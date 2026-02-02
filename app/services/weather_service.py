import requests
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json
import numpy as np
from ..config import settings
from ..database import get_db

class WeatherService:
    def __init__(self):
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.noaa_url = "https://www.ncei.noaa.gov/access/services/data/v1"
        
    async def get_nasa_power_data(self, lat: float, lon: float, start_date: str, end_date: str) -> List[Dict]:
        """
        Récupérer les données de la NASA POWER API avec paramètres corrigés
        
        IMPORTANT: NASA POWER utilise:
        - T2M: Température moyenne à 2m (°C)
        - T2M_MAX, T2M_MIN: Temp max/min à 2m (°C)
        - PRECTOTCORR: Précipitation corrigée (mm/jour)
        - RH2M: Humidité relative à 2m (%)
        - ALLSKY_SFC_SW_DWN: Rayonnement solaire incident (MJ/m²/jour)
        - WS2M: Vitesse vent à 2m (m/s)
        - EVPTRNS: Évapotranspiration nette (mm/jour) - PAS POTENTIELLE!
        
        Pour l'Algérie, on doit calculer ET0 avec FAO Penman-Monteith
        """
        # PARAMÈTRES CORRIGÉS - Demander plus de données
        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M,PS,QV2M",
            # ^^ Ajout de PS (pression surface) et QV2M (humidité spécifique)
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.nasa_power_url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_nasa_data(data, lat)
                    else:
                        error_text = await response.text()
                        print(f"NASA API Error: {response.status}, {error_text}")
                        return None
        except Exception as e:
            print(f"Error fetching NASA data: {e}")
            return None
    
    def _process_nasa_data(self, raw_data: Dict, latitude: float) -> List[Dict]:
        """
        Process NASA POWER API response avec calculs corrigés
        """
        processed_data = []
        
        if 'properties' not in raw_data or 'parameter' not in raw_data['properties']:
            return processed_data
        
        parameters = raw_data['properties']['parameter']
        
        # Get dates from the response
        dates = list(parameters.get('T2M', {}).keys())
        
        for date_str in dates:
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                day_of_year = date_obj.timetuple().tm_yday
                
                # Extraire les données brutes
                weather_record = {
                    'date': date_obj,
                    'temperature_avg': parameters.get('T2M', {}).get(date_str),
                    'temperature_max': parameters.get('T2M_MAX', {}).get(date_str),
                    'temperature_min': parameters.get('T2M_MIN', {}).get(date_str),
                    'precipitation': parameters.get('PRECTOTCORR', {}).get(date_str),  # mm/day
                    'humidity': parameters.get('RH2M', {}).get(date_str),  # %
                    'solar_radiation': parameters.get('ALLSKY_SFC_SW_DWN', {}).get(date_str),  # MJ/m²/day
                    'wind_speed': parameters.get('WS2M', {}).get(date_str),  # m/s
                    'surface_pressure': parameters.get('PS', {}).get(date_str),  # kPa
                    'specific_humidity': parameters.get('QV2M', {}).get(date_str),  # kg/kg
                    'nasa_evapotranspiration': parameters.get('EVPTRNS', {}).get(date_str),  # mm/day (NETTE!)
                    'source': 'NASA_POWER',
                    'latitude': latitude
                }
                
                # CORRECTION 1: Calculer l'évapotranspiration de référence (ET0) avec FAO
                if all([
                    weather_record['temperature_avg'] is not None,
                    weather_record['temperature_max'] is not None,
                    weather_record['temperature_min'] is not None,
                    weather_record['solar_radiation'] is not None,
                    weather_record['wind_speed'] is not None,
                    weather_record['humidity'] is not None
                ]):
                    et0 = self.calculate_fao_et0(
                        t_mean=weather_record['temperature_avg'],
                        t_max=weather_record['temperature_max'],
                        t_min=weather_record['temperature_min'],
                        solar_rad=weather_record['solar_radiation'],  # MJ/m²/day
                        wind_speed=weather_record['wind_speed'],  # m/s
                        humidity=weather_record['humidity'],  # %
                        latitude=latitude,
                        day_of_year=day_of_year,
                        altitude=0  # Pour Alger, altitude ≈ 0
                    )
                    weather_record['et0_fao'] = et0
                    
                    # CORRECTION 2: Calculer l'évapotranspiration réelle (ETA)
                    # ETA = ET0 * Kc * Kr, où Kc dépend de la culture
                    # Pour estimation générale en Algérie:
                    kc = self._get_crop_coefficient(date_obj.month)
                    weather_record['et_real'] = et0 * kc
                else:
                    weather_record['et0_fao'] = None
                    weather_record['et_real'] = None
                
                processed_data.append(weather_record)
                
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
        
        return processed_data
    
    def calculate_fao_et0(self, t_mean, t_max, t_min, solar_rad, wind_speed, 
                         humidity, latitude, day_of_year, altitude=0):
        """
        FAO Penman-Monteith ET0 calculation (FORMULE STANDARD FAO 56)
        
        Formule: ET0 = [0.408Δ(Rn-G) + γ(900/(T+273))u2(es-ea)] / [Δ + γ(1+0.34u2)]
        
        Où:
        - Δ = pente de la courbe de pression de vapeur (kPa/°C)
        - Rn = rayonnement net (MJ/m²/jour)
        - G = flux de chaleur du sol (MJ/m²/jour)
        - γ = constante psychrométrique (kPa/°C)
        - T = température moyenne (°C)
        - u2 = vitesse du vent à 2m (m/s)
        - es = pression de vapeur saturante (kPa)
        - ea = pression de vapeur réelle (kPa)
        """
        try:
            # 1. Pression atmosphérique (kPa)
            P = 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26
            
            # 2. Constante psychrométrique (kPa/°C)
            gamma = 0.665 * 10 ** -3 * P
            
            # 3. Pression de vapeur saturante (kPa)
            es_tmax = 0.6108 * np.exp(17.27 * t_max / (t_max + 237.3))
            es_tmin = 0.6108 * np.exp(17.27 * t_min / (t_min + 237.3))
            es = (es_tmax + es_tmin) / 2
            
            # 4. Pression de vapeur réelle (kPa)
            ea = es * humidity / 100
            
            # 5. Pente de la courbe de pression de vapeur (kPa/°C)
            delta = 4098 * es / ((t_mean + 237.3) ** 2)
            
            # 6. Rayonnement extraterrestre (Ra)
            Ra = self._calculate_extraterrestrial_radiation(latitude, day_of_year)
            
            # 7. Rayonnement net à ondes courtes (Rns)
            # albedo = 0.23 pour surface herbeuse (référence FAO)
            albedo = 0.23
            Rns = (1 - albedo) * solar_rad  # solar_rad en MJ/m²/day
            
            # 8. Rayonnement net à ondes longues (Rnl)
            sigma = 4.903 * 10 ** -9  # MJ/K⁴/m²/day
            Rnl = sigma * ((t_max + 273.16) ** 4 + (t_min + 273.16) ** 4) / 2
            Rnl *= (0.34 - 0.14 * np.sqrt(ea))
            Rnl *= (1.35 * solar_rad / (0.75 * Ra) - 0.35)
            
            # 9. Rayonnement net (Rn)
            Rn = Rns - Rnl
            
            # 10. Flux de chaleur du sol (G) - négligeable pour quotidien
            G = 0
            
            # 11. Calcul final ET0 (mm/day)
            numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (t_mean + 273)) * wind_speed * (es - ea)
            denominator = delta + gamma * (1 + 0.34 * wind_speed)
            
            et0 = numerator / denominator
            
            # Limiter aux valeurs réalistes pour l'Algérie
            et0 = max(0, min(et0, 15))  # Max 15 mm/jour en Algérie
            
            return round(et0, 2)
            
        except Exception as e:
            print(f"Error calculating FAO ET0: {e}")
            return None
    
    def _calculate_extraterrestrial_radiation(self, latitude, day_of_year):
        """Rayonnement extraterrestre (MJ/m²/day) - FAO 56 Eq. 21"""
        lat_rad = latitude * np.pi / 180
        
        # Constante solaire
        Gsc = 0.0820  # MJ/m²/min
        
        # Rayonnement solaire inverse
        dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        
        # Déclinaison solaire
        delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        
        # Angle horaire du coucher de soleil
        omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        
        # Rayonnement extraterrestre
        Ra = (24 * 60 / np.pi) * Gsc * dr * (
            omega_s * np.sin(lat_rad) * np.sin(delta) + 
            np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
        )
        
        return Ra
    
    def _get_crop_coefficient(self, month):
        """
        Coefficient cultural (Kc) pour l'Algérie par mois
        Basé sur les cultures principales (céréales, maraîchage)
        """
        # Valeurs typiques pour l'Algérie méditerranéenne
        kc_values = {
            1: 0.4,   # Janvier - début croissance
            2: 0.5,
            3: 0.7,   # Mars - croissance active
            4: 0.9,
            5: 1.0,   # Mai - pic de croissance
            6: 1.1,
            7: 1.2,   # Juillet - maturation
            8: 1.1,
            9: 0.9,   # Septembre - récolte
            10: 0.6,
            11: 0.4,
            12: 0.3
        }
        return kc_values.get(month, 0.7)
    
    def calculate_hargreaves_et0(self, t_mean, t_max, t_min, latitude, day_of_year):
        """
        Équation de Hargreaves simplifiée pour quand on manque de données
        ET0 = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
        """
        try:
            Ra = self._calculate_extraterrestrial_radiation(latitude, day_of_year)
            et0 = 0.0023 * Ra * (t_mean + 17.8) * np.sqrt(t_max - t_min)
            return max(0, round(et0, 2))
        except:
            return None
    def _convert_numpy_types(self, data_dict: Dict) -> Dict:
        """
        Convertir les types NumPy en types Python natifs pour PostgreSQL
        """
        converted = {}
        for key, value in data_dict.items():
            if value is None:
                converted[key] = None
            elif isinstance(value, np.float64):
                converted[key] = float(value)
            elif isinstance(value, np.float32):
                converted[key] = float(value)
            elif isinstance(value, np.int64):
                converted[key] = int(value)
            elif isinstance(value, np.int32):
                converted[key] = int(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            else:
                converted[key] = value
        return converted
    
    async def save_weather_data_to_db(self, region_id: int, processed_data: List[Dict], db):
        """
        Sauvegarder avec les nouvelles colonnes et conversion des types
        """
        from .. import models
        
        for record in processed_data:
            # CONVERTIR LES TYPES NP.FLOAT64 AVANT DE SAUVEGARDER
            converted_record = self._convert_numpy_types(record)
            
            existing = db.query(models.WeatherData).filter(
                models.WeatherData.region_id == region_id,
                models.WeatherData.date == converted_record['date']
            ).first()
            
            if not existing:
                weather_entry = models.WeatherData(
                    region_id=region_id,
                    date=converted_record['date'],
                    temperature_avg=converted_record['temperature_avg'],
                    temperature_max=converted_record['temperature_max'],
                    temperature_min=converted_record['temperature_min'],
                    precipitation=converted_record['precipitation'],
                    humidity=converted_record['humidity'],
                    solar_radiation=converted_record['solar_radiation'],
                    wind_speed=converted_record['wind_speed'],
                    evapotranspiration=converted_record.get('et_real') or 
                                    converted_record.get('et0_fao') or 
                                    converted_record.get('nasa_evapotranspiration'),
                    source=converted_record['source'],
                    # Nouvelles colonnes
                    et0_fao=converted_record.get('et0_fao'),
                    et_real=converted_record.get('et_real'),
                    surface_pressure=converted_record.get('surface_pressure'),
                    specific_humidity=converted_record.get('specific_humidity')
                )
                db.add(weather_entry)
        
        db.commit()
        return len(processed_data)
    
    async def get_historical_weather(self, wilaya_code: str, years_back: int = 5) -> Dict:
        """
        Récupérer les données historiques avec calculs corrigés
        """
        with open('algeria_cordinates.json', 'r') as f:
            algeria_data = json.load(f)
        
        wilaya_key = f"DZ{wilaya_code.zfill(2)}"
        if wilaya_key not in algeria_data:
            raise ValueError(f"Wilaya code {wilaya_code} not found")
        
        wilaya_info = algeria_data[wilaya_key]
        lat = wilaya_info['lat']
        lon = wilaya_info['lon']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        data = await self.get_nasa_power_data(lat, lon, start_str, end_str)
        
        return {
            'wilaya': wilaya_info['name'],
            'wilaya_code': wilaya_code,
            'coordinates': {'lat': lat, 'lon': lon},
            'data': data,
            'period': {'start': start_str, 'end': end_str}
        }
    
    async def get_climate_summary(self, lat: float, lon: float) -> Dict:
        """
        Résumé climatique avec indicateurs corrigés
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        data = await self.get_nasa_power_data(lat, lon, start_str, end_str)
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Calculer des indicateurs clés pour l'Algérie
        summary = {
            'avg_temperature': round(df['temperature_avg'].mean(), 2) if not df.empty else None,
            'total_precipitation': round(df['precipitation'].sum(), 1) if not df.empty else None,
            'avg_humidity': round(df['humidity'].mean(), 1) if not df.empty else None,
            'avg_solar_radiation': round(df['solar_radiation'].mean(), 2) if not df.empty else None,
            'total_evapotranspiration': round(df['et0_fao'].sum(), 1) if 'et0_fao' in df.columns else None,
            'water_balance': round(df['precipitation'].sum() - df['et0_fao'].sum(), 1) 
                          if 'et0_fao' in df.columns else None,
            'aridity_index': round(df['et0_fao'].mean() / (df['precipitation'].mean() + 0.1), 2) 
                           if not df.empty else None,
            'data_points': len(data)
        }
        
        return summary

# Fonctions utilitaires
def format_date_for_nasa(date_obj: datetime) -> str:
    return date_obj.strftime("%Y%m%d")

def parse_nasa_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y%m%d")