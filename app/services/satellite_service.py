import ee
from .ee_init import init_ee
init_ee()
import geemap
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ..config import settings

class SatelliteService:
    def __init__(self):
        self._initialized = False

    # def _init_ee(self):
    #     try:
    #         ee.Initialize()
    #         print("✅ Earth Engine initialized")
    #     except Exception:
    #         ee.Authenticate()
    #         ee.Initialize()
    #         print("✅ Earth Engine authenticated and initialized")
        
    def get_sentinel2_data(self, geometry: Dict, start_date: str, end_date: str, 
                        cloud_percentage: int = 20) -> ee.ImageCollection:
        """Récupérer les images Sentinel-2 (version harmonisée)"""
        # UTILISER LA VERSION HARMONISÉE
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(ee.Geometry(geometry))
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage))
                    .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        return collection
    
    def get_landsat_data(self, geometry: Dict, start_date: str, end_date: str, 
                         cloud_percentage: int = 20) -> ee.ImageCollection:
        """Récupérer les images Landsat 8 (pour données historiques avant 2015)"""
        # Landsat 8 Collection 2 Tier 1
        collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                    .filterBounds(ee.Geometry(geometry))
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUD_COVER', cloud_percentage))
                    .sort('CLOUD_COVER'))
        
        return collection
    
    def calculate_landsat_indices(self, image: ee.Image) -> ee.Image:
        """Calculer les indices spectraux pour Landsat 8"""
        # Landsat 8 bandes: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B10=Thermal
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        
        # EVI pour Landsat
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('SR_B5'),
                'RED': image.select('SR_B4'),
                'BLUE': image.select('SR_B2')
            }
        ).rename('EVI')
        
        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        
        # LST - Utiliser la bande thermique B10 (température de surface)
        # Convertir de Kelvin à Celsius: (B10 * 0.00341802 + 149.0) - 273.15
        lst = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        
        return image.addBands([ndvi, evi, ndwi, lst])

    def calculate_indices(self, image: ee.Image) -> ee.Image:
        """Calculer les indices spectraux"""
        # NDVI
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        # NDWI
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # LST: Get from Landsat 8 instead (PROPER thermal bands)
        landsat_image = self.get_landsat_for_date(image.date(), image.geometry())
        
        if landsat_image:
            # PROPER LST calculation from Landsat thermal bands
            lst = self.calculate_landsat_lst(landsat_image)
        else:
            # Fallback: Estimate from NDVI
            lst = self.estimate_lst_from_ndvi(ndvi)
        
        return image.addBands([ndvi, evi, ndwi, lst])
    
    def get_landsat_for_date(self, date, geometry):
        """Get Landsat 8/9 image for same date/location"""
        landsat_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                            .filterDate(date, date.advance(1, 'day'))
                            .filterBounds(geometry)
                            .filter(ee.Filter.lt('CLOUD_COVER', 20)))
        
        if landsat_collection.size().getInfo() > 0:
            return landsat_collection.first()
        return None

    def calculate_landsat_lst(self, landsat_image):
        """Calculate actual LST from Landsat thermal bands"""
        # Convert to TOA radiance
        ML = 0.0003342  # Band 10 multiplicative rescaling factor
        AL = 0.1        # Band 10 additive rescaling factor
        K1 = 774.8853   # Band 10 thermal constant
        K2 = 1321.0789  # Band 10 thermal constant
        
        thermal_band = landsat_image.select('ST_B10')
        
        # Convert to radiance
        radiance = thermal_band.multiply(ML).add(AL)
        
        # Convert to brightness temperature (Kelvin)
        bt = radiance.expression(
            'K2 / log((K1 / radiance) + 1)',
            {
                'K1': K1,
                'K2': K2,
                'radiance': radiance
            }
        )
        
        # Convert to Celsius
        lst_celsius = bt.subtract(273.15).rename('LST')
        
        return lst_celsius

    def estimate_lst_from_ndvi(self, ndvi):
        """Fallback: Estimate LST from NDVI when no Landsat data"""
        # Empirical relationship for Algeria
        lst = ndvi.expression(
            '35 - (15 * NDVI)',  # More vegetation = cooler
            {'NDVI': ndvi}
        ).rename('LST')
        
        return lst
    

    def get_monthly_composite(self, geometry: Dict, year: int, month: int) -> Optional[Dict]:
        """
        Get monthly composite satellite data with better cloud filtering
        """
        start_date = f"{year}-{month:02d}-01"
        # Calculate last day of month
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"
        
        try:
            # Use Sentinel-2 for 2015+, Landsat for earlier
            if year >= 2015:
                collection = self.get_sentinel2_data(geometry, start_date, end_date, cloud_percentage=30)
            else:
                collection = self.get_landsat_data(geometry, start_date, end_date, cloud_percentage=30)
            
            collection_size = collection.size().getInfo()
            
            if collection_size == 0:
                return None
            
            # Create median composite (better than first image)
            if collection_size > 1:
                # Use median to reduce cloud/noise
                composite = collection.median()
            else:
                composite = collection.first()
            
            # Calculate indices
            if year >= 2015:
                composite_with_indices = self.calculate_indices(composite)
            else:
                composite_with_indices = self.calculate_landsat_indices(composite)
            
            # Get statistics
            scale = 10 if year >= 2015 else 30
            stats = composite_with_indices.select(['NDVI', 'NDWI', 'LST']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry(geometry),
                scale=scale,
                maxPixels=1e9
            )
            
            return stats.getInfo()
            
        except Exception as e:
            print(f"Error creating monthly composite for {year}-{month}: {e}")
            return None
    
    
    def get_region_statistics(self, geometry: Dict, start_date: str, 
                             end_date: str) -> Optional[Dict]:
        """Obtenir les statistiques des indices pour une région"""
        try:
            # Déterminer quelle année pour choisir le bon satellite
            year = int(start_date.split('-')[0])
            use_landsat = year < 2015  # Sentinel-2 disponible à partir de 2015
            
            # Si c'est un Point, créer un buffer de 5km pour avoir une zone d'analyse
            if geometry.get('type') == 'Point':
                point = ee.Geometry.Point(geometry['coordinates'])
                # Buffer de 5000 mètres (5km) pour créer une zone d'analyse
                ee_geometry = point.buffer(5000)
                # Convertir en dict pour les fonctions de collection
                geometry_for_filter = ee_geometry.getInfo()
            else:
                ee_geometry = ee.Geometry(geometry)
                geometry_for_filter = geometry
            
            # Choisir le bon satellite selon l'année
            if use_landsat:
                # Utiliser Landsat 8 pour les années avant 2015
                collection = self.get_landsat_data(geometry_for_filter, start_date, end_date)
                collection_size = collection.size().getInfo()
                
                if collection_size == 0:
                    # Ne pas afficher de warning pour chaque mois, juste retourner None silencieusement
                    return None
                
                # Calculer les indices Landsat
                best_image = collection.first()
                image_with_indices = self.calculate_landsat_indices(best_image)
                scale = 30  # Résolution Landsat
            else:
                # Utiliser Sentinel-2 pour 2015 et après
                collection = self.get_sentinel2_data(geometry_for_filter, start_date, end_date)
                collection_size = collection.size().getInfo()
                
                if collection_size == 0:
                    # Ne pas afficher de warning pour chaque mois
                    return None
                
                # Calculer les indices Sentinel-2
                best_image = collection.first()
                image_with_indices = self.calculate_indices(best_image)
                scale = 10  # Résolution Sentinel-2
            
            # Calculer les statistiques avec meilleure gestion d'erreur
            stats = image_with_indices.select(['NDVI', 'EVI', 'NDWI', 'LST']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=scale,
                maxPixels=1e9  # Augmenter la limite de pixels
            )
            
            stats_info = stats.getInfo()
            
            # Vérifier si les résultats sont valides
            if not stats_info or len(stats_info) == 0:
                return None
            
            # Vérifier que les valeurs ne sont pas toutes None
            valid_stats = {k: v for k, v in stats_info.items() if v is not None}
            if len(valid_stats) == 0:
                return None
            
            return valid_stats
            
        except Exception as e:
            # Ne pas afficher d'erreur pour chaque tentative, juste logger silencieusement
            return None

    def generate_sample_data(self, count: int, wilaya_code: Optional[int] = None) -> List[Dict]:
        """Generate sample satellite data for demo purposes"""
        import random
        from datetime import date
        
        sample_data = []
        now = datetime.utcnow()
        
        # Sample wilayas if no specific code provided
        wilayas = [{"code": 16, "name": "Alger"}, {"code": 31, "name": "Oran"}, 
                   {"code": 25, "name": "Constantine"}, {"code": 19, "name": "Setif"}]
        
        if wilaya_code:
            wilayas = [w for w in wilayas if w["code"] == wilaya_code]
            if not wilayas:
                wilayas = [{"code": wilaya_code, "name": f"Wilaya {wilaya_code}"}]
        
        for i in range(min(count, len(wilayas))):
            wilaya = wilayas[i % len(wilayas)]
            
            # Generate realistic values for Algeria
            base_ndvi = 0.4 + random.random() * 0.3
            base_ndwi = -0.1 + random.random() * 0.2
            base_lst = 25 + random.random() * 10
            
            sample_data.append({
                "name": wilaya["name"],
                "date": (now - timedelta(days=i*7)).isoformat(),
                "ndvi": round(base_ndvi + random.random() * 0.1, 3),
                "ndwi": round(base_ndwi + random.random() * 0.05, 3),
                "lst": round(base_lst + random.random() * 3, 1),
                "stress_score": round(30 + random.random() * 40, 1),
                "wilaya_code": wilaya["code"],
                "is_sample": True
            })
        
        return sample_data