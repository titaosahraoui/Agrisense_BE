from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import json
from ..services.weather_service import WeatherService
from ..services.stress_analysis import StressAnalysisService
from ..database import get_db
from .. import models, schemas
import pandas as pd

router = APIRouter(prefix="/weather", tags=["Weather Data"])
weather_service = WeatherService()
analysis_service = StressAnalysisService()
@router.get("/wilaya/{wilaya_code}")
async def get_wilaya_weather_data(
    wilaya_code: str,
    start_date: str = Query((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")),
    end_date: str = Query(datetime.now().strftime("%Y-%m-%d")),
    db: Session = Depends(get_db)
):
    """
    Récupérer les données météo pour une wilaya
    wilaya_code: Code de la wilaya (ex: '16' pour Alger)
    """
    # Load coordinates from JSON file
    try:
        with open('algeria_cordinates.json', 'r') as f:
            algeria_data = json.load(f)
    except FileNotFoundError:
        # If file not found in current directory, try parent
        with open('./algeria_cordinates.json', 'r') as f:
            algeria_data = json.load(f)
    
    wilaya_key = f"DZ{wilaya_code.zfill(2)}"
    if wilaya_key not in algeria_data:
        raise HTTPException(status_code=404, detail=f"Wilaya code {wilaya_code} non trouvée")
    
    wilaya_info = algeria_data[wilaya_key]
    
    # Check if region exists in database
    region = db.query(models.Region).filter(
        models.Region.wilaya_code == int(wilaya_code)
    ).first()
    
    if not region:
        # Create region entry if it doesn't exist
        region = models.Region(
            name=wilaya_info['name'],
            wilaya_code=int(wilaya_code),
            centroid_lat=wilaya_info['lat'],
            centroid_lon=wilaya_info['lon']
        )
        db.add(region)
        db.commit()
        db.refresh(region)
    
    # Format dates for NASA API
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Convert to NASA format
    nasa_start = start_dt.strftime("%Y%m%d")
    nasa_end = end_dt.strftime("%Y%m%d")
    
    # Fetch from NASA API
    nasa_data = await weather_service.get_nasa_power_data(
        region.centroid_lat,
        region.centroid_lon,
        nasa_start,
        nasa_end
    )
    
    if not nasa_data:
        raise HTTPException(status_code=503, detail="Impossible de récupérer les données météo")
    
    # Save to database
    saved_count = await weather_service.save_weather_data_to_db(region.id, nasa_data, db)
    
    return {
        "wilaya": wilaya_info['name'],
        "wilaya_code": wilaya_code,
        "coordinates": {
            "lat": wilaya_info['lat'],
            "lon": wilaya_info['lon']
        },
        "period": {
            "start": start_date,
            "end": end_date
        },
        "data": nasa_data,
        "saved_records": saved_count
    }

@router.post("/analyze/{wilaya_code}")
async def analyze_water_stress(
    wilaya_code: str,
    db: Session = Depends(get_db)
):
    """
    Analyser le stress hydrique pour une wilaya
    
    This endpoint will:
    1. Check if region exists, create if not
    2. Check for recent weather data in database (last 7 days)
    3. If no recent data, fetch from NASA POWER API
    4. Perform stress analysis
    """
    try:
        # Load coordinates
        with open('algeria_cordinates.json', 'r') as f:
            algeria_data = json.load(f)
        
        wilaya_key = f"DZ{wilaya_code.zfill(2)}"
        if wilaya_key not in algeria_data:
            raise HTTPException(status_code=404, detail=f"Wilaya code {wilaya_code} non trouvée")
        
        wilaya_info = algeria_data[wilaya_key]
        
        # Find or create region
        region = db.query(models.Region).filter(
            models.Region.wilaya_code == int(wilaya_code)
        ).first()
        
        if not region:
            region = models.Region(
                name=wilaya_info['name'],
                wilaya_code=int(wilaya_code),
                centroid_lat=wilaya_info['lat'],
                centroid_lon=wilaya_info['lon']
            )
            db.add(region)
            db.commit()
            db.refresh(region)
        
        # Calculate date range for last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Format for database query
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Check for existing weather data in the last 7 days
        weather_records = (db.query(models.WeatherData)
                          .filter(models.WeatherData.region_id == region.id)
                          .filter(models.WeatherData.date >= start_date_str)
                          .filter(models.WeatherData.date <= end_date_str)
                          .order_by(models.WeatherData.date.desc())
                          .all())
        
        if not weather_records:
            # No recent data, fetch from NASA
            print(f"No recent data found for region {region.id}, fetching from NASA...")
            
            nasa_start = start_date.strftime("%Y%m%d")
            nasa_end = end_date.strftime("%Y%m%d")
            
            nasa_data = await weather_service.get_nasa_power_data(
                region.centroid_lat,
                region.centroid_lon,
                nasa_start,
                nasa_end
            )
            
            if nasa_data:
                await weather_service.save_weather_data_to_db(region.id, nasa_data, db)
                # Re-query to get the saved data
                weather_records = (db.query(models.WeatherData)
                                  .filter(models.WeatherData.region_id == region.id)
                                  .filter(models.WeatherData.date >= start_date_str)
                                  .filter(models.WeatherData.date <= end_date_str)
                                  .order_by(models.WeatherData.date.desc())
                                  .all())
            else:
                raise HTTPException(status_code=503, detail="Could not fetch data from NASA")
        
        if not weather_records:
            raise HTTPException(status_code=404, detail="No weather data available for analysis")
        
        # Use the most recent record for analysis
        latest_weather = weather_records[0]
        
        # Get satellite data if available
        satellite_data = (db.query(models.SatelliteData)
                         .filter(models.SatelliteData.region_id == region.id)
                         .order_by(models.SatelliteData.acquisition_date.desc())
                         .first())
        
        # Prepare indicators - using actual data, not defaults
        indicators: dict[str, float] = {
            'temperature': float(latest_weather.temperature_avg) if latest_weather.temperature_avg else 0,
            'precipitation': float(latest_weather.precipitation) if latest_weather.precipitation else 0,
            'humidity': float(latest_weather.humidity) if latest_weather.humidity else 0,
            'evapotranspiration': float(latest_weather.evapotranspiration) if latest_weather.evapotranspiration else 0,
            'solar_radiation': float(latest_weather.solar_radiation) if latest_weather.solar_radiation else 0
        }
        
        # Add satellite indicators if available
        if satellite_data:
            if satellite_data.ndvi is not None:
                indicators['ndvi'] = float(satellite_data.ndvi)
            if satellite_data.lst is not None:
                indicators['lst'] = float(satellite_data.lst)
            if satellite_data.ndwi is not None:
                indicators['ndwi'] = float(satellite_data.ndwi)
        
        # Calculate averages for the period for better analysis
        avg_temperature = sum([r.temperature_avg for r in weather_records if r.temperature_avg]) / len(weather_records)
        total_precipitation = sum([r.precipitation for r in weather_records if r.precipitation])
        avg_humidity = sum([r.humidity for r in weather_records if r.humidity]) / len(weather_records)
        
        # Use averages for more accurate analysis
        indicators.update({
            'avg_temperature': avg_temperature,
            'total_precipitation': total_precipitation,
            'avg_humidity': avg_humidity
        })
        
        print(f"Indicators for analysis: {indicators}")
        
        # Calculate stress score
        stress_score = analysis_service.calculate_stress_score(indicators)
        stress_level = analysis_service.determine_stress_level(stress_score)
        recommendations = analysis_service.generate_recommendations(indicators, stress_level)
        
        # Save analysis to database
        analysis = models.StressAnalysis(
            region_id=region.id,
            stress_level=stress_level,
            stress_score=stress_score,
            indicators=indicators,
            recommendations=recommendations
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        return {
            "wilaya": region.name,
            "wilaya_code": wilaya_code,
            "stress_level": stress_level,
            "stress_score": round(stress_score, 2),
            "indicators": {
                "temperature": round(indicators.get('temperature', 0), 2),
                "precipitation": round(indicators.get('precipitation', 0), 2),
                "humidity": round(indicators.get('humidity', 0), 2),
                "evapotranspiration": round(indicators.get('evapotranspiration', 0), 2),
                "avg_temperature": round(indicators.get('avg_temperature', 0), 2),
                "total_precipitation": round(indicators.get('total_precipitation', 0), 2),
                "avg_humidity": round(indicators.get('avg_humidity', 0), 2)
            },
            "recommendations": recommendations,
            "analysis_date": datetime.utcnow(),
            "data_period": {
                "start": start_date_str,
                "end": end_date_str,
                "data_points": len(weather_records)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/historical/{wilaya_code}")
async def get_historical_weather(
    wilaya_code: str,
    years: int = Query(5, ge=1, le=20, description="Nombre d'années historiques"),
    db: Session = Depends(get_db)
):
    """
    Récupérer les données historiques pour une wilaya
    """
    historical_data = await weather_service.get_historical_weather(wilaya_code, years)
    
    # Save to database
    if historical_data and 'data' in historical_data and historical_data['data']:
        # Find region
        region = db.query(models.Region).filter(
            models.Region.wilaya_code == int(wilaya_code)
        ).first()
        
        if region:
            saved_count = await weather_service.save_weather_data_to_db(
                region.id, 
                historical_data['data'], 
                db
            )
            historical_data['saved_records'] = saved_count
    
    return historical_data

@router.get("/summary/{wilaya_code}")
async def get_climate_summary(
    wilaya_code: str,
    db: Session = Depends(get_db)
):
    """
    Obtenir un résumé climatique pour une wilaya
    """
    # Load coordinates
    with open('algeria_cordinates.json', 'r') as f:
        algeria_data = json.load(f)
    
    wilaya_key = f"DZ{wilaya_code.zfill(2)}"
    if wilaya_key not in algeria_data:
        raise HTTPException(status_code=404, detail=f"Wilaya code {wilaya_code} non trouvée")
    
    wilaya_info = algeria_data[wilaya_key]
    
    # Get climate summary
    summary = await weather_service.get_climate_summary(
        wilaya_info['lat'],
        wilaya_info['lon']
    )
    
    if not summary:
        raise HTTPException(status_code=503, detail="Impossible de récupérer le résumé climatique")
    
    return {
        "wilaya": wilaya_info['name'],
        "summary": summary,
        "coordinates": {
            "lat": wilaya_info['lat'],
            "lon": wilaya_info['lon']
        }
    }

@router.get("/compare/{wilaya_codes}")
async def compare_wilayas_weather(
    wilaya_codes: str,
    start_date: str = Query((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")),
    end_date: str = Query(datetime.now().strftime("%Y-%m-%d"))
):
    """
    Comparer les données météo entre plusieurs wilayas
    wilaya_codes: Codes séparés par des virgules (ex: '16,31,12')
    """
    codes = wilaya_codes.split(',')
    results = []
    
    for code in codes:
        code = code.strip()
        try:
            data = await weather_service.get_historical_weather(code, 1)
            if data and 'data' in data:
                # Calculate averages
                df = pd.DataFrame(data['data'])
                summary = {
                    'wilaya': data['wilaya'],
                    'code': code,
                    'avg_temp': df['temperature_avg'].mean(),
                    'total_precip': df['precipitation'].sum(),
                    'avg_humidity': df['humidity'].mean()
                }
                results.append(summary)
        except Exception as e:
            print(f"Error processing wilaya {code}: {e}")
    
    return {"comparison": results}

# @router.post("/analyze/{wilaya_code}")
# async def analyze_water_stress(
#     wilaya_code: str,
#     db: Session = Depends(get_db)
# ):
#     """
#     Analyser le stress hydrique pour une wilaya
#     """
#     # Find region
#     region = db.query(models.Region).filter(
#         models.Region.wilaya_code == int(wilaya_code)
#     ).first()
    
#     if not region:
#         raise HTTPException(status_code=404, detail="Région non trouvée")
    
#     # Get latest weather data
#     weather_data = (db.query(models.WeatherData)
#                    .filter(models.WeatherData.region_id == region.id)
#                    .order_by(models.WeatherData.date.desc())
#                    .first())
    
#     if not weather_data:
#         # Fetch recent data if not in DB
#         nasa_data = await weather_service.get_nasa_power_data(
#             region.centroid_lat,
#             region.centroid_lon,
#             (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
#             datetime.now().strftime("%Y%m%d")
#         )
        
#         if nasa_data:
#             await weather_service.save_weather_data_to_db(region.id, nasa_data, db)
#             weather_data = nasa_data[0] if nasa_data else None
    
#     if not weather_data:
#         raise HTTPException(status_code=404, detail="Données météo non disponibles")
    
#     # Get satellite data
#     satellite_data = (db.query(models.SatelliteData)
#                      .filter(models.SatelliteData.region_id == region.id)
#                      .order_by(models.SatelliteData.acquisition_date.desc())
#                      .first())
    
#     # Prepare indicators
#     indicators = {
#         'temperature': weather_data.temperature_avg,
#         'precipitation': weather_data.precipitation,
#         'humidity': weather_data.humidity,
#         'evapotranspiration': weather_data.evapotranspiration,
#         'solar_radiation': weather_data.solar_radiation
#     }
    
#     if satellite_data:
#         indicators.update({
#             'ndvi': satellite_data.ndvi,
#             'lst': satellite_data.lst,
#             'ndwi': satellite_data.ndwi
#         })
    
#     # Calculate stress score
#     stress_score = analysis_service.calculate_stress_score(indicators)
#     stress_level = analysis_service.determine_stress_level(stress_score)
#     recommendations = analysis_service.generate_recommendations(indicators, stress_level)
    
#     # Save analysis
#     analysis = models.StressAnalysis(
#         region_id=region.id,
#         stress_level=stress_level,
#         stress_score=stress_score,
#         indicators=indicators,
#         recommendations=recommendations
#     )
    
#     db.add(analysis)
#     db.commit()
#     db.refresh(analysis)
    
#     return {
#         "wilaya": region.name,
#         "wilaya_code": wilaya_code,
#         "stress_level": stress_level,
#         "stress_score": stress_score,
#         "indicators": indicators,
#         "recommendations": recommendations,
#         "analysis_date": datetime.utcnow()
#     }