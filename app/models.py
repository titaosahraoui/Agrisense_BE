from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

class Region(Base):
    __tablename__ = "regions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    wilaya_code = Column(Integer, unique=True)
    geometry = Column(JSON)  # GeoJSON geometry
    area_km2 = Column(Float)
    centroid_lat = Column(Float)
    centroid_lon = Column(Float)

class WeatherData(Base):
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(Integer, index=True)
    date = Column(DateTime, index=True)
    temperature_avg = Column(Float)
    temperature_max = Column(Float)
    temperature_min = Column(Float)
    precipitation = Column(Float)
    humidity = Column(Float)
    solar_radiation = Column(Float)
    wind_speed = Column(Float)
    evapotranspiration = Column(Float)  # Évapotranspiration réelle (ETA)
    
    # NOUVELLES COLONNES
    et0_fao = Column(Float)  # Évapotranspiration de référence FAO
    et_real = Column(Float)  # Évapotranspiration réelle (ETA = ET0 * Kc)
    surface_pressure = Column(Float)
    specific_humidity = Column(Float)
    
    source = Column(String)

class SatelliteData(Base):
    __tablename__ = "satellite_data"
    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(Integer, index=True)
    acquisition_date = Column(DateTime, index=True)
    satellite = Column(String)  # Sentinel-2, Landsat-8, etc.
    ndvi = Column(Float)  # Normalized Difference Vegetation Index
    evi = Column(Float)   # Enhanced Vegetation Index
    ndwi = Column(Float)  # Normalized Difference Water Index
    lst = Column(Float)   # Land Surface Temperature
    cloud_cover = Column(Float)
    geometry = Column(JSON)  # GeoJSON polygon

class StressAnalysis(Base):
    __tablename__ = "stress_analysis"
    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(Integer, index=True)
    analysis_date = Column(DateTime, index=True, default=datetime.utcnow)
    stress_level = Column(String)  # low, medium, high, severe
    stress_score = Column(Float)   # 0-100
    indicators = Column(JSON)      # {"ndvi": 0.4, "lst": 35, "precipitation": 15, ...}
    recommendations = Column(JSON) # Liste de recommandations

class TrainingData(Base):
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    wilaya_code = Column(Integer, index=True)
    date = Column(DateTime, index=True)
    
    # Features météo
    temperature_avg = Column(Float)
    temperature_max = Column(Float)
    temperature_min = Column(Float)
    precipitation = Column(Float)
    humidity = Column(Float)
    solar_radiation = Column(Float)
    wind_speed = Column(Float)
    evapotranspiration = Column(Float)
    
    # Features satellites
    ndvi = Column(Float)
    ndwi = Column(Float)
    lst = Column(Float)
    
    # Features temporelles
    month = Column(Integer)
    season = Column(String)
    day_of_year = Column(Integer)
    
    # Features calculées (lag, cumul, etc.)
    precip_cumul_30d = Column(Float)
    temp_avg_7d = Column(Float)
    ndvi_trend = Column(Float)
    
    # Target variable (ce qu'on veut prédire)
    stress_score = Column(Float)
    stress_level = Column(String)  # low, moderate, high, severe
    
    # Source
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


__all__ = [
    "User",
    "Region",
    "WeatherData",
    "SatelliteData",
    "StressAnalysis",
    "TrainingData",
]