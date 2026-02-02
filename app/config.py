import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # JWT
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    
    # API Keys
    NASA_API_KEY = os.getenv("NASA_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    GOOGLE_EARTH_ENGINE_KEY = os.getenv("GOOGLE_EARTH_ENGINE_KEY")
    
    # Google Earth Engine
    GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT")
    GEE_PRIVATE_KEY = os.getenv("GEE_PRIVATE_KEY")
    
    # Algérie Météo API (si disponible)
    ALGERIA_WEATHER_API = os.getenv("ALGERIA_WEATHER_API_URL")

settings = Settings()