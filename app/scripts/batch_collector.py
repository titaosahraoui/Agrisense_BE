import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..services.weather_service import WeatherService
from ..services.satellite_service import SatelliteService
from ..services.daily_data_collector import DailyDataCollector

async def collect_complete_dataset():
    """
    Collect complete dataset from 2018 to 2025
    """
    db = SessionLocal()
    
    try:
        # Initialize services
        weather_service = WeatherService()
        satellite_service = SatelliteService()
        collector = DailyDataCollector(weather_service, satellite_service)
        
        # Wilayas to collect (example: wilaya 16 - Alger)
        wilayas_to_collect = [16]  # Add more wilaya codes as needed
        
        for wilaya_code in wilayas_to_collect:
            print(f"\n{'='*60}")
            print(f"Collecting data for Wilaya {wilaya_code}")
            print(f"{'='*60}")
            
            # Collect from 2018-01-01 to 2025-12-31
            result = await collector.collect_daily_data_for_period(
                wilaya_code=str(wilaya_code),
                start_date="2018-01-01",
                end_date="2025-12-31",
                db=db
            )
            
            print(f"\nResult for Wilaya {wilaya_code}:")
            print(f"  Days collected: {result['days_collected']}")
            print(f"  Days saved: {result['days_saved']}")
            print(f"  Period: {result['period']}")
    
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(collect_complete_dataset())