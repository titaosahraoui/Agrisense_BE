
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy.orm import Session
import json
from .. import models
from .stress_analysis import StressAnalysisService

class DailyDataCollector:
    def __init__(self, weather_service, satellite_service):
        self.weather_service = weather_service
        self.satellite_service = satellite_service
        
    async def collect_daily_data_for_period(
        self,
        wilaya_code: str,
        start_date: str,
        end_date: str,
        db: Session
    ) -> Dict:
        """
        Collect DAILY data for the specified period
        """
        print(f"Collecting daily data for wilaya {wilaya_code} from {start_date} to {end_date}")
        
        # Load coordinates
        with open('algeria_cordinates.json', 'r') as f:
            algeria_data = json.load(f)
        
        wilaya_key = f"DZ{wilaya_code.zfill(2)}"
        if wilaya_key not in algeria_data:
            raise ValueError(f"Wilaya {wilaya_code} not found")
        
        wilaya_info = algeria_data[wilaya_key]
        
        # Get region from database or create it
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
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        raw_data_buffer = []  # Buffer for raw data before cleaning
        current_year = start_dt.year
        end_year = end_dt.year

        for year in range(current_year, end_year + 1):
            print(f"\nProcessing year {year}...")
            
            # Determine start and end for this year
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31)
            
            if year == current_year:
                year_start = max(year_start, start_dt)
            if year == end_year:
                year_end = min(year_end, end_dt)
            
            # Get daily weather data
            nasa_start = year_start.strftime("%Y%m%d")
            nasa_end = year_end.strftime("%Y%m%d")
            
            weather_data = await self.weather_service.get_nasa_power_data(
                region.centroid_lat,
                region.centroid_lon,
                nasa_start,
                nasa_end
            )
            
            if not weather_data:
                print(f"  ⚠️ No weather data for {year}")
                continue
            
            # Process each month for satellite data (monthly composites)
            for month in range(1, 13):
                month_start = datetime(year, month, 1)
                month_end = (datetime(year, month + 1, 1) if month < 12 
                           else datetime(year + 1, 1, 1)) - timedelta(days=1)
                
                # Skip if outside our range
                if month_end < year_start or month_start > year_end:
                    continue
                
                print(f"  Month {month:02d}: ", end="")
                
                # Get monthly satellite data (composite)
                if region.geometry:
                    geometry_to_use = region.geometry
                else:
                    geometry_to_use = {
                        "type": "Point",
                        "coordinates": [region.centroid_lon, region.centroid_lat]
                    }
                
                satellite_stats = self.satellite_service.get_region_statistics(
                    geometry_to_use,
                    month_start.strftime("%Y-%m-%d"),
                    month_end.strftime("%Y-%m-%d")
                )
                
                if satellite_stats:
                    print(f"✓ Satellite data available")
                else:
                    print(f"✗ No satellite data")
                
                # Assign satellite data to each day of the month
                daily_weather_for_month = [
                    w for w in weather_data 
                    if month_start.date() <= w['date'].date() <= month_end.date()
                ]
                
                # Collect RAW data for batch cleaning
                for daily_weather in daily_weather_for_month:
                    raw_entry = {
                        'wilaya_code': int(wilaya_code),
                        'date': daily_weather['date'],
                        'year': year,
                        
                        # Weather fields
                        'temperature_avg': daily_weather.get('temperature_avg'),
                        'temperature_max': daily_weather.get('temperature_max'),
                        'temperature_min': daily_weather.get('temperature_min'),
                        'precipitation': daily_weather.get('precipitation'),
                        'humidity': daily_weather.get('humidity'),
                        'solar_radiation': daily_weather.get('solar_radiation'),
                        'wind_speed': daily_weather.get('wind_speed'),
                        'et0_fao': daily_weather.get('et0_fao'),
                        
                        # Satellite fields (Mapped to lowercase for DataCleaner)
                        'ndvi': satellite_stats.get('NDVI') if satellite_stats else None,
                        'ndwi': satellite_stats.get('NDWI') if satellite_stats else None,
                        'lst': satellite_stats.get('LST') if satellite_stats else None,
                    }
                    raw_data_buffer.append(raw_entry)
            
            print(f"  Year {year}: Collected raw data points: {len([d for d in raw_data_buffer if d['date'].year == year])}")
        
        # 1. Clean collected data
        # We collected ALL raw data first to allow efficient time-series interpolation/imputation
        all_daily_data = []
        
        if raw_data_buffer:
            print(f"\n🧹 Cleaning {len(raw_data_buffer)} raw records before processing...")
            from .data_cleaner import DataCleaner
            cleaner = DataCleaner(db)
            
            # Convert to DataFrame
            df_raw = pd.DataFrame(raw_data_buffer)
            
            # Clean (Interpolate Satellite gaps, Impute Weather gaps) - No Normalization
            df_cleaned = cleaner.clean_daily_validation_data(df_raw)
            
            print("✨ Creating final training entries with calculated stress scores...")
            
            # 2. Create entries from CLEANED data
            cleaned_records = df_cleaned.to_dict('records')
            
            for row in cleaned_records:
                # Reconstruct inputs for _create_daily_training_entry
                # This ensures we use the CLEANED values for stress calculation
                
                simulated_weather = {
                    'temperature_avg': row.get('temperature_avg'),
                    'temperature_max': row.get('temperature_max'),
                    'temperature_min': row.get('temperature_min'),
                    'precipitation': row.get('precipitation'),
                    'humidity': row.get('humidity'),
                    'solar_radiation': row.get('solar_radiation'),
                    'wind_speed': row.get('wind_speed'),
                    'et0_fao': row.get('et0_fao'),
                    'date': row.get('date') # vital
                }
                
                # Map back to Uppercase for consistency if needed, though service handles both usually.
                # But _create_daily_training_entry expects dict to extract 'NDVI' etc.
                simulated_sat = {
                    'NDVI': row.get('ndvi'),
                    'NDWI': row.get('ndwi'),
                    'LST': row.get('lst')
                } if row.get('ndvi') is not None else {}
                
                entry = self._create_daily_training_entry(
                    wilaya_code=int(row['wilaya_code']),
                    date=row['date'],
                    weather_data=simulated_weather,
                    satellite_stats=simulated_sat,
                    year=row.get('year', row['date'].year)
                )
                
                if entry:
                    all_daily_data.append(entry)

        # Save to database
        saved_count = await self._save_daily_training_data(all_daily_data, db)
        
        return {
            "wilaya": wilaya_info['name'],
            "wilaya_code": wilaya_code,
            "period": f"{start_date} to {end_date}",
            "days_collected": len(all_daily_data),
            "days_saved": saved_count,
            "sample_dates": [d['date'].strftime("%Y-%m-%d") for d in all_daily_data[:5]] if all_daily_data else []
        }
    
    def _create_daily_training_entry(self, wilaya_code: int, date: datetime, 
                                   weather_data: Dict, satellite_stats: Dict, year: int) -> Dict:
        """Create a daily training data entry with stress score calculated from clean data"""
        
        # Prepare indicators for stress calculation
        indicators = {
            'temperature': weather_data.get('temperature_avg'),
            'precipitation': weather_data.get('precipitation'),
            'humidity': weather_data.get('humidity'),
            'evapotranspiration': weather_data.get('et0_fao'),
            'ndvi': satellite_stats.get('NDVI'),
            'lst': satellite_stats.get('LST'),
            'ndwi': satellite_stats.get('NDWI')
        }
        
        # Calculate stress score immediately
        from .stress_analysis import StressAnalysisService
        analysis_service = StressAnalysisService()
        
        try:
            stress_score = analysis_service.calculate_stress_score(indicators)
            stress_level = analysis_service.determine_stress_level(stress_score)
        except Exception:
            # Fallback if calculation fails (though data should be clean now)
            stress_score = 0.5
            stress_level = "moderate"
        
        return {
            'wilaya_code': wilaya_code,
            'date': date,
            
            # Weather data
            'temperature_avg': weather_data.get('temperature_avg'),
            'temperature_max': weather_data.get('temperature_max'),
            'temperature_min': weather_data.get('temperature_min'),
            'precipitation': weather_data.get('precipitation'),
            'humidity': weather_data.get('humidity'),
            'solar_radiation': weather_data.get('solar_radiation'),
            'wind_speed': weather_data.get('wind_speed'),
            'evapotranspiration': weather_data.get('et0_fao'),
            
            # Satellite data
            'ndvi': satellite_stats.get('NDVI'),
            'ndwi': satellite_stats.get('NDWI'),
            'lst': satellite_stats.get('LST'),
            
            # Time features
            'month': date.month,
            'season': self._get_season(date.month),
            'day_of_year': date.timetuple().tm_yday,
            'year': date.year,
            
            # Target
            'stress_score': stress_score,
            'stress_level': stress_level,
            
            # Meta
            'source': f'NASA_GEE_{year}',
            'created_at': datetime.utcnow()
        }
    
    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "hiver"
        elif month in [3, 4, 5]:
            return "printemps"
        elif month in [6, 7, 8]:
            return "été"
        else:
            return "automne"
    
    
    async def _save_daily_training_data(self, data: List[Dict], db: Session) -> int:
        """Save daily training data to database"""
        saved = 0
        
        for entry in data:
            # Check if entry already exists
            exists = db.query(models.TrainingData).filter(
                models.TrainingData.wilaya_code == entry['wilaya_code'],
                models.TrainingData.date == entry['date']
            ).first()
            
            if not exists:
                # Convert numpy types
                converted_entry = self._convert_numpy_types(entry)
                # Drop fields not present in TrainingData model (keep DB schema consistent)
                converted_entry.pop("year", None)
                training_data = models.TrainingData(**converted_entry)
                db.add(training_data)
                saved += 1
        
        db.commit()
        return saved
    
    def _convert_numpy_types(self, data_dict: Dict) -> Dict:
        """Convert NumPy types to Python native types"""
        import numpy as np
        
        converted = {}
        for key, value in data_dict.items():
            if value is None:
                converted[key] = None
            elif isinstance(value, (np.float64, np.float32)):
                converted[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                converted[key] = int(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            else:
                converted[key] = value
        
        return converted