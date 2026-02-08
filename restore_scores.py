from app.database import SessionLocal
from app.models import TrainingData
from app.services.stress_analysis import StressAnalysisService, RegionType
import sys

def restore():
    print("🔄 Restoring stress scores to original Rule-Based Logic...")
    
    db = SessionLocal()
    # Initialize service with default region or handle dynamic regions if possible
    # For now, default to MEDITERRANEAN as basic restoration
    service = StressAnalysisService(RegionType.MEDITERRANEAN)
    
    try:
        records = db.query(TrainingData).all()
        total = len(records)
        print(f"📊 Found {total} records to process.")
        
        count = 0
        for r in records:
            indicators = {
                'ndvi': r.ndvi,
                'temperature': r.temperature_avg,
                'precipitation': r.precipitation,
                'evapotranspiration': r.evapotranspiration,
                'humidity': r.humidity
            }
            # Remove None values so the service handles them correctly
            indicators = {k: v for k, v in indicators.items() if v is not None}
            
            # Recalculate using pure rule-based logic
            result = service.calculate_stress_score(indicators)
            if isinstance(result, tuple):
                score = result[0]
            else:
                score = result
            
            level = service.determine_stress_level(score)
            
            # Update record
            r.stress_score = score
            r.stress_level = level
            
            count += 1
            if count % 1000 == 0:
                print(f"   Processed {count}/{total}...")
        
        db.commit()
        print(f"✅ Successfully restored {count} records to rule-based scores.")
        
    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    restore()
