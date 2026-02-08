from app.database import SessionLocal
from app.models import TrainingData, WeatherData
from sqlalchemy import func, extract

def check_coverage():
    session = SessionLocal()
    try:
        print("Checking TrainingData coverage:")
        query = session.query(
            extract('year', TrainingData.date).label('year'),
            func.count(TrainingData.id),
            func.count(func.distinct(TrainingData.wilaya_code))
        ).group_by('year').order_by('year')
        
        for year, count, wilayas in query.all():
            print(f"Year {int(year or 0)}: {count} records, covering {wilayas} wilayas")
            
        print("\nChecking WeatherData coverage:")
        query = session.query(
            extract('year', WeatherData.date).label('year'),
            func.count(WeatherData.id)
        ).group_by('year').order_by('year')
        
        for year, count in query.all():
            print(f"Year {int(year or 0)}: {count} records")

    finally:
        session.close()

if __name__ == "__main__":
    check_coverage()
