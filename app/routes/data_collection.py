# routes/data_collection.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func 
from typing import Optional
from ..services.weather_service import WeatherService
from ..services.satellite_service import SatelliteService
from ..services.daily_data_collector import DailyDataCollector
from ..database import get_db
from .. import models
import json
import asyncio

router = APIRouter(prefix="/data-collection", tags=["Data Collection"])

# @router.post("/collect/wilaya/{wilaya_code}")
# async def collect_wilaya_data(
#     wilaya_code: str,
#     start_year: int = 2015,
#     end_year: int = 2025,
#     background_tasks: BackgroundTasks = None,
#     db: Session = Depends(get_db)
# ):
#     """
#     Lancer la collecte de données pour une wilaya
#     (Peut être exécuté en arrière-plan)
#     """
#     # Initialiser les services
#     weather_service = WeatherService()
#     satellite_service = SatelliteService()
#     collector = DailyDataCollector(weather_service, satellite_service)
    
#     if background_tasks:
#         # Exécuter en arrière-plan
#         background_tasks.add_task(
#             collector.collect_historical_data_for_wilaya,
#             wilaya_code, start_year, end_year, db
#         )
        
#         return {
#             "message": f"Collecte démarrée en arrière-plan pour la wilaya {wilaya_code}",
#             "task": "started"
#         }
#     else:
#         # Exécuter immédiatement
#         result = await collector.collect_historical_data_for_wilaya(
#             wilaya_code, start_year, end_year, db
#         )
        
#         return result
        
@router.post("/collect/wilaya/daily/{wilaya_code}")
async def collect_wilaya_data(
    wilaya_code: str,
    start_year: int = 2015,
    end_year: int = 2025,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Lancer la collecte de données pour une wilaya
    (Peut être exécuté en arrière-plan)
    """
    # Initialiser les services
    weather_service = WeatherService()
    satellite_service = SatelliteService()
    collector = DailyDataCollector(weather_service, satellite_service)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    if background_tasks:
        # Exécuter en arrière-plan
        background_tasks.add_task(
            collector.collect_daily_data_for_period,
            wilaya_code, start_date, end_date, db
        )
        
        return {
            "message": f"Collecte daily démarrée en arrière-plan pour la wilaya {wilaya_code}",
            "task": "started"
        }
    else:
        # Exécuter immédiatement
        result = await collector.collect_daily_data_for_period(
            wilaya_code, start_date, end_date, db
        )
        
        return result

@router.post("/collect/all-wilayas")
async def collect_all_wilayas_data(
    start_year: int = 2015,
    end_year: int = 2025,
    data_type: str = "daily",  # "daily" or "historical"
    batch_size: int = 5,  # Process N wilayas at a time
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Collect data for all Algerian wilayas
    """
    # Load all wilayas from JSON
    with open('algeria_cordinates.json', 'r') as f:
        algeria_data = json.load(f)
    
    wilaya_codes = list(algeria_data.keys())
    
    if background_tasks:
        background_tasks.add_task(
            collect_wilayas_batch,
            wilaya_codes, start_year, end_year, data_type, batch_size, db
        )
        
        return {
            "message": f"Started collection for all {len(wilaya_codes)} wilayas",
            "wilaya_count": len(wilaya_codes),
            "batch_size": batch_size,
            "data_type": data_type
        }
    else:
        results = await collect_wilayas_batch(
            wilaya_codes, start_year, end_year, data_type, batch_size, db
        )
        
        return {
            "results": results,
            "total_wilayas": len(results),
            "successful": len([r for r in results if r.get('success')])
        }

async def collect_wilayas_batch(wilaya_codes, start_year, end_year, data_type, batch_size, db):
    """Process wilayas in batches to avoid overwhelming APIs"""
    results = []
    
    for i in range(0, len(wilaya_codes), batch_size):
        batch = wilaya_codes[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {batch}")
        
        # Process each wilaya in the batch
        batch_tasks = []
        for wilaya_key in batch:
            wilaya_code = wilaya_key[2:]  # Remove "DZ" prefix
            
            if data_type == "daily":
                collector = DailyDataCollector(WeatherService(), SatelliteService())
                task = collector.collect_daily_data_for_period(
                    wilaya_code, 
                    f"{start_year}-01-01", 
                    f"{end_year}-12-31", 
                    db
                )
            else:
                collector = DataCollector(WeatherService(), SatelliteService())
                task = collector.collect_historical_data_for_wilaya(
                    wilaya_code, start_year, end_year, db
                )
            
            batch_tasks.append(task)
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for wilaya_key, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                results.append({
                    "wilaya": wilaya_key,
                    "success": False,
                    "error": str(result)
                })
            else:
                results.append({
                    "wilaya": wilaya_key,
                    "success": True,
                    **result
                })
        
        # Small delay between batches
        await asyncio.sleep(2)
    
    return results

@router.get("/status")
async def get_collection_status(
    wilaya_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Vérifier le statut des données collectées
    """
    try:
        query = db.query(models.TrainingData)
        
        if wilaya_code:
            query = query.filter(models.TrainingData.wilaya_code == int(wilaya_code))
        
        count = query.count()
        
        # Statistiques par wilaya - CORRIGER CES LIGNES :
        wilaya_stats = db.query(
            models.TrainingData.wilaya_code,
            func.count(models.TrainingData.id).label('count'),  # REMPLACER db.func par func
            func.min(models.TrainingData.date).label('min_date'),
            func.max(models.TrainingData.date).label('max_date')
        ).group_by(models.TrainingData.wilaya_code).all()
        
        # Récupérer les noms des wilayas depuis la table Region
        stats_with_names = []
        for stat in wilaya_stats:
            region = db.query(models.Region).filter(
                models.Region.wilaya_code == stat.wilaya_code
            ).first()
            
            wilaya_name = region.name if region else f"Wilaya {stat.wilaya_code}"
            
            stats_with_names.append({
                "wilaya_code": stat.wilaya_code,
                "wilaya_name": wilaya_name,
                "record_count": stat.count,
                "date_range": {
                    "start": stat.min_date.strftime("%Y-%m-%d") if stat.min_date else None,
                    "end": stat.max_date.strftime("%Y-%m-%d") if stat.max_date else None
                }
            })
        
        return {
            "total_records": count,
            "wilaya_stats": stats_with_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")