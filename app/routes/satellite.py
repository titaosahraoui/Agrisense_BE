from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
from ..services.satellite_service import SatelliteService
from ..database import get_db
from .. import models, schemas

router = APIRouter(prefix="/satellite", tags=["Satellite Data"])
satellite_service = SatelliteService()

@router.get("/indices/{region_id}")
async def get_satellite_indices(
    region_id: int,
    start_date: str = Query((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")),
    end_date: str = Query(datetime.now().strftime("%Y-%m-%d")),
    db: Session = Depends(get_db)
):
    """Récupérer les indices satellitaires pour une région"""
    region = db.query(models.Region).filter(models.Region.id == region_id).first()
    if not region:
        raise HTTPException(status_code=404, detail="Région non trouvée")
    
    # Récupérer depuis Google Earth Engine
    stats = satellite_service.get_region_statistics(
        region.geometry,
        start_date,
        end_date
    )
    
    if not stats:
        raise HTTPException(status_code=404, detail="Aucune donnée satellitaire disponible")
    
    # Sauvegarder dans la base de données
    satellite_data = models.SatelliteData(
        region_id=region_id,
        acquisition_date=datetime.now(),
        satellite="Sentinel-2",
        ndvi=stats.get('NDVI', 0),
        evi=stats.get('EVI', 0),
        ndwi=stats.get('NDWI', 0),
        lst=stats.get('LST', 0),
        geometry=region.geometry
    )
    
    db.add(satellite_data)
    db.commit()
    db.refresh(satellite_data)
    
    return satellite_data