from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List
import json
from ..database import get_db
from .. import models

router = APIRouter(prefix="/zones", tags=["Zones"])

@router.get("/wilayas")
async def get_all_wilayas(db: Session = Depends(get_db)):
    """Get all wilayas with their information"""
    
    try:
        # Load from JSON file first (fallback)
        with open('algeria_cordinates.json', 'r') as f:
            coordinates_data = json.load(f)
        
        # Get wilayas from database
        regions = db.query(models.Region).all()
        
        wilayas = []
        
        # Combine database data with coordinates
        for region in regions:
            wilaya_key = f"DZ{str(region.wilaya_code).zfill(2)}"
            coord_data = coordinates_data.get(wilaya_key, {})
            
            wilayas.append({
                "wilaya_code": region.wilaya_code,
                "name": region.name,
                "lat": region.centroid_lat,
                "lon": region.centroid_lon,
                "area_km2": region.area_km2,
                "geometry": region.geometry,
                "has_data": True
            })
        
        # Add any wilayas from JSON that aren't in database
        for code, data in coordinates_data.items():
            wilaya_code = int(code[2:])
            
            # Check if already added
            if not any(w["wilaya_code"] == wilaya_code for w in wilayas):
                wilayas.append({
                    "wilaya_code": wilaya_code,
                    "name": data["name"],
                    "lat": data["lat"],
                    "lon": data["lon"],
                    "area_km2": None,
                    "geometry": None,
                    "has_data": False
                })
        
        # Sort by wilaya code
        wilayas.sort(key=lambda x: x["wilaya_code"])
        
        return {
            "wilayas": wilayas,
            "count": len(wilayas),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wilayas: {str(e)}")

@router.get("/wilayas/{wilaya_code}")
async def get_wilaya_details(wilaya_code: int, db: Session = Depends(get_db)):
    """Get detailed information for a specific wilaya"""
    
    try:
        # Get from database
        region = db.query(models.Region).filter(
            models.Region.wilaya_code == wilaya_code
        ).first()
        
        # Load coordinates from JSON
        with open('algeria_cordinates.json', 'r') as f:
            coordinates_data = json.load(f)
        
        wilaya_key = f"DZ{str(wilaya_code).zfill(2)}"
        coord_data = coordinates_data.get(wilaya_key, {})
        
        if not region and not coord_data:
            raise HTTPException(status_code=404, detail=f"Wilaya {wilaya_code} not found")
        
        # Get statistics for this wilaya
        stats = db.query(
            func.count(models.TrainingData.id).label('record_count'),
            func.min(models.TrainingData.date).label('first_date'),
            func.max(models.TrainingData.date).label('last_date'),
            func.avg(models.TrainingData.stress_score).label('avg_stress'),
            func.avg(models.TrainingData.ndvi).label('avg_ndvi'),
            func.avg(models.TrainingData.precipitation).label('avg_precipitation')
        ).filter(
            models.TrainingData.wilaya_code == wilaya_code
        ).first()
        
        # Get latest stress level
        latest_data = db.query(models.TrainingData).filter(
            models.TrainingData.wilaya_code == wilaya_code
        ).order_by(models.TrainingData.date.desc()).first()
        
        wilaya_info = {
            "wilaya_code": wilaya_code,
            "name": region.name if region else coord_data.get("name", f"Wilaya {wilaya_code}"),
            "lat": region.centroid_lat if region else coord_data.get("lat"),
            "lon": region.centroid_lon if region else coord_data.get("lon"),
            "area_km2": region.area_km2 if region else None,
            "geometry": region.geometry if region else None,
            "in_database": region is not None,
            "statistics": {
                "record_count": stats.record_count or 0,
                "first_date": stats.first_date.isoformat() if stats.first_date else None,
                "last_date": stats.last_date.isoformat() if stats.last_date else None,
                "avg_stress": float(stats.avg_stress) if stats.avg_stress else None,
                "avg_ndvi": float(stats.avg_ndvi) if stats.avg_ndvi else None,
                "avg_precipitation": float(stats.avg_precipitation) if stats.avg_precipitation else None,
                "current_stress_level": latest_data.stress_level if latest_data else None,
                "current_stress_score": float(latest_data.stress_score) if latest_data and latest_data.stress_score else None
            }
        }
        
        return wilaya_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wilaya details: {str(e)}")