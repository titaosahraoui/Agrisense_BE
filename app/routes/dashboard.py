# routes/dashboard.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, distinct
from datetime import datetime, timedelta
from typing import List, Optional
from ..database import get_db
from .. import models

from ..services.satellite_service import SatelliteService
from datetime import datetime, timedelta

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
satellite_service = SatelliteService()

@router.get("/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get dashboard summary statistics"""
    
    try:
        # Get total records from TrainingData
        total_records = db.query(func.count(models.TrainingData.id)).scalar() or 0
        
        # Get unique wilayas
        wilaya_count = db.query(func.count(func.distinct(models.TrainingData.wilaya_code))).scalar() or 0
        
        # Get latest training date
        latest_training = db.query(func.max(models.TrainingData.created_at)).scalar()
        
        # Get active alerts (high/severe stress in last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        alerts = db.query(func.count(models.TrainingData.id)).filter(
            and_(
                models.TrainingData.date >= week_ago,
                models.TrainingData.stress_level.in_(["high", "severe"])
            )
        ).scalar() or 0
        
        return {
            "summary": {
                "total_records": total_records,
                "total_wilayas": wilaya_count,
                "alerts_count": alerts,
                "last_training": latest_training.isoformat() if latest_training else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard summary: {str(e)}")

@router.get("/wilayas/stats")
async def get_wilayas_statistics(
    stress_level: Optional[str] = Query(None, description="Filter by stress level"),
    limit: Optional[int] = Query(None, description="Limit results"),
    db: Session = Depends(get_db)
):
    """Get statistics for all wilayas"""
    
    try:
        
        # Check if we have any training data
        total_records = db.query(func.count(models.TrainingData.id)).scalar() or 0
        
        if total_records == 0:
            # Return empty array with message
            return {
                "data": [],
                "count": 0,
                "message": "No training data available. Please run data collection first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get date for last 30 days
        month_ago = datetime.utcnow() - timedelta(days=30)
        
        # First, get all distinct wilaya codes from TrainingData (those that actually have data)
        wilaya_codes_with_data = db.query(
            func.distinct(models.TrainingData.wilaya_code)
        ).all()
        wilaya_codes_with_data = [code[0] for code in wilaya_codes_with_data]
        
        if not wilaya_codes_with_data:
            return {
                "data": [],
                "count": 0,
                "message": "No wilayas have training data yet.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get regions for these wilayas
        regions = db.query(models.Region).filter(
            models.Region.wilaya_code.in_(wilaya_codes_with_data)
        ).all()
        
        # If no regions in database, create fallback from coordinates
        if not regions:
            # Load coordinates file
            import json
            try:
                with open('algeria_cordinates.json', 'r') as f:
                    coordinates_data = json.load(f)
                
                regions = []
                for code in wilaya_codes_with_data:
                    wilaya_key = f"DZ{str(code).zfill(2)}"
                    if wilaya_key in coordinates_data:
                        coord_data = coordinates_data[wilaya_key]
                        # Create a mock region
                        regions.append(type('Region', (), {
                            'wilaya_code': code,
                            'name': coord_data['name'],
                            'centroid_lat': coord_data['lat'],
                            'centroid_lon': coord_data['lon']
                        })())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading coordinates: {str(e)}")
        
        wilaya_stats = []
        
        for region in regions:
            # Get latest data for this wilaya
            latest_data = db.query(models.TrainingData).filter(
                models.TrainingData.wilaya_code == region.wilaya_code
            ).order_by(models.TrainingData.date.desc()).first()
            
            if not latest_data:
                # Skip if no data at all for this wilaya
                continue
            
            # Apply stress level filter
            if stress_level and latest_data.stress_level != stress_level:
                continue
            
            # Calculate average statistics for last 30 days in a single query
            avg_scores = db.query(
                func.avg(models.TrainingData.stress_score).label('avg_stress_score'),
                func.avg(models.TrainingData.ndvi).label('avg_ndvi'),
                func.avg(models.TrainingData.ndwi).label('avg_ndwi'),
                func.avg(models.TrainingData.lst).label('avg_lst'),
                func.avg(models.TrainingData.precipitation).label('avg_precipitation'),
                func.avg(models.TrainingData.temperature_avg).label('avg_temperature'),
                func.count(models.TrainingData.id).label('record_count')
            ).filter(
                and_(
                    models.TrainingData.wilaya_code == region.wilaya_code,
                    models.TrainingData.date >= month_ago
                )
            ).first()
            
            # Get date range for this wilaya
            date_range = db.query(
                func.min(models.TrainingData.date).label('min_date'),
                func.max(models.TrainingData.date).label('max_date')
            ).filter(
                models.TrainingData.wilaya_code == region.wilaya_code
            ).first()
            
            # Prepare the wilaya stats
            wilaya_stat = {
                "wilaya_code": region.wilaya_code,
                "wilaya_name": region.name if hasattr(region, 'name') else f"Wilaya {region.wilaya_code}",
                "stress_score": float(latest_data.stress_score) if latest_data.stress_score is not None else None,
                "stress_level": latest_data.stress_level,
                "ndvi": float(latest_data.ndvi) if latest_data.ndvi is not None else None,
                "ndwi": float(latest_data.ndwi) if latest_data.ndwi is not None else None,
                "lst": float(latest_data.lst) if latest_data.lst is not None else None,
                "last_update": latest_data.date.isoformat() if latest_data.date else None,
                "avg_stress_score": float(avg_scores.avg_stress_score) if avg_scores.avg_stress_score is not None else None,
                "avg_ndvi": float(avg_scores.avg_ndvi) if avg_scores.avg_ndvi is not None else None,
                "avg_ndwi": float(avg_scores.avg_ndwi) if avg_scores.avg_ndwi is not None else None,
                "avg_lst": float(avg_scores.avg_lst) if avg_scores.avg_lst is not None else None,
                "avg_precipitation": float(avg_scores.avg_precipitation) if avg_scores.avg_precipitation is not None else None,
                "avg_temperature": float(avg_scores.avg_temperature) if avg_scores.avg_temperature is not None else None,
                "record_count": avg_scores.record_count or 0,
                "date_range": {
                    "start": date_range.min_date.isoformat() if date_range and date_range.min_date else None,
                    "end": date_range.max_date.isoformat() if date_range and date_range.max_date else None
                }
            }
            
            wilaya_stats.append(wilaya_stat)
        
        # Sort by average stress score (highest first)
        wilaya_stats.sort(key=lambda x: (x.get('avg_stress_score') or 0), reverse=True)
        
        # Apply limit if specified
        if limit:
            wilaya_stats = wilaya_stats[:limit]
        
        # Return as a consistent response format
        # return wilaya_stat
        return {
            "data": wilaya_stats,
            "count": len(wilaya_stats),
            "has_real_data": total_records > 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_wilayas_statistics: {e}")
        print(f"Traceback: {error_details}")
        
@router.get("/climate-summary")
async def get_climate_summary(
    time_range: str = Query("month", description="Time range: week, month, or season"),
    db: Session = Depends(get_db)
):
    """Get climate summary for dashboard"""
    
    try:
        # Define days based on time range
        if time_range == "week":
            days_back = 7
        elif time_range == "season":
            days_back = 90
        else:  # month
            days_back = 30
        
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Get climate data from training data
        climate_data = db.query(
            func.avg(models.TrainingData.temperature_avg).label('avg_temperature'),
            func.sum(models.TrainingData.precipitation).label('total_precipitation'),
            func.avg(models.TrainingData.humidity).label('avg_humidity'),
            func.sum(models.TrainingData.evapotranspiration).label('total_evapotranspiration'),
            func.avg(models.TrainingData.ndvi).label('avg_ndvi'),
            func.avg(models.TrainingData.ndwi).label('avg_ndwi'),
            func.count(models.TrainingData.id).label('data_points')
        ).filter(
            models.TrainingData.date >= start_date
        ).first()
        
        # Calculate derived metrics
        water_balance = 0
        if climate_data.total_precipitation and climate_data.total_evapotranspiration:
            water_balance = climate_data.total_precipitation - climate_data.total_evapotranspiration
        
        aridity_index = 0
        if climate_data.total_evapotranspiration and climate_data.total_precipitation:
            aridity_index = climate_data.total_evapotranspiration / (climate_data.total_precipitation + 1e-6)
        
        return {
            "time_range": time_range,
            "days_analyzed": days_back,
            "data_points": climate_data.data_points or 0,
            "avg_temperature": round(float(climate_data.avg_temperature), 2) if climate_data.avg_temperature else None,
            "total_precipitation": round(float(climate_data.total_precipitation), 1) if climate_data.total_precipitation else None,
            "avg_humidity": round(float(climate_data.avg_humidity), 1) if climate_data.avg_humidity else None,
            "total_evapotranspiration": round(float(climate_data.total_evapotranspiration), 1) if climate_data.total_evapotranspiration else None,
            "water_balance": round(float(water_balance), 1),
            "aridity_index": round(float(aridity_index), 2),
            "avg_ndvi": round(float(climate_data.avg_ndvi), 3) if climate_data.avg_ndvi else None,
            "avg_ndwi": round(float(climate_data.avg_ndwi), 3) if climate_data.avg_ndwi else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching climate summary: {str(e)}")

@router.get("/timeline/{wilaya_code}")
async def get_stress_timeline(
    wilaya_code: int,
    days: int = Query(30, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """Get stress timeline data for a specific wilaya"""
    
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        timeline_data = db.query(models.TrainingData).filter(
            and_(
                models.TrainingData.wilaya_code == wilaya_code,
                models.TrainingData.date >= start_date
            )
        ).order_by(models.TrainingData.date.asc()).all()
        
        if not timeline_data:
            raise HTTPException(status_code=404, detail="No timeline data found for this wilaya")
        
        # Format response
        formatted_data = []
        for record in timeline_data:
            formatted_data.append({
                "date": record.date.isoformat() if record.date else None,
                "stress_score": float(record.stress_score) if record.stress_score else None,
                "stress_level": record.stress_level,
                "precipitation": float(record.precipitation) if record.precipitation else None,
                "temperature_avg": float(record.temperature_avg) if record.temperature_avg else None,
                "humidity": float(record.humidity) if record.humidity else None,
                "ndvi": float(record.ndvi) if record.ndvi else None,
                "ndwi": float(record.ndwi) if record.ndwi else None,
                "lst": float(record.lst) if record.lst else None
            })
        
        return {
            "wilaya_code": wilaya_code,
            "days": days,
            "data": formatted_data,
            "count": len(formatted_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching timeline data: {str(e)}")

@router.get("/satellite/indices")
async def get_satellite_indices(
    wilaya_code: Optional[int] = Query(None, description="Filter by wilaya code"),
    latest: int = Query(5, description="Number of latest records to return"),
    db: Session = Depends(get_db)
):
    """Get satellite indices data"""
    
    try:
        query = db.query(models.TrainingData).filter(
            models.TrainingData.ndvi.isnot(None)
        )
        
        if wilaya_code:
            query = query.filter(models.TrainingData.wilaya_code == wilaya_code)
        
        # Get latest records
        satellite_data = query.order_by(models.TrainingData.date.desc()).limit(latest).all()
        
        if not satellite_data:
            # Return sample data if no real data
            # Use the service method instead of local function
            return satellite_service.generate_sample_data(latest, wilaya_code)
        
        # Format response
        formatted_data = []
        for record in satellite_data:
            # Get wilaya name
            region = db.query(models.Region).filter(
                models.Region.wilaya_code == record.wilaya_code
            ).first()
            
            formatted_data.append({
                "name": region.name if region else f"Wilaya {record.wilaya_code}",
                "date": record.date.isoformat() if record.date else None,
                "ndvi": float(record.ndvi) if record.ndvi else None,
                "ndwi": float(record.ndwi) if record.ndwi else None,
                "lst": float(record.lst) if record.lst else None,
                "stress_score": float(record.stress_score) if record.stress_score else None,
                "wilaya_code": record.wilaya_code
            })
        
        return formatted_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching satellite indices: {str(e)}")