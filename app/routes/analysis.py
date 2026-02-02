from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta
from ..database import get_db
from .. import models

router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.get("/latest")
async def get_latest_analysis(
    wilaya_code: Optional[int] = Query(None, description="Filter by wilaya code"),
    limit: int = Query(10, description="Limit results"),
    db: Session = Depends(get_db)
):
    """Get latest stress analysis records"""
    try:
        query = db.query(models.StressAnalysis)
        
        if wilaya_code:
            query = query.join(models.Region).filter(models.Region.wilaya_code == wilaya_code)
            
        analyses = query.order_by(models.StressAnalysis.analysis_date.desc()).limit(limit).all()
        
        return analyses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_analysis_summary(db: Session = Depends(get_db)):
    """Get summary of stress levels"""
    try:
        results = db.query(
            models.StressAnalysis.stress_level, 
            func.count(models.StressAnalysis.id)
        ).group_by(models.StressAnalysis.stress_level).all()
        
        # Convert to dictionary with default values
        summary = {
            "low": 0,
            "moderate": 0,
            "high": 0,
            "severe": 0
        }
        
        for level, count in results:
            if level:
                summary[level.lower()] = count
                
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
