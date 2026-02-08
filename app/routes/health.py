# routes/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..database import get_db
from ..services.prediction_service import PredictionService
import os

router = APIRouter(prefix="/health", tags=["Health & Testing"])

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Water Stress Prediction API",
        "version": "2.0.0"
    }

@router.get("/database")
async def check_database(db: Session = Depends(get_db)):
    """Check database connectivity"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        
        # Count records
        from .. import models
        training_count = db.query(models.TrainingData).count()
        region_count = db.query(models.Region).count()
        
        return {
            "status": "connected",
            "training_records": training_count,
            "regions": region_count
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/models")
async def check_models():
    """Check available models"""
    try:
        service = PredictionService()
        info = service.get_model_info()
        
        # Check model directory
        possible_dirs = ["ml_Models", "models"]
        model_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                model_dir = d
                break
        
        model_files = []
        if model_dir:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        return {
            "status": "ok",
            "model_directory": model_dir,
            "models_on_disk": len(model_files),
            "current_mode": info['current_mode'],
            "supervised_loaded": info['supervised_available'],
            "hybrid_loaded": info['hybrid_available'],
            "rule_based_available": info['rule_based_available']
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.post("/test/prediction")
async def test_prediction():
    """Test prediction with sample data"""
    try:
        service = PredictionService()
        
        # Sample environmental data
        test_features = {
            'ndvi': 0.45,
            'temperature': 32,
            'precipitation': 2.5,
            'evapotranspiration': 6.0,
            'humidity': 55
        }
        
        result = service.predict_stress(test_features)
        
        return {
            "status": "success",
            "test_features": test_features,
            "prediction": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/system/info")
async def system_info():
    """Get system information"""
    import sys
    import platform
    
    # Check if Earth Engine is initialized
    ee_status = "unknown"
    try:
        import ee
        ee.Initialize()
        ee_status = "initialized"
    except Exception as e:
        ee_status = f"not initialized: {str(e)[:50]}"
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "earth_engine_status": ee_status,
        "working_directory": os.getcwd(),
        "model_directories": {
            "ml_Models": os.path.exists("ml_Models"),
            "models": os.path.exists("models")
        }
    }

@router.get("/test/all")
async def test_all_systems(db: Session = Depends(get_db)):
    """Run comprehensive system test"""
    results = {
        "database": "not tested",
        "models": "not tested",
        "prediction": "not tested",
        "earth_engine": "not tested"
    }
    
    # Test database
    try:
        db.execute(text("SELECT 1"))
        results["database"] = "✅ connected"
    except Exception as e:
        results["database"] = f"❌ {str(e)[:50]}"
    
    # Test models
    try:
        service = PredictionService()
        info = service.get_model_info()
        if info['supervised_available'] or info['hybrid_available']:
            results["models"] = f"✅ {info['current_mode']} mode"
        else:
            results["models"] = "⚠️ no models loaded (rule-based only)"
    except Exception as e:
        results["models"] = f"❌ {str(e)[:50]}"
    
    # Test prediction
    try:
        service = PredictionService()
        test_result = service.predict_stress({
            'ndvi': 0.45,
            'temperature': 32,
            'precipitation': 2.5,
            'evapotranspiration': 6.0,
            'humidity': 55
        })
        results["prediction"] = f"✅ score: {test_result['predicted_stress_score']:.1f}"
    except Exception as e:
        results["prediction"] = f"❌ {str(e)[:50]}"
    
    # Test Earth Engine
    try:
        import ee
        ee.Initialize()
        results["earth_engine"] = "✅ initialized"
    except Exception as e:
        results["earth_engine"] = f"❌ {str(e)[:50]}"
    
    # Overall status
    failures = [k for k, v in results.items() if "❌" in str(v)]
    warnings = [k for k, v in results.items() if "⚠️" in str(v)]
    
    overall = "healthy"
    if failures:
        overall = "unhealthy"
    elif warnings:
        overall = "degraded"
    
    return {
        "overall_status": overall,
        "tests": results,
        "failures": failures,
        "warnings": warnings
    }
