# routes/predictions.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
import os
import json
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["Predictions"])

# Dependency to get prediction service
def get_prediction_service():
    return PredictionService()

@router.post("/predict/stress-score")
async def predict_stress_score(
    features: Dict,
    service: PredictionService = Depends(get_prediction_service)
):
    """Predict stress score from features"""
    try:
        result = service.predict_stress(features)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/performance")
async def get_model_performance(
    service: PredictionService = Depends(get_prediction_service)
):
    """Get performance of the latest model"""
    try:
        # Try getting from loaded model first
        perf = service.get_model_performance()
        if perf:
            return perf
            
        # Fallback to reading disk if service doesn't have it (e.g. model failed to load)
        model_dir = "models"
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail="No models directory found")
            
        results_files = [f for f in os.listdir(model_dir) if f.startswith("training_results_")]
        
        if not results_files:
            raise HTTPException(status_code=404, detail="No results found")
        
        results_files.sort(reverse=True)
        latest_results = os.path.join(model_dir, results_files[0])
        
        with open(latest_results, 'r') as f:
            results = json.load(f)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/list")
async def list_trained_models():
    """List all trained models available"""
    try:
        model_dir = "models"
        if not os.path.exists(model_dir):
            return {"models": [], "message": "No models directory found"}
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        json_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
        
        return {
            "models": model_files,
            "results": json_files,
            "count": {
                "models": len(model_files),
                "results": len(json_files)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
