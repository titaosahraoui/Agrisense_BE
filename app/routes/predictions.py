# routes/predictions.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Optional, Any
import os
import json
from sqlalchemy.orm import Session
from ..database import get_db
from .. import models
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["Predictions"])

# Dependency to get prediction service (singleton)
def get_prediction_service():
    return PredictionService()

@router.post("/predict/stress-score")
async def predict_stress_score(
    features: Dict[str, Any],
    mode: Optional[str] = None,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict stress score from features
    
    Args:
        features: Environmental indicators (ndvi, temperature, precipitation, etc.)
        mode: Optional prediction mode ('supervised', 'hybrid', 'rule-based')
    
    Returns:
        Prediction result with score, level, confidence, etc.
    """
    try:
        result = service.predict_stress(features, mode=mode)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/compare")
async def compare_predictions(
    features: Dict[str, Any],
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Compare predictions from all available models
    
    Args:
        features: Environmental indicators
    
    Returns:
        Predictions from supervised, hybrid, and rule-based models with consensus
    """
    try:
        comparison = service.compare_predictions(features)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/recommendations")
async def get_recommendations(
    ndvi: float,
    temperature: float,
    precipitation: float,
    evapotranspiration: float,
    humidity: float,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get stress prediction and actionable recommendations
    
    Query parameters are the environmental indicators
    """
    try:
        features = {
            'ndvi': ndvi,
            'temperature': temperature,
            'precipitation': precipitation,
            'evapotranspiration': evapotranspiration,
            'humidity': humidity
        }
        
        recommendations = service.get_recommendations(features)
        prediction = service.predict_stress(features)
        
        return {
            "prediction": prediction,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
async def get_model_info(
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Get information about available models and current prediction mode
    
    Returns:
        - current_mode: Active prediction mode
        - supervised_available: Whether supervised model is loaded
        - hybrid_available: Whether hybrid model is loaded
        - Model details and performance metrics
    """
    try:
        info = service.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/set-mode")
async def set_prediction_mode(
    mode: str,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Manually set the prediction mode
    
    Args:
        mode: 'supervised', 'hybrid', or 'rule-based'
    """
    try:
        service.set_prediction_mode(mode)
        return {
            "message": f"Prediction mode set to {mode}",
            "current_mode": mode
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/reload")
async def reload_models(
    service: PredictionService = Depends(get_prediction_service)
):
    """Force reload all models from disk"""
    try:
        service.reload_models()
        info = service.get_model_info()
        
        return {
            "message": "Models reloaded successfully",
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/performance")
async def get_model_performance(
    service: PredictionService = Depends(get_prediction_service)
):
    """Get performance metrics of loaded models"""
    try:
        # Try getting from loaded model first
        perf = service.get_model_performance()
        if perf:
            return perf
            
        # Fallback to reading from disk
        possible_dirs = ["ml_Models", "models"]
        model_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                model_dir = d
                break
        
        if not model_dir:
            raise HTTPException(status_code=404, detail="No models directory found")
            
        results_files = [
            f for f in os.listdir(model_dir) 
            if f.startswith("training_metrics_") or 
               f.startswith("training_results_") or
               f.startswith("training_comparison_")
        ]
        
        if not results_files:
            raise HTTPException(status_code=404, detail="No training results found")
        
        results_files.sort(reverse=True)
        latest_results = os.path.join(model_dir, results_files[0])
        
        with open(latest_results, 'r') as f:
            results = json.load(f)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/list")
async def list_trained_models():
    """List all trained models available"""
    try:
        possible_dirs = ["ml_Models", "models"]
        model_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                model_dir = d
                break
                
        if not model_dir:
            return {
                "models": [], 
                "message": "No models directory found",
                "directory": None
            }
        
        # Get model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        json_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
        
        # Categorize models
        supervised_models = [f for f in model_files if 'regression' in f or 'classification' in f]
        hybrid_models = [f for f in model_files if 'hybrid' in f]
        other_models = [f for f in model_files if f not in supervised_models + hybrid_models]
        
        return {
            "directory": model_dir,
            "supervised_models": supervised_models,
            "hybrid_models": hybrid_models,
            "other_models": other_models,
            "result_files": json_files,
            "count": {
                "total_models": len(model_files),
                "supervised": len(supervised_models),
                "hybrid": len(hybrid_models),
                "results": len(json_files)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/predict")
async def batch_predict(
    features_list: list[Dict[str, Any]],
    mode: Optional[str] = None,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict stress scores for multiple samples
    
    Args:
        features_list: List of feature dictionaries
        mode: Optional prediction mode
    
    Returns:
        List of predictions
    """
    try:
        predictions = service.predict_batch(features_list, mode=mode)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "mode_used": predictions[0].get('prediction_mode') if predictions else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Backward compatibility endpoint
@router.post("/predict/recalculate-all")
async def recalculate_all_stress_scores(
    background_tasks: BackgroundTasks,
    mode: Optional[str] = None,
    service: PredictionService = Depends(get_prediction_service),
    db: Session = Depends(get_db)
):
    """
    Recalculate stress scores for ALL training data records using the loaded model.
    
    Args:
        mode: Optional prediction mode to use ('supervised', 'hybrid', 'rule-based')
    """
    try:
        # Check if we have any model available
        info = service.get_model_info()
        
        if not info['supervised_available'] and not info['hybrid_available']:
            raise HTTPException(
                status_code=400, 
                detail="No models available. Please train a model first."
            )

        # Define the background task function
        def _process_recalculation(db_session: Session, pred_service: PredictionService, pred_mode: Optional[str]):
            try:
                print("🔄 Starting recalculation of all stress scores...")
                
                # Load all training data
                all_records = db_session.query(models.TrainingData).all()
                
                if not all_records:
                    print("⚠️  No training data found")
                    return
                
                count = 0
                errors = 0
                
                for record in all_records:
                    try:
                        # Prepare features
                        features = {
                            'ndvi': record.ndvi,
                            'temperature': record.temperature_avg,
                            'precipitation': record.precipitation,
                            'evapotranspiration': record.evapotranspiration,
                            'humidity': record.humidity
                        }
                        
                        # Remove None values
                        features = {k: v for k, v in features.items() if v is not None}
                        
                        if not features:
                            continue
                        
                        # Predict
                        result = pred_service.predict_stress(features, mode=pred_mode)
                        
                        # Update record
                        record.stress_score = result['predicted_stress_score']
                        record.stress_level = result['stress_level']
                        
                        count += 1
                        
                        # Commit in batches of 100
                        if count % 100 == 0:
                            db_session.commit()
                            print(f"   Processed {count} records...")
                    
                    except Exception as e:
                        errors += 1
                        print(f"   Error processing record {record.id}: {e}")
                        continue
                
                # Final commit
                db_session.commit()
                
                print(f"✅ Recalculation complete!")
                print(f"   Successful: {count}")
                print(f"   Errors: {errors}")
                print(f"   Mode used: {pred_mode or 'auto'}")
            
            except Exception as e:
                print(f"❌ Fatal error in recalculation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                db_session.close()

        # Add to background tasks
        background_tasks.add_task(_process_recalculation, db, service, mode)
        
        return {
            "message": "Recalculation started in background",
            "mode": mode or info['current_mode'],
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
