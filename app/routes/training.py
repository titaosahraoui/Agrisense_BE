from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
import json
from ..services.data_cleaner import DataCleaner
from ..database import get_db
import pandas as pd
import numpy as np
import os
from ..services.data_cleaner import DataCleaner
from ..services.optimized_trainer import OptimizedModelTrainer

router = APIRouter(prefix="/training", tags=["Model Training"])


@router.get("/data-quality")
async def get_data_quality(
    wilaya_code: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Analyze data quality and missing values"""
    cleaner = DataCleaner(db)
    
    wilaya_codes = [wilaya_code] if wilaya_code else None
    df = cleaner.load_training_data(wilaya_codes)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    analysis = cleaner.analyze_missing_data(df)
    
    # Convert any remaining numpy types in the response
    def clean_response(obj):
        """Recursively clean response for JSON serialization"""
        if obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, dict):
            return {key: clean_response(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [clean_response(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # Handle datetime
            return obj.isoformat()
        else:
            # Try to convert to native Python type
            try:
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                else:
                    return str(obj)
            except:
                return str(obj)
    
    # Clean the entire response
    cleaned_analysis = clean_response(analysis)
    
    # Get date range safely
    date_start = None
    date_end = None
    if 'date' in df.columns and not df['date'].empty:
        try:
            date_start = df['date'].min()
            date_end = df['date'].max()
            # Convert to string if they're datetime objects
            if hasattr(date_start, 'isoformat'):
                date_start = date_start.isoformat()
            if hasattr(date_end, 'isoformat'):
                date_end = date_end.isoformat()
        except:
            date_start = None
            date_end = None
    
    response = {
        "data_summary": {
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "date_range": {
                "start": date_start,
                "end": date_end
            }
        },
        "missing_data_analysis": cleaned_analysis
    }
    
    return response

@router.post("/clean-data")
async def clean_data(
    wilaya_codes: Optional[List[int]] = None,
    strategy: str = "advanced",
    db: Session = Depends(get_db)
):
    """Clean data and return analysis"""
    cleaner = DataCleaner(db)
    
    # Load data
    df = cleaner.load_training_data(wilaya_codes)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Analyze before cleaning
    before_analysis = cleaner.analyze_missing_data(df)
    
    # Clean data
    df_clean = cleaner.clean_data(df, strategy=strategy)
    
    # Analyze after cleaning
    after_analysis = cleaner.analyze_missing_data(df_clean)
    
    # Add engineered features
    df_engineered = cleaner.add_engineered_features(df_clean)
    
    # Save cleaned data
    filename = cleaner.save_cleaned_data(df_engineered)
    
    return {
        "before_cleaning": before_analysis,
        "after_cleaning": after_analysis,
        "cleaning_strategy": strategy,
        "saved_file": filename,
        "engineered_features_count": len(df_engineered.columns) - len(df.columns),
        "sample_size": len(df_engineered)
    }


@router.post("/train-optimized")
async def train_optimized_models(
    wilaya_codes: Optional[List[int]] = None,
    cleaning_strategy: str = "advanced",
    test_size: float = 0.2,
    random_state: int = 42,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Train and compare multiple models with optimized parameters"""
    
    if background_tasks:
        background_tasks.add_task(
            train_models_async,
            wilaya_codes, cleaning_strategy, test_size, random_state, db
        )
        
        return {
            "message": "Optimized model training started in background",
            "wilaya_codes": wilaya_codes,
            "cleaning_strategy": cleaning_strategy,
            "test_size": test_size,
            "status": "started"
        }
    else:
        return await train_models_async(wilaya_codes, cleaning_strategy, test_size, random_state, db)

async def train_models_async(wilaya_codes, cleaning_strategy, test_size, random_state, db):
    """Async wrapper for model training"""
    try:

        cleaner = DataCleaner(db)
        
        print("📊 Loading data...")
        df = cleaner.load_training_data(wilaya_codes)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        print(f"✅ Loaded {len(df)} records")
        
        print("🧹 Cleaning data...")
        df_clean = cleaner.clean_data(df, strategy=cleaning_strategy)
        
        print("⚙️ Engineering features...")
        df_engineered = cleaner.add_engineered_features(df_clean)
        
        print("📈 Preparing training data...")
        X_train, X_test, y_train, y_test, feature_cols = cleaner.prepare_for_training(df_engineered)
        
        print("🤖 Training and comparing models...")
        trainer = OptimizedModelTrainer()
        results = trainer.train_models(X_train, y_train, X_test, y_test)
        
        print("🔄 Performing time series cross-validation...")
        cv_results = trainer.time_series_cross_validate(
            pd.concat([X_train, X_test]), 
            pd.concat([y_train, y_test])
        )
        
        print("🔍 Analyzing feature importance...")
        feature_analysis = trainer.analyze_feature_importance()
        
        print("💾 Saving models...")
        saved_files = trainer.save_models()
        
        # Prepare response
        response = {
            "status": "success",
            "data_info": {
                "total_samples": len(df),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "features_count": len(feature_cols),
                "cleaning_strategy": cleaning_strategy
            },
            "model_results": results,
            "best_model": trainer.best_model_name,
            "cross_validation": cv_results,
            "feature_importance": feature_analysis,
            "saved_files": [os.path.basename(f) for f in saved_files],
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error in model training: {e}")
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "details": "Check server logs for full traceback"
            }
        )

# Also add these helper endpoints to your existing training.py
@router.get("/list-trained-models")
async def list_trained_models():
    """List all trained models available"""
    import os
    import glob
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        return {"models": [], "message": "No models directory found"}
    
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    json_files = glob.glob(os.path.join(model_dir, "*.json"))
    
    return {
        "models_directory": model_dir,
        "model_files": [os.path.basename(f) for f in model_files],
        "results_files": [os.path.basename(f) for f in json_files],
        "count": {
            "models": len(model_files),
            "results": len(json_files)
        }
    }