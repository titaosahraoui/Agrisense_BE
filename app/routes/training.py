# routes/training.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
import json
import os
from ..database import get_db
from ..services.feature_engineering import FeatureEngineer
from ..services.unified_trainer import UnifiedModelTrainer
from ..services.stress_analysis import RegionType
from .. import models



router = APIRouter(prefix="/training", tags=["Model Training"])

@router.get("/data-quality")
async def get_data_quality(
    wilaya_codes: Optional[List[int]] = None,
    db: Session = Depends(get_db)
):
    """Analyze data quality and missing values"""
    try:
        engineer = FeatureEngineer()
        
        # Load data
        data = engineer.load_training_data(db, wilaya_codes)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Analyze missing data
        missing_analysis = {}
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                missing_analysis[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_count / len(data) * 100)
                }
        
        # Get date range
        date_start = None
        date_end = None
        if 'date' in data.columns and not data['date'].isnull().all():
            date_start = data['date'].min()
            date_end = data['date'].max()
            if hasattr(date_start, 'isoformat'):
                date_start = date_start.isoformat()
            if hasattr(date_end, 'isoformat'):
                date_end = date_end.isoformat()
        
        return {
            "data_summary": {
                "total_records": int(len(data)),
                "total_columns": int(len(data.columns)),
                "date_range": {
                    "start": date_start,
                    "end": date_end
                },
                "wilaya_codes": sorted(data['wilaya_code'].unique().tolist()) if 'wilaya_code' in data.columns else []
            },
            "missing_data_analysis": missing_analysis,
            "column_dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/hybrid")
async def train_hybrid_model(
    wilaya_codes: Optional[List[int]] = None,
    region_type: str = "mediterranean",
    n_clusters: int = 5,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Train hybrid clustering model (NO LABELS NEEDED)
    
    Args:
        wilaya_codes: Optional list of wilaya codes to include
        region_type: 'mediterranean', 'saharan', 'high_plateaus', or 'tell_atlas'
        n_clusters: Number of stress zones to discover (3-8 recommended)
        background_tasks: Optional background execution
    
    Returns:
        Training results including cluster summary and metrics
    """
    
    # Convert region type string to enum
    region_map = {
        'mediterranean': RegionType.MEDITERRANEAN,
        'saharan': RegionType.SAHARAN,
        'high_plateaus': RegionType.HIGH_PLATEAUS,
        'tell_atlas': RegionType.TELL_ATLAS
    }
    
    if region_type.lower() not in region_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region_type. Must be one of: {list(region_map.keys())}"
        )
    
    region_enum = region_map[region_type.lower()]
    
    def _train_task(db_session: Session, codes: Optional[List[int]], 
                   reg_type: RegionType, clusters: int):
        try:
            print(f"🔬 Starting hybrid model training...")
            print(f"   Region: {reg_type.value}")
            print(f"   Clusters: {clusters}")
            
            # Load data
            engineer = FeatureEngineer()
            data = engineer.load_training_data(db_session, codes)
            
            if data.empty:
                print("❌ No training data found")
                return {"error": "No data found"}
            
            print(f"   Loaded {len(data)} records")
            
            # Train hybrid model
            trainer = UnifiedModelTrainer()
            result = trainer.train_hybrid(
                data=data,
                region_type=reg_type,
                n_clusters=clusters
            )
            
            print("✅ Hybrid model training complete!")
            return result
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            db_session.close()
    
    if background_tasks:
        background_tasks.add_task(_train_task, db, wilaya_codes, region_enum, n_clusters)
        
        return {
            "message": "Hybrid model training started in background",
            "region_type": region_type,
            "n_clusters": n_clusters,
            "wilaya_codes": wilaya_codes,
            "status": "started"
        }
    else:
        result = _train_task(db, wilaya_codes, region_enum, n_clusters)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result

@router.post("/train/supervised")
async def train_supervised_model(
    wilaya_codes: Optional[List[int]] = None,
    model_name: str = "xgb",
    test_size: float = 0.2,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Train supervised ML model (NEEDS LABELED STRESS SCORES)
    
    Args:
        wilaya_codes: Optional list of wilaya codes
        model_name: 'xgb', 'rf', 'gbr', or 'best' (to auto-select best model)
        test_size: Test set proportion (0.1 - 0.3)
        background_tasks: Optional background execution
    
    Returns:
        Training results including performance metrics
    """
    
    # Validation relaxed to allow 'best' and other models supported by underlying trainer
    if not (0.1 <= test_size <= 0.3):
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0.1 and 0.3"
        )
    
    def _train_task(db_session: Session, codes: Optional[List[int]], 
                   model: str, test_sz: float):
        try:
            print(f"🤖 Starting supervised model training...")
            print(f"   Model: {model}")
            print(f"   Test size: {test_sz}")
            
            # Load and prepare data
            engineer = FeatureEngineer()
            data = engineer.load_training_data(db_session, codes)
            
            if data.empty:
                print("❌ No training data found")
                return {"error": "No data found"}
            
            # Check for stress_score labels
            if 'stress_score' not in data.columns or data['stress_score'].isnull().all():
                print("❌ No stress_score labels found")
                return {"error": "No labeled data (stress_score) found. Use hybrid training instead."}
            
            print(f"   Loaded {len(data)} records")
            
            # Create features
            features_df = engineer.create_features_from_training_data(data)
            X, y = engineer.prepare_for_training(features_df, target_type='score')
            
            print(f"   Features: {X.shape[1]}")
            print(f"   Samples: {len(X)}")
            
            # Train model
            trainer = UnifiedModelTrainer()
            result = trainer.train_supervised(
                X=X,
                y=y,
                model_name=model,
                test_size=test_sz
            )
            
            print("✅ Supervised model training complete!")
            return result
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            db_session.close()
    
    if background_tasks:
        background_tasks.add_task(_train_task, db, wilaya_codes, model_name, test_size)
        
        return {
            "message": "Supervised model training started in background",
            "model_name": model_name,
            "test_size": test_size,
            "wilaya_codes": wilaya_codes,
            "status": "started"
        }
    else:
        result = _train_task(db, wilaya_codes, model_name, test_size)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result

@router.post("/train/compare")
async def train_and_compare_models(
    wilaya_codes: Optional[List[int]] = None,
    region_type: str = "mediterranean",
    supervised_model: str = "xgb",
    n_clusters: int = 5,
    test_size: float = 0.2,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Train both supervised and hybrid models, then compare performance
    
    Returns:
        Comparison results and recommendation
    """
    
    # Convert region type
    region_map = {
        'mediterranean': RegionType.MEDITERRANEAN,
        'saharan': RegionType.SAHARAN,
        'high_plateaus': RegionType.HIGH_PLATEAUS,
        'tell_atlas': RegionType.TELL_ATLAS
    }
    
    region_enum = region_map.get(region_type.lower(), RegionType.MEDITERRANEAN)
    
    def _compare_task(db_session: Session, codes: Optional[List[int]], 
                     reg_type: RegionType, sup_model: str, clusters: int, test_sz: float):
        try:
            print("🥊 Starting model comparison training...")
            
            # Load data
            engineer = FeatureEngineer()
            data = engineer.load_training_data(db_session, codes)
            
            if data.empty:
                return {"error": "No data found"}
            
            # Train both models
            trainer = UnifiedModelTrainer()
            results = trainer.train_both(
                data=data,
                region_type=reg_type,
                supervised_model_name=sup_model,
                n_clusters=clusters,
                test_size=test_sz
            )
            
            print("✅ Comparison training complete!")
            return results
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            db_session.close()
    
    if background_tasks:
        background_tasks.add_task(
            _compare_task, db, wilaya_codes, region_enum, 
            supervised_model, n_clusters, test_size
        )
        
        return {
            "message": "Comparison training started in background",
            "status": "started"
        }
    else:
        result = _compare_task(db, wilaya_codes, region_enum, supervised_model, n_clusters, test_size)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result

@router.post("/optimize/clusters")
async def optimize_cluster_count(
    wilaya_codes: Optional[List[int]] = None,
    region_type: str = "mediterranean",
    min_clusters: int = 3,
    max_clusters: int = 8,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Find optimal number of clusters for hybrid model using elbow method
    
    Returns:
        Optimal cluster count and evaluation metrics for each count tested
    """
    
    region_map = {
        'mediterranean': RegionType.MEDITERRANEAN,
        'saharan': RegionType.SAHARAN,
        'high_plateaus': RegionType.HIGH_PLATEAUS,
        'tell_atlas': RegionType.TELL_ATLAS
    }
    
    region_enum = region_map.get(region_type.lower(), RegionType.MEDITERRANEAN)
    
    def _optimize_task(db_session: Session, codes: Optional[List[int]], 
                      reg_type: RegionType, min_c: int, max_c: int):
        try:
            print("🔍 Starting cluster optimization...")
            
            # Load data
            engineer = FeatureEngineer()
            data = engineer.load_training_data(db_session, codes)
            
            if data.empty:
                return {"error": "No data found"}
            
            # Optimize
            trainer = UnifiedModelTrainer()
            result = trainer.optimize_clusters(
                data=data,
                region_type=reg_type,
                min_clusters=min_c,
                max_clusters=max_c
            )
            
            print(f"✅ Optimal clusters: {result['optimal_n_clusters']}")
            return result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            db_session.close()
    
    if background_tasks:
        background_tasks.add_task(
            _optimize_task, db, wilaya_codes, region_enum, min_clusters, max_clusters
        )
        
        return {
            "message": "Cluster optimization started in background",
            "status": "started"
        }
    else:
        result = _optimize_task(db, wilaya_codes, region_enum, min_clusters, max_clusters)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result

@router.get("/list-models")
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
                "message": "No models directory found"
            }
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        result_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
        
        # Categorize models
        supervised = [f for f in model_files if 'regression' in f or 'classification' in f]
        hybrid = [f for f in model_files if 'hybrid' in f]
        
        return {
            "directory": model_dir,
            "supervised_models": supervised,
            "hybrid_models": hybrid,
            "result_files": result_files,
            "count": {
                "total": len(model_files),
                "supervised": len(supervised),
                "hybrid": len(hybrid),
                "results": len(result_files)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/status")
async def get_training_status(db: Session = Depends(get_db)):
    """
    Get information about available training data
    
    Returns:
        Statistics about data availability for training
    """
    try:
        engineer = FeatureEngineer()
        data = engineer.load_training_data(db)
        
        if data.empty:
            return {
                "data_available": False,
                "message": "No training data found in database"
            }
        
        # Check for labels
        has_labels = 'stress_score' in data.columns and not data['stress_score'].isnull().all()
        
        # Get wilaya distribution
        wilaya_dist = {}
        if 'wilaya_code' in data.columns:
            wilaya_dist = data['wilaya_code'].value_counts().to_dict()
        
        return {
            "data_available": True,
            "total_records": len(data),
            "has_stress_labels": has_labels,
            "labeled_records": int(data['stress_score'].notna().sum()) if 'stress_score' in data.columns else 0,
            "date_range": {
                "start": data['date'].min().isoformat() if 'date' in data.columns else None,
                "end": data['date'].max().isoformat() if 'date' in data.columns else None
            },
            "wilaya_distribution": {int(k): int(v) for k, v in wilaya_dist.items()},
            "available_features": data.columns.tolist(),
            "recommended_approach": "hybrid" if not has_labels else "both"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
