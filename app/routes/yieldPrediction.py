# yieldPrediction.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd

# Import your service
from ..services.YieldPredictionService import YieldPredictionService

# Pydantic models for request/response validation
class MeteorologicalFeatures(BaseModel):
    """Meteorological features for prediction"""
    avg_temperature_hiver: float = Field(..., description="Average winter temperature (°C)")
    avg_temperature_ete: float = Field(..., description="Average summer temperature (°C)")
    avg_rainfall_hiver: float = Field(..., description="Average winter rainfall (mm)")
    avg_rainfall_ete: float = Field(..., description="Average summer rainfall (mm)")
    avg_humidity_hiver: float = Field(..., description="Average winter humidity (%)")
    avg_humidity_ete: float = Field(..., description="Average summer humidity (%)")
    avg_ndvi_hiver: float = Field(..., description="Average winter NDVI")
    avg_ndvi_ete: float = Field(..., description="Average summer NDVI")


class PredictionRequest(BaseModel):
    """Request model for single wilaya prediction"""
    wilaya_code: int = Field(..., description="Wilaya code (1-58)")
    season: str = Field(..., description="Season: 'summer' or 'winter'")
    meteorological_features: MeteorologicalFeatures
    
    class Config:
        json_schema_extra = {
            "example": {
                "wilaya_code": 16,
                "season": "summer",
                "meteorological_features": {
                    "avg_temperature_hiver": 12.5,
                    "avg_temperature_ete": 28.3,
                    "avg_rainfall_hiver": 45.2,
                    "avg_rainfall_ete": 12.8,
                    "avg_humidity_hiver": 68.5,
                    "avg_humidity_ete": 42.1,
                    "avg_ndvi_hiver": 0.35,
                    "avg_ndvi_ete": 0.22
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    wilaya_code: int
    wilaya_name: Optional[str] = None
    season: str
    predicted_yield: float
    unit: str = "qx/ha"
    timestamp: datetime = Field(default_factory=datetime.now)


class DashboardData(BaseModel):
    """Response model for dashboard data"""
    predictions: List[Dict]
    total_wilayas: int
    average_summer_yield: float
    average_winter_yield: float
    recommendations: Dict
    timestamp: datetime = Field(default_factory=datetime.now)


class WilayaDetailResponse(BaseModel):
    """Detailed response for a specific wilaya"""
    wilaya_code: int
    wilaya_name: str
    predictions: Dict
    historical_data: List[Dict]
    cluster: Optional[int] = None
    recommendations: List[str]


class TrainingRequest(BaseModel):
    """Request to trigger model training"""
    retrain: bool = Field(default=False, description="Force retrain even if models exist")


class TrainingResponse(BaseModel):
    """Response from training endpoint"""
    status: str
    message: str
    results: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Create router
router = APIRouter(prefix="/yield-prediction", tags=["Yield Predictions"])

# Database configuration - adjust to your settings
# Database configuration - adjust to your settings
from ..database import DATABASE_URL
DB_URL = DATABASE_URL

# Path to agricultural data
from pathlib import Path
CSV_BASE_PATH = Path(__file__).resolve().parent.parent.parent / "ml_Models" / "outputs" / "data"

# Initialize service (singleton pattern)
_service_instance = None

def get_service() -> YieldPredictionService:
    """Dependency to get service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = YieldPredictionService(
            db_url=DB_URL,
            csv_base_path=CSV_BASE_PATH,
            models_dir="models"
        )
        
        # Try to load existing models
        loaded = _service_instance.load_models()
        if not loaded:
            print("⚠️  No pre-trained models found. Please train models first using /train endpoint")
        
        # Load data if models loaded but data not loaded
        if _service_instance.master_df is None and loaded:
            print("Loading data...")
            _service_instance.load_all_data()
    
    return _service_instance


@router.post("/train", response_model=TrainingResponse, status_code=202)
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    service: YieldPredictionService = Depends(get_service)
):
    """
    Train or retrain yield prediction models
    
    This endpoint loads data from the database and trains models for:
    - Summer cereals
    - Winter cereals
    - Maize
    - Sorghum
    
    Note: This is a long-running operation
    """
    try:
        # Check if models already exist
        if not request.retrain and service.models:
            return TrainingResponse(
                status="skipped",
                message="Models already trained. Set retrain=true to force retraining"
            )
        
        # Load all data
        print("Loading data...")
        service.load_all_data()
        
        # Train models (in background for production)
        def train_in_background():
            try:
                results = service.train_all_models()
                print("Training completed successfully")
                return results
            except Exception as e:
                print(f"Training failed: {str(e)}")
                raise
        
        if background_tasks:
            background_tasks.add_task(train_in_background)
            return TrainingResponse(
                status="started",
                message="Training started in background. Check logs for progress."
            )
        else:
            results = train_in_background()
            return TrainingResponse(
                status="success",
                message="Models trained successfully",
                results={
                    'n_models_trained': len(service.models),
                    'model_metrics': service.model_metrics,
                    'clusters_created': service.cluster_results is not None
                }
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict_yield(
    request: PredictionRequest,
    service: YieldPredictionService = Depends(get_service)
):
    """
    Predict cereal yield for a specific wilaya
    
    Provides yield prediction based on meteorological features and wilaya characteristics.
    """
    try:
        # Validate season
        if request.season not in ['summer', 'winter']:
            raise HTTPException(status_code=400, detail="Season must be 'summer' or 'winter'")
        
        # Check if model exists
        if request.season not in service.models:
            raise HTTPException(
                status_code=400, 
                detail=f"No model trained for {request.season} season. Please train models first."
            )
        
        # Convert Pydantic model to dict
        meteo_features = request.meteorological_features.dict()
        
        # Make prediction
        predicted_yield = service.predict_yield(
            wilaya_code=request.wilaya_code,
            season=request.season,
            meteorological_features=meteo_features
        )
        
        # Get wilaya name
        wilaya_name = None
        if service.master_df is not None:
            wilaya_data = service.master_df[service.master_df['Wilaya_Code'] == request.wilaya_code]
            if not wilaya_data.empty:
                wilaya_name = wilaya_data.iloc[0]['Wilaya_Name']
        
        return PredictionResponse(
            wilaya_code=request.wilaya_code,
            wilaya_name=wilaya_name,
            season=request.season,
            predicted_yield=round(predicted_yield, 2)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data(
    year: int = 2024,
    service: YieldPredictionService = Depends(get_service)
):
    """
    Get comprehensive dashboard data for all wilayas
    
    Returns predictions, averages, and recommendations for the dashboard view.
    """
    try:
        if service.master_df is None or not service.models:
            raise HTTPException(
                status_code=503, 
                detail="Models not trained. Please train models first using /train endpoint"
            )
        
        # Get all predictions
        predictions = service.get_all_predictions(year=year)
        
        # Calculate averages
        if predictions:
            avg_summer = sum(p['summer_yield'] for p in predictions) / len(predictions)
            avg_winter = sum(p['winter_yield'] for p in predictions) / len(predictions)
        else:
            avg_summer = avg_winter = 0
        
        # Get recommendations
        recommendations = service.generate_recommendations()
        
        return DashboardData(
            predictions=predictions,
            total_wilayas=len(predictions),
            average_summer_yield=round(avg_summer, 2),
            average_winter_yield=round(avg_winter, 2),
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {str(e)}")


@router.get("/wilaya/{wilaya_code}", response_model=WilayaDetailResponse)
async def get_wilaya_details(
    wilaya_code: int,
    service: YieldPredictionService = Depends(get_service)
):
    """
    Get detailed information and predictions for a specific wilaya
    
    Returns historical data, predictions, cluster assignment, and recommendations.
    """
    try:
        if service.master_df is None:
            raise HTTPException(
                status_code=503,
                detail="Models not trained. Please train models first"
            )
        
        # Filter data for this wilaya
        wilaya_data = service.master_df[service.master_df['Wilaya_Code'] == wilaya_code]
        
        if wilaya_data.empty:
            raise HTTPException(status_code=404, detail=f"Wilaya {wilaya_code} not found")
        
        wilaya_name = wilaya_data.iloc[0]['Wilaya_Name']
        
        # Get historical data
        historical = []
        for _, row in wilaya_data.iterrows():
            historical.append({
                'year': int(row['Year']),
                'summer_yield': float(row['Total_Summer_Yield']),
                'winter_yield': float(row['Total_Winter_Yield']),
                'cultivated_area': float(row['Cultivated_Area'])
            })
        
        # Get latest predictions
        latest_row = wilaya_data.iloc[-1]
        predictions = {
            'summer_yield': float(latest_row['Total_Summer_Yield']),
            'winter_yield': float(latest_row['Total_Winter_Yield']),
            'maize_yield': float(latest_row.get('Summer_Maize_Yield', 0)),
            'sorghum_yield': float(latest_row.get('Summer_Sorghum_Yield', 0))
        }
        
        # Get cluster assignment
        cluster = None
        if service.cluster_results is not None:
            cluster_data = service.cluster_results['cluster_data']
            wilaya_cluster = cluster_data[cluster_data['Wilaya_Code'] == wilaya_code]
            if not wilaya_cluster.empty:
                cluster = int(wilaya_cluster.iloc[0]['Cluster'])
        
        # Generate wilaya-specific recommendations
        recommendations = []
        if predictions['summer_yield'] > predictions['winter_yield']:
            recommendations.append("Focus on summer cereal cultivation for higher yields")
        else:
            recommendations.append("Winter cereals show better performance in this region")
        
        if latest_row['Total_Agricultural_Area'] > latest_row['Cultivated_Area'] * 2:
            recommendations.append("Significant unused agricultural area - consider expanding cultivation")
        
        return WilayaDetailResponse(
            wilaya_code=wilaya_code,
            wilaya_name=wilaya_name,
            predictions=predictions,
            historical_data=historical,
            cluster=cluster,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get wilaya details: {str(e)}")


@router.get("/recommendations")
async def get_recommendations(service: YieldPredictionService = Depends(get_service)):
    """
    Get strategic recommendations for cereal production
    
    Returns insights about high-potential regions, seasonal performance, and land use optimization.
    """
    try:
        if service.master_df is None:
            raise HTTPException(
                status_code=503,
                detail="Models not trained. Please train models first"
            )
        
        recommendations = service.generate_recommendations()
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.get("/clusters")
async def get_cluster_analysis(service: YieldPredictionService = Depends(get_service)):
    """
    Get cluster analysis of wilayas based on agricultural characteristics
    
    Returns cluster assignments and summary statistics.
    """
    try:
        if service.cluster_results is None:
            raise HTTPException(
                status_code=503,
                detail="Clustering not performed. Please train models first"
            )
        
        cluster_data = service.cluster_results['cluster_data']
        cluster_summary = service.cluster_results['cluster_summary']
        
        # Format cluster data
        clusters = {}
        for cluster_id in cluster_data['Cluster'].unique():
            cluster_wilayas = cluster_data[cluster_data['Cluster'] == cluster_id]
            clusters[int(cluster_id)] = {
                'wilayas': cluster_wilayas[['Wilaya_Code', 'Wilaya_Name']].to_dict('records'),
                'count': len(cluster_wilayas),
                'characteristics': cluster_summary.loc[cluster_id].to_dict()
            }
        
        return {
            "status": "success",
            "clusters": clusters,
            "total_clusters": len(clusters),
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cluster analysis: {str(e)}")


@router.get("/model-performance")
async def get_model_performance(service: YieldPredictionService = Depends(get_service)):
    """
    Get detailed model performance metrics
    """
    try:
        if not service.model_metrics:
            raise HTTPException(
                status_code=503,
                detail="Models not trained. Please train models first"
            )
        
        # Sanitize metrics for JSON response
        sanitized_metrics = {}
        for crop_type, metrics in service.model_metrics.items():
            sanitized_metrics[crop_type] = {}
            for k, v in metrics.items():
                # Skip large arrays like predictions
                if k == 'predictions':
                    continue
                # proper conversion for feature_importances dict
                if k == 'feature_importances' and isinstance(v, dict):
                     sanitized_metrics[crop_type][k] = {fk: float(fv) for fk, fv in v.items()}
                # Convert numpy types to native python types
                elif hasattr(v, 'item'): 
                    sanitized_metrics[crop_type][k] = v.item()
                else:
                    sanitized_metrics[crop_type][k] = v
        
        return {
            "status": "success",
            "model_metrics": sanitized_metrics,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.get("/health")
async def health_check(service: YieldPredictionService = Depends(get_service)):
    """
    Health check endpoint
    
    Returns the status of the prediction service and loaded models.
    """
    return {
        "status": "healthy",
        "models_loaded": {
            "summer": 'summer' in service.models,
            "winter": 'winter' in service.models,
            "maize": 'maize' in service.models,
            "sorghum": 'sorghum' in service.models
        },
        "models_count": len(service.models),
        "data_loaded": service.master_df is not None,
        "data_shape": service.master_df.shape if service.master_df is not None else None,
        "timestamp": datetime.now()
    }


@router.get("/wilayas")
async def get_all_wilayas(service: YieldPredictionService = Depends(get_service)):
    """
    Get list of all wilayas with basic information
    
    Returns a list of all wilayas in the system.
    """
    try:
        if service.master_df is None:
            raise HTTPException(
                status_code=503,
                detail="Data not loaded. Please train models first"
            )
        
        wilayas = service.master_df.groupby('Wilaya_Code').first()[['Wilaya_Name']].reset_index()
        wilaya_list = [
            {'code': int(row['Wilaya_Code']), 'name': row['Wilaya_Name']}
            for _, row in wilayas.iterrows()
        ]
        
        return {
            "status": "success",
            "wilayas": sorted(wilaya_list, key=lambda x: x['code']),
            "total": len(wilaya_list),
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get wilayas list: {str(e)}")


@router.get("/features")
async def get_feature_importance(service: YieldPredictionService = Depends(get_service)):
    """
    Get feature importance for trained models
    """
    try:
        if not service.model_metrics:
            raise HTTPException(
                status_code=503,
                detail="Models not trained. Please train models first"
            )
        
        feature_importance = {}
        for crop_type, metrics in service.model_metrics.items():
            if 'feature_importances' in metrics and metrics['feature_importances'] is not None:
                feature_importance[crop_type] = metrics['feature_importances']
        
        return {
            "status": "success",
            "feature_importance": feature_importance,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")