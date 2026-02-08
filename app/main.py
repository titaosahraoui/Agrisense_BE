
from fastapi import FastAPI

from app.routes import zones
from . import models
from .database import engine
from .routes import auth, users, weather, satellite, data_collection, predictions,training, zones,dashboard,analysis,health,yieldPrediction
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Water Stress Prediction API",
    description="Enhanced water stress prediction system for Algeria with hybrid ML",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routeurs
app.include_router(health.router) 
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(weather.router)
app.include_router(satellite.router)
app.include_router(data_collection.router)
app.include_router(dashboard.router)
app.include_router(zones.router)
app.include_router(predictions.router)
app.include_router(training.router)
app.include_router(analysis.router)
app.include_router(yieldPrediction.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Water Stress Prediction API",
        "version": "2.0.0",
        "features": [
            "Hybrid Unsupervised Learning",
            "Supervised Machine Learning",
            "Rule-Based Predictions",
            "Multi-Model Comparison",
            "Region-Specific Calibration"
        ],
        "docs": "/docs",
        "health": "/health/test/all"
    }

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("🚀 Water Stress Prediction API Starting...")
    print("=" * 60)
    
    # Initialize prediction service (loads models)
    try:
        from .services.prediction_service import PredictionService
        service = PredictionService()
        info = service.get_model_info()
        
        print(f"📊 Prediction Mode: {info['current_mode']}")
        print(f"   Supervised: {'✅' if info['supervised_available'] else '❌'}")
        print(f"   Hybrid: {'✅' if info['hybrid_available'] else '❌'}")
        print(f"   Rule-Based: ✅")
    except Exception as e:
        print(f"⚠️  Model loading error: {e}")
    
    # Check database connection
    try:
        from .database import SessionLocal
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.close()
        print("✅ Database: Connected")
    except Exception as e:
        print(f"❌ Database: {e}")
    
    # Check Earth Engine
    try:
        import ee
        ee.Initialize()
        print("✅ Earth Engine: Initialized")
    except Exception as e:
        print(f"⚠️  Earth Engine: {str(e)[:50]}")
    
    print("=" * 60)
    print("✅ API Ready!")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health/test/all")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("\n👋 Shutting down Water Stress Prediction API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)