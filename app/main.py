
from fastapi import FastAPI

from app.routes import zones
from . import models
from .database import engine
from .routes import auth, users, weather, satellite, data_collection, predictions,training, zones,dashboard,analysis
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Algeria Water Stress Detection API 🌍",
    description="API pour la détection du stress hydrique en Algérie",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routeurs
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

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API de Détection du Stress Hydrique en Algérie",
        "endpoints": {
            "auth": "/auth",
            "users": "/users",
            "weather": "/weather",
            "satellite": "/satellite",
            "analysis": "/analysis",
            "data-collection": "/data-collection",
            "predictions": "/predictions",
            "zones": "/zones",
            "dashboard": "/dashboard"
        }
    }



# from fastapi import FastAPI
# from . import models
# from .database import engine
# from .routes import auth, users
# from fastapi.middleware.cors import CORSMiddleware


# models.Base.metadata.create_all(bind=engine)

# app = FastAPI(title="FastAPI Auth Example 🚀")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080"],  # React frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(auth.router)
# app.include_router(users.router)

# @app.get("/")
# def root():
#     return {"message": "Welcome to the FastAPI Backend!"}
