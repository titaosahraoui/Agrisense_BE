from typing import Dict, Any, Optional
from .ml_model import WaterStressModel
from .stress_analysis import StressAnalysisService
import pandas as pd
import os
from datetime import datetime

class PredictionService:
    _instance = None
    _model = None
    _stress_service = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize models and services"""
        self._stress_service = StressAnalysisService()
        self._load_best_model()

    def _load_best_model(self):
        """Load the best available model from disk"""
        try:
            model_dir = "models"
            if not os.path.exists(model_dir):
                print("Warning: No models directory found.")
                return

            # Find latest model
            model_files = [f for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pkl")]
            if not model_files:
                print("Warning: No trained models found.")
                return

            model_files.sort(reverse=True)
            latest_model_path = os.path.join(model_dir, model_files[0])
            
            # Use WaterStressModel to load
            self._model = WaterStressModel()
            # Note: WaterStressModel.load_model expects just the dictionary, but our current file might be just the dict
            # Let's adjust based on how ml_model.py works.
            # ml_model.py's load_model uses joblib.load which returns the dict.
            # It then sets self.model, self.scaler etc from that dict.
            self._model.load_model(latest_model_path)
            print(f"PredictionService: Loaded model from {latest_model_path}")
            
        except Exception as e:
            print(f"Error loading model in PredictionService: {e}")
            self._model = None

    def predict_stress(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict stress score and return detailed result
        """
        if self._model is None:
            # Try reloading in case it was added later
            self._load_best_model()
            if self._model is None:
                raise ValueError("Model not available. Please train a model first.")

        # Prepare features
        # The model's predict method expects a DataFrame
        # We need to ensure we have the right columns
        
        expected_features = self._model.feature_names
        
        # Create DataFrame with 0 for missing features
        features_df = pd.DataFrame([features])
        if expected_features:
            for col in expected_features:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Reorder columns to match training
            features_df = features_df[expected_features]

        # Predict
        prediction_val = self._model.predict(features_df)[0]
        
        # Determine level using the centralized logic
        stress_level = self._stress_service.determine_stress_level(prediction_val)
        
        return {
            "predicted_stress_score": float(prediction_val),
            "stress_level": stress_level,
            "model_used": self._model.model_info.get('model_name', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }

    def reload_model(self):
        """Force reload of the best model"""
        self._load_best_model()

    def get_model_performance(self):
        """Get performance metrics of loaded model"""
        if self._model and self._model.model_info:
            return self._model.model_info
        return None
