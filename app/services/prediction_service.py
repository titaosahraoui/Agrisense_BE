from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import os
from datetime import datetime
import joblib

from .ml_model import WaterStressModel
from .hybrid_stress_model import HybridStressPredictor, create_stress_predictor
from .stress_analysis import StressAnalysisService, RegionType


class PredictionService:
    """
    Enhanced Prediction Service supporting multiple prediction approaches:
    1. Supervised ML (trained on labeled stress scores)
    2. Hybrid unsupervised (clustering + domain knowledge)
    3. Rule-based (pure domain knowledge)
    """
    
    _instance = None
    _supervised_model = None
    _hybrid_model = None
    _stress_service = None
    _prediction_mode = None  # 'supervised', 'hybrid', or 'rule-based'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize models and services"""
        self._stress_service = StressAnalysisService()
        self._load_models()

    def _load_models(self):
        """Load all available models from disk"""
        # Try to load supervised model
        self._load_supervised_model()
        
        # Try to load hybrid model
        self._load_hybrid_model()
        
        # Determine default prediction mode
        if self._supervised_model is not None:
            self._prediction_mode = 'supervised'
            print("✅ PredictionService: Using supervised ML model")
        elif self._hybrid_model is not None:
            self._prediction_mode = 'hybrid'
            print("✅ PredictionService: Using hybrid clustering model")
        else:
            self._prediction_mode = 'rule-based'
            print("⚠️  PredictionService: Using rule-based predictions (no trained models found)")

    def _load_supervised_model(self):
        """Load the best available supervised ML model from disk"""
        try:
            # Check likely directories
            possible_dirs = ["ml_Models", "models"]
            model_dir = None
            for d in possible_dirs:
                if os.path.exists(d):
                    model_dir = d
                    break
            
            if not model_dir:
                print("ℹ️  No supervised models directory found (checked ml_Models, models).")
                return

            # Find latest regression model
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith("regression_model_") and f.endswith(".pkl")]
            
            # Fallback to generic best_model
            if not model_files:
                model_files = [f for f in os.listdir(model_dir) 
                              if f.startswith("best_model_") and f.endswith(".pkl")]
            
            if not model_files:
                print("ℹ️  No supervised models found.")
                return

            # Sort to get latest
            model_files.sort(reverse=True)
            latest_model_path = os.path.join(model_dir, model_files[0])
            
            # Initialize and load model
            self._supervised_model = WaterStressModel(model_type='regression')
            self._supervised_model.load_model(latest_model_path)
            print(f"✅ Loaded supervised model: {model_files[0]}")
            
        except Exception as e:
            print(f"⚠️  Error loading supervised model: {e}")
            self._supervised_model = None

    def _load_hybrid_model(self):
        """Load the hybrid clustering model from disk"""
        try:
            possible_dirs = ["ml_Models", "models"]
            model_dir = None
            for d in possible_dirs:
                if os.path.exists(d):
                    model_dir = d
                    break
            
            if not model_dir:
                return

            # Find hybrid model files
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith("hybrid_stress_model") and f.endswith(".pkl")]
            
            if not model_files:
                print("ℹ️  No hybrid models found.")
                return

            # Get latest
            model_files.sort(reverse=True)
            latest_model_path = os.path.join(model_dir, model_files[0])
            
            # Load hybrid model
            self._hybrid_model = HybridStressPredictor()
            self._hybrid_model.load_model(latest_model_path)
            print(f"✅ Loaded hybrid model: {model_files[0]}")
            
        except Exception as e:
            print(f"⚠️  Error loading hybrid model: {e}")
            self._hybrid_model = None

    def predict_stress(self, features: Dict[str, Any], 
                      mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict stress score and return detailed result
        
        Args:
            features: Dictionary with environmental indicators
            mode: Prediction mode override ('supervised', 'hybrid', 'rule-based')
                 If None, uses default mode
        
        Returns:
            Dictionary with predicted_stress_score, stress_level, confidence, etc.
        """
        # Determine which mode to use
        prediction_mode = mode if mode else self._prediction_mode
        
        # Validate mode availability
        if prediction_mode == 'supervised' and self._supervised_model is None:
            print("⚠️  Supervised model not available, falling back to hybrid/rule-based")
            prediction_mode = 'hybrid' if self._hybrid_model else 'rule-based'
        
        if prediction_mode == 'hybrid' and self._hybrid_model is None:
            print("⚠️  Hybrid model not available, falling back to rule-based")
            prediction_mode = 'rule-based'
        
        # Perform prediction based on mode
        if prediction_mode == 'supervised':
            return self._predict_supervised(features)
        elif prediction_mode == 'hybrid':
            return self._predict_hybrid(features)
        else:
            return self._predict_rule_based(features)

    def _predict_supervised(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using supervised ML model"""
        expected_features = self._supervised_model.feature_names
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        if expected_features:
            # Handle missing features
            fill_value = np.nan if isinstance(self._supervised_model.model, Pipeline) else 0
            
            for col in expected_features:
                if col not in features_df.columns:
                    features_df[col] = fill_value
            
            # Reorder columns
            features_df = features_df[expected_features]

        # Predict
        prediction_val = self._supervised_model.predict(features_df)[0]
        stress_level = self._stress_service.determine_stress_level(prediction_val)
        
        return {
            "predicted_stress_score": float(prediction_val),
            "stress_level": stress_level,
            "prediction_mode": "supervised",
            "model_used": self._supervised_model.model_info.get('model_name', 'unknown'),
            "confidence": 0.85,  # Default confidence for supervised models
            "timestamp": datetime.now().isoformat()
        }

    def _predict_hybrid(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using hybrid clustering model"""
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        # Predict with confidence
        result = self._hybrid_model.predict_with_confidence(features_df)
        
        prediction_val = result.iloc[0]['stress_score']
        stress_level = result.iloc[0]['stress_level']
        confidence = result.iloc[0]['confidence']
        cluster_id = result.iloc[0]['cluster']
        
        return {
            "predicted_stress_score": float(prediction_val),
            "stress_level": stress_level,
            "prediction_mode": "hybrid",
            "cluster_id": int(cluster_id),
            "confidence": float(confidence),
            "model_used": "hybrid_clustering",
            "timestamp": datetime.now().isoformat()
        }

    def _predict_rule_based(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using pure rule-based approach"""
        analysis = self._stress_service.get_stress_analysis(features)
        
        return {
            "predicted_stress_score": analysis['score'],
            "stress_level": analysis['level'],
            "prediction_mode": "rule-based",
            "confidence": 1.0 if len(analysis['missing_indicators']) == 0 else 0.7,
            "model_used": "domain_knowledge",
            "recommendations": analysis.get('recommendations', []),
            "timestamp": datetime.now().isoformat()
        }

    def predict_batch(self, features_list: List[Dict[str, Any]], 
                     mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict stress for multiple samples
        
        Args:
            features_list: List of feature dictionaries
            mode: Prediction mode ('supervised', 'hybrid', 'rule-based')
        
        Returns:
            List of prediction results
        """
        return [self.predict_stress(features, mode) for features in features_list]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models and current mode"""
        return {
            "current_mode": self._prediction_mode,
            "supervised_available": self._supervised_model is not None,
            "hybrid_available": self._hybrid_model is not None,
            "rule_based_available": True,  # Always available
            "supervised_info": self._supervised_model.model_info if self._supervised_model else None,
            "hybrid_info": {
                "n_clusters": self._hybrid_model.n_clusters if self._hybrid_model else None,
                "region_type": self._hybrid_model.region_type.value if self._hybrid_model else None
            } if self._hybrid_model else None
        }

    def set_prediction_mode(self, mode: str):
        """
        Manually set the prediction mode
        
        Args:
            mode: 'supervised', 'hybrid', or 'rule-based'
        """
        valid_modes = ['supervised', 'hybrid', 'rule-based']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from {valid_modes}")
        
        # Check if mode is available
        if mode == 'supervised' and self._supervised_model is None:
            raise ValueError("Supervised model not available")
        if mode == 'hybrid' and self._hybrid_model is None:
            raise ValueError("Hybrid model not available")
        
        self._prediction_mode = mode
        print(f"✅ Prediction mode set to: {mode}")

    def reload_models(self):
        """Force reload of all models"""
        print("🔄 Reloading all models...")
        self._supervised_model = None
        self._hybrid_model = None
        self._load_models()

    def compare_predictions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare predictions from all available models
        
        Args:
            features: Feature dictionary
        
        Returns:
            Dictionary with predictions from each available model
        """
        results = {
            "features": features,
            "predictions": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Try each mode
        if self._supervised_model:
            try:
                results["predictions"]["supervised"] = self._predict_supervised(features)
            except Exception as e:
                results["predictions"]["supervised"] = {"error": str(e)}
        
        if self._hybrid_model:
            try:
                results["predictions"]["hybrid"] = self._predict_hybrid(features)
            except Exception as e:
                results["predictions"]["hybrid"] = {"error": str(e)}
        
        try:
            results["predictions"]["rule_based"] = self._predict_rule_based(features)
        except Exception as e:
            results["predictions"]["rule_based"] = {"error": str(e)}
        
        # Calculate consensus
        valid_scores = [
            p["predicted_stress_score"] 
            for p in results["predictions"].values() 
            if "predicted_stress_score" in p
        ]
        
        if valid_scores:
            results["consensus"] = {
                "mean_score": float(np.mean(valid_scores)),
                "std_score": float(np.std(valid_scores)),
                "min_score": float(np.min(valid_scores)),
                "max_score": float(np.max(valid_scores)),
                "agreement": float(np.std(valid_scores)) < 10.0  # Low variance = agreement
            }
        
        return results

    def get_recommendations(self, features: Dict[str, Any]) -> List[str]:
        """Get actionable recommendations based on prediction"""
        prediction = self.predict_stress(features)
        
        # Get recommendations from stress service
        recommendations = self._stress_service.generate_recommendations(
            features, 
            prediction['stress_level']
        )
        
        return recommendations

    # Backward compatibility methods
    def get_model_performance(self):
        """Get performance metrics of loaded model (backward compatible)"""
        if self._supervised_model and self._supervised_model.model_info:
            return self._supervised_model.model_info
        return self.get_model_info()
