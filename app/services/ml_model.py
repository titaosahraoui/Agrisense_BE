import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib
from pathlib import Path

# Import ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class WaterStressModel:
    def __init__(self, model_type: str = 'regression'):
        """
        Initialize Water Stress Prediction Model
        
        model_type: 'regression' to predict score (0-100)
                   'classification' to predict level (low/medium/high/severe)
        """
        self.model_type = model_type
        self.model = None  # Will hold Pipeline or legacy model
        self.scaler = None # Legacy support
        self.label_encoder = LabelEncoder() if model_type == 'classification' else None
        self.feature_names = None
        self.model_info = {}
        
        # Model Configurations
        self.model_configs = {
            'regression': {
                'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
                'xgb': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
                'gbr': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            },
            'classification': {
                'rf': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
                'xgb': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
                'gbr': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            }
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, model_name: str = 'xgb'):
        """
        Train the model using a Scikit-Learn Pipeline
        """
        print(f"Training {self.model_type} model with {model_name}...")
        
        # Encode targets for classification
        if self.model_type == 'classification':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        self.feature_names = X.columns.tolist()
        
        # Define Preprocessing Pipeline
        # Identify column types
        categorical_cols = [c for c in X.columns if X[c].dtype == 'object' or c in ['wilaya_code', 'season']]
        numerical_cols = [c for c in X.columns if c not in categorical_cols]
        
        # Numerical Transformer: Impute missing -> Scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical Transformer: Impute missing -> OneHotEncode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine in ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
            
        # Create Full Pipeline
        regressor = self.model_configs[self.model_type][model_name]
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', regressor)])
        
        # Train
        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.scaler = None # Not used in pipeline mode
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Train Score: {train_score:.3f}")
        print(f"Test Score: {test_score:.3f}")
        
        # Detailed Metrics
        y_pred = self.model.predict(X_test)
        
        self._calculate_metrics(y_test, y_pred, train_score, test_score)
        
        # Save model info
        self.model_info.update({
            'model_type': self.model_type,
            'model_name': model_name,
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'pipeline': True
        })
        
        return {
            'model': self.model,
            'metrics': self.model_info['metrics']
        }

    def _calculate_metrics(self, y_true, y_pred, train_score, test_score):
        if self.model_type == 'regression':
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            self.model_info['metrics'] = {
                'train_score': float(train_score), 'test_score': float(test_score),
                'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)
            }
        else:
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            y_true_labels = self.label_encoder.inverse_transform(y_true)
            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            self.model_info['metrics'] = {
                'train_score': float(train_score), 'test_score': float(test_score),
                'accuracy': float(accuracy)
            }

    def predict(self, X: pd.DataFrame, return_proba: bool = False):
        """Make predictions using pipeline or legacy model"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Check if using Pipeline (new way) or Legacy
        if isinstance(self.model, Pipeline):
            # Pipeline handles imputation and scaling
            if self.model_type == 'regression':
                return self.model.predict(X)
            else:
                if return_proba and hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                preds = self.model.predict(X)
                return self.label_encoder.inverse_transform(preds)
        else:
            # Legacy: Manual scaling
            if self.scaler is None:
                raise ValueError("Legacy model detected but scaler is missing")
            
            # Fill missing for legacy (simple 0 fill as fallback)
            X_filled = X.fillna(0)
            X_scaled = self.scaler.transform(X_filled)
            
            if self.model_type == 'regression':
                return self.model.predict(X_scaled)
            else:
                preds = self.model.predict(X_scaled)
                return self.label_encoder.inverse_transform(preds)

    def save_model(self, filepath: str = "models/water_stress_model.pkl"):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler, # None if pipeline
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_info': self.model_info,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from disk (supports legacy and pipeline)"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.label_encoder = model_data.get('label_encoder')
            self.feature_names = model_data.get('feature_names')
            self.model_info = model_data.get('model_info', {})
            self.model_type = model_data.get('model_type', 'regression')
            
            print(f"Loaded {self.model_type} model from {filepath}")
            if isinstance(self.model, Pipeline):
                print("Type: Scikit-Learn Pipeline (Robust Preprocessing)")
            else:
                print("Type: Legacy Model (Manual Scaling)")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        if self.model is None: return None
        if self.model_type == 'classification': y = self.label_encoder.transform(y)
        return cross_val_score(self.model, X, y, cv=cv, n_jobs=-1)

    def _calculate_feature_importance(self, feature_names):
        # Difficult with Pipeline as features are transformed
        pass