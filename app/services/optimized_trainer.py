import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class OptimizedModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train and compare multiple models optimized for agricultural data"""
        
        print("🤖 Training multiple models for comparison...")
        
        # Define models with optimized parameters for time series/agricultural data
        model_configs = {
            "random_forest": {
                "model": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                "description": "Random Forest - Good for nonlinear relationships"
            },
            "xgboost": {
                "model": xgb.XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                "description": "XGBoost - Great for tabular data with missing values"
            },
            "lightgbm": {
                "model": lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                "description": "LightGBM - Fast and handles large datasets well"
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                "description": "Gradient Boosting - Robust to outliers"
            },
            "ridge": {
                "model": Ridge(alpha=1.0, random_state=42),
                "description": "Ridge Regression - Good baseline linear model"
            }
        }
        
        
        # Prepare Preprocessor (Scaling + Encoding)
        # Identify columns
        numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        # Explicitly check for season or object types
        categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
        
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
        transformers = [
            ('num', StandardScaler(), numeric_features)
        ]
        
        if len(categorical_features) > 0:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            )
            
        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        
        # Fit and Transform
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after transformation (for importance)
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = numeric_features
            
        # Train each model
        for name, config in model_configs.items():
            print(f"\n📊 Training {name}...")
            
            try:
                model = config["model"]
                
                # Train model
                model.fit(X_train_processed, y_train)
                self.models[name] = {
                    "model": model,
                    "preprocessor": preprocessor, # Store preprocessor
                    "scaler": preprocessor,       # Alias for compatibility
                    "description": config["description"]
                }
                
                # Make predictions
                y_pred = model.predict(X_test_processed)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                self.results[name] = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "description": config["description"]
                }
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    if len(importances) == len(feature_names):
                        self.feature_importance[name] = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                
                print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"   Error training {name}: {e}")
                self.results[name] = {"error": str(e)}
        
        # Determine best model
        if self.results:
            valid_results = {k: v for k, v in self.results.items() if 'rmse' in v}
            if valid_results:
                self.best_model_name = min(valid_results.items(), 
                                         key=lambda x: x[1]['rmse'])[0]
                self.best_model = self.models[self.best_model_name]
                
                print(f"\n🏆 Best model: {self.best_model_name}")
                print(f"   RMSE: {self.results[self.best_model_name]['rmse']:.4f}")
                print(f"   R²: {self.results[self.best_model_name]['r2']:.4f}")
        
        return self.results
    
    def time_series_cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                                 n_splits: int = 5) -> Dict:
        """Time series cross-validation to avoid data leakage"""
        
        print("\n🔄 Performing time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features for this fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Train best model on this fold
            if self.best_model:
                model = self.best_model["model"]
                model.fit(X_train_scaled, y_train_fold)
                y_pred = model.predict(X_val_scaled)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(rmse)
                
                print(f"   Fold {fold}: RMSE = {rmse:.4f}")
        
        if cv_scores:
            cv_results = {
                "mean_rmse": float(np.mean(cv_scores)),
                "std_rmse": float(np.std(cv_scores)),
                "cv_scores": [float(score) for score in cv_scores],
                "n_splits": n_splits
            }
            
            print(f"\n📈 CV Results: Mean RMSE = {cv_results['mean_rmse']:.4f} "
                  f"(± {cv_results['std_rmse']:.4f})")
            
            return cv_results
        
        return {}
    
    def analyze_feature_importance(self) -> Dict:
        """Analyze and compare feature importance across models"""
        
        if not self.feature_importance:
            return {}
        
        print("\n🔍 Analyzing feature importance...")
        
        # Combine importances from all models
        all_importances = {}
        for model_name, importance_df in self.feature_importance.items():
            print(f"\n   Top features for {model_name}:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"     {row['feature']}: {row['importance']:.4f}")
            
            # Store for comparison
            for _, row in importance_df.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                if feature not in all_importances:
                    all_importances[feature] = {}
                all_importances[feature][model_name] = importance
        
        # Calculate average importance
        avg_importance = {}
        for feature, model_importances in all_importances.items():
            avg_importance[feature] = np.mean(list(model_importances.values()))
        
        # Sort by average importance
        avg_importance_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'avg_importance': list(avg_importance.values())
        }).sort_values('avg_importance', ascending=False)
        
        print("\n📊 Top features (average across all models):")
        for idx, row in avg_importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['avg_importance']:.4f}")
        
        return {
            "by_model": {k: v.to_dict('records') for k, v in self.feature_importance.items()},
            "average": avg_importance_df.to_dict('records')
        }
    
    def save_models(self, output_dir: str = "models"):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each model
        saved_files = []
        for name, model_data in self.models.items():
            filename = os.path.join(output_dir, f"{name}_model_{timestamp}.pkl")
            
            save_data = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'name': name,
                'description': model_data['description'],
                'performance': self.results.get(name, {}),
                'timestamp': timestamp
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            
            saved_files.append(filename)
        
        # Save best model separately
        if self.best_model:
            best_filename = os.path.join(output_dir, f"best_model_{timestamp}.pkl")
            with open(best_filename, 'wb') as f:
                pickle.dump({
                    'model': self.best_model['model'],
                    'scaler': self.best_model['scaler'],
                    'name': self.best_model_name,
                    'performance': self.results[self.best_model_name],
                    'timestamp': timestamp,
                    'all_results': self.results
                }, f)
            
            saved_files.append(best_filename)
        
        # Save results as JSON
        results_file = os.path.join(output_dir, f"training_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'best_model': self.best_model_name,
                'feature_importance': self.analyze_feature_importance(),
                'timestamp': timestamp,
                'models_trained': list(self.models.keys())
            }, f, indent=2)
        
        print(f"\n💾 Saved {len(saved_files)} model files to {output_dir}/")
        print(f"📄 Results saved to {results_file}")
        
        return saved_files