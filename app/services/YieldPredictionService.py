# YieldPredictionService.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive
import seaborn as sns
from scipy import stats

from .data_loader import DataLoader

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> Dict:
        """Initialize ML models"""
        return {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                objective='reg:squarederror'
            ),
            'Linear Regression': LinearRegression(),
            'Support Vector Regression': SVR(
                kernel='rbf', 
                C=100, 
                gamma=0.1, 
                epsilon=0.1
            )
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train multiple models and return results"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='r2'
            )
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
        
        return results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate a trained model"""
        y_pred = model.predict(X_test)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'predictions': y_pred,
            'feature_importances': self._get_feature_importance(model, X_test.columns) 
                                   if hasattr(model, 'feature_importances_') else None
        }
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        return None
    
    def select_best_model(self, results: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict]:
        """Select the best model based on performance"""
        best_model_name = None
        best_score = -np.inf
        best_metrics = {}
        
        for name, result in results.items():
            metrics = self.evaluate_model(result['model'], X_test, y_test)
            score = metrics['r2'] - 0.1 * metrics['rmse']  # Composite score
            
            if score > best_score:
                best_score = score
                best_model_name = name
                best_metrics = metrics
                best_metrics.update({
                    'model_name': name,
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                })
        
        return results[best_model_name]['model'], best_metrics


class FeatureEngineer:
    """Handles feature engineering and preparation"""
    
    def __init__(self):
        self.feature_names = []
    
    def prepare_features(self, df: pd.DataFrame, target_type: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Prepare features for ML models"""
        # Base agricultural features
        base_features = [
            'Total_Agricultural_Area',
            'Cultivated_Area',
            'Fallow_Land',
            'Fruit_Trees'
        ]
        
        # Meteorological features
        meteo_features = [
            'avg_temperature_hiver', 'avg_temperature_été',
            'avg_rainfall_hiver', 'avg_rainfall_été',
            'avg_humidity_hiver', 'avg_humidity_été',
            'avg_ndvi_hiver', 'avg_ndvi_été'
        ]
        
        # Select target
        target_mapping = {
            'summer': 'Total_Summer_Yield',
            'winter': 'Total_Winter_Yield',
            'maize': 'Summer_Maize_Yield',
            'sorghum': 'Summer_Sorghum_Yield'
        }
        
        if target_type not in target_mapping:
            raise ValueError(f"Invalid target_type: {target_type}")
        
        target_col = target_mapping[target_type]
        
        # Filter data with positive yield
        data = df[df[target_col] > 0].copy()
        
        # Combine available features
        available_features = [f for f in base_features + meteo_features 
                            if f in data.columns]
        
        # Add engineered features
        engineered_features = self._add_engineered_features(data, available_features)
        all_features = available_features + engineered_features
        
        X = data[all_features].copy()
        y = data[target_col].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Store info
        info = {
            'wilaya_codes': data['Wilaya_Code'].values,
            'wilaya_names': data['Wilaya_Name'].values,
            'years': data['Year'].values,
            'feature_names': all_features
        }
        
        self.feature_names = all_features
        
        return X, y, info
    
    def _add_engineered_features(self, data: pd.DataFrame, base_features: List[str]) -> List[str]:
        """Add engineered features"""
        engineered = []
        
        # Seasonal interaction features
        if all(col in data.columns for col in ['avg_temperature_hiver', 'avg_temperature_été']):
            data['temp_diff_summer_winter'] = data['avg_temperature_été'] - data['avg_temperature_hiver']
            engineered.append('temp_diff_summer_winter')
        
        if all(col in data.columns for col in ['avg_rainfall_hiver', 'avg_rainfall_été']):
            data['rainfall_ratio_summer_winter'] = data['avg_rainfall_été'] / (data['avg_rainfall_hiver'] + 0.001)
            engineered.append('rainfall_ratio_summer_winter')
        
        # Land use ratios
        if all(col in data.columns for col in ['Cultivated_Area', 'Total_Agricultural_Area']):
            data['cultivation_intensity'] = data['Cultivated_Area'] / (data['Total_Agricultural_Area'] + 0.001)
            engineered.append('cultivation_intensity')
        
        return engineered
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        
        # Specific handling for engineered features that might be NaN due to division by zero
        if 'cultivation_intensity' in X.columns:
            X['cultivation_intensity'] = X['cultivation_intensity'].fillna(0)
        if 'rainfall_ratio_summer_winter' in X.columns:
            X['rainfall_ratio_summer_winter'] = X['rainfall_ratio_summer_winter'].fillna(0)
            
        for col in X.columns:
            if X[col].isnull().any():
                if col.startswith('avg_'):
                    # For meteorological features, use column mean
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    # For other features, use median
                    X[col] = X[col].fillna(X[col].median())
        
        # Final fallback for any remaining NaNs (e.g. if mean is NaN because all are NaN)
        X = X.fillna(0)
        return X


class YieldPredictionService:
    """Main service for cereal yield prediction and analysis"""
    
    def __init__(self, db_url: str, csv_base_path: str = "./outputs/data", models_dir: str = "models"):
        self.db_url = db_url
        self.csv_base_path = csv_base_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(db_url, csv_base_path)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # Models and data
        self.models = {}
        self.model_metrics = {}
        self.master_df = None
        self.cluster_results = None
        self.all_data = None
        
    def load_all_data(self) -> Dict:
        """Load all data from database and CSV files"""
        self.all_data = self.data_loader.load_all_data()
        self.master_df = self.all_data['master_df']
        return self.all_data
    
    def prepare_training_data(self, target_type: str, test_size: float = 0.2, 
                             time_based_split: bool = True) -> Dict:
        """Prepare training and testing data"""
        X, y, info = self.feature_engineer.prepare_features(self.master_df, target_type)
        
        if time_based_split:
            # Time-based split: use earlier years for training
            unique_years = sorted(info['years'])
            split_idx = int(len(unique_years) * (1 - test_size))
            train_years = unique_years[:split_idx]
            test_years = unique_years[split_idx:]
            
            train_mask = np.isin(info['years'], train_years)
            test_mask = np.isin(info['years'], test_years)
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            print(f"Time-based split: Train years {train_years}, Test years {test_years}")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'info': info,
            'feature_names': self.feature_engineer.feature_names
        }
    
    def train_crop_model(self, target_type: str, save_model: bool = True) -> Dict:
        """Train model for a specific crop type"""
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL FOR {target_type.upper()} CEREALS")
        print(f"{'='*80}")
        
        # Prepare data
        data = self.prepare_training_data(target_type, time_based_split=True)
        
        # Train models
        training_results = self.model_trainer.train_models(
            data['X_train'], data['y_train']
        )
        
        # Select and evaluate best model
        best_model, metrics = self.model_trainer.select_best_model(
            training_results, data['X_test'], data['y_test']
        )
        
        # Store results
        self.models[target_type] = best_model
        self.model_metrics[target_type] = metrics
        
        # Save model if requested
        if save_model:
            self._save_model(best_model, target_type)
        
        # Generate visualizations
        self._generate_training_visualizations(
            target_type, training_results, data, metrics
        )
        
        return {
            'model': best_model,
            'metrics': metrics,
            'training_results': training_results,
            'data_info': data['info']
        }
    
    def train_all_models(self) -> Dict:
        """Train models for all crop types"""
        results = {}
        
        for target_type in ['summer', 'winter', 'maize', 'sorghum']:
            try:
                result = self.train_crop_model(target_type)
                results[target_type] = result
            except Exception as e:
                print(f"Error training {target_type} model: {str(e)}")
                continue
        
        # Perform clustering
        self.perform_clustering()
        
        # Save all models
        self.save_models()
        
        return results
    
    def perform_clustering(self, n_clusters: int = 4) -> Dict:
        """Perform K-means clustering on wilayas"""
        cluster_features = [
            'Total_Agricultural_Area',
            'Cultivated_Area',
            'Total_Summer_Yield',
            'Total_Winter_Yield'
        ]
        
        available_features = [f for f in cluster_features if f in self.master_df.columns]
        latest_data = self.master_df.groupby('Wilaya_Code').last().reset_index()
        
        cluster_data = latest_data[available_features + ['Wilaya_Code', 'Wilaya_Name']].copy()
        cluster_data = cluster_data.fillna(0)
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data[available_features])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        cluster_data['Cluster'] = cluster_labels
        
        # Cluster summary
        cluster_summary = cluster_data.groupby('Cluster')[available_features].mean()
        
        self.cluster_results = {
            'cluster_data': cluster_data,
            'cluster_summary': cluster_summary,
            'kmeans': kmeans,
            'scaler': scaler
        }
        
        # Generate cluster visualization
        self._visualize_clusters(cluster_data, cluster_summary)
        
        return self.cluster_results
    
    def predict_yield(self, wilaya_code: int, season: str, 
                     meteorological_features: Dict) -> float:
        """Predict yield for a specific wilaya and season"""
        if season not in self.models:
            raise ValueError(f"No model trained for {season} season")
        
        model = self.models[season]
        
        # Get wilaya data
        wilaya_data = self.master_df[
            self.master_df['Wilaya_Code'] == wilaya_code
        ].iloc[-1]
        
        # Prepare features
        features = {
            'Total_Agricultural_Area': wilaya_data['Total_Agricultural_Area'],
            'Cultivated_Area': wilaya_data['Cultivated_Area'],
            'Fallow_Land': wilaya_data['Fallow_Land'],
            'Fruit_Trees': wilaya_data.get('Fruit_Trees', 0),
            **meteorological_features
        }
        
        # Create feature DataFrame
        X = pd.DataFrame([features])
        
        # Add engineered features
        self.feature_engineer._add_engineered_features(X, list(X.columns))
        
        # Handle missing values rigorously
        X = self.feature_engineer._handle_missing_values(X)
        
        # Ensure all required features are present
        required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X.columns
        missing_features = set(required_features) - set(X.columns)
        
        if missing_features:
            for feature in missing_features:
                X[feature] = 0  # Add missing features with default value
        
        # Reorder columns to match training
        X = X[list(required_features)]
        
        # Predict
        prediction = model.predict(X)[0]
        
        return float(prediction)
    
    def get_all_predictions(self, year: int = 2024) -> List[Dict]:
        """Get predictions for all wilayas"""
        if not self.models:
            raise ValueError("No models trained")
        
        predictions = []
        latest_data = self.master_df.groupby('Wilaya_Code').last()
        
        # Determine features required by models to ensure we have them all
        # We'll use the summer model as a reference, or union of all
        required_cols = set()
        for model in self.models.values():
             if hasattr(model, 'feature_names_in_'):
                 required_cols.update(model.feature_names_in_)
        
        for wilaya_code, row in latest_data.iterrows():
            # Prepare features for each model
            features = {
                'Total_Agricultural_Area': row['Total_Agricultural_Area'],
                'Cultivated_Area': row['Cultivated_Area'],
                'Fallow_Land': row['Fallow_Land'],
                'Fruit_Trees': row.get('Fruit_Trees', 0),
                'avg_temperature_hiver': row.get('avg_temperature_hiver', 0),
                'avg_temperature_été': row.get('avg_temperature_été', 0),
                'avg_rainfall_hiver': row.get('avg_rainfall_hiver', 0),
                'avg_rainfall_été': row.get('avg_rainfall_été', 0),
                'avg_humidity_hiver': row.get('avg_humidity_hiver', 0),
                'avg_humidity_été': row.get('avg_humidity_été', 0),
                'avg_ndvi_hiver': row.get('avg_ndvi_hiver', 0),
                'avg_ndvi_été': row.get('avg_ndvi_été', 0)
            }
            
            X = pd.DataFrame([features])
            
            # Add engineered features
            self.feature_engineer._add_engineered_features(X, list(X.columns))
            
            # Handle missing values rigorously
            X = self.feature_engineer._handle_missing_values(X)
            
            # Ensure columns match for each model before prediction
            wilaya_pred = {'wilaya_code': int(wilaya_code), 'wilaya_name': row['Wilaya_Name'], 'year': year}
            
            for crop_type, model in self.models.items():
                if hasattr(model, 'feature_names_in_'):
                    # Align features
                    X_crop = X.copy()
                    missing = set(model.feature_names_in_) - set(X_crop.columns)
                    for c in missing: 
                        X_crop[c] = 0
                    X_crop = X_crop[list(model.feature_names_in_)]
                    wilaya_pred[f'{crop_type}_yield'] = float(model.predict(X_crop)[0])
                else:
                    wilaya_pred[f'{crop_type}_yield'] = float(model.predict(X)[0])
            
            predictions.append(wilaya_pred)
        
        return predictions
    
    def generate_recommendations(self) -> Dict:
        """Generate strategic recommendations based on model insights"""
        if self.master_df is None:
            raise ValueError("No data available for recommendations")
        
        # High potential regions
        avg_yield_by_region = self.master_df.groupby('Wilaya_Name')['Total_Summer_Yield'].mean()
        high_potential = avg_yield_by_region.nlargest(5).to_dict()
        low_yield = avg_yield_by_region.nsmallest(5).to_dict()
        
        # Seasonal performance
        summer_avg = self.master_df['Total_Summer_Yield'].mean()
        winter_avg = self.master_df['Total_Winter_Yield'].mean()
        
        # Land use optimization
        area_yield_corr = self.master_df[['Total_Summer_Area', 'Total_Summer_Yield']].corr().iloc[0, 1]
        
        # Model-based insights
        model_insights = {}
        for crop_type, metrics in self.model_metrics.items():
            model_insights[crop_type] = {
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'best_features': self._get_top_features(crop_type, top_n=5)
            }
        
        return {
            'high_potential_regions': high_potential,
            'low_yield_regions': low_yield,
            'seasonal_performance': {
                'summer_avg': float(summer_avg),
                'winter_avg': float(winter_avg),
                'recommendation': 'summer' if summer_avg > winter_avg else 'winter'
            },
            'land_use_optimization': {
                'area_yield_correlation': float(area_yield_corr),
                'recommendation': 'intensive' if area_yield_corr < 0 else 'extensive'
            },
            'model_insights': model_insights
        }
    
    def _get_top_features(self, crop_type: str, top_n: int = 5) -> List[Dict]:
        """Get top important features for a model"""
        if crop_type not in self.model_metrics:
            return []
        
        metrics = self.model_metrics[crop_type]
        if 'feature_importances' not in metrics or metrics['feature_importances'] is None:
            return []
        
        importances = metrics['feature_importances']
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{'feature': feat, 'importance': float(imp)} for feat, imp in sorted_features]
    
    def _generate_training_visualizations(self, target_type: str, training_results: Dict, 
                                         data: Dict, metrics: Dict):
        """Generate training visualizations"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Model comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Model performance comparison
            model_names = list(training_results.keys())
            cv_scores = [training_results[name]['cv_mean'] for name in model_names]
            cv_stds = [training_results[name]['cv_std'] for name in model_names]
            
            axes[0].bar(model_names, cv_scores, yerr=cv_stds, capsize=5, alpha=0.7)
            axes[0].set_title('Cross-Validation R² Scores')
            axes[0].set_ylabel('R² Score')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Actual vs Predicted
            best_model = self.models[target_type]
            y_pred = best_model.predict(data['X_test'])
            
            axes[1].scatter(data['y_test'], y_pred, alpha=0.6)
            max_val = max(data['y_test'].max(), y_pred.max())
            axes[1].plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
            axes[1].set_xlabel('Actual Yield')
            axes[1].set_ylabel('Predicted Yield')
            axes[1].set_title(f'Actual vs Predicted (R²={metrics["r2"]:.3f})')
            axes[1].grid(True, alpha=0.3)
            
            # Residuals
            residuals = data['y_test'] - y_pred
            axes[2].scatter(y_pred, residuals, alpha=0.6)
            axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[2].set_xlabel('Predicted Values')
            axes[2].set_ylabel('Residuals')
            axes[2].set_title('Residual Plot')
            axes[2].grid(True, alpha=0.3)
            
            # Error distribution
            axes[3].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[3].axvline(x=0, color='r', linestyle='--', alpha=0.7)
            axes[3].set_xlabel('Prediction Error')
            axes[3].set_ylabel('Frequency')
            axes[3].set_title(f'Error Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}')
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle(f'{target_type.upper()} Cereals - Model Training Results', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save figure
            fig_path = self.models_dir / f'{target_type}_training_results.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved training visualization to {fig_path}")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
    
    def _visualize_clusters(self, cluster_data: pd.DataFrame, cluster_summary: pd.DataFrame):
        """Visualize clustering results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cluster scatter plot
            scatter = axes[0].scatter(
                cluster_data['Total_Agricultural_Area'],
                cluster_data['Total_Summer_Yield'],
                c=cluster_data['Cluster'],
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            axes[0].set_xlabel('Total Agricultural Area (ha)')
            axes[0].set_ylabel('Summer Yield (qx/ha)')
            axes[0].set_title('Wilaya Clusters')
            axes[0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0])
            
            # Cluster summary
            cluster_summary.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Cluster Characteristics')
            axes[1].set_xlabel('Cluster')
            axes[1].set_ylabel('Average Value')
            axes[1].legend(title='Features')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=0)
            
            plt.suptitle('Wilaya Clustering Analysis', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save figure
            fig_path = self.models_dir / 'clustering_analysis.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved clustering visualization to {fig_path}")
            
        except Exception as e:
            print(f"Error generating clustering visualization: {str(e)}")
    
    def _save_model(self, model, target_type: str):
        """Save a single model"""
        model_path = self.models_dir / f'{target_type}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Saved {target_type} model to {model_path}")
    
    def save_models(self):
        """Save all models and data to disk"""
        # Save models
        for target_type, model in self.models.items():
            model_path = self.models_dir / f'{target_type}_model.pkl'
            joblib.dump(model, model_path)
        
        # Save metrics
        metrics_path = self.models_dir / 'model_metrics.pkl'
        joblib.dump(self.model_metrics, metrics_path)
        
        # Save master dataframe
        if self.master_df is not None:
            data_path = self.models_dir / 'master_data.pkl'
            self.master_df.to_pickle(data_path)
        
        # Save cluster results
        if self.cluster_results is not None:
            cluster_path = self.models_dir / 'cluster_results.pkl'
            joblib.dump(self.cluster_results, cluster_path)
        
        print(f"All models and data saved to {self.models_dir}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            # Load models
            model_files = list(self.models_dir.glob('*_model.pkl'))
            for model_file in model_files:
                target_type = model_file.stem.replace('_model', '')
                self.models[target_type] = joblib.load(model_file)
            
            # Load metrics
            metrics_path = self.models_dir / 'model_metrics.pkl'
            if metrics_path.exists():
                self.model_metrics = joblib.load(metrics_path)
            
            # Load data
            data_path = self.models_dir / 'master_data.pkl'
            if data_path.exists():
                self.master_df = pd.read_pickle(data_path)
            
            # Load cluster results
            cluster_path = self.models_dir / 'cluster_results.pkl'
            if cluster_path.exists():
                self.cluster_results = joblib.load(cluster_path)
            
            print(f"Loaded {len(self.models)} models from {self.models_dir}")
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading models: {str(e)}")
            return False