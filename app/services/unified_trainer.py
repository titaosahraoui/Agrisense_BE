"""
Complete Unified Training System - FIXED VERSION
All methods required by routes included
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from .ml_model import WaterStressModel
from .hybrid_stress_model import HybridStressPredictor
from .stress_analysis import StressAnalysisService, RegionType
from .feature_engineering import FeatureEngineer


class UnifiedModelTrainer:
    """
    Complete unified trainer with all required methods
    """
    
    def __init__(self, output_dir: str = "ml_Models"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.supervised_model = None
        self.hybrid_model = None
        self.training_results = {}
        
    def train_supervised(self, X: pd.DataFrame, y: pd.Series, 
                        model_name: str = 'xgb',
                        test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train supervised ML model
        
        Args:
            X: Feature DataFrame
            y: Target (stress scores)
            model_name: 'xgb', 'rf', or 'gbr'
            test_size: Test set proportion
        
        Returns:
            Training results dictionary
        """
        print("=" * 60)
        print("🤖 TRAINING SUPERVISED ML MODEL")
        print("=" * 60)
        
        # Initialize model
        self.supervised_model = WaterStressModel(model_type='regression')
        
        # Train
        result = self.supervised_model.train(
            X=X, 
            y=y, 
            test_size=test_size, 
            model_name=model_name
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.output_dir}/regression_model_{timestamp}.pkl"
        self.supervised_model.save_model(model_path)
        
        self.training_results['supervised'] = {
            'metrics': result['metrics'],
            'model_path': model_path,
            'model_name': model_name,
            'timestamp': timestamp
        }
        
        print(f"\n✅ Supervised model saved to {model_path}")
        return self.training_results['supervised']
    
    def train_hybrid(self, data: pd.DataFrame, 
                    region_type: RegionType = RegionType.MEDITERRANEAN,
                    n_clusters: int = 5) -> Dict[str, Any]:
        """
        Train hybrid clustering model (no labels needed)
        
        Args:
            data: DataFrame with features (ndvi, temperature, precipitation, etc.)
            region_type: Algeria region type
            n_clusters: Number of clusters to create
        
        Returns:
            Training results dictionary
        """
        print("=" * 60)
        print("🔬 TRAINING HYBRID CLUSTERING MODEL (UNSUPERVISED)")
        print("=" * 60)
        
        # Initialize model
        self.hybrid_model = HybridStressPredictor(
            region_type=region_type,
            n_clusters=n_clusters
        )
        
        # Train
        self.hybrid_model.fit(data)
        
        # Get cluster summary
        cluster_summary = self.hybrid_model.get_cluster_summary()
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{self.output_dir}/hybrid_stress_model_{timestamp}.pkl"
        self.hybrid_model.save_model(model_path)
        
        # Calculate metrics by comparing with rule-based predictions
        metrics = self._evaluate_hybrid_model(data)
        
        self.training_results['hybrid'] = {
            'cluster_summary': cluster_summary.to_dict('records'),
            'metrics': metrics,
            'model_path': model_path,
            'region_type': region_type.value,
            'n_clusters': n_clusters,
            'timestamp': timestamp
        }
        
        print(f"\n✅ Hybrid model saved to {model_path}")
        return self.training_results['hybrid']
    
    def _evaluate_hybrid_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate hybrid model by comparing with rule-based predictions"""
        print("\n📊 Evaluating hybrid model...")
        
        try:
            # Get predictions from hybrid model
            predictions = self.hybrid_model.predict(data)
            
            # Get rule-based predictions for comparison
            stress_service = StressAnalysisService(self.hybrid_model.region_type)
            
            rule_based_scores = []
            for _, row in data.iterrows():
                indicators = {
                    'ndvi': row.get('ndvi'),
                    'temperature': row.get('temperature', row.get('temperature_avg')),
                    'precipitation': row.get('precipitation'),
                    'evapotranspiration': row.get('evapotranspiration'),
                    'humidity': row.get('humidity')
                }
                score = stress_service.calculate_stress_score(indicators)
                rule_based_scores.append(score)
            
            rule_based_scores = np.array(rule_based_scores)
            hybrid_scores = predictions['stress_score'].values
            
            # Calculate metrics
            mae = np.mean(np.abs(hybrid_scores - rule_based_scores))
            rmse = np.sqrt(np.mean((hybrid_scores - rule_based_scores) ** 2))
            correlation = np.corrcoef(hybrid_scores, rule_based_scores)[0, 1]
            
            # Level agreement
            rule_based_levels = [stress_service.determine_stress_level(s) for s in rule_based_scores]
            hybrid_levels = predictions['stress_level'].values
            level_agreement = np.mean([rl == hl for rl, hl in zip(rule_based_levels, hybrid_levels)])
            
            metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'correlation': float(correlation),
                'level_agreement': float(level_agreement)
            }
            
            print(f"   MAE vs Rule-based: {mae:.2f}")
            print(f"   RMSE vs Rule-based: {rmse:.2f}")
            print(f"   Correlation: {correlation:.3f}")
            print(f"   Level Agreement: {level_agreement:.1%}")
            
            return metrics
        except Exception as e:
            print(f"   ⚠️  Evaluation failed: {e}")
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'correlation': 0.0,
                'level_agreement': 0.0
            }
    
    def train_both(self, data: pd.DataFrame, 
                  region_type: RegionType = RegionType.MEDITERRANEAN,
                  supervised_model_name: str = 'xgb',
                  n_clusters: int = 5,
                  test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train both supervised and hybrid models for comparison
        
        Args:
            data: Complete dataset with features and stress_score column
            region_type: Algeria region type
            supervised_model_name: ML algorithm for supervised model
            n_clusters: Number of clusters for hybrid model
            test_size: Test set proportion for supervised model
        
        Returns:
            Combined training results
        """
        print("\n" + "=" * 60)
        print("🚀 TRAINING BOTH MODELS FOR COMPARISON")
        print("=" * 60)
        
        results = {
            'hybrid': None,
            'supervised': None,
            'comparison': None
        }
        
        # 1. Train Hybrid Model (doesn't need stress_score labels)
        print("\n1️⃣  Training Hybrid Model (Unsupervised)...")
        try:
            hybrid_result = self.train_hybrid(data, region_type, n_clusters)
            results['hybrid'] = hybrid_result
        except Exception as e:
            print(f"   ❌ Hybrid training failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Train Supervised Model (needs stress_score labels)
        if 'stress_score' in data.columns:
            print("\n2️⃣  Training Supervised Model...")
            try:
                # Prepare features
                feature_engineer = FeatureEngineer()
                features_df = feature_engineer.create_features_from_training_data(data)
                X, y = feature_engineer.prepare_for_training(features_df, target_type='score')
                
                supervised_result = self.train_supervised(X, y, supervised_model_name, test_size)
                results['supervised'] = supervised_result
            except Exception as e:
                print(f"   ❌ Supervised training failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️  Skipping supervised training (no stress_score labels)")
        
        # 3. Compare results
        print("\n" + "=" * 60)
        print("📊 MODEL COMPARISON")
        print("=" * 60)
        
        comparison = self._compare_models(results['hybrid'], results['supervised'])
        results['comparison'] = comparison
        
        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"{self.output_dir}/training_comparison_{timestamp}.json"
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {results_path}")
        except Exception as e:
            print(f"   ⚠️  Could not save results: {e}")
        
        return results
    
    def _compare_models(self, hybrid_result: Optional[Dict], 
                       supervised_result: Optional[Dict]) -> Dict:
        """Compare performance of hybrid and supervised models"""
        comparison = {
            'hybrid_available': hybrid_result is not None,
            'supervised_available': supervised_result is not None
        }
        
        if supervised_result and hybrid_result:
            print("\n🥊 Hybrid vs Supervised:")
            
            hybrid_mae = hybrid_result.get('metrics', {}).get('mae', 999)
            supervised_mae = supervised_result.get('metrics', {}).get('mae', 999)
            
            print(f"   Hybrid MAE:         {hybrid_mae:.2f}")
            print(f"   Supervised MAE:     {supervised_mae:.2f}")
            
            hybrid_rmse = hybrid_result.get('metrics', {}).get('rmse', 999)
            supervised_rmse = supervised_result.get('metrics', {}).get('rmse', 999)
            
            print(f"   Hybrid RMSE:        {hybrid_rmse:.2f}")
            print(f"   Supervised RMSE:    {supervised_rmse:.2f}")
            
            # Determine winner
            if supervised_mae < hybrid_mae:
                comparison['recommended'] = 'supervised'
                print(f"\n   ✅ Recommendation: Use Supervised Model")
            else:
                comparison['recommended'] = 'hybrid'
                print(f"\n   ✅ Recommendation: Use Hybrid Model")
        elif hybrid_result:
            comparison['recommended'] = 'hybrid'
            print("\n   ✅ Recommendation: Use Hybrid Model (only option)")
        elif supervised_result:
            comparison['recommended'] = 'supervised'
            print("\n   ✅ Recommendation: Use Supervised Model (only option)")
        else:
            comparison['recommended'] = 'none'
            print("\n   ❌ No models successfully trained")
        
        return comparison
    
    def quick_train_hybrid(self, db_session, wilaya_codes: List[int] = None,
                          region_type: RegionType = RegionType.MEDITERRANEAN,
                          n_clusters: int = 5) -> Dict:
        """
        Quick training of hybrid model directly from database
        
        Args:
            db_session: SQLAlchemy database session
            wilaya_codes: List of wilaya codes to include (None = all)
            region_type: Algeria region type
            n_clusters: Number of clusters
        
        Returns:
            Training results
        """
        print("🔄 Loading training data from database...")
        
        # Load data
        feature_engineer = FeatureEngineer()
        data = feature_engineer.load_training_data(db_session, wilaya_codes)
        
        if data.empty:
            raise ValueError("No training data found in database")
        
        print(f"   Loaded {len(data)} records")
        
        # Train hybrid model
        return self.train_hybrid(data, region_type, n_clusters)
    
    def optimize_clusters(self, data: pd.DataFrame, 
                         region_type: RegionType = RegionType.MEDITERRANEAN,
                         min_clusters: int = 3,
                         max_clusters: int = 8) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method
        
        Args:
            data: Training data
            region_type: Algeria region type
            min_clusters: Minimum clusters to try
            max_clusters: Maximum clusters to try
        
        Returns:
            Results for each cluster count
        """
        print(f"\n🔍 Finding optimal number of clusters ({min_clusters}-{max_clusters})...")
        
        results = {}
        
        for n in range(min_clusters, max_clusters + 1):
            print(f"\n   Testing {n} clusters...")
            
            model = HybridStressPredictor(region_type=region_type, n_clusters=n)
            model.fit(data)
            
            # Evaluate
            metrics = self._evaluate_hybrid_model_for_optimization(model, data)
            
            results[n] = {
                'n_clusters': n,
                'metrics': metrics,
                'cluster_summary': model.get_cluster_summary().to_dict('records')
            }
            
            print(f"      Inertia: {metrics['inertia']:.2f}")
            print(f"      Silhouette: {metrics['silhouette']:.3f}")
        
        # Find elbow
        inertias = [results[n]['metrics']['inertia'] for n in range(min_clusters, max_clusters + 1)]
        optimal_n = self._find_elbow(inertias) + min_clusters
        
        print(f"\n✅ Optimal number of clusters: {optimal_n}")
        
        return {
            'optimal_n_clusters': optimal_n,
            'all_results': results
        }
    
    def _evaluate_hybrid_model_for_optimization(self, model: HybridStressPredictor, 
                                               data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model for cluster optimization"""
        from sklearn.metrics import silhouette_score
        
        # Prepare features
        features, _ = model.prepare_features(data)
        features_imputed = model.imputer.transform(features)
        features_scaled = model.scaler.transform(features_imputed)
        
        # Get predictions
        labels = model.kmeans.predict(features_scaled)
        
        # Calculate metrics
        inertia = model.kmeans.inertia_
        silhouette = silhouette_score(features_scaled, labels)
        
        return {
            'inertia': float(inertia),
            'silhouette': float(silhouette)
        }
    
    def _find_elbow(self, values: List[float]) -> int:
        """Find elbow point in curve"""
        # Simple elbow detection: maximum second derivative
        if len(values) < 3:
            return 0
        
        second_derivatives = []
        for i in range(1, len(values) - 1):
            second_deriv = values[i-1] - 2*values[i] + values[i+1]
            second_derivatives.append(abs(second_deriv))
        
        return np.argmax(second_derivatives) + 1