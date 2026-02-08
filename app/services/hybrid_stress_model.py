"""
Hybrid Water Stress Prediction System
Combines unsupervised clustering with domain knowledge for robust stress prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from datetime import datetime

from .stress_analysis import StressAnalysisService, RegionType


class HybridStressPredictor:
    """
    Hybrid approach combining:
    - Unsupervised clustering to discover natural patterns
    - Domain knowledge (stress calculator) to interpret clusters
    - No labeled training data required
    
    This is the RECOMMENDED approach for your use case
    """
    
    def __init__(self, region_type: RegionType = RegionType.MEDITERRANEAN, n_clusters: int = 5):
        self.region_type = region_type
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.stress_calculator = StressAnalysisService(region_type)
        self.cluster_profiles = {}
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract and prepare features from data"""
        # Define feature columns in order of importance
        feature_cols = ['ndvi', 'temperature', 'precipitation', 
                       'evapotranspiration', 'humidity']
        
        # Also check for alternative names
        alt_names = {
            'temperature': 'temperature_avg',
            'precipitation': 'total_precipitation',
            'humidity': 'avg_humidity'
        }
        
        # Build feature dataframe
        features_list = []
        actual_feature_names = []
        
        for col in feature_cols:
            if col in data.columns:
                features_list.append(data[col].values)
                actual_feature_names.append(col)
            elif col in alt_names and alt_names[col] in data.columns:
                features_list.append(data[alt_names[col]].values)
                actual_feature_names.append(col)
            else:
                # Create placeholder column filled with median if feature missing
                features_list.append(np.full(len(data), np.nan))
                actual_feature_names.append(col)
        
        if not features_list:
            raise ValueError("No valid features found in data")
        
        features = np.column_stack(features_list)
        return features, actual_feature_names
    
    def fit(self, data: pd.DataFrame):
        """
        Train clustering model and assign stress levels to clusters
        based on their centroid characteristics using domain knowledge
        
        Args:
            data: DataFrame with columns: ndvi, temperature, precipitation, evapotranspiration, humidity
        """
        print(f"🔄 Training Hybrid Stress Predictor ({self.region_type.value})...")
        
        features, feature_names = self.prepare_features(data)
        self.feature_names = feature_names
        
        print(f"   Features: {', '.join(feature_names)}")
        print(f"   Samples: {len(features)}")
        
        # Handle missing values
        features_imputed = self.imputer.fit_transform(features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features_imputed)
        
        # Perform clustering
        print(f"   Clustering into {self.n_clusters} groups...")
        self.kmeans.fit(features_scaled)
        
        # Analyze clusters and assign stress levels using domain knowledge
        self._assign_cluster_stress_levels(features_imputed)
        
        self.is_trained = True
        
        print(f"✅ Training complete!")
        print(f"\n📊 Cluster Summary:")
        for cluster_id, profile in self.cluster_profiles.items():
            print(f"   Cluster {cluster_id}: {profile['level']} stress (score: {profile['score']:.1f})")
        
        return self
    
    def _assign_cluster_stress_levels(self, features: np.ndarray):
        """
        Assign stress levels to clusters based on centroid characteristics
        Uses domain knowledge without labeled data
        """
        # Get centroids back to original scale
        centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        for cluster_id in range(self.n_clusters):
            centroid = centroids[cluster_id]
            
            # Build indicators dict from centroid
            indicators = {}
            for i, feature_name in enumerate(self.feature_names):
                indicators[feature_name] = centroid[i]
            
            # Use existing stress calculator to get score and level
            stress_score = self.stress_calculator.calculate_stress_score(indicators)
            stress_level = self.stress_calculator.determine_stress_level(stress_score)
            
            # Get cluster size
            cluster_mask = self.kmeans.labels_ == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            self.cluster_profiles[cluster_id] = {
                'score': stress_score,
                'level': stress_level,
                'centroid': indicators,
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(features) * 100)
            }
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict stress scores and levels for new data
        
        Returns:
            DataFrame with columns: cluster, stress_score, stress_level
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        features, _ = self.prepare_features(data)
        
        # Impute and scale
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Get cluster assignments
        clusters = self.kmeans.predict(features_scaled)
        
        # Map clusters to stress scores and levels
        results = []
        for cluster_id in clusters:
            cluster_info = self.cluster_profiles[cluster_id]
            results.append({
                'cluster': cluster_id,
                'stress_score': cluster_info['score'],
                'stress_level': cluster_info['level']
            })
        
        return pd.DataFrame(results)
    
    def predict_with_confidence(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with confidence scores based on distance to cluster centroid
        
        Returns:
            DataFrame with: cluster, stress_score, stress_level, confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        features, _ = self.prepare_features(data)
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Get cluster assignments and distances
        clusters = self.kmeans.predict(features_scaled)
        distances = self.kmeans.transform(features_scaled)
        
        results = []
        for i, cluster_id in enumerate(clusters):
            cluster_info = self.cluster_profiles[cluster_id]
            
            # Calculate confidence: inverse of distance to centroid (normalized)
            distance_to_centroid = distances[i, cluster_id]
            max_distance = np.max(distances[i])
            confidence = 1.0 - (distance_to_centroid / max_distance) if max_distance > 0 else 1.0
            
            results.append({
                'cluster': cluster_id,
                'stress_score': cluster_info['score'],
                'stress_level': cluster_info['level'],
                'confidence': confidence
            })
        
        return pd.DataFrame(results)
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of all clusters and their characteristics"""
        if not self.is_trained:
            return pd.DataFrame()
        
        summary = []
        for cluster_id, info in self.cluster_profiles.items():
            summary.append({
                'cluster': cluster_id,
                'stress_score': info['score'],
                'stress_level': info['level'],
                'size': info['size'],
                'percentage': info['percentage'],
                **info['centroid']
            })
        
        return pd.DataFrame(summary).sort_values('stress_score', ascending=False)
    
    def save_model(self, filepath: str = "models/hybrid_stress_model.pkl"):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'cluster_profiles': self.cluster_profiles,
            'feature_names': self.feature_names,
            'region_type': self.region_type,
            'n_clusters': self.n_clusters,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Hybrid model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.kmeans = model_data['kmeans']
            self.scaler = model_data['scaler']
            self.imputer = model_data['imputer']
            self.cluster_profiles = model_data['cluster_profiles']
            self.feature_names = model_data['feature_names']
            self.region_type = model_data['region_type']
            self.n_clusters = model_data['n_clusters']
            self.is_trained = model_data['is_trained']
            
            # Reinitialize stress calculator with loaded region type
            self.stress_calculator = StressAnalysisService(self.region_type)
            
            print(f"✅ Loaded hybrid model from {filepath}")
            print(f"   Region: {self.region_type.value}")
            print(f"   Clusters: {self.n_clusters}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise


class ClusterBasedStressPredictor:
    """
    Pure clustering approach without domain knowledge
    Discovers stress patterns purely from data
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_stats = {}
        self.feature_names = None
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract features"""
        feature_cols = ['ndvi', 'temperature', 'precipitation', 
                       'evapotranspiration', 'humidity']
        
        features_list = []
        actual_names = []
        
        for col in feature_cols:
            if col in data.columns:
                features_list.append(data[col].values)
                actual_names.append(col)
        
        if not features_list:
            raise ValueError("No valid features found")
        
        return np.column_stack(features_list), actual_names
    
    def fit(self, data: pd.DataFrame):
        """Train clustering model"""
        print(f"🔄 Training Pure Clustering Model...")
        
        features, feature_names = self.prepare_features(data)
        self.feature_names = feature_names
        
        # Impute and scale
        features_imputed = self.imputer.fit_transform(features)
        features_scaled = self.scaler.fit_transform(features_imputed)
        
        # Cluster
        self.kmeans.fit(features_scaled)
        
        # Calculate cluster statistics
        self._calculate_cluster_stats(features_imputed, self.kmeans.labels_)
        
        self.is_trained = True
        print(f"✅ Clustering complete with {self.n_clusters} clusters")
        
        return self
    
    def _calculate_cluster_stats(self, features: np.ndarray, labels: np.ndarray):
        """Calculate statistics for each cluster"""
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_features = features[mask]
            
            stats = {}
            for i, feature_name in enumerate(self.feature_names):
                stats[f'{feature_name}_mean'] = float(np.mean(cluster_features[:, i]))
                stats[f'{feature_name}_std'] = float(np.std(cluster_features[:, i]))
            
            stats['size'] = int(np.sum(mask))
            stats['percentage'] = float(np.sum(mask) / len(features) * 100)
            
            self.cluster_stats[cluster_id] = stats
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster assignments"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        features, _ = self.prepare_features(data)
        features_imputed = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features_imputed)
        
        return self.kmeans.predict(features_scaled)
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get cluster statistics"""
        if not self.is_trained:
            return pd.DataFrame()
        
        summary = []
        for cluster_id, stats in self.cluster_stats.items():
            summary.append({'cluster': cluster_id, **stats})
        
        return pd.DataFrame(summary)


# Utility function to choose the best model for the task
def create_stress_predictor(approach: str = 'hybrid', 
                            region_type: RegionType = RegionType.MEDITERRANEAN,
                            n_clusters: int = 5):
    """
    Factory function to create the appropriate stress predictor
    
    Args:
        approach: 'hybrid' (recommended), 'clustering', or 'supervised'
        region_type: Region type for Algeria
        n_clusters: Number of clusters for unsupervised approaches
    
    Returns:
        Appropriate predictor instance
    """
    if approach == 'hybrid':
        return HybridStressPredictor(region_type=region_type, n_clusters=n_clusters)
    elif approach == 'clustering':
        return ClusterBasedStressPredictor(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'hybrid' or 'clustering'")
