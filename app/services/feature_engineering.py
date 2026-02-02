import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from .. import models

class FeatureEngineer:
    def __init__(self):
        self.season_mapping = {'hiver': 0, 'printemps': 1, 'été': 2, 'automne': 3}
        
    def load_training_data(self, db: Session, wilaya_codes: List[int] = None) -> pd.DataFrame:
        """Charger les données depuis la base"""

        
        query = db.query(models.TrainingData)
        
        if wilaya_codes:
            query = query.filter(models.TrainingData.wilaya_code.in_(wilaya_codes))
        
        data = query.order_by(models.TrainingData.wilaya_code, 
                             models.TrainingData.date).all()
        
        records = []
        for record in data:
            records.append({
                'wilaya_code': record.wilaya_code,
                'date': record.date,
                'temperature_avg': record.temperature_avg,
                'precipitation': record.precipitation,
                'humidity': record.humidity,
                'solar_radiation': record.solar_radiation,
                'evapotranspiration': record.evapotranspiration,
                'ndvi': record.ndvi,
                'ndwi': record.ndwi,
                'lst': record.lst,
                'month': record.month,
                'season': record.season,
                'stress_score': record.stress_score,
                'stress_level': record.stress_level
            })
        
        return pd.DataFrame(records)
    
    def create_features_from_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer les features pour le ML"""
        print("🔧 Création des features...")
        
        if df.empty:
            return df
        
        features_df = df.copy()
        
        # 1. Features temporelles
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # 2. Encodage saison
        features_df['season_encoded'] = features_df['season'].map(self.season_mapping)
        
        # 3. Pour chaque wilaya
        all_features = []
        
        for wilaya_code in features_df['wilaya_code'].unique():
            wilaya_df = features_df[features_df['wilaya_code'] == wilaya_code].copy()
            wilaya_df = wilaya_df.sort_values('date')
            
            # Lag features
            for lag in [1, 2, 3]:
                wilaya_df[f'precip_lag_{lag}'] = wilaya_df['precipitation'].shift(lag)
                wilaya_df[f'ndvi_lag_{lag}'] = wilaya_df['ndvi'].shift(lag)
                wilaya_df[f'temp_lag_{lag}'] = wilaya_df['temperature_avg'].shift(lag)
            
            # Moyennes mobiles
            for window in [3, 6]:
                wilaya_df[f'precip_ma_{window}'] = wilaya_df['precipitation'].rolling(
                    window=window, min_periods=1
                ).mean()
                wilaya_df[f'temp_ma_{window}'] = wilaya_df['temperature_avg'].rolling(
                    window=window, min_periods=1
                ).mean()
            
            # Cumul pluie
            wilaya_df['precip_cumul_3m'] = wilaya_df['precipitation'].rolling(
                window=3, min_periods=1
            ).sum()
            wilaya_df['precip_cumul_6m'] = wilaya_df['precipitation'].rolling(
                window=6, min_periods=1
            ).sum()
            
            # Ratios
            wilaya_df['et_precip_ratio'] = wilaya_df['evapotranspiration'] / (
                wilaya_df['precipitation'] + 0.1
            )
            
            all_features.append(wilaya_df)
        
        # Combiner
        if all_features:
            features_df = pd.concat(all_features, ignore_index=True)
        
        # Nettoyage
        features_df = self._clean_data(features_df)
        
        print(f"✅ Features créées: {features_df.shape[1]} colonnes")
        return features_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyer les données"""
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def prepare_for_training(self, features_df: pd.DataFrame, 
                           target_type: str = 'score') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Préparer X et y pour l'entraînement
        CORRECTION : Retourne seulement X et y selon target_type
        """
        # Colonnes à exclure
        exclude = ['wilaya_code', 'date', 'season', 'stress_level', 'stress_score']
        
        if target_type == 'score':
            target_col = 'stress_score'
            exclude.append('stress_level')
        else:
            target_col = 'stress_level'
            exclude.append('stress_score')
        
        # Features (X)
        X = features_df.drop(columns=[c for c in exclude if c in features_df.columns])
        
        # Target (y)
        y = features_df[target_col]
        
        print(f"📊 {target_type}: X shape: {X.shape}, y shape: {y.shape}")
        return X, y  # SEULEMENT 2 VALEURS DE RETOUR