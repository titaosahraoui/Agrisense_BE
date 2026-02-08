import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from .. import models

class FeatureEngineer:
    """
    Enhanced Feature Engineering with:
    - Better data loading with debugging
    - Advanced temporal features
    - Climate indices
    - Spatial normalization
    - Interaction terms
    """
    
    def __init__(self):
        self.season_mapping = {'hiver': 0, 'printemps': 1, 'été': 2, 'automne': 3}
        
    def load_training_data(self, db: Session, wilaya_codes: List[int] = None) -> pd.DataFrame:
        """
        Load training data from database with enhanced debugging
        
        Args:
            db: Database session
            wilaya_codes: Optional list of wilaya codes to filter
        
        Returns:
            DataFrame with training data
        """
        try:
            print(f"📊 Loading training data...")
            
            # Build query
            query = db.query(models.TrainingData)
            
            if wilaya_codes:
                query = query.filter(models.TrainingData.wilaya_code.in_(wilaya_codes))
                print(f"   Filtering by wilayas: {wilaya_codes}")
            
            # Get count first for debugging
            total_count = query.count()
            print(f"   Found {total_count} records in database")
            
            if total_count == 0:
                print("   ⚠️  No records found!")
                print("   Checking if table exists and has data...")
                
                # Debug: Check total records without filter
                all_records = db.query(models.TrainingData).count()
                print(f"   Total records in TrainingData table: {all_records}")
                
                if all_records == 0:
                    print("   ❌ TrainingData table is empty!")
                    return pd.DataFrame()
                
                # If filtered query is empty, return all data
                if wilaya_codes:
                    print(f"   ⚠️  No records for wilayas {wilaya_codes}, loading all data instead")
                    query = db.query(models.TrainingData)
            
            # Order by wilaya and date
            data = query.order_by(
                models.TrainingData.wilaya_code, 
                models.TrainingData.date
            ).all()
            
            print(f"   Retrieved {len(data)} records")
            
            # Convert to DataFrame
            records = []
            for record in data:
                records.append({
                    'id': record.id,
                    'wilaya_code': record.wilaya_code,
                    'date': record.date,
                    'temperature_avg': record.temperature_avg,
                    'temperature_max': record.temperature_max,
                    'temperature_min': record.temperature_min,
                    'precipitation': record.precipitation,
                    'humidity': record.humidity,
                    'solar_radiation': record.solar_radiation,
                    'wind_speed': record.wind_speed,
                    'evapotranspiration': record.evapotranspiration,
                    'ndvi': record.ndvi,
                    'ndwi': record.ndwi,
                    'lst': record.lst,
                    'month': record.month,
                    'season': record.season,
                    'day_of_year': record.day_of_year,
                    'stress_score': record.stress_score,
                    'stress_level': record.stress_level
                })
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                print(f"   ✅ Loaded {len(df)} records")
                print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"   Wilayas: {sorted(df['wilaya_code'].unique().tolist())}")
                print(f"   Columns: {df.columns.tolist()}")
            else:
                print("   ❌ DataFrame is empty after conversion!")
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_features_from_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features for ML with advanced engineering
        
        Features added:
        - Cyclical time encoding
        - Lag features
        - Rolling statistics
        - Climate indices
        - Interaction terms
        """
        print("🔧 Creating enhanced features...")
        
        if df.empty:
            print("   ⚠️  Empty dataframe, returning as-is")
            return df
        
        features_df = df.copy()
        
        # 1. Cyclical temporal features
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        if 'day_of_year' in features_df.columns:
            features_df['doy_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
            features_df['doy_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        
        # 2. Season encoding
        features_df['season_encoded'] = features_df['season'].map(self.season_mapping)
        
        # 3. Climate indices
        features_df = self._add_climate_indices(features_df)
        
        # 4. Interaction features
        features_df = self._add_interaction_features(features_df)
        
        # 5. Per-wilaya features (lag, rolling stats)
        all_features = []
        
        for wilaya_code in features_df['wilaya_code'].unique():
            print(f"   Processing wilaya {wilaya_code}...")
            wilaya_df = features_df[features_df['wilaya_code'] == wilaya_code].copy()
            wilaya_df = wilaya_df.sort_values('date')
            
            # Lag features
            for lag in [1, 2, 3, 7, 14]:
                wilaya_df[f'precip_lag_{lag}'] = wilaya_df['precipitation'].shift(lag)
                if 'ndvi' in wilaya_df.columns:
                    wilaya_df[f'ndvi_lag_{lag}'] = wilaya_df['ndvi'].shift(lag)
                wilaya_df[f'temp_lag_{lag}'] = wilaya_df['temperature_avg'].shift(lag)
            
            # Rolling statistics
            for window in [3, 7, 14, 30]:
                # Precipitation
                wilaya_df[f'precip_sum_{window}d'] = wilaya_df['precipitation'].rolling(
                    window=window, min_periods=1
                ).sum()
                wilaya_df[f'precip_mean_{window}d'] = wilaya_df['precipitation'].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Temperature
                wilaya_df[f'temp_mean_{window}d'] = wilaya_df['temperature_avg'].rolling(
                    window=window, min_periods=1
                ).mean()
                wilaya_df[f'temp_max_{window}d'] = wilaya_df['temperature_avg'].rolling(
                    window=window, min_periods=1
                ).max()
                
                # NDVI if available
                if 'ndvi' in wilaya_df.columns:
                    wilaya_df[f'ndvi_mean_{window}d'] = wilaya_df['ndvi'].rolling(
                        window=window, min_periods=1
                    ).mean()
            
            # Trend features
            if len(wilaya_df) > 7:
                wilaya_df['temp_trend_7d'] = wilaya_df['temperature_avg'].rolling(7).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 7 else 0
                )
                wilaya_df['precip_trend_7d'] = wilaya_df['precipitation'].rolling(7).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 7 else 0
                )
            
            all_features.append(wilaya_df)
        
        # Combine all wilayas
        if all_features:
            features_df = pd.concat(all_features, ignore_index=True)
        
        # Clean data
        features_df = self._clean_data(features_df)
        
        print(f"✅ Features created: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
        return features_df
    
    def _add_climate_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add climate-related indices"""
        
        # Aridity index (Precipitation / Evapotranspiration)
        if 'precipitation' in df.columns and 'evapotranspiration' in df.columns:
            df['aridity_index'] = df['precipitation'] / (df['evapotranspiration'] + 0.1)
        
        # Water deficit
        if 'precipitation' in df.columns and 'evapotranspiration' in df.columns:
            df['water_deficit'] = df['evapotranspiration'] - df['precipitation']
            df['water_surplus'] = (df['precipitation'] - df['evapotranspiration']).clip(lower=0)
        
        # Temperature range
        if 'temperature_max' in df.columns and 'temperature_min' in df.columns:
            df['temp_range'] = df['temperature_max'] - df['temperature_min']
        
        # Heat stress indicator
        if 'temperature_avg' in df.columns:
            df['heat_stress'] = (df['temperature_avg'] - 25).clip(lower=0)  # Stress above 25°C
        
        # Moisture index (combines humidity and precipitation)
        if 'humidity' in df.columns and 'precipitation' in df.columns:
            df['moisture_index'] = (df['humidity'] / 100) * np.log1p(df['precipitation'])
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction terms between key features"""
        
        # NDVI * Temperature (vegetation under heat stress)
        if 'ndvi' in df.columns and 'temperature_avg' in df.columns:
            df['ndvi_temp_interact'] = df['ndvi'] * df['temperature_avg']
        
        # Precipitation * Temperature (evaporation pressure)
        if 'precipitation' in df.columns and 'temperature_avg' in df.columns:
            df['precip_temp_interact'] = df['precipitation'] * (30 - df['temperature_avg'])
        
        # ET / Precipitation ratio
        if 'evapotranspiration' in df.columns and 'precipitation' in df.columns:
            df['et_precip_ratio'] = df['evapotranspiration'] / (df['precipitation'] + 0.1)
        
        # Humidity * Precipitation (moisture availability)
        if 'humidity' in df.columns and 'precipitation' in df.columns:
            df['humidity_precip'] = (df['humidity'] / 100) * df['precipitation']
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data: handle inf, fill missing values"""
        
        # Replace inf with nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def prepare_for_training(self, features_df: pd.DataFrame, 
                           target_type: str = 'score') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X and y for training
        
        Args:
            features_df: DataFrame with all features
            target_type: 'score' for regression, 'level' for classification
        
        Returns:
            X, y tuple
        """
        print(f"📊 Preparing data for training (target: {target_type})...")
        
        # Columns to exclude from features
        exclude = ['id', 'date', 'stress_level', 'stress_score', 'created_at']
        
        # Determine target
        if target_type == 'score':
            target_col = 'stress_score'
        else:
            target_col = 'stress_level'
        
        # Check target exists
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data!")
        
        # Drop excluded columns that exist
        X = features_df.drop(columns=[c for c in exclude if c in features_df.columns])
        
        # Get target
        y = features_df[target_col]
        
        # Drop rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Features: {X.columns.tolist()[:10]}... ({len(X.columns)} total)")
        
        if len(X) == 0:
            raise ValueError("No valid training samples after removing missing targets!")
        
        return X, y
