# services/data_cleaner.py
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

class DataCleaner:
    def __init__(self, db: Session):
        self.db = db
    
    def load_training_data(self, wilaya_codes: Optional[List[int]] = None) -> pd.DataFrame:
        """Load training data from database"""
        from .. import models
        
        query = self.db.query(models.TrainingData)
        
        if wilaya_codes:
            query = query.filter(models.TrainingData.wilaya_code.in_(wilaya_codes))
        
        # Convert to pandas DataFrame
        records = []
        for record in query.all():
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
        
        if df.empty:
            print("⚠️ No data found")
            return df
        
        print(f"✅ Loaded {len(df)} records")
        return df
    
    def convert_numpy_types(self, obj: Any) -> Any:
        """Recursively convert NumPy types to Python native types"""
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self.convert_numpy_types(obj.tolist())
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif isinstance(obj, pd.Period):
            return str(obj)
        else:
            return obj
    
    def analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns"""
        # First convert all numpy types in the dataframe
        df = df.copy()
        for col in df.columns:
            if df[col].dtype in [np.int64, np.int32, np.float64, np.float32]:
                df[col] = df[col].astype(object).where(df[col].notnull(), None)
        
        missing_stats = {}
        
        # Total missing values
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        # Missing by column
        missing_by_column = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_by_column[column] = {
                'count': int(missing_count),
                'percentage': round(float(missing_percentage), 2)
            }
        
        # Missing by wilaya
        missing_by_wilaya = {}
        if 'wilaya_code' in df.columns:
            for wilaya in df['wilaya_code'].unique():
                wilaya_data = df[df['wilaya_code'] == wilaya]
                missing_count = wilaya_data.isnull().sum().sum()
                missing_by_wilaya[int(wilaya)] = {
                    'count': int(missing_count),
                    'percentage': round(float((missing_count / wilaya_data.size) * 100), 2)
                }
        
        # Time-based missing patterns
        time_missing = {}
        if 'date' in df.columns:
            df['year_month'] = df['date'].dt.to_period('M')
            
            # Fix for pandas deprecation warning
            missing_by_month = df.groupby('year_month').apply(
                lambda x: x.drop(columns=['year_month']).isnull().sum().sum() / x.drop(columns=['year_month']).size * 100
            )
            
            # Convert Period index to string
            time_missing['by_month'] = {
                str(period): float(value) 
                for period, value in missing_by_month.items()
            }
        
        result = {
            'summary': {
                'total_cells': int(total_cells),
                'missing_cells': int(missing_cells),
                'missing_percentage': round(float((missing_cells / total_cells) * 100), 2)
            },
            'by_column': self.convert_numpy_types(missing_by_column),
            'by_wilaya': self.convert_numpy_types(missing_by_wilaya),
            'time_patterns': self.convert_numpy_types(time_missing)
        }
        
        return result
    
    
    def clean_daily_validation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean daily collected data without normalization or encoding.
        Focuses on filling missing values and ensuring data integrity.
        """
        df_clean = df.copy()
        print("🧹 Running Specialized Daily Data Cleaning...")
        
        # 1. Satellite Data Interpolation
        satellite_cols = ['ndvi', 'ndwi', 'lst']
        print("  🛰️ Applying time interpolation for satellite data...")
        
        if 'date' in df_clean.columns and df_clean['date'].dtype == 'object':
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            
        if 'wilaya_code' in df_clean.columns and 'date' in df_clean.columns:
            df_clean = df_clean.sort_values(['wilaya_code', 'date'])
            for col in satellite_cols:
                if col in df_clean.columns:
                    current_missing = df_clean[col].isnull().sum()
                    if current_missing > 0:
                        df_clean[col] = df_clean.groupby('wilaya_code')[col].transform(
                            lambda x: x.interpolate(method='linear', limit_direction='both')
                        )
                        print(f"    - {col}: filled values via interpolation")

        # 2. KNN Imputation for remaining numeric nulls
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['id', 'wilaya_code', 'day_of_year', 'month', 'year']
        impute_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if impute_cols and df_clean[impute_cols].isnull().values.any():
            print(f"  Unknown values detected, running KNN Imputation on {len(impute_cols)} columns...")
            imputer = KNNImputer(n_neighbors=5)
            
            non_imputed = df_clean.drop(columns=impute_cols)
            to_impute = df_clean[impute_cols]
            
            imputed_array = imputer.fit_transform(to_impute)
            df_imputed = pd.DataFrame(imputed_array, columns=impute_cols, index=df_clean.index)
            
            df_clean = pd.concat([non_imputed, df_imputed], axis=1)
        

        # 4. Type Conversion (No Normalization)
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].dtype in [np.float64, np.float32]:
                df_clean[col] = df_clean[col].astype(float)
            elif df_clean[col].dtype in [np.int64, np.int32]:
                df_clean[col] = df_clean[col].astype(int)
        
        print("✅ Daily Data Cleaning Complete.")
        return df_clean

    def clean_data(self, df: pd.DataFrame, strategy: str = "advanced") -> pd.DataFrame:
        """
        Clean data using various strategies
        
        Strategies:
        - "simple": Fill with mean/median
        - "advanced": Use KNN imputation
        - "time_series": Use forward/backward fill for time series
        """
        df_clean = df.copy()
        
        print(f"🧹 Cleaning data with {strategy} strategy...")
        
        # 1. SPECIALIZED CLEANING: Satellite Data Interpolation
        # Satellite data follows trends, so we interpolate over time per Wilaya
        satellite_cols = ['ndvi', 'ndwi', 'lst']
        print("  🛰️ Applying time interpolation for satellite data...")
        
        # Ensure date is datetime
        if 'date' in df_clean.columns and df_clean['date'].dtype == 'object':
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            
        if 'wilaya_code' in df_clean.columns and 'date' in df_clean.columns:
            # Sort by wilaya and date to ensure correct interpolation
            df_clean = df_clean.sort_values(['wilaya_code', 'date'])
            
            for col in satellite_cols:
                if col in df_clean.columns:
                    # Group by wilaya and interpolate
                    current_missing = df_clean[col].isnull().sum()
                    if current_missing > 0:
                        df_clean[col] = df_clean.groupby('wilaya_code')[col].transform(
                            lambda x: x.interpolate(method='linear', limit_direction='both')
                        )
                        new_missing = df_clean[col].isnull().sum()
                        print(f"    - {col}: filled {current_missing - new_missing} values via interpolation")

        if strategy == "simple":
            # Fill numeric columns with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    print(f"  Filled {col} with median: {median_val}")
        
        elif strategy == "advanced":
            # Use KNN imputation for better accuracy
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove ID columns from imputation
            exclude_cols = ['id', 'wilaya_code', 'day_of_year', 'month']
            impute_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if impute_cols:
                # Create imputer
                imputer = KNNImputer(n_neighbors=5)
                
                # Store non-numeric columns
                non_numeric_data = df_clean.drop(columns=impute_cols)
                numeric_data = df_clean[impute_cols]
                
                # Impute missing values
                imputed_array = imputer.fit_transform(numeric_data)
                df_imputed = pd.DataFrame(imputed_array, columns=impute_cols)
                
                # Convert numpy types to python types
                for col in df_imputed.columns:
                    if df_imputed[col].dtype in [np.float64, np.float32]:
                        df_imputed[col] = df_imputed[col].astype(float)
                    elif df_imputed[col].dtype in [np.int64, np.int32]:
                        df_imputed[col] = df_imputed[col].astype(int)
                
                # Combine back
                df_clean = pd.concat([non_numeric_data.reset_index(drop=True), 
                                    df_imputed.reset_index(drop=True)], axis=1)
        
        elif strategy == "time_series":
            # Sort by date
            if 'date' in df_clean.columns:
                df_clean = df_clean.sort_values('date')
            
            # Group by wilaya and fill with forward/backward fill
            if 'wilaya_code' in df_clean.columns:
                grouped = df_clean.groupby('wilaya_code')
                df_clean = grouped.apply(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                df_clean = df_clean.reset_index(drop=True)
        
        # Handle categorical columns
        if 'season' in df_clean.columns:
            df_clean['season'] = df_clean['season'].fillna(df_clean['season'].mode()[0] 
                                                         if not df_clean['season'].mode().empty 
                                                         else 'printemps')
        
        if 'stress_level' in df_clean.columns:
            df_clean['stress_level'] = df_clean['stress_level'].fillna('moderate')
        
        print(f"✅ Cleaning complete. Missing values after cleaning:")
        missing_after = df_clean.isnull().sum()
        print(missing_after[missing_after > 0])
        
        # Convert numpy types to python types
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].dtype in [np.float64, np.float32]:
                df_clean[col] = df_clean[col].astype(float)
            elif df_clean[col].dtype in [np.int64, np.int32]:
                df_clean[col] = df_clean[col].astype(int)
        
        return df_clean
    
    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for better model performance"""
        df_engineered = df.copy()
        
        # Convert numpy types first
        for col in df_engineered.select_dtypes(include=[np.number]).columns:
            if df_engineered[col].dtype in [np.float64, np.float32]:
                df_engineered[col] = df_engineered[col].astype(float)
            elif df_engineered[col].dtype in [np.int64, np.int32]:
                df_engineered[col] = df_engineered[col].astype(int)
        
        # Lag features for time series
        lag_cols = ['temperature_avg', 'precipitation', 'ndvi', 'humidity']
        for col in lag_cols:
            if col in df_engineered.columns:
                df_engineered[f'{col}_lag7'] = df_engineered.groupby('wilaya_code')[col].shift(7).astype(float)
                df_engineered[f'{col}_lag30'] = df_engineered.groupby('wilaya_code')[col].shift(30).astype(float)
        
        # Rolling averages
        if 'precipitation' in df_engineered.columns:
            precip_cumul = df_engineered.groupby('wilaya_code')['precipitation']\
                .rolling(window=30, min_periods=1).sum().reset_index(level=0, drop=True)
            df_engineered['precip_cumul_30d'] = precip_cumul.astype(float)
        
        if 'temperature_avg' in df_engineered.columns:
            temp_avg = df_engineered.groupby('wilaya_code')['temperature_avg']\
                .rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
            df_engineered['temp_avg_7d'] = temp_avg.astype(float)
        
        # NDVI trend (simplified to avoid complex calculation)
        if 'ndvi' in df_engineered.columns:
            ndvi_trend = df_engineered.groupby('wilaya_code')['ndvi']\
                .rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
            df_engineered['ndvi_trend'] = ndvi_trend.astype(float)
        
        # Climate indices
        if all(col in df_engineered.columns for col in ['precipitation', 'evapotranspiration']):
            df_engineered['aridity_index'] = (df_engineered['precipitation'] / 
                                            (df_engineered['evapotranspiration'] + 1e-6)).astype(float)
        
        # Heat stress index (simplified)
        if all(col in df_engineered.columns for col in ['temperature_avg', 'humidity']):
            df_engineered['heat_index'] = (0.5 * (df_engineered['temperature_avg'] + 61.0 + 
                                                (df_engineered['temperature_avg'] - 68.0) * 1.2 + 
                                                df_engineered['humidity'] * 0.094)).astype(float)
        
        # Convert any new numpy columns to python types
        for col in df_engineered.columns:
            if df_engineered[col].dtype in [np.float64, np.float32]:
                df_engineered[col] = df_engineered[col].astype(float)
            elif df_engineered[col].dtype in [np.int64, np.int32]:
                df_engineered[col] = df_engineered[col].astype(int)
        
        print(f"✅ Added {len(df_engineered.columns) - len(df.columns)} engineered features")
        
        return df_engineered
    
    def save_cleaned_data(self, df: pd.DataFrame, table_name: str = "cleaned_training_data"):
        """Save cleaned data to database or CSV"""
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name}_{timestamp}.csv"
        
        # Ensure all data is JSON serializable
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if df_to_save[col].dtype == 'object':
                # Convert any remaining numpy types in object columns
                df_to_save[col] = df_to_save[col].apply(
                    lambda x: self.convert_numpy_types(x) if isinstance(x, (np.integer, np.floating, np.ndarray)) else x
                )
        
        df_to_save.to_csv(filename, index=False)
        print(f"💾 Saved cleaned data to {filename}")
        
        return filename
    
    def prepare_for_training(self, df: pd.DataFrame) -> tuple:
        """Prepare data for model training"""
        # Select features and target
        feature_cols = [
            'temperature_avg', 'temperature_max', 'temperature_min',
            'precipitation', 'humidity', 'solar_radiation', 'wind_speed',
            'evapotranspiration', 'ndvi', 'ndwi', 'lst',
            'month', 'day_of_year'
        ]
        
        # Only include columns that exist in dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Add engineered features
        engineered_cols = [col for col in df.columns if any(x in col for x in ['lag', 'trend', 'index'])]
        feature_cols.extend(engineered_cols)
        
        # Remove non-numeric columns from features
        feature_cols = [col for col in feature_cols if col in df.select_dtypes(include=[np.number]).columns]
        
        # Target variable
        target_col = 'stress_score'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        # Ensure all data is Python native types
        for col in X.columns:
            if X[col].dtype in [np.float64, np.float32]:
                X[col] = X[col].astype(float)
            elif X[col].dtype in [np.int64, np.int32]:
                X[col] = X[col].astype(int)
        
        if y.dtype in [np.float64, np.float32]:
            y = y.astype(float)
        elif y.dtype in [np.int64, np.int32]:
            y = y.astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"📊 Data prepared for training:")
        print(f"   Features: {len(feature_cols)} columns")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, feature_cols