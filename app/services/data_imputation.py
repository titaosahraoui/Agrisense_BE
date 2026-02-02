
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class DataImputer:
    def __init__(self):
        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=42
        )
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataset
        """
        df_imputed = df.copy()
        
        # 1. For NDVI, NDWI, LST: Use temporal interpolation
        satellite_cols = ['ndvi', 'ndwi', 'lst']
        for col in satellite_cols:
            if col in df.columns:
                # Sort by date for interpolation
                df_imputed = df_imputed.sort_values(['wilaya_code', 'date'])
                
                # Group by wilaya and interpolate within each group
                df_imputed[col] = df_imputed.groupby('wilaya_code')[col].transform(
                    lambda x: x.interpolate(method='linear', limit=2)
                )
        
        # 2. For remaining numeric columns, use MICE imputation
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that shouldn't be imputed
        cols_to_impute = [col for col in numeric_cols 
                         if col not in ['id', 'wilaya_code', 'month', 'day_of_year', 'stress_score']]
        
        if cols_to_impute:
            # Separate features to impute
            data_to_impute = df_imputed[cols_to_impute].values
            
            # Fit and transform using MICE
            imputed_data = self.imputer.fit_transform(data_to_impute)
            
            # Update dataframe
            df_imputed[cols_to_impute] = imputed_data
        
        return df_imputed
    
    def create_monthly_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data to monthly features
        """
        # Ensure date is datetime
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Create month-year identifier
        daily_df['year_month'] = daily_df['date'].dt.to_period('M')
        
        # Group by wilaya and month
        monthly_features = []
        
        for (wilaya_code, year_month), group in daily_df.groupby(['wilaya_code', 'year_month']):
            monthly_record = {
                'wilaya_code': wilaya_code,
                'date': group['date'].iloc[15] if len(group) > 15 else group['date'].iloc[0],  # Mid-month
                'year': group['date'].dt.year.iloc[0],
                'month': group['date'].dt.month.iloc[0],
                
                # Aggregated weather features
                'temperature_avg': group['temperature_avg'].mean(),
                'temperature_max': group['temperature_max'].max(),
                'temperature_min': group['temperature_min'].min(),
                'precipitation_total': group['precipitation'].sum(),
                'humidity_avg': group['humidity'].mean(),
                'solar_radiation_avg': group['solar_radiation'].mean(),
                'wind_speed_avg': group['wind_speed'].mean(),
                'evapotranspiration_total': group['evapotranspiration'].sum(),
                
                # Satellite features (use last available)
                'ndvi': group['ndvi'].iloc[-1] if not group['ndvi'].isna().all() else None,
                'ndwi': group['ndwi'].iloc[-1] if not group['ndwi'].isna().all() else None,
                'lst': group['lst'].iloc[-1] if not group['lst'].isna().all() else None,
                
                # Derived features
                'days_with_precip': (group['precipitation'] > 0.1).sum(),
                'heat_days': (group['temperature_max'] > 30).sum(),
                'dry_days': (group['humidity'] < 30).sum()
            }
            
            monthly_features.append(monthly_record)
        
        return pd.DataFrame(monthly_features)