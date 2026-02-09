import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import Dict, List, Optional, Tuple
import warnings
import os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class DataLoader:
    """Load and preprocess all agricultural and meteorological data"""
    
    # Define constants for better maintainability
    DATA_YEARS = [2016, 2017, 2018, 2019]
    SEASONS = ['été', 'hiver']
    
    # Mapping for consistent column names
    CEREAL_COLUMN_MAPPING = {
        'Ble_Dur_Sup_Recoltee_ha': 'Durum_Wheat_Area',
        'Ble_Tendre_Sup_Recoltee_ha': 'Soft_Wheat_Area',
        'Ble_Dur_Prod_qx': 'Durum_Wheat_Production',
        'Ble_Tendre_Prod_qx': 'Soft_Wheat_Production',
        'Orge_Sup_Recoltee_ha': 'Barley_Area',
        'Orge_Prod_qx': 'Barley_Production',
        'Maïs_Sup_ha': 'Maize_Area',
        'Maïs_Prod_qx': 'Maize_Production',
        'Maïs_Rdt_qx_ha': 'Maize_Yield',
        'Sorgho_Sup_ha': 'Sorghum_Area',
        'Sorgho_Prod_qx': 'Sorghum_Production',
        'Sorgho_Rdt_qx_ha': 'Sorghum_Yield',
        'Total_Cereales_Ete_Sup_ha': 'Total_Summer_Area',
        'Total_Cereales_Ete_Prod_qx': 'Total_Summer_Production',
        'Total_Cereales_Ete_Rdt_qx_ha': 'Total_Summer_Yield',
        'Total_Cereales_Hiver_Sup_Recoltee_ha': 'Total_Winter_Area',
        'Total_Cereales_Hiver_Prod_qx': 'Total_Winter_Production',
        'Total_Cereales_Hiver_Rdt_qx_ha': 'Total_Winter_Yield'
    }
    
    LAND_USE_COLUMN_MAPPING = {
        'Cultures_herbacées_ha': 'Cultivated_Area',
        'Terres_au_repos_ha': 'Fallow_Land',
        'Plantations_arbres_fruit_ha': 'Fruit_Trees',
        'TOTAL_SAU_ha': 'Total_Agricultural_Area'
    }
    
    def __init__(self, db_url: str = None, csv_base_path: str = "../ml_Models/outputs/data"):
        self.db_url = db_url
        self.csv_base_path = Path(csv_base_path)
        
        # Validate paths
        if not self.csv_base_path.exists():
            raise FileNotFoundError(f"CSV base path not found: {csv_base_path}")
    
    def load_meteorological_data(self) -> pd.DataFrame:
        """Load meteorological data from PostgreSQL database"""
        if not self.db_url:
            raise ValueError("Database URL is required for loading meteorological data")
            
        try:
            engine = create_engine(self.db_url)
            
            query = """
            SELECT
                wilaya_code,
                date,
                temperature_avg,
                precipitation,
                humidity,
                ndvi,
                season,
                EXTRACT(YEAR FROM date) AS year
            FROM training_data
            WHERE season IN ('été', 'hiver')
            ORDER BY wilaya_code, date
            """
            
            df = pd.read_sql(query, engine, parse_dates=['date'])
            
            # Validate data
            if df.empty:
                raise ValueError("No meteorological data found in the database")
                
            # Check for missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                print(f"Warning: Missing values found in columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load meteorological data: {str(e)}")
    
    def preprocess_meteo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess meteorological data with better year handling"""
        df = df.copy()
        
        # Filter valid seasons
        valid_seasons = df['season'].isin(self.SEASONS)
        if not valid_seasons.any():
            raise ValueError(f"No valid seasons found. Expected: {self.SEASONS}")
        df = df[valid_seasons].copy()
        
        # Handle year consistency - keep original years but ensure they match agricultural data
        # The agricultural data uses 2016-2019, so we need to map meteorological years accordingly
        # If meteorological data has different years, we can align them logically
        df['original_year'] = df['year']
        
        # Ensure we have data for all required years
        available_years = sorted(df['year'].unique())
        print(f"Meteorological data available for years: {available_years}")
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal aggregated features with robust aggregation"""
        # Define aggregation functions
        agg_funcs = {
            'temperature_avg': ['mean', 'min', 'max', 'std'],
            'precipitation': ['mean', 'sum', 'max'],
            'humidity': ['mean', 'min', 'max'],
            'ndvi': ['mean', 'max', 'min']
        }
        
        seasonal_df = (
            df
            .groupby(['wilaya_code', 'year', 'season'])
            .agg(agg_funcs)
            .round(3)
        )
        
        # Flatten column names
        seasonal_df.columns = ['_'.join(col).strip() for col in seasonal_df.columns.values]
        seasonal_df = seasonal_df.reset_index()
        
        return seasonal_df
    
    def pivot_seasonal_data(self, seasonal_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot seasonal data to wide format with proper naming"""
        # Create mapping for better column names
        value_vars = [
            'temperature_avg_mean', 'temperature_avg_std',
            'precipitation_mean', 'precipitation_sum',
            'humidity_mean', 'ndvi_mean', 'ndvi_max'
        ]
        
        pivot_dfs = []
        for var in value_vars:
            if var in seasonal_df.columns:
                temp_pivot = seasonal_df.pivot_table(
                    index=['wilaya_code', 'year'],
                    columns='season',
                    values=var
                )
                temp_pivot.columns = [f"{var}_{col}" for col in temp_pivot.columns]
                pivot_dfs.append(temp_pivot)
        
        if pivot_dfs:
            pivot_df = pd.concat(pivot_dfs, axis=1).reset_index()
            
            # Fill missing values with column means
            numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
            pivot_df[numeric_cols] = pivot_df[numeric_cols].fillna(pivot_df[numeric_cols].mean())
            
            return pivot_df
        else:
            return pd.DataFrame(columns=['wilaya_code', 'year'])
    
    def load_agricultural_data(self) -> Dict[str, pd.DataFrame]:
        """Load all agricultural CSV files with error handling"""
        data_files = {}
        
        # Define file patterns for better flexibility
        file_patterns = {
            'annual': "annual_{year}.csv",
            'summer': "CEREALES D'ETE {year}.csv",
            'winter': "CEREALES D'HIVER {year}.csv",
            'land_use_2016': "SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE (2016).csv",
            'land_use_2017': "SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE (2017).csv",
            'land_use_2018': "SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE (2018).csv",
            'land_use_2019': "SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE.csv"
        }
        
        # Load annual files
        for year in self.DATA_YEARS:
            for data_type, pattern in [('annual', file_patterns['annual']),
                                      ('summer', file_patterns['summer']),
                                      ('winter', file_patterns['winter'])]:
                file_key = f"{data_type}_{year}"
                file_name = pattern.format(year=year)
                file_path = self.csv_base_path / file_name
                
                try:
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        data_files[file_key] = df
                        print(f"Loaded: {file_name}")
                    else:
                        print(f"Warning: File not found: {file_name}")
                        data_files[file_key] = pd.DataFrame()
                except Exception as e:
                    print(f"Error loading {file_name}: {str(e)}")
                    data_files[file_key] = pd.DataFrame()
        
        # Load land use files
        for key, file_name in [(k, v) for k, v in file_patterns.items() if k.startswith('land_use')]:
            file_path = self.csv_base_path / file_name
            try:
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()
                    data_files[key] = df
                    print(f"Loaded: {file_name}")
                else:
                    print(f"Warning: File not found: {file_name}")
                    data_files[key] = pd.DataFrame()
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")
                data_files[key] = pd.DataFrame()
        
        return data_files
    
    def _standardize_column_names(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Standardize column names using mapping"""
        df = df.copy()
        
        # Rename columns based on mapping
        rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        return df
    
    def _validate_wilaya_data(self, wilaya_code, year, data_dict):
        """Validate data for a specific wilaya and year"""
        try:
            wilaya_int = int(wilaya_code)
            if wilaya_int == 0 or wilaya_int > 58:  # Assuming 58 wilayas in Algeria
                return False
        except (ValueError, TypeError):
            return False
        
        # Check if all required dataframes have data for this wilaya
        required_keys = [f'summer_{year}', f'winter_{year}', f'land_use_{year}']
        
        for key in required_keys:
            if key not in data_dict or data_dict[key].empty:
                return False
            
            df = data_dict[key]
            if 'Wilaya_Code' not in df.columns:
                return False
            
            if wilaya_code not in df['Wilaya_Code'].values:
                return False
        
        return True
    
    def create_master_dataset(self, agricultural_data: Dict, meteo_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Combine agricultural and meteorological data into master dataset"""
        master_records = []
        
        # First, create yearly agricultural summary
        yearly_dfs = []
        for year in self.DATA_YEARS:
            annual_key = f'annual_{year}'
            if annual_key in agricultural_data and not agricultural_data[annual_key].empty:
                df = agricultural_data[annual_key].copy()
                df['Year'] = year
                # Standardize column names
                df = self._standardize_column_names(df, self.CEREAL_COLUMN_MAPPING)
                yearly_dfs.append(df)
        
        # Process each wilaya-year combination
        for year in self.DATA_YEARS:
            for summer_key in [f'summer_{year}']:
                if summer_key not in agricultural_data or agricultural_data[summer_key].empty:
                    continue
                
                summer_df = agricultural_data[summer_key].copy()
                winter_df = agricultural_data.get(f'winter_{year}', pd.DataFrame()).copy()
                land_df = agricultural_data.get(f'land_use_{year}', pd.DataFrame()).copy()
                
                # Standardize column names
                summer_df = self._standardize_column_names(summer_df, self.CEREAL_COLUMN_MAPPING)
                winter_df = self._standardize_column_names(winter_df, self.CEREAL_COLUMN_MAPPING)
                land_df = self._standardize_column_names(land_df, self.LAND_USE_COLUMN_MAPPING)
                
                # Get unique wilayas
                if 'Wilaya_Code' not in summer_df.columns:
                    continue
                
                wilaya_codes = summer_df['Wilaya_Code'].dropna().unique()
                
                for wilaya_code in wilaya_codes:
                    if not self._validate_wilaya_data(wilaya_code, year, agricultural_data):
                        continue
                    
                    try:
                        # Extract data for this wilaya
                        summer_data = summer_df[summer_df['Wilaya_Code'] == wilaya_code].iloc[0]
                        winter_data = winter_df[winter_df['Wilaya_Code'] == wilaya_code].iloc[0] if not winter_df.empty else pd.Series()
                        land_data = land_df[land_df['Wilaya_Code'] == wilaya_code].iloc[0] if not land_df.empty else pd.Series()
                        
                        record = {
                            'Year': year,
                            'Wilaya_Code': int(wilaya_code),
                            'Wilaya_Name': summer_data.get('Wilaya_Name', f'Wilaya_{wilaya_code}'),
                        }
                        
                        # Add summer cereal features with safe access
                        summer_fields = [
                            ('Maize_Area', 'Summer_Maize_Area', 0),
                            ('Maize_Production', 'Summer_Maize_Production', 0),
                            ('Maize_Yield', 'Summer_Maize_Yield', 0),
                            ('Sorghum_Area', 'Summer_Sorghum_Area', 0),
                            ('Sorghum_Production', 'Summer_Sorghum_Production', 0),
                            ('Sorghum_Yield', 'Summer_Sorghum_Yield', 0),
                            ('Total_Summer_Area', 'Total_Summer_Area', 0),
                            ('Total_Summer_Production', 'Total_Summer_Production', 0),
                            ('Total_Summer_Yield', 'Total_Summer_Yield', 0),
                        ]
                        
                        for src, dest, default in summer_fields:
                            record[dest] = summer_data.get(src, default)
                        
                        # Add winter cereal features
                        if not winter_data.empty:
                            winter_fields = [
                                ('Durum_Wheat_Area', 'Winter_Durum_Wheat_Area', 0),
                                ('Soft_Wheat_Area', 'Winter_Soft_Wheat_Area', 0),
                                ('Durum_Wheat_Production', 'Winter_Durum_Wheat_Production', 0),
                                ('Soft_Wheat_Production', 'Winter_Soft_Wheat_Production', 0),
                                ('Barley_Area', 'Winter_Barley_Area', 0),
                                ('Barley_Production', 'Winter_Barley_Production', 0),
                                ('Total_Winter_Area', 'Total_Winter_Area', 0),
                                ('Total_Winter_Production', 'Total_Winter_Production', 0),
                                ('Total_Winter_Yield', 'Total_Winter_Yield', 0),
                            ]
                            
                            for src, dest, default in winter_fields:
                                record[dest] = winter_data.get(src, default)
                            
                            # Calculate combined wheat
                            record['Winter_Total_Wheat_Area'] = record.get('Winter_Durum_Wheat_Area', 0) + record.get('Winter_Soft_Wheat_Area', 0)
                            record['Winter_Total_Wheat_Production'] = record.get('Winter_Durum_Wheat_Production', 0) + record.get('Winter_Soft_Wheat_Production', 0)
                        
                        # Add land use features
                        if not land_data.empty:
                            land_fields = [
                                ('Cultivated_Area', 'Cultivated_Area', 0),
                                ('Fallow_Land', 'Fallow_Land', 0),
                                ('Fruit_Trees', 'Fruit_Trees', 0),
                                ('Total_Agricultural_Area', 'Total_Agricultural_Area', 0),
                            ]
                            
                            for src, dest, default in land_fields:
                                record[dest] = land_data.get(src, default)
                        
                        # Calculate derived features
                        record['Maize_Yield_Potential'] = record.get('Summer_Maize_Yield', 0) if record.get('Summer_Maize_Area', 0) > 0 else 0
                        record['Sorghum_Yield_Potential'] = record.get('Summer_Sorghum_Yield', 0) if record.get('Summer_Sorghum_Area', 0) > 0 else 0
                        record['Winter_Yield_Potential'] = record.get('Total_Winter_Yield', 0) if record.get('Total_Winter_Area', 0) > 0 else 0
                        
                        # Calculate total cereals
                        record['Total_Cereal_Area'] = record.get('Total_Summer_Area', 0) + record.get('Total_Winter_Area', 0)
                        record['Total_Cereal_Production'] = record.get('Total_Summer_Production', 0) + record.get('Total_Winter_Production', 0)
                        
                        master_records.append(record)
                        
                    except Exception as e:
                        print(f"Error processing wilaya {wilaya_code}, year {year}: {str(e)}")
                        continue
        
        master_df = pd.DataFrame(master_records)
        
        if master_df.empty:
            print("Warning: No valid records found for master dataset")
            return master_df
        
        # Add meteorological data if provided
        if meteo_df is not None and not meteo_df.empty:
            meteo_df = meteo_df.copy()
            
            # Ensure consistent column names
            if 'wilaya_code' in meteo_df.columns:
                meteo_df = meteo_df.rename(columns={'wilaya_code': 'Wilaya_Code'})
            if 'year' in meteo_df.columns:
                meteo_df = meteo_df.rename(columns={'year': 'Year'})
            
            # Ensure proper data types
            meteo_df['Wilaya_Code'] = pd.to_numeric(meteo_df['Wilaya_Code'], errors='coerce').astype('Int64')
            meteo_df['Year'] = pd.to_numeric(meteo_df['Year'], errors='coerce').astype('Int64')
            
            # Merge with master dataset
            master_df = pd.merge(
                master_df,
                meteo_df,
                on=['Wilaya_Code', 'Year'],
                how='left'
            )
            
            # Report on merge results
            merged_count = master_df['Wilaya_Code'].notnull().sum()
            total_count = len(master_df)
            print(f"Merged meteorological data for {merged_count}/{total_count} records")
        
        # Fill missing values
        numeric_cols = master_df.select_dtypes(include=[np.number]).columns
        master_df[numeric_cols] = master_df[numeric_cols].fillna(0)
        
        # Sort and reset index
        master_df = master_df.sort_values(['Year', 'Wilaya_Code']).reset_index(drop=True)
        
        return master_df
    
    def load_all_data(self) -> Dict:
        """Load and process all data with comprehensive logging"""
        print("=" * 50)
        print("Starting data loading process...")
        print("=" * 50)
        
        result = {}
        
        try:
            # Load meteorological data if DB URL is provided
            if self.db_url:
                print("\n1. Loading meteorological data...")
                meteo_raw = self.load_meteorological_data()
                print(f"   Loaded {len(meteo_raw)} raw meteorological records")
                
                meteo_processed = self.preprocess_meteo_data(meteo_raw)
                print(f"   Processed {len(meteo_processed)} meteorological records")
                
                seasonal_df = self.create_seasonal_features(meteo_processed)
                print(f"   Created seasonal features for {len(seasonal_df)} season-wilaya combinations")
                
                pivot_df = self.pivot_seasonal_data(seasonal_df)
                print(f"   Created pivot table with {len(pivot_df)} wilaya-year combinations")
                
                result.update({
                    'meteo_raw': meteo_raw,
                    'meteo_processed': meteo_processed,
                    'seasonal_df': seasonal_df,
                    'pivot_df': pivot_df
                })
            else:
                print("\n1. Skipping meteorological data (no DB URL provided)")
                pivot_df = None
            
            # Load agricultural data
            print("\n2. Loading agricultural data...")
            agricultural_data = self.load_agricultural_data()
            loaded_count = sum(1 for df in agricultural_data.values() if not df.empty)
            print(f"   Loaded {loaded_count} agricultural datasets")
            
            # Create master dataset
            print("\n3. Creating master dataset...")
            master_df = self.create_master_dataset(agricultural_data, pivot_df)
            print(f"   Created master dataset with {len(master_df)} records")
            print(f"   Dataset shape: {master_df.shape}")
            print(f"   Columns: {list(master_df.columns)}")
            
            result.update({
                'agricultural_data': agricultural_data,
                'master_df': master_df
            })
            
            print("\n" + "=" * 50)
            print("Data loading completed successfully!")
            print("=" * 50)
            
            return result
            
        except Exception as e:
            print(f"\nError during data loading: {str(e)}")
            raise
    
    def save_master_dataset(self, output_path: str = "master_dataset.csv"):
        """Save the master dataset to CSV"""
        try:
            data = self.load_all_data()
            master_df = data['master_df']
            
            if not master_df.empty:
                master_df.to_csv(output_path, index=False)
                print(f"Master dataset saved to: {output_path}")
                print(f"Records saved: {len(master_df)}")
            else:
                print("Warning: Master dataset is empty, nothing to save")
                
        except Exception as e:
            print(f"Error saving master dataset: {str(e)}")
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary statistics of the loaded data"""
        try:
            data = self.load_all_data()
            master_df = data['master_df']
            
            if master_df.empty:
                return pd.DataFrame()
            
            summary = {
                'total_records': len(master_df),
                'years_covered': sorted(master_df['Year'].unique()),
                'wilayas_covered': len(master_df['Wilaya_Code'].unique()),
                'missing_values': master_df.isnull().sum().sum(),
                'columns': len(master_df.columns),
                'numeric_columns': len(master_df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(master_df.select_dtypes(include=['object']).columns)
            }
            
            return pd.DataFrame([summary])
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return pd.DataFrame()