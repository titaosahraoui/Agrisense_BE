import pandas as pd
from sqlalchemy import create_engine
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Load and preprocess all agricultural and meteorological data"""
    
    def __init__(self, db_url: str, csv_base_path: str = "../ml_Models/outputs/data"):
        self.db_url = db_url
        self.csv_base_path = csv_base_path
        
    def load_meteorological_data(self) -> pd.DataFrame:
        """Load meteorological data from PostgreSQL database"""
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
        """
        
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def preprocess_meteo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess meteorological data"""
        df = df[df['season'].isin(['été', 'hiver'])].copy()
        
        # Year mapping
        year_mapping = {
            2020: 2016,
            2021: 2017,
            2018: 2018,
            2019: 2019
        }
        
        df_replaced = []
        for source_year, target_year in year_mapping.items():
            temp = df[df['year'] == source_year].copy()
            if not temp.empty:
                temp['year'] = target_year
                temp['date'] = temp['date'].apply(lambda d: d.replace(year=target_year))
                df_replaced.append(temp)
        
        if df_replaced:
            df = pd.concat(df_replaced, ignore_index=True)
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal aggregated features"""
        seasonal_df = (
            df
            .groupby(['wilaya_code', 'year', 'season'])
            .agg(
                avg_temperature=('temperature_avg', 'mean'),
                avg_rainfall=('precipitation', 'mean'),
                avg_humidity=('humidity', 'mean'),
                avg_ndvi=('ndvi', 'mean')
            )
            .reset_index()
        )
        
        # Round numeric columns
        numeric_cols = ['avg_temperature', 'avg_rainfall', 'avg_humidity', 'avg_ndvi']
        seasonal_df[numeric_cols] = seasonal_df[numeric_cols].round(3)
        
        return seasonal_df
    
    def pivot_seasonal_data(self, seasonal_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot seasonal data to wide format"""
        pivot_df = seasonal_df.pivot_table(
            index=['wilaya_code', 'year'],
            columns='season',
            values=['avg_temperature', 'avg_rainfall', 'avg_humidity', 'avg_ndvi']
        )
        
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
        pivot_df = pivot_df.reset_index()
        
        return pivot_df
    
    def load_agricultural_data(self) -> Dict[str, pd.DataFrame]:
        """Load all agricultural CSV files"""
        data_files = {
            'annual_2016': pd.read_csv(f'{self.csv_base_path}/2016.csv'),
            'annual_2017': pd.read_csv(f'{self.csv_base_path}/2017.csv'),
            'annual_2018': pd.read_csv(f'{self.csv_base_path}/2018.csv'),
            'annual_2019': pd.read_csv(f'{self.csv_base_path}/2019.csv'),
            'summer_2016': pd.read_csv(f"{self.csv_base_path}/CEREALES D'ETE 2016.csv"),
            'summer_2017': pd.read_csv(f"{self.csv_base_path}/CEREALES D'ETE 2017.csv"),
            'summer_2018': pd.read_csv(f"{self.csv_base_path}/CEREALES D'ETE 2018.csv"),
            'summer_2019': pd.read_csv(f"{self.csv_base_path}/CEREALES D'ETE 2019.csv"),
            'winter_2016': pd.read_csv(f"{self.csv_base_path}/CEREALES D'HIVER 2016.csv"),
            'winter_2017': pd.read_csv(f"{self.csv_base_path}/CEREALES D'HIVER 2017.csv"),
            'winter_2018': pd.read_csv(f"{self.csv_base_path}/CEREALES D'HIVER 2018.csv"),
            'winter_2019': pd.read_csv(f"{self.csv_base_path}/CEREALES D'HIVER 2019.csv"),
            'land_use_2016': pd.read_csv(f"{self.csv_base_path}/SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE (2016).csv"),
            'land_use_2017': pd.read_csv(f"{self.csv_base_path}/SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE (2017).csv"),
            'land_use_2018': pd.read_csv(f"{self.csv_base_path}/SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE (2018).csv"),
            'land_use_2019': pd.read_csv(f"{self.csv_base_path}/SUPERFICIES DES TERRES UTILISEES PAR L'AGRICULTURE.csv")
        }
        return data_files
    
    def create_master_dataset(self, agricultural_data: Dict, meteo_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Combine agricultural and meteorological data into master dataset"""
        # Extract yearly cereal data
        yearly_data = []
        for year, df in [('2016', agricultural_data['annual_2016']), 
                         ('2017', agricultural_data['annual_2017']),
                         ('2018', agricultural_data['annual_2018']),
                         ('2019', agricultural_data['annual_2019'])]:
            df['Year'] = int(year)
            yearly_data.append(df)
        
        yearly_df = pd.concat(yearly_data, ignore_index=True)
        
        # Process wilaya-level data
        master_records = []
        
        for year in ['2016', '2017', '2018', '2019']:
            summer_df = agricultural_data[f'summer_{year}']
            winter_df = agricultural_data[f'winter_{year}']
            land_df = agricultural_data[f'land_use_{year}']
            
            for wilaya_code in summer_df['Wilaya_Code'].unique():
                try:
                    wilaya_int = int(wilaya_code)
                    if wilaya_int == 0:
                        continue
                except (ValueError, TypeError):
                    continue
                    
                summer_row = summer_df[summer_df['Wilaya_Code'] == wilaya_code].iloc[0]
                winter_row = winter_df[winter_df['Wilaya_Code'] == wilaya_code].iloc[0]
                land_row = land_df[land_df['Wilaya_Code'] == wilaya_code].iloc[0]
                
                record = {
                    'Year': int(year),
                    'Wilaya_Code': int(wilaya_code),
                    'Wilaya_Name': summer_row['Wilaya_Name'],
                    
                    # Summer cereal features
                    'Summer_Maize_Area': summer_row.get('Maïs_Sup_ha', 0),
                    'Summer_Maize_Production': summer_row.get('Maïs_Prod_qx', 0),
                    'Summer_Maize_Yield': summer_row.get('Maïs_Rdt_qx_ha', 0),
                    'Summer_Sorghum_Area': summer_row.get('Sorgho_Sup_ha', 0),
                    'Summer_Sorghum_Production': summer_row.get('Sorgho_Prod_qx', 0),
                    'Summer_Sorghum_Yield': summer_row.get('Sorgho_Rdt_qx_ha', 0),
                    'Total_Summer_Area': summer_row.get('Total_Cereales_Ete_Sup_ha', 0),
                    'Total_Summer_Production': summer_row.get('Total_Cereales_Ete_Prod_qx', 0),
                    'Total_Summer_Yield': summer_row.get('Total_Cereales_Ete_Rdt_qx_ha', 0),
                    
                    # Winter cereal features
                    'Winter_Wheat_Area': winter_row.get('Ble_Dur_Sup_Recoltee_ha', 0) + 
                                        winter_row.get('Ble_Tendre_Sup_Recoltee_ha', 0),
                    'Winter_Wheat_Production': winter_row.get('Ble_Dur_Prod_qx', 0) + 
                                             winter_row.get('Ble_Tendre_Prod_qx', 0),
                    'Winter_Barley_Area': winter_row.get('Orge_Sup_Recoltee_ha', 0),
                    'Winter_Barley_Production': winter_row.get('Orge_Prod_qx', 0),
                    'Total_Winter_Area': winter_row.get('Total_Cereales_Hiver_Sup_Recoltee_ha', 0),
                    'Total_Winter_Production': winter_row.get('Total_Cereales_Hiver_Prod_qx', 0),
                    'Total_Winter_Yield': winter_row.get('Total_Cereales_Hiver_Rdt_qx_ha', 0),
                    
                    # Land use features
                    'Cultivated_Area': land_row.get('Cultures_herbacées_ha', 0),
                    'Fallow_Land': land_row.get('Terres_au_repos_ha', 0),
                    'Fruit_Trees': land_row.get('Plantations_arbres_fruit_ha', 0),
                    'Total_Agricultural_Area': land_row.get('TOTAL_SAU_ha', 0)
                }
                
                # Calculate derived features
                record['Maize_Yield_Potential'] = record['Summer_Maize_Yield'] if record['Summer_Maize_Area'] > 0 else 0
                record['Sorghum_Yield_Potential'] = record['Summer_Sorghum_Yield'] if record['Summer_Sorghum_Area'] > 0 else 0
                record['Winter_Yield_Potential'] = record['Total_Winter_Yield'] if record['Total_Winter_Area'] > 0 else 0
                
                master_records.append(record)
        
        master_df = pd.DataFrame(master_records)
        
        # Add meteorological data if provided
        if meteo_df is not None:
            meteo_df = meteo_df.copy()
            
            if 'wilaya_code' in meteo_df.columns:
                meteo_df = meteo_df.rename(columns={'wilaya_code': 'Wilaya_Code'})
            if 'year' in meteo_df.columns:
                meteo_df = meteo_df.rename(columns={'year': 'Year'})
            
            meteo_df['Wilaya_Code'] = meteo_df['Wilaya_Code'].astype(int)
            meteo_df['Year'] = meteo_df['Year'].astype(int)
            
            master_df = pd.merge(
                master_df,
                meteo_df,
                on=['Wilaya_Code', 'Year'],
                how='left'
            )
        
        return master_df
    
    def load_all_data(self) -> Dict:
        """Load and process all data"""
        # Load meteorological data
        meteo_raw = self.load_meteorological_data()
        meteo_processed = self.preprocess_meteo_data(meteo_raw)
        seasonal_df = self.create_seasonal_features(meteo_processed)
        pivot_df = self.pivot_seasonal_data(seasonal_df)
        
        # Load agricultural data
        agricultural_data = self.load_agricultural_data()
        
        # Create master dataset
        master_df = self.create_master_dataset(agricultural_data, pivot_df)
        
        return {
            'meteo_raw': meteo_raw,
            'meteo_processed': meteo_processed,
            'seasonal_df': seasonal_df,
            'pivot_df': pivot_df,
            'agricultural_data': agricultural_data,
            'master_df': master_df
        }