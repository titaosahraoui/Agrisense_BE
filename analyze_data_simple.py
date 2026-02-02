
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('cleaned_training_data_20260121_195915.csv')
    print(f"SHAPE: {df.shape}")
    print("COLUMNS: " + ", ".join(df.columns))
    
    print("\nNULL COUNTS:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0])
    
    if 'stress_score' in df.columns:
        print("\nSTRESS_SCORE STATS:")
        print(df['stress_score'].describe())
    
    if 'ndvi' in df.columns:
        print("\nNDVI STATS:")
        print(df['ndvi'].describe())
        
except Exception as e:
    print(str(e))
