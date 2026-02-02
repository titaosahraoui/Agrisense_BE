
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('cleaned_training_data_20260121_195915.csv')
    print("=== DATA SHAPE ===")
    print(df.shape)
    
    print("\n=== COLUMNS & TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\n=== SAMPLE DATA (Head) ===")
    print(df.head(3).T)
    
    print("\n=== BASIC STATISTICS ===")
    # Filter numeric columns only for describe
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe().T[['count', 'mean', 'min', 'max']])
    
    print("\n=== CORRELATIONS WITH TARGET (yield_quintal/stress_score) ===")
    if 'yield_quintal' in df.columns:
        print("Yield Correlations:")
        print(numeric_df.corrwith(df['yield_quintal']).sort_values(ascending=False).head(5))
        print(numeric_df.corrwith(df['yield_quintal']).sort_values(ascending=False).tail(5))
        
    if 'stress_score' in df.columns:
        print("\nStress Score Correlations:")
        print(numeric_df.corrwith(df['stress_score']).sort_values(ascending=False).head(5))
        print(numeric_df.corrwith(df['stress_score']).sort_values(ascending=False).tail(5))

except Exception as e:
    print(f"Error analyzing CSV: {e}")
