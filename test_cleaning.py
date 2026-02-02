
import pandas as pd
import numpy as np
from app.services.data_cleaner import DataCleaner

# Mock print to keep output clean or just rely on standard output
def run_test():
    print("📢 Loading dataset...")
    try:
        # Load the existing data
        df = pd.read_csv('cleaned_training_data_20260121_195915.csv')
        
        # We need to simulate the state BEFORE cleaning to test the cleaner
        # Ideally we would have raw data, but here we can take the 'cleaned' data 
        # (which apparently still has missing values based on our analysis) and try to clean it further.
        
        # Check specific columns
        target_cols = ['ndvi', 'ndwi', 'lst']
        
        print("\n📊 INITIAL MISSING VALUES:")
        initial_missing = df[target_cols].isnull().sum()
        print(initial_missing)
        
        # Initialize cleaner (db not needed for clean_data method)
        cleaner = DataCleaner(db=None) 
        
        print("\n🚀 Running DataCleaner with interpolation...")
        # We use a strategy that includes our new logic. 
        # The new logic is applied unconditionally before the specific strategy block in the modified code.
        df_cleaned = cleaner.clean_data(df, strategy="simple") 
        
        print("\n📊 MISSING VALUES AFTER INTERPOLATION:")
        final_missing = df_cleaned[target_cols].isnull().sum()
        print(final_missing)
        
        print("\n🎉 RESULTS:")
        for col in target_cols:
            filled = initial_missing[col] - final_missing[col]
            pct = (filled / initial_missing[col] * 100) if initial_missing[col] > 0 else 0
            print(f"{col}: Recovered {filled} values ({pct:.1f}% of missing)")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
