#!/usr/bin/env python3

"""
Test data loading directly to see if files are being read
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from chatbot.smart_data_fetcher import SmartDataFetcher

def test_data_loading():
    """Test if data loading is working"""
    
    print("🧪 Testing Data Loading")
    print("=" * 60)
    
    fetcher = SmartDataFetcher()
    
    # Test with known parameters
    print("📂 Testing AMZN FRACTAL TRACK data loading...")
    
    try:
        data_dict = fetcher.fetch_data(
            signal_types=['entry'],
            required_columns=None,  # Get all columns
            assets=['AMZN'],
            functions=['FRACTAL TRACK'],
            from_date='2025-10-10',
            to_date='2025-10-25'
        )
        
        # Combine data from all signal types
        import pandas as pd
        all_data = []
        for signal_type, df in data_dict.items():
            if not df.empty:
                all_data.append(df)
        
        data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        
        print(f"✅ Loaded data: {len(data)} rows")
        
        if not data.empty:
            print("📊 Sample data:")
            sample = data.iloc[0]
            print(f"  Function: {sample.get('Function', 'N/A')}")
            print(f"  Symbol: {sample.get('Symbol, Signal, Signal Date/Price[$]', 'N/A')}")
            print(f"  Date columns: {[col for col in data.columns if 'Date' in col]}")
        
        return len(data) > 0
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_existence():
    """Check if the data files exist"""
    
    print("\n🔍 Checking File Existence")
    print("=" * 60)
    
    import glob
    
    # Check for AMZN FRACTAL TRACK files
    pattern = "chatbot/data/entry/AMZN/FRACTAL TRACK/*.csv"
    files = glob.glob(pattern)
    
    print(f"📁 Found {len(files)} AMZN FRACTAL TRACK files:")
    for f in files:
        print(f"  {f}")
    
    if files:
        # Read first file to see structure
        import pandas as pd
        try:
            sample_file = files[0]
            df = pd.read_csv(sample_file)
            print(f"\n📄 Sample file: {sample_file}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            if len(df) > 0:
                print("  Sample row:")
                for col, val in df.iloc[0].items():
                    print(f"    {col}: {val}")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    return len(files) > 0

if __name__ == "__main__":
    print("🚀 Data Loading Test")
    print("=" * 60)
    
    # Test 1: File existence
    files_exist = test_file_existence()
    
    # Test 2: Data loading
    if files_exist:
        data_loads = test_data_loading()
    else:
        print("⚠️  No files found, skipping data loading test")
        data_loads = False
    
    print(f"\n📊 Results:")
    print(f"Files exist: {'✅ YES' if files_exist else '❌ NO'}")
    print(f"Data loads: {'✅ YES' if data_loads else '❌ NO'}")
    
    if files_exist and data_loads:
        print("\n🎉 Data system is working!")
    else:
        print("\n⚠️  Data loading issues detected")