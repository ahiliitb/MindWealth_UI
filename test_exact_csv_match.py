#!/usr/bin/env python3
"""
Test script to verify tables show EXACTLY the same structure as CSV files.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.getcwd())

from chatbot.signal_extractor import SignalExtractor
from chatbot.smart_data_fetcher import SmartDataFetcher

def test_exact_csv_match():
    """Test that extracted tables match original CSV files exactly."""
    
    print("🧪 Testing EXACT CSV Structure Match")
    print("=" * 50)
    
    # Initialize components
    smart_fetcher = SmartDataFetcher()
    extractor = SignalExtractor()
    
    print("\n1️⃣ Testing signal extraction with AI response...")
    
    # Simulate an AI response that mentions entry signals
    ai_response = "The AAPL entry signal shows a long position at $247.45 with FRACTAL TRACK function."
    
    # Get fetched data first
    fetched_data = smart_fetcher.fetch_data(
        signal_types=['entry'],
        required_columns=None,  # All columns
        assets=['AAPL'],
        functions=['FRACTAL TRACK'],
        from_date='2025-10-16',
        to_date='2025-10-16'
    )
    
    # Extract signal tables
    query_params = {
        'assets': ['AAPL'],
        'functions': ['FRACTAL TRACK'],
        'from_date': '2025-10-16',
        'to_date': '2025-10-16'
    }
    
    signal_tables = extractor.extract_full_signal_tables(
        ai_response=ai_response,
        fetched_data=fetched_data,
        query_params=query_params
    )
    
    if 'entry' in signal_tables and not signal_tables['entry'].empty:
        extracted_df = signal_tables['entry']
        print(f"✅ Extracted {len(extracted_df)} rows with {len(extracted_df.columns)} columns")
        
        # Load original CSV
        original_csv = Path('chatbot/data/entry/AAPL/FRACTAL TRACK/2025-10-16.csv')
        if original_csv.exists():
            original_df = pd.read_csv(original_csv)
            
            print(f"\n📊 Structure Comparison:")
            print(f"   Original CSV: {len(original_df.columns)} columns, {len(original_df)} rows")
            print(f"   Extracted:    {len(extracted_df.columns)} columns, {len(extracted_df)} rows")
            
            # Check if columns match exactly
            original_cols = original_df.columns.tolist()
            extracted_cols = extracted_df.columns.tolist()
            
            if original_cols == extracted_cols:
                print(f"   ✅ PERFECT MATCH: Columns are identical!")
            else:
                print(f"   ❌ MISMATCH: Columns differ")
                print(f"   Original has: {set(original_cols) - set(extracted_cols)}")
                print(f"   Extracted has: {set(extracted_cols) - set(original_cols)}")
                
            # Show column comparison
            print(f"\n📋 Column-by-Column Comparison:")
            max_cols = max(len(original_cols), len(extracted_cols))
            for i in range(max_cols):
                orig_col = original_cols[i] if i < len(original_cols) else "❌ MISSING"
                extr_col = extracted_cols[i] if i < len(extracted_cols) else "❌ MISSING"
                match = "✅" if orig_col == extr_col else "❌"
                print(f"   {i+1:2d}. {match} Original: '{orig_col}'")
                if orig_col != extr_col:
                    print(f"       Extracted: '{extr_col}'")
                    
            # Check first row data match
            if len(extracted_df) > 0 and len(original_df) > 0:
                print(f"\n📈 First Row Data Comparison:")
                for col in original_cols[:5]:  # Show first 5 columns
                    orig_val = original_df.iloc[0][col]
                    extr_val = extracted_df.iloc[0][col] if col in extracted_df.columns else "MISSING"
                    match = "✅" if str(orig_val) == str(extr_val) else "❌"
                    print(f"   {match} {col}: {orig_val}")
                    if str(orig_val) != str(extr_val):
                        print(f"       Extracted: {extr_val}")
        else:
            print(f"❌ Original CSV file not found: {original_csv}")
    else:
        print(f"❌ No entry signals extracted")
        
    print(f"\n🏁 Test Complete!")

if __name__ == "__main__":
    test_exact_csv_match()