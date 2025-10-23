#!/usr/bin/env python3
"""
Test script to verify that signal extraction preserves original CSV structure.
This tests that ALL columns are fetched in their original order and format.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.getcwd())

from chatbot.smart_data_fetcher import SmartDataFetcher

def test_csv_structure_preservation():
    """Test that we can fetch ALL columns preserving original CSV structure."""
    
    print("🧪 Testing CSV Structure Preservation")
    print("=" * 50)
    
    # Initialize smart data fetcher
    fetcher = SmartDataFetcher()
    
    # Test 1: Fetch ALL columns (no column filtering)
    print("\n1️⃣ Testing ALL columns fetch (required_columns=None)")
    
    try:
        result = fetcher.fetch_data(
            signal_types=['entry'],
            required_columns=None,  # Fetch ALL columns to preserve structure
            assets=['AAPL'],
            functions=['FRACTAL TRACK'],
            from_date='2025-10-16',
            to_date='2025-10-16'
        )
        
        if 'entry' in result and not result['entry'].empty:
            df = result['entry']
            print(f"✅ Successfully fetched {len(df)} rows with {len(df.columns)} columns")
            
            # Check original CSV structure
            original_csv = Path('chatbot/data/entry/AAPL/FRACTAL TRACK/2025-10-16.csv')
            if original_csv.exists():
                original_df = pd.read_csv(original_csv)
                
                print(f"\n📊 Column Comparison:")
                print(f"   Original CSV: {len(original_df.columns)} columns")
                print(f"   Fetched Data: {len(df.columns)} columns")
                
                # Compare column names and order
                original_cols = original_df.columns.tolist()
                fetched_cols = [col for col in df.columns if not col.startswith('_')]  # Exclude metadata cols
                
                print(f"\n🔍 Column Structure Analysis:")
                if original_cols == fetched_cols:
                    print(f"   ✅ Column names and order PERFECTLY MATCH original CSV!")
                else:
                    print(f"   ❌ Column structure differs from original")
                    print(f"   Original: {original_cols[:5]}... ({len(original_cols)} total)")
                    print(f"   Fetched:  {fetched_cols[:5]}... ({len(fetched_cols)} total)")
                
                # Display first few column names from both
                print(f"\n📋 First 10 Columns:")
                for i in range(min(10, len(original_cols))):
                    orig_col = original_cols[i] if i < len(original_cols) else "N/A"
                    fetched_col = fetched_cols[i] if i < len(fetched_cols) else "N/A"
                    match = "✅" if orig_col == fetched_col else "❌"
                    print(f"   {i+1:2d}. {match} Original: '{orig_col}'")
                    if orig_col != fetched_col:
                        print(f"       Fetched:  '{fetched_col}'")
                
                # Check data values for first row
                if len(df) > 0 and len(original_df) > 0:
                    print(f"\n📈 Data Sample Comparison:")
                    print(f"   Function: {df.iloc[0]['Function']} (Original: {original_df.iloc[0]['Function']})")
                    symbol_col = 'Symbol, Signal, Signal Date/Price[$]'
                    if symbol_col in df.columns and symbol_col in original_df.columns:
                        print(f"   Symbol: {df.iloc[0][symbol_col]}")
                        print(f"   Original: {original_df.iloc[0][symbol_col]}")
            
        else:
            print(f"❌ No data fetched for entry signals")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Test Complete!")

if __name__ == "__main__":
    test_csv_structure_preservation()