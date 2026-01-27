#!/usr/bin/env python3
"""
Deduplicate chatbot CSV files.
Removes duplicate rows from entry.csv, exit.csv, portfolio_target_achieved.csv, and breadth.csv
based on their respective deduplication keys.
"""

import pandas as pd
from pathlib import Path

def deduplicate_file(file_path, dedup_cols):
    """Deduplicate a CSV file based on specified columns."""
    if not Path(file_path).exists():
        print(f"⚠ {file_path}: File not found")
        return
    
    df = pd.read_csv(file_path)
    original = len(df)
    
    # Remove duplicates, keeping first occurrence
    df_deduped = df.drop_duplicates(subset=dedup_cols, keep='first')
    removed = original - len(df_deduped)
    
    if removed > 0:
        df_deduped.to_csv(file_path, index=False)
        print(f"✓ {file_path}: Removed {removed} duplicates ({original} → {len(df_deduped)})")
    else:
        print(f"✓ {file_path}: No duplicates found ({original} rows)")

def main():
    """Deduplicate all chatbot CSV files."""
    print("\n" + "="*80)
    print("CHATBOT DATA DEDUPLICATION")
    print("="*80 + "\n")
    
    # Standard dedup columns for signal files (entry, exit, portfolio_target_achieved)
    signal_dedup_cols = ['Function', 'Symbol', 'Interval', 'Signal', 'Signal Open Price']
    
    # Breadth dedup columns
    breadth_dedup_cols = ['Function', 'Date']
    
    # Deduplicate each file
    files_to_dedup = [
        ('chatbot/data/entry.csv', signal_dedup_cols),
        ('chatbot/data/exit.csv', signal_dedup_cols),
        ('chatbot/data/portfolio_target_achieved.csv', signal_dedup_cols),
        ('chatbot/data/breadth.csv', breadth_dedup_cols)
    ]
    
    for file_path, dedup_cols in files_to_dedup:
        deduplicate_file(file_path, dedup_cols)
    
    print("\n" + "="*80)
    print("✓ Deduplication Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
