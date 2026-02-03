#!/usr/bin/env python3
"""
Script to update column names from "Current" to "Today's" in all CSV and JSON files.
Updates:
- "Current Trading Date/Price[$], Current Price vs Signal" -> "Today's Trading Date/Price[$], Today's Price vs Signal"
- "Trading Days between Signal and Current Date" -> "Trading Days between Signal and Today's Date"
- "Current Price" -> "Today's Price"
"""

import os
import json
import csv
from pathlib import Path

def update_csv_file(file_path):
    """Update column names in a CSV file."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if updates are needed
        if "Current Trading Date/Price" not in content and "Trading Days between Signal and Current Date" not in content and "Current Price" not in content:
            return False
        
        # Make replacements
        original_content = content
        content = content.replace(
            "Current Trading Date/Price[$], Current Price vs Signal",
            "Today's Trading Date/Price[$], Today's Price vs Signal"
        )
        content = content.replace(
            "Trading Days between Signal and Current Date",
            "Trading Days between Signal and Today's Date"
        )
        # Only replace "Current Price" if it's a column header (followed by comma or end of line)
        # and not part of the longer column name
        if '"Current Price"' in content or ',Current Price,' in content or 'Current Price,' in content:
            content = content.replace('"Current Price"', '"Today\'s Price"')
            content = content.replace(',Current Price,', ',Today\'s Price,')
            # Handle case where it might be at the end
            if content.endswith('Current Price'):
                content = content[:-len('Current Price')] + 'Today\'s Price'
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Error updating {file_path}: {e}")
        return False

def update_json_file(file_path):
    """Update column names in a JSON file."""
    try:
        # Read the file as text first for string replacement
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if updates are needed
        if "Current Trading Date/Price" not in content and "Trading Days between Signal and Current Date" not in content:
            return False
        
        # Make replacements
        original_content = content
        content = content.replace(
            "Current Trading Date/Price[$], Current Price vs Signal",
            "Today's Trading Date/Price[$], Today's Price vs Signal"
        )
        content = content.replace(
            "Trading Days between Signal and Current Date",
            "Trading Days between Signal and Today's Date"
        )
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all CSV and JSON files."""
    base_dir = Path(__file__).parent
    
    # Directories to search
    search_dirs = [
        base_dir / "chatbot" / "data",
        base_dir / "trade_store",
        base_dir / "chatbot" / "history",
    ]
    
    csv_updated = 0
    csv_total = 0
    json_updated = 0
    json_total = 0
    
    print("=" * 60)
    print("UPDATING COLUMN NAMES FROM 'CURRENT' TO 'TODAY'S'")
    print("=" * 60)
    
    # Update CSV files
    print("\n📄 Updating CSV files...")
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for csv_file in search_dir.rglob("*.csv"):
            csv_total += 1
            if update_csv_file(csv_file):
                csv_updated += 1
                print(f"  ✓ Updated: {csv_file.relative_to(base_dir)}")
    
    # Update JSON files
    print("\n📄 Updating JSON files...")
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for json_file in search_dir.rglob("*.json"):
            json_total += 1
            if update_json_file(json_file):
                json_updated += 1
                print(f"  ✓ Updated: {json_file.relative_to(base_dir)}")
    
    # Also update monitored_trades.json in root
    monitored_trades = base_dir / "monitored_trades.json"
    if monitored_trades.exists():
        json_total += 1
        if update_json_file(monitored_trades):
            json_updated += 1
            print(f"  ✓ Updated: monitored_trades.json")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CSV files: {csv_updated}/{csv_total} updated")
    print(f"JSON files: {json_updated}/{json_total} updated")
    print(f"Total: {csv_updated + json_updated}/{csv_total + json_total} files updated")
    print("=" * 60)

if __name__ == "__main__":
    main()
