#!/bin/bash

# Script to copy trade data files and push to GitHub
# This script copies CSV files from ../MindWealth/cache to trade_store/stock_data

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Starting trade data update process..."
echo "📁 Working directory: $SCRIPT_DIR"

# Define directories
CACHE_DIR="../MindWealth/cache"
TARGET_STOCK_DATA_DIR="trade_store/stock_data"
SOURCE_TRADE_DIR="../MindWealth/trade_store/US"
SOURCE_VIRTUAL_TRADING_DIR="../MindWealth/trade_store"
TARGET_TRADE_DIR="trade_store/US"

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ Error: Cache directory $CACHE_DIR does not exist!"
    exit 1
fi

# Check if trade store source directory exists
if [ ! -d "$SOURCE_TRADE_DIR" ]; then
    echo "❌ Error: Trade store directory $SOURCE_TRADE_DIR does not exist!"
    exit 1
fi

# Copy all CSV files from cache to stock_data
echo "📊 Copying stock data CSV files from cache..."
cp "$CACHE_DIR"/*.csv "$TARGET_STOCK_DATA_DIR"/ 2>/dev/null || echo "⚠️  No CSV files found in cache"

# Copy all CSV files from trade_store/US
# Exclude dated versions of forward_testing.csv and latest_performance.csv
echo "📊 Copying trade signal CSV files..."
for file in "$SOURCE_TRADE_DIR"/*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Skip dated versions of forward_testing.csv and latest_performance.csv
        if [[ $filename =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_(forward_testing|latest_performance)\.csv$ ]]; then
            echo "⏭️  Skipping dated file: $filename (using non-dated version only)"
            continue
        fi
        cp "$file" "$TARGET_TRADE_DIR"/
    fi
done

# Copy all TXT files from trade_store/US
echo "📄 Copying trade signal TXT files..."
cp "$SOURCE_TRADE_DIR"/*.txt "$TARGET_TRADE_DIR"/ 2>/dev/null || echo "⚠️  No TXT files found in trade_store/US"

# Clean up old dated CSV and TXT files after copying new ones
# This ensures we keep only the latest dated file for each base name
echo "🧹 Cleaning up old dated files (keeping only latest for each base name)..."
if [ -d "$TARGET_TRADE_DIR" ]; then
    cd "$TARGET_TRADE_DIR"
    
    # Create a temporary file to track files to keep
    KEEP_FILES=$(mktemp)
    
    # Process all CSV files matching date pattern (YYYY-MM-DD_*.csv)
    # Exclude forward_testing.csv and latest_performance.csv from cleanup
    for file in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.csv; do
        if [ -f "$file" ]; then
            # Extract date and base name (e.g., 2025-11-06_new_signal.csv -> date: 2025-11-06, base: new_signal.csv)
            filename=$(basename "$file")
            if [[ $filename =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})_(.+)$ ]]; then
                date_part="${BASH_REMATCH[1]}"
                base_name="${BASH_REMATCH[2]}"
                
                # Skip cleanup for forward_testing.csv and latest_performance.csv
                if [ "$base_name" = "forward_testing.csv" ] || [ "$base_name" = "latest_performance.csv" ]; then
                    continue
                fi
                
                # Check if we already have a file for this base name
                existing_date=$(grep -F "^${base_name}:" "$KEEP_FILES" 2>/dev/null | cut -d: -f3)
                
                if [ -z "$existing_date" ]; then
                    # First file for this base name, add it
                    echo "${base_name}:${filename}:${date_part}" >> "$KEEP_FILES"
                else
                    # Compare dates - keep the newer one
                    if [ "$date_part" \> "$existing_date" ]; then
                        # New file is newer, update the entry
                        tmp_keep=$(mktemp)
                        grep -Fv "^${base_name}:" "$KEEP_FILES" > "$tmp_keep" 2>/dev/null || true
                        mv "$tmp_keep" "$KEEP_FILES"
                        echo "${base_name}:${filename}:${date_part}" >> "$KEEP_FILES"
                    fi
                fi
            fi
        fi
    done
    
    # Collect CSV files to keep
    CSV_FILES_TO_KEEP=""
    if [ -f "$KEEP_FILES" ]; then
        while IFS=: read -r base_name filename date_part; do
            CSV_FILES_TO_KEEP="$CSV_FILES_TO_KEEP $filename"
        done < "$KEEP_FILES"
    fi
    
    # Delete old dated CSV files (keep only the latest for each base name)
    # Exclude forward_testing.csv and latest_performance.csv from deletion
    deleted_count=0
    for file in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.csv; do
        if [ -f "$file" ]; then
            # Extract base name to check if it should be excluded
            filename=$(basename "$file")
            if [[ $filename =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})_(.+)$ ]]; then
                base_name="${BASH_REMATCH[2]}"
                
                # Skip deletion for forward_testing.csv and latest_performance.csv
                if [ "$base_name" = "forward_testing.csv" ] || [ "$base_name" = "latest_performance.csv" ]; then
                    continue
                fi
            fi
            
            # Check if this file should be kept
            should_keep=false
            for keep_file in $CSV_FILES_TO_KEEP; do
                if [ "$file" = "$keep_file" ]; then
                    should_keep=true
                    break
                fi
            done
            
            if [ "$should_keep" = false ]; then
                rm -f "$file"
                deleted_count=$((deleted_count + 1))
            fi
        fi
    done
    
    if [ $deleted_count -gt 0 ]; then
        echo "✅ Deleted $deleted_count old dated CSV file(s)"
    else
        echo "ℹ️  No old dated CSV files to delete"
    fi
    
    # Clean up temporary file and process TXT files
    rm -f "$KEEP_FILES" "$KEEP_FILES.bak" 2>/dev/null
    KEEP_FILES=$(mktemp)
    
    # Process all TXT files matching date pattern (YYYY-MM-DD_*.txt)
    for file in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.txt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [[ $filename =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})_(.+)$ ]]; then
                date_part="${BASH_REMATCH[1]}"
                base_name="${BASH_REMATCH[2]}"
                
                # Check if we already have a file for this base name
                existing_date=$(grep -F "^${base_name}:" "$KEEP_FILES" 2>/dev/null | cut -d: -f3)
                
                if [ -z "$existing_date" ]; then
                    # First file for this base name, add it
                    echo "${base_name}:${filename}:${date_part}" >> "$KEEP_FILES"
                else
                    # Compare dates - keep the newer one
                    if [ "$date_part" \> "$existing_date" ]; then
                        # New file is newer, update the entry
                        tmp_keep=$(mktemp)
                        grep -Fv "^${base_name}:" "$KEEP_FILES" > "$tmp_keep" 2>/dev/null || true
                        mv "$tmp_keep" "$KEEP_FILES"
                        echo "${base_name}:${filename}:${date_part}" >> "$KEEP_FILES"
                    fi
                fi
            fi
        fi
    done
    
    # Collect TXT files to keep
    TXT_FILES_TO_KEEP=""
    if [ -f "$KEEP_FILES" ]; then
        while IFS=: read -r base_name filename date_part; do
            TXT_FILES_TO_KEEP="$TXT_FILES_TO_KEEP $filename"
        done < "$KEEP_FILES"
    fi
    
    # Delete old dated TXT files (keep only the latest for each base name)
    deleted_txt_count=0
    for file in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.txt; do
        if [ -f "$file" ]; then
            # Check if this file should be kept
            should_keep=false
            for keep_file in $TXT_FILES_TO_KEEP; do
                if [ "$file" = "$keep_file" ]; then
                    should_keep=true
                    break
                fi
            done
            
            if [ "$should_keep" = false ]; then
                rm -f "$file"
                deleted_txt_count=$((deleted_txt_count + 1))
            fi
        fi
    done
    
    if [ $deleted_txt_count -gt 0 ]; then
        echo "✅ Deleted $deleted_txt_count old dated TXT file(s)"
    fi
    
    # Clean up temporary file
    rm -f "$KEEP_FILES" "$KEEP_FILES.bak" 2>/dev/null
    
    cd "$SCRIPT_DIR"
fi

# Copy virtual trading CSV files specifically (from trade_store root, not US subfolder)
echo "📊 Copying virtual trading CSV files..."
if [ -f "$SOURCE_VIRTUAL_TRADING_DIR/virtual_trading_long.csv" ]; then
    cp "$SOURCE_VIRTUAL_TRADING_DIR/virtual_trading_long.csv" "$TARGET_TRADE_DIR/virtual_trading_long.csv"
    echo "✅ Copied virtual_trading_long.csv → virtual_trading_long.csv"
else
    echo "⚠️  virtual_trading_long.csv not found in $SOURCE_VIRTUAL_TRADING_DIR"
fi

if [ -f "$SOURCE_VIRTUAL_TRADING_DIR/virtual_trading_short.csv" ]; then
    cp "$SOURCE_VIRTUAL_TRADING_DIR/virtual_trading_short.csv" "$TARGET_TRADE_DIR/virtual_trading_short.csv"
    echo "✅ Copied virtual_trading_short.csv → virtual_trading_short.csv"
else
    echo "⚠️  virtual_trading_short.csv not found in $SOURCE_VIRTUAL_TRADING_DIR"
fi

# Copy breadth_us.csv from trade_store to trade_store/US
echo "📊 Copying breadth_us.csv..."
if [ -f "$SOURCE_VIRTUAL_TRADING_DIR/breadth_us.csv" ]; then
    cp "$SOURCE_VIRTUAL_TRADING_DIR/breadth_us.csv" "$TARGET_TRADE_DIR/breadth_us.csv"
    echo "✅ Copied breadth_us.csv → breadth_us.csv"
else
    echo "⚠️  breadth_us.csv not found in $SOURCE_VIRTUAL_TRADING_DIR"
fi


# Convert signals to data structure
echo "🔄 Converting signals to chatbot data structure..."
echo "🐍 Activating virtual environment..."
source venv/bin/activate
python3 chatbot/convert_signals_to_data_structure.py

# Git operations
echo "🔄 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Update trade data: CSV files from cache and trade_store/US"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ All done! Data updated and pushed to GitHub."