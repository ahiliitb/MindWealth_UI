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
    cd "$TARGET_TRADE_DIR" || { echo "❌ Error: Cannot cd to $TARGET_TRADE_DIR"; exit 1; }
    
    # Process CSV files: Find latest file for each base name and delete older ones
    # Count files before cleanup
    files_before=$(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.csv" 2>/dev/null | wc -l | tr -d ' ')
    
    # Get all unique base names from dated CSV files
    # Use process substitution to avoid subshell issues
    while IFS= read -r base_name; do
        # Skip empty lines
        [ -z "$base_name" ] && continue
                
                # Skip cleanup for forward_testing.csv and latest_performance.csv
                if [ "$base_name" = "forward_testing.csv" ] || [ "$base_name" = "latest_performance.csv" ]; then
                    continue
                fi
                
        # Find all files for this base name using find (more reliable than glob)
        matching_files=$(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_${base_name}" 2>/dev/null)
                
        if [ -n "$matching_files" ]; then
            # Get the latest file (first when sorted descending by filename)
            latest_file=$(echo "$matching_files" | sed 's|^\./||' | sort -r | head -1)
            
            if [ -n "$latest_file" ] && [ -f "$latest_file" ]; then
                # Delete all other files with the same base name
                echo "$matching_files" | sed 's|^\./||' | while IFS= read -r file; do
                    [ -z "$file" ] && continue
                    if [ -f "$file" ] && [ "$file" != "$latest_file" ]; then
                        rm -f "$file" && echo "  🗑️  Deleted old file: $file (keeping $latest_file)"
                    fi
                done
            fi
        fi
    done < <(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.csv" 2>/dev/null | \
        sed 's|^\./||' | \
        sed 's/^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_//' | \
        sort -u)
    
    # Count files after cleanup
    files_after=$(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.csv" 2>/dev/null | wc -l | tr -d ' ')
    deleted_count=$((files_before - files_after))
    
    if [ $deleted_count -gt 0 ]; then
        echo "✅ Deleted $deleted_count old dated CSV file(s) (kept latest for each base name)"
    else
        echo "ℹ️  No old dated CSV files to delete (all files are already the latest)"
    fi
    
    # Process TXT files the same way
    # Count files before cleanup
    txt_files_before=$(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.txt" 2>/dev/null | wc -l | tr -d ' ')
    
    # Get all unique base names from dated TXT files
    while IFS= read -r base_name; do
        # Skip empty lines
        [ -z "$base_name" ] && continue
        
        # Find all files for this base name using find (more reliable than glob)
        matching_files=$(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_${base_name}" 2>/dev/null)
                
        if [ -n "$matching_files" ]; then
            # Get the latest file (first when sorted descending by filename)
            latest_file=$(echo "$matching_files" | sed 's|^\./||' | sort -r | head -1)
            
            if [ -n "$latest_file" ] && [ -f "$latest_file" ]; then
                # Delete all other files with the same base name
                echo "$matching_files" | sed 's|^\./||' | while IFS= read -r file; do
                    [ -z "$file" ] && continue
                    if [ -f "$file" ] && [ "$file" != "$latest_file" ]; then
                        rm -f "$file" && echo "  🗑️  Deleted old file: $file (keeping $latest_file)"
                    fi
                done
            fi
        fi
    done < <(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.txt" 2>/dev/null | \
        sed 's|^\./||' | \
        sed 's/^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_//' | \
        sort -u)
    
    # Count files after cleanup
    txt_files_after=$(find . -maxdepth 1 -type f -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.txt" 2>/dev/null | wc -l | tr -d ' ')
    deleted_txt_count=$((txt_files_before - txt_files_after))
    
    if [ $deleted_txt_count -gt 0 ]; then
        echo "✅ Deleted $deleted_txt_count old dated TXT file(s) (kept latest for each base name)"
    fi
    
    cd "$SCRIPT_DIR" || true
else
    echo "⚠️  Warning: Target trade directory $TARGET_TRADE_DIR does not exist, skipping cleanup"
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