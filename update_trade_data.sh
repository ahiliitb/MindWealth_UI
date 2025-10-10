#!/bin/bash

# Script to copy trade data files and push to GitHub
# This script copies CSV files from ../MindWealth/cache to trade_store/stock_data

set -e  # Exit on any error

echo "🔄 Starting trade data update process..."

# Define directories
CACHE_DIR="../MindWealth/cache"
TARGET_STOCK_DATA_DIR="trade_store/stock_data"
SOURCE_TRADE_DIR="../MindWealth/trade_store/US"
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
echo "📊 Copying trade signal CSV files..."
cp "$SOURCE_TRADE_DIR"/*.csv "$TARGET_TRADE_DIR"/ 2>/dev/null || echo "⚠️  No CSV files found in trade_store/US"


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