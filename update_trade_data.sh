#!/bin/bash

# Script to copy trade data files and push to GitHub
# This script copies files from ../MindWealth/trade_store/US to trade_store/US

set -e  # Exit on any error

echo "🔄 Starting trade data update process..."

# Check if source directory exists
SOURCE_DIR="../MindWealth/trade_store/US"
TARGET_DIR="trade_store/US"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory $SOURCE_DIR does not exist!"
    exit 1
fi

# Create target directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "📁 Creating target directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# Copy all files from source to target
cp -r "$SOURCE_DIR"/* "$TARGET_DIR"/

git add .

# Commit changes
git commit -m "Update trade data files from MindWealth/trade_store/US"
git push origin main