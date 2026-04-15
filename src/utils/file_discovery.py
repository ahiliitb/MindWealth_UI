"""
File discovery and CSV structure detection utilities
"""

import os
import glob
import re
from datetime import datetime


def extract_date_from_filename(filename):
    """
    Extract date from filename in format: YYYY-MM-DD_filename.ext (date prefix convention)
    Returns datetime object if found, None otherwise
    Works with .csv, .txt, and other extensions
    """
    # Pattern to match dates at the START of filename: YYYY-MM-DD_filename.ext
    date_pattern = r'^(\d{4}-\d{2}-\d{2})_(.+)\.(csv|txt|json)$'
    match = re.search(date_pattern, filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d')
        except ValueError:
            return None
    return None


def get_base_filename(filename):
    """
    Extract base filename without date prefix.
    e.g., '2025-10-31_bollinger_band.csv' -> 'bollinger_band.csv'
    """
    # Pattern to match dates at the START: YYYY-MM-DD_filename.csv
    date_pattern = r'^\d{4}-\d{2}-\d{2}_(.+)\.csv$'
    match = re.search(date_pattern, filename)
    if match:
        return match.group(1) + '.csv'
    # If no date prefix, return filename as-is
    return filename


def detect_csv_structure(file_path):
    """Detect the structure and type of CSV file based on filename (handles dated filenames)"""
    filename = os.path.basename(file_path)
    
    # Get base filename (without date)
    base_filename = get_base_filename(filename)
    
    # Map filenames to their specific parsers
    file_mapping = {
        'all_signal.csv': 'all_signal',
        'breadth.csv': 'breadth',
        'outstanding_signal.csv': 'outstanding_signal',
        'new_signal.csv': 'new_signal',
        'target_signal.csv': 'target_signal',
        'combined_performance_report.csv': 'combined_performance_report',
        'F-Stack-Analyzer.csv': 'f_stack_analyzer'
    }
    
    return file_mapping.get(base_filename, 'unknown')


def get_latest_csv_file(base_filename, trade_store_path="./trade_store/US"):
    """Return the latest matching CSV for a base filename, preferring dated files."""
    if not os.path.exists(trade_store_path):
        return None

    csv_pattern = os.path.join(trade_store_path, "*.csv")
    selected_file = None
    selected_date = None

    for file_path in glob.glob(csv_pattern):
        filename = os.path.basename(file_path)
        if get_base_filename(filename) != base_filename:
            continue

        current_date = extract_date_from_filename(filename)

        if selected_file is None:
            selected_file = file_path
            selected_date = current_date
            continue

        if current_date and selected_date:
            if current_date > selected_date:
                selected_file = file_path
                selected_date = current_date
        elif current_date and not selected_date:
            selected_file = file_path
            selected_date = current_date

    return selected_file


def discover_csv_files():
    """Dynamically discover all CSV files in the trade_store/US directory (handles dated filenames)"""
    # Define the specific order for page names
    ordered_pages = [
        'All Signal Report',
        'Signal Breadth Indicator (SBI)',
        'Outstanding Signals',
        'Portfolio Risk Management',
        'New Signals',
        'Combined Performance Report',
        'F-Stack'
    ]
    
    # Map base file names (without date) to model function names
    base_name_mapping = {
        'all_signal.csv': 'All Signal Report',
        'breadth.csv': 'Signal Breadth Indicator (SBI)',
        'outstanding_signal.csv': 'Outstanding Signals',
        'new_signal.csv': 'New Signals',
        'combined_performance_report.csv': 'Combined Performance Report',
        'target_signal.csv': 'Portfolio Risk Management',
        'F-Stack-Analyzer.csv': 'F-Stack'
    }
    
    csv_files = {}
    trade_store_path = "./trade_store/US"
    
    if os.path.exists(trade_store_path):
        # Find all CSV files
        csv_pattern = os.path.join(trade_store_path, "*.csv")
        csv_file_paths = glob.glob(csv_pattern)
        
        # Create a mapping from base filename to filepath
        # If multiple dated versions exist, prefer the most recent
        file_mapping = {}
        
        for file_path in csv_file_paths:
            filename = os.path.basename(file_path)
            base_filename = get_base_filename(filename)
            
            if base_filename in base_name_mapping:
                current_date = extract_date_from_filename(filename)

                # If we haven't seen this base file, add it
                if base_filename not in file_mapping:
                    file_mapping[base_filename] = file_path
                else:
                    existing_date = extract_date_from_filename(os.path.basename(file_mapping[base_filename]))

                    # Prefer dated files over non-dated and latest dated among duplicates
                    if current_date and existing_date:
                        if current_date > existing_date:
                            file_mapping[base_filename] = file_path
                    elif current_date and not existing_date:
                        file_mapping[base_filename] = file_path
                    # If both are dated and current is older, keep existing
                    # If both are non-dated, keep first one found
        
        # Add files in the specified order
        for page_name in ordered_pages:
            # Find the corresponding file(s) - may have multiple mappings for same page
            candidates = []
            for base_name, mapped_name in base_name_mapping.items():
                if mapped_name == page_name and base_name in file_mapping:
                    candidates.append(base_name)
            
            if candidates:
                selected_base = candidates[0]
                csv_files[page_name] = file_mapping[selected_base]
    
    india_trade_store_path = "./trade_store/INDIA"
    if os.path.exists(india_trade_store_path):
        f_stack_pattern = os.path.join(india_trade_store_path, "*.csv")
        selected_f_stack_file = None
        selected_f_stack_date = None
        
        for file_path in glob.glob(f_stack_pattern):
            filename = os.path.basename(file_path)
            base_filename = get_base_filename(filename)
            if base_filename != 'F-Stack-Analyzer.csv':
                continue
            
            current_date = extract_date_from_filename(filename)
            
            if selected_f_stack_file is None:
                selected_f_stack_file = file_path
                selected_f_stack_date = current_date
            else:
                if current_date and selected_f_stack_date:
                    if current_date > selected_f_stack_date:
                        selected_f_stack_file = file_path
                        selected_f_stack_date = current_date
                elif current_date and not selected_f_stack_date:
                    selected_f_stack_file = file_path
                    selected_f_stack_date = current_date
                elif not selected_f_stack_date and not current_date:
                    # keep first non-dated file
                    pass
        
        if selected_f_stack_file:
            csv_files['F-Stack'] = selected_f_stack_file
    
    return csv_files

