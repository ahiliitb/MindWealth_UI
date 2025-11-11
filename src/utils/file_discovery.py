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
        'bollinger_band.csv': 'bollinger_band',
        'Distance.csv': 'distance',
        'Fib-Ret.csv': 'fib_ret',
        'General-Divergence.csv': 'general_divergence',
        'new_high.csv': 'new_high',
        'Stochastic-Divergence.csv': 'stochastic_divergence',
        'sigma.csv': 'sigma',
        'sentiment.csv': 'sentiment',
        'Trendline.csv': 'trendline',
        'breadth.csv': 'breadth',
        'outstanding_signal.csv': 'outstanding_signal',
        'outstanding_exit_signal.csv': 'outstanding_exit_signal',
        'new_signal.csv': 'new_signal',
        'target_signal.csv': 'target_signal',
        'latest_performance.csv': 'latest_performance',
        'forward_backtesting.csv': 'forward_backtesting',
        'forward_testing.csv': 'forward_backtesting'  # Use same parser as forward_backtesting
    }
    
    return file_mapping.get(base_filename, 'unknown')


def discover_csv_files():
    """Dynamically discover all CSV files in the trade_store/US directory (handles dated filenames)"""
    # Define the specific order for page names
    ordered_pages = [
        'Band Matrix',
        'DeltaDrift', 
        'Fractal Track',
        'BaselineDiverge',
        'Altitude Alpha',
        'Oscillator Delta',
        'SigmaShell',
        'PulseGauge',
        'TrendPulse',
        'Signal Breadth Indicator (SBI)',
        'Outstanding Signals',
        'Outstanding Target',
        'Outstanding Signals Exit',
        'New Signals',
        'Latest Performance',
        'Forward Testing Performance',
        'Horizontal'
    ]
    
    # Map base file names (without date) to model function names
    base_name_mapping = {
        'bollinger_band.csv': 'Band Matrix',
        'Distance.csv': 'DeltaDrift',
        'Fib-Ret.csv': 'Fractal Track',
        'General-Divergence.csv': 'BaselineDiverge',
        'new_high.csv': 'Altitude Alpha',
        'Stochastic-Divergence.csv': 'Oscillator Delta',
        'sigma.csv': 'SigmaShell',
        'sentiment.csv': 'PulseGauge',
        'Trendline.csv': 'TrendPulse',
        'breadth.csv': 'Signal Breadth Indicator (SBI)',
        'outstanding_signal.csv': 'Outstanding Signals',
        'outstanding_exit_signal.csv': 'Outstanding Signals Exit',
        'new_signal.csv': 'New Signals',
        'latest_performance.csv': 'Latest Performance',
        'forward_backtesting.csv': 'Forward Testing Performance',
        'forward_testing.csv': 'Forward Testing Performance',  # Alternative filename
        'target_signal.csv': 'Outstanding Target',
        'Horizontal.csv': 'Horizontal'
    }
    
    csv_files = {}
    trade_store_path = "./trade_store/US"
    
    if os.path.exists(trade_store_path):
        # Find all CSV files
        csv_pattern = os.path.join(trade_store_path, "*.csv")
        csv_file_paths = glob.glob(csv_pattern)
        
        # Create a mapping from base filename to filepath
        # If multiple dated versions exist, prefer the most recent
        # For forward_testing.csv and latest_performance.csv, ONLY use non-dated versions
        file_mapping = {}
        
        # Files that MUST use non-dated versions only
        non_dated_only_files = ['latest_performance.csv', 'forward_testing.csv']
        
        for file_path in csv_file_paths:
            filename = os.path.basename(file_path)
            base_filename = get_base_filename(filename)
            
            if base_filename in base_name_mapping:
                current_date = extract_date_from_filename(filename)
                
                # For forward_testing.csv and latest_performance.csv, skip any dated versions
                if base_filename in non_dated_only_files:
                    if current_date:
                        # Skip dated versions of these files
                        continue
                
                # If we haven't seen this base file, add it
                if base_filename not in file_mapping:
                    file_mapping[base_filename] = file_path
                else:
                    existing_date = extract_date_from_filename(os.path.basename(file_mapping[base_filename]))
                    
                    # For forward_testing.csv and latest_performance.csv, only keep non-dated
                    if base_filename in non_dated_only_files:
                        # If existing is dated and current is not, replace it
                        if existing_date and not current_date:
                            file_mapping[base_filename] = file_path
                        # If both are non-dated, keep first one found
                        # If current is dated, skip it (already handled above)
                    else:
                        # For other files, prefer dated versions (existing logic)
                        if current_date and existing_date:
                            if current_date > existing_date:
                                file_mapping[base_filename] = file_path
                        elif current_date and not existing_date:
                            # Prefer dated file over non-dated
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
            
            # If multiple candidates, prefer specific ones (e.g., forward_testing.csv over forward_backtesting.csv)
            if candidates:
                # Prefer forward_testing.csv if both exist
                if 'forward_testing.csv' in candidates:
                    selected_base = 'forward_testing.csv'
                else:
                    selected_base = candidates[0]  # Use first found
                csv_files[page_name] = file_mapping[selected_base]
    
    return csv_files

