"""
File discovery and CSV structure detection utilities
"""

import os
import glob


def detect_csv_structure(file_path):
    """Detect the structure and type of CSV file based on filename"""
    filename = os.path.basename(file_path)
    
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
        'forward_backtesting.csv': 'forward_backtesting'
    }
    
    return file_mapping.get(filename, 'unknown')


def discover_csv_files():
    """Dynamically discover all CSV files in the trade_store/US directory"""
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
        'Forward Testing Performance'
    ]
    
    # Map file names to model function names
    name_mapping = {
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
        'target_signal.csv': 'Outstanding Target'
    }
    
    csv_files = {}
    trade_store_path = "./trade_store/US"
    
    if os.path.exists(trade_store_path):
        # Find all CSV files
        csv_pattern = os.path.join(trade_store_path, "*.csv")
        csv_file_paths = glob.glob(csv_pattern)
        
        # Create a mapping from filename to filepath
        file_mapping = {}
        for file_path in csv_file_paths:
            filename = os.path.basename(file_path)
            file_mapping[filename] = file_path
        
        # Add files in the specified order
        for page_name in ordered_pages:
            # Find the corresponding file
            for original_name, mapped_name in name_mapping.items():
                if mapped_name == page_name and original_name in file_mapping:
                    csv_files[page_name] = file_mapping[original_name]
                    break
    
    return csv_files

