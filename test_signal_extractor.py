"""
Test script to validate the SignalExtractor functionality with sample data.
"""

import pandas as pd
from chatbot.signal_extractor import SignalExtractor

def test_signal_extraction():
    """Test both legacy and new full signal extraction."""
    
    print("Testing SignalExtractor...")
    
    # Create sample data similar to what would come from smart_data_fetcher
    sample_data = {
        'entry': pd.DataFrame({
            'Symbol, Signal, Signal Date/Price[$]': [
                'AAPL, Long, 2025-10-14 (Price: 247.66)',
                'MSFT, Short, 2025-10-13 (Price: 415.23)',
                'GOOGL, Long, 2025-10-12 (Price: 182.45)'
            ],
            'Function': [
                'FRACTAL TRACK',
                'BOLLINGER BANDS',
                'MATRIX DIVERGENCE'
            ],
            'Interval, Confirmation Status': [
                '1D, Confirmed',
                '4H, Pending',
                '1H, Confirmed'
            ],
            'Exit Signal Date/Price[$]': [
                'No Exit Yet',
                '2025-10-15 (Price: 410.88)',
                'No Exit Yet'
            ],
            'Volume': [1000000, 2500000, 800000],
            'RSI': [65.4, 25.8, 72.1],
            'MACD': [0.45, -0.23, 0.78]
        }),
        'exit': pd.DataFrame({
            'Symbol, Signal, Signal Date/Price[$]': [
                'TSLA, Exit, 2025-10-15 (Price: 245.78)'
            ],
            'Function': [
                'BREAKOUT MATRIX'
            ],
            'Interval, Confirmation Status': [
                '1D, Confirmed'
            ],
            'Volume': [3200000],
            'Profit_Loss': [15.2]
        })
    }
    
    # Test AI response text
    sample_response = """
    Based on the analysis of AAPL, MSFT, and GOOGL entry signals, here are the key findings:
    
    **AAPL Long Signal**: FRACTAL TRACK indicates a bullish breakout at $247.66 on the daily timeframe.
    **MSFT Short Signal**: BOLLINGER BANDS shows oversold conditions leading to a short entry at $415.23.
    **GOOGL Long Signal**: MATRIX DIVERGENCE detected positive momentum at $182.45.
    
    Additionally, TSLA exit signal from BREAKOUT MATRIX at $245.78 shows profit taking.
    """
    
    # Initialize SignalExtractor
    extractor = SignalExtractor()
    
    # Test NEW full table extraction
    print("\n=== TESTING NEW FULL TABLE EXTRACTION ===")
    query_params = {
        'assets': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'functions': ['FRACTAL TRACK', 'BOLLINGER BANDS', 'MATRIX DIVERGENCE', 'BREAKOUT MATRIX'],
        'from_date': '2025-10-01',
        'to_date': '2025-10-20'
    }
    
    full_tables = extractor.extract_full_signal_tables(sample_response, sample_data, query_params)
    
    print(f"Extracted {len(full_tables)} signal type tables:")
    for signal_type, df in full_tables.items():
        print(f"\n{signal_type.upper()} TABLE ({len(df)} rows, {len(df.columns)} columns):")
        print(df.to_string(index=False))
        print(f"Columns: {list(df.columns)}")
    
    # Test legacy extraction for comparison
    print("\n=== TESTING LEGACY EXTRACTION ===")
    signals_df = extractor.extract_signals_from_response(sample_response, sample_data)
    print(f"Legacy extracted {len(signals_df)} signals:")
    print(signals_df.to_string(index=False))
    
    return full_tables

if __name__ == "__main__":
    test_signal_extraction()