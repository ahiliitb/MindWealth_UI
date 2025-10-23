"""
Test script to validate signal type filtering based on AI response content.
"""

import pandas as pd
from chatbot.signal_extractor import SignalExtractor

def test_entry_only_response():
    """Test that only entry signals are shown when AI response only mentions entries."""
    
    print("=== TESTING ENTRY-ONLY FILTERING ===")
    
    # Sample data with both entry and exit signals
    sample_data = {
        'entry': pd.DataFrame({
            'Symbol, Signal, Signal Date/Price[$]': [
                'AAPL, Long, 2025-10-14 (Price: 247.66)',
                'MSFT, Short, 2025-10-13 (Price: 415.23)'
            ],
            'Function': [
                'FRACTAL TRACK',
                'BOLLINGER BANDS'
            ],
            'Interval, Confirmation Status': [
                '1D, Confirmed',
                '4H, Pending'
            ],
            '_signal_type': ['entry', 'entry'],
            '_asset': ['AAPL', 'MSFT'],
            '_function': ['FRACTAL TRACK', 'BOLLINGER BANDS'],
            '_date': ['2025-10-14', '2025-10-13']
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
            '_signal_type': ['exit'],
            '_asset': ['TSLA'],
            '_function': ['BREAKOUT MATRIX'],
            '_date': ['2025-10-15']
        })
    }
    
    # AI response that ONLY mentions entry signals (no exit mentions)
    entry_only_response = """
    Based on the recent market analysis, I've identified strong entry opportunities:
    
    **AAPL Long Entry**: The FRACTAL TRACK analysis shows a bullish breakout pattern at $247.66. 
    This entry signal on the daily chart indicates strong upward momentum with good risk/reward ratio.
    
    **MSFT Short Entry**: BOLLINGER BANDS analysis reveals an overbought condition at $415.23, 
    presenting a short entry opportunity on the 4-hour timeframe.
    
    These entry positions offer excellent potential with proper risk management.
    """
    
    # Initialize SignalExtractor
    extractor = SignalExtractor()
    
    # Extract signal tables
    query_params = {
        'assets': ['AAPL', 'MSFT', 'TSLA'],
        'functions': ['FRACTAL TRACK', 'BOLLINGER BANDS', 'BREAKOUT MATRIX'],
        'from_date': '2025-10-01',
        'to_date': '2025-10-20'
    }
    
    full_tables = extractor.extract_full_signal_tables(entry_only_response, sample_data, query_params)
    
    print(f"\n✅ SUCCESS: Extracted {len(full_tables)} signal type tables (should be 1 for entry only)")
    print(f"Signal types found: {list(full_tables.keys())}")
    
    # Should only have entry table, no exit table
    assert 'entry' in full_tables, "Entry table should be present"
    assert 'exit' not in full_tables, "Exit table should NOT be present when not mentioned in response"
    
    for signal_type, df in full_tables.items():
        print(f"\n{signal_type.upper()} TABLE ({len(df)} rows):")
        print(f"Assets: {df['_asset'].unique().tolist() if '_asset' in df.columns else 'N/A'}")
        print(f"Functions: {df['_function'].unique().tolist() if '_function' in df.columns else 'N/A'}")
    
    return full_tables

def test_exit_only_response():
    """Test that only exit signals are shown when AI response only mentions exits."""
    
    print("\n=== TESTING EXIT-ONLY FILTERING ===")
    
    # Same sample data with both entry and exit signals
    sample_data = {
        'entry': pd.DataFrame({
            'Symbol, Signal, Signal Date/Price[$]': [
                'AAPL, Long, 2025-10-14 (Price: 247.66)',
                'MSFT, Short, 2025-10-13 (Price: 415.23)'
            ],
            'Function': [
                'FRACTAL TRACK',
                'BOLLINGER BANDS'
            ],
            'Interval, Confirmation Status': [
                '1D, Confirmed',
                '4H, Pending'
            ],
            '_signal_type': ['entry', 'entry'],
            '_asset': ['AAPL', 'MSFT'],
            '_function': ['FRACTAL TRACK', 'BOLLINGER BANDS'],
            '_date': ['2025-10-14', '2025-10-13']
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
            '_signal_type': ['exit'],
            '_asset': ['TSLA'],
            '_function': ['BREAKOUT MATRIX'],
            '_date': ['2025-10-15']
        })
    }
    
    # AI response that ONLY mentions exit signals (no entry mentions)
    exit_only_response = """
    Portfolio update: We're closing positions based on technical analysis.
    
    **TSLA Exit Signal**: BREAKOUT MATRIX analysis indicates it's time to close the TSLA position 
    at $245.78. The exit signal shows we've reached our target and should take profits.
    
    This exit will lock in gains and reduce portfolio risk exposure.
    """
    
    # Initialize SignalExtractor
    extractor = SignalExtractor()
    
    # Extract signal tables
    query_params = {
        'assets': ['AAPL', 'MSFT', 'TSLA'],
        'functions': ['FRACTAL TRACK', 'BOLLINGER BANDS', 'BREAKOUT MATRIX'],
        'from_date': '2025-10-01',
        'to_date': '2025-10-20'
    }
    
    full_tables = extractor.extract_full_signal_tables(exit_only_response, sample_data, query_params)
    
    print(f"\n✅ SUCCESS: Extracted {len(full_tables)} signal type tables (should be 1 for exit only)")
    print(f"Signal types found: {list(full_tables.keys())}")
    
    # Should only have exit table, no entry table  
    assert 'exit' in full_tables, "Exit table should be present"
    assert 'entry' not in full_tables, "Entry table should NOT be present when not mentioned in response"
    
    for signal_type, df in full_tables.items():
        print(f"\n{signal_type.upper()} TABLE ({len(df)} rows):")
        print(f"Assets: {df['_asset'].unique().tolist() if '_asset' in df.columns else 'N/A'}")
        print(f"Functions: {df['_function'].unique().tolist() if '_function' in df.columns else 'N/A'}")
    
    return full_tables

if __name__ == "__main__":
    print("Testing Signal Type Filtering Based on AI Response Content\n")
    
    # Test 1: Entry-only response should only show entry table
    test_entry_only_response()
    
    # Test 2: Exit-only response should only show exit table  
    test_exit_only_response()
    
    print("\n🎉 All tests passed! Signal filtering is working correctly.")