#!/usr/bin/env python3

"""
Test SIGNAL_KEYS filtering to ensure only specific signals are shown
"""

import sys
import os
import json

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from chatbot.signal_extractor import SignalExtractor
import pandas as pd

def test_signal_keys_filtering():
    """Test that SIGNAL_KEYS filtering works correctly"""
    
    print("🧪 Testing SIGNAL_KEYS Filtering")
    print("=" * 60)
    
    # Sample GPT response with SIGNAL_KEYS (like the user provided)
    sample_response = """
Based on your analysis, here are the top signals:

1. **OSCILLATOR DELTA - MA** - Strong momentum signal from October 20, 2025
2. **FRACTAL TRACK - NKE** - Breakout pattern detected on October 20, 2025  
3. **FRACTAL TRACK - UBER** - Support bounce signal from October 20, 2025

SIGNAL_KEYS: [
  {"function": "OSCILLATOR DELTA", "symbol": "MA", "interval": "Daily", "signal_date": "2025-10-20"},
  {"function": "FRACTAL TRACK", "symbol": "NKE", "interval": "Daily", "signal_date": "2025-10-20"},
  {"function": "FRACTAL TRACK", "symbol": "UBER", "interval": "Daily", "signal_date": "2025-10-20"}
]

These signals show strong technical patterns in the current market.
"""
    
    # Create sample data that includes MORE signals than just the ones mentioned
    sample_data = {
        'entry': pd.DataFrame([
            # The 3 signals mentioned in SIGNAL_KEYS
            {'Function': 'OSCILLATOR DELTA', 'Symbol, Signal, Signal Date/Price[$]': 'MA, Long, 2025-10-20 (Price: 100.00)', 'CAGR difference (Strategy - Buy and Hold) [%]': '5.2%'},
            {'Function': 'FRACTAL TRACK', 'Symbol, Signal, Signal Date/Price[$]': 'NKE, Long, 2025-10-20 (Price: 150.00)', 'CAGR difference (Strategy - Buy and Hold) [%]': '4.8%'},
            {'Function': 'FRACTAL TRACK', 'Symbol, Signal, Signal Date/Price[$]': 'UBER, Long, 2025-10-20 (Price: 75.00)', 'CAGR difference (Strategy - Buy and Hold) [%]': '3.9%'},
            
            # Additional signals that should NOT appear in the filtered result
            {'Function': 'FRACTAL TRACK', 'Symbol, Signal, Signal Date/Price[$]': 'AAPL, Long, 2025-10-21 (Price: 200.00)', 'CAGR difference (Strategy - Buy and Hold) [%]': '3.5%'},
            {'Function': 'OSCILLATOR DELTA', 'Symbol, Signal, Signal Date/Price[$]': 'MSFT, Long, 2025-10-21 (Price: 300.00)', 'CAGR difference (Strategy - Buy and Hold) [%]': '3.2%'},
            {'Function': 'BOLLINGER BAND', 'Symbol, Signal, Signal Date/Price[$]': 'GOOGL, Long, 2025-10-22 (Price: 150.00)', 'CAGR difference (Strategy - Buy and Hold) [%]': '2.8%'},
        ])
    }
    
    extractor = SignalExtractor()
    
    # Step 1: Test signal identification
    print("🔍 Testing signal identification...")
    used_signals = extractor._identify_used_signals(sample_response, sample_data)
    
    print(f"✅ Signal types identified: {used_signals['signal_types']}")
    print(f"✅ Functions: {used_signals['functions']}")
    print(f"✅ Symbols: {used_signals['symbols']}")
    print(f"✅ Signal keys count: {len(used_signals.get('signal_keys', []))}")
    
    # Step 2: Test filtering
    print(f"\n🧹 Testing signal filtering...")
    print(f"📊 Original data: {len(sample_data['entry'])} rows")
    
    filtered_data = extractor._filter_used_signals(
        sample_data['entry'],
        used_signals,
        used_signals.get('signal_keys', [])
    )
    
    print(f"📋 Filtered data: {len(filtered_data)} rows")
    
    # Step 3: Verify results
    print(f"\n🎯 Filtered Results:")
    if not filtered_data.empty:
        for i, row in filtered_data.iterrows():
            function = row.get('Function', 'Unknown')
            symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', 'Unknown')
            print(f"  {len(filtered_data) - i}. {function} - {symbol_info}")
    
    # Step 4: Check correctness
    expected_symbols = ['MA', 'NKE', 'UBER']
    expected_functions = ['OSCILLATOR DELTA', 'FRACTAL TRACK']
    
    success = True
    
    # Check if we got exactly 3 signals
    if len(filtered_data) != 3:
        print(f"❌ Expected 3 signals, got {len(filtered_data)}")
        success = False
    else:
        print(f"✅ Correct number of signals: 3")
    
    # Check if the right symbols are included
    found_symbols = []
    for _, row in filtered_data.iterrows():
        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', '')
        for symbol in expected_symbols:
            if symbol in symbol_info:
                found_symbols.append(symbol)
                break
    
    if set(found_symbols) == set(expected_symbols):
        print(f"✅ All expected symbols found: {expected_symbols}")
    else:
        print(f"❌ Symbol mismatch. Expected: {expected_symbols}, Found: {found_symbols}")
        success = False
    
    return success

if __name__ == "__main__":
    print("🚀 SIGNAL_KEYS Filtering Test")
    print("=" * 60)
    
    success = test_signal_keys_filtering()
    
    print(f"\n📊 Result: {'✅ SUCCESS - SIGNAL_KEYS filtering works correctly!' if success else '❌ FAILED - Issues found'}")
    
    if success:
        print("🎉 The system will now show only the signals mentioned in SIGNAL_KEYS!")
    else:
        print("⚠️  The filtering needs additional fixes.")