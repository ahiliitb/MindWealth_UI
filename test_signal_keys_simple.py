#!/usr/bin/env python3

"""
Simple SIGNAL_KEYS JSON extraction test without data loading
"""

import json
import re
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from chatbot.signal_extractor import SignalExtractor

# Sample GPT response with SIGNAL_KEYS
sample_response = """
Based on your analysis, here are the top 5 entry signals:

1. **FRACTAL TRACK - FXI** - Strong momentum with CAGR difference of 3.12%
2. **FRACTAL TRACK - SPY** - Consistent performance with 2.85% CAGR difference  
3. **FRACTAL TRACK - AAPL** - Tech sector strength with 2.45% CAGR difference
4. **FRACTAL TRACK - MSFT** - Cloud computing momentum with 2.21% CAGR difference
5. **FRACTAL TRACK - NVDA** - AI sector leadership with 1.98% CAGR difference

SIGNAL_KEYS: [
    {"function": "FRACTAL TRACK", "symbol": "FXI", "interval": "Daily", "signal_date": "2025-10-20"},
    {"function": "FRACTAL TRACK", "symbol": "SPY", "interval": "Daily", "signal_date": "2025-10-21"},
    {"function": "FRACTAL TRACK", "symbol": "AAPL", "interval": "Daily", "signal_date": "2025-10-22"},
    {"function": "FRACTAL TRACK", "symbol": "MSFT", "interval": "Daily", "signal_date": "2025-10-21"},
    {"function": "FRACTAL TRACK", "symbol": "NVDA", "interval": "Daily", "signal_date": "2025-10-20"}
]

These signals show strong technical momentum across different sectors.
"""

def test_signal_keys_extraction():
    """Test SIGNAL_KEYS extraction from GPT response"""
    
    print("🧪 Testing SIGNAL_KEYS extraction...")
    print("=" * 60)
    
    # Create signal extractor instance  
    extractor = SignalExtractor()
    
    # Test the new extraction method
    extracted_signals = extractor._extract_specific_signals_from_response(sample_response)
    
    print(f"✅ Extracted {len(extracted_signals)} signals")
    print("\n📋 Extracted Signals:")
    
    for i, signal in enumerate(extracted_signals, 1):
        function, symbol, interval, signal_date = signal
        print(f"  {i}. Function: {function}")
        print(f"     Symbol: {symbol}")
        print(f"     Interval: {interval}")
        print(f"     Date: {signal_date}")
        print()
    
    # Verify expected signals
    expected_symbols = ['FXI', 'SPY', 'AAPL', 'MSFT', 'NVDA']
    extracted_symbols = [signal[1] for signal in extracted_signals]
    
    print(f"🎯 Expected: {expected_symbols}")
    print(f"🔍 Extracted: {extracted_symbols}")
    
    if extracted_symbols == expected_symbols:
        print("✅ SUCCESS: All expected signals extracted correctly!")
        return True
    else:
        print("❌ FAILURE: Signal extraction mismatch")
        return False

def test_json_parsing():
    """Test direct JSON parsing of SIGNAL_KEYS"""
    
    print("\n🔧 Testing direct JSON parsing...")
    print("=" * 60)
    
    # Extract SIGNAL_KEYS JSON block
    pattern = r'SIGNAL_KEYS:\s*(\[[\s\S]*?\])'
    match = re.search(pattern, sample_response)
    
    if match:
        json_str = match.group(1)
        print(f"📋 Found JSON block: {json_str[:100]}...")
        
        try:
            parsed_signals = json.loads(json_str)
            print(f"✅ Successfully parsed {len(parsed_signals)} signals from JSON")
            
            for i, signal in enumerate(parsed_signals, 1):
                print(f"  {i}. {signal}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            return False
    else:
        print("❌ No SIGNAL_KEYS JSON block found")
        return False

if __name__ == "__main__":
    print("🚀 SIGNAL_KEYS Extraction Test")
    print("=" * 60)
    
    # Test 1: JSON parsing
    json_success = test_json_parsing()
    
    # Test 2: Full extraction
    extraction_success = test_signal_keys_extraction()
    
    print("\n📊 Test Results:")
    print("=" * 60)
    print(f"JSON Parsing: {'✅ PASS' if json_success else '❌ FAIL'}")
    print(f"Signal Extraction: {'✅ PASS' if extraction_success else '❌ FAIL'}")
    
    if json_success and extraction_success:
        print("\n🎉 All tests passed! SIGNAL_KEYS extraction is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")