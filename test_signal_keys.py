#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.signal_extractor import SignalExtractor

def test_signal_keys_extraction():
    """Test the new SIGNAL_KEYS JSON extraction functionality"""
    
    print("🧪 Testing SIGNAL_KEYS JSON Extraction")
    print("=" * 50)
    
    # Initialize signal extractor
    extractor = SignalExtractor()
    
    # Test with sample GPT response that includes SIGNAL_KEYS
    sample_ai_response = """
    ## Top 5 Signals Analysis

    Based on my analysis of the data, here are the top 5 signals by Sharpe Ratio:

    1. **AAPL** - Sharpe Ratio: 1.85, CAGR Difference: 13.2%
    2. **MSFT** - Sharpe Ratio: 1.72, CAGR Difference: 12.8%
    3. **GOOGL** - Sharpe Ratio: 1.68, CAGR Difference: 11.5%
    4. **TSLA** - Sharpe Ratio: 1.45, CAGR Difference: 10.2%
    5. **NVDA** - Sharpe Ratio: 1.32, CAGR Difference: 9.8%

    These signals highlight the top performers based on risk-adjusted returns and strategy outperformance.

    SIGNAL_KEYS: [
      {"function": "FRACTAL TRACK", "symbol": "AAPL", "interval": "Daily", "signal_date": "2025-10-16"},
      {"function": "BAND MATRIX", "symbol": "MSFT", "interval": "Daily", "signal_date": "2025-10-15"},
      {"function": "MATRIX DIVERGENCE", "symbol": "GOOGL", "interval": "Daily", "signal_date": "2025-10-14"},
      {"function": "FRACTAL TRACK", "symbol": "TSLA", "interval": "Daily", "signal_date": "2025-10-13"},
      {"function": "BOLLINGER BAND", "symbol": "NVDA", "interval": "Daily", "signal_date": "2025-10-12"}
    ]
    """
    
    print("📝 Sample AI Response with SIGNAL_KEYS:")
    print("..." + sample_ai_response[sample_ai_response.find("SIGNAL_KEYS"):] + "...")
    
    # Extract specific signals
    specific_signals = extractor._extract_specific_signals_from_response(sample_ai_response)
    
    print(f"\n🎯 Extracted Specific Signals: {len(specific_signals)}")
    for i, (function, symbol, interval, date) in enumerate(specific_signals, 1):
        print(f"   {i}. {symbol} - {function} - {interval} - {date}")
    
    # Verify results
    expected_count = 5
    print(f"\n✅ Expected: {expected_count} signals")
    print(f"✅ Result: {len(specific_signals)} signals extracted")
    
    if len(specific_signals) == expected_count:
        print("✅ SUCCESS: Correctly extracted exact number of signals from SIGNAL_KEYS")
    else:
        print("❌ ISSUE: Signal count doesn't match expected")
    
    # Verify first signal
    if specific_signals and specific_signals[0] == ('FRACTAL TRACK', 'AAPL', 'Daily', '2025-10-16'):
        print("✅ SUCCESS: First signal correctly parsed")
    else:
        print(f"❌ ISSUE: First signal parsing failed. Got: {specific_signals[0] if specific_signals else 'None'}")

if __name__ == "__main__":
    test_signal_keys_extraction()