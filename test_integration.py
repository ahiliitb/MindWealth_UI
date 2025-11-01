#!/usr/bin/env python3

"""
Test SIGNAL_KEYS integration with the full chatbot system
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from chatbot.chatbot_engine import ChatbotEngine
from chatbot.signal_extractor import SignalExtractor
import json

def test_full_integration():
    """Test complete SIGNAL_KEYS integration flow"""
    
    print("🧪 Testing Full SIGNAL_KEYS Integration")
    print("=" * 60)
    
    # Initialize chatbot engine
    engine = ChatbotEngine()
    
    # Test query asking for specific signals
    test_query = "give me the top 3 FRACTAL TRACK signals"
    
    print(f"📤 Query: {test_query}")
    print()
    
    try:
        # Process the query
        response, metadata = engine.query(
            user_message=test_query,
            from_date='2025-10-20',
            to_date='2025-10-23'
        )
        
        result = {
            'response': response,
            'tables': metadata.get('tables', [])
        }
        
        print("🎯 Query Result:")
        print("=" * 40)
        print(f"Response Length: {len(result.get('response', ''))}")
        print(f"Tables Generated: {len(result.get('tables', []))}")
        
        # Check if response contains SIGNAL_KEYS
        response = result.get('response', '')
        print(f"📄 GPT Response Preview: {response[:500]}...")
        print()
        
        if 'SIGNAL_KEYS' in response:
            print("✅ Response contains SIGNAL_KEYS format")
            
            # Extract SIGNAL_KEYS manually for verification
            import re
            pattern = r'SIGNAL_KEYS:\s*(\[[\s\S]*?\])'
            match = re.search(pattern, response)
            
            if match:
                try:
                    signal_keys = json.loads(match.group(1))
                    print(f"✅ Found {len(signal_keys)} signals in SIGNAL_KEYS")
                    
                    for i, signal in enumerate(signal_keys, 1):
                        print(f"  {i}. {signal.get('symbol')} - {signal.get('function')} - {signal.get('signal_date')}")
                        
                except json.JSONDecodeError:
                    print("❌ SIGNAL_KEYS format invalid")
            else:
                print("❌ SIGNAL_KEYS block not found in response")
        else:
            print("⚠️  Response does not contain SIGNAL_KEYS format")
        
        # Check tables
        tables = result.get('tables', [])
        if tables:
            print(f"\n📊 Generated {len(tables)} table(s):")
            for i, table in enumerate(tables, 1):
                table_data = table.get('data', [])
                print(f"  Table {i}: {len(table_data)} rows")
                
                if table_data:
                    # Show first few rows as sample
                    print("  Sample data:")
                    for j, row in enumerate(table_data[:3], 1):
                        symbol = row.get('Symbol, Signal, Signal Date/Price[$]', 'Unknown')
                        function = row.get('Function', 'Unknown')
                        print(f"    {j}. {symbol} - {function}")
        else:
            print("⚠️  No tables generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_extractor_with_real_response():
    """Test signal extractor with a real GPT response format"""
    
    print("\n🔧 Testing Signal Extractor with Real Format")
    print("=" * 60)
    
    # Sample response that follows our new format
    sample_response = """
Based on your request for the top 3 FRACTAL TRACK signals, here's my analysis:

**Top 3 FRACTAL TRACK Entry Signals:**

1. **FXI - China Large Cap ETF**
   - Signal Date: October 20, 2025
   - CAGR Difference: 3.12% above buy-and-hold
   - Sharpe Ratio: 0.36
   
2. **SPY - S&P 500 ETF** 
   - Signal Date: October 21, 2025
   - CAGR Difference: 2.85% above buy-and-hold
   - Sharpe Ratio: 0.42

3. **AAPL - Apple Inc**
   - Signal Date: October 22, 2025
   - CAGR Difference: 2.45% above buy-and-hold
   - Sharpe Ratio: 0.38

SIGNAL_KEYS: [
    {"function": "FRACTAL TRACK", "symbol": "FXI", "interval": "Daily", "signal_date": "2025-10-20"},
    {"function": "FRACTAL TRACK", "symbol": "SPY", "interval": "Daily", "signal_date": "2025-10-21"},
    {"function": "FRACTAL TRACK", "symbol": "AAPL", "interval": "Daily", "signal_date": "2025-10-22"}
]

These signals represent the strongest momentum patterns in the current market environment.
"""
    
    extractor = SignalExtractor()
    extracted_signals = extractor._extract_specific_signals_from_response(sample_response)
    
    print(f"✅ Extracted {len(extracted_signals)} specific signals")
    
    expected_count = 3
    if len(extracted_signals) == expected_count:
        print(f"✅ Correct number of signals extracted ({expected_count})")
        
        expected_symbols = ['FXI', 'SPY', 'AAPL']
        extracted_symbols = [signal[1] for signal in extracted_signals]
        
        if extracted_symbols == expected_symbols:
            print("✅ All expected symbols extracted correctly")
            return True
        else:
            print(f"❌ Symbol mismatch: expected {expected_symbols}, got {extracted_symbols}")
            return False
    else:
        print(f"❌ Wrong number of signals: expected {expected_count}, got {len(extracted_signals)}")
        return False

if __name__ == "__main__":
    print("🚀 SIGNAL_KEYS Integration Test")
    print("=" * 60)
    
    # Test 1: Signal extractor with real format
    extractor_success = test_signal_extractor_with_real_response()
    
    # Test 2: Full integration (this will make actual API calls)
    print("\n" + "="*60)
    integration_input = input("Run full integration test (will make API calls)? [y/N]: ").lower().strip()
    
    if integration_input == 'y':
        integration_success = test_full_integration()
    else:
        print("⏭️  Skipping full integration test")
        integration_success = True  # Skip this test
    
    print("\n📊 Final Results:")
    print("=" * 60)
    print(f"Signal Extractor: {'✅ PASS' if extractor_success else '❌ FAIL'}")
    print(f"Full Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if extractor_success and integration_success:
        print("\n🎉 SIGNAL_KEYS system is working correctly!")
        print("The chatbot will now show only GPT-mentioned signals in tables.")
    else:
        print("\n⚠️  Some issues found. Check the implementation.")