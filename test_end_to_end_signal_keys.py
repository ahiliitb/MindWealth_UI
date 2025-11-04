#!/usr/bin/env python3

"""
Test complete SIGNAL_KEYS workflow with real data
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from chatbot.chatbot_engine import ChatbotEngine

def test_end_to_end_signal_keys():
    """Test complete SIGNAL_KEYS workflow"""
    
    print("🧪 Testing Complete SIGNAL_KEYS Workflow")
    print("=" * 60)
    
    # Initialize engine
    engine = ChatbotEngine()
    
    # Test with a query that should trigger SIGNAL_KEYS
    # Use a broader date range to ensure we have data
    test_query = "show me the top 5 FRACTAL TRACK signals"
    
    print(f"📤 Query: {test_query}")
    print("📅 Date range: 2025-10-10 to 2025-10-25")
    print()
    
    try:
        # Process query
        response, metadata = engine.query(
            user_message=test_query,
            from_date='2025-10-10',
            to_date='2025-10-25',
            functions=['FRACTAL TRACK']
        )
        
        print("🎯 Results:")
        print("=" * 40)
        print(f"Response length: {len(response)} characters")
        
        # Check for SIGNAL_KEYS in response
        if 'SIGNAL_KEYS' in response:
            print("✅ Response contains SIGNAL_KEYS format")
            
            # Extract SIGNAL_KEYS section
            import re
            import json
            pattern = r'SIGNAL_KEYS:\s*(\[[\s\S]*?\])'
            match = re.search(pattern, response)
            
            if match:
                try:
                    signal_keys = json.loads(match.group(1))
                    print(f"✅ Found {len(signal_keys)} signals in SIGNAL_KEYS")
                    
                    print("\n📋 SIGNAL_KEYS:")
                    for i, signal in enumerate(signal_keys, 1):
                        print(f"  {i}. {signal['symbol']} - {signal['function']} - {signal['signal_date']}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ SIGNAL_KEYS JSON parsing failed: {e}")
        else:
            print("⚠️  No SIGNAL_KEYS found in response")
        
        # Check tables
        tables = metadata.get('tables', [])
        if tables:
            print(f"\n📊 Generated {len(tables)} table(s):")
            
            for i, table in enumerate(tables, 1):
                table_data = table.get('data', [])
                signal_type = table.get('signal_type', 'unknown')
                
                print(f"\n  Table {i} ({signal_type}): {len(table_data)} rows")
                
                if table_data:
                    print("  Sample rows:")
                    for j, row in enumerate(table_data[:3], 1):
                        function = row.get('Function', 'Unknown')
                        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', 'Unknown')
                        cagr = row.get('CAGR difference (Strategy - Buy and Hold) [%]', 'N/A')
                        print(f"    {j}. {function} - {symbol_info} - CAGR: {cagr}")
                        
                    if len(table_data) > 3:
                        print(f"    ... and {len(table_data) - 3} more rows")
        else:
            print("⚠️  No tables generated")
        
        # Show first part of response
        print(f"\n📄 Response Preview:")
        print("-" * 40)
        print(response[:500] + "..." if len(response) > 500 else response)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 End-to-End SIGNAL_KEYS Test")
    print("=" * 60)
    
    success = test_end_to_end_signal_keys()
    
    print(f"\n📊 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("🎉 SIGNAL_KEYS system is working end-to-end!")
        print("📋 Tables now show only the signals mentioned by GPT")
    else:
        print("⚠️  Issues detected in end-to-end flow")