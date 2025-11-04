#!/usr/bin/env python3

"""
Test SIGNAL_KEYS with known data and AMZN (which we know works)
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from chatbot.chatbot_engine import ChatbotEngine

def test_with_amzn():
    """Test with AMZN which we know has data and works"""
    
    print("🧪 Testing SIGNAL_KEYS with AMZN (known working)")
    print("=" * 60)
    
    engine = ChatbotEngine()
    
    # Test with AMZN which we know works and has FRACTAL TRACK data
    test_query = "give me AMZN FRACTAL TRACK signals"
    
    print(f"📤 Query: {test_query}")
    
    try:
        response, metadata = engine.query(
            user_message=test_query,
            tickers=['AMZN'],  # Explicitly specify AMZN
            from_date='2025-10-16',  # Exact date where we know data exists
            to_date='2025-10-16',
            functions=['FRACTAL TRACK']  # Explicitly specify function
        )
        
        print("\n🎯 Query Result:")
        print("=" * 40)
        print(f"Response Length: {len(response)}")
        
        # Show response preview
        print(f"📄 Response Preview: {response[:600]}...")
        
        # Check for SIGNAL_KEYS
        if 'SIGNAL_KEYS' in response:
            print("\n✅ Response contains SIGNAL_KEYS format!")
            
            # Extract and show SIGNAL_KEYS
            import re
            import json
            pattern = r'SIGNAL_KEYS:\s*(\[[\s\S]*?\])'
            match = re.search(pattern, response)
            
            if match:
                try:
                    signal_keys = json.loads(match.group(1))
                    print(f"✅ Found {len(signal_keys)} signals in SIGNAL_KEYS:")
                    
                    for i, signal in enumerate(signal_keys, 1):
                        print(f"  {i}. {signal}")
                        
                except json.JSONDecodeError:
                    print("❌ SIGNAL_KEYS format invalid")
        else:
            print("\n⚠️  Response does not contain SIGNAL_KEYS format")
            print("This could mean:")
            print("  - No data available for this query")
            print("  - GPT needs more explicit instruction")
            print("  - Date range issue")
        
        # Check tables
        tables = metadata.get('tables', [])
        if tables:
            print(f"\n📊 Generated {len(tables)} table(s)")
            for i, table in enumerate(tables, 1):
                table_data = table.get('data', [])
                print(f"  Table {i}: {len(table_data)} rows")
                
                # Show sample data
                if table_data:
                    sample_row = table_data[0]
                    print(f"  Sample: {sample_row.get('Function', 'N/A')} - {sample_row.get('Symbol, Signal, Signal Date/Price[$]', 'N/A')}")
        else:
            print("\n⚠️  No tables generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 AMZN SIGNAL_KEYS Test")
    print("=" * 60)
    
    success = test_with_amzn()
    
    print(f"\n📊 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("🎉 SIGNAL_KEYS system ready for use!")
    else:
        print("⚠️  Need to investigate further")