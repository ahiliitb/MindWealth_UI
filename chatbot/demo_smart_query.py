"""
Simple example demonstrating the two-stage smart query system.
This shows how to use the smart_query method with real data.
"""

import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.chatbot_engine import ChatbotEngine


def main():
    print("\n" + "="*80)
    print("Two-Stage Smart Query System - Simple Demo")
    print("="*80)
    
    # Check if API key is available
    from chatbot.config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        print("Please set OPENAI_API_KEY in your .env file or environment variables")
        return 1
    
    # Initialize chatbot engine
    print("\nInitializing chatbot engine...")
    engine = ChatbotEngine()
    print("✅ Engine initialized successfully")
    
    # Example 1: Query about specific asset performance
    print("\n" + "-"*80)
    print("Example 1: Asset Performance Query")
    print("-"*80)
    
    query1 = "What is the current performance of TSM?"
    print(f"\nQuery: {query1}")
    print(f"Selected signal types: ['entry']")
    print(f"Assets: ['TSM']")
    print(f"Date: 2025-10-14")
    
    print("\n⏳ Processing query...")
    response1, metadata1 = engine.smart_query(
        user_message=query1,
        selected_signal_types=["entry"],
        assets=["TSM"],
        from_date="2025-10-14",
        to_date="2025-10-14"
    )
    
    print("\n📊 Results:")
    print(f"Columns used: {metadata1.get('required_columns', [])}")
    print(f"Rows fetched: {metadata1.get('rows_fetched', 0)}")
    print(f"Column selection reasoning: {metadata1.get('column_selection_reasoning', '')}")
    print(f"\n💬 Response:")
    print(response1)
    
    # Example 2: Breadth indicators
    print("\n" + "-"*80)
    print("Example 2: Breadth Indicators Query")
    print("-"*80)
    
    query2 = "Show me the breadth indicators"
    print(f"\nQuery: {query2}")
    print(f"Selected signal types: ['breadth']")
    print(f"Date: 2025-10-14")
    
    print("\n⏳ Processing query...")
    response2, metadata2 = engine.smart_query(
        user_message=query2,
        selected_signal_types=["breadth"],
        from_date="2025-10-14"
    )
    
    print("\n📊 Results:")
    print(f"Columns used: {metadata2.get('required_columns', [])}")
    print(f"Rows fetched: {metadata2.get('rows_fetched', 0)}")
    print(f"Column selection reasoning: {metadata2.get('column_selection_reasoning', '')}")
    print(f"\n💬 Response:")
    print(response2)
    
    print("\n" + "="*80)
    print("Demo completed successfully! ✅")
    print("="*80)
    print("\nKey benefits demonstrated:")
    print("  ✓ Intelligent column selection reduces token usage")
    print("  ✓ Only relevant data is fetched from files")
    print("  ✓ Clear reasoning for what columns are needed")
    print("  ✓ Works with checkbox-style signal type selection")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
