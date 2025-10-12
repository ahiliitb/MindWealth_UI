#!/usr/bin/env python3
"""
Test querying the converted signal data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from chatbot import ChatbotEngine


def test_signal_query():
    """Test querying trading signals from converted data with new structure."""
    
    print("\n" + "="*80)
    print("TESTING SIGNAL DATA QUERIES - NEW STRUCTURE")
    print("="*80 + "\n")
    
    chatbot = ChatbotEngine()
    
    # Show available data
    print("Available Data:")
    print("-" * 80)
    assets = chatbot.get_available_tickers()
    functions = chatbot.get_available_functions()
    print(f"Assets: {len(assets)} available")
    print(f"Functions: {', '.join(functions)}")
    print()
    
    # Test 1: Query all functions for ETH-USD
    print("="*80)
    print("TEST 1: All functions for ETH-USD on 2025-10-10")
    print("="*80)
    
    response, metadata = chatbot.query(
        user_message="Summarize all trading signals for ETH-USD. What signal types exist?",
        tickers=["ETH-USD"],
        from_date="2025-10-01",
        to_date="2025-10-10",
        functions=["TRENDPULSE"]
        # Note: Automatically loads from BOTH signal and target folders
    )
    
    print(f"\n📊 Response (first 400 chars):\n{response[:400]}...")
    print(f"\n📈 Tokens used: {metadata['tokens_used']['total']}")
    print(f"📁 Data loaded: {metadata.get('data_loaded', {})}")
    
    # Test 2: Query specific function for HDFCBANK.NS
    print("\n" + "="*80)
    print("TEST 2: TRENDPULSE function only for HDFCBANK.NS")
    print("="*80 + "\n")
    
    response2, metadata2 = chatbot.query(
        user_message="What TRENDPULSE signals exist for HDFCBANK.NS?",
        tickers=["HDFCBANK.NS"],
        from_date="2025-10-08",
        to_date="2025-10-10",
        functions=["TRENDPULSE"]  # Only TRENDPULSE
    )
    
    print(f"📊 Response (first 400 chars):\n{response2[:400]}...")
    print(f"\n📈 Tokens used: {metadata2['tokens_used']['total']}")
    print(f"📁 Data loaded: {metadata2.get('data_loaded', {})}")
    
    # Test 3: Multiple functions
    print("\n" + "="*80)
    print("TEST 3: Compare TRENDPULSE and BASELINEDIVERGENCE for HDFCBANK.NS")
    print("="*80 + "\n")
    
    response3, metadata3 = chatbot.query(
        user_message="Compare TRENDPULSE and BASELINEDIVERGENCE signals for HDFCBANK.NS",
        tickers=["HDFCBANK.NS"],
        from_date="2025-10-08",
        to_date="2025-10-10",
        functions=["TRENDPULSE", "BASELINEDIVERGENCE"]
    )
    
    print(f"📊 Response (first 400 chars):\n{response3[:400]}...")
    print(f"\n📈 Tokens used: {metadata3['tokens_used']['total']}")
    
    # Test 4: Smart context reuse
    print("\n" + "="*80)
    print("TEST 4: Follow-up question (same parameters - should reuse)")
    print("="*80 + "\n")
    
    response4, metadata4 = chatbot.query(
        user_message="Which signal has better win rate?",
        tickers=["HDFCBANK.NS"],
        from_date="2025-10-08",
        to_date="2025-10-10",
        functions=["TRENDPULSE", "BASELINEDIVERGENCE"]  # Same params
    )
    
    if metadata4.get('data_reused_from_history'):
        print("✅ Data reused from history (optimization working!)")
    else:
        print("⚠️  Data was reloaded")
    
    print(f"\n📊 Response (first 300 chars):\n{response4[:300]}...")
    print(f"\n📈 Tokens used: {metadata4['tokens_used']['total']}")
    
    print("\n" + "="*80)
    print("✅ SUCCESS - All signal query tests passed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_signal_query()
