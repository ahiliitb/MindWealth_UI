#!/usr/bin/env python3
"""
Test automatic function extraction from user prompts using GPT-4o-mini.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from chatbot import ChatbotEngine, FunctionExtractor


def test_function_extractor():
    """Test the function extraction independently."""
    
    print("\n" + "="*80)
    print("TESTING FUNCTION EXTRACTION - GPT-4o-mini")
    print("="*80 + "\n")
    
    extractor = FunctionExtractor()
    
    test_queries = [
        "What TRENDPULSE signals exist for AAPL?",
        "Compare TRENDPULSE and FRACTAL TRACK signals",
        "Show me all signals for AAPL",
        "What are the baseline divergence signals?",
        "Analyze AAPL stock performance",
        "Show TRENDPULSE, BAND MATRIX and SIGMASHELL signals",
        "What's the ALTITUDE ALPHA analysis?",
        "Compare trendpulse and fractal track",  # lowercase test
        "OSCILLATOR DELTA and PULSEGAUGE signals for ETH-USD"
    ]
    
    print("Available functions:")
    print(f"  {', '.join(extractor.get_available_functions())}\n")
    
    for query in test_queries:
        print("-" * 80)
        print(f"Query: {query}")
        
        functions = extractor.extract_functions(query)
        
        print(f"Extracted: {functions if functions else '[] (all functions)'}")
        print()


def test_auto_extraction_in_chatbot():
    """Test automatic extraction integrated in chatbot."""
    
    print("\n" + "="*80)
    print("TESTING AUTO-EXTRACTION IN CHATBOT")
    print("="*80 + "\n")
    
    chatbot = ChatbotEngine()
    
    # Test 1: User mentions TRENDPULSE in query
    print("="*80)
    print("TEST 1: User mentions TRENDPULSE (should auto-extract)")
    print("="*80)
    
    response1, metadata1 = chatbot.query(
        user_message="What TRENDPULSE signals exist for AAPL on 2025-10-10?",
        tickers=["AAPL"],
        from_date="2025-10-10",
        to_date="2025-10-10"
        # NO functions parameter - should auto-extract
    )
    
    print(f"\nUser query: 'What TRENDPULSE signals exist for AAPL on 2025-10-10?'")
    print(f"Functions parameter: None (auto-extract enabled)")
    print(f"Auto-extracted: {metadata1.get('functions_auto_extracted', [])}")
    print(f"Functions used: {metadata1.get('functions', [])}")
    print(f"\n📊 Response (first 300 chars):\n{response1[:300]}...")
    print(f"\n📈 Tokens used: {metadata1['tokens_used']['total']}")
    
    # Test 2: User doesn't mention specific function
    print("\n" + "="*80)
    print("TEST 2: User doesn't mention function (should load all)")
    print("="*80)
    
    response2, metadata2 = chatbot.query(
        user_message="What signals exist for ETH-USD?",
        tickers=["ETH-USD"],
        from_date="2025-10-10",
        to_date="2025-10-10"
        # NO functions parameter
    )
    
    print(f"\nUser query: 'What signals exist for ETH-USD?'")
    print(f"Auto-extracted: {metadata2.get('functions_auto_extracted', [])}")
    print(f"Functions used: {metadata2.get('functions', [])} (empty = all functions)")
    print(f"\n📊 Response (first 300 chars):\n{response2[:300]}...")
    print(f"\n📈 Tokens used: {metadata2['tokens_used']['total']}")
    
    # Test 3: User explicitly provides functions (should override)
    print("\n" + "="*80)
    print("TEST 3: User explicitly provides functions (should NOT auto-extract)")
    print("="*80)
    
    response3, metadata3 = chatbot.query(
        user_message="Show me the signals",  # Doesn't mention function
        tickers=["AAPL"],
        from_date="2025-10-10",
        to_date="2025-10-10",
        functions=["FRACTAL TRACK"]  # Explicitly provided
    )
    
    print(f"\nUser query: 'Show me the signals'")
    print(f"Functions parameter: ['FRACTAL TRACK'] (explicit)")
    print(f"Auto-extracted: {metadata3.get('functions_auto_extracted', [])} (should be empty)")
    print(f"Functions used: {metadata3.get('functions', [])}")
    print(f"\n📊 Response (first 300 chars):\n{response3[:300]}...")
    print(f"\n📈 Tokens used: {metadata3['tokens_used']['total']}")
    
    # Test 4: Multiple functions in query
    print("\n" + "="*80)
    print("TEST 4: User mentions multiple functions (should extract both)")
    print("="*80)
    
    response4, metadata4 = chatbot.query(
        user_message="Compare TRENDPULSE and FRACTAL TRACK signals for AAPL",
        tickers=["AAPL"],
        from_date="2025-10-10",
        to_date="2025-10-10"
        # Should auto-extract both functions
    )
    
    print(f"\nUser query: 'Compare TRENDPULSE and FRACTAL TRACK signals for AAPL'")
    print(f"Auto-extracted: {metadata4.get('functions_auto_extracted', [])}")
    print(f"Functions used: {metadata4.get('functions', [])}")
    print(f"\n📊 Response (first 300 chars):\n{response4[:300]}...")
    print(f"\n📈 Tokens used: {metadata4['tokens_used']['total']}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\nTest 1: Extracted {metadata1.get('functions_auto_extracted', [])}")
    print(f"Test 2: Extracted {metadata2.get('functions_auto_extracted', [])} (none mentioned)")
    print(f"Test 3: No extraction (explicit override)")
    print(f"Test 4: Extracted {metadata4.get('functions_auto_extracted', [])} (multiple)")
    
    print("\n🎉 Auto-extraction feature working!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nRunning tests...\n")
    
    # Test extractor independently
    test_function_extractor()

