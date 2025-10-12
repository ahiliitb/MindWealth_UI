"""
Example usage of the chatbot engine.
This file demonstrates how to use the chatbot functionality.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from chatbot import ChatbotEngine

def example_1_basic_query():
    """Example 1: Basic query with four parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Query with Four Parameters")
    print("="*80)
    
    # Initialize chatbot
    chatbot = ChatbotEngine()
    
    # Query with tickers, dates, and functions
    response, metadata = chatbot.query(
        user_message="What TRENDPULSE signals exist for AAPL? Summarize.",
        tickers=["AAPL"],              # Assets
        from_date="2025-10-10",        # From date
        to_date="2025-10-10",          # To date
        functions=["TRENDPULSE"]       # Functions
    )
    
    print(f"\nSession ID: {chatbot.get_session_id()}")
    print(f"\nResponse:\n{response}")
    print(f"\nTokens used: {metadata.get('tokens_used', {})}")


def example_2_multiple_assets_and_functions():
    """Example 2: Compare multiple assets with specific function"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Multiple Assets with Specific Function")
    print("="*80)
    
    chatbot = ChatbotEngine()
    
    response, metadata = chatbot.query(
        user_message="Compare TRENDPULSE signals for AAPL and ETH-USD. Which has better metrics?",
        tickers=["AAPL", "ETH-USD"],       # Multiple assets
        from_date="2025-10-10",            # From date
        to_date="2025-10-10",              # To date
        functions=["TRENDPULSE"]           # Specific function
    )
    
    print(f"\nResponse:\n{response}")


def example_3_multiple_functions():
    """Example 3: Compare multiple functions for same asset"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Multiple Functions for Same Asset")
    print("="*80)
    
    chatbot = ChatbotEngine()
    
    response, metadata = chatbot.query(
        user_message="Compare TRENDPULSE and FRACTAL TRACK signals for AAPL. Which is more reliable?",
        tickers=["AAPL"],                          # Single asset
        from_date="2025-10-10",                    # From date
        to_date="2025-10-10",                      # To date
        functions=["TRENDPULSE", "FRACTAL TRACK"] # Multiple functions
    )
    
    print(f"\nResponse:\n{response}")


def example_4_conversation_context():
    """Example 4: Multi-turn conversation with smart context reuse"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-turn Conversation with Context Reuse")
    print("="*80)
    
    chatbot = ChatbotEngine()
    
    # First query
    response1, metadata1 = chatbot.query(
        user_message="What BASELINEDIVERGENCE signals exist for HDFCBANK.NS?",
        tickers=["HDFCBANK.NS"],
        from_date="2025-10-10",
        to_date="2025-10-10",
        functions=["BASELINEDIVERGENCE"]
    )
    print(f"\nFirst Response (truncated):\n{response1[:300]}...\n")
    print(f"Tokens: {metadata1['tokens_used']['total']}")
    
    # Follow-up query with same parameters (will reuse data)
    response2, metadata2 = chatbot.query(
        user_message="What is the win rate of that signal?",
        tickers=["HDFCBANK.NS"],           # Same parameters
        from_date="2025-10-10",
        to_date="2025-10-10",
        functions=["BASELINEDIVERGENCE"]
    )
    print(f"\nFollow-up Response:\n{response2}\n")
    
    if metadata2.get('data_reused_from_history'):
        print("✨ Data reused from history (optimization working!)")
    
    print(f"Tokens: {metadata2['tokens_used']['total']}")
    
    # Check session summary
    summary = chatbot.get_session_summary()
    print(f"\nSession Summary:\n{summary}")


def example_5_csv_text_input():
    """Example 5: Query with raw CSV text"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Query with Raw CSV Text")
    print("="*80)
    
    chatbot = ChatbotEngine()
    
    # Example CSV text
    csv_text = """Date,Close,Volume
2024-01-01,150.25,1000000
2024-01-02,152.30,1200000
2024-01-03,148.50,1500000
2024-01-04,155.00,1800000"""
    
    response, metadata = chatbot.query_with_csv_text(
        user_message="What is the trend in this data?",
        csv_text=csv_text,
        additional_context="This is sample stock data for analysis"
    )
    
    print(f"\nResponse:\n{response}")


def example_6_continue_session():
    """Example 6: Continue an existing session"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Continue Existing Session")
    print("="*80)
    
    # Create new session
    chatbot1 = ChatbotEngine()
    session_id = chatbot1.get_session_id()
    
    chatbot1.query(
        user_message="Tell me about NFLX stock",
        tickers=["NFLX"],
        from_date="2024-01-01",
        to_date="2024-12-31"
    )
    
    print(f"\nCreated session: {session_id}")
    
    # Later, continue the same session
    chatbot2 = ChatbotEngine(session_id=session_id)
    response, _ = chatbot2.query(
        user_message="What was the highest price in that period?"
    )
    
    print(f"\nContinued session response:\n{response}")
    
    # Show full conversation history
    history = chatbot2.get_conversation_history()
    print(f"\nFull conversation has {len(history)} messages")


def example_7_available_data():
    """Example 7: Check available assets, functions, and data files"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Available Data Sources")
    print("="*80)
    
    chatbot = ChatbotEngine()
    
    # Get available assets
    assets = chatbot.get_available_tickers()
    print(f"\nTotal available assets: {len(assets)}")
    print(f"Sample assets: {assets[:10]}")
    
    # Get available functions
    functions = chatbot.get_available_functions()
    print(f"\nAvailable functions: {len(functions)}")
    print(f"Functions: {functions}")
    
    # Get functions for specific asset
    if assets:
        asset_functions = chatbot.get_available_functions(assets[0])
        print(f"\nFunctions for {assets[0]}: {asset_functions}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MINDWEALTH CHATBOT - USAGE EXAMPLES")
    print("="*80)
    print("\nNote: Set OPENAI_API_KEY environment variable before running")
    print("Example: export OPENAI_API_KEY='your-key-here'")
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nShowing examples anyway (will fail when executed)...\n")
    
    try:
        # Run examples
        # Uncomment the examples you want to run
        
        # example_1_basic_query()
        # example_2_multiple_assets_and_functions()
        # example_3_multiple_functions()
        # example_4_conversation_context()
        # example_5_csv_text_input()
        # example_6_continue_session()
        example_7_available_data()
        
        print("\n" + "="*80)
        print("Examples completed!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure OPENAI_API_KEY is set and valid.")

