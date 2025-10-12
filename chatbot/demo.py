"""
Simple demonstration script for MindWealth Chatbot.
Run this to test the chatbot functionality.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import ChatbotEngine


def main():
    """Main demonstration function."""
    
    print("\n" + "="*80)
    print("MINDWEALTH CHATBOT - DEMONSTRATION")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr set it in Python:")
        print("  import os")
        print("  os.environ['OPENAI_API_KEY'] = 'your-api-key-here'")
        return
    
    try:
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = ChatbotEngine()
        print(f"✓ Chatbot initialized with session ID: {chatbot.get_session_id()}\n")
        
        # Show available data
        print("Available data sources:")
        assets = chatbot.get_available_tickers()
        print(f"  - Assets: {len(assets)} available")
        print(f"    Examples: {', '.join(assets[:10])}...")
        
        functions = chatbot.get_available_functions()
        print(f"  - Functions: {len(functions)} available")
        print(f"    Types: {', '.join(functions)}\n")
        
        # Example 1: Simple query with stock data
        print("-" * 80)
        print("DEMO 1: Analyze single stock")
        print("-" * 80)
        
        user_query = "How many TRENDPULSE signals exist for AAPL?"
        print(f"Query: {user_query}")
        print(f"Assets: ['AAPL']")
        print(f"Functions: ['TRENDPULSE']")
        print(f"Date Range: 2025-10-01 to 2025-10-10")
        print(f"Note: Loads BOTH signal and target data automatically")
        print("\nProcessing...")
        
        response, metadata = chatbot.query(
            user_message=user_query,
            tickers=["AAPL"],
            from_date="2025-10-01",
            to_date="2025-10-10",
            functions=["TRENDPULSE"]
        )
        
        print(f"\n📊 Response:\n{response}")
        print(f"\n📈 Metadata:")
        print(f"  - Tokens used: {metadata.get('tokens_used', {}).get('total', 'N/A')}")
        print(f"  - Data loaded: {metadata.get('data_loaded', {})}")
        
        # Example 2: Follow-up question
        print("-" * 80)
        
        follow_up = "What was the stretegy sharpe ratio in that whole data for AAPL for TRENDPULSE?"
        print(f"\nFollow-up Query: {follow_up}")
        print("Processing...")
        
        # Note: For follow-up questions with same parameters, data will be reused from history
        response2, metadata2 = chatbot.query(
            user_message=follow_up,
            tickers=["AAPL"],
            from_date="2025-10-10",
            to_date="2025-10-10",
            functions=["TRENDPULSE"]  # Same parameters - will reuse data
        )
        
        if metadata2.get('data_reused_from_history'):
            print("\n✨ Note: Data reused from history (smart optimization!)")
        
        print(f"\n📊 Response:\n{response2}")
        print(f"\n📈 Tokens used: {metadata2.get('tokens_used', {}).get('total', 'N/A')}")
        
        # Demo 3: Query both signals and targets
        print("\n" + "="*80)
        print("DEMO 3: Query Both Signals and Targets for AMD")
        print("="*80 + "\n")
        
        multi_query = "What signals and targets exist for AMD? Summarize both."
        print(f"Query: {multi_query}")
        print(f"Assets: ['AMD']")
        print(f"Note: Automatically loads BOTH signal and target data")
        print("\nProcessing...")
        
        response_multi, metadata_multi = chatbot.query(
            user_message=multi_query,
            tickers=["AMD"],
            functions=None  # Load all functions from both signal and target
        )
        
        print(f"\n📊 Response (first 500 chars):\n{response_multi[:500]}...")
        print(f"\n📈 Metadata:")
        print(f"  - Tokens used: {metadata_multi['tokens_used']['total']}")
        print(f"  - Data loaded: {metadata_multi.get('data_loaded', {})}")
        print("="*80)
        
        # Show session summary
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        
        summary = chatbot.get_session_summary()
        print(f"\nSession ID: {summary['session_id']}")
        print(f"Total Messages: {summary['message_count']}")
        print(f"User Messages: {summary['user_messages']}")
        print(f"Assistant Messages: {summary['assistant_messages']}")
        
        print("\n" + "="*80)
        print("✓ DEMONSTRATION COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
