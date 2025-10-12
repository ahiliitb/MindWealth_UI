#!/usr/bin/env python3
"""
Test to verify conversation history/context is working.
This will prove that the chatbot remembers previous messages.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from chatbot import ChatbotEngine


def test_history_context():
    """Test that conversation history is maintained across queries with new structure."""
    
    print("\n" + "="*80)
    print("TESTING CONVERSATION HISTORY & CONTEXT - NEW STRUCTURE")
    print("="*80 + "\n")
    
    # Initialize chatbot
    chatbot = ChatbotEngine()
    session_id = chatbot.get_session_id()
    print(f"✓ Created session: {session_id}\n")
    
    # Show available data
    assets = chatbot.get_available_tickers()
    functions = chatbot.get_available_functions()
    print(f"Available: {len(assets)} assets, {len(functions)} functions")
    print(f"Functions: {', '.join(functions)}\n")
    
    # Query 1: Ask about AAPL with specific function
    print("-" * 80)
    print("QUERY 1: Initial question with TRENDPULSE data for AAPL")
    print("-" * 80)
    
    response1, metadata1 = chatbot.query(
        user_message="What TRENDPULSE signals exist for AAPL? Give a brief summary.",
        tickers=["AAPL"],
        from_date="2025-10-10",
        to_date="2025-10-10",
        functions=["TRENDPULSE"]
    )
    
    print(f"\nResponse 1:\n{response1}")
    print(f"\nTokens used: {metadata1['tokens_used']['total']}")
    
    # Check conversation history
    history = chatbot.get_conversation_history()
    print(f"\n✓ Messages in history: {len(history)}")
    print(f"  - System: 1")
    print(f"  - User: 1") 
    print(f"  - Assistant: 1")
    
    # Query 2: Follow-up WITHOUT providing ticker/dates/functions
    # This will test if it remembers we were talking about AAPL and TRENDPULSE
    print("\n" + "-" * 80)
    print("QUERY 2: Follow-up WITHOUT re-providing data")
    print("-" * 80)
    print("Testing: Does it remember the asset and function?")
    
    response2, metadata2 = chatbot.query(
        user_message="What asset and function were we just discussing?"
        # NO tickers, dates, functions, or data provided!
    )
    
    print(f"\nResponse 2:\n{response2}")
    print(f"\nTokens used: {metadata2['tokens_used']['total']}")
    
    # Query 3: Reference previous answer using context
    print("\n" + "-" * 80)
    print("QUERY 3: Reference your previous response")
    print("-" * 80)
    print("Testing: Can it reference its own previous answer?")
    
    response3, metadata3 = chatbot.query(
        user_message="Based on the trend you mentioned in your first response, do you think the price was going up or down?"
        # Still no data provided - testing pure context memory
    )
    
    print(f"\nResponse 3:\n{response3}")
    print(f"\nTokens used: {metadata3['tokens_used']['total']}")
    
    # Query 4: Follow-up WITH same parameters - should reuse data
    print("\n" + "-" * 80)
    print("QUERY 4: Follow-up WITH same parameters (should reuse data)")
    print("-" * 80)
    print("Testing: Smart context optimization")
    
    response4, metadata4 = chatbot.query(
        user_message="What was the signal price in that TRENDPULSE signal?",
        tickers=["AAPL"],
        from_date="2025-10-10",
        to_date="2025-10-10",
        functions=["TRENDPULSE"]  # Same as Query 1
    )
    
    if metadata4.get('data_reused_from_history'):
        print("✅ Data reused from history!")
    else:
        print("⚠️  Data was reloaded")
    
    print(f"\nResponse 4:\n{response4}")
    print(f"\nTokens used: {metadata4['tokens_used']['total']}")
    
    # Query 5: Different function - should load new data
    print("\n" + "-" * 80)
    print("QUERY 5: Different function (should load new data)")
    print("-" * 80)
    print("Testing: Different function loads fresh data")
    
    response5, metadata5 = chatbot.query(
        user_message="What FRACTAL TRACK signals exist for AAPL?",
        tickers=["AAPL"],
        from_date="2025-10-10",
        to_date="2025-10-10",
        functions=["FRACTAL TRACK"]  # Different function
    )
    
    if metadata5.get('data_reused_from_history'):
        print("⚠️  Data was reused (shouldn't happen)")
    else:
        print("✅ New data loaded for different function!")
    
    print(f"\nResponse 5 (first 200 chars):\n{response5[:200]}...")
    print(f"\nTokens used: {metadata5['tokens_used']['total']}")
    
    # Show final session summary
    print("\n" + "="*80)
    print("FINAL SESSION SUMMARY")
    print("="*80)
    
    summary = chatbot.get_session_summary()
    print(f"\nSession ID: {summary['session_id']}")
    print(f"Total Messages: {summary['message_count']}")
    print(f"  - System messages: 1")
    print(f"  - User messages: {summary['user_messages']}")
    print(f"  - Assistant messages: {summary['assistant_messages']}")
    
    print("\nQuery Summary:")
    print(f"  Query 1: TRENDPULSE data loaded - {metadata1['tokens_used']['total']} tokens")
    print(f"  Query 2: No data (context only) - {metadata2['tokens_used']['total']} tokens")
    print(f"  Query 3: No data (context only) - {metadata3['tokens_used']['total']} tokens")
    print(f"  Query 4: {'Reused ✅' if metadata4.get('data_reused_from_history') else 'Reloaded'} - {metadata4['tokens_used']['total']} tokens")
    print(f"  Query 5: FRACTAL TRACK loaded - {metadata5['tokens_used']['total']} tokens")
    
    # Show all conversation turns
    print("\n" + "-" * 80)
    print("FULL CONVERSATION FLOW:")
    print("-" * 80)
    
    history = chatbot.get_conversation_history()
    for i, msg in enumerate(history, 1):
        role = msg['role'].upper()
        content_preview = msg['content'][:100].replace('\n', ' ')
        if len(msg['content']) > 100:
            content_preview += "..."
        print(f"{i}. {role}: {content_preview}")
    
    print("\n" + "="*80)
    print("✅ HISTORY CONTEXT TEST COMPLETE")
    print("="*80)
    
    # Check the saved history file
    print(f"\n📁 History saved to: chatbot/history/{session_id}.json")
    print("   You can open this file to see the full conversation!")
    
    # Don't delete session so user can inspect it
    print(f"\n💡 Session kept for inspection. To view:")
    print(f"   cat chatbot/history/{session_id}.json | python3 -m json.tool")
    
    return True


if __name__ == "__main__":
    test_history_context()

