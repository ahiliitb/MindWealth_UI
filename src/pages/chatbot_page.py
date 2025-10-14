"""
AI Chatbot Page for Trading Analysis
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from chatbot import ChatbotEngine


def render_chatbot_page():
    """Render the AI Chatbot page."""
    
    st.title("🤖 AI Trading Analysis Chatbot")
    st.markdown("Ask questions about your trading signals and get AI-powered insights!")
    
    # Sidebar configuration
    st.sidebar.header("📊 Query Configuration")
    
    # Initialize chatbot engine
    if 'chatbot_engine' not in st.session_state:
        try:
            st.session_state.chatbot_engine = ChatbotEngine()
            st.session_state.chat_history = []
            st.session_state.last_settings = None
        except Exception as e:
            st.error(f"❌ Failed to initialize chatbot: {e}")
            st.error("Please check:")
            st.error("1. OpenAI API key is set in .streamlit/secrets.toml")
            st.error("2. API key is valid and active")
            st.error("3. openai library is properly installed (version 1.12.0+)")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    chatbot = st.session_state.chatbot_engine
    
    # Get available tickers
    available_tickers = chatbot.get_available_tickers()
    
    # Ticker selection with auto-extraction option
    st.sidebar.subheader("Select Assets")
    
    use_auto_extract_tickers = st.sidebar.checkbox(
        "🤖 Auto-extract tickers from query",
        value=True,
        help="Let AI automatically detect ticker symbols from your question"
    )
    
    selected_tickers = None
    if not use_auto_extract_tickers:
        selected_tickers = st.sidebar.multiselect(
            "Choose one or more tickers:",
            options=sorted(available_tickers),
            default=["AAPL"] if "AAPL" in available_tickers else (available_tickers[:1] if available_tickers else []),
            help="Select the assets you want to analyze"
        )
    else:
        st.sidebar.info("🤖 Tickers will be extracted from your query automatically")
    
    # Date range selection
    st.sidebar.subheader("Select Date Range")
    
    col1, col2 = st.sidebar.columns(2)
    
    # Set default dates (last 10 days)
    default_from_date = datetime.now() - timedelta(days=10)
    default_to_date = datetime.now()
    
    with col1:
        from_date = st.date_input(
            "From Date",
            value=default_from_date,
            help="Start date for data (default: 10 days ago)"
        )
    
    with col2:
        to_date = st.date_input(
            "To Date",
            value=default_to_date,
            help="End date for data (default: today)"
        )
    
    # Signal Type selection (4 types)
    st.sidebar.subheader("Select Signal Types")
    
    col_sig1, col_sig2 = st.sidebar.columns(2)
    
    with col_sig1:
        include_entry = st.checkbox(
            "🔵 Entry Signals",
            value=True,
            help="Open positions (no exit yet)"
        )
        include_exit = st.checkbox(
            "🟢 Exit Signals",
            value=True,
            help="Completed trades (with exit dates)"
        )
    
    with col_sig2:
        include_target = st.checkbox(
            "🎯 Target Achievements",
            value=True,
            help="Target price achievements (90%+ gains)"
        )
        include_breadth = st.checkbox(
            "📊 Market Breadth",
            value=False,
            help="Market-wide sentiment analysis"
        )
    
    # Build signal_types list
    selected_signal_types = []
    if include_entry:
        selected_signal_types.append("entry")
    if include_exit:
        selected_signal_types.append("exit")
    if include_target:
        selected_signal_types.append("target")
    if include_breadth:
        selected_signal_types.append("breadth")
    
    # If nothing selected, include entry/exit/target (not breadth by default)
    if not selected_signal_types:
        selected_signal_types = ["entry", "exit", "target"]
    
    # Function selection (optional)
    st.sidebar.subheader("Function Filter")
    available_functions = chatbot.get_available_functions()
    
    use_auto_extract = st.sidebar.checkbox(
        "🤖 Auto-extract functions from query",
        value=True,
        help="Let AI automatically detect function names from your question"
    )
    
    selected_functions = None
    if not use_auto_extract:
        selected_functions = st.sidebar.multiselect(
            "Choose functions:",
            options=sorted(available_functions),
            default=[],
            help="Leave empty to load all functions"
        )
        if not selected_functions:
            selected_functions = None
    
    # Advanced settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Settings")
    
    enable_batch = st.sidebar.checkbox(
        "🔄 Enable batch processing for large queries",
        value=False,
        help="Enable for comprehensive analysis of many tickers (slower but more complete)"
    )
    
    # Update chatbot's batch processing setting
    import chatbot.config as config
    config.ENABLE_BATCH_PROCESSING = enable_batch
    
    if enable_batch:
        st.sidebar.warning("⏱️ Batch mode: Queries may take several minutes for comprehensive analysis")
    else:
        st.sidebar.info("⚡ Fast mode: Limited to 15 tickers for quick responses")
    
    # Check if settings have changed - if yes, clear history
    current_settings = {
        'tickers': tuple(sorted(selected_tickers)) if selected_tickers else None,
        'from_date': from_date.strftime('%Y-%m-%d'),
        'to_date': to_date.strftime('%Y-%m-%d'),
        'signal_types': tuple(sorted(selected_signal_types)) if selected_signal_types else None,
        'functions': tuple(sorted(selected_functions)) if selected_functions else None
    }
    
    if st.session_state.last_settings is not None:
        if current_settings != st.session_state.last_settings:
            logger_msg = "Settings changed - clearing chat history"
            st.sidebar.info("⚠️ Settings changed - chat history cleared")
            chatbot.clear_history()
            st.session_state.chat_history = []
            st.session_state.last_settings = current_settings
    else:
        st.session_state.last_settings = current_settings
    
    # Clear history button
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat History"):
        chatbot.clear_history()
        st.session_state.chat_history = []
        st.session_state.last_settings = current_settings
        st.rerun()
    
    # Main chat area
    st.markdown("---")
    
    # Display current configuration
    with st.expander("📋 Current Configuration", expanded=False):
        st.write(f"**Mode:** {'🔄 Batch Processing (Comprehensive)' if enable_batch else '⚡ Fast Mode (Limited to 15 tickers)'}")
        st.write(f"**Auto Ticker Extraction:** {'Enabled 🤖' if use_auto_extract_tickers else 'Disabled'}")
        if not use_auto_extract_tickers:
            st.write(f"**Selected Tickers:** {', '.join(selected_tickers) if selected_tickers else 'None'}")
        st.write(f"**Date Range:** {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
        st.write(f"**Signal Types:** {', '.join(selected_signal_types) if selected_signal_types else 'All'}")
        st.write(f"**Auto Function Extraction:** {'Enabled 🤖' if use_auto_extract else 'Disabled'}")
        if selected_functions and not use_auto_extract:
            st.write(f"**Selected Functions:** {', '.join(selected_functions)}")
        st.write(f"**Session ID:** {chatbot.get_session_id()}")
    
    # Chat history display
    st.markdown("### 💬 Conversation")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your trading signals...")
    
    if user_input:
        # Validate configuration (only if not using auto-extraction AND not breadth-only)
        # If only breadth is selected, we don't need tickers
        breadth_only = selected_signal_types == ['breadth']
        if not use_auto_extract_tickers and not selected_tickers and not breadth_only:
            st.error("⚠️ Please select at least one ticker or enable auto-extraction!")
            st.stop()
        
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response, metadata = chatbot.query(
                        user_message=user_input,
                        tickers=selected_tickers,  # None if auto-extracting
                        from_date=from_date.strftime('%Y-%m-%d'),
                        to_date=to_date.strftime('%Y-%m-%d'),
                        functions=selected_functions,
                        signal_types=selected_signal_types,  # From checkboxes (manual selection)
                        auto_extract_functions=use_auto_extract,
                        auto_extract_tickers=use_auto_extract_tickers
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'metadata': metadata
                    })
                    
                    # Rerun to update chat display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>💡 <b>Tips:</b></p>
        <ul style='list-style: none; padding: 0;'>
            <li>• Enable auto-extraction for full natural language: "What signals exist for AAPL?"</li>
            <li>• Ask about specific signals: "What TRENDPULSE signals exist for AAPL?"</li>
            <li>• Compare functions: "Compare FRACTAL TRACK and BASELINEDIVERGENCE for MSFT"</li>
            <li>• Ask follow-up questions - the chatbot remembers context!</li>
            <li>• 🤖 Auto-extraction uses GPT-4o-mini to understand your intent</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_chatbot_page()

