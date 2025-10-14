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
        pass
    
    # Date range selection
    st.sidebar.subheader("Select Date Range")
    
    col1, col2 = st.sidebar.columns(2)
    
    # Set default dates (last 10 days)
    default_from_date = datetime.now() - timedelta(days=5)
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
            "Entry Signals",
            value=True,
            help="Open positions (no exit yet)"
        )
        include_exit = st.checkbox(
            "Exit Signals",
            value=False,
            help="Completed trades (with exit dates)"
        )
    
    with col_sig2:
        include_target = st.checkbox(
            "Target Achieved",
            value=False,
            help="Target price achievements (90%+ gains)"
        )
        include_breadth = st.checkbox(
            "Bullish Breadth Index",
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
    
    # Smart batch processing is always enabled
    import chatbot.config as config
    config.ENABLE_BATCH_PROCESSING = True
    
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
            logger_msg = "Settings changed - clearing backend history (chat visible)"
            st.sidebar.warning("⚠️ Settings changed - Starting fresh context (previous chat still visible)")
            # Clear backend history but keep chat visible for reference
            chatbot.clear_history()
            # Don't clear st.session_state.chat_history - keep it visible
            st.session_state.last_settings = current_settings
    else:
        st.session_state.last_settings = current_settings
    
    # Clear everything button
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear All Chat (Visible + History)"):
        chatbot.clear_history()
        st.session_state.chat_history = []
        st.session_state.last_settings = current_settings
        st.rerun()
    
    # Chat history display
    st.markdown("### 💬 Conversation")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'separator':
                # Display separator for context reset
                st.markdown("---")
                st.info(message['content'])
                st.markdown("---")
            elif message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
                    
                    # Show batch processing metadata for historical messages too
                    msg_metadata = message.get('metadata', {})
                    if msg_metadata.get('batch_processing_used'):
                        batch_mode = msg_metadata.get('batch_mode', 'unknown')
                        batch_count = msg_metadata.get('batch_count', 0)
                        tokens_used = msg_metadata.get('tokens_used', {})
                        finish_reason = msg_metadata.get('finish_reason', '')
                        
                        with st.expander("📊 Processing Details", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if batch_mode == 'single':
                                    st.metric("Batch Mode", "Single 🎯", help="All data processed in one optimized batch")
                                else:
                                    if 'synthesis' in finish_reason:
                                        st.metric("Batch Mode", f"Multi + Synthesis ✨", help=f"{batch_count} batches with AI synthesis for unified response")
                                    else:
                                        st.metric("Batch Mode", f"Multi ({batch_count}) 🔄", help="Data split across multiple batches for optimal processing")
                            
                            with col2:
                                total_tokens = tokens_used.get('total', 0)
                                st.metric("Total Tokens", f"{total_tokens:,}", help="Total tokens used (prompt + completion)")
                            
                            with col3:
                                tickers_processed = len(msg_metadata.get('tickers', []))
                                st.metric("Tickers Processed", tickers_processed, help="Number of tickers analyzed")
                            
                            # Additional info
                            if 'synthesis' in finish_reason:
                                st.caption(f"✨ Multi-batch results synthesized into single response | Total: {tokens_used.get('prompt', 0):,} prompt + {tokens_used.get('completion', 0):,} completion tokens")
                            else:
                                st.caption(f"💡 Prompt: {tokens_used.get('prompt', 0):,} tokens | Completion: {tokens_used.get('completion', 0):,} tokens")
    
    # Chat input with follow-up question mode
    st.markdown("### 💬 Ask a Question")
    
    # Follow-up mode toggle
    is_followup = st.checkbox(
        "🔗 Follow-up Question Mode",
        value=False,
        help="✅ Enabled: Ask follow-up questions using previous data context (no new data loaded)\n❌ Disabled: Start a new question with fresh data (clears history)"
    )
    
    if is_followup:
        st.info("💡 **Follow-up Mode Active**: Your question will use the previous data context. Conversation continues.")
    else:
        st.warning("🆕 **New Question Mode**: Fresh data will be loaded. Previous chat visible but context resets.")
    
    user_input = st.chat_input("Ask a question about your trading signals...")
    
    if user_input:
        # Handle follow-up vs new question
        if not is_followup:
            # New question - clear backend history but keep chat visible
            chatbot.clear_history()
            # Don't clear st.session_state.chat_history - keep previous messages visible
            
            # Add visual separator to indicate new conversation context
            if st.session_state.chat_history:  # Only if there's previous chat
                st.session_state.chat_history.append({
                    'role': 'separator',
                    'content': '--- 🆕 New Question (Fresh Context) ---'
                })
        
        # Validate configuration (only if not follow-up AND not using auto-extraction AND not breadth-only)
        # Skip validation for follow-up questions as they reuse previous data
        if not is_followup:
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
            spinner_text = "🤔 Analyzing follow-up question..." if is_followup else "🤔 Analyzing your query..."
            with st.spinner(spinner_text):
                try:
                    response, metadata = chatbot.query(
                        user_message=user_input,
                        tickers=selected_tickers if not is_followup else None,  # None if follow-up or auto-extracting
                        from_date=from_date.strftime('%Y-%m-%d') if not is_followup else None,
                        to_date=to_date.strftime('%Y-%m-%d') if not is_followup else None,
                        functions=selected_functions if not is_followup else None,
                        signal_types=selected_signal_types if not is_followup else None,  # From checkboxes (manual selection)
                        auto_extract_functions=use_auto_extract if not is_followup else False,
                        auto_extract_tickers=use_auto_extract_tickers if not is_followup else False,
                        is_followup=is_followup
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display batch processing metadata if available
                    if metadata.get('batch_processing_used'):
                        batch_mode = metadata.get('batch_mode', 'unknown')
                        batch_count = metadata.get('batch_count', 0)
                        tokens_used = metadata.get('tokens_used', {})
                        finish_reason = metadata.get('finish_reason', '')
                        
                        with st.expander("📊 Processing Details", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if batch_mode == 'single':
                                    st.metric("Batch Mode", "Single 🎯", help="All data processed in one optimized batch")
                                else:
                                    if 'synthesis' in finish_reason:
                                        st.metric("Batch Mode", f"Multi + Synthesis ✨", help=f"{batch_count} batches with AI synthesis for unified response")
                                    else:
                                        st.metric("Batch Mode", f"Multi ({batch_count}) 🔄", help="Data split across multiple batches for optimal processing")
                            
                            with col2:
                                total_tokens = tokens_used.get('total', 0)
                                st.metric("Total Tokens", f"{total_tokens:,}", help="Total tokens used (prompt + completion)")
                            
                            with col3:
                                tickers_processed = len(metadata.get('tickers', []))
                                st.metric("Tickers Processed", tickers_processed, help="Number of tickers analyzed")
                            
                            # Additional info
                            if 'synthesis' in finish_reason:
                                st.caption(f"✨ Multi-batch results synthesized into single response | Total: {tokens_used.get('prompt', 0):,} prompt + {tokens_used.get('completion', 0):,} completion tokens")
                            else:
                                st.caption(f"💡 Prompt: {tokens_used.get('prompt', 0):,} tokens | Completion: {tokens_used.get('completion', 0):,} tokens")
                    
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


if __name__ == "__main__":
    render_chatbot_page()

