"""
AI Chatbot Page for Trading Analysis
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from chatbot import ChatbotEngine, SessionManager
from chatbot.signal_type_selector import SIGNAL_TYPE_DESCRIPTIONS, DEFAULT_SIGNAL_TYPES
from chatbot.config import MAX_CHATS_DISPLAY

def get_signal_type_label(signal_type: str, uppercase: bool = False) -> str:
    """Return a user-facing label for a signal type key."""
    title = SIGNAL_TYPE_DESCRIPTIONS.get(signal_type, (signal_type.replace("_", " ").title(), ""))[0]
    return title.upper() if uppercase else title


def extract_user_prompt(content: str, metadata: Optional[dict] = None) -> str:
    """Return the original user prompt without appended data payloads."""
    if metadata and metadata.get("display_prompt"):
        return metadata["display_prompt"]
    
    cleaned = content or ""
    
    if 'FOLLOW-UP QUESTION:' in cleaned:
        cleaned = cleaned.split('FOLLOW-UP QUESTION:', 1)[1].strip()
    elif 'User Query:' in cleaned:
        cleaned = cleaned.split('User Query:', 1)[1].strip()
    
    if '===' in cleaned:
        cleaned = cleaned.split('===', 1)[0].strip()
    
    return cleaned


def apply_table_styling():
    """Apply custom CSS styling for larger table fonts."""
    st.markdown("""
    <style>
    /* Enhanced CSS for larger table fonts with comprehensive targeting */
    
    /* Primary dataframe container targeting */
    .stDataFrame {
        font-size: 16px !important;
    }
    
    [data-testid="stDataFrame"] {
        font-size: 16px !important;
    }
    
    /* Target AG Grid components (Streamlit's dataframe implementation) */
    .ag-root-wrapper {
        font-size: 16px !important;
    }
    
    .ag-header {
        font-size: 17px !important;
        font-weight: 600 !important;
    }
    
    .ag-header-cell-text {
        font-size: 17px !important;
        font-weight: 600 !important;
    }
    
    .ag-cell {
        font-size: 16px !important;
        padding: 10px 12px !important;
        line-height: 1.4 !important;
    }
    
    .ag-cell-value {
        font-size: 16px !important;
    }
    
    /* Target table elements within dataframes */
    .stDataFrame table,
    [data-testid="stDataFrame"] table {
        font-size: 16px !important;
    }
    
    .stDataFrame th,
    [data-testid="stDataFrame"] th {
        font-size: 17px !important;
        font-weight: 600 !important;
        padding: 12px 14px !important;
    }
    
    .stDataFrame td,
    [data-testid="stDataFrame"] td {
        font-size: 16px !important;
        padding: 10px 14px !important;
        line-height: 1.4 !important;
    }
    
    /* Target all text elements within dataframe containers */
    .stDataFrame *,
    [data-testid="stDataFrame"] * {
        font-size: 16px !important;
    }
    
    /* Additional comprehensive targeting */
    .element-container div[data-testid="stDataFrame"] * {
        font-size: 16px !important;
    }
    
    /* Streamlit specific dataframe elements */
    .streamlit-expanderHeader {
        font-size: 16px !important;
    }
    
    /* Style for dataframe in expander */
    .streamlit-expander .stDataFrame {
        font-size: 16px !important;
    }
    
    /* Override any inherited smaller font sizes */
    .stApp .stDataFrame {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def display_styled_dataframe(df, height=400, key_suffix=""):
    """Display dataframe with enhanced styling and larger fonts."""
    # Apply additional styling through column configuration
    column_config = {}
    for col in df.columns:
        column_config[col] = st.column_config.TextColumn(
            col,
            width="medium",
            help=None
        )
    
    # Display with enhanced parameters
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        height=min(height, (len(df) + 1) * 40),  # Slightly larger row height for better readability
        column_config=column_config,
        key=f"styled_df_{key_suffix}_{hash(str(df.shape))}"  # Unique key
    )


def _coerce_to_dataframe(data: Any) -> Optional[pd.DataFrame]:
    """Best-effort conversion of legacy signal tables to pandas DataFrame."""
    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        return data

    try:
        if isinstance(data, list):
            # Empty list -> empty DataFrame
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # Dict of iterables (column mapping) or a single record
            if any(isinstance(v, (list, tuple, set)) for v in data.values()):
                return pd.DataFrame(data)
            return pd.DataFrame([data])
    except Exception:
        return None

    return None


def render_chat_history_sidebar():
    """Render the chat history sidebar for managing sessions."""
    st.sidebar.title("💬 Chat History")
    
    # New Chat button at the top
    if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
        # Create new session
        new_session_id = SessionManager.create_new_session()
        st.session_state.current_session_id = new_session_id
        st.session_state.chatbot_engine = None  # Will be recreated with new session
        st.session_state.chat_history = []
        st.session_state.last_settings = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Search box
    search_query = st.sidebar.text_input("🔍 Search chats", placeholder="Type to search...")
    
    # Get sessions
    if search_query:
        sessions = SessionManager.search_sessions(search_query)
    else:
        sessions = SessionManager.list_all_sessions(sort_by='last_updated')
    
    # Limit displayed sessions to MAX_CHATS_DISPLAY (unless searching)
    total_sessions = len(sessions)
    if not search_query and total_sessions > MAX_CHATS_DISPLAY:
        sessions = sessions[:MAX_CHATS_DISPLAY]
        showing_limited = True
    else:
        showing_limited = False
    
    # Display sessions (compact, no chat number/time, use preview as title)
    if not sessions:
        st.sidebar.info("No chat history yet. Start a new conversation!")
    else:
        for session in sessions:
            session_id = session['session_id']
            # Use preview (first user message) as the display title, fallback to title
            preview = session.get('preview', '').strip()
            display_title = preview if preview else session.get('title', 'New Chat')
            # Only show first 6 words for compactness
            display_title = ' '.join(display_title.split()[:6])
            if len(display_title) < len(preview):
                display_title += '...'
            is_current = st.session_state.get('current_session_id') == session_id
            # Compact row: title + rename + delete
            # Use two columns: title (wide) and icons (narrow, side-by-side)
            cols = st.sidebar.columns([8, 2], gap="small")
            with cols[0]:
                if st.button(f"{'🟢 ' if is_current else ''}{display_title}", key=f"load_{session_id}", use_container_width=True, disabled=is_current):
                    st.session_state.current_session_id = session_id
                    st.session_state.chatbot_engine = None
                    st.session_state.chat_history = []
                    st.session_state.last_settings = None
                    st.rerun()
            with cols[1]:
                icon_cols = st.columns([1, 1], gap="small")
                with icon_cols[0]:
                    if st.button("✏️", key=f"rename_{session_id}", help="Rename"):
                        st.session_state[f'renaming_{session_id}'] = True
                        st.rerun()
                with icon_cols[1]:
                    if st.button("🗑️", key=f"delete_{session_id}", help="Delete"):
                        if not is_current or len(sessions) > 1:
                            SessionManager.delete_session(session_id)
                            if is_current:
                                remaining = [s for s in sessions if s['session_id'] != session_id]
                                if remaining:
                                    st.session_state.current_session_id = remaining[0]['session_id']
                                    st.session_state.chatbot_engine = None
                                    st.session_state.chat_history = []
                                    st.session_state.last_settings = None
                            st.rerun()
            # Inline rename input (compact)
            if st.session_state.get(f'renaming_{session_id}', False):
                new_title = st.text_input("Rename chat:", value=display_title, key=f"rename_input_{session_id}")
                col_save, col_cancel = st.columns([1,1], gap="small")
                with col_save:
                    if st.button("✅", key=f"save_rename_{session_id}"):
                        SessionManager.update_session_title(session_id, new_title)
                        st.session_state[f'renaming_{session_id}'] = False
                        st.rerun()
                with col_cancel:
                    if st.button("❌", key=f"cancel_rename_{session_id}"):
                        st.session_state[f'renaming_{session_id}'] = False
                        st.rerun()


def render_chatbot_page():
    """Render the AI Chatbot page."""
    
    st.title("🤖 AI Trading Analysis Chatbot")
    st.markdown("Ask questions about your trading signals and get AI-powered insights!")
    
    # Apply custom styling for larger table fonts
    apply_table_styling()
    
    # Initialize current session if not exists
    if 'current_session_id' not in st.session_state:
        # Check if there are existing sessions
        existing_sessions = SessionManager.list_all_sessions()
        if existing_sessions:
            # Use the most recent session
            st.session_state.current_session_id = existing_sessions[0]['session_id']
        else:
            # Create a new session
            st.session_state.current_session_id = SessionManager.create_new_session()
    
    # Initialize chatbot engine with current session
    if 'chatbot_engine' not in st.session_state or st.session_state.chatbot_engine is None:
        try:
            st.session_state.chatbot_engine = ChatbotEngine(
                session_id=st.session_state.current_session_id
            )
            # Load chat history from the session
            history_manager = st.session_state.chatbot_engine.history_manager
            st.session_state.chat_history = []
            
            # Convert history to chat format - clean user messages for display
            for msg in history_manager.get_full_history():
                if msg['role'] in ['user', 'assistant']:
                    metadata = msg.get('metadata', {}) or {}
                    content = msg['content']
                    if msg['role'] == 'user':
                        content = extract_user_prompt(content, metadata)
                    
                    st.session_state.chat_history.append({
                        'role': msg['role'],
                        'content': content,
                        'metadata': metadata
                    })
            
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

    # Initialize signal type session defaults
    if 'last_signal_types' not in st.session_state:
        st.session_state.last_signal_types = DEFAULT_SIGNAL_TYPES.copy()
    if 'last_signal_reason' not in st.session_state:
        st.session_state.last_signal_reason = "Default selection: entry, exit, target."
    
    # --- SIDEBAR QUERY CONFIGURATION ---
    st.sidebar.header("📊 Query Configuration")
    
    # Auto-extraction is always enabled (no manual selection)
    use_auto_extract_tickers = True
    selected_tickers = None
    
    # Date range selection
    st.sidebar.subheader("Select Date Range")
    
    col1, col2 = st.sidebar.columns(2)
    
    # Set default dates (last 5 days)
    default_from_date = datetime.now() - timedelta(days=5)
    default_to_date = datetime.now()
    
    with col1:
        from_date = st.date_input(
            "From Date",
            value=default_from_date,
            help="Start date for data (default: 5 days ago)"
        )
    
    with col2:
        to_date = st.date_input(
            "To Date",
            value=default_to_date,
            help="End date for data (default: today)"
        )
    
    # Signal Type selection is AI-driven
    st.sidebar.subheader("Signal Types (auto-selected)")
    st.sidebar.caption("The assistant reads your question and chooses the relevant signal categories.")
    
    signal_selection_placeholder = st.sidebar.empty()
    
    def render_signal_selection(selected, reasoning):
        with signal_selection_placeholder.container():
            selection_text = ", ".join(get_signal_type_label(sig) for sig in selected) if selected else "None"
            st.markdown(f"**AI Selection:** {selection_text}")
            if reasoning:
                st.caption(f"💡 {reasoning}")
    
    last_signal_types = st.session_state.get("last_signal_types", DEFAULT_SIGNAL_TYPES)
    last_signal_reason = st.session_state.get("last_signal_reason", "")
    render_signal_selection(last_signal_types, last_signal_reason)
    
    with st.sidebar.expander("Available Signal Types", expanded=False):
        for key, (title, description) in SIGNAL_TYPE_DESCRIPTIONS.items():
            st.markdown(f"**{title}**")
            st.markdown(description)
    
    # Auto-extract functions is always enabled (no manual selection)
    use_auto_extract = True
    selected_functions = None
    selected_signal_types = list(last_signal_types)
    
    # Smart batch processing is always enabled
    import chatbot.config as config
    config.ENABLE_BATCH_PROCESSING = True
    
    # Check if settings have changed - if yes, clear history
    current_settings = {
        'tickers': tuple(sorted(selected_tickers)) if selected_tickers else None,
        'from_date': from_date.strftime('%Y-%m-%d'),
        'to_date': to_date.strftime('%Y-%m-%d'),
        'functions': tuple(sorted(selected_functions)) if selected_functions else None
    }
    
    # Clear everything button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Current Chat"):
        chatbot.clear_history()
        # Update title to "New Chat"
        chatbot.history_manager.update_session_title("New Chat")
        st.session_state.chat_history = []
        st.session_state.last_settings = current_settings
        st.session_state.last_signal_types = DEFAULT_SIGNAL_TYPES.copy()
        st.session_state.last_signal_reason = "Default selection: entry, exit, target."
        st.rerun()
    
    # Now render chat history sidebar AFTER query configuration
    st.sidebar.markdown("---")
    render_chat_history_sidebar()
    
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

    # --- CHAT HISTORY UI ---
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
                    
                    # Display full signal tables if available
                    msg_metadata = message.get('metadata', {})
                    full_signal_tables = msg_metadata.get('full_signal_tables', {})
                    
                    if full_signal_tables:
                        st.markdown("### 📊 Complete Signal Data Used in Analysis")
                        
                        # Display each signal type in separate sections
                        for signal_type, signal_df in full_signal_tables.items():
                            if not signal_df.empty:
                                st.markdown(f"#### {get_signal_type_label(signal_type, uppercase=True)} Signals ({len(signal_df)} records)")
                                
                                # Display the complete table with all columns and enhanced styling
                                display_styled_dataframe(
                                    signal_df, 
                                    height=min(350, (len(signal_df) + 1) * 40),  # Adaptive height for history
                                    key_suffix=f"history_{signal_type}"
                                )
                                
                                # Show summary info (same as new responses)
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Records", len(signal_df))
                                with col2:
                                    try:
                                        # Try to extract symbols from various columns
                                        symbols_found = set()
                                        for col in signal_df.columns:
                                            if any(keyword in col.lower() for keyword in ['symbol', 'asset']):
                                                symbols = signal_df[col].astype(str).str.extract(r'([A-Z]{2,5})').dropna()
                                                symbols_found.update(symbols.iloc[:, 0].tolist() if not symbols.empty else [])
                                        unique_symbols = len(symbols_found) if symbols_found else "N/A"
                                    except:
                                        unique_symbols = "N/A"
                                    st.metric("Unique Symbols", unique_symbols)
                                with col3:
                                    st.metric("Total Columns", len(signal_df.columns))
                    
                    else:
                        # Fallback to legacy simple table
                        signals_df = _coerce_to_dataframe(msg_metadata.get('signals_table'))
                        if signals_df is not None and not signals_df.empty:
                            with st.expander(f"Signals Referenced ({len(signals_df)} signals)", expanded=False):
                                # Use styled dataframe for legacy tables too
                                display_styled_dataframe(
                                    signals_df,
                                    height=300,
                                    key_suffix="legacy"
                                )
                    
                    # Show metadata
                    # Check if it's a smart query
                    if msg_metadata.get('input_type') == 'smart_query':
                        with st.expander("📊 Smart Query Details", expanded=False):
                            signal_types_meta = msg_metadata.get('selected_signal_types', [])
                            signal_reason_meta = msg_metadata.get('signal_type_reasoning', '')
                            if signal_types_meta:
                                st.markdown(f"**AI Signal Types:** {', '.join(get_signal_type_label(sig) for sig in signal_types_meta)}")
                                if signal_reason_meta:
                                    st.caption(f"💡 {signal_reason_meta}")
                            # Show column selection per signal type
                            st.subheader("🎯 Column Selection by Signal Type")
                            columns_by_type = msg_metadata.get('columns_by_signal_type', {})
                            reasoning_by_type = msg_metadata.get('reasoning_by_signal_type', {})
                            for signal_type in msg_metadata.get('selected_signal_types', []):
                                if signal_type in columns_by_type:
                                    cols = columns_by_type[signal_type]
                                    reasoning = reasoning_by_type.get(signal_type, '')
                                    st.markdown(f"**{get_signal_type_label(signal_type, uppercase=True)}** ({len(cols)} columns)")
                                    st.caption(f"💡 {reasoning}")
                                    with st.expander(f"View {signal_type} columns"):
                                        for col in cols:
                                            st.text(f"  • {col}")
                            
                            # Show data statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Rows Fetched", msg_metadata.get('rows_fetched', 0))
                            with col2:
                                st.metric("Signal Types", len(msg_metadata.get('signal_types_with_data', [])))
                            with col3:
                                total_tokens = msg_metadata.get('tokens_used', {}).get('total', 0)
                                st.metric("Tokens Used", f"{total_tokens:,}")
                    
                    # Show batch processing metadata for old query() method
                    elif msg_metadata.get('batch_processing_used'):
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
    
    # Chat input
    st.markdown("### 💬 Ask a Question")
    
    user_input = st.chat_input("Ask a question about your trading signals...")
    
    if user_input:
        ai_signal_types, ai_reason = chatbot.determine_signal_types(user_input)
        if not ai_signal_types:
            ai_signal_types = DEFAULT_SIGNAL_TYPES.copy()
        selected_signal_types = list(ai_signal_types)
        st.session_state.last_signal_types = selected_signal_types
        st.session_state.last_signal_reason = ai_reason
        render_signal_selection(selected_signal_types, ai_reason)
        
        # Add user message to history (store clean user input for UI display)
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'metadata': {'display_prompt': user_input}
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            selection_text = ", ".join(get_signal_type_label(sig) for sig in selected_signal_types)
            st.markdown(f"**AI Signal Type Selection:** {selection_text}")
            if ai_reason:
                st.caption(f"💡 {ai_reason}")
            with st.spinner("🤔 Analyzing your query with conversation context..."):
                try:
                    # Always use smart follow-up query to maintain conversation context (like ChatGPT)
                    response, metadata = chatbot.smart_followup_query(
                        user_message=user_input,
                        selected_signal_types=selected_signal_types,
                        assets=selected_tickers if not use_auto_extract_tickers else None,
                        from_date=from_date.strftime('%Y-%m-%d'),
                        to_date=to_date.strftime('%Y-%m-%d'),
                        functions=selected_functions if not use_auto_extract else None,
                        auto_extract_tickers=use_auto_extract_tickers,
                        signal_type_reasoning=ai_reason
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display full signal tables with all columns if available
                    full_signal_tables = metadata.get('full_signal_tables', {})
                    if full_signal_tables:
                        st.markdown("### 📊 Complete Signal Data Used in Analysis")
                        
                        # Display each signal type in separate sections
                        for signal_type, signal_df in full_signal_tables.items():
                            if not signal_df.empty:
                                st.markdown(f"#### {get_signal_type_label(signal_type, uppercase=True)} Signals ({len(signal_df)} records)")
                                
                                # Display the complete table with all columns and enhanced styling
                                display_styled_dataframe(
                                    signal_df, 
                                    height=min(400, (len(signal_df) + 1) * 40),  # Adaptive height for new responses
                                    key_suffix=f"new_{signal_type}"
                                )
                                
                                # Show summary info
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Records", len(signal_df))
                                with col2:
                                    try:
                                        # Try to extract symbols from various columns
                                        symbols_found = set()
                                        for col in signal_df.columns:
                                            if any(keyword in col.lower() for keyword in ['symbol', 'asset']):
                                                symbols = signal_df[col].astype(str).str.extract(r'([A-Z]{2,5})').dropna()
                                                symbols_found.update(symbols.iloc[:, 0].tolist() if not symbols.empty else [])
                                        unique_symbols = len(symbols_found) if symbols_found else "N/A"
                                    except:
                                        unique_symbols = "N/A"
                                    st.metric("Unique Symbols", unique_symbols)
                                with col3:
                                    st.metric("Total Columns", len(signal_df.columns))
                    
                    else:
                        # Fallback to legacy simple table if full tables not available
                        signals_df = _coerce_to_dataframe(metadata.get('signals_table'))
                        if signals_df is not None and not signals_df.empty:
                            st.markdown("### 📊 Signals Referenced in Analysis")
                            st.dataframe(
                                signals_df,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                                    "Function": st.column_config.TextColumn("Function", width="medium"),
                                    "Signal_Type": st.column_config.TextColumn("Signal Type", width="small"),
                                    "Signal_Direction": st.column_config.TextColumn("Direction", width="small"),
                                    "Signal_Date": st.column_config.DateColumn("Signal Date", width="medium"),
                                    "Price": st.column_config.TextColumn("Price", width="small"),
                                    "Interval": st.column_config.TextColumn("Interval", width="small"),
                                    "Status": st.column_config.TextColumn("Status", width="small")
                                }
                            )
                            st.caption(f"📈 {len(signals_df)} signal(s) were used to generate this analysis")
                    
                    # Show smart query metadata
                    input_type = metadata.get('input_type', '')
                    
                    if input_type in ['smart_query', 'smart_followup']:
                        with st.expander("📊 Smart Query Details", expanded=False):
                            selection_list = metadata.get('selected_signal_types', [])
                            selection_reason = metadata.get('signal_type_reasoning', '')
                            if selection_list:
                                st.markdown(f"**AI Signal Types:** {', '.join(get_signal_type_label(sig) for sig in selection_list)}")
                                if selection_reason:
                                    st.caption(f"💡 {selection_reason}")
                            # Show follow-up specific info if applicable
                            if input_type == 'smart_followup':
                                followup_mode = metadata.get('followup_mode', 'unknown')
                                needs_new_data = metadata.get('needs_new_data', False)
                                analysis_reasoning = metadata.get('analysis_reasoning', '')
                                history_used = metadata.get('history_exchanges_used', 0)
                                filters_changed = metadata.get('filters_changed', False)
                                filter_change_details = metadata.get('filter_change_details', [])
                                data_passing_mode = metadata.get('data_passing_mode', 'unknown')
                                
                                # Build info message
                                info_msg = f"""**Follow-up Mode**: {followup_mode.replace('_', ' ').title()}
- **New Data Needed**: {'Yes' if needs_new_data else 'No'}
- **Reasoning**: {analysis_reasoning}
- **History Context**: Last {history_used} exchange(s) used"""
                                
                                if filters_changed:
                                    info_msg += f"\n- **⚠️ Filters Changed**: {', '.join(filter_change_details)}"
                                
                                # Add token optimization info
                                if data_passing_mode == "full_data":
                                    info_msg += f"\n- **💰 Data Passed**: Full data (filters changed)"
                                elif data_passing_mode == "new_columns_only":
                                    info_msg += f"\n- **💰 Data Passed**: Only NEW columns (token optimized ⚡)"
                                elif data_passing_mode == "no_new_data":
                                    info_msg += f"\n- **💰 Data Passed**: Nothing (all in context ⚡⚡)"
                                
                                # Add batch processing info if applicable
                                batch_mode = metadata.get('batch_mode', '')
                                batch_count = metadata.get('batch_count', 0)
                                if batch_mode == 'multi':
                                    info_msg += f"\n- **⚡ Batch Processing**: {batch_count} batches (data too large for single call)"
                                elif batch_mode == 'single' and batch_count == 1:
                                    info_msg += f"\n- **⚡ Processing**: Single batch (data within token limit)"
                                
                                st.info(info_msg)
                            
                            # Show column selection per signal type
                            st.subheader("🎯 Column Selection by Signal Type")
                            
                            columns_by_type = metadata.get('columns_by_signal_type', {})
                            reasoning_by_type = metadata.get('reasoning_by_signal_type', {})
                            new_columns_by_type = metadata.get('new_columns_by_type', {})
                            existing_columns_by_type = metadata.get('existing_columns_by_type', {})
                            
                            for signal_type in metadata.get('selected_signal_types', []):
                                if signal_type in columns_by_type:
                                    cols = columns_by_type[signal_type]
                                    reasoning = reasoning_by_type.get(signal_type, '')
                                    new_cols = new_columns_by_type.get(signal_type, [])
                                    existing_cols = existing_columns_by_type.get(signal_type, [])
                                    
                                    # Show counts
                                    total_cols = len(cols)
                                    new_count = len(new_cols)
                                    existing_count = len(existing_cols)
                                    
                                    if new_count > 0 and existing_count > 0:
                                        st.markdown(f"**{get_signal_type_label(signal_type, uppercase=True)}** ({total_cols} columns: {new_count} new ✨ + {existing_count} existing 📦)")
                                    elif new_count > 0:
                                        st.markdown(f"**{get_signal_type_label(signal_type, uppercase=True)}** ({total_cols} columns: all new ✨)")
                                    elif existing_count > 0:
                                        st.markdown(f"**{get_signal_type_label(signal_type, uppercase=True)}** ({total_cols} columns: all existing 📦)")
                                    else:
                                        st.markdown(f"**{get_signal_type_label(signal_type, uppercase=True)}** ({total_cols} columns)")
                                    
                                    st.caption(f"💡 {reasoning}")
                                    
                                    with st.expander(f"View {signal_type} columns"):
                                        # Show new columns first
                                        if new_cols:
                                            st.markdown("**✨ NEW Columns (freshly fetched):**")
                                            for col in new_cols:
                                                st.markdown(f"  <span style='color: #00cc00;'>✨ {col}</span>", unsafe_allow_html=True)
                                        
                                        # Then show existing columns
                                        if existing_cols:
                                            if new_cols:  # Add separator if we showed new cols
                                                st.markdown("---")
                                            st.markdown("**📦 EXISTING Columns (already in context):**")
                                            for col in existing_cols:
                                                st.markdown(f"  <span style='color: #888888;'>📦 {col}</span>", unsafe_allow_html=True)
                                        
                                        # Fallback: if no split info, show all columns
                                        if not new_cols and not existing_cols:
                                            for col in cols:
                                                st.text(f"  • {col}")
                            
                            # Show data statistics
                            batch_mode = metadata.get('batch_mode', '')
                            batch_count = metadata.get('batch_count', 0)
                            
                            if batch_mode == 'multi' and batch_count > 1:
                                # Show batch-specific metrics for multi-batch processing
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Rows Fetched", metadata.get('rows_fetched', 0))
                                with col2:
                                    signal_types_count = len(metadata.get('signal_types_with_data', metadata.get('selected_signal_types', [])))
                                    st.metric("Signal Types", signal_types_count)
                                with col3:
                                    st.metric("Batch Count", batch_count, help="Data split across multiple API calls")
                                with col4:
                                    total_tokens = metadata.get('tokens_used', {}).get('total', 0)
                                    st.metric("Total Tokens", f"{total_tokens:,}")
                            else:
                                # Standard metrics for single batch
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Rows Fetched", metadata.get('rows_fetched', 0))
                                with col2:
                                    signal_types_count = len(metadata.get('signal_types_with_data', metadata.get('selected_signal_types', [])))
                                    st.metric("Signal Types", signal_types_count)
                                with col3:
                                    total_tokens = metadata.get('tokens_used', {}).get('total', 0)
                                    st.metric("Tokens Used", f"{total_tokens:,}")
                    
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

