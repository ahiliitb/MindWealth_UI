"""
All Data Page - Display all chatbot data (entry, exit, portfolio, breadth) with filters
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from ..components.cards import create_summary_cards, create_strategy_cards
from ..utils.helpers import format_days
from ..config_paths import (
    CHATBOT_ENTRY_CSV, 
    CHATBOT_EXIT_CSV, 
    CHATBOT_TARGET_CSV, 
    CHATBOT_BREADTH_CSV
)


def create_all_data_page():
    """Create All Data page with chatbot data files"""    # Info button at the top
    if st.button("ℹ️ Info About Page", key="info_all_data", help="Click to learn about this page"):
        st.session_state['show_info_all_data'] = not st.session_state.get('show_info_all_data', False)
    
    if st.session_state.get('show_info_all_data', False):
        with st.expander("📖 Outstanding Signals Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The Outstanding Signals page consolidates all trading signal data from multiple sources including entry signals, exit signals, portfolio targets, and market breadth indicators.
            
            ### Why is it used?
            - **Centralized View**: Access all signal types in one place
            - **Comprehensive Analysis**: View entry, exit, portfolio targets, and breadth data together
            - **Cross-Signal Comparison**: Compare different signal types and their performance
            - **Unified Filtering**: Apply consistent filters across all signal types
            
            ### How to use?
            1. **Select Tab**: Choose from Entry Signals, Exit Signals, Portfolio Targets, or Market Breadth
            2. **Apply Filters**: Use sidebar filters to narrow down by functions and symbols
            3. **View Summary**: Check summary cards at the top for quick metrics
            4. **Explore Cards**: Scroll through strategy cards for detailed information
            5. **Analyze Table**: Review the comprehensive data table at the bottom
            
            ### Key Features:
            - Multi-tab interface for different signal types
            - Unified filtering across all tabs
            - Entry signals: Current open positions
            - Exit signals: Completed trades
            - Portfolio targets: Target achievement tracking
            - Market breadth: Overall market health indicators
            """)
        st.title("� Outstanding Signals")

    # Display data fetch datetime at top of page
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    st.markdown("---")

    # Define the 4 chatbot data files using config paths
    data_files = {
        "Entry Signals": str(CHATBOT_ENTRY_CSV),
        "Exit Signals": str(CHATBOT_EXIT_CSV),
        "Portfolio Targets": str(CHATBOT_TARGET_CSV),
        "Market Breadth": str(CHATBOT_BREADTH_CSV)
    }

    # Load all data files to get combined functions and symbols
    all_data = {}
    all_functions = set()
    all_symbols = set()
    
    for tab_name, file_path in data_files.items():
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            all_data[tab_name] = df
            
            # Collect all functions
            if 'Function' in df.columns:
                all_functions.update(df['Function'].dropna().unique())
            
            # Collect all symbols
            if 'Symbol, Signal, Signal Date/Price[$]' in df.columns:
                all_symbols.update(df['Symbol, Signal, Signal Date/Price[$]'].dropna().unique())
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            all_data[tab_name] = pd.DataFrame()

    # Create SINGLE set of sidebar filters for ALL tabs
    st.sidebar.markdown("#### 🔍 Filters (Apply to All Tabs)")
    
    # Function filter
    if all_functions:
        st.sidebar.markdown("**Functions:**")
        all_functions_list = sorted(list(all_functions))

        all_functions_label = "All Functions"
        function_options_with_all = [all_functions_label] + list(all_functions_list)

        if 'selected_functions_global' not in st.session_state:
            st.session_state['selected_functions_global'] = all_functions_list

        stored_functions = st.session_state.get('selected_functions_global', all_functions_list)
        valid_stored_functions = [f for f in stored_functions if f in all_functions_list]

        selected_functions = st.sidebar.multiselect(
            "Select Functions",
            options=function_options_with_all,
            default=valid_stored_functions,
            key="functions_multiselect_global",
            help=f"Choose one or more functions. Select '{all_functions_label}' to include all."
        )

        if all_functions_label in selected_functions or not selected_functions:
            st.session_state['selected_functions_global'] = all_functions_list
        else:
            st.session_state['selected_functions_global'] = [f for f in selected_functions if f in all_functions_list]

        selected_functions = st.session_state['selected_functions_global']
    else:
        selected_functions = []

    # Symbol filter
    if all_symbols:
        st.sidebar.markdown("**Symbols:**")
        all_symbols_list = sorted(list(all_symbols))

        all_symbols_label = "All Symbols"
        symbol_options_with_all = [all_symbols_label] + list(all_symbols_list)

        if 'selected_symbols_global' not in st.session_state:
            st.session_state['selected_symbols_global'] = all_symbols_list

        stored_symbols = st.session_state.get('selected_symbols_global', all_symbols_list)
        valid_stored_symbols = [s for s in stored_symbols if s in all_symbols_list]

        selected_symbols = st.sidebar.multiselect(
            "Select Symbols",
            options=symbol_options_with_all,
            default=valid_stored_symbols,
            key="symbols_multiselect_global",
            help=f"Choose one or more symbols. Select '{all_symbols_label}' to include all."
        )

        if all_symbols_label in selected_symbols or not selected_symbols:
            st.session_state['selected_symbols_global'] = all_symbols_list
        else:
            st.session_state['selected_symbols_global'] = [s for s in selected_symbols if s in all_symbols_list]

        selected_symbols = st.session_state['selected_symbols_global']
    else:
        selected_symbols = []

    # Win rate filter (slider)
    st.sidebar.markdown("---")
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Minimum win rate threshold",
        key="win_rate_slider_all_data"
    )
    
    # Sharpe ratio filter
    min_sharpe_ratio = st.sidebar.slider(
        "Min Strategy Sharpe Ratio",
        min_value=-5.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Minimum Strategy Sharpe Ratio threshold",
        key="sharpe_ratio_slider_all_data"
    )

    # Create main tabs for each data file
    main_tabs = st.tabs(list(data_files.keys()))

    for i, tab_name in enumerate(data_files.keys()):
        with main_tabs[i]:
            display_data_file(tab_name, all_data[tab_name], selected_functions, selected_symbols, min_win_rate, min_sharpe_ratio)


def display_data_file(tab_name, df, selected_functions, selected_symbols, min_win_rate=0, min_sharpe_ratio=-5.0):
    """Display a specific data file with filters and tabs"""

    # Check if data is loaded
    if df.empty:
        st.warning(f"No data available for {tab_name}")
        return

    # Determine signal type for filtering logic
    if tab_name == "Market Breadth":
        signal_type = "breadth"
    else:
        signal_type = "signals"

    # Apply filters
    filtered_df = df.copy()

    if signal_type == "signals":
        # Apply function filter if functions exist
        if 'Function' in filtered_df.columns and selected_functions:
            filtered_df = filtered_df[filtered_df['Function'].isin(selected_functions)]
        
        # Apply symbol filter if symbol column exists and we have selected symbols
        if 'Symbol, Signal, Signal Date/Price[$]' in filtered_df.columns and selected_symbols:
            filtered_df = filtered_df[filtered_df['Symbol, Signal, Signal Date/Price[$]'].isin(selected_symbols)]
        
        # Apply win rate filter if column exists
        if 'Win_Rate' in filtered_df.columns and min_win_rate > 0:
            filtered_df = filtered_df[filtered_df['Win_Rate'].fillna(0) >= min_win_rate]
        
        # Apply Sharpe ratio filter if column exists
        if 'Strategy_Sharpe' in filtered_df.columns:
            sharpe_series = filtered_df['Strategy_Sharpe'].fillna(-999)
            filtered_df = filtered_df[sharpe_series >= min_sharpe_ratio]

    if filtered_df.empty:
        st.warning(f"No data matches the current filters for {tab_name}")
        return

    # Create position tabs (Long/Short/All) for signals, skip for breadth
    if signal_type == "signals" and 'Symbol, Signal, Signal Date/Price[$]' in filtered_df.columns:
        position_tab1, position_tab2, position_tab3 = st.tabs(["� ALL Positions", "📈 Long Positions", "📉 Short Positions"])

        with position_tab1:
            display_interval_tabs(filtered_df, "ALL Positions", tab_name)

        with position_tab2:
            df_long = filtered_df[filtered_df['Symbol, Signal, Signal Date/Price[$]'].str.contains('Long', na=False)]
            display_interval_tabs(df_long, "Long Positions", tab_name)

        with position_tab3:
            df_short = filtered_df[filtered_df['Symbol, Signal, Signal Date/Price[$]'].str.contains('Short', na=False)]
            display_interval_tabs(df_short, "Short Positions", tab_name)
    else:
        # For breadth or data without position info, just show all data
        display_interval_tabs(filtered_df, tab_name, tab_name)


def display_interval_tabs(df, position_name, tab_name):
    """Display interval tabs for data"""

    if df.empty:
        st.info(f"No {position_name} available for {tab_name}")
        return

    # Determine which interval column to use
    interval_column = None
    if 'Interval' in df.columns:
        # Portfolio Targets use simple 'Interval' column
        interval_column = 'Interval'
    elif 'Interval, Confirmation Status' in df.columns:
        # Entry/Exit use 'Interval, Confirmation Status' column
        interval_column = 'Interval, Confirmation Status'

    # Get unique intervals
    if interval_column:
        try:
            # Extract intervals and convert to strings to handle mixed types
            interval_list = []
            for val in df[interval_column].unique():
                if pd.notna(val):
                    str_val = str(val).strip()
                    # For 'Interval, Confirmation Status', extract the interval part
                    if ',' in str_val and interval_column == 'Interval, Confirmation Status':
                        interval = str_val.split(',')[0].strip()
                    else:
                        # For simple 'Interval' column, use as-is
                        interval = str_val
                    if interval and interval not in interval_list:
                        interval_list.append(interval)

            intervals = ['ALL Intervals'] + sorted(interval_list)
        except Exception as e:
            # Fallback for any data type issues
            print(f"Warning: Could not parse intervals: {e}")
            intervals = ['ALL Intervals']
    else:
        intervals = ['ALL Intervals']

    # Create interval tabs
    interval_tabs = st.tabs(intervals)

    for i, interval in enumerate(intervals):
        with interval_tabs[i]:
            if interval == 'ALL Intervals':
                interval_df = df
            else:
                # Filter by interval based on which column we're using
                if interval_column == 'Interval':
                    # Simple exact match for 'Interval' column
                    interval_df = df[df[interval_column].astype(str) == interval]
                elif interval_column == 'Interval, Confirmation Status':
                    # For 'Interval, Confirmation Status', check if the interval string starts with our interval
                    interval_df = df[df[interval_column].astype(str).str.startswith(interval)]
                else:
                    interval_df = df

            if interval_df.empty:
                st.info(f"No data available for {interval}")
                continue

            # Add Key Performance Metrics section
            st.markdown(f"### 🎯 Key Performance Metrics - {position_name} {interval}")
            
            # Parse required fields from interval_df
            parsed_data = []
            for _, row in interval_df.iterrows():
                try:
                    # Extract Win Rate
                    win_rate_str = str(row.get('Win Rate [%], History Tested, Number of Trades', '0%'))
                    win_rate = float(win_rate_str.split('%')[0].split(',')[0].strip()) if '%' in win_rate_str else 0.0
                    
                    # Extract Number of Trades
                    num_trades_str = str(row.get('Win Rate [%], History Tested, Number of Trades', '0'))
                    num_trades_parts = num_trades_str.split(',')
                    num_trades = int(num_trades_parts[-1].strip()) if len(num_trades_parts) >= 3 else 0
                    
                    # Extract Strategy CAGR
                    cagr_str = str(row.get('Backtested Strategy CAGR [%]', '0'))
                    strategy_cagr = float(cagr_str.strip().replace('%', '')) if cagr_str.strip() and cagr_str != 'nan' else 0.0
                    
                    # Extract Strategy Sharpe Ratio
                    sharpe_str = str(row.get('Backtested Strategy Sharpe Ratio', '0'))
                    strategy_sharpe = float(sharpe_str.strip()) if sharpe_str.strip() and sharpe_str != 'nan' else 0.0
                    
                    parsed_data.append({
                        'Win_Rate': win_rate,
                        'Num_Trades': num_trades,
                        'Strategy_CAGR': strategy_cagr,
                        'Strategy_Sharpe': strategy_sharpe
                    })
                except Exception as e:
                    # Skip rows that can't be parsed
                    continue
            
            if parsed_data:
                # Create a dataframe with parsed metrics
                metrics_df = pd.DataFrame(parsed_data)
                create_summary_cards(metrics_df)
            
            st.markdown("---")

            # Display detailed data table
            st.markdown("### 📊 Detailed Data Table")

            # Prepare dataframe for display
            display_df = interval_df.copy()

            # Format Signal Open Price for better display
            if 'Signal Open Price' in display_df.columns:
                display_df['Signal Open Price'] = display_df['Signal Open Price'].apply(
                    lambda x: f"${x}" if pd.notna(x) and str(x).strip() else "N/A"
                )

            # Display the dataframe with autosize for ALL columns
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    col: st.column_config.Column(
                        col
                        # No width parameter = autosize
                    ) for col in display_df.columns
                },
                height=min(900, max(500, (len(display_df) + 1) * 35))
            )


def display_data_metrics(df, tab_name, signal_type):
    """Display summary metrics for data"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = len(df)
        st.metric("Total Records", total_records)

    with col2:
        if signal_type == "signals" and 'Symbol, Signal, Signal Date/Price[$]' in df.columns:
            unique_symbols = df['Symbol, Signal, Signal Date/Price[$]'].nunique()
            st.metric("Unique Symbols", unique_symbols)
        elif 'Function' in df.columns:
            unique_functions = df['Function'].nunique()
            st.metric("Unique Functions", unique_functions)
        else:
            st.metric("Data Points", total_records)

    with col3:
        if 'Function' in df.columns:
            unique_functions = df['Function'].nunique()
            st.metric("Functions", unique_functions)
        else:
            st.metric("Data Points", total_records)

    with col4:
        if 'Date' in df.columns:
            unique_dates = df['Date'].nunique()
            st.metric("Unique Dates", unique_dates)
        else:
            st.metric("Records", total_records)

    st.markdown("---")


def display_interval_metrics(df, interval, position_name):
    """Display summary metrics for interval data"""

    col1, col2, col3 = st.columns(3)

    with col1:
        records_count = len(df)
        st.metric(f"{interval} Records", records_count)

    with col2:
        if 'Function' in df.columns:
            unique_functions = df['Function'].nunique()
            st.metric("Functions", unique_functions)
        else:
            st.metric("Data Points", records_count)

    with col3:
        if 'Date' in df.columns:
            # Count records by date
            date_counts = df['Date'].value_counts()
            if not date_counts.empty:
                most_common_date = date_counts.index[0]
                st.metric("Most Common Date", most_common_date)
            else:
                st.metric("Date", "N/A")
        else:
            st.metric("Interval", interval)