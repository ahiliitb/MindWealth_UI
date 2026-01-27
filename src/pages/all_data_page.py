"""
All Data Page - Display all chatbot data (entry, exit, portfolio, breadth) with filters
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from ..utils.helpers import format_days
from ..config_paths import (
    CHATBOT_ENTRY_CSV, 
    CHATBOT_EXIT_CSV, 
    CHATBOT_TARGET_CSV, 
    CHATBOT_BREADTH_CSV
)


def create_all_data_page():
    """Create All Data page with chatbot data files"""
    st.title("📊 All Data")

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
        
        if st.sidebar.button("All Functions", key="select_all_functions_global",
                           help="Select all functions", use_container_width=True):
            st.session_state['selected_functions_global'] = all_functions_list
            st.session_state["functions_multiselect_global"] = all_functions_list

        if 'selected_functions_global' not in st.session_state:
            st.session_state['selected_functions_global'] = all_functions_list

        if len(st.session_state['selected_functions_global']) == len(all_functions_list):
            st.sidebar.markdown("*All functions selected*")
        else:
            st.sidebar.markdown(f"*{len(st.session_state['selected_functions_global'])} of {len(all_functions_list)} selected*")

        with st.sidebar.expander("Select Functions", expanded=False):
            selected_functions = st.multiselect(
                "",
                options=all_functions_list,
                default=st.session_state['selected_functions_global'],
                key="functions_multiselect_global",
                label_visibility="collapsed"
            )

        st.session_state['selected_functions_global'] = selected_functions
    else:
        selected_functions = []

    # Symbol filter
    if all_symbols:
        st.sidebar.markdown("**Symbols:**")
        all_symbols_list = sorted(list(all_symbols))
        
        if st.sidebar.button("All Symbols", key="select_all_symbols_global",
                           help="Select all symbols", use_container_width=True):
            st.session_state['selected_symbols_global'] = all_symbols_list
            st.session_state["symbols_multiselect_global"] = all_symbols_list

        if 'selected_symbols_global' not in st.session_state:
            st.session_state['selected_symbols_global'] = all_symbols_list

        if len(st.session_state['selected_symbols_global']) == len(all_symbols_list):
            st.sidebar.markdown("*All symbols selected*")
        else:
            st.sidebar.markdown(f"*{len(st.session_state['selected_symbols_global'])} of {len(all_symbols_list)} selected*")

        with st.sidebar.expander("Select Symbols", expanded=False):
            selected_symbols = st.multiselect(
                "",
                options=all_symbols_list,
                default=st.session_state['selected_symbols_global'],
                key="symbols_multiselect_global",
                label_visibility="collapsed"
            )

        st.session_state['selected_symbols_global'] = selected_symbols
    else:
        selected_symbols = []

    # Create main tabs for each data file
    main_tabs = st.tabs(list(data_files.keys()))

    for i, tab_name in enumerate(data_files.keys()):
        with main_tabs[i]:
            display_data_file(tab_name, all_data[tab_name], selected_functions, selected_symbols)


def display_data_file(tab_name, df, selected_functions, selected_symbols):
    """Display a specific data file with filters and tabs"""
    st.subheader(f"📊 {tab_name}")

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

    if filtered_df.empty:
        st.warning(f"No data matches the current filters for {tab_name}")
        return

    # Display summary metrics
    display_data_metrics(filtered_df, tab_name, signal_type)

    # Create position tabs (Long/Short/All) for signals, skip for breadth
    if signal_type == "signals" and 'Symbol, Signal, Signal Date/Price[$]' in filtered_df.columns:
        position_tab1, position_tab2, position_tab3 = st.tabs(["📈 Long Positions", "📉 Short Positions", "📊 ALL Positions"])

        with position_tab1:
            df_long = filtered_df[filtered_df['Symbol, Signal, Signal Date/Price[$]'].str.contains('Long', na=False)]
            display_interval_tabs(df_long, "Long Positions", tab_name)

        with position_tab2:
            df_short = filtered_df[filtered_df['Symbol, Signal, Signal Date/Price[$]'].str.contains('Short', na=False)]
            display_interval_tabs(df_short, "Short Positions", tab_name)

        with position_tab3:
            display_interval_tabs(filtered_df, "All Positions", tab_name)
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

            # Display summary metrics for this interval
            display_interval_metrics(interval_df, interval, position_name)

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