"""
General analysis page for CSV data files
"""

import streamlit as st
import pandas as pd
import os

from ..components.cards import create_summary_cards, create_strategy_cards
from ..utils.data_loader import load_data_from_file
from ..utils.file_discovery import extract_date_from_filename
from .performance_page import create_performance_summary_page
from .breadth_page import create_breadth_page


def create_analysis_page(data_file, page_title):
    """Create an analysis page similar to Signal Analysis for any CSV file"""
    # Load data first to check page type before displaying title
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Check if this is a breadth data page (after processing) - handle separately
    if 'Function' in df.columns and 'Bullish_Asset_Percentage' in df.columns and 'Bullish_Signal_Percentage' in df.columns:
        create_breadth_page(data_file, page_title)
        return
    
    # Check if this is a performance summary page (after processing)
    if 'Strategy' in df.columns and 'Interval' in df.columns and 'Total_Trades' in df.columns:
        create_performance_summary_page(data_file, page_title)
        return
    
    # Display title
    st.title(f"📈 {page_title}")
    
    # Display data fetch datetime at top of page (from JSON file)
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    
    st.markdown("---")
    
    # Add interval and position type extraction
    def extract_interval(row):
        # First check if Interval is already in the row (from parser)
        if 'Interval' in row.index and pd.notna(row.get('Interval')):
            interval = str(row['Interval']).strip()
            if interval and interval != 'Unknown':
                return interval
        
        # Fallback to Raw_Data
        interval_info = row['Raw_Data'].get('Interval, Confirmation Status', 'Unknown')
        if interval_info == 'Unknown' or not interval_info:
            interval_info = row['Raw_Data'].get('Interval', 'Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = str(interval_info).strip()
        return interval if interval else 'Unknown'
    
    def extract_position_type(row):
        signal_info = row['Raw_Data'].get('Symbol, Signal, Signal Date/Price[$]', '')
        if 'Long' in str(signal_info):
            return 'Long'
        elif 'Short' in str(signal_info):
            return 'Short'
        else:
            return 'Unknown'
    
    df['Interval'] = df.apply(extract_interval, axis=1)
    df['Position_Type'] = df.apply(extract_position_type, axis=1)
    
    # Create main tabs for position types
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 ALL Positions", "📈 Long Positions", "📉 Short Positions"])
    
    # Sidebar filters (same as Signal Analysis)
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Function filter with select all button
    st.sidebar.markdown("**Functions:**")
    if st.sidebar.button("All", key=f"select_all_functions_{page_title}", help="Select all functions", use_container_width=True):
        all_functions = list(df['Function'].unique())
        st.session_state[f'selected_functions_{page_title}'] = all_functions
        st.session_state[f"functions_multiselect_{page_title}"] = all_functions
    
    # Initialize session state for functions
    if f'selected_functions_{page_title}' not in st.session_state:
        st.session_state[f'selected_functions_{page_title}'] = list(df['Function'].unique())
    
    # Display function selection status
    if len(st.session_state[f'selected_functions_{page_title}']) == len(df['Function'].unique()):
        st.sidebar.markdown("*All functions selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state[f'selected_functions_{page_title}'])} of {len(df['Function'].unique())} selected*")
    
    with st.sidebar.expander("Select Functions", expanded=False):
        functions = st.multiselect(
            "",
            options=df['Function'].unique(),
            default=st.session_state[f'selected_functions_{page_title}'],
            key=f"functions_multiselect_{page_title}",
            label_visibility="collapsed"
        )
    
    # Update session state
    st.session_state[f'selected_functions_{page_title}'] = functions
    
    # Symbol filter with select all button
    st.sidebar.markdown("**Symbols:**")
    if st.sidebar.button("All", key=f"select_all_symbols_{page_title}", help="Select all symbols", use_container_width=True):
        all_symbols = list(df['Symbol'].unique())
        st.session_state[f'selected_symbols_{page_title}'] = all_symbols
        st.session_state[f"symbols_multiselect_{page_title}"] = all_symbols
    
    # Initialize session state for symbols
    if f'selected_symbols_{page_title}' not in st.session_state:
        st.session_state[f'selected_symbols_{page_title}'] = list(df['Symbol'].unique())
    
    # Display symbol selection status
    if len(st.session_state[f'selected_symbols_{page_title}']) == len(df['Symbol'].unique()):
        st.sidebar.markdown("*All symbols selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state[f'selected_symbols_{page_title}'])} of {len(df['Symbol'].unique())} selected*")
    
    with st.sidebar.expander("Select Symbols", expanded=False):
        symbols = st.multiselect(
            "",
            options=df['Symbol'].unique(),
            default=st.session_state[f'selected_symbols_{page_title}'],
            key=f"symbols_multiselect_{page_title}",
            label_visibility="collapsed"
        )
    
    # Update session state
    st.session_state[f'selected_symbols_{page_title}'] = symbols
    
    # Win rate filter
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=0,
        help="Minimum win rate threshold",
        key=f"win_rate_slider_{page_title}"
    )
    
    # Sharpe ratio filter - allow negative values by default
    min_sharpe_ratio = st.sidebar.slider(
        "Min Strategy Sharpe Ratio",
        min_value=-5.0,
        max_value=5.0,
        value=-5.0,
        step=0.1,
        help="Minimum Strategy Sharpe Ratio threshold (default: -5.0 to show all)",
        key=f"sharpe_ratio_slider_{page_title}"
    )
    
    # Use the same display_interval_tabs function but with unique keys
    def display_interval_tabs_for_page(position_df, position_name):
        """Display interval tabs within each position tab for this page"""
        # Create interval sub-tabs
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        def display_tab_content(filtered_df, tab_name):
            """Display content for each tab"""
            if filtered_df.empty:
                st.warning(f"No {tab_name} data matches the selected filters. Please adjust your filters.")
                return
            
            # Create unique key prefix for charts
            chart_key = f"{page_title.lower().replace(' ', '_')}_{position_name.lower().replace(' ', '_')}_{tab_name.lower().replace(' ', '_')}"
            
            # Summary cards
            st.markdown(f"### 🎯 Key Performance Metrics - {position_name} {tab_name}")
            create_summary_cards(filtered_df)
            
            st.markdown("---")
            
            # Strategy cards - pass tab context to ensure unique keys across tabs
            create_strategy_cards(filtered_df, page_title, chart_key)
            
            st.markdown("---")
            
            # Data table - Original CSV format
            st.markdown(f"### 📋 Detailed Data Table - {position_name} {tab_name} (Original CSV Format)")
            
            # Create a dataframe with original CSV data
            csv_data = []
            for _, row in filtered_df.iterrows():
                csv_data.append(row['Raw_Data'])
            
            if csv_data:
                original_df = pd.DataFrame(csv_data)
                
                # Columns to exclude from detail table (only show in strategy cards if not "No Information")
                columns_to_exclude = [
                    'Sigmashell, Success Rate of Past Analysis [%]',
                    'Divergence observed with, Signal Type',
                    'Maxima Broken Date/Price[$]',
                    'Track Level/Price($), Price on Latest Trading day vs Track Level, Signal Type',
                    'Reference Upmove or Downmove start Date/Price($), end Date/Price($)',
                    '% Change in Price on Latest Trading day vs Price on Trendpulse Breakout day/Earliest Unconfirmed Signal day/Confirmed Signal day'
                ]
                
                # Remove excluded columns if they exist
                columns_to_display = [col for col in original_df.columns if col not in columns_to_exclude]
                filtered_original_df = original_df[columns_to_display]
                
                # Reorder columns: Symbol/Signal first, Exit Signal second, Function third
                from ..utils.helpers import reorder_dataframe_columns, find_column_by_keywords
                filtered_original_df = reorder_dataframe_columns(filtered_original_df)
                
                # Find Symbol and Exit Signal columns for pinning
                symbol_col = find_column_by_keywords(filtered_original_df.columns, ['Symbol, Signal', 'Symbol'])
                if not symbol_col:
                    for col in filtered_original_df.columns:
                        if 'Symbol' in col and 'Signal' in col and 'Exit' not in col:
                            symbol_col = col
                            break
                exit_col = find_column_by_keywords(filtered_original_df.columns, ['Exit Signal Date', 'Exit Signal', 'Exit'])
                
                # Display with better formatting and autosize for ALL columns
                column_config = {}
                for col in filtered_original_df.columns:
                    column_config[col] = st.column_config.TextColumn(
                        col,
                        help=f"Original CSV column: {col}"
                        # No width parameter = autosize
                    )
                
                st.dataframe(
                    filtered_original_df,
                    use_container_width=True,
                    height=600,
                    column_config=column_config
                )
        
        # ALL Intervals
        with interval_tab1:
            # Fill NaN values with 0 to ensure they pass filters
            position_df_filtered = position_df.copy()
            position_df_filtered['Win_Rate'] = position_df_filtered['Win_Rate'].fillna(0)
            position_df_filtered['Strategy_Sharpe'] = position_df_filtered['Strategy_Sharpe'].fillna(0)
            
            # Ensure Symbol column doesn't have NaN values that would cause filtering issues
            if 'Symbol' in position_df_filtered.columns:
                position_df_filtered['Symbol'] = position_df_filtered['Symbol'].fillna('')
            
            # Build filter conditions
            filter_mask = pd.Series([True] * len(position_df_filtered), index=position_df_filtered.index)
            
            # Function filter
            if 'Function' in position_df_filtered.columns:
                filter_mask = filter_mask & position_df_filtered['Function'].isin(functions)
            
            # Symbol filter
            if 'Symbol' in position_df_filtered.columns:
                filter_mask = filter_mask & position_df_filtered['Symbol'].isin(symbols)
            
            # Win Rate filter
            if 'Win_Rate' in position_df_filtered.columns:
                filter_mask = filter_mask & (position_df_filtered['Win_Rate'] >= min_win_rate)
            
            # Strategy Sharpe filter - handle NaN and ensure comparison works correctly
            if 'Strategy_Sharpe' in position_df_filtered.columns:
                # Replace NaN with a very negative number so they pass the filter if min_sharpe_ratio allows
                sharpe_series = position_df_filtered['Strategy_Sharpe'].fillna(-999)
                filter_mask = filter_mask & (sharpe_series >= min_sharpe_ratio)
            
            filtered_df = position_df_filtered[filter_mask]
            display_tab_content(filtered_df, "ALL Intervals")
        
        # Daily
        with interval_tab2:
            daily_df = position_df[position_df['Interval'].str.contains('Daily', case=False, na=False)].copy()
            daily_df['Win_Rate'] = daily_df['Win_Rate'].fillna(0)
            daily_df['Strategy_Sharpe'] = daily_df['Strategy_Sharpe'].fillna(0)
            if 'Symbol' in daily_df.columns:
                daily_df['Symbol'] = daily_df['Symbol'].fillna('')
            filter_mask = pd.Series([True] * len(daily_df), index=daily_df.index)
            if 'Function' in daily_df.columns:
                filter_mask = filter_mask & daily_df['Function'].isin(functions)
            if 'Symbol' in daily_df.columns:
                filter_mask = filter_mask & daily_df['Symbol'].isin(symbols)
            if 'Win_Rate' in daily_df.columns:
                filter_mask = filter_mask & (daily_df['Win_Rate'] >= min_win_rate)
            if 'Strategy_Sharpe' in daily_df.columns:
                filter_mask = filter_mask & (daily_df['Strategy_Sharpe'] >= min_sharpe_ratio)
            filtered_df = daily_df[filter_mask]
            display_tab_content(filtered_df, "Daily")
        
        # Weekly
        with interval_tab3:
            weekly_df = position_df[position_df['Interval'].str.contains('Weekly', case=False, na=False)].copy()
            weekly_df['Win_Rate'] = weekly_df['Win_Rate'].fillna(0)
            weekly_df['Strategy_Sharpe'] = weekly_df['Strategy_Sharpe'].fillna(0)
            if 'Symbol' in weekly_df.columns:
                weekly_df['Symbol'] = weekly_df['Symbol'].fillna('')
            filter_mask = pd.Series([True] * len(weekly_df), index=weekly_df.index)
            if 'Function' in weekly_df.columns:
                filter_mask = filter_mask & weekly_df['Function'].isin(functions)
            if 'Symbol' in weekly_df.columns:
                filter_mask = filter_mask & weekly_df['Symbol'].isin(symbols)
            if 'Win_Rate' in weekly_df.columns:
                filter_mask = filter_mask & (weekly_df['Win_Rate'] >= min_win_rate)
            if 'Strategy_Sharpe' in weekly_df.columns:
                filter_mask = filter_mask & (weekly_df['Strategy_Sharpe'] >= min_sharpe_ratio)
            filtered_df = weekly_df[filter_mask]
            display_tab_content(filtered_df, "Weekly")
        
        # Monthly
        with interval_tab4:
            monthly_df = position_df[position_df['Interval'].str.contains('Monthly', case=False, na=False)].copy()
            monthly_df['Win_Rate'] = monthly_df['Win_Rate'].fillna(0)
            monthly_df['Strategy_Sharpe'] = monthly_df['Strategy_Sharpe'].fillna(0)
            if 'Symbol' in monthly_df.columns:
                monthly_df['Symbol'] = monthly_df['Symbol'].fillna('')
            filter_mask = pd.Series([True] * len(monthly_df), index=monthly_df.index)
            if 'Function' in monthly_df.columns:
                filter_mask = filter_mask & monthly_df['Function'].isin(functions)
            if 'Symbol' in monthly_df.columns:
                filter_mask = filter_mask & monthly_df['Symbol'].isin(symbols)
            if 'Win_Rate' in monthly_df.columns:
                filter_mask = filter_mask & (monthly_df['Win_Rate'] >= min_win_rate)
            if 'Strategy_Sharpe' in monthly_df.columns:
                filter_mask = filter_mask & (monthly_df['Strategy_Sharpe'] >= min_sharpe_ratio)
            filtered_df = monthly_df[filter_mask]
            display_tab_content(filtered_df, "Monthly")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = position_df[position_df['Interval'].str.contains('Quarterly', case=False, na=False)].copy()
            quarterly_df['Win_Rate'] = quarterly_df['Win_Rate'].fillna(0)
            quarterly_df['Strategy_Sharpe'] = quarterly_df['Strategy_Sharpe'].fillna(0)
            if 'Symbol' in quarterly_df.columns:
                quarterly_df['Symbol'] = quarterly_df['Symbol'].fillna('')
            filter_mask = pd.Series([True] * len(quarterly_df), index=quarterly_df.index)
            if 'Function' in quarterly_df.columns:
                filter_mask = filter_mask & quarterly_df['Function'].isin(functions)
            if 'Symbol' in quarterly_df.columns:
                filter_mask = filter_mask & quarterly_df['Symbol'].isin(symbols)
            if 'Win_Rate' in quarterly_df.columns:
                filter_mask = filter_mask & (quarterly_df['Win_Rate'] >= min_win_rate)
            if 'Strategy_Sharpe' in quarterly_df.columns:
                filter_mask = filter_mask & (quarterly_df['Strategy_Sharpe'] >= min_sharpe_ratio)
            filtered_df = quarterly_df[filter_mask]
            display_tab_content(filtered_df, "Quarterly")
        
        # Yearly
        with interval_tab6:
            yearly_df = position_df[position_df['Interval'].str.contains('Yearly', case=False, na=False)].copy()
            yearly_df['Win_Rate'] = yearly_df['Win_Rate'].fillna(0)
            yearly_df['Strategy_Sharpe'] = yearly_df['Strategy_Sharpe'].fillna(0)
            if 'Symbol' in yearly_df.columns:
                yearly_df['Symbol'] = yearly_df['Symbol'].fillna('')
            filter_mask = pd.Series([True] * len(yearly_df), index=yearly_df.index)
            if 'Function' in yearly_df.columns:
                filter_mask = filter_mask & yearly_df['Function'].isin(functions)
            if 'Symbol' in yearly_df.columns:
                filter_mask = filter_mask & yearly_df['Symbol'].isin(symbols)
            if 'Win_Rate' in yearly_df.columns:
                filter_mask = filter_mask & (yearly_df['Win_Rate'] >= min_win_rate)
            if 'Strategy_Sharpe' in yearly_df.columns:
                filter_mask = filter_mask & (yearly_df['Strategy_Sharpe'] >= min_sharpe_ratio)
            filtered_df = yearly_df[filter_mask]
            display_tab_content(filtered_df, "Yearly")
    
    # ALL Positions Tab
    with main_tab1:
        display_interval_tabs_for_page(df, "ALL Positions")
    
    # Long Positions Tab
    with main_tab2:
        long_df = df[df['Position_Type'] == 'Long']
        display_interval_tabs_for_page(long_df, "Long Positions")
    
    # Short Positions Tab
    with main_tab3:
        short_df = df[df['Position_Type'] == 'Short']
        display_interval_tabs_for_page(short_df, "Short Positions")

