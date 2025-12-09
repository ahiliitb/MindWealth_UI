"""
Monitored Trades Page - Personal Portfolio Analysis
Displays and manages user's monitored trades with daily price updates
Similar structure to Virtual Trading page
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.monitored_trades import (
    load_monitored_trades,
    save_monitored_trades,
    remove_trade_from_monitored,
    update_monitored_trades_prices,
    update_monitored_trades_with_outstanding,
    get_latest_price_from_stock_data
)
from src.utils.data_loader import load_data_from_file
from src.components.cards import create_summary_cards, create_strategy_cards
from constant import OUTSTANDING_SIGNAL_CSV_PATH_US


def create_monitored_trades_page():
    """Create the Monitored Trades page"""
    st.title("⭐ Monitored Trades")
    
    # Display data fetch datetime at top of page
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    st.markdown("### Personal Portfolio Analysis")
    st.markdown("---")
    
    # Load monitored trades
    df = load_monitored_trades()
    
    if df.empty:
        st.info("📋 No monitored trades yet. Add trades from the 'Outstanding Signals' or 'New Signals' page using the '⭐ Add to Monitored' button.")
        return
    
    # Sidebar controls
    st.sidebar.markdown("### 🔧 Controls")
    
    # Update prices button
    if st.sidebar.button("🔄 Update Prices", help="Update current prices from stock data"):
        with st.spinner("Updating prices..."):
            # Load outstanding signals to check for exits
            try:
                outstanding_df = load_data_from_file(OUTSTANDING_SIGNAL_CSV_PATH_US, "Outstanding Signals")
                if not outstanding_df.empty:
                    update_monitored_trades_with_outstanding(outstanding_df)
                else:
                    update_monitored_trades_prices()
            except Exception as e:
                st.error(f"Error updating: {e}")
                update_monitored_trades_prices()
            st.success("✅ Prices updated!")
            st.rerun()
    
    # Prepare data - determine open/closed status
    df['Status'] = df.apply(
        lambda row: 'Closed' if pd.notna(row.get('Exit_Date')) and str(row.get('Exit_Date')).strip() else 'Open',
        axis=1
    )
    
    # Ensure required columns exist
    df['Interval'] = df.get('Interval', 'Unknown').fillna('Unknown')
    df['Function'] = df.get('Function', 'Unknown').fillna('Unknown')
    df['Symbol'] = df.get('Symbol', '').fillna('')
    df['Signal_Type'] = df.get('Signal_Type', 'Unknown').fillna('Unknown')
    
    # Determine position type (Long/Short)
    df['Position'] = df['Signal_Type'].apply(
        lambda x: 'Long' if str(x).upper() == 'LONG' else ('Short' if str(x).upper() == 'SHORT' else 'Long')
    )
    
    # Main tabs for trade status
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 All Trades", "Open Trades", "Closed Trades"])
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Function filter
    st.sidebar.markdown("**Functions:**")
    available_functions = sorted(df['Function'].unique())
    
    if st.sidebar.button("All", key="select_all_functions_mt", help="Select all functions", use_container_width=True):
        st.session_state['selected_functions_mt'] = list(available_functions)
        st.session_state["functions_multiselect_mt"] = list(available_functions)
    
    if 'selected_functions_mt' not in st.session_state:
        st.session_state['selected_functions_mt'] = list(available_functions)
    
    # Filter out any functions that no longer exist in the dataframe
    valid_selected_functions = [f for f in st.session_state['selected_functions_mt'] if f in available_functions]
    if len(valid_selected_functions) != len(st.session_state['selected_functions_mt']):
        st.session_state['selected_functions_mt'] = valid_selected_functions if valid_selected_functions else list(available_functions)
    
    if len(st.session_state['selected_functions_mt']) == len(available_functions):
        st.sidebar.markdown("*All functions selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state['selected_functions_mt'])} of {len(available_functions)} selected*")
    
    with st.sidebar.expander("Select Functions", expanded=False):
        functions = st.multiselect(
            "",
            options=available_functions,
            default=st.session_state['selected_functions_mt'],
            key="functions_multiselect_mt",
            label_visibility="collapsed"
        )
    
    st.session_state['selected_functions_mt'] = functions if functions else list(available_functions)
    
    # Symbol filter
    st.sidebar.markdown("**Symbols:**")
    available_symbols = sorted(df['Symbol'].unique())
    
    if st.sidebar.button("All", key="select_all_symbols_mt", help="Select all symbols", use_container_width=True):
        st.session_state['selected_symbols_mt'] = list(available_symbols)
        st.session_state["symbols_multiselect_mt"] = list(available_symbols)
    
    if 'selected_symbols_mt' not in st.session_state:
        st.session_state['selected_symbols_mt'] = list(available_symbols)
    
    # Filter out any symbols that no longer exist in the dataframe
    valid_selected_symbols = [s for s in st.session_state['selected_symbols_mt'] if s in available_symbols]
    if len(valid_selected_symbols) != len(st.session_state['selected_symbols_mt']):
        st.session_state['selected_symbols_mt'] = valid_selected_symbols if valid_selected_symbols else list(available_symbols)
    
    if len(st.session_state['selected_symbols_mt']) == len(available_symbols):
        st.sidebar.markdown("*All symbols selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state['selected_symbols_mt'])} of {len(available_symbols)} selected*")
    
    with st.sidebar.expander("Select Symbols", expanded=False):
        symbols = st.multiselect(
            "",
            options=available_symbols,
            default=st.session_state['selected_symbols_mt'],
            key="symbols_multiselect_mt",
            label_visibility="collapsed"
        )
    
    st.session_state['selected_symbols_mt'] = symbols if symbols else list(available_symbols)
    
    # Win rate filter (slider)
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=0,
        help="Minimum win rate threshold",
        key="win_rate_slider_mt"
    )
    
    # Process each main tab
    with main_tab1:
        st.subheader("📊 All Trades")
        display_monitored_trades_content(df, "All Trades", functions, symbols, min_win_rate)
    
    with main_tab2:
        st.subheader("Open Trades")
        df_open = df[df['Status'] == 'Open']
        display_monitored_trades_content(df_open, "Open Trades", functions, symbols, min_win_rate)
    
    with main_tab3:
        st.subheader("Closed Trades")
        df_closed = df[df['Status'] != 'Open']
        display_monitored_trades_content(df_closed, "Closed Trades", functions, symbols, min_win_rate)


def display_monitored_trades_content(df, tab_name, selected_functions, selected_symbols, min_win_rate):
    """Display monitored trades content with position and interval tabs"""
    
    if df.empty:
        st.warning(f"No data available for {tab_name}")
        return
    
    # Apply filters
    filtered_df = df[
        (df['Function'].isin(selected_functions)) &
        (df['Symbol'].isin(selected_symbols))
    ]
    
    # Apply win rate filter if column exists
    if 'Win_Rate' in filtered_df.columns and min_win_rate > 0:
        filtered_df = filtered_df[filtered_df['Win_Rate'].fillna(0) >= min_win_rate]
    
    if filtered_df.empty:
        st.warning(f"No data matches the current filters for {tab_name}")
        return
    
    # Create position tabs (Long/Short/All)
    position_tab1, position_tab2, position_tab3 = st.tabs(["📈 Long Positions", "📉 Short Positions", "📊 ALL Positions"])
    
    with position_tab1:
        df_long = filtered_df[filtered_df['Position'] == 'Long']
        display_interval_tabs(df_long, "Long Positions", tab_name)
    
    with position_tab2:
        df_short = filtered_df[filtered_df['Position'] == 'Short']
        display_interval_tabs(df_short, "Short Positions", tab_name)
    
    with position_tab3:
        display_interval_tabs(filtered_df, "All Positions", tab_name)


def display_interval_tabs(df, position_name, trade_status):
    """Display interval tabs for monitored trades data"""
    
    if df.empty:
        st.info(f"No {position_name} available for {trade_status}")
        return
    
    # Get unique intervals and create tabs
    unique_intervals = sorted([i for i in df['Interval'].unique() if pd.notna(i) and str(i) != 'Unknown'])
    intervals = ['ALL Intervals'] + unique_intervals
    
    # Create interval tabs
    interval_tabs = st.tabs(intervals)
    
    for i, interval in enumerate(intervals):
        with interval_tabs[i]:
            if interval == 'ALL Intervals':
                interval_df = df
            else:
                interval_df = df[df['Interval'] == interval]
            
            if interval_df.empty:
                st.info(f"No data available for {interval}")
                continue
            
            # Display summary metrics
            display_monitored_trades_metrics(interval_df, interval, position_name)
            
            # Display strategy cards
            create_strategy_cards(interval_df, page_name="Monitored Trades", tab_context=f"{trade_status}_{position_name}_{interval}")
            st.markdown("---")
            
            # Display detailed data table
            st.markdown("### 📋 Detailed Data Table (Original CSV Format)")
            display_trades_table(interval_df, f"{position_name} - {interval}")


def display_monitored_trades_metrics(df, interval, position_name):
    """Display summary metrics for monitored trades"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(df)
        st.metric("Total Trades", total_trades)
    
    with col2:
        # Calculate actual win rate:
        # - For closed trades: based on actual profit (exit price vs signal price)
        # - For open trades: based on current mark to market (current price vs signal price)
        winning_trades = 0
        total_trades_counted = 0
        
        for _, row in df.iterrows():
            signal_price = row.get('Signal_Price', 0)
            signal_type = str(row.get('Signal_Type', '')).upper()
            
            if pd.isna(signal_price) or signal_price <= 0:
                continue
            
            if row['Status'] == 'Closed':
                # For closed trades: use exit price to calculate profit
                exit_price = row.get('Exit_Price')
                if pd.notna(exit_price):
                    total_trades_counted += 1
                    # Calculate profit percentage
                    pnl = ((exit_price - signal_price) / signal_price) * 100
                    # For short positions, invert the P&L
                    if signal_type == 'SHORT':
                        pnl = -pnl
                    # If profit > 0, it's a winning trade
                    if pnl > 0:
                        winning_trades += 1
            elif row['Status'] == 'Open':
                # For open trades: use current price to calculate mark to market
                current_price = row.get('Current_Price')
                if pd.notna(current_price):
                    total_trades_counted += 1
                    # Calculate mark to market percentage
                    mtm = ((current_price - signal_price) / signal_price) * 100
                    # For short positions, invert the MTM
                    if signal_type == 'SHORT':
                        mtm = -mtm
                    # If MTM > 0, it's currently a winning trade
                    if mtm > 0:
                        winning_trades += 1
        
        if total_trades_counted > 0:
            actual_win_rate = (winning_trades / total_trades_counted) * 100
            st.metric("Actual Win Rate", f"{actual_win_rate:.2f}%")
        else:
            st.metric("Actual Win Rate", "N/A")
    
    with col3:
        # Calculate average profit (for all trades - realized for closed, unrealized for open)
        profits = []
        for _, row in df.iterrows():
            if row['Status'] == 'Closed' and pd.notna(row.get('Exit_Price')) and pd.notna(row.get('Signal_Price')):
                # Realized profit
                exit_price = row.get('Exit_Price')
                signal_price = row.get('Signal_Price')
                signal_type = str(row.get('Signal_Type', '')).upper()
                if signal_price > 0:
                    pnl = ((exit_price - signal_price) / signal_price) * 100
                    if signal_type == 'SHORT':
                        pnl = -pnl
                    profits.append(pnl)
            elif row['Status'] == 'Open' and pd.notna(row.get('Current_Price')) and pd.notna(row.get('Signal_Price')):
                # Unrealized profit
                current_price = row.get('Current_Price')
                signal_price = row.get('Signal_Price')
                signal_type = str(row.get('Signal_Type', '')).upper()
                if signal_price > 0:
                    pnl = ((current_price - signal_price) / signal_price) * 100
                    if signal_type == 'SHORT':
                        pnl = -pnl
                    profits.append(pnl)
        
        if profits:
            avg_profit = sum(profits) / len(profits)
            st.metric("Avg Profit", f"{avg_profit:.2f}%")
        else:
            st.metric("Avg Profit", "N/A")
    
    with col4:
        # Calculate average backtested win rate
        if 'Win_Rate' in df.columns:
            win_rates = df['Win_Rate'].dropna()
            if len(win_rates) > 0:
                avg_win_rate = win_rates.mean()
                st.metric("Avg Backtested Win Rate", f"{avg_win_rate:.2f}%")
            else:
                st.metric("Avg Backtested Win Rate", "N/A")
        else:
            st.metric("Avg Backtested Win Rate", "N/A")
    
    st.markdown("---")


def display_trades_table(df: pd.DataFrame, title: str):
    """Display trades in a formatted table - using original CSV columns like Outstanding Signals"""
    if df.empty:
        st.warning(f"No {title.lower()} to display")
        return
    
    # Create a dataframe with original CSV data from Raw_Data (same as Outstanding Signals)
    csv_data = []
    for _, row in df.iterrows():
        if 'Raw_Data' in row and pd.notna(row.get('Raw_Data')):
            # Raw_Data might be a dict or already a dict
            raw_data = row['Raw_Data']
            if isinstance(raw_data, dict):
                csv_data.append(raw_data)
            elif isinstance(raw_data, str):
                # Try to parse if it's a string
                import json
                try:
                    csv_data.append(json.loads(raw_data))
                except:
                    csv_data.append({})
            else:
                csv_data.append({})
        else:
            csv_data.append({})
    
    if csv_data:
        original_df = pd.DataFrame(csv_data)
        
        # Columns to exclude from detail table (same as Outstanding Signals)
        # Only show in strategy cards if not "No Information"
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
        
        # Display with better formatting, pinning, and autosize
        column_config = {}
        for col in filtered_original_df.columns:
            # Pin Symbol and Exit Signal columns
            if col == symbol_col or col == exit_col:
                column_config[col] = st.column_config.TextColumn(
                    col,
                    help=f"Original CSV column: {col}",
                    pinned="left"
                    # No width parameter = autosize
                )
            else:
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
    else:
        st.warning("No original CSV data available for display")
