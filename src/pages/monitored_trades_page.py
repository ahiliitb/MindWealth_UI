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
    # Info button at the top
    if st.button("ℹ️ Info About Page", key="info_monitored_trades", help="Click to learn about this page"):
        st.session_state['show_info_monitored_trades'] = not st.session_state.get('show_info_monitored_trades', False)
    
    if st.session_state.get('show_info_monitored_trades', False):
        with st.expander("📖 Monitored Trades Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The Monitored Trades page is your personal portfolio tracker where you can add and monitor specific trades that interest you.
            
            ### Why is it used?
            - **Personal Tracking**: Monitor trades you're personally interested in or invested in
            - **Portfolio Management**: Keep track of your selected positions in one place
            - **Price Updates**: Get automated price updates for your monitored trades
            - **Performance Analysis**: Track individual trade performance over time
            
            ### How to use?
            1. **Add Trades**: Use the "⭐ Add to Monitored" button on Outstanding Signals or New Signals pages
            2. **Update Prices**: Click "🔄 Update Prices" button in sidebar to refresh today prices
            3. **View Status**: Switch between All Trades, Open Trades, and Closed Trades tabs
            4. **Apply Filters**: Use sidebar filters to focus on specific functions, symbols, or intervals
            5. **Remove Trades**: Remove trades you no longer want to monitor
            
            ### Key Features:
            - Personal portfolio tracking
            - Automated price updates from stock data
            - Integration with outstanding signals for exit detection
            - Open vs closed trade segregation
            - Real-time profit/loss tracking
            - Customizable watchlist
            """)
    
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
    if st.sidebar.button("🔄 Update Prices", help="Update today prices from stock data"):
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

    all_functions_label = "All Functions"
    function_options_with_all = [all_functions_label] + list(available_functions)

    if 'selected_functions_mt' not in st.session_state:
        st.session_state['selected_functions_mt'] = list(available_functions)

    stored_functions = st.session_state.get('selected_functions_mt', list(available_functions))
    valid_stored_functions = [f for f in stored_functions if f in available_functions]

    functions = st.sidebar.multiselect(
        "Select Functions",
        options=function_options_with_all,
        default=valid_stored_functions,
        key="functions_multiselect_mt",
        help=f"Choose one or more functions. Select '{all_functions_label}' to include all."
    )

    if all_functions_label in functions or not functions:
        st.session_state['selected_functions_mt'] = list(available_functions)
    else:
        st.session_state['selected_functions_mt'] = [f for f in functions if f in available_functions]

    functions = st.session_state['selected_functions_mt']
    
    # Symbol filter
    st.sidebar.markdown("**Symbols:**")
    available_symbols = sorted(df['Symbol'].unique())

    all_symbols_label = "All Symbols"
    symbol_options_with_all = [all_symbols_label] + list(available_symbols)

    if 'selected_symbols_mt' not in st.session_state:
        st.session_state['selected_symbols_mt'] = list(available_symbols)

    stored_symbols = st.session_state.get('selected_symbols_mt', list(available_symbols))
    valid_stored_symbols = [s for s in stored_symbols if s in available_symbols]

    symbols = st.sidebar.multiselect(
        "Select Symbols",
        options=symbol_options_with_all,
        default=valid_stored_symbols,
        key="symbols_multiselect_mt",
        help=f"Choose one or more symbols. Select '{all_symbols_label}' to include all."
    )

    if all_symbols_label in symbols or not symbols:
        st.session_state['selected_symbols_mt'] = list(available_symbols)
    else:
        st.session_state['selected_symbols_mt'] = [s for s in symbols if s in available_symbols]

    symbols = st.session_state['selected_symbols_mt']
    
    # Win rate filter (slider)
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Minimum win rate threshold",
        key="win_rate_slider_mt"
    )
    
    # Sharpe ratio filter
    min_sharpe_ratio = st.sidebar.slider(
        "Min Strategy Sharpe Ratio",
        min_value=-5.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Minimum Strategy Sharpe Ratio threshold",
        key="sharpe_ratio_slider_mt"
    )
    
    # Process each main tab
    with main_tab1:
        st.subheader("📊 All Trades")
        display_monitored_trades_content(df, "All Trades", functions, symbols, min_win_rate, min_sharpe_ratio)
    
    with main_tab2:
        st.subheader("Open Trades")
        df_open = df[df['Status'] == 'Open']
        display_monitored_trades_content(df_open, "Open Trades", functions, symbols, min_win_rate, min_sharpe_ratio)
    
    with main_tab3:
        st.subheader("Closed Trades")
        df_closed = df[df['Status'] != 'Open']
        display_monitored_trades_content(df_closed, "Closed Trades", functions, symbols, min_win_rate, min_sharpe_ratio)


def display_monitored_trades_content(df, tab_name, selected_functions, selected_symbols, min_win_rate, min_sharpe_ratio=-5.0):
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
    
    # Apply Sharpe ratio filter if column exists
    if 'Strategy_Sharpe' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Strategy_Sharpe'].fillna(-999) >= min_sharpe_ratio]
    
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
            
            # Display strategy cards (with search)
            search_filtered_df = create_strategy_cards(interval_df, page_name="Monitored Trades", tab_context=f"{trade_status}_{position_name}_{interval}")
            st.markdown("---")
            
            # Display detailed data table (uses search-filtered data)
            st.markdown("### 📋 Detailed Data Table (Original CSV Format)")
            display_trades_table(search_filtered_df, f"{position_name} - {interval}")


def display_monitored_trades_metrics(df, interval, position_name):
    """Display summary metrics for monitored trades"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(df)
        st.metric("Total Trades", total_trades)
    
    with col2:
        # Calculate actual win rate:
        # - For closed trades: based on actual profit (exit price vs signal price)
        # - For open trades: based on current mark to market (today price vs signal price)
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
                # For open trades: use today price to calculate mark to market
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
        # Signal Open Price: backend deduplication only - never display
        # Only show in strategy cards if not "No Information"
        columns_to_exclude = [
            'Signal Open Price',
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
                    help=f"Original CSV column: {col}"
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
