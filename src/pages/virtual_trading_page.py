"""
Virtual Trading Page - Display open/closed trades with filters
"""

import streamlit as st
import pandas as pd
from ..utils.helpers import format_days
from ..config_paths import VIRTUAL_TRADING_LONG_CSV, VIRTUAL_TRADING_SHORT_CSV


def create_virtual_trading_page():
    """Create virtual trading page with open/closed/all trades tabs"""
    # Info button at the top
    if st.button("ℹ️ Info About Page", key="info_virtual_trading", help="Click to learn about this page"):
        st.session_state['show_info_virtual_trading'] = not st.session_state.get('show_info_virtual_trading', False)
    
    if st.session_state.get('show_info_virtual_trading', False):
        with st.expander("📖 Virtual Trading Page Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The Virtual Trading page simulates real trading scenarios, tracking both open and closed positions for long and short trades.
            
            ### Why is it used?
            - **Paper Trading**: Test strategies without real money
            - **Performance Tracking**: Monitor virtual portfolio performance
            - **Strategy Validation**: Validate trading strategies before live implementation
            - **Risk-Free Learning**: Learn trading mechanics without financial risk
            
            ### How to use?
            1. **View All Trades**: Start with the "All Trades" tab for complete overview
            2. **Filter by Status**: Use "Open Trades" or "Closed Trades" tabs to focus on specific trade states
            3. **Apply Filters**: Use sidebar filters for functions, symbols, and intervals
            4. **Analyze Performance**: Review win rates, profits, and holding periods
            5. **Track Positions**: Monitor position types (Long/Short) and their outcomes
            
            ### Key Features:
            - Separate tracking for long and short positions
            - Open vs closed trade analysis
            - Real-time profit/loss calculations
            - Holding period analysis
            - Comprehensive filtering options
            - Performance metrics and summaries
            """)
    
    st.title("📈 Virtual Trading")
    
    # Display data fetch datetime at top of page
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    st.markdown("---")
    
    # Load data from both long and short CSV files
    try:
        df_long = pd.read_csv(VIRTUAL_TRADING_LONG_CSV)
        df_long['Position'] = 'Long'
    except Exception as e:
        st.error(f"Error loading virtual_trading_long.csv: {str(e)}")
        df_long = pd.DataFrame()
    
    try:
        df_short = pd.read_csv(VIRTUAL_TRADING_SHORT_CSV)
        df_short['Position'] = 'Short'
    except Exception as e:
        st.error(f"Error loading virtual_trading_short.csv: {str(e)}")
        df_short = pd.DataFrame()
    
    # Combine both dataframes
    df = pd.concat([df_long, df_short], ignore_index=True)
    
    if df.empty:
        st.warning("No virtual trading data available")
        return
    
    # Clean and prepare data
    df['Status'] = df['Status'].fillna('Open').str.strip()
    df['Interval'] = df['Interval'].fillna('Unknown')
    df['Function'] = df['Function'].fillna('Unknown')
    
    # Main tabs for trade status
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 All Trades", "Open Trades", "Closed Trades"])
    
    # Sidebar filters
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Function filter
    st.sidebar.markdown("**Functions:**")
    available_functions = sorted(df['Function'].unique())

    all_functions_label = "All Functions"
    function_options_with_all = [all_functions_label] + list(available_functions)

    if 'selected_functions_vt' not in st.session_state:
        st.session_state['selected_functions_vt'] = list(available_functions)

    stored_functions = st.session_state.get('selected_functions_vt', list(available_functions))
    valid_stored_functions = [f for f in stored_functions if f in available_functions]

    functions = st.sidebar.multiselect(
        "Select Functions",
        options=function_options_with_all,
        default=valid_stored_functions,
        key="functions_multiselect_vt",
        help=f"Choose one or more functions. Select '{all_functions_label}' to include all."
    )

    if all_functions_label in functions or not functions:
        st.session_state['selected_functions_vt'] = list(available_functions)
    else:
        st.session_state['selected_functions_vt'] = [f for f in functions if f in available_functions]

    functions = st.session_state['selected_functions_vt']
    
    # Symbol filter
    st.sidebar.markdown("**Symbols:**")
    available_symbols = sorted(df['Symbol'].unique())

    all_symbols_label = "All Symbols"
    symbol_options_with_all = [all_symbols_label] + list(available_symbols)

    if 'selected_symbols_vt' not in st.session_state:
        st.session_state['selected_symbols_vt'] = list(available_symbols)

    stored_symbols = st.session_state.get('selected_symbols_vt', list(available_symbols))
    valid_stored_symbols = [s for s in stored_symbols if s in available_symbols]

    symbols = st.sidebar.multiselect(
        "Select Symbols",
        options=symbol_options_with_all,
        default=valid_stored_symbols,
        key="symbols_multiselect_vt",
        help=f"Choose one or more symbols. Select '{all_symbols_label}' to include all."
    )

    if all_symbols_label in symbols or not symbols:
        st.session_state['selected_symbols_vt'] = list(available_symbols)
    else:
        st.session_state['selected_symbols_vt'] = [s for s in symbols if s in available_symbols]

    symbols = st.session_state['selected_symbols_vt']
    
    # Win rate filter (slider)
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Minimum win rate threshold",
        key="win_rate_slider_vt"
    )
    
    # Process each main tab
    with main_tab1:
        st.subheader("📊 All Trades")
        display_virtual_trading_content(df, "All Trades", functions, symbols, min_win_rate)
    
    with main_tab2:
        st.subheader("Open Trades")
        df_open = df[df['Status'] == 'Open']
        display_virtual_trading_content(df_open, "Open Trades", functions, symbols, min_win_rate)
    
    with main_tab3:
        st.subheader("Closed Trades")
        df_closed = df[df['Status'] != 'Open']
        display_virtual_trading_content(df_closed, "Closed Trades", functions, symbols, min_win_rate)


def display_virtual_trading_content(df, tab_name, selected_functions, selected_symbols, min_win_rate):
    """Display virtual trading content with position and interval tabs"""
    
    if df.empty:
        st.warning(f"No data available for {tab_name}")
        return
    
    # Apply filters
    filtered_df = df[
        (df['Function'].isin(selected_functions)) &
        (df['Symbol'].isin(selected_symbols))
    ]
    
    # Apply win rate filter if column exists
    if 'Backtested Win Rate [%]' in filtered_df.columns and min_win_rate > 0:
        filtered_df = filtered_df[filtered_df['Backtested Win Rate [%]'] >= min_win_rate]
    
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
    """Display interval tabs for virtual trading data"""
    
    if df.empty:
        st.info(f"No {position_name} available for {trade_status}")
        return
    
    # Get unique intervals
    intervals = ['ALL Intervals'] + sorted(df['Interval'].unique().tolist())
    
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
            display_virtual_trading_metrics(interval_df, interval, position_name)
            
            # Display detailed data table
            st.markdown("### 📊 Detailed Data Table")
            
            # Prepare dataframe for display (remove internal Position column if needed, or keep it)
            display_df = interval_df.copy()
            # Exclude Signal Open Price - backend deduplication only, never display
            if 'Signal Open Price' in display_df.columns:
                display_df = display_df.drop(columns=['Signal Open Price'])
            
            # Format numeric columns for better display
            if 'Entry Price' in display_df.columns:
                display_df['Entry Price'] = display_df['Entry Price'].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) else "N/A"
                )
            
            if 'Exit Price' in display_df.columns:
                display_df['Exit Price'] = display_df['Exit Price'].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) and str(x).strip() else "N/A"
                )
            
            # Support both old and new column names
            if "Today Price" in display_df.columns:
                display_df["Today Price"] = display_df["Today Price"].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) and str(x).strip() else "N/A"
                )
            elif 'Current Price' in display_df.columns:
                # Rename old column to new name
                display_df.rename(columns={'Current Price': "Today Price"}, inplace=True)
                display_df["Today Price"] = display_df["Today Price"].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) and str(x).strip() else "N/A"
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


def display_virtual_trading_metrics(df, interval, position_name):
    """Display summary metrics for virtual trading"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(df)
        st.metric("Total Trades", total_trades)
        
    with col2:
        # Calculate actual win rate based on trades
        profit_col = 'Realised/Unrealised Profit'
        if profit_col in df.columns:
            # Parse profit percentage to count winning trades
            winning_trades = 0
            total_with_profit = 0
            for val in df[profit_col]:
                if pd.notna(val) and str(val) != '':
                    try:
                        profit_str = str(val).replace('%', '').strip()
                        if profit_str:
                            profit_val = float(profit_str)
                            total_with_profit += 1
                            if profit_val > 0:
                                winning_trades += 1
                    except:
                        pass
            
            if total_with_profit > 0:
                actual_win_rate = (winning_trades / total_with_profit) * 100
                st.metric("Actual Win Rate", f"{actual_win_rate:.2f}%")
            else:
                st.metric("Actual Win Rate", "N/A")
        else:
            st.metric("Actual Win Rate", "N/A")
    
    with col3:
        # Calculate average profit (for trades with profit data)
        profit_col = 'Realised/Unrealised Profit'
        if profit_col in df.columns:
            # Parse profit percentage
            profits = []
            for val in df[profit_col]:
                if pd.notna(val) and str(val) != '':
                    try:
                        # Remove % and convert to float
                        profit_str = str(val).replace('%', '').strip()
                        if profit_str:
                            profits.append(float(profit_str))
                    except:
                        pass
            
            if profits:
                avg_profit = sum(profits) / len(profits)
                st.metric("Avg Profit", f"{avg_profit:.2f}%")
            else:
                st.metric("Avg Profit", "N/A")
        else:
            st.metric("Avg Profit", "N/A")
        
    with col4:
        # Calculate average holding period
        # For closed trades: Entry Date to Exit Date (actual holding period)
        # For open trades: Entry Date to current date (mark-to-market holding period, updates daily)
        holding_periods = []
        
        if 'Entry Date' in df.columns:
            current_date = pd.Timestamp.now()
            
            for _, row in df.iterrows():
                entry_date = row.get('Entry Date')
                status = row.get('Status', 'Open')
                exit_date = row.get('Exit Date')
                
                if pd.notna(entry_date):
                    try:
                        entry = pd.to_datetime(entry_date)
                        
                        # For closed trades: use Exit Date
                        if status != 'Open' and pd.notna(exit_date) and str(exit_date).strip():
                            exit = pd.to_datetime(exit_date)
                            days = (exit - entry).days
                            if days >= 0:
                                holding_periods.append(days)
                        # For open trades: use current date (mark-to-market)
                        elif status == 'Open':
                            days = (current_date - entry).days
                            if days >= 0:
                                holding_periods.append(days)
                    except:
                        pass
        
        if holding_periods:
            avg_holding = sum(holding_periods) / len(holding_periods)
            st.metric("Avg Holding Period", format_days(f"{avg_holding:.1f}"))
        else:
            st.metric("Avg Holding Period", "N/A")
    
    st.markdown("---")


