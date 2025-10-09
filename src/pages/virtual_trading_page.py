"""
Virtual Trading Page - Display open/closed trades with filters
"""

import streamlit as st
import pandas as pd


def create_virtual_trading_page():
    """Create virtual trading page with open/closed/all trades tabs"""
    st.title("📈 Virtual Trading")
    st.markdown("---")
    
    # Load data from both long and short CSV files
    try:
        df_long = pd.read_csv('./trade_store/US/virtual_trading_long.csv')
        df_long['Position'] = 'Long'
    except Exception as e:
        st.error(f"Error loading virtual_trading_long.csv: {str(e)}")
        df_long = pd.DataFrame()
    
    try:
        df_short = pd.read_csv('./trade_store/US/virtual_trading_short.csv')
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
    if st.sidebar.button("All", key="select_all_functions_vt", help="Select all functions", use_container_width=True):
        st.session_state['selected_functions_vt'] = list(df['Function'].unique())
    
    if 'selected_functions_vt' not in st.session_state:
        st.session_state['selected_functions_vt'] = list(df['Function'].unique())
    
    if len(st.session_state['selected_functions_vt']) == len(df['Function'].unique()):
        st.sidebar.markdown("*All functions selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state['selected_functions_vt'])} of {len(df['Function'].unique())} selected*")
    
    with st.sidebar.expander("Select Functions", expanded=False):
        functions = st.multiselect(
            "",
            options=df['Function'].unique(),
            default=st.session_state['selected_functions_vt'],
            key="functions_multiselect_vt",
            label_visibility="collapsed"
        )
    
    st.session_state['selected_functions_vt'] = functions
    
    # Symbol filter
    st.sidebar.markdown("**Symbols:**")
    if st.sidebar.button("All", key="select_all_symbols_vt", help="Select all symbols", use_container_width=True):
        st.session_state['selected_symbols_vt'] = list(df['Symbol'].unique())
    
    if 'selected_symbols_vt' not in st.session_state:
        st.session_state['selected_symbols_vt'] = list(df['Symbol'].unique())
    
    if len(st.session_state['selected_symbols_vt']) == len(df['Symbol'].unique()):
        st.sidebar.markdown("*All symbols selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state['selected_symbols_vt'])} of {len(df['Symbol'].unique())} selected*")
    
    with st.sidebar.expander("Select Symbols", expanded=False):
        symbols = st.multiselect(
            "",
            options=df['Symbol'].unique(),
            default=st.session_state['selected_symbols_vt'],
            key="symbols_multiselect_vt",
            label_visibility="collapsed"
        )
    
    st.session_state['selected_symbols_vt'] = symbols
    
    # Win rate filter (slider)
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=0,
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
            
            # Display trades in cards
            display_virtual_trading_cards(interval_df, interval, position_name)
            
            # Display detailed data table
            st.markdown("---")
            st.markdown("### 📊 Detailed Data Table")
            
            # Prepare dataframe for display (remove internal Position column if needed, or keep it)
            display_df = interval_df.copy()
            
            # Format numeric columns for better display
            if 'Entry Price' in display_df.columns:
                display_df['Entry Price'] = display_df['Entry Price'].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) else "N/A"
                )
            
            if 'Exit Price' in display_df.columns:
                display_df['Exit Price'] = display_df['Exit Price'].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) and str(x).strip() else "N/A"
                )
            
            if 'Current Price' in display_df.columns:
                display_df['Current Price'] = display_df['Current Price'].apply(
                    lambda x: f"${x:.4f}" if pd.notna(x) and str(x).strip() else "N/A"
                )
            
            # Display the dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
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
        # Calculate average backtested win rate
        if 'Backtested Win Rate [%]' in df.columns:
            win_rates = df['Backtested Win Rate [%]'].dropna()
            if len(win_rates) > 0:
                avg_win_rate = win_rates.mean()
                st.metric("Avg Backtested Win Rate", f"{avg_win_rate:.2f}%")
            else:
                st.metric("Avg Backtested Win Rate", "N/A")
        else:
            st.metric("Avg Backtested Win Rate", "N/A")
    
    st.markdown("---")


def display_virtual_trading_cards(df, interval, position_name):
    """Display virtual trading cards in scrollable container"""
    
    if df.empty:
        st.info(f"No trades available for {interval} - {position_name}")
        return
    
    # Create scrollable container for cards
    with st.container(height=1000, border=True):
        for idx, row in df.iterrows():
            # Determine card color based on status and profit
            status = row.get('Status', 'Unknown')
            profit_str = str(row.get('Realised/Unrealised Profit', '0%'))
            
            # Parse profit
            try:
                profit_val = float(profit_str.replace('%', '').strip())
            except:
                profit_val = 0
            
            # Create expander title
            profit_indicator = "🟢" if profit_val > 0 else "🔴" if profit_val < 0 else "⚪"
            
            expander_title = f"{profit_indicator} {row['Function']} - {row['Symbol']} | {row['Interval']} | {row['Signal']} | {status}"
            
            with st.expander(expander_title, expanded=False):
                st.markdown("**📋 Trade Information**")
                
                # Create three columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🎯 Entry Details**")
                    st.write(f"**Function:** {row['Function']}")
                    st.write(f"**Symbol:** {row['Symbol']}")
                    st.write(f"**Signal:** {row['Signal']}")
                    st.write(f"**Interval:** {row['Interval']}")
                    st.write(f"**Entry Date:** {row['Entry Date']}")
                    st.write(f"**Entry Price:** ${row['Entry Price']:.4f}" if pd.notna(row['Entry Price']) else "**Entry Price:** N/A")
                
                with col2:
                    st.markdown("**📊 Exit & Current Status**")
                    st.write(f"**Status:** {status}")
                    
                    if pd.notna(row['Exit Date']) and str(row['Exit Date']).strip():
                        st.write(f"**Exit Date:** {row['Exit Date']}")
                        if pd.notna(row['Exit Price']):
                            st.write(f"**Exit Price:** ${row['Exit Price']:.4f}")
                    else:
                        st.write(f"**Exit Date:** Not yet exited")
                    
                    if pd.notna(row['Current Price']):
                        st.write(f"**Current Price:** ${row['Current Price']:.4f}")
                    
                    if pd.notna(row['Holding Period']) and str(row['Holding Period']).strip():
                        st.write(f"**Holding Period:** {row['Holding Period']}")
                
                with col3:
                    st.markdown("**💰 Performance**")
                    
                    # Display profit with color
                    if profit_val > 0:
                        st.markdown(f"**Profit:** <span style='color: green; font-weight: bold;'>{profit_str}</span>", unsafe_allow_html=True)
                    elif profit_val < 0:
                        st.markdown(f"**Profit:** <span style='color: red; font-weight: bold;'>{profit_str}</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"**Profit:** {profit_str}")
                    
                    # Display Backtested Win Rate if available
                    if 'Backtested Win Rate [%]' in row.index and pd.notna(row['Backtested Win Rate [%]']):
                        win_rate = row['Backtested Win Rate [%]']
                        st.write(f"**Backtested Win Rate:** {win_rate:.2f}%")
                    
                    # Calculate days in trade if entry date is available
                    if pd.notna(row['Entry Date']):
                        try:
                            from datetime import datetime
                            entry_dt = pd.to_datetime(row['Entry Date'])
                            if pd.notna(row['Exit Date']) and str(row['Exit Date']).strip():
                                exit_dt = pd.to_datetime(row['Exit Date'])
                                days_in_trade = (exit_dt - entry_dt).days
                            else:
                                days_in_trade = (datetime.now() - entry_dt).days
                            st.write(f"**Days in Trade:** {days_in_trade}")
                        except:
                            pass
                    
                    # Show if trade is realized or unrealized
                    if status == 'Open':
                        st.info("📌 **Unrealised Profit**")
                    else:
                        st.success("✅ **Realised Profit**")

