"""
Performance summary page for displaying trading performance metrics
"""

import streamlit as st
import pandas as pd

from ..components.cards import create_performance_summary_cards, create_performance_cards
from ..utils.data_loader import load_data_from_file
from ..utils.helpers import format_days

DETAILED_TABLE_ONLY_PAGES = {"Latest Performance", "Forward Testing Performance"}


def display_summary_metrics_basic(df, tab_name):
    """Display summary/average metrics without card UI."""
    if df.empty:
        return
    
    st.markdown(f"### 🎯 Summary Metrics - {tab_name}")
    
    avg_win_rate = df['Win_Percentage'].mean()
    total_trades = df['Total_Trades'].sum()
    avg_profit = df['Avg_Profit'].mean()
    avg_holding = df['Avg_Holding_Days'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
    with col2:
        st.metric("Total Trades", f"{int(total_trades):,}")
    with col3:
        st.metric("Avg Profit", f"{avg_profit:.1f}%")
    with col4:
        st.metric("Avg Holding Days", format_days(f"{avg_holding:.0f}"))
    
    st.markdown("---")


def create_performance_summary_page(data_file, page_title):
    """Create a performance summary page for CSV files with different structure"""
    # Info button at the top
    if st.button("ℹ️ Info About Page", key=f"info_performance_{page_title}", help="Click to learn about this page"):
        st.session_state[f'show_info_performance_{page_title}'] = not st.session_state.get(f'show_info_performance_{page_title}', False)
    
    if st.session_state.get(f'show_info_performance_{page_title}', False):
        with st.expander("📖 Performance Summary Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The Performance Summary page provides comprehensive statistics and metrics about trading strategy performance, including win rates, average profits, and holding periods.
            
            ### Why is it used?
            - **Strategy Evaluation**: Evaluate the effectiveness of different trading strategies
            - **Performance Metrics**: Review key performance indicators like win rate and profit
            - **Comparison**: Compare performance across strategies, intervals, and signal types
            - **Historical Analysis**: Analyze historical performance data
            
            ### How to use?
            1. **Select Signal Type**: Choose between All, Long, or Short signals using tabs
            2. **Apply Filters**: Use sidebar filters for strategies, intervals, and performance thresholds
            3. **Review Summary**: Check summary metrics at the top (average win rate, total trades, etc.)
            4. **View Cards**: Scroll through strategy cards for detailed performance breakdown
            5. **Analyze Table**: Review the complete performance table at the bottom
            
            ### Key Features:
            - Win rate and profit analysis
            - Average holding period calculation
            - Total trades tracking
            - Strategy and interval filtering
            - Long vs Short performance comparison
            - Summary metrics and aggregations
            - Historical performance tracking
            """)
    
    st.title(f"📊 {page_title}")
    st.markdown("---")
    
    # Load data from the specific file
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Display data fetch datetime at top of page
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    
    # Create main tabs for signal types
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 ALL Signal Types", "📈 Long Signals", "📉 Short Signals"])
    
    # Sidebar filters for performance data
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Function filter
    st.sidebar.markdown("**Functions:**")
    available_strategies = sorted(df['Strategy'].dropna().unique())

    all_functions_label = "All Functions"
    function_options_with_all = [all_functions_label] + list(available_strategies)

    # Initialize session state for strategies
    if f'selected_strategies_{page_title}' not in st.session_state:
        st.session_state[f'selected_strategies_{page_title}'] = list(available_strategies)

    stored_strategies = st.session_state.get(f'selected_strategies_{page_title}', list(available_strategies))
    valid_stored_strategies = [s for s in stored_strategies if s in available_strategies]

    strategies = st.sidebar.multiselect(
        "Select Functions",
        options=function_options_with_all,
        default=valid_stored_strategies,
        key=f"strategies_multiselect_{page_title}",
        help=f"Choose one or more functions. Select '{all_functions_label}' to include all."
    )

    # Treat empty selection as "All" for consistency
    if all_functions_label in strategies or not strategies:
        st.session_state[f'selected_strategies_{page_title}'] = list(available_strategies)
    else:
        st.session_state[f'selected_strategies_{page_title}'] = [s for s in strategies if s in available_strategies]

    strategies = st.session_state[f'selected_strategies_{page_title}']
    
    
    # Win rate filter
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Minimum win rate threshold",
        key=f"win_rate_slider_{page_title}"
    )
    
    def display_performance_content(filtered_df, tab_name, signal_type_filter=None):
        """Display performance content for each tab"""
        if filtered_df.empty:
            st.warning(f"No {tab_name} data matches the selected filters. Please adjust your filters.")
            return
        
        show_cards = page_title not in DETAILED_TABLE_ONLY_PAGES
        
        if show_cards:
            # Performance summary cards
            st.markdown(f"### 🎯 Performance Summary - {tab_name}")
            create_performance_summary_cards(filtered_df)
            
            st.markdown("---")
            
            # Performance cards
            create_performance_cards(filtered_df)
            
            st.markdown("---")
        else:
            display_summary_metrics_basic(filtered_df, tab_name)
        
        # Data table - Original CSV format
        if show_cards:
            detail_heading = f"### 📋 Detailed Data Table - {tab_name} (Original CSV Format)"
        else:
            detail_heading = f"### 📋 Detailed Performance Table - {tab_name}"
        st.markdown(detail_heading)
        
        # Create a dataframe with original CSV data
        csv_data = []
        for _, row in filtered_df.iterrows():
            csv_data.append(row['Raw_Data'])
        
        if csv_data:
            original_df = pd.DataFrame(csv_data)
            
            # Display with better formatting and autosize
            st.dataframe(
                original_df,
                use_container_width=True,
                height=600,
                column_config={
                    col: st.column_config.TextColumn(
                        col,
                        help=f"Original CSV column: {col}"
                        # No width parameter = autosize
                    ) for col in original_df.columns
                }
            )
    
    # ALL Signal Types
    with main_tab1:
        # Create interval tabs for ALL signal types
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = df[
                (df['Strategy'].isin(strategies)) &
                (df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "ALL Intervals", "ALL")
        
        # Daily
        with interval_tab2:
            daily_df = df[df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Strategy'].isin(strategies)) &
                (daily_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Daily", "ALL")
        
        # Weekly
        with interval_tab3:
            weekly_df = df[df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Strategy'].isin(strategies)) &
                (weekly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Weekly", "ALL")
        
        # Monthly
        with interval_tab4:
            monthly_df = df[df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Strategy'].isin(strategies)) &
                (monthly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Monthly", "ALL")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = df[df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Strategy'].isin(strategies)) &
                (quarterly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Quarterly", "ALL")
        
        # Yearly
        with interval_tab6:
            yearly_df = df[df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Strategy'].isin(strategies)) &
                (yearly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Yearly", "ALL")
    
    # Long Signals
    with main_tab2:
        # Create interval tabs for Long signals
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = df[
                (df['Strategy'].isin(strategies)) &
                (df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "ALL Intervals", "Long")
        
        # Daily
        with interval_tab2:
            daily_df = df[df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Strategy'].isin(strategies)) &
                (daily_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (daily_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Daily", "Long")
        
        # Weekly
        with interval_tab3:
            weekly_df = df[df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Strategy'].isin(strategies)) &
                (weekly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (weekly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Weekly", "Long")
        
        # Monthly
        with interval_tab4:
            monthly_df = df[df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Strategy'].isin(strategies)) &
                (monthly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (monthly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Monthly", "Long")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = df[df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Strategy'].isin(strategies)) &
                (quarterly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (quarterly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Quarterly", "Long")
        
        # Yearly
        with interval_tab6:
            yearly_df = df[df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Strategy'].isin(strategies)) &
                (yearly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (yearly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Yearly", "Long")
    
    # Short Signals
    with main_tab3:
        # Create interval tabs for Short signals
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = df[
                (df['Strategy'].isin(strategies)) &
                (df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "ALL Intervals", "Short")
        
        # Daily
        with interval_tab2:
            daily_df = df[df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Strategy'].isin(strategies)) &
                (daily_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (daily_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Daily", "Short")
        
        # Weekly
        with interval_tab3:
            weekly_df = df[df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Strategy'].isin(strategies)) &
                (weekly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (weekly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Weekly", "Short")
        
        # Monthly
        with interval_tab4:
            monthly_df = df[df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Strategy'].isin(strategies)) &
                (monthly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (monthly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Monthly", "Short")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = df[df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Strategy'].isin(strategies)) &
                (quarterly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (quarterly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Quarterly", "Short")
        
        # Yearly
        with interval_tab6:
            yearly_df = df[df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Strategy'].isin(strategies)) &
                (yearly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (yearly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Yearly", "Short")

