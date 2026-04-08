"""
Performance summary page for displaying trading performance metrics
"""

import streamlit as st
import pandas as pd

from ..components.cards import create_performance_summary_cards, create_performance_cards
from ..utils.data_loader import load_data_from_file
from ..utils.helpers import format_days

DETAILED_TABLE_ONLY_PAGES = {"Combined Performance Report"}


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
            - **Historical Analysis**: Analyze historical performance signal data
            
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
        st.warning(f"No signal data available for {page_title}")
        return
    
    # Display data fetch datetime at top of page
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    
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

    def get_filtered_df(source_df, interval=None, signal_type=None):
        filtered_df = source_df[
            (source_df['Strategy'].isin(strategies)) &
            (source_df['Win_Percentage'] >= min_win_rate)
        ]

        if interval:
            filtered_df = filtered_df[
                filtered_df['Interval'].str.contains(interval, case=False, na=False)
            ]

        if signal_type:
            filtered_df = filtered_df[
                filtered_df['Signal_Type'].str.contains(signal_type, case=False, na=False)
            ]

        return filtered_df
    
    def display_performance_content(filtered_df, tab_name, signal_type_filter=None):
        """Display performance content for each tab"""
        if filtered_df.empty:
            st.warning(f"No {tab_name} signal data matches the selected filters. Please adjust your filters.")
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
            detail_heading = f"### 📋 Detailed Signal Data Table - {tab_name} (Original CSV Format)"
        else:
            detail_heading = f"### 📋 Detailed Performance Table - {tab_name}"
        st.markdown(detail_heading)
        
        # Create a dataframe with original CSV data
        csv_data = []
        for _, row in filtered_df.iterrows():
            csv_data.append(row['Raw_Data'])
        
        if csv_data:
            original_df = pd.DataFrame(csv_data)
            # Exclude Signal Open Price - backend deduplication only, never display
            if 'Signal Open Price' in original_df.columns:
                original_df = original_df.drop(columns=['Signal Open Price'])
            
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

    def render_interval_tabs(source_df, signal_type=None):
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])

        tab_configs = [
            (interval_tab1, "ALL Intervals", None),
            (interval_tab2, "Daily", "Daily"),
            (interval_tab3, "Weekly", "Weekly"),
            (interval_tab4, "Monthly", "Monthly"),
            (interval_tab5, "Quarterly", "Quarterly"),
            (interval_tab6, "Yearly", "Yearly"),
        ]

        for interval_tab, label, interval in tab_configs:
            with interval_tab:
                filtered_df = get_filtered_df(source_df, interval=interval, signal_type=signal_type)
                display_performance_content(filtered_df, label, signal_type or "ALL")

    def render_signal_type_tabs(source_df):
        main_tab1, main_tab2, main_tab3 = st.tabs(["📊 ALL Signal Types", "📈 Long Signals", "📉 Short Signals"])

        with main_tab1:
            render_interval_tabs(source_df)

        with main_tab2:
            render_interval_tabs(source_df, signal_type="Long")

        with main_tab3:
            render_interval_tabs(source_df, signal_type="Short")

    if page_title == "Combined Performance Report" and 'Function' in df.columns:
        latest_df = df[df['Function'].astype(str).str.contains('Latest Performance', case=False, na=False)]
        forward_df = df[df['Function'].astype(str).str.contains('Forward Testing', case=False, na=False)]

        combined_tabs = []
        tab_datasets = []

        if not latest_df.empty:
            combined_tabs.append("📊 Latest Performance")
            tab_datasets.append(("Latest Performance", latest_df))

        if not forward_df.empty:
            combined_tabs.append("🚀 Forward Testing")
            tab_datasets.append(("Forward Testing", forward_df))

        if tab_datasets:
            for tab, (tab_name, tab_df) in zip(st.tabs(combined_tabs), tab_datasets):
                with tab:
                    st.caption(f"Showing `{tab_name}` rows from the combined performance report.")
                    render_signal_type_tabs(tab_df)
        else:
            render_signal_type_tabs(df)
    else:
        render_signal_type_tabs(df)

