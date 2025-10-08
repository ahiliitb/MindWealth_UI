"""
Performance summary page for displaying trading performance metrics
"""

import streamlit as st
import pandas as pd

from ..components.cards import create_performance_summary_cards, create_performance_cards
from ..utils.data_loader import load_data_from_file


def create_performance_summary_page(data_file, page_title):
    """Create a performance summary page for CSV files with different structure"""
    st.title(f"📊 {page_title}")
    st.markdown("---")
    
    # Load data from the specific file
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Create main tabs for signal types
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 ALL Signal Types", "📈 Long Signals", "📉 Short Signals"])
    
    # Sidebar filters for performance data
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Strategy filter
    st.sidebar.markdown("**Strategies:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("All", key=f"select_all_strategies_{page_title}", help="Select all strategies", use_container_width=True):
            st.session_state[f'selected_strategies_{page_title}'] = list(df['Strategy'].unique())
    with col2:
        if st.button("None", key=f"deselect_all_strategies_{page_title}", help="Deselect all strategies", use_container_width=True):
            st.session_state[f'selected_strategies_{page_title}'] = []
    
    # Initialize session state for strategies
    if f'selected_strategies_{page_title}' not in st.session_state:
        st.session_state[f'selected_strategies_{page_title}'] = list(df['Strategy'].unique())
    
    # Display strategy selection status
    if len(st.session_state[f'selected_strategies_{page_title}']) == len(df['Strategy'].unique()):
        st.sidebar.markdown("*All strategies selected*")
    elif len(st.session_state[f'selected_strategies_{page_title}']) == 0:
        st.sidebar.markdown("*No strategies selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state[f'selected_strategies_{page_title}'])} of {len(df['Strategy'].unique())} selected*")
    
    with st.sidebar.expander("Select Strategies", expanded=False):
        strategies = st.multiselect(
            "",
            options=df['Strategy'].unique(),
            default=st.session_state[f'selected_strategies_{page_title}'],
            key=f"strategies_multiselect_{page_title}",
            label_visibility="collapsed"
        )
    
    # Update session state
    st.session_state[f'selected_strategies_{page_title}'] = strategies
    
    
    # Win rate filter
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=0,
        help="Minimum win rate threshold",
        key=f"win_rate_slider_{page_title}"
    )
    
    def display_performance_content(filtered_df, tab_name, signal_type_filter=None):
        """Display performance content for each tab"""
        if filtered_df.empty:
            st.warning(f"No {tab_name} data matches the selected filters. Please adjust your filters.")
            return
        
        # Performance summary cards
        st.markdown(f"### 🎯 Performance Summary - {tab_name}")
        create_performance_summary_cards(filtered_df)
        
        st.markdown("---")
        
        # Performance cards
        create_performance_cards(filtered_df)
        
        st.markdown("---")
        
        # Data table - Original CSV format
        st.markdown(f"### 📋 Detailed Data Table - {tab_name} (Original CSV Format)")
        
        # Create a dataframe with original CSV data
        csv_data = []
        for _, row in filtered_df.iterrows():
            csv_data.append(row['Raw_Data'])
        
        if csv_data:
            original_df = pd.DataFrame(csv_data)
            
            # Display with better formatting
            st.dataframe(
                original_df,
                use_container_width=True,
                height=600,
                column_config={
                    col: st.column_config.TextColumn(
                        col,
                        width="medium",
                        help=f"Original CSV column: {col}"
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

