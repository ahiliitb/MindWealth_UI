"""
F-Stack Analyzer page for visualising band/extension data.
"""

import streamlit as st
import pandas as pd

from ..utils.data_loader import load_data_from_file
from ..utils.helpers import format_days


def _format_number(value, suffix=""):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"


def _render_metric_card(value, label):
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def _format_date(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    return str(value)


def _render_signal_cards(df):
    st.markdown("### 📊 F-Stack Signal Cards")
    if df.empty:
        st.warning("No signals match current filters for card display.")
        return
    
    for _, row in df.iterrows():
        header = (
            f"🔍 {row['Symbol']} • {row['Signal']} | {row['Interval']} | "
            f"{_format_date(row['Signal_Date'])}"
        )
        with st.expander(header, expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Signal Basics**")
                st.write(f"Signal Type: {row['Signal']}")
                st.write(f"Signal Date: {_format_date(row['Signal_Date'])}")
                st.write(f"Signal Price: {_format_number(row['Signal_Price'])}")
                st.write(f"Latest Date: {_format_date(row['Latest_Date'])}")
                st.write(f"Latest Price: {_format_number(row['Latest_Price'])}")
                st.write(f"Price vs Signal: {row['Price_vs_Signal']}")
            
            with col2:
                st.markdown("**Band Structure**")
                st.write(f"Current Extension Level: {_format_number(row['Current_Extension_Level'])}")
                st.write(f"Current Band Range: {row['Current_Band_Range']}")
                st.write(f"Band Width: {_format_number(row['Current_Band_Width'], '%')}")
                st.write(f"Band Composition: {row['Band_Composition']}")
                st.write(f"Trading Days Since Signal: {format_days(str(int(row['Trading_Days']))) if pd.notna(row['Trading_Days']) else 'N/A'}")
            
            with col3:
                st.markdown("**Next Targets**")
                st.write(f"Next Band Level: {_format_number(row['Next_Band_Level'])}")
                st.write(f"Next Band Range: {row['Next_Band_Range']}")
                st.write(f"Next Fib Ret: {row['Next_Fib_Ret']}")
                st.write(f"Next Fib vs Price: {row['Next_Fib_vs_Price']}")
                st.write(f"Next Band vs Price: {row['Next_Band_vs_Price']}")


def create_f_stack_page(data_file, page_title="F-Stack"):
    """Render the F-Stack analyzer page."""
    # Info button at the top
    if st.button("ℹ️ Info About Page", key="info_f_stack", help="Click to learn about this page"):
        st.session_state['show_info_f_stack'] = not st.session_state.get('show_info_f_stack', False)
    
    if st.session_state.get('show_info_f_stack', False):
        with st.expander("📖 F-Stack Analyzer Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The F-Stack Analyzer visualizes and analyzes Fibonacci-based band structures and extension levels for trading signals.
            
            ### Why is it used?
            - **Technical Analysis**: Understand Fibonacci-based price levels and bands
            - **Target Identification**: Identify next potential price targets
            - **Band Analysis**: Analyze band composition and width for trend strength
            - **Extension Tracking**: Monitor current extension levels relative to signal price
            
            ### How to use?
            1. **Review Cards**: Scroll through signal cards for overview of each position
            2. **Check Band Structure**: View current extension level, band range, and composition
            3. **Identify Targets**: Look at next band levels and Fibonacci retracements
            4. **Apply Filters**: Use sidebar filters for functions, symbols, and intervals
            5. **Analyze Table**: Review the complete data table for detailed metrics
            
            ### Key Features:
            - Fibonacci extension level analysis
            - Band structure visualization (width, range, composition)
            - Next target identification
            - Price vs signal tracking
            - Trading days calculation
            - Comprehensive filtering options
            """)
    
    st.title("📐 F-Stack Analyzer")
    
    # Display data fetch datetime at top of page (from JSON file)
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    
    st.caption(
        "Review extension levels, band composition, and upcoming targets derived from the F-Stack Analyzer report."
    )
    
    st.markdown("---")

    df = load_data_from_file(data_file, page_title)

    if df.empty:
        st.warning("No F-Stack data available for the selected report.")
        return

    df = df.copy()

    numeric_columns = [
        'Current_Extension_Level',
        'Current_Band_Width',
        'Trading_Days',
        'Next_Band_Level',
        'Signal_Price',
        'Latest_Price'
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    for column in ['Signal_Date', 'Latest_Date']:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors='coerce')

    st.sidebar.markdown("#### 🔍 Filters")

    # Signal filter
    signal_values = sorted([val for val in df['Signal'].dropna().unique()])
    selected_signals = st.sidebar.multiselect(
        "Signal Type",
        options=signal_values,
        default=signal_values or None,
    )

    # Interval filter
    interval_values = sorted([val for val in df['Interval'].dropna().unique()])
    selected_intervals = st.sidebar.multiselect(
        "Interval",
        options=interval_values,
        default=interval_values or None,
    )

    # Symbol search
    symbol_query = st.sidebar.text_input("Search Symbol").strip()

    # Band width slider
    width_slider = None
    if df['Current_Band_Width'].notna().any():
        width_min = float(df['Current_Band_Width'].min(skipna=True))
        width_max = float(df['Current_Band_Width'].max(skipna=True))
        if width_min == width_max:
            width_slider = (width_min, width_max)
        else:
            width_slider = st.sidebar.slider(
                "Current Band Width (%)",
                min_value=float(width_min),
                max_value=float(width_max),
                value=(float(width_min), float(width_max)),
            )

    # Trading days slider
    days_slider = None
    if df['Trading_Days'].notna().any():
        days_min = int(df['Trading_Days'].min(skipna=True))
        days_max = int(df['Trading_Days'].max(skipna=True))
        if days_min == days_max:
            days_slider = (days_min, days_max)
        else:
            days_slider = st.sidebar.slider(
                "Trading Days Between Signal & Latest",
                min_value=int(days_min),
                max_value=int(days_max),
                value=(int(days_min), int(days_max)),
            )

    filtered_df = df.copy()

    if selected_signals:
        filtered_df = filtered_df[filtered_df['Signal'].isin(selected_signals)]

    if selected_intervals:
        filtered_df = filtered_df[filtered_df['Interval'].isin(selected_intervals)]

    if symbol_query:
        filtered_df = filtered_df[
            filtered_df['Symbol'].str.contains(symbol_query, case=False, na=False)
        ]

    if width_slider and filtered_df['Current_Band_Width'].notna().any():
        filtered_df = filtered_df[
            (filtered_df['Current_Band_Width'].between(width_slider[0], width_slider[1], inclusive="both"))
            | filtered_df['Current_Band_Width'].isna()
        ]

    if days_slider and filtered_df['Trading_Days'].notna().any():
        filtered_df = filtered_df[
            (filtered_df['Trading_Days'].between(days_slider[0], days_slider[1], inclusive="both"))
            | filtered_df['Trading_Days'].isna()
        ]

    if filtered_df.empty:
        st.warning("No F-Stack records match the selected filters.")
        return

    filtered_df = filtered_df.sort_values(by='Latest_Date', ascending=False, na_position='last')

    st.markdown("### 🎯 F-Stack Highlights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_metric_card(f"{len(filtered_df)}", "Total Signals")
    with col2:
        _render_metric_card(_format_number(filtered_df['Current_Extension_Level'].mean()), "Avg Extension Level ($)")
    with col3:
        _render_metric_card(_format_number(filtered_df['Current_Band_Width'].mean(), "%"), "Avg Band Width")
    with col4:
        avg_days = filtered_df['Trading_Days'].mean()
        formatted_avg_days = format_days(f"{avg_days:.0f}") if pd.notna(avg_days) else "N/A"
        _render_metric_card(formatted_avg_days, "Avg Trading Days")

    st.markdown("---")

    _render_signal_cards(filtered_df)

    st.markdown("---")

    st.markdown("### 📋 Detailed Data Table - F-Stack Analyzer")
    display_columns = [
        'Symbol',
        'Signal',
        'Signal_Date',
        'Signal_Price',
        'Latest_Date',
        'Latest_Price',
        'Interval',
        'Current_Extension_Level',
        'Current_Band_Range',
        'Current_Band_Width',
        'Band_Composition',
        'Trading_Days',
        'Price_vs_Signal',
        'Next_Band_Level',
        'Next_Band_Range',
        'Next_Fib_Ret',
        'Next_Fib_vs_Price',
        'Next_Band_vs_Price',
    ]

    available_columns = [col for col in display_columns if col in filtered_df.columns]
    display_df = filtered_df[available_columns]

    # Ensure ALL columns get autosize (no width parameter = autosize)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(
                col
                # No width parameter = autosize
            ) for col in display_df.columns
        }
    )

    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="f_stack_analyzer_filtered.csv",
        mime="text/csv",
    )

