"""
Horizontal page: display Horizontal.csv data with cards and a candlestick + horizontal line, then table
"""

import os
import pandas as pd
import streamlit as st
import hashlib

from ..utils.file_discovery import extract_date_from_filename
from ..components.charts import create_horizontal_chart


def create_horizontal_page(data_file: str, page_title: str):
    """Render the Horizontal analysis page."""
    # Info button at the top
    if st.button("ℹ️ Info About Page", key=f"info_horizontal_{page_title}", help="Click to learn about this page"):
        st.session_state[f'show_info_horizontal_{page_title}'] = not st.session_state.get(f'show_info_horizontal_{page_title}', False)
    
    if st.session_state.get(f'show_info_horizontal_{page_title}', False):
        with st.expander("📖 Horizontal Analysis Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The Horizontal Analysis page displays horizontal support and resistance levels for different assets across various time intervals.
            
            ### Why is it used?
            - **Level Identification**: Identify key horizontal support/resistance levels
            - **Price Comparison**: Compare today price with identified horizontal levels
            - **Technical Analysis**: Use horizontal levels for entry/exit decisions
            - **Chart Visualization**: View interactive charts with horizontal level overlays
            
            ### How to use?
            1. **Browse Cards**: Scroll through strategy cards showing horizontal levels for each symbol
            2. **View Charts**: Click "📊 View Interactive Chart" to see candlestick chart with horizontal line
            3. **Check Difference**: Review the percentage difference between today price and horizontal level
            4. **Analyze Status**: See if price is above or below the horizontal level
            5. **Compare Intervals**: Analyze horizontal levels across different time intervals
            
            ### Key Features:
            - Horizontal support/resistance level identification
            - Interactive candlestick charts
            - Percentage difference calculation
            - Multi-interval analysis
            - Visual price vs level comparison
            - Expandable detail cards
            """)
    
    st.title(f"📊 {page_title}")
    
    # Display data fetch datetime at top of page (from JSON file)
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")

    st.markdown("---")

    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Failed to read Horizontal CSV: {str(e)}")
        return

    if df.empty:
        st.info("No data available in Horizontal report.")
        return

    # Normalize expected columns
    expected_cols = ['Symbol', 'Interval', 'Latest Horizontal', 'Today\'s Price', 'Difference (%)']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in Horizontal CSV: {missing}")

    st.markdown("### 📊 Horizontal Strategy Cards")
    st.markdown("Click on any card to see details and view interactive chart")
    
    total_signals = len(df)
    st.markdown(f"**Total Signals: {total_signals}**")
    
    # Create scrollable container for cards (similar to other pages)
    with st.container(height=1000, border=True):
        for card_num, (idx, row) in enumerate(df.iterrows()):
            symbol = str(row.get('Symbol', 'Unknown'))
            interval = str(row.get('Interval', 'Daily'))
            latest_horizontal = row.get('Latest Horizontal', None)
            current_price = row.get('Today\'s Price', None)
            if current_price is None:
                current_price = row.get('Current Price', None)  # Fallback for old data
            difference = row.get('Difference (%)', None)
            
            # Create expandable card with title format similar to other strategy cards
            expander_title = f"🔍 {symbol} | {interval}"
            
            with st.expander(expander_title, expanded=False):
                st.markdown("**📋 Key Information**")
                
                # Add interactive chart button (similar to strategy cards)
                # Create a unique identifier for the chart button
                unique_str = f"horizontal_{page_title}_{card_num}_{symbol}_{interval}_{idx}"
                unique_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
                chart_key = f"chart_{unique_hash}_{card_num}"
                
                if st.button(f"📊 View Interactive Chart", key=chart_key):
                    create_horizontal_chart(symbol, interval, latest_horizontal)
                
                # Create three columns for better layout (similar to strategy cards)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🎯 Symbol Details**")
                    st.write(f"**Symbol:** {symbol}")
                    st.write(f"**Interval:** {interval}")
                    if latest_horizontal is not None:
                        st.write(f"**Latest Horizontal:** {latest_horizontal}")
                
                with col2:
                    st.markdown("**📊 Price Information**")
                    if current_price is not None:
                        st.write(f"**Today Price:** {current_price}")
                    if difference is not None and difference != "":
                        # Color code the difference
                        try:
                            diff_value = float(difference)
                            color_class = "positive" if diff_value > 0 else "negative"
                            st.markdown(f"**Difference:** <span class='{color_class}'>{difference}%</span>", unsafe_allow_html=True)
                        except:
                            st.write(f"**Difference:** {difference}%")
                
                with col3:
                    st.markdown("**📈 Horizontal Analysis**")
                    if latest_horizontal is not None and current_price is not None:
                        try:
                            latest = float(latest_horizontal)
                            current = float(current_price)
                            if current > latest:
                                st.markdown("**Status:** <span style='color: green;'>Above Horizontal</span>", unsafe_allow_html=True)
                            elif current < latest:
                                st.markdown("**Status:** <span style='color: red;'>Below Horizontal</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("**Status:** <span style='color: orange;'>At Horizontal</span>", unsafe_allow_html=True)
                        except:
                            pass

    st.markdown("---")
    st.markdown("### 📋 Detailed Data Table (Original CSV)")
    # Exclude Signal Open Price - backend deduplication only, never display
    display_df = df.drop(columns=['Signal Open Price']) if 'Signal Open Price' in df.columns else df
    # Ensure ALL columns get autosize (no width parameter = autosize)
    st.dataframe(
        display_df, 
        use_container_width=True, 
        height=600,
        column_config={
            col: st.column_config.Column(
                col
                # No width parameter = autosize
            ) for col in display_df.columns
        }
    )


