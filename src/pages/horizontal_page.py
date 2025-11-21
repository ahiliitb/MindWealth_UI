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
    st.title(f"📊 {page_title}")

    # Show report date from filename if available
    filename = os.path.basename(data_file)
    file_date = extract_date_from_filename(filename)
    if file_date:
        st.markdown(f"**📅 Report Date: {file_date.strftime('%B %d, %Y')} at 5:00 PM EST**")

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
    expected_cols = ['Symbol', 'Interval', 'Latest Horizontal', 'Current Price', 'Difference (%)']
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
            current_price = row.get('Current Price', None)
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
                        st.write(f"**Current Price:** {current_price}")
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
    st.dataframe(df, use_container_width=True, height=600)


