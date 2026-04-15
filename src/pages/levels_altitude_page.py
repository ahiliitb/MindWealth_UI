"""
Combined page for horizontal levels and new high signals.
"""

import hashlib
import pandas as pd
import streamlit as st

from .horizontal_page import render_horizontal_dataframe


def _render_new_high_dataframe(df: pd.DataFrame, page_title: str):
    """Render New High cards/table from the combined report schema."""
    if df.empty:
        st.info("No New High data is currently available.")
        return

    new_high_display_columns = [
        'Report Type',
        'Symbol',
        'Today price',
        'New Highest',
    ]

    st.markdown("### 📊 New High Cards")
    st.markdown("Click on any card to review the latest New High information")
    st.markdown(f"**Total Signals: {len(df)}**")

    with st.container(height=700, border=True):
        for card_num, (_, row) in enumerate(df.iterrows()):
            symbol = str(row.get('Symbol', 'Unknown'))
            today_price = row.get('Today price', 'No Information')
            new_high = row.get('New Highest', 'No Information')
            report_type = row.get('Report Type', 'New High')

            expander_title = f"🔍 {symbol} | New High"

            with st.expander(expander_title, expanded=False):
                st.markdown("**📋 Key Information**")

                unique_str = f"new_high_{page_title}_{card_num}_{symbol}"
                unique_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**🎯 Symbol Details**")
                    st.write(f"**Symbol:** {symbol}")
                    st.write(f"**Today Price:** {today_price}")

                with col2:
                    st.markdown("**📈 New High Status**")
                    st.write(f"**New Highest:** {new_high}")
                    st.write(f"**Report Type:** {report_type}")

                with col3:
                    st.markdown("**📝 Report Info**")
                    st.write(f"**Today Price:** {today_price}")
                    st.caption(f"Card ID: `{unique_hash}`")

    st.markdown("---")
    st.markdown("### 📋 Detailed New High Data Table (Original CSV)")
    display_columns = [col for col in new_high_display_columns if col in df.columns]
    display_df = df[display_columns].copy() if display_columns else df.copy()
    st.dataframe(display_df, use_container_width=True, height=400)


def create_levels_altitude_page(report_file=None, page_title="Horizontal & New High Report"):
    """Render a combined page with Horizontal Levels and New High tabs."""
    if st.button("ℹ️ Info About Page", key=f"info_horizontal_new_high_{page_title}", help="Click to learn about this page"):
        st.session_state[f"show_info_horizontal_new_high_{page_title}"] = not st.session_state.get(
            f"show_info_horizontal_new_high_{page_title}", False
        )

    if st.session_state.get(f"show_info_horizontal_new_high_{page_title}", False):
        with st.expander("📖 Horizontal & New High Report Information", expanded=True):
            st.markdown("""
            ### What is this page?
            This page uses one combined report file and separates horizontal analysis and new high signals into two tabs.

            ### Why is it used?
            - **Horizontal Levels**: Review support and resistance levels with the same cards and charts as before
            - **New High**: Review new-high entries from the same combined report
            - **Single Source File**: Load both tabs from one report instead of separate CSVs

            ### How to use?
            1. Open the tab you want to review
            2. Review the cards and details for that report type
            3. Compare horizontal setups and new highs from one combined page
            """)

    st.title(f"📊 {page_title}")

    from ..utils.helpers import display_data_fetch_info

    display_data_fetch_info(location="header")
    st.markdown("---")

    if not report_file:
        st.info("No combined Horizontal & New High report is currently available.")
        return

    try:
        combined_df = pd.read_csv(report_file)
    except Exception as e:
        st.error(f"Failed to read combined Horizontal & New High report: {str(e)}")
        return

    if combined_df.empty:
        st.info("No data is currently available in the combined Horizontal & New High report.")
        return

    report_type_series = combined_df.get('Report Type', pd.Series(dtype=str)).astype(str).str.strip()
    horizontal_df = combined_df[report_type_series.str.lower() == 'horizontal'].copy()
    new_high_df = combined_df[report_type_series.str.lower() == 'new high'].copy()

    horizontal_tab, altitude_tab = st.tabs(["📏 Horizontal Levels", "🚀 New High"])

    with horizontal_tab:
        st.caption("Horizontal support and resistance levels loaded from the combined report.")
        if not horizontal_df.empty:
            render_horizontal_dataframe(horizontal_df, "Horizontal Levels")
        else:
            st.info("No Horizontal Levels rows are currently available in the combined report.")

    with altitude_tab:
        st.caption("New High entries loaded from the combined report.")
        if not new_high_df.empty:
            _render_new_high_dataframe(new_high_df, page_title)
        else:
            st.info("No New High rows are currently available in the combined report.")
