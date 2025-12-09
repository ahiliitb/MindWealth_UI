"""
Breadth analysis page for signal breadth indicators
"""

import streamlit as st
import pandas as pd
import os

from ..components.cards import create_breadth_summary_cards, create_breadth_cards
from ..utils.data_loader import load_data_from_file
from constant import BREADTH_SIGNAL_STORE_CSV_PATH_US
from ..utils.file_discovery import extract_date_from_filename


def create_breadth_page(data_file, page_title):
    """Create a specialized page for breadth data"""
    st.title(f"📊 {page_title}")
    
    # Display data fetch datetime at top of page (from JSON file)
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
    
    st.markdown("---")
    
    # Load data from the specific file
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Breadth summary cards
    st.markdown("### 🎯 Market Breadth Summary")
    create_breadth_summary_cards(df)
    
    st.markdown("---")
    
    # Breadth analysis cards
    st.markdown("### 📈 Strategy Breadth Analysis")
    create_breadth_cards(df)
    
    # Insert SBI graph (single chart) after cards and before the table
    st.markdown("---")
    st.markdown("### 📊 Bullish SBI Chart")
    try:
        # Reuse the plotting logic from test_breadth_plot.py
        sbi_df = pd.read_csv(BREADTH_SIGNAL_STORE_CSV_PATH_US, index_col=False)

        # Keep only 'All Function Combined' rows
        data_rows = sbi_df[sbi_df['Function'] == 'All Function Combined'].copy()

        if not data_rows.empty:
            def pct_to_float(val):
                if pd.isna(val) or val == '' or val == ' ':
                    return 0.0
                s = str(val).strip().replace('%', '').replace(',', '')
                if s == '' or s == 'nan':
                    return 0.0
                try:
                    return float(s)
                except:
                    return 0.0

            y1 = [pct_to_float(v) for v in data_rows['Bullish Asset vs Total Asset (%)'].tolist()]
            y2 = [pct_to_float(v) for v in data_rows['Bullish Signal vs Total Signal (%)'].tolist()]

            # X-axis: use Date if present and non-empty, else observation index
            if 'Date' in data_rows.columns:
                dates = data_rows['Date'].tolist()
                has_dates = any(str(d).strip() != '' and not pd.isna(d) for d in dates)
                if has_dates:
                    x = []
                    obs_counter = 1
                    for d in dates:
                        if str(d).strip() == '' or pd.isna(d):
                            x.append(f"Obs {obs_counter}")
                            obs_counter += 1
                        else:
                            x.append(str(d))
                else:
                    x = list(range(1, len(data_rows) + 1))
                xaxis_title = 'Date / Observation' if has_dates else 'Observation'
            else:
                x = list(range(1, len(data_rows) + 1))
                xaxis_title = 'Observation'

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y1, mode='lines', name='Bullish Asset vs Total Asset (%)',
                line=dict(color='#1f77b4', width=3), opacity=0.95
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y2, mode='lines', name='Bullish Signal vs Total Signal (%)',
                line=dict(color='#ff7f0e', width=3), opacity=0.95
            ))

            # Improved axis readability to match other charts
            fig.update_layout(
                title='Bullish Signal Breadth Indicator (SBI) plot for All function',
                xaxis_title=xaxis_title,
                yaxis_title='Percentage (%)',
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.05,
                    xanchor='right', x=1,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#333', borderwidth=1,
                    font=dict(size=13, color='#111')
                ),
                margin=dict(t=110, b=70, r=20, l=65),
                plot_bgcolor='white',
                paper_bgcolor='#fafafa',
                xaxis=dict(
                    showgrid=True, gridcolor='#e9e9e9', gridwidth=1,
                    title=dict(text=xaxis_title, font=dict(size=16, color='#111')),
                    tickfont=dict(size=13, color='#111'),
                    showline=True, linewidth=1.5, linecolor='#333', mirror=True
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='#e9e9e9', gridwidth=1,
                    title=dict(text='Percentage (%)', font=dict(size=16, color='#111')),
                    tickfont=dict(size=13, color='#111'),
                    showline=True, linewidth=1.5, linecolor='#333', mirror=True
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'All Function Combined' observations found in breadth data.")
    except Exception as e:
        st.warning(f"Unable to render SBI graph: {e}")

    st.markdown("---")
    
    # Data table - Original CSV format
    st.markdown("### 📋 Detailed Data Table (Original CSV Format)")
    
    # Create a dataframe with original CSV data
    csv_data = []
    for _, row in df.iterrows():
        csv_data.append(row['Raw_Data'])
    
    if csv_data:
        original_df = pd.DataFrame(csv_data)
        
        # Display with better formatting and autosize for ALL columns
        st.dataframe(
            original_df,
            use_container_width=True,
            height=400,
            column_config={
                col: st.column_config.TextColumn(
                    col,
                    help=f"Original CSV column: {col}"
                    # No width parameter = autosize
                ) for col in original_df.columns
            }
        )

