"""
Breadth analysis page for signal breadth indicators
"""

import streamlit as st
import pandas as pd

from ..components.cards import create_breadth_summary_cards, create_breadth_cards
from ..utils.data_loader import load_data_from_file


def create_breadth_page(data_file, page_title):
    """Create a specialized page for breadth data"""
    st.title(f"📊 {page_title}")
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
    
    st.markdown("---")
    
    # Data table - Original CSV format
    st.markdown("### 📋 Detailed Data Table (Original CSV Format)")
    
    # Create a dataframe with original CSV data
    csv_data = []
    for _, row in df.iterrows():
        csv_data.append(row['Raw_Data'])
    
    if csv_data:
        original_df = pd.DataFrame(csv_data)
        
        # Display with better formatting
        st.dataframe(
            original_df,
            use_container_width=True,
            height=400,
            column_config={
                col: st.column_config.TextColumn(
                    col,
                    width="medium",
                    help=f"Original CSV column: {col}"
                ) for col in original_df.columns
            }
        )

