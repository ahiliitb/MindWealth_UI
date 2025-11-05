"""
MindWealth Trading Strategy Analysis - Main Application
Modular version with organized code structure
"""

import streamlit as st

from constant import *
from src.pages import (
    create_top_signals_dashboard,
    create_analysis_page,
    create_text_file_page,
    create_virtual_trading_page,
    render_chatbot_page,
    create_trade_details_page
)
from src.pages.horizontal_page import create_horizontal_page
from src.utils import discover_csv_files

# Set page config
st.set_page_config(
    page_title="Trading Strategy Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    .strategy-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .strategy-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .positive {
        color: #00C851;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .neutral {
        color: #ffbb33;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""
    # Add refresh button at the top
    col1, col2 = st.columns([10, 1])
    with col1:
        st.title("📈 Trading Strategy Analysis")
    with col2:
        if st.button("🔄 Refresh", help="Refresh data and reload page"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("**Select Page**")
    
    # Dynamically discover CSV files
    csv_files = discover_csv_files()
    
    # Define all available pages in the correct order
    page_options = {
        "Dashboard": None,
        "🤖 AI Chatbot": "chatbot",
        "Virtual Trading": "virtual_trading",
        "AI Output": "text_files",
        "Trade Details": "trade_details",
    }
    
    # Add CSV files in the specified order
    page_options.update(csv_files)
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        list(page_options.keys()),
        key="page_selector"
    )
    
    # Display selected page
    if page == "Dashboard":
        create_top_signals_dashboard()
    elif page == "🤖 AI Chatbot":
        render_chatbot_page()
    elif page == "Virtual Trading":
        create_virtual_trading_page()
    elif page == "AI Output":
        create_text_file_page()
    elif page == "Trade Details":
        create_trade_details_page()
    else:
        # Create analysis page for CSV files
        csv_file = page_options[page]
        if csv_file and csv_file not in ["text_files", "virtual_trading", "chatbot"]:
            if page == 'Horizontal':
                create_horizontal_page(csv_file, page)
            else:
                create_analysis_page(csv_file, page)
        else:
            st.error(f"No data file found for {page}")


if __name__ == "__main__":
    main()

