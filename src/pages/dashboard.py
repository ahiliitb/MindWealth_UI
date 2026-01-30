"""
Dashboard page for MindWealth Trading Strategy Analysis
"""

import streamlit as st

from ..utils.file_discovery import discover_csv_files


def create_top_signals_dashboard():
    """Create the dashboard page"""
    # Display data fetch datetime in sidebar (only for dashboard)
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="sidebar")
    
    # Info button at the top
    if st.button("ℹ️ Info About Page", key="info_dashboard", help="Click to learn about this page"):
        st.session_state['show_info_dashboard'] = not st.session_state.get('show_info_dashboard', False)
    
    if st.session_state.get('show_info_dashboard', False):
        with st.expander("📖 Dashboard Page Information", expanded=True):
            st.markdown("""
            ### What is this page?
            The Trading Strategy Dashboard is your central hub for accessing all trading strategy analysis tools and pages.
            
            ### Why is it used?
            - **Overview**: Get a quick overview of all available strategy analysis pages
            - **Navigation**: Easily navigate to different sections of the application
            - **Status Check**: See which strategies have data available and which don't
            
            ### How to use?
            1. **View Available Strategies**: Check the list of model function strategies and signal pages
            2. **Check Data Availability**: Green checkmarks (✅) indicate available data, red crosses (❌) indicate missing data
            3. **Navigate**: Use the sidebar menu to access specific strategy pages
            4. **Explore Features**: Review the additional features section to see what else is available
            
            ### Key Features:
            - Real-time data loading from CSV files
            - Dynamic strategy analysis with filters and visualizations
            - Access to Claude AI output and reports
            - Comprehensive view of all trading strategies
            """)
    
    st.title("📊 Trading Strategy Dashboard")
    st.markdown("---")
    
    st.info("Welcome to the Trading Strategy Analysis Dashboard! Use the navigation menu to explore different strategy analysis pages.")
    
    # Display overview of available strategies
    st.markdown("### 🎯 Available Strategy Analysis Pages")
    
    # Get list of available CSV files
    csv_files = discover_csv_files()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Function Strategies:**")
        strategy_pages = [
            "Band Matrix", "DeltaDrift", "Fractal Track", "BaselineDiverge",
            "Altitude Alpha", "Oscillator Delta", "SigmaShell", "PulseGauge",
            "TrendPulse", "Signal Breadth Indicator (SBI)"
        ]
        for page in strategy_pages:
            if page in csv_files:
                st.markdown(f"✅ {page}")
            else:
                st.markdown(f"❌ {page} (No data)")
    
    with col2:
        st.markdown("**Signal & Performance Pages:**")
        signal_pages = [
            "Outstanding Signals", "Portfolio Risk Management", "Outstanding Signals Exit",
            "New Signals", "Latest Performance", "Forward Testing Performance"
        ]
        for page in signal_pages:
            if page in csv_files:
                st.markdown(f"✅ {page}")
            else:
                st.markdown(f"❌ {page} (No data)")
    
    st.markdown("### 📄 Additional Features")
    st.markdown("• **Claude Output**: View text file outputs")
    st.markdown("• **Dynamic Analysis**: Each page provides detailed strategy analysis with filters and visualizations")
    st.markdown("• **Real-time Data**: All data is loaded dynamically from CSV files")

