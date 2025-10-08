"""
Dashboard page for MindWealth Trading Strategy Analysis
"""

import streamlit as st

from ..utils.file_discovery import discover_csv_files


def create_top_signals_dashboard():
    """Create the dashboard page"""
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
            "Outstanding Signals", "Outstanding Target", "Outstanding Signals Exit",
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

