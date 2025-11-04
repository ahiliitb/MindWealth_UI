"""
Trade Details page for displaying CSV files from success_rate, forward_testing, and latest_performance folders
"""

import streamlit as st
import pandas as pd
import os


BASE_DIR = "./trade_store/US"


def discover_folder_structure(base_folder):
    """
    Discover the structure of functions, intervals, and assets in a folder
    
    Returns:
        dict: {
            'functions': [list of function names],
            'assets_by_function': {
                'FUNCTION': {
                    'intervals': [list of intervals],
                    'assets': [list of assets]
                }
            }
        }
    """
    structure = {
        'functions': [],
        'assets_by_function': {}
    }
    
    if not os.path.exists(base_folder):
        return structure
    
    # Get all function folders
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            structure['functions'].append(item)
            structure['assets_by_function'][item] = {
                'intervals': [],
                'assets': []
            }
            
            # Get all asset folders within this function
            for asset_item in os.listdir(item_path):
                asset_path = os.path.join(item_path, asset_item)
                if os.path.isdir(asset_path):
                    structure['assets_by_function'][item]['assets'].append(asset_item)
                    
                    # Get all interval CSV files within this asset
                    for csv_file in os.listdir(asset_path):
                        if csv_file.endswith('.csv'):
                            interval_name = csv_file.replace('.csv', '')
                            if interval_name not in structure['assets_by_function'][item]['intervals']:
                                structure['assets_by_function'][item]['intervals'].append(interval_name)
    
    # Sort for consistency
    structure['functions'].sort()
    for func in structure['assets_by_function']:
        structure['assets_by_function'][func]['intervals'].sort()
        structure['assets_by_function'][func]['assets'].sort()
    
    return structure


def get_available_intervals_for_function(base_folder, function):
    """Get all available intervals for a specific function across all assets"""
    function_path = os.path.join(base_folder, function)
    if not os.path.exists(function_path):
        return []
    
    intervals = set()
    for asset_folder in os.listdir(function_path):
        asset_path = os.path.join(function_path, asset_folder)
        if os.path.isdir(asset_path):
            for csv_file in os.listdir(asset_path):
                if csv_file.endswith('.csv'):
                    intervals.add(csv_file.replace('.csv', ''))
    
    return sorted(intervals)


def get_available_assets_for_function_interval(base_folder, function, interval):
    """Get available assets for a specific function and interval"""
    function_path = os.path.join(base_folder, function)
    if not os.path.exists(function_path):
        return []
    
    assets = []
    for asset_folder in os.listdir(function_path):
        asset_path = os.path.join(function_path, asset_folder)
        if os.path.isdir(asset_path):
            csv_file = os.path.join(asset_path, f"{interval}.csv")
            if os.path.exists(csv_file):
                assets.append(asset_folder)
    
    return sorted(assets)


def load_trade_data(base_folder, function, asset, interval):
    """Load CSV data for the selected combination"""
    csv_path = os.path.join(base_folder, function, asset, f"{interval}.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_trade_details_page():
    """Create the Trade Details page with tabs and selection dropdowns"""
    st.title("📊 Trade Details")
    st.markdown("---")
    
    # Define the three main folders
    main_folders = {
        "Backtest History": "success_rate",
        "Forward Testing": "forward_testing",
        "Latest Performance": "latest_performance"
    }
    
    # Create tabs for the three main folders
    tab1, tab2, tab3 = st.tabs(list(main_folders.keys()))
    
    tabs = [tab1, tab2, tab3]
    
    for idx, (tab_name, folder_name) in enumerate(main_folders.items()):
        with tabs[idx]:
            folder_path = os.path.join(BASE_DIR, folder_name)
            
            if not os.path.exists(folder_path):
                st.warning(f"Folder '{folder_name}' not found at {folder_path}")
                continue
            
            # Discover folder structure
            structure = discover_folder_structure(folder_path)
            
            if not structure['functions']:
                st.info(f"No functions found in {folder_name} folder")
                continue
            
            # Create three columns for selection
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_function = st.selectbox(
                    "Select Function",
                    options=structure['functions'],
                    key=f"function_{folder_name}",
                    help="Choose a trading function/strategy"
                )
            
            # Get available intervals for the selected function
            available_intervals = get_available_intervals_for_function(folder_path, selected_function)
            
            with col2:
                if available_intervals:
                    selected_interval = st.selectbox(
                        "Select Interval",
                        options=available_intervals,
                        key=f"interval_{folder_name}_{selected_function}",
                        help="Choose a time interval"
                    )
                else:
                    selected_interval = None
                    st.info("No intervals available")
            
            with col3:
                # Get available assets for the selected function and interval
                if selected_interval:
                    available_assets = get_available_assets_for_function_interval(folder_path, selected_function, selected_interval)
                    
                    if available_assets:
                        # Add search functionality for assets
                        asset_search = st.text_input(
                            "Search Asset",
                            key=f"asset_search_{folder_name}_{selected_function}_{selected_interval}",
                            placeholder="Type to filter assets...",
                            help="Type to filter assets by name"
                        )
                        
                        # Filter assets based on search
                        if asset_search:
                            filtered_assets = [a for a in available_assets if asset_search.upper() in a.upper()]
                        else:
                            filtered_assets = available_assets
                        
                        if filtered_assets:
                            selected_asset = st.selectbox(
                                "Select Asset",
                                options=filtered_assets,
                                key=f"asset_{folder_name}_{selected_function}_{selected_interval}",
                                help="Choose an asset/ticker"
                            )
                        else:
                            selected_asset = None
                            st.info("No assets found matching search")
                    else:
                        selected_asset = None
                        st.info(f"No assets available for {selected_interval} interval")
                else:
                    selected_asset = None
                    st.info("Please select an interval first")
            
            # Load and display data
            if selected_function and selected_interval and selected_asset:
                # Verify the file exists for this combination
                csv_path = os.path.join(folder_path, selected_function, selected_asset, f"{selected_interval}.csv")
                
                if not os.path.exists(csv_path):
                    st.warning(f"Data file not found: {csv_path}")
                else:
                    df = load_trade_data(folder_path, selected_function, selected_asset, selected_interval)
                    
                    if df is not None and not df.empty:
                        st.markdown("---")
                        st.markdown(f"### 📈 Trade Data: {selected_function} - {selected_asset} - {selected_interval}")
                        
                        # Check if Signal column exists and get unique signal types
                        if 'Signal' in df.columns:
                            unique_signals = df['Signal'].unique()
                            # Filter out any NaN or empty values
                            unique_signals = [s for s in unique_signals if pd.notna(s) and str(s).strip() != '']
                            
                            # Normalize signal types: Buy -> Long, Sell -> Short
                            # Create mapping for tab names
                            signal_mapping = {
                                'Buy': 'Long',
                                'Sell': 'Short',
                                'Long': 'Long',
                                'Short': 'Short'
                            }
                            
                            # Get normalized tab names (Long, Short)
                            normalized_tabs = set()
                            for signal in unique_signals:
                                signal_str = str(signal).strip()
                                if signal_str in signal_mapping:
                                    normalized_tabs.add(signal_mapping[signal_str])
                                else:
                                    # Keep other signal types as-is
                                    normalized_tabs.add(signal_str)
                            
                            # Sort tabs: Long first, then Short, then others
                            tab_order = ['Long', 'Short']
                            normalized_tabs_list = sorted(normalized_tabs, key=lambda x: (
                                tab_order.index(x) if x in tab_order else 999,
                                x
                            ))
                            
                            if normalized_tabs_list:
                                # Create tabs with normalized names
                                signal_tabs = st.tabs([f"{tab_name}" for tab_name in normalized_tabs_list])
                                
                                for idx, tab_name in enumerate(normalized_tabs_list):
                                    with signal_tabs[idx]:
                                        # Filter data based on tab name
                                        # Long tab: filter Buy OR Long
                                        # Short tab: filter Sell OR Short
                                        if tab_name == 'Long':
                                            filtered_df = df[df['Signal'].isin(['Buy', 'Long'])].copy()
                                        elif tab_name == 'Short':
                                            filtered_df = df[df['Signal'].isin(['Sell', 'Short'])].copy()
                                        else:
                                            # For other signal types, filter exactly
                                            filtered_df = df[df['Signal'] == tab_name].copy()
                                        
                                        if not filtered_df.empty:
                                            st.markdown(f"### {tab_name} Signals ({len(filtered_df)} trades)")
                                            
                                            # Display the filtered CSV data table
                                            st.dataframe(filtered_df, use_container_width=True, height=600)
                                            
                                            # Download button for filtered data
                                            csv_data = filtered_df.to_csv(index=False)
                                            st.download_button(
                                                label=f"📥 Download {tab_name} CSV",
                                                data=csv_data,
                                                file_name=f"{selected_function}_{selected_asset}_{selected_interval}_{tab_name}.csv",
                                                mime="text/csv",
                                                key=f"download_{folder_name}_{selected_function}_{selected_asset}_{selected_interval}_{tab_name}"
                                            )
                                        else:
                                            st.info(f"No {tab_name} signals found")
                            else:
                                # No signal types found, show all data
                                st.dataframe(df, use_container_width=True, height=600)
                                
                                # Download button
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"{selected_function}_{selected_asset}_{selected_interval}.csv",
                                    mime="text/csv",
                                    key=f"download_{folder_name}_{selected_function}_{selected_asset}_{selected_interval}"
                                )
                        else:
                            # No Signal column, show all data
                            st.dataframe(df, use_container_width=True, height=600)
                            
                            # Download button
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=f"{selected_function}_{selected_asset}_{selected_interval}.csv",
                                mime="text/csv",
                                key=f"download_{folder_name}_{selected_function}_{selected_asset}_{selected_interval}"
                            )
                    elif df is not None and df.empty:
                        st.info("No data available for the selected combination")
                    else:
                        st.error("Failed to load data")
            else:
                st.info("Please select Function, Interval, and Asset to view trade data")

