"""
Trade Details page for displaying CSV files from success_rate, forward_testing, and latest_performance folders
"""

import streamlit as st
import pandas as pd
import os


from ..config_paths import TRADE_STORE_US_DIR

BASE_DIR = str(TRADE_STORE_US_DIR)


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
    
    # Display data fetch datetime at top of page
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="header")
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
            
            # Initialize session state keys for this folder
            function_key = f"function_{folder_name}"
            interval_key = f"interval_{folder_name}"
            assets_key = f"assets_{folder_name}"
            
            # Initialize session state if not exists
            if function_key not in st.session_state:
                st.session_state[function_key] = None
            if interval_key not in st.session_state:
                st.session_state[interval_key] = None
            if assets_key not in st.session_state:
                st.session_state[assets_key] = []
            
            with col1:
                # Get default function from session state if available and valid
                default_function_idx = 0
                if st.session_state[function_key] and st.session_state[function_key] in structure['functions']:
                    default_function_idx = structure['functions'].index(st.session_state[function_key])
                
                selected_function = st.selectbox(
                    "Select Function",
                    options=structure['functions'],
                    index=default_function_idx,
                    key=f"function_select_{folder_name}",
                    help="Choose a trading function/strategy"
                )
                
                # Update session state
                function_changed = selected_function != st.session_state.get(function_key)
                if function_changed:
                    st.session_state[function_key] = selected_function
                    # Clear interval when function changes (assets will be validated later)
                    st.session_state[interval_key] = None
            
            # Get available intervals for the selected function
            available_intervals = get_available_intervals_for_function(folder_path, selected_function)
            
            with col2:
                if available_intervals:
                    # Get default interval from session state if available and valid
                    default_interval_idx = 0
                    if st.session_state[interval_key] and st.session_state[interval_key] in available_intervals:
                        default_interval_idx = available_intervals.index(st.session_state[interval_key])
                    
                    selected_interval = st.selectbox(
                        "Select Interval",
                        options=available_intervals,
                        index=default_interval_idx,
                        key=f"interval_select_{folder_name}_{selected_function}",
                        help="Choose a time interval"
                    )
                    
                    # Update session state (assets will be validated later, not cleared)
                    interval_changed = selected_interval != st.session_state.get(interval_key)
                    if interval_changed:
                        st.session_state[interval_key] = selected_interval
                else:
                    selected_interval = None
                    st.session_state[interval_key] = None
                    st.info("No intervals available")
            
            with col3:
                # Get available assets for the selected function and interval
                if selected_interval:
                    available_assets = get_available_assets_for_function_interval(folder_path, selected_function, selected_interval)
                    
                    if available_assets:
                        filtered_assets = available_assets
                        if filtered_assets:
                            # Add "All Assets" option at the beginning
                            options_with_all = ["All Assets"] + filtered_assets
                            
                            # Get stored assets from session state, but filter to only include valid ones
                            stored_assets = st.session_state.get(assets_key, [])
                            valid_stored_assets = [a for a in stored_assets if a in filtered_assets]
                            
                            # Use valid stored assets as default, preserving selections
                            selected_assets = st.multiselect(
                                "Select Asset(s)",
                                options=options_with_all,
                                default=valid_stored_assets,
                                key=f"asset_multiselect_{folder_name}_{selected_function}_{selected_interval}",
                                help="Choose one or more assets/tickers. Select 'All Assets' to include all available assets."
                            )
                            
                            # Handle "All Assets" selection
                            if "All Assets" in selected_assets:
                                selected_assets = filtered_assets
                            
                            # Always update session state with current valid selections
                            valid_selected_assets = [a for a in selected_assets if a in filtered_assets and a != "All Assets"]
                            if valid_selected_assets:
                                st.session_state[assets_key] = valid_selected_assets
                            elif selected_assets:
                                st.session_state[assets_key] = filtered_assets
                            
                            # Clear selection button
                            if st.button(
                                "Clear Selection",
                                key=f"clear_assets_{folder_name}_{selected_function}_{selected_interval}",
                                help="Clear all asset selections"
                            ):
                                st.session_state[assets_key] = []
                                st.rerun()
                            
                            if not selected_assets:
                                st.info("Please select at least one asset")
                    else:
                        selected_assets = []
                        # Don't clear stored assets if no assets available (might be temporary)
                        st.info(f"No assets available for {selected_interval} interval")
                else:
                    selected_assets = []
                    # Don't clear stored assets when waiting for interval selection
                    st.info("Please select an interval first")
            
            # Load and display data for multiple assets
            if selected_function and selected_interval and selected_assets:
                # Load data for all selected assets
                all_dataframes = {}
                missing_assets = []
                
                for selected_asset in selected_assets:
                    csv_path = os.path.join(folder_path, selected_function, selected_asset, f"{selected_interval}.csv")
                    
                    if not os.path.exists(csv_path):
                        missing_assets.append(selected_asset)
                        continue
                    
                    df = load_trade_data(folder_path, selected_function, selected_asset, selected_interval)
                    
                    if df is not None and not df.empty:
                        # Add asset column to distinguish data
                        if 'Asset' not in df.columns:
                            df['Asset'] = selected_asset
                        all_dataframes[selected_asset] = df
                
                if missing_assets:
                    st.warning(f"Data files not found for: {', '.join(missing_assets)}")
                
                if all_dataframes:
                    # Combine all dataframes if multiple assets selected
                    if len(all_dataframes) > 1:
                        combined_df = pd.concat(all_dataframes.values(), ignore_index=True)
                        st.markdown("---")
                        st.markdown(f"### 📈 Trade Data: {selected_function} - {len(selected_assets)} Assets - {selected_interval}")
                        st.markdown(f"**Selected Assets:** {', '.join(selected_assets)}")
                    else:
                        # Single asset - use the first (and only) dataframe
                        asset_name = list(all_dataframes.keys())[0]
                        combined_df = all_dataframes[asset_name]
                        st.markdown("---")
                        st.markdown(f"### 📈 Trade Data: {selected_function} - {asset_name} - {selected_interval}")
                    
                    # Check if Signal column exists and get unique signal types
                    if 'Signal' in combined_df.columns:
                        unique_signals = combined_df['Signal'].unique()
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
                                    with st.container(height=1000, border=True):
                                        # Filter data based on tab name
                                        # Long tab: filter Buy OR Long
                                        # Short tab: filter Sell OR Short
                                        if tab_name == 'Long':
                                            filtered_df = combined_df[combined_df['Signal'].isin(['Buy', 'Long'])].copy()
                                        elif tab_name == 'Short':
                                            filtered_df = combined_df[combined_df['Signal'].isin(['Sell', 'Short'])].copy()
                                        else:
                                            # For other signal types, filter exactly
                                            filtered_df = combined_df[combined_df['Signal'] == tab_name].copy()
                                        
                                        if not filtered_df.empty:
                                            # Show asset breakdown if multiple assets
                                            if len(selected_assets) > 1:
                                                asset_counts = filtered_df['Asset'].value_counts()
                                                st.markdown(f"#### {tab_name} Signals - {len(filtered_df)} total trades")
                                                st.markdown("**Breakdown by Asset:**")
                                                with st.container(height=320, border=True):
                                                    for asset, count in asset_counts.items():
                                                        st.markdown(f"- {asset}: {count} trades")
                                                st.markdown("---")
                                            else:
                                                st.markdown(f"### {tab_name} Signals ({len(filtered_df)} trades)")
                                            
                                            # Display the filtered CSV data table with autosize
                                            table_height = min(700, max(400, (len(filtered_df) + 1) * 35))
                                            st.dataframe(
                                                filtered_df, 
                                                use_container_width=True, 
                                                height=table_height,
                                                column_config={
                                                    col: st.column_config.Column(
                                                        col
                                                        # No width parameter = autosize
                                                    ) for col in filtered_df.columns
                                                }
                                            )
                                            
                                            # Download button for filtered data
                                            csv_data = filtered_df.to_csv(index=False)
                                            assets_str = "_".join(selected_assets[:3])  # Limit filename length
                                            if len(selected_assets) > 3:
                                                assets_str += f"_and_{len(selected_assets)-3}_more"
                                            st.download_button(
                                                label=f"📥 Download {tab_name} CSV",
                                                data=csv_data,
                                                file_name=f"{selected_function}_{assets_str}_{selected_interval}_{tab_name}.csv",
                                                mime="text/csv",
                                                key=f"download_{folder_name}_{selected_function}_{selected_interval}_{tab_name}_{idx}"
                                            )
                                        else:
                                            st.info(f"No {tab_name} signals found")
                        else:
                            # No signal types found, show all data
                            with st.container(height=850, border=True):
                                table_height = min(700, max(400, (len(combined_df) + 1) * 35))
                                st.dataframe(
                                    combined_df, 
                                    use_container_width=True, 
                                    height=table_height,
                                    column_config={
                                        col: st.column_config.Column(
                                            col
                                            # No width parameter = autosize
                                        ) for col in combined_df.columns
                                    }
                                )
                                
                                # Download button
                                csv_data = combined_df.to_csv(index=False)
                                assets_str = "_".join(selected_assets[:3])
                                if len(selected_assets) > 3:
                                    assets_str += f"_and_{len(selected_assets)-3}_more"
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"{selected_function}_{assets_str}_{selected_interval}.csv",
                                    mime="text/csv",
                                    key=f"download_{folder_name}_{selected_function}_{selected_interval}_combined"
                                )
                    else:
                        # No Signal column, show all data
                        with st.container(height=850, border=True):
                            table_height = min(700, max(400, (len(combined_df) + 1) * 35))
                            st.dataframe(
                                combined_df, 
                                use_container_width=True, 
                                height=table_height,
                                column_config={
                                    col: st.column_config.Column(
                                        col
                                        # No width parameter = autosize
                                    ) for col in combined_df.columns
                                }
                            )
                            
                            # Download button
                            csv_data = combined_df.to_csv(index=False)
                            assets_str = "_".join(selected_assets[:3])
                            if len(selected_assets) > 3:
                                assets_str += f"_and_{len(selected_assets)-3}_more"
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=f"{selected_function}_{assets_str}_{selected_interval}.csv",
                                mime="text/csv",
                                key=f"download_{folder_name}_{selected_function}_{selected_interval}_no_signal"
                            )
                elif not missing_assets:
                    st.info("No data available for the selected combination")
            else:
                st.info("Please select Function, Interval, and at least one Asset to view trade data")

