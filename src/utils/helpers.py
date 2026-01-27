"""
Helper utility functions
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


from ..config_paths import DATA_FETCH_DATETIME_JSON


def find_column_by_keywords(columns, keywords):
    """Find a column name that contains any of the keywords"""
    for col in columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    return None


def reorder_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder dataframe columns to have:
    1. Symbol/Signal column first (e.g., "Symbol, Signal, Signal Date/Price[$]")
    2. Exit Signal column second (e.g., "Exit Signal Date/Price[$]")
    3. Function column third
    4. Rest of columns after
    
    Args:
        df: DataFrame to reorder
        
    Returns:
        DataFrame with reordered columns
    """
    if df.empty or df.columns.empty:
        return df
    
    # Find the key columns with comprehensive keyword matching
    # Symbol column: look for "Symbol, Signal" or just "Symbol" (but not "Exit Signal")
    symbol_col = find_column_by_keywords(df.columns, ['Symbol, Signal', 'Symbol'])
    if not symbol_col:
        # Try alternative patterns
        for col in df.columns:
            if 'Symbol' in col and 'Signal' in col and 'Exit' not in col:
                symbol_col = col
                break
    
    # Exit Signal column: look for "Exit Signal" or "Exit" (but prioritize "Exit Signal")
    exit_col = find_column_by_keywords(df.columns, ['Exit Signal Date', 'Exit Signal', 'Exit'])
    
    # Function column
    function_col = find_column_by_keywords(df.columns, ['Function'])
    
    # Build ordered column list
    ordered_cols = []
    
    # Add symbol column first if found
    if symbol_col:
        ordered_cols.append(symbol_col)
    
    # Add exit signal column second if found
    if exit_col:
        ordered_cols.append(exit_col)
    
    # Add function column third if found
    if function_col:
        ordered_cols.append(function_col)
    
    # Add remaining columns (excluding already added ones)
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining_cols)
    
    # Reorder dataframe (only if we have columns to reorder)
    if ordered_cols:
        return df[ordered_cols]
    return df


def get_pinned_column_config(df: pd.DataFrame, base_config=None):
    """
    Create column configuration with Symbol and Exit Signal columns pinned by default.
    Users can unpin/change pinning via the table's three-dot menu.
    
    Args:
        df: DataFrame to create config for
        base_config: Optional base column config dict to merge with
        
    Returns:
        Dictionary of column configurations with pinning
    """
    import streamlit as st
    
    # Find the key columns
    symbol_col = find_column_by_keywords(df.columns, ['Symbol, Signal', 'Symbol'])
    if not symbol_col:
        for col in df.columns:
            if 'Symbol' in col and 'Signal' in col and 'Exit' not in col:
                symbol_col = col
                break
    
    exit_col = find_column_by_keywords(df.columns, ['Exit Signal Date', 'Exit Signal', 'Exit'])
    
    # Start with base config if provided
    column_config = base_config.copy() if base_config else {}
    
    # Create config for all columns
    for col in df.columns:
        if col not in column_config:
            # Default config for unpinned columns
            column_config[col] = st.column_config.TextColumn(
                col,
                width="medium"
            )
        
        # Pin Symbol and Exit Signal columns
        if col == symbol_col or col == exit_col:
            # Update existing config or create new one with pinning
            if isinstance(column_config[col], dict):
                column_config[col]['pinned'] = 'left'
            else:
                # If it's already a column config, we need to recreate it with pinned
                column_config[col] = st.column_config.TextColumn(
                    col,
                    width="medium",
                    pinned="left"
                )
    
    return column_config


def get_data_fetch_datetime(json_path=None):
    """
    Read the data fetch datetime from JSON file.
    
    Args:
        json_path: Path to the data_fetch_datetime.json file (uses config default if None)
        
    Returns:
        Dictionary with date, time, datetime, and timezone, or None if file not found
    """
    if json_path is None:
        json_path = str(DATA_FETCH_DATETIME_JSON)
    try:
        json_file = Path(json_path)
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                return data
        return None
    except Exception as e:
        return None


def display_data_fetch_info(json_path=None, location="sidebar"):
    """
    Display the data fetch date and time information.
    
    Args:
        json_path: Path to the data_fetch_datetime.json file (uses config default if None)
        location: Where to display - "sidebar" or "header"
    """
    if json_path is None:
        json_path = str(DATA_FETCH_DATETIME_JSON)
    import streamlit as st
    from datetime import datetime
    
    datetime_info = get_data_fetch_datetime(json_path)
    
    if datetime_info:
        date = datetime_info.get('date', 'N/A')
        time = datetime_info.get('time', 'N/A')
        datetime_str = datetime_info.get('datetime', 'N/A')
        timezone = datetime_info.get('timezone', 'N/A')
        
        if location == "sidebar":
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📅 Data Last Updated")
            st.sidebar.markdown(f"**Date:** {date}")
            st.sidebar.markdown(f"**Time:** {time} {timezone}")
            st.sidebar.caption(f"Last fetch: {datetime_str} {timezone}")
        else:  # header - format nicely
            try:
                # Parse date and time
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%B %d, %Y")
                
                # Parse time and format as 12-hour with AM/PM
                time_obj = datetime.strptime(time, "%H:%M:%S")
                formatted_time = time_obj.strftime("%I:%M %p")
                
                # Get timezone abbreviation (e.g., EST, EDT)
                tz_abbrev = timezone
                try:
                    import pytz
                    tz_map = {
                        "US/Eastern": "US/Eastern",
                        "US/Central": "US/Central",
                        "US/Pacific": "US/Pacific",
                        "America/New_York": "US/Eastern",
                        "America/Chicago": "US/Central",
                        "America/Los_Angeles": "US/Pacific"
                    }
                    tz_name = tz_map.get(timezone, timezone)
                    if tz_name in ["US/Eastern", "US/Central", "US/Pacific"]:
                        tz = pytz.timezone(tz_name)
                        dt = tz.localize(datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S"))
                        tz_abbrev = dt.strftime("%Z")
                except (ImportError, Exception):
                    # Fallback to timezone string if pytz fails
                    tz_abbrev = timezone
                
                st.markdown(f"**📅 Report Date:** {formatted_date} at {formatted_time} {tz_abbrev}")
            except Exception:
                # Fallback to original format if parsing fails
                st.markdown(f"**📅 Report Date:** {date} at {time} {timezone}")
    else:
        if location == "sidebar":
            st.sidebar.markdown("---")
            st.sidebar.caption("⚠️ Data fetch time not available")
        else:  # header
            st.markdown("**📅 Report Date:** Not available")


def format_days(days_str):
    """
    Format days string to appropriate unit:
    - If < 30: output as "X days"
    - If >= 30 and < 360: output as "X months" (30 days = 1 month)
    - If >= 360: output as "X years" (360 days = 12 months = 1 year, 365 days = 1 year)
    
    Args:
        days_str: String containing days (e.g., "15", "45", "400")
    
    Returns:
        Formatted string (e.g., "15 days", "1.5 months", "1.1 years")
    """
    import re
    
    try:
        # Extract numeric value, handling various formats
        days_str_clean = str(days_str).strip()
        # Remove common suffixes to extract the number
        days_str_clean = days_str_clean.replace(" days", "").replace(" day", "").replace(" months", "").replace(" month", "").replace(" years", "").replace(" year", "").replace("(business days)", "").replace("(calendar days)", "").strip()
        # Extract first number (handles decimals and handles cases where there might be text after)
        match = re.search(r'(\d+\.?\d*)', days_str_clean)
        if match:
            days = float(match.group(1))
        else:
            # Fallback: try direct conversion
            days = float(days_str_clean.split()[0] if days_str_clean.split() else days_str_clean)
        
        if days < 30:
            return f"{int(days)} day{'s' if int(days) != 1 else ''}"
        elif days < 360:
            # 30 days = 1 month, so convert to months
            months = days / 30
            if months == int(months):
                # If it's a whole number of months
                month_int = int(months)
                if month_int == 1:
                    return f"1 month"
                else:
                    return f"{month_int} months"
            else:
                return f"{months:.1f} months"
        else:
            # >= 360 days (12 months) = 1 year, convert to years
            # Since 30 days = 1 month, 360 days = 12 months = 1 year
            years = days / 360  # 360 days = 1 year (12 months)
            if years == int(years):
                # If it's a whole number of years
                year_int = int(years)
                if year_int == 1:
                    return f"1 year"
                else:
                    return f"{year_int} years"
            else:
                return f"{years:.1f} years"
    except (ValueError, AttributeError):
        # If parsing fails, return original string
        return str(days_str)


def extract_days_from_formatted_string(formatted_str):
    """
    Extract numeric days from a formatted string that may contain units.
    Handles formats like "60", "2.0 months", "1.5 month", "1 year", "60 days", etc.
    
    Args:
        formatted_str: String that may be formatted (e.g., "2.0 months", "60 days", "60")
    
    Returns:
        Integer number of days
    """
    import re
    
    try:
        # Clean the string
        cleaned = str(formatted_str).strip()
        
        # Extract the numeric value (handles decimals)
        match = re.search(r'(\d+\.?\d*)', cleaned)
        if not match:
            return 0
        
        numeric_value = float(match.group(1))
        
        # Check for units and convert to days
        if 'year' in cleaned.lower():
            return round(numeric_value * 360)  # 360 days = 1 year
        elif 'month' in cleaned.lower():
            return round(numeric_value * 30)  # 30 days = 1 month
        elif 'day' in cleaned.lower():
            return round(numeric_value)
        else:
            # No unit specified, assume it's already in days
            return round(numeric_value)
    except (ValueError, AttributeError, TypeError):
        return 0


def format_days_list(days_list_str, separator="/"):
    """
    Format a list of days separated by a delimiter (e.g., "10/20/30")
    
    Args:
        days_list_str: String containing days separated by delimiter
        separator: Delimiter used (default: "/")
    
    Returns:
        Formatted string with each value formatted appropriately
    """
    try:
        parts = str(days_list_str).split(separator)
        formatted_parts = [format_days(part.strip()) for part in parts]
        return separator.join(formatted_parts)
    except:
        return str(days_list_str)

