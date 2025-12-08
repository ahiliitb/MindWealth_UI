"""
Helper utility functions
"""
import pandas as pd


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

