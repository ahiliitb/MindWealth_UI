"""
Utility module for managing monitored trades
Stores and updates personal portfolio trades
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re


MONITORED_TRADES_FILE = "monitored_trades.json"


def get_monitored_trades_path() -> Path:
    """Get the path to the monitored trades storage file"""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / MONITORED_TRADES_FILE


def generate_trade_id(symbol: str, signal_date: str, interval: str, signal_type: str, function: str) -> str:
    """Generate a unique ID for a trade based on its identifying characteristics"""
    # Normalize values
    symbol = str(symbol).strip().upper()
    signal_date = str(signal_date).strip()
    interval = str(interval).strip()
    signal_type = str(signal_type).strip()
    function = str(function).strip()
    
    # Create a unique identifier
    trade_id = f"{symbol}_{signal_date}_{interval}_{signal_type}_{function}"
    # Replace any problematic characters
    trade_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', trade_id)
    return trade_id


def load_monitored_trades() -> pd.DataFrame:
    """Load monitored trades from JSON file"""
    file_path = get_monitored_trades_path()
    
    if not file_path.exists():
        return pd.DataFrame()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data or 'trades' not in data:
            return pd.DataFrame()
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data['trades'])
        return df
    except Exception as e:
        print(f"Error loading monitored trades: {e}")
        return pd.DataFrame()


def save_monitored_trades(df: pd.DataFrame) -> bool:
    """Save monitored trades to JSON file"""
    file_path = get_monitored_trades_path()
    
    try:
        # Convert DataFrame to list of dicts
        trades = df.to_dict('records') if not df.empty else []
        
        data = {
            'last_updated': datetime.now().isoformat(),
            'trades': trades
        }
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return True
    except Exception as e:
        print(f"Error saving monitored trades: {e}")
        return False


def add_trade_to_monitored(trade_data: Dict) -> bool:
    """Add a new trade to monitored trades"""
    df = load_monitored_trades()
    
    # Generate unique ID
    trade_id = generate_trade_id(
        trade_data.get('Symbol', ''),
        trade_data.get('Signal_Date', ''),
        trade_data.get('Interval', ''),
        trade_data.get('Signal_Type', ''),
        trade_data.get('Function', '')
    )
    
    # Check if trade already exists
    if not df.empty and 'Trade_ID' in df.columns:
        if trade_id in df['Trade_ID'].values:
            return False  # Trade already exists
    
    # Add trade ID and metadata
    trade_data['Trade_ID'] = trade_id
    trade_data['Added_Date'] = datetime.now().isoformat()
    trade_data['Last_Updated'] = datetime.now().isoformat()
    
    # Initialize exit fields if not present
    if 'Exit_Date' not in trade_data:
        trade_data['Exit_Date'] = None
    if 'Exit_Price' not in trade_data:
        trade_data['Exit_Price'] = None
    if 'Current_Price' not in trade_data:
        trade_data['Current_Price'] = None
    if 'Current_Date' not in trade_data:
        trade_data['Current_Date'] = None
    
    # Add to DataFrame
    new_row = pd.DataFrame([trade_data])
    if df.empty:
        df = new_row
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    
    return save_monitored_trades(df)


def remove_trade_from_monitored(trade_id: str) -> bool:
    """Remove a trade from monitored trades"""
    df = load_monitored_trades()
    
    if df.empty or 'Trade_ID' not in df.columns:
        return False
    
    # Filter out the trade
    df = df[df['Trade_ID'] != trade_id]
    
    return save_monitored_trades(df)


def get_latest_price_from_stock_data(symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """Get the latest price and date from stock_data CSV file"""
    project_root = Path(__file__).resolve().parent.parent.parent
    stock_data_path = project_root / "trade_store" / "stock_data" / f"{symbol}.csv"
    
    if not stock_data_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(stock_data_path)
        
        if df.empty:
            return None, None
        
        # Find Date column (case-insensitive)
        date_col = None
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
                break
        
        if date_col is None:
            return None, None
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col, ascending=False)
        
        # Get the latest row
        latest_row = df.iloc[0]
        
        # Find Close or Price column
        price = None
        for col in ['Close', 'close', 'Price', 'price', 'Adj Close', 'Adj Close']:
            if col in df.columns:
                price = latest_row[col]
                break
        
        if price is None and len(df.columns) > 1:
            # Try to find numeric column
            for col in df.columns:
                if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                    price = latest_row[col]
                    break
        
        date_str = latest_row[date_col]
        if pd.notna(date_str):
            if isinstance(date_str, pd.Timestamp):
                date_str = date_str.strftime('%Y-%m-%d')
            else:
                date_str = str(date_str)
        
        if price is not None:
            try:
                price = float(price)
                return price, date_str
            except:
                return None, None
        
        return None, None
    except Exception as e:
        print(f"Error reading stock data for {symbol}: {e}")
        return None, None


def update_monitored_trades_prices() -> bool:
    """Update current prices for all monitored trades"""
    df = load_monitored_trades()
    
    if df.empty:
        return True
    
    updated = False
    for idx, row in df.iterrows():
        symbol = row.get('Symbol', '')
        if not symbol:
            continue
        
        # Get latest price
        current_price, current_date = get_latest_price_from_stock_data(symbol)
        
        if current_price is not None:
            df.at[idx, 'Current_Price'] = current_price
            df.at[idx, 'Current_Date'] = current_date
            df.at[idx, 'Last_Updated'] = datetime.now().isoformat()
            updated = True
    
    if updated:
        return save_monitored_trades(df)
    
    return True


def check_exit_signals_in_outstanding(df: pd.DataFrame, outstanding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for exit signals in outstanding signals for monitored trades.
    Updates Exit_Date and Exit_Price if exit signal is found.
    
    Args:
        df: Monitored trades DataFrame
        outstanding_df: Outstanding signals DataFrame
    
    Returns:
        Updated monitored trades DataFrame
    """
    if df.empty or outstanding_df.empty:
        return df
    
    # Ensure required columns exist
    required_cols = ['Symbol', 'Signal_Date', 'Interval', 'Signal_Type', 'Function']
    if not all(col in df.columns for col in required_cols):
        return df
    
    if 'Symbol' not in outstanding_df.columns:
        return df
    
    # Parse exit signal column from outstanding signals
    exit_col = 'Exit Signal Date/Price[$]'
    if exit_col not in outstanding_df.columns:
        return df
    
    for idx, monitored_row in df.iterrows():
        # Skip if already has exit
        if pd.notna(monitored_row.get('Exit_Date')) and monitored_row.get('Exit_Date'):
            continue
        
        # Match criteria
        symbol = str(monitored_row['Symbol']).strip().upper()
        signal_date = str(monitored_row['Signal_Date']).strip()
        interval = str(monitored_row['Interval']).strip()
        signal_type = str(monitored_row['Signal_Type']).strip()
        function = str(monitored_row['Function']).strip()
        
        # Find matching signal in outstanding - match on symbol and function first
        matches = outstanding_df[
            (outstanding_df['Symbol'].str.strip().str.upper() == symbol) &
            (outstanding_df['Function'].str.strip() == function)
        ]
        
        if matches.empty:
            continue
        
        # Check each match for exit signal and verify it's the same trade
        for _, match_row in matches.iterrows():
            # Verify this is the same signal by checking signal date and type
            signal_info = str(match_row.get('Symbol, Signal, Signal Date/Price[$]', ''))
            
            # Check if signal date matches
            if signal_date not in signal_info:
                continue
            
            # Check if signal type matches (Long/Short)
            if signal_type.upper() not in signal_info.upper():
                continue
            
            # Check interval if available in outstanding data
            if 'Interval' in match_row.index:
                match_interval = str(match_row['Interval']).strip()
                if interval and match_interval and interval.lower() not in match_interval.lower() and match_interval.lower() not in interval.lower():
                    continue
            
            # Now check for exit signal
            exit_info = str(match_row.get(exit_col, ''))
            
            # Check if exit exists
            if 'No Exit Yet' in exit_info or not exit_info or exit_info.lower() in ['nan', 'none', '']:
                continue
            
            # Extract exit date and price
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', exit_info)
            price_match = re.search(r'Price:\s*([\d.]+)', exit_info)
            
            if date_match and price_match:
                exit_date = date_match.group(1)
                try:
                    exit_price = float(price_match.group(1))
                    
                    # Update exit information
                    df.at[idx, 'Exit_Date'] = exit_date
                    df.at[idx, 'Exit_Price'] = exit_price
                    df.at[idx, 'Last_Updated'] = datetime.now().isoformat()
                    break
                except:
                    continue
    
    return df


def update_monitored_trades_with_outstanding(outstanding_df: pd.DataFrame) -> bool:
    """Update monitored trades with exit signals from outstanding signals"""
    df = load_monitored_trades()
    
    if df.empty:
        return True
    
    # Update prices first
    update_monitored_trades_prices()
    df = load_monitored_trades()  # Reload after price update
    
    # Check for exit signals
    df = check_exit_signals_in_outstanding(df, outstanding_df)
    
    return save_monitored_trades(df)

