"""
Data loading utilities for CSV and stock data files
"""

import pandas as pd
import streamlit as st
import csv
import io
from datetime import datetime, timedelta

from .file_discovery import detect_csv_structure
from ..parsers import (
    parse_bollinger_band, parse_distance, parse_fib_ret, parse_general_divergence,
    parse_new_high, parse_stochastic_divergence, parse_sigma, parse_sentiment,
    parse_trendline, parse_outstanding_signal, parse_outstanding_exit_signal,
    parse_new_signal, parse_target_signals, parse_breadth,
    parse_latest_performance, parse_forward_backtesting, parse_signal_csv
)


@st.cache_data
def load_data_from_file(file_path, page_name="Unknown"):
    """Load and process trading data from any CSV file with specific parsers"""
    try:
        # Detect CSV structure
        csv_type = detect_csv_structure(file_path)
        
        # Load the full CSV with special handling for sentiment.csv
        if 'sentiment.csv' in file_path:
            # Handle sentiment.csv with complex column names - use manual parsing
            try:
                # Read the file and manually parse it
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use csv.Sniffer to detect the dialect
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(content[:1000])
                
                # Parse the CSV manually
                reader = csv.reader(io.StringIO(content), dialect=dialect)
                rows = list(reader)
                
                if len(rows) < 2:
                    st.warning(f"No data found in {file_path}")
                    return pd.DataFrame()
                
                # Create DataFrame from parsed rows
                df = pd.DataFrame(rows[1:], columns=rows[0])
                
            except Exception as e:
                st.error(f"Error parsing sentiment.csv: {str(e)}")
                return pd.DataFrame()
        else:
            df = pd.read_csv(file_path)
        
        if df.empty:
            st.warning(f"No data found in {file_path}")
            return pd.DataFrame()
        
        # Parse based on detected structure using specific parsers
        parser_mapping = {
            'bollinger_band': parse_bollinger_band,
            'distance': parse_distance,
            'fib_ret': parse_fib_ret,
            'general_divergence': parse_general_divergence,
            'new_high': parse_new_high,
            'stochastic_divergence': parse_stochastic_divergence,
            'sigma': parse_sigma,
            'sentiment': parse_sentiment,
            'trendline': parse_trendline,
            'breadth': parse_breadth,
            'outstanding_signal': parse_outstanding_signal,
            'outstanding_exit_signal': parse_outstanding_exit_signal,
            'new_signal': parse_new_signal,
            'target_signal': parse_target_signals,
            'latest_performance': parse_latest_performance,
            'forward_backtesting': parse_forward_backtesting
        }
        
        if csv_type in parser_mapping:
            return parser_mapping[csv_type](df)
        else:
            # Fallback to basic parsing for unknown structures
            st.warning(f"Unknown CSV structure for {file_path}, using basic parsing")
            return parse_signal_csv(df, page_name)
            
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame()


def load_stock_data_file(symbol, start_date, end_date, interval='Daily'):
    """Load and process stock data from CSV file for a given symbol"""
    import os
    
    # Map interval to pandas frequency
    INTERVAL_LETTER_DICT = {
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Yearly': 'Y'
    }
    
    # Construct the file path - now using CSV files
    csv_file_path = f'./trade_store/stock_data/{symbol}.csv'
    
    if not os.path.exists(csv_file_path):
        return None
    
    try:
        # Read CSV file
        # Assuming CSV format with columns: Date, Open, High, Low, Close, Volume
        df = pd.read_csv(csv_file_path)
        
        # Check if CSV is empty
        if df.empty:
            return None
        
        # Convert Date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'Date'}, inplace=True)
        else:
            st.error(f"No 'Date' column found in {csv_file_path}")
            return None
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Ensure we have required OHLC columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in {csv_file_path}: {missing_cols}")
            return None
        
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Convert to required interval if not Daily
        if interval != 'Daily':
            agg_dict = {
                'Open': 'first',
                'Close': 'last',
                'High': 'max',
                'Low': 'min'
            }
            
            # Add Volume to aggregation if it exists
            if 'Volume' in df.columns:
                agg_dict['Volume'] = 'sum'
            
            df = df.groupby(pd.Grouper(freq=INTERVAL_LETTER_DICT[interval])).agg(agg_dict)
            df = df.dropna()
        
        df.reset_index(inplace=True)
        return df
        
    except Exception as e:
        st.error(f"Error reading CSV file for {symbol}: {str(e)}")
        return None

