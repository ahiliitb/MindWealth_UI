"""
Parsers for individual signal strategy CSV files
"""

import pandas as pd
import re
from .base_parsers import parse_signal_csv


def parse_bollinger_band(df):
    """Parse bollinger_band.csv"""
    return parse_signal_csv(df, 'Band Matrix')


def parse_distance(df):
    """Parse Distance.csv"""
    return parse_signal_csv(df, 'DeltaDrift')


def parse_fib_ret(df):
    """Parse Fib-Ret.csv"""
    return parse_signal_csv(df, 'Fractal Track')


def parse_general_divergence(df):
    """Parse General-Divergence.csv"""
    return parse_signal_csv(df, 'BaselineDiverge')


def parse_new_high(df):
    """Parse new_high.csv"""
    return parse_signal_csv(df, 'Altitude Alpha')


def parse_stochastic_divergence(df):
    """Parse Stochastic-Divergence.csv"""
    return parse_signal_csv(df, 'Oscillator Delta')


def parse_sigma(df):
    """Parse sigma.csv"""
    return parse_signal_csv(df, 'SigmaShell')


def parse_sentiment(df):
    """Parse sentiment.csv with specific handling for quoted first column"""
    processed_data = []

    for _, row in df.iterrows():
        # Check if 'Signal Open Price' column exists and has a valid value
        signal_open_price = row.get('Signal Open Price', '')
        if signal_open_price and str(signal_open_price).strip():
            try:
                signal_price = float(str(signal_open_price).strip())
            except:
                signal_price = 0
        else:
            # Fallback to parsing from the complex string
            signal_price = 0

        # Parse symbol and signal info - handle quoted first column
        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', '')
        # Remove quotes and parse
        symbol_info_clean = str(symbol_info).strip('"')
        symbol_match = re.search(r'([^,]+),\s*([^,]+),\s*([^(]+)\(Price:\s*([^)]+)\)', symbol_info_clean)

        if symbol_match:
            symbol = symbol_match.group(1).strip()
            signal_type = symbol_match.group(2).strip()
            signal_date = symbol_match.group(3).strip()
            # Use the extracted signal_price if not already set from Signal Open Price column
            if signal_price == 0:
                try:
                    signal_price = float(symbol_match.group(4).strip())
                except:
                    signal_price = 0
        else:
            symbol, signal_type, signal_date = "Unknown", "Unknown", "Unknown"
        
        # Parse win rate and number of trades
        win_rate_info = row.get('Win Rate [%], History Tested, Number of Trades', '')
        win_rate_match = re.search(r'([0-9.]+)%.*?([0-9]+)$', str(win_rate_info))
        
        if win_rate_match:
            try:
                win_rate = float(win_rate_match.group(1))
                num_trades = int(win_rate_match.group(2))
            except:
                win_rate, num_trades = 0, 0
        else:
            win_rate, num_trades = 0, 0
        
        # Parse CAGR
        strategy_cagr = 0
        buy_hold_cagr = 0
        if 'Backtested Strategy CAGR [%]' in row:
            try:
                strategy_cagr = float(str(row['Backtested Strategy CAGR [%]']).replace('%', ''))
            except:
                strategy_cagr = 0
        if 'CAGR of Buy and Hold [%]' in row:
            try:
                buy_hold_cagr = float(str(row['CAGR of Buy and Hold [%]']).replace('%', ''))
            except:
                buy_hold_cagr = 0
        
        # Parse Sharpe ratios
        strategy_sharpe = 0
        buy_hold_sharpe = 0
        if 'Backtested Strategy Sharpe Ratio' in row:
            try:
                strategy_sharpe = float(row['Backtested Strategy Sharpe Ratio'])
            except:
                strategy_sharpe = 0
        if 'Sharpe Ratio of Buy and Hold' in row:
            try:
                buy_hold_sharpe = float(row['Sharpe Ratio of Buy and Hold'])
            except:
                buy_hold_sharpe = 0
        
        # Parse returns
        returns_info = row.get('Backtested Returns(Win Trades) [%] (Best/Worst/Avg)', '')
        returns_match = re.search(r'([0-9.]+)%/([0-9.]+)%/([0-9.]+)%', str(returns_info))
        
        if returns_match:
            try:
                best_return = float(returns_match.group(1))
                worst_return = float(returns_match.group(2))
                avg_return = float(returns_match.group(3))
            except:
                best_return, worst_return, avg_return = 0, 0, 0
        else:
            best_return, worst_return, avg_return = 0, 0, 0
        
        processed_data.append({
            'Function': 'PulseGauge',
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)


def parse_trendline(df):
    """Parse Trendline.csv"""
    return parse_signal_csv(df, 'TrendPulse')

