"""
Parsers for advanced signal CSV files (outstanding, new signals, breadth, etc.)
"""

import pandas as pd
import re
from .base_parsers import parse_detailed_signal_csv


def parse_outstanding_signal(df):
    """Parse outstanding_signal.csv"""
    return parse_detailed_signal_csv(df)


def parse_outstanding_exit_signal(df):
    """Parse outstanding_exit_signal.csv"""
    return parse_detailed_signal_csv(df)


def parse_new_signal(df):
    """Parse new_signal.csv"""
    return parse_detailed_signal_csv(df)


def parse_breadth(df):
    """Parse breadth.csv"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Extract function name
        function = row.get('Function', 'Unknown')
        
        # Extract bullish asset percentage
        bullish_asset_str = str(row.get('Bullish Asset vs Total Asset (%).', '0%')).replace('%', '')
        try:
            bullish_asset_pct = float(bullish_asset_str)
        except:
            bullish_asset_pct = 0
        
        # Extract bullish signal percentage
        bullish_signal_str = str(row.get('Bullish Signal vs Total Signal (%)', '0%')).replace('%', '')
        try:
            bullish_signal_pct = float(bullish_signal_str)
        except:
            bullish_signal_pct = 0
        
        processed_data.append({
            'Function': function,
            'Bullish_Asset_Percentage': bullish_asset_pct,
            'Bullish_Signal_Percentage': bullish_signal_pct,
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)


def parse_target_signals(df, page_name="Unknown"):
    """Parse target signals CSV (target_signal.csv)"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Parse symbol and signal info
        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', '')
        symbol_match = re.search(r'([^,]+),\s*([^,]+),\s*([^(]+)\(Price:\s*([^)]+)\)', str(symbol_info))
        
        if symbol_match:
            symbol = symbol_match.group(1).strip()
            signal_type = symbol_match.group(2).strip()
            signal_date = symbol_match.group(3).strip()
            try:
                signal_price = float(symbol_match.group(4).strip())
            except:
                signal_price = 0
        else:
            symbol, signal_type, signal_date, signal_price = "Unknown", "Unknown", "Unknown", 0
        
        # Parse win rate and number of trades
        win_rate_info = row.get('Number of Trades/Historic Win Rate [%]', '')
        win_rate_match = re.search(r'([0-9]+)/([0-9.]+)%', str(win_rate_info))
        
        if win_rate_match:
            try:
                num_trades = int(win_rate_match.group(1))
                win_rate = float(win_rate_match.group(2))
            except:
                win_rate, num_trades = 0, 0
        else:
            win_rate, num_trades = 0, 0
        
        # Parse current trading date and price
        current_info = row.get('Current Trading Date/Price[$]', '')
        current_match = re.search(r'([^(]+)\(Price:\s*([^)]+)\)', str(current_info))
        if current_match:
            current_date = current_match.group(1).strip()
            try:
                current_price = float(current_match.group(2).strip())
            except:
                current_price = 0
        else:
            current_date, current_price = "Unknown", 0
        
        # Parse entry signal date and price
        entry_info = row.get('Entry Signal Date/Price[$]', '')
        entry_match = re.search(r'([^(]+)\(Price:\s*([^)]+)\)', str(entry_info))
        
        if entry_match:
            entry_date = entry_match.group(1).strip()
            try:
                entry_price = float(entry_match.group(2).strip())
            except:
                entry_price = 0
        else:
            entry_date, entry_price = "Unknown", 0
        
        # Parse gain and holding period
        gain_info = row.get('% Gain, Holding Period (days)', '')
        gain_match = re.search(r'([0-9.]+)%,\s*([0-9]+)\s*days', str(gain_info))
        
        if gain_match:
            try:
                gain_pct = float(gain_match.group(1))
                holding_days = int(gain_match.group(2))
            except:
                gain_pct, holding_days = 0, 0
        else:
            gain_pct, holding_days = 0, 0
        
        # Parse backtested returns
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
        
        # Parse interval and function
        interval = row.get('Interval', 'Unknown')
        function = row.get('Function', 'Unknown')
        
        # Parse target information
        target_info = row.get('Target for which Price has achieved over 90 percent of gain %', '')
        target_price = 0
        target_type = "Unknown"
        
        if '(' in str(target_info) and ')' in str(target_info):
            # Extract price and type from format like "0.8118 (Historic Rise or Fall to Pivot)"
            target_match = re.search(r'([0-9.]+)\s*\(([^)]+)\)', str(target_info))
            if target_match:
                try:
                    target_price = float(target_match.group(1))
                    target_type = target_match.group(2).strip()
                except:
                    target_price, target_type = 0, "Unknown"
        
        # Parse next targets
        next_targets = row.get('Next Two Target % from Latest Trading Price', 'N/A')
        
        # Parse remaining potential exit prices
        exit_prices = row.get('Remaining Potential Exit Prices [$]', 'N/A')
        
        # Calculate performance metrics (simplified)
        strategy_cagr = 0
        buy_hold_cagr = 0
        strategy_sharpe = 0
        buy_hold_sharpe = 0
        
        # Try to extract from performance data if available
        performance_info = row.get('Latest Past 6 Months Performance[%]/No. of Analysed Trades/Avg Holding Period (days) (Across ALL Assets)', '')
        if performance_info and '/' in str(performance_info):
            try:
                perf_parts = str(performance_info).split('/')
                if len(perf_parts) >= 1:
                    # Use performance percentage as a rough CAGR estimate
                    strategy_cagr = float(perf_parts[0].replace('%', ''))
            except:
                pass
        
        processed_data.append({
            'Symbol': symbol,
            'Function': function,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Current_Date': current_date,
            'Current_Price': current_price,
            'Gain_Percentage': gain_pct,
            'Holding_Days': holding_days,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Target_Price': target_price,
            'Target_Type': target_type,
            'Next_Targets': next_targets,
            'Exit_Prices': exit_prices,
            'Interval': interval,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

