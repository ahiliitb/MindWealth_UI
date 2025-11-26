"""
Base parser functions that are reused across different CSV types
"""

import pandas as pd
import re


def parse_signal_csv(df, function_name):
    """Parse signal CSV files with common structure"""
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
            win_rate_info = row.get('Win Rate [%], History Tested, Number of Trades', '')
            win_rate_match = re.search(r'([0-9.]+)%.*?([0-9]+)', str(win_rate_info))
            
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
            'Function': function_name,
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


def parse_detailed_signal_csv(df):
    """Parse detailed signal CSV files with Function column"""
    processed_data = []
    
    for idx, row in df.iterrows():
        try:
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
                # Fallback: try to extract symbol from the beginning of the string
                parts = str(symbol_info).split(',')
                if len(parts) >= 1:
                    symbol = parts[0].strip()
                    signal_type = parts[1].strip() if len(parts) >= 2 else "Unknown"
                    # Try to extract date and price from the string
                    date_match = re.search(r'([0-9]{4}-[0-9]{2}-[0-9]{2})', str(symbol_info))
                    signal_date = date_match.group(1) if date_match else "Unknown"
                    price_match = re.search(r'Price:\s*([0-9.]+)', str(symbol_info))
                    signal_price = float(price_match.group(1)) if price_match else 0
                else:
                    symbol, signal_type, signal_date, signal_price = "Unknown", "Unknown", "Unknown", 0
            
            # Parse win rate and number of trades
            win_rate_info = row.get('Win Rate [%], History Tested, Number of Trades', '')
            win_rate_match = re.search(r'([0-9.]+)%.*?([0-9]+)', str(win_rate_info))
            
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
                'Function': row.get('Function', 'Unknown'),
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
        except Exception as e:
            # Log error but still add the row with default values to ensure no data is lost
            import sys
            print(f"Warning: Error parsing row {idx}: {e}", file=sys.stderr)
            processed_data.append({
                'Function': row.get('Function', 'Unknown'),
                'Symbol': 'Unknown',
                'Signal_Type': 'Unknown',
                'Signal_Date': 'Unknown',
                'Signal_Price': 0,
                'Win_Rate': 0,
                'Num_Trades': 0,
                'Strategy_CAGR': 0,
                'Buy_Hold_CAGR': 0,
                'Strategy_Sharpe': 0,
                'Buy_Hold_Sharpe': 0,
                'Best_Return': 0,
                'Worst_Return': 0,
                'Avg_Return': 0,
                'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
                'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
                'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
                'Raw_Data': row.to_dict()
            })
    
    return pd.DataFrame(processed_data)


def parse_performance_csv(df):
    """Parse performance CSV files"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Extract win rate from percentage string
        win_rate_str = str(row.get('Win Percentage', '0%')).replace('%', '')
        try:
            win_rate = float(win_rate_str)
        except:
            win_rate = 0
        
        # Extract number of trades
        try:
            num_trades = int(row.get('Total Analysed Trades', 0))
        except:
            num_trades = 0
        
        # Extract average profit
        profit_str = str(row.get('Profit [%] (Best/Worst/Avg.)', '0/0/0'))
        profit_match = re.search(r'([0-9.-]+)%/([0-9.-]+)%/([0-9.-]+)%', profit_str)
        
        if profit_match:
            try:
                best_return = float(profit_match.group(1))
                worst_return = float(profit_match.group(2))
                avg_return = float(profit_match.group(3))
            except:
                best_return, worst_return, avg_return = 0, 0, 0
        else:
            best_return, worst_return, avg_return = 0, 0, 0
        
        # Extract holding period information
        holding_period_str = str(row.get('Holding Period (days) (Max./Min./Avg.)', '0/0/0'))
        holding_match = re.search(r'([0-9.]+) days/([0-9.]+) days/([0-9.]+) days', holding_period_str)
        
        if holding_match:
            try:
                max_holding = float(holding_match.group(1))
                min_holding = float(holding_match.group(2))
                avg_holding = float(holding_match.group(3))
            except:
                max_holding, min_holding, avg_holding = 0, 0, 0
        else:
            max_holding, min_holding, avg_holding = 0, 0, 0
        
        # Extract average backtested win rate
        try:
            avg_backtested_win_rate = float(str(row.get('Avg Backtested Win Rate [%]', '0%')).replace('%', ''))
        except:
            avg_backtested_win_rate = 0
        
        # Extract average backtested holding period
        try:
            avg_backtested_holding = float(str(row.get('Avg Backtested Holding Period (days)', '0')).replace(' days', ''))
        except:
            avg_backtested_holding = 0
        
        processed_data.append({
            'Strategy': row.get('Strategy', 'Unknown'),
            'Interval': row.get('Interval', 'Unknown'),
            'Signal_Type': row.get('Signal Type', 'Unknown'),
            'Total_Trades': num_trades,
            'Win_Percentage': win_rate,
            'Max_Holding_Days': max_holding,
            'Min_Holding_Days': min_holding,
            'Avg_Holding_Days': avg_holding,
            'Best_Profit': best_return,
            'Worst_Profit': worst_return,
            'Avg_Profit': avg_return,
            'Avg_Backtested_Win_Rate': avg_backtested_win_rate,
            'Avg_Backtested_Holding_Days': avg_backtested_holding,
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

