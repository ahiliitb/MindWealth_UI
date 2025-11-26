#!/usr/bin/env python3
"""
Convert trading signal CSV files to chatbot data structure.
Extracts Symbol and Signal Date from signal files and organizes them into:
chatbot/data/{Symbol}/YYYY-MM-DD.csv

Features:
- Automatic deduplication based on DEDUP_COLUMNS from .env
- Prevents duplicate rows when appending to existing files
"""

import pandas as pd
import re
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()


def parse_symbol_signal_column(value):
    """
    Parse the "Symbol, Signal, Signal Date/Price[$]" column.
    
    Example: "ETH-USD, Long, 2025-10-10 (Price: 4369.1436)"
    Returns: ("ETH-USD", "2025-10-10", "Long", 4369.1436)
    """
    try:
        # Handle NaN, None, or empty values
        if value is None or pd.isna(value):
            return None, None, None, None
        
        # Convert to string to handle any numeric types
        value_str = str(value).strip()
        
        if not value_str or value_str.lower() in ['nan', 'none', '']:
            return None, None, None, None
        
        # Split by comma
        parts = [p.strip() for p in value_str.split(',')]
        
        if len(parts) < 3:
            return None, None, None, None
        
        symbol = parts[0]
        signal_type = parts[1]
        
        # Extract date and price from third part
        # Format: "2025-10-10 (Price: 4369.1436)"
        date_price_part = parts[2]
        
        # Extract date using regex
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_price_part)
        date = date_match.group(1) if date_match else None
        
        # Extract price using regex
        price_match = re.search(r'Price:\s*([\d.]+)', date_price_part)
        price = float(price_match.group(1)) if price_match else None
        
        return symbol, date, signal_type, price
        
    except Exception as e:
        print(f"Error parsing: {value} - {e}")
        return None, None, None, None


def parse_exit_signal_column(value):
    """
    Parse the "Exit Signal Date/Price[$]" column.
    
    Examples:
        "No Exit Yet" -> (None, None)
        "2025-10-10 (Price: 5.98) (Today)" -> ("2025-10-10", 5.98)
    
    Returns: (exit_date, exit_price) or (None, None) if no exit
    """
    try:
        # Handle NaN, None, or empty values
        if value is None or pd.isna(value):
            return None, None
        
        # Convert to string to handle any numeric types
        value_str = str(value).strip()
        
        if not value_str or value_str.lower() in ['nan', 'none', '']:
            return None, None
        
        # Check if no exit
        if "No Exit Yet" in value_str:
            return None, None
        
        # Extract date using regex
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', value_str)
        exit_date = date_match.group(1) if date_match else None
        
        # Extract price using regex
        price_match = re.search(r'Price:\s*([\d.]+)', value_str)
        exit_price = float(price_match.group(1)) if price_match else None
        
        return exit_date, exit_price
        
    except Exception as e:
        print(f"Error parsing exit: {value} - {e}")
        return None, None


def parse_interval_from_status(value):
    """
    Parse the interval from "Interval, Confirmation Status" column.
    
    Example: "Daily, is CONFIRMED on 2025-11-18" -> "Daily"
    Example: "Weekly, Nullified" -> "Weekly"
    
    Args:
        value: Value from "Interval, Confirmation Status" column
    
    Returns:
        Interval string (e.g., "Daily", "Weekly") or None
    """
    try:
        if not value or pd.isna(value):
            return None
        
        # Split by comma and take the first part (interval)
        parts = str(value).split(',')
        if parts:
            interval = parts[0].strip()
            return interval if interval else None
        
        return None
        
    except Exception as e:
        print(f"  ⚠ Error parsing interval: {e}")
        return None


def is_confirmed_signal(confirmation_status_value):
    """
    Check if the confirmation status indicates the signal is or was confirmed.
    
    Only accepts:
    - "is CONFIRMED" (current confirmed status)
    - "was CONFIRMED" (previously confirmed status)
    
    Rejects:
    - "will be confirmed"
    - "nullified"
    - Any other status
    
    Args:
        confirmation_status_value: Value from "Interval, Confirmation Status" column
            Example: "Daily, is CONFIRMED on 2025-11-18"
    
    Returns:
        True if signal is confirmed, False otherwise
    """
    try:
        if not confirmation_status_value or pd.isna(confirmation_status_value):
            return False
        
        status_str = str(confirmation_status_value).strip()
        
        # Check for "is CONFIRMED" or "was CONFIRMED" (case-insensitive)
        # Using regex to match these patterns
        confirmed_pattern = re.search(r'\b(is|was)\s+CONFIRMED\b', status_str, re.IGNORECASE)
        
        if confirmed_pattern:
            return True
        
        return False
        
    except Exception as e:
        print(f"  ⚠ Error checking confirmation status: {e}")
        return False


def get_dedup_columns(signal_type="entry"):
    """
    Get deduplication columns based on signal type.
    
    Different signal types use different deduplication keys:
    - entry: Date, Symbol, Interval, Signal
    - exit: Exit Signal Date, Symbol, Signal, Interval, Signal Date
    - target: Symbol, Signal, Interval, Signal Date, Target for which Price has achieved over 90 percent of gain %, Backtested Target Exit Date, Exit Signal Date/Price[$]
    - breadth: Date
    
    Args:
        signal_type: Type of signal ('entry', 'exit', 'target', 'breadth')
    
    Returns:
        List of column names to use for deduplication
    """
    if signal_type == "entry":
        dedup_cols_str = os.getenv("ENTRY_DEDUP_COLUMNS", "Date,Symbol,Interval,Signal")
    elif signal_type == "exit":
        dedup_cols_str = os.getenv("EXIT_DEDUP_COLUMNS", "Exit Signal Date,Symbol,Signal,Interval,Signal Date")
    elif signal_type == "target":
        dedup_cols_str = os.getenv("TARGET_DEDUP_COLUMNS", "Symbol,Signal,Interval,Signal Date,Target for which Price has achieved over 90 percent of gain %,Backtested Target Exit Date,Exit Signal Date/Price[$]")
    elif signal_type == "breadth":
        dedup_cols_str = os.getenv("BREADTH_DEDUP_COLUMNS", "Date")
    else:
        # Fallback to entry defaults
        dedup_cols_str = os.getenv("DEDUP_COLUMNS", "Date,Symbol,Interval,Signal")
    
    dedup_cols = [col.strip() for col in dedup_cols_str.split(",")]
    return dedup_cols


def deduplicate_dataframe(df, dedup_columns=None, signal_type="entry"):
    """
    Remove duplicates from dataframe based on specified columns.
    
    Args:
        df: DataFrame to deduplicate
        dedup_columns: List of column names to check for duplicates (if None, uses signal_type to get defaults)
        signal_type: Type of signal ('entry', 'exit', 'target', 'breadth') - used if dedup_columns is None
        
    Returns:
        Deduplicated DataFrame
    """
    if dedup_columns is None:
        dedup_columns = get_dedup_columns(signal_type)
    
    # Only use columns that exist in the dataframe
    available_cols = [col for col in dedup_columns if col in df.columns]
    
    if available_cols:
        original_count = len(df)
        df = df.drop_duplicates(subset=available_cols, keep='first')
        removed_count = original_count - len(df)
        
        if removed_count > 0:
            print(f"  ℹ Removed {removed_count} duplicate rows based on: {', '.join(available_cols)}")
    
    return df


def check_target_duplicate(row, master_csv_path):
    """
    Check if target signal already exists in master CSV based on the three key columns.
    
    Args:
        row: DataFrame row to check
        master_csv_path: Path to all_targets.csv
        
    Returns:
        True if duplicate found, False otherwise
    """
    import pandas as pd
    from pathlib import Path
    
    # Target dedup columns - THESE THREE COLUMNS MUST MATCH FOR DUPLICATE
    dedup_cols = [
        "Symbol",
        "Target for which Price has achieved over 90 percent of gain %",
        "Entry Signal Date/Price[$]"
    ]
    
    master_file = Path(master_csv_path)
    
    # If master file doesn't exist, not a duplicate
    if not master_file.exists():
        return False
    
    try:
        master_df = pd.read_csv(master_file)
        
        # Check which columns exist in both row and master
        available_cols = [col for col in dedup_cols if col in master_df.columns and col in row.index]
        
        if not available_cols:
            print(f"  ⚠ Warning: No dedup columns found for comparison")
            return False
        
        # Check for exact match on ALL three columns
        for _, existing_row in master_df.iterrows():
            match = True
            for col in available_cols:
                # Convert to string for comparison to handle different data types
                existing_val = str(existing_row[col]).strip()
                new_val = str(row[col]).strip()
                
                if existing_val != new_val:
                    match = False
                    break
            
            if match:
                # Found exact duplicate
                print(f"  🚫 Duplicate found: {row.get('Symbol', 'N/A')} - {row.get('Entry Signal Date/Price[$]', 'N/A')}")
                return True
        
        return False
        
    except Exception as e:
        print(f"  ⚠ Error checking duplicate: {e}")
        return False


def add_to_master_targets(row, master_csv_path):
    """
    Add target signal to master CSV (all_targets.csv).
    This maintains a complete record of all unique targets for deduplication.
    
    Args:
        row: DataFrame row to add
        master_csv_path: Path to all_targets.csv
    """
    import pandas as pd
    from pathlib import Path
    
    master_file = Path(master_csv_path)
    
    try:
        if master_file.exists():
            # Read existing master file
            master_df = pd.read_csv(master_file)
            # Append new row
            master_df = pd.concat([master_df, pd.DataFrame([row])], ignore_index=True)
            print(f"  ✅ Added to master CSV: {row.get('Symbol', 'N/A')} (Total: {len(master_df)} targets)")
        else:
            # Create new master file
            master_df = pd.DataFrame([row])
            print(f"  ✅ Created master CSV with first target: {row.get('Symbol', 'N/A')}")
        
        # Save updated master file
        master_df.to_csv(master_file, index=False)
        
    except Exception as e:
        print(f"  ⚠ Error updating master targets: {e}")


def convert_signal_file_to_data_structure(
    input_file,
    signal_type="signal",
    output_base_dir="chatbot/data",
    overwrite=False,
    dedup_columns=None
):
    """
    Convert a trading signal/target CSV file to the data folder structure.
    Automatically deduplicates data based on DEDUP_COLUMNS from .env.
    
    For signals: Splits into entry/ and exit/ folders based on exit date:
        - If exit date exists → exit/ folder (completed trades)
        - If no exit yet → entry/ folder (open positions)
        - Only entry signals with "is CONFIRMED" or "was CONFIRMED" status are processed
        - Exit signals are always processed (they're completed trades)
    
    For targets: Checks against master CSV before storing.
    
    Args:
        input_file: Path to input CSV file (e.g., outstanding_signal.csv, target_signal.csv)
        signal_type: 'signal' or 'target'
        output_base_dir: Base directory for output (default: chatbot/data)
        overwrite: Whether to overwrite existing files
        dedup_columns: List of columns for deduplication (uses .env if None)
    """
    # Note: dedup_columns parameter is now handled per-row based on signal type (entry/exit/target/breadth)
    # Each signal type uses different deduplication keys
    print("\n" + "="*80)
    print(f"CONVERTING: {input_file}")
    print("="*80 + "\n")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return
    
    # Get the column names
    symbol_column = df.columns[1]  # "Symbol, Signal, Signal Date/Price[$]"
    exit_column = df.columns[2] if len(df.columns) > 2 else None  # "Exit Signal Date/Price[$]"
    
    print(f"✓ Parsing column: '{symbol_column}'")
    if exit_column:
        print(f"✓ Exit column: '{exit_column}'")
    
    # Parse each row
    # Note: For signals, we'll determine entry/exit folder per-row based on exit date
    if signal_type == "target":
        output_base = Path(output_base_dir) / "portfolio_target_achieved"
        output_base.mkdir(parents=True, exist_ok=True)
    # For signals, we create both entry and exit folders
    elif signal_type == "signal":
        entry_base = Path(output_base_dir) / "entry"
        exit_base = Path(output_base_dir) / "exit"
        entry_base.mkdir(parents=True, exist_ok=True)
        exit_base.mkdir(parents=True, exist_ok=True)
    
    # For targets, set up master CSV path
    master_csv_path = None
    if signal_type == "target":
        master_csv_path = Path(output_base_dir) / "portfolio_target_achieved" / "all_targets.csv"
    
    processed = 0
    skipped = 0
    duplicates_rejected = 0
    unconfirmed_skipped = 0
    created_symbols = set()
    created_functions = set()
    signals_with_exit = 0
    signals_no_exit = 0
    
    # Handle different column structures for signal vs target
    if signal_type == "target":
        # For target_signal.csv column order:
        # col 0: Function
        # col 1: "Symbol, Signal, Signal Date/Price[$]"
        # col 2: Interval
        # col 3: Exit Signal Date/Price[$]
        # col 4: Target for which Price has achieved over 90 percent of gain %
        # col 5: Backtested Target Exit Date
        # col 6+: Other columns...
        function_column = df.columns[0]  # "Function"
        symbol_column = df.columns[1]  # "Symbol, Signal, Signal Date/Price[$]"
        exit_column = None  # Exit Signal Date/Price[$] is in column 3, but handled separately in target parsing section
        confirmation_column = None  # No confirmation column for targets
        use_current_date = True  # Use current date for targets
    else:
        # For signal files, function is first column, symbol is in column 1
        function_column = df.columns[0]  # Function name
        symbol_column = df.columns[1]
        exit_column = df.columns[2] if len(df.columns) > 2 else None
        # Find "Interval, Confirmation Status" column (usually column 5)
        confirmation_column = None
        for col in df.columns:
            if "Confirmation Status" in col or "confirmation" in col.lower():
                confirmation_column = col
                break
        use_current_date = False
        
        if confirmation_column:
            print(f"✓ Confirmation column found: '{confirmation_column}'")
        else:
            print(f"⚠ Warning: Confirmation Status column not found. All signals will be processed.")
    
    for idx, row in df.iterrows():
        # For targets, check duplicate first
        if signal_type == "target" and master_csv_path:
            is_duplicate = check_target_duplicate(row, master_csv_path)
            if is_duplicate:
                duplicates_rejected += 1
                continue  # Skip this row
        
        # Get function name
        function_name = row[function_column]
        if pd.isna(function_name) or not function_name:
            function_name = "UNKNOWN"
        
        # Parse symbol based on file type
        if signal_type == "target":
            # For targets, parse the compound column "Symbol, Signal, Signal Date/Price[$]"
            symbol_data = row[symbol_column]
            if pd.notna(symbol_data):
                symbol, signal_date, sig_type, price = parse_symbol_signal_column(symbol_data)
            else:
                symbol, signal_date, sig_type, price = None, None, None, None
            
            if not symbol or not signal_date:
                # Fallback: try to get symbol directly if parsing fails
                if pd.notna(row[symbol_column]):
                    symbol = str(row[symbol_column]).strip()
                    # Try to extract symbol from the beginning if it's a compound field
                    if ',' in symbol:
                        symbol = symbol.split(',')[0].strip()
                else:
                    symbol = ""
                signal_date = datetime.now().strftime("%Y-%m-%d")
                sig_type = ""
        else:
            # For signals, parse the compound column
            symbol_data = row[symbol_column]
            if pd.notna(symbol_data):
            symbol, signal_date, sig_type, price = parse_symbol_signal_column(symbol_data)
            else:
                symbol, signal_date, sig_type, price = None, None, None, None
        
        if not symbol or not signal_date:
            skipped += 1
            continue
        
        # Check for exit date (only for signals, not targets)
        exit_date = None
        exit_price = None
        
        if signal_type != "target" and exit_column and exit_column in row.index:
            exit_data = row[exit_column]
            if pd.notna(exit_data):
            exit_date, exit_price = parse_exit_signal_column(exit_data)
            else:
                exit_date, exit_price = None, None
        
        # For entry signals (signals without exit), check confirmation status
        # Exit signals are always processed (they're completed trades)
        if signal_type == "signal" and not exit_date and confirmation_column and confirmation_column in row.index:
            confirmation_status = row[confirmation_column]
            if pd.notna(confirmation_status) and not is_confirmed_signal(confirmation_status):
                unconfirmed_skipped += 1
                skipped += 1
                continue  # Skip unconfirmed entry signals
            elif pd.isna(confirmation_status):
                # If confirmation status is NaN, skip the signal (can't verify it's confirmed)
                unconfirmed_skipped += 1
                skipped += 1
                continue
        
        # Determine which folder and date to use based on signal type and exit date
        if signal_type == "target":
            # For targets, always use current date and target folder
            date_to_use = datetime.now().strftime("%Y-%m-%d")
            row_signal_type = "portfolio_target_achieved"
            output_base = Path(output_base_dir) / "portfolio_target_achieved"
        elif exit_date:
            # For signals with exit, use exit date and EXIT folder
            date_to_use = exit_date
            signals_with_exit += 1
            row_signal_type = "exit"  # Completed trade
            output_base = exit_base
        else:
            # For signals without exit, use signal date and ENTRY folder
            date_to_use = signal_date
            signals_no_exit += 1
            row_signal_type = "entry"  # Open position
            output_base = entry_base
        
        # Extract columns for deduplication based on signal type
        if signal_type == "signal":
            # Extract Interval from "Interval, Confirmation Status" column
            if confirmation_column and confirmation_column in row.index:
                interval = parse_interval_from_status(row[confirmation_column])
                row['Interval'] = interval if interval else ""
            else:
                row['Interval'] = ""
            
            # Extract Signal (Long/Short) from parsed symbol data
            # We already parsed sig_type from parse_symbol_signal_column
            row['Signal'] = sig_type if sig_type else ""
            
            # Add Symbol column for deduplication
            row['Symbol'] = symbol if symbol else ""
            
            if row_signal_type == "entry":
                # For entry signals: Date, Symbol, Interval, Signal
                row['Date'] = signal_date if signal_date else ""
            elif row_signal_type == "exit":
                # For exit signals: Exit Signal Date, Symbol, Signal, Interval, Signal Date
                row['Exit Signal Date'] = exit_date if exit_date else ""
                row['Signal Date'] = signal_date if signal_date else ""
        elif signal_type == "target":
            # For targets deduplication: Symbol, Signal, Interval, Signal Date, 
            # Target for which Price has achieved over 90 percent of gain %,
            # Backtested Target Exit Date, Exit Signal Date/Price[$]
            # 
            # Column order in target_signal.csv:
            # col 0: Function
            # col 1: "Symbol, Signal, Signal Date/Price[$]"
            # col 2: Interval
            # col 3: Exit Signal Date/Price[$]
            # col 4: Target for which Price has achieved over 90 percent of gain %
            # col 5: Backtested Target Exit Date
            
            # Extract Symbol, Signal, and Signal Date (already parsed above from col 1)
            row['Symbol'] = symbol if symbol else ""
            row['Signal'] = sig_type if sig_type else ""
            row['Signal Date'] = signal_date if signal_date else ""
            
            # Extract Interval (column 2)
            interval_column = df.columns[2] if len(df.columns) > 2 else None
            if interval_column and interval_column in row.index:
                row['Interval'] = str(row[interval_column]).strip() if pd.notna(row[interval_column]) else ""
            else:
                row['Interval'] = ""
            
            # Extract "Exit Signal Date/Price[$]" (column 3)
            exit_signal_column = df.columns[3] if len(df.columns) > 3 else None
            if exit_signal_column and exit_signal_column in row.index:
                row['Exit Signal Date/Price[$]'] = str(row[exit_signal_column]).strip() if pd.notna(row[exit_signal_column]) else ""
            else:
                row['Exit Signal Date/Price[$]'] = ""
            
            # Extract "Target for which Price has achieved over 90 percent of gain %" (column 4)
            target_column = df.columns[4] if len(df.columns) > 4 else None
            if target_column and target_column in row.index:
                row['Target for which Price has achieved over 90 percent of gain %'] = str(row[target_column]).strip() if pd.notna(row[target_column]) else ""
            else:
                row['Target for which Price has achieved over 90 percent of gain %'] = ""
            
            # Extract "Backtested Target Exit Date" (column 5)
            backtested_exit_column = df.columns[5] if len(df.columns) > 5 else None
            if backtested_exit_column and backtested_exit_column in row.index:
                row['Backtested Target Exit Date'] = str(row[backtested_exit_column]).strip() if pd.notna(row[backtested_exit_column]) else ""
            else:
                row['Backtested Target Exit Date'] = ""
        
        # Add SignalType column to the row
        row['SignalType'] = row_signal_type
        
        # Create asset/function directory structure
        # Structure: data/{entry|exit|target}/{asset}/{function}/YYYY-MM-DD.csv
        asset_dir = output_base / symbol
        function_dir = asset_dir / function_name
        function_dir.mkdir(parents=True, exist_ok=True)
        
        created_symbols.add(symbol)
        created_functions.add(function_name)
        
        # Create output file path using the determined date
        output_file = function_dir / f"{date_to_use}.csv"
        
        # Check if file exists
        if output_file.exists() and not overwrite:
            # Append to existing file with deduplication
            try:
                existing_df = pd.read_csv(output_file)
                new_row_df = pd.DataFrame([row])
                
                # Combine and deduplicate using the appropriate signal type
                combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                # Map row_signal_type to deduplication signal_type
                dedup_signal_type = "target" if row_signal_type == "portfolio_target_achieved" else row_signal_type
                combined_df = deduplicate_dataframe(combined_df, dedup_columns=None, signal_type=dedup_signal_type)
                
                # Save deduplicated data
                combined_df.to_csv(output_file, index=False)
                processed += 1
            except Exception as e:
                print(f"  ⚠ Error appending to {output_file}: {e}")
                skipped += 1
        else:
            # Create new file (still deduplicate in case of multiple rows with same params)
            try:
                # Save the entire row as a new CSV
                row_df = pd.DataFrame([row])
                # Map row_signal_type to deduplication signal_type
                dedup_signal_type = "target" if row_signal_type == "portfolio_target_achieved" else row_signal_type
                row_df = deduplicate_dataframe(row_df, dedup_columns=None, signal_type=dedup_signal_type)
                row_df.to_csv(output_file, index=False)
                
                # For targets, add to master CSV
                if signal_type == "target" and master_csv_path:
                    add_to_master_targets(row, master_csv_path)
                
                processed += 1
            except Exception as e:
                print(f"  ⚠ Error creating {output_file}: {e}")
                skipped += 1
    
    print("\n" + "-"*80)
    print("CONVERSION SUMMARY")
    print("-"*80)
    print(f"Signal Type: {signal_type.upper()}")
    print(f"✓ Total rows processed: {processed}")
    print(f"⚠ Rows skipped: {skipped}")
    if signal_type == "target":
        print(f"🚫 Duplicates rejected: {duplicates_rejected}")
        print(f"   → Deduplication keys: Symbol, Signal, Interval, Signal Date, Target for which Price has achieved over 90 percent of gain %, Backtested Target Exit Date, Exit Signal Date/Price[$]")
    if signal_type == "signal" and unconfirmed_skipped > 0:
        print(f"🚫 Unconfirmed entry signals skipped: {unconfirmed_skipped} (only 'is CONFIRMED' or 'was CONFIRMED' are processed)")
    print(f"✓ Unique assets: {len(created_symbols)}")
    print(f"✓ Unique functions: {len(created_functions)}")
    if signal_type != "target":
        print(f"\n📂 Folder Distribution:")
        print(f"   ✓ EXIT signals (completed trades): {signals_with_exit}")
        print(f"      → Stored in: chatbot/data/exit/{{asset}}/{{function}}/{{exit_date}}.csv")
        print(f"      → Deduplication keys: Exit Signal Date, Symbol, Signal, Interval, Signal Date")
        print(f"   ✓ ENTRY signals (confirmed open positions): {signals_no_exit}")
        print(f"      → Stored in: chatbot/data/entry/{{asset}}/{{function}}/{{signal_date}}.csv")
        print(f"      → Deduplication keys: Date, Symbol, Interval, Signal")
        if unconfirmed_skipped > 0:
            print(f"   ⚠ Unconfirmed entry signals filtered out: {unconfirmed_skipped}")
    print(f"\n✓ Functions: {', '.join(sorted(list(created_functions)))}")
    print(f"\n✓ Assets (sample): {', '.join(sorted(list(created_symbols)[:10]))}")
    if len(created_symbols) > 10:
        print(f"  ... and {len(created_symbols) - 10} more")
    
    if signal_type == "target":
        print(f"\n✓ Output structure: chatbot/data/portfolio_target_achieved/{{asset}}/{{function}}/YYYY-MM-DD.csv")
    else:
        print(f"\n✓ Output structures:")
        print(f"   - chatbot/data/entry/{{asset}}/{{function}}/YYYY-MM-DD.csv")
        print(f"   - chatbot/data/exit/{{asset}}/{{function}}/YYYY-MM-DD.csv")
    print("="*80 + "\n")
    
    return processed, skipped, created_symbols


def get_latest_price_from_stock_data(symbol, stock_data_dir="trade_store/stock_data"):
    """
    Get the latest (most recent) price from stock_data CSV file.
    
    Args:
        symbol: Stock symbol (e.g., "TSLA", "WMT", "^NDX")
        stock_data_dir: Directory containing stock_data CSV files
    
    Returns:
        Tuple of (latest_date, latest_price) or (None, None) if not found
    """
    try:
        stock_data_path = Path(stock_data_dir) / f"{symbol}.csv"
        
        if not stock_data_path.exists():
            return None, None
        
        # Read the stock data CSV
        stock_df = pd.read_csv(stock_data_path)
        
        if stock_df.empty or 'Close' not in stock_df.columns:
            return None, None
        
        # Get the most recent row (last row, assuming data is sorted by date)
        latest_row = stock_df.iloc[-1]
        latest_date = latest_row.get('Date', None)
        latest_price = latest_row.get('Close', None)
        
        if pd.isna(latest_date) or pd.isna(latest_price):
            return None, None
        
        return str(latest_date), float(latest_price)
        
    except Exception as e:
        print(f"  ⚠ Error reading stock data for {symbol}: {e}")
        return None, None


def parse_current_price_column(value):
    """
    Parse the "Current Trading Date/Price[$], Current Price vs Signal" column.
    
    Example: "2025-11-18 (Price: 401.25), 0.0% below"
    Returns: (date, price, percentage_change) or (None, None, None)
    """
    try:
        if not value or pd.isna(value):
            return None, None, None
        
        value_str = str(value).strip()
        
        # Extract date
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', value_str)
        date = date_match.group(1) if date_match else None
        
        # Extract price
        price_match = re.search(r'Price:\s*([\d.]+)', value_str)
        price = float(price_match.group(1)) if price_match else None
        
        # Extract percentage change (optional)
        pct_match = re.search(r'([\d.]+)%\s*(above|below)', value_str)
        percentage = None
        if pct_match:
            pct_value = float(pct_match.group(1))
            direction = pct_match.group(2)
            percentage = pct_value if direction == 'above' else -pct_value
        
        return date, price, percentage
        
    except Exception as e:
        print(f"  ⚠ Error parsing current price: {value} - {e}")
        return None, None, None


def calculate_price_change_percentage(current_price, signal_price):
    """
    Calculate percentage change between current price and signal price.
    
    Args:
        current_price: Current/live price
        signal_price: Original signal price
    
    Returns:
        Percentage change string (e.g., "5.2% above" or "3.1% below")
    """
    if not current_price or not signal_price or signal_price == 0:
        return "0.0% below"
    
    try:
        change_pct = ((current_price - signal_price) / signal_price) * 100
        
        if change_pct >= 0:
            return f"{change_pct:.2f}% above"
        else:
            return f"{abs(change_pct):.2f}% below"
            
    except Exception as e:
        return "0.0% below"


def update_current_prices_in_data_files(data_base_dir="chatbot/data", stock_data_dir="trade_store/stock_data"):
    """
    Update current prices in all chatbot data files using live prices from stock_data.
    
    This function:
    1. Scans all CSV files in chatbot/data (entry, exit, portfolio_target_achieved)
    2. For each file, extracts symbols and updates "Current Trading Date/Price[$], Current Price vs Signal" column
    3. Uses the latest price from stock_data CSV files
    
    Args:
        data_base_dir: Base directory for chatbot data (default: chatbot/data)
        stock_data_dir: Directory containing stock_data CSV files (default: trade_store/stock_data)
    """
    print("\n" + "="*80)
    print("UPDATING CURRENT PRICES FROM LIVE STOCK DATA")
    print("="*80 + "\n")
    
    data_base = Path(data_base_dir)
    stock_data_base = Path(stock_data_dir)
    
    if not data_base.exists():
        print(f"✗ Data directory not found: {data_base_dir}")
        return
    
    if not stock_data_base.exists():
        print(f"✗ Stock data directory not found: {stock_data_dir}")
        return
    
    # Find all CSV files in entry, exit, and portfolio_target_achieved folders
    csv_files = []
    for folder in ['entry', 'exit', 'portfolio_target_achieved']:
        folder_path = data_base / folder
        if folder_path.exists():
            csv_files.extend(folder_path.rglob("*.csv"))
    
    if not csv_files:
        print("⚠ No CSV files found to update")
        return
    
    print(f"✓ Found {len(csv_files)} CSV files to process")
    
    updated_count = 0
    skipped_count = 0
    price_not_found_count = 0
    
    current_price_column = "Current Trading Date/Price[$], Current Price vs Signal"
    
    for csv_file in csv_files:
        try:
            # Skip master files
            if csv_file.name == "all_targets.csv":
                continue
            
            df = pd.read_csv(csv_file)
            
            if df.empty:
                continue
            
            # Check if the current price column exists
            if current_price_column not in df.columns:
                skipped_count += 1
                continue
            
            file_updated = False
            
            # Process each row
            for idx, row in df.iterrows():
                # Extract symbol from the row
                symbol = None
                
                # Try to get symbol from different possible columns
                if 'Symbol' in row.index:
                    symbol_val = row['Symbol']
                    if pd.notna(symbol_val):
                        symbol = str(symbol_val).strip()
                elif 'Symbol, Signal, Signal Date/Price[$]' in row.index:
                    symbol_data = row['Symbol, Signal, Signal Date/Price[$]']
                    if pd.notna(symbol_data):
                        symbol, _, _, _ = parse_symbol_signal_column(symbol_data)
                
                if not symbol or pd.isna(symbol) or symbol == '':
                    continue
                
                # Get latest price from stock_data
                latest_date, latest_price = get_latest_price_from_stock_data(symbol, stock_data_dir)
                
                if latest_date is None or latest_price is None:
                    price_not_found_count += 1
                    continue
                
                # Extract signal price to calculate percentage change
                signal_price = None
                if 'Symbol, Signal, Signal Date/Price[$]' in row.index:
                    symbol_data = row['Symbol, Signal, Signal Date/Price[$]']
                    if pd.notna(symbol_data):
                        _, _, _, signal_price = parse_symbol_signal_column(symbol_data)
                
                # Calculate percentage change
                if signal_price:
                    price_change_str = calculate_price_change_percentage(latest_price, signal_price)
                else:
                    price_change_str = "0.0% below"
                
                # Update the current price column
                new_current_price_value = f"{latest_date} (Price: {latest_price:.4f}), {price_change_str}"
                df.at[idx, current_price_column] = new_current_price_value
                file_updated = True
            
            # Save updated file if changes were made
            if file_updated:
                df.to_csv(csv_file, index=False)
                updated_count += 1
                
        except Exception as e:
            print(f"  ⚠ Error updating {csv_file}: {e}")
            skipped_count += 1
    
    print("\n" + "-"*80)
    print("PRICE UPDATE SUMMARY")
    print("-"*80)
    print(f"✓ Files updated: {updated_count}")
    print(f"⚠ Files skipped: {skipped_count}")
    print(f"⚠ Symbols with no price data: {price_not_found_count}")
    print("="*80 + "\n")


def convert_breadth_report(
    input_file,
    output_base_dir="chatbot/data"
):
    """
    Convert breadth report to data folder structure.
    Breadth is market-wide, so structure is: chatbot/data/breadth/YYYY-MM-DD.csv
    
    Args:
        input_file: Path to breadth.csv file
        output_base_dir: Base directory for output (default: chatbot/data)
    """
    print("\n" + "="*80)
    print(f"CONVERTING BREADTH REPORT: {input_file}")
    print("="*80 + "\n")
    
    # Read the breadth CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return 0, 0
    
    # Create breadth directory
    breadth_dir = Path(output_base_dir) / "breadth"
    breadth_dir.mkdir(parents=True, exist_ok=True)
    
    # Use current date for filename and add Date column
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = breadth_dir / f"{current_date}.csv"
    
    # Add Date column to the dataframe
    df['Date'] = current_date
    
    # Save breadth report with current date
    try:
        df.to_csv(output_file, index=False)
        print(f"✓ Saved breadth report to: {output_file}")
        print(f"✓ Functions in report: {len(df)}")
        print(f"✓ Columns: {', '.join(df.columns.tolist())}")
        processed = 1
        skipped = 0
    except Exception as e:
        print(f"✗ Error saving breadth report: {e}")
        processed = 0
        skipped = 1
    
    print("\n" + "-"*80)
    print("BREADTH CONVERSION SUMMARY")
    print("-"*80)
    print(f"✓ Report saved: {output_file.name}")
    print(f"✓ Total functions: {len(df)}")
    print("="*80 + "\n")
    
    return processed, skipped


def main():
    """Main function with examples."""
    
    print("\n" + "="*80)
    print("TRADING SIGNAL, TARGET & BREADTH DATA CONVERTER")
    print("="*80 + "\n")
    
    print("This script converts trading CSV files to the chatbot data structure.")
    print("Structure:")
    print("  - chatbot/data/entry/{asset}/{function}/YYYY-MM-DD.csv (open positions)")
    print("  - chatbot/data/exit/{asset}/{function}/YYYY-MM-DD.csv (completed trades)")
    print("  - chatbot/data/portfolio_target_achieved/{asset}/{function}/YYYY-MM-DD.csv (portfolio target achieved)")
    print("  - chatbot/data/breadth/YYYY-MM-DD.csv (market breadth)\n")
    
    # Convert outstanding_signal.csv (signals)
    # Handle both naming conventions: outstanding_signal.csv and YYYY-MM-DD_outstanding_signal.csv
    print("-" * 80)
    print("Converting SIGNAL data (outstanding_signal.csv)")
    print("-" * 80)
    
    # Try to find the most recent outstanding_signal file
    signal_file = None
    
    # First try exact match
    signal_file_exact = Path("trade_store/US/outstanding_signal.csv")
    if signal_file_exact.exists():
        signal_file = signal_file_exact
    else:
        # Try pattern matching for date_name.csv format
        signal_pattern_files = list(Path("trade_store/US").glob("*_outstanding_signal.csv"))
        if signal_pattern_files:
            # Sort by modification time and get the most recent
            signal_pattern_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            signal_file = signal_pattern_files[0]
            print(f"ℹ Found dated file: {signal_file.name}")
    
    if signal_file and signal_file.exists():
        convert_signal_file_to_data_structure(
            input_file=signal_file,
            signal_type="signal",
            output_base_dir="chatbot/data",
            overwrite=False
        )
    else:
        print(f"⚠ File not found: outstanding_signal.csv (tried exact match and date_name.csv pattern)")
    
    # Convert target_signal.csv (targets)
    # Handle both naming conventions: target_signal.csv and YYYY-MM-DD_target_signal.csv
    print("\n" + "-" * 80)
    print("Converting TARGET data (target_signal.csv)")
    print("-" * 80)
    
    # Try to find the most recent target_signal file
    target_file = None
    
    # First try exact match
    target_file_exact = Path("trade_store/US/target_signal.csv")
    if target_file_exact.exists():
        target_file = target_file_exact
    else:
        # Try pattern matching for date_name.csv format
        target_pattern_files = list(Path("trade_store/US").glob("*_target_signal.csv"))
        if target_pattern_files:
            # Sort by modification time and get the most recent
            target_pattern_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            target_file = target_pattern_files[0]
            print(f"ℹ Found dated file: {target_file.name}")
    
    if target_file and target_file.exists():
        convert_signal_file_to_data_structure(
            input_file=target_file,
            signal_type="target",
            output_base_dir="chatbot/data",
            overwrite=False
        )
    else:
        print(f"⚠ File not found: target_signal.csv (tried exact match and date_name.csv pattern)")
    
    # Convert breadth.csv (market-wide breadth report)
    # Handle both naming conventions: breadth.csv and YYYY-MM-DD_breadth.csv
    print("\n" + "-" * 80)
    print("Converting BREADTH data (breadth.csv)")
    print("-" * 80)
    
    # Try to find the most recent breadth file
    breadth_file = None
    
    # First try exact match
    breadth_file_exact = Path("trade_store/US/breadth.csv")
    if breadth_file_exact.exists():
        breadth_file = breadth_file_exact
    else:
        # Try pattern matching for date_name.csv format (excluding breadth_us.csv)
        breadth_pattern_files = [f for f in Path("trade_store/US").glob("*_breadth.csv") 
                                if "breadth_us" not in f.name]
        if breadth_pattern_files:
            # Sort by modification time and get the most recent
            breadth_pattern_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            breadth_file = breadth_pattern_files[0]
            print(f"ℹ Found dated file: {breadth_file.name}")
    
    if breadth_file and breadth_file.exists():
        convert_breadth_report(
            input_file=breadth_file,
            output_base_dir="chatbot/data"
        )
    else:
        print(f"⚠ File not found: breadth.csv (tried exact match and date_name.csv pattern)")
    
    # Update current prices in all chatbot data files using live prices from stock_data
    print("\n" + "-" * 80)
    print("Updating current prices from live stock data")
    print("-" * 80)
    update_current_prices_in_data_files(
        data_base_dir="chatbot/data",
        stock_data_dir="trade_store/stock_data"
    )
    
    print("\n" + "="*80)
    print("✓ Conversion Complete!")
    print("="*80)
    print("\nData structure created (4 signal types):")
    print("  1. ENTRY:   chatbot/data/entry/{asset}/{function}/YYYY-MM-DD.csv")
    print("  2. EXIT:    chatbot/data/exit/{asset}/{function}/YYYY-MM-DD.csv")
    print("  3. TARGET:  chatbot/data/portfolio_target_achieved/{asset}/{function}/YYYY-MM-DD.csv")
    print("  4. BREADTH: chatbot/data/breadth/YYYY-MM-DD.csv")
    print("\nMaster files:")
    print("  - chatbot/data/portfolio_target_achieved/all_targets.csv (portfolio target deduplication)")
    print("\n✓ Current prices updated from live stock data")
    print()


if __name__ == "__main__":
    main()

