"""
Data processor for loading and filtering CSV data based on user parameters.
New structure: chatbot/data/{ticker_name}/YYYY-MM-DD.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from .config import (
    CHATBOT_DATA_DIR,
    CHATBOT_ENTRY_DIR,
    CHATBOT_EXIT_DIR,
    CHATBOT_TARGET_DIR,
    CHATBOT_BREADTH_DIR,
    STOCK_DATA_DIR,
    DATE_FORMAT,
    CSV_ENCODING,
    MAX_ROWS_TO_INCLUDE,
    DEDUP_COLUMNS,
    MAX_INPUT_TOKENS_PER_CALL,
    ESTIMATED_CHARS_PER_TOKEN
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles loading and processing of CSV data for chatbot queries."""
    
    def __init__(self, use_new_structure: bool = True):
        """
        Initialize DataProcessor.
        
        Args:
            use_new_structure: If True, uses chatbot/data/{entry|exit|target|breadth} structure.
                             If False, uses legacy trade_store/stock_data structure.
        """
        self.use_new_structure = use_new_structure
        self.chatbot_data_dir = Path(CHATBOT_DATA_DIR)
        self.entry_data_dir = Path(CHATBOT_ENTRY_DIR)
        self.exit_data_dir = Path(CHATBOT_EXIT_DIR)
        self.target_data_dir = Path(CHATBOT_TARGET_DIR)
        self.breadth_data_dir = Path(CHATBOT_BREADTH_DIR)
        self.stock_data_dir = Path(STOCK_DATA_DIR)
        
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available ticker/asset symbols from entry, exit, and target folders.
        
        Returns:
            List of ticker symbols
        """
        try:
            if self.use_new_structure:
                # Get folder names from entry/, exit/, and target/
                tickers = set()
                
                if self.entry_data_dir.exists():
                    entry_tickers = [d.name for d in self.entry_data_dir.iterdir() if d.is_dir()]
                    tickers.update(entry_tickers)
                
                if self.exit_data_dir.exists():
                    exit_tickers = [d.name for d in self.exit_data_dir.iterdir() if d.is_dir()]
                    tickers.update(exit_tickers)
                
                if self.target_data_dir.exists():
                    target_tickers = [d.name for d in self.target_data_dir.iterdir() if d.is_dir()]
                    tickers.update(target_tickers)
                
                return sorted(list(tickers))
            else:
                # Legacy: Get CSV files from trade_store/stock_data/
                csv_files = list(self.stock_data_dir.glob("*.csv"))
                tickers = [f.stem for f in csv_files if f.stem != "today_date"]
                return sorted(tickers)
        except Exception as e:
            logger.error(f"Error getting available tickers: {e}")
            return []
    
    def get_available_functions(self, ticker: Optional[str] = None) -> List[str]:
        """
        Get list of available function names from entry, exit, and target folders.
        
        Args:
            ticker: Optional ticker to get functions for. If None, gets all unique functions.
            
        Returns:
            List of function names
        """
        try:
            if not self.use_new_structure:
                return []
            
            functions = set()
            
            if ticker:
                # Get functions for specific ticker from entry, exit, and target
                for base_dir in [self.entry_data_dir, self.exit_data_dir, self.target_data_dir]:
                    if base_dir.exists():
                        ticker_dir = base_dir / ticker
                        if ticker_dir.exists():
                            function_dirs = [d.name for d in ticker_dir.iterdir() if d.is_dir()]
                            functions.update(function_dirs)
                return sorted(list(functions))
            else:
                # Get all unique functions across all tickers from entry, exit, and target
                for base_dir in [self.entry_data_dir, self.exit_data_dir, self.target_data_dir]:
                    if base_dir.exists():
                        for ticker_dir in base_dir.iterdir():
                            if ticker_dir.is_dir():
                                for function_dir in ticker_dir.iterdir():
                                    if function_dir.is_dir():
                                        functions.add(function_dir.name)
                
                return sorted(list(functions))
        except Exception as e:
            logger.error(f"Error getting available functions: {e}")
            return []
    
    def get_available_dates_for_ticker(
        self, 
        ticker: str, 
        function: Optional[str] = None
    ) -> List[str]:
        """
        Get list of available dates for a specific ticker and function from entry, exit, and target.
        
        Args:
            ticker: Ticker/asset symbol
            function: Function name (optional). If None, gets dates across all functions.
            
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        try:
            dates = set()
            
            # Check entry, exit, and target directories
            for base_dir in [self.entry_data_dir, self.exit_data_dir, self.target_data_dir]:
                if not base_dir.exists():
                    continue
                    
                ticker_dir = base_dir / ticker
                if not ticker_dir.exists():
                    continue
                
                if function:
                    # Get dates for specific function
                    function_dir = ticker_dir / function
                    if not function_dir.exists():
                        continue
                    csv_files = list(function_dir.glob("*.csv"))
                else:
                    # Get dates across all functions
                    csv_files = list(ticker_dir.rglob("*.csv"))
                
                # Extract dates from filenames (YYYY-MM-DD.csv)
                for f in csv_files:
                    date_str = f.stem
                    # Validate date format
                    try:
                        datetime.strptime(date_str, DATE_FORMAT)
                        dates.add(date_str)
                    except ValueError:
                        logger.warning(f"Invalid date format in filename: {f.name}")
                        continue
            
            return sorted(list(dates))
        except Exception as e:
            logger.error(f"Error getting available dates for {ticker}: {e}")
            return []
    
    def get_date_range(self, from_date: str, to_date: str) -> List[str]:
        """
        Generate list of dates between from_date and to_date.
        
        Args:
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            List of date strings
        """
        try:
            start = datetime.strptime(from_date, DATE_FORMAT)
            end = datetime.strptime(to_date, DATE_FORMAT)
            
            dates = []
            current = start
            while current <= end:
                dates.append(current.strftime(DATE_FORMAT))
                current += timedelta(days=1)
            
            return dates
        except Exception as e:
            logger.error(f"Error generating date range: {e}")
            return []
    
    def load_stock_data_new_structure(
        self,
        tickers: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        dedup_columns: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        signal_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data from new structure: chatbot/data/{signal|target}/{asset}/{function}/YYYY-MM-DD.csv
        Loads from signal and/or target folders based on signal_types parameter.
        
        Args:
            tickers: List of ticker/asset symbols
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            dedup_columns: Columns to use for deduplication (placeholder)
            functions: List of function names to filter (None = all functions)
            signal_types: List of signal types - controls which folders to load from:
                         - ['entry_exit'] → load from signal/ folder only
                         - ['target_achieved'] → load from target/ folder only
                         - None or [] → load from both (fallback)
            
        Returns:
            Dictionary mapping ticker to combined DataFrame
        """
        if dedup_columns is None:
            dedup_columns = DEDUP_COLUMNS
        
        result = {}
        
        for ticker in tickers:
            try:
                # Determine which functions to load
                if functions:
                    # User specified functions - use only those
                    functions_to_load = functions
                    logger.info(f"Loading specific functions for {ticker}: {functions_to_load}")
                else:
                    # No functions specified - load ALL available functions
                    functions_to_load = self.get_available_functions(ticker)
                    logger.info(f"No functions specified - loading ALL {len(functions_to_load)} functions for {ticker}")
                
                if not functions_to_load:
                    logger.warning(f"No functions found for asset: {ticker}")
                    continue
                
                # Determine which directories to load from based on signal_types
                base_dirs_to_load = []
                if signal_types:
                    if 'entry' in signal_types:
                        base_dirs_to_load.append(('entry', self.entry_data_dir))
                    if 'exit' in signal_types:
                        base_dirs_to_load.append(('exit', self.exit_data_dir))
                    if 'target' in signal_types:
                        base_dirs_to_load.append(('target', self.target_data_dir))
                    # Note: breadth is handled separately, not asset-specific
                    logger.info(f"Loading based on signal_types {signal_types}: {[name for name, _ in base_dirs_to_load]}")
                else:
                    # No signal_types specified - load from entry, exit, and target (fallback)
                    base_dirs_to_load = [
                        ('entry', self.entry_data_dir),
                        ('exit', self.exit_data_dir),
                        ('target', self.target_data_dir)
                    ]
                    logger.info(f"No signal_types specified - loading from entry, exit, and target folders")
                
                # Load data for each function from selected folders
                all_dfs = []
                for function_name in functions_to_load:
                    # Load from selected directories
                    for dir_name, base_dir in base_dirs_to_load:
                        if not base_dir.exists():
                            continue
                            
                        ticker_dir = base_dir / ticker
                        if not ticker_dir.exists():
                            continue
                            
                        function_dir = ticker_dir / function_name
                        if not function_dir.exists():
                            continue
                        
                        # Get available dates for this ticker/function
                        available_dates = self.get_available_dates_for_ticker(ticker, function_name)
                        
                        if not available_dates:
                            continue
                        
                        # Filter dates by from_date and to_date
                        dates_to_load = available_dates
                        if from_date:
                            dates_to_load = [d for d in dates_to_load if d >= from_date]
                        if to_date:
                            dates_to_load = [d for d in dates_to_load if d <= to_date]
                        
                        if not dates_to_load:
                            continue
                        
                        # Determine data type based on directory name
                        data_type = dir_name  # 'entry', 'exit', or 'target'
                        
                        # Load all CSV files for the date range and function
                        for date_str in dates_to_load:
                            file_path = function_dir / f"{date_str}.csv"
                            
                            if not file_path.exists():
                                continue
                            
                            try:
                                df = pd.read_csv(file_path, encoding=CSV_ENCODING)
                                
                                # Add ticker column if not present
                                if 'Symbol' not in df.columns and 'Ticker' not in df.columns:
                                    df['Symbol'] = ticker
                                
                                # Add date column if not present
                                if 'Date' not in df.columns:
                                    df['Date'] = date_str
                                
                                # Add function column if not present
                                if 'Function' not in df.columns:
                                    df['Function'] = function_name
                                
                                # Add data type column
                                df['DataType'] = data_type
                                
                                all_dfs.append(df)
                                
                            except Exception as e:
                                logger.error(f"Error loading file {file_path}: {e}")
                                continue
                
                if not all_dfs:
                    logger.warning(f"No valid data loaded for ticker: {ticker}")
                    continue
                
                # Combine all dataframes
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # Convert Date column to datetime if present
                if 'Date' in combined_df.columns:
                    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
                
                # Remove duplicates based on specified columns
                # Placeholder: Use the columns specified in config or parameter
                available_dedup_cols = [col for col in dedup_columns if col in combined_df.columns]
                
                if available_dedup_cols:
                    original_count = len(combined_df)
                    combined_df = combined_df.drop_duplicates(subset=available_dedup_cols, keep='first')
                    removed_count = original_count - len(combined_df)
                    
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} duplicate rows for {ticker} based on columns: {available_dedup_cols}")
                else:
                    logger.warning(f"No deduplication columns found for {ticker}. Available columns: {combined_df.columns.tolist()}")
                
                # Sort by date if Date column exists (LATEST FIRST)
                if 'Date' in combined_df.columns:
                    combined_df = combined_df.sort_values('Date', ascending=False)
                elif 'LoadedDate' in combined_df.columns:
                    combined_df = combined_df.sort_values('LoadedDate', ascending=False)
                
                # Limit rows if too many - KEEP LATEST DATA
                if len(combined_df) > MAX_ROWS_TO_INCLUDE:
                    logger.info(f"Limiting {ticker} data from {len(combined_df)} to {MAX_ROWS_TO_INCLUDE} rows (keeping latest)")
                    # Keep the most recent rows
                    combined_df = combined_df.head(MAX_ROWS_TO_INCLUDE)
                
                result[ticker] = combined_df
                
                # Count by SignalType (entry_exit vs target_achieved)
                if 'SignalType' in combined_df.columns:
                    entry_exit_count = len(combined_df[combined_df['SignalType'] == 'entry_exit'])
                    target_count = len(combined_df[combined_df['SignalType'] == 'target_achieved'])
                    logger.info(f"Loaded {len(combined_df)} rows for {ticker} (entry_exit: {entry_exit_count}, target_achieved: {target_count})")
                else:
                    logger.info(f"Loaded {len(combined_df)} rows for {ticker}")
                
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return result
    
    def load_stock_data_legacy(
        self,
        tickers: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data from legacy structure: trade_store/stock_data/{ticker}.csv
        
        Args:
            tickers: List of ticker symbols
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping ticker to filtered DataFrame
        """
        result = {}
        
        for ticker in tickers:
            try:
                file_path = self.stock_data_dir / f"{ticker}.csv"
                
                if not file_path.exists():
                    logger.warning(f"Data file not found for ticker: {ticker}")
                    continue
                
                df = pd.read_csv(file_path, encoding=CSV_ENCODING)
                
                # Ensure Date column exists
                if 'Date' not in df.columns:
                    logger.warning(f"No 'Date' column in {ticker} data")
                    continue
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter by date range if provided
                if from_date:
                    from_dt = pd.to_datetime(from_date)
                    df = df[df['Date'] >= from_dt]
                
                if to_date:
                    to_dt = pd.to_datetime(to_date)
                    df = df[df['Date'] <= to_dt]
                
                # Sort by date
                df = df.sort_values('Date')
                
                # Limit rows if too many
                if len(df) > MAX_ROWS_TO_INCLUDE:
                    logger.info(f"Limiting {ticker} data from {len(df)} to {MAX_ROWS_TO_INCLUDE} rows")
                    step = len(df) // MAX_ROWS_TO_INCLUDE
                    df = df.iloc[::step][:MAX_ROWS_TO_INCLUDE]
                
                result[ticker] = df
                logger.info(f"Loaded {len(df)} rows for {ticker}")
                
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
                continue
        
        return result
    
    def load_stock_data(
        self,
        tickers: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        dedup_columns: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        signal_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for specified tickers, functions, and date range.
        Loads from signal and/or target folders based on signal_types.
        Uses new structure if available, falls back to legacy.
        
        Args:
            tickers: List of ticker/asset symbols
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            dedup_columns: Columns to use for deduplication
            functions: List of function names to filter (None = all functions)
            signal_types: List of signal types - controls which folders to load from:
                         - ['entry_exit'] → load from signal/ folder only
                         - ['target_achieved'] → load from target/ folder only
                         - ['entry_exit', 'target_achieved'] → load from both
            
        Returns:
            Dictionary mapping ticker to DataFrame (includes DataType column: 'signal' or 'target')
        """
        if self.use_new_structure:
            return self.load_stock_data_new_structure(
                tickers, from_date, to_date, dedup_columns, functions, signal_types
            )
        else:
            return self.load_stock_data_legacy(tickers, from_date, to_date)
    
    def load_breadth_data(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load breadth report data for the specified date range.
        Breadth data is market-wide (not asset-specific).
        
        Args:
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with breadth data or None if no data found
        """
        if not self.breadth_data_dir.exists():
            logger.warning(f"Breadth directory not found: {self.breadth_data_dir}")
            return None
        
        # Get all CSV files in breadth directory
        csv_files = list(self.breadth_data_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning("No breadth reports found")
            return None
        
        # Filter by date range
        files_to_load = []
        for file_path in csv_files:
            file_date = file_path.stem  # filename without extension (YYYY-MM-DD)
            
            # Check if date is within range
            include_file = True
            if from_date and file_date < from_date:
                include_file = False
            if to_date and file_date > to_date:
                include_file = False
            
            if include_file:
                files_to_load.append((file_date, file_path))
        
        if not files_to_load:
            logger.info(f"No breadth reports found in date range {from_date} to {to_date}")
            return None
        
        # Sort by date (most recent first)
        files_to_load.sort(reverse=True)
        
        # Load the most recent breadth report
        most_recent_date, most_recent_file = files_to_load[0]
        
        try:
            df = pd.read_csv(most_recent_file, encoding=CSV_ENCODING)
            df['Date'] = most_recent_date
            df['DataType'] = 'breadth'
            
            logger.info(f"Loaded breadth report from {most_recent_date}: {len(df)} functions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading breadth report {most_recent_file}: {e}")
            return None
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // ESTIMATED_CHARS_PER_TOKEN
    
    def format_data_for_prompt(
        self,
        stock_data: Dict[str, pd.DataFrame],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Format stock data as JSON for inclusion in GPT prompt.
        Automatically limits data to fit within token constraints.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            max_tokens: Maximum tokens allowed (default from config)
            
        Returns:
            Formatted JSON string for prompt (limited to max_tokens)
        """
        if not stock_data:
            return ""
        
        if max_tokens is None:
            max_tokens = MAX_INPUT_TOKENS_PER_CALL
        
        import json
        
        formatted_parts = ["=== TRADING DATA (JSON Format) ===\n"]
        current_tokens = 0
        tickers_included = []
        
        # NO TOKEN LIMITS - Include ALL tickers
        # Smart batch processing will handle splitting across multiple API calls
        for ticker, df in stock_data.items():
            if df.empty:
                continue
            
            # Convert DataFrame to list of dictionaries (each row as key-value pairs)
            records = df.to_dict('records')
            
            # Create JSON structure for this ticker
            ticker_data = {
                "asset": ticker,
                "record_count": len(records),
                "data": records
            }
            
            # Convert to JSON string (pretty printed for readability)
            ticker_json = json.dumps(ticker_data, indent=2, default=str)
            
            ticker_token_estimate = self.estimate_token_count(ticker_json)
            
            # Add ALL ticker data - no skipping
            formatted_parts.append(f"\n{ticker_json}")
            current_tokens += ticker_token_estimate
            tickers_included.append(ticker)
        
        result = "\n".join(formatted_parts)
        logger.info(f"Formatted ALL data as JSON: ~{current_tokens} tokens, {len(tickers_included)} assets included (NO LIMITS)")
        
        return result
    
    def validate_date_format(self, date_str: str) -> bool:
        """
        Validate date string format.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            datetime.strptime(date_str, DATE_FORMAT)
            return True
        except ValueError:
            return False
    
    def get_data_summary(
        self,
        tickers: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        dedup_columns: Optional[List[str]] = None,
        functions: Optional[List[str]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
        """
        Complete data loading and formatting pipeline.
        Loads from BOTH signal and target folders.
        
        Args:
            tickers: List of ticker/asset symbols
            from_date: Start date
            to_date: End date
            dedup_columns: Columns for deduplication
            functions: List of function names to filter
            
        Returns:
            Tuple of (stock_data_dict, formatted_text)
        """
        # Load stock data (from both signal and target)
        stock_data = self.load_stock_data(
            tickers, from_date, to_date, dedup_columns, functions
        )
        
        # Format for prompt
        formatted_text = self.format_data_for_prompt(stock_data)
        
        return stock_data, formatted_text
