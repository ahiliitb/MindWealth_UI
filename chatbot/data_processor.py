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
    CHATBOT_SIGNAL_DIR,
    CHATBOT_TARGET_DIR,
    STOCK_DATA_DIR,
    DATE_FORMAT,
    CSV_ENCODING,
    MAX_ROWS_TO_INCLUDE,
    DEDUP_COLUMNS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles loading and processing of CSV data for chatbot queries."""
    
    def __init__(self, use_new_structure: bool = True):
        """
        Initialize DataProcessor.
        
        Args:
            use_new_structure: If True, uses chatbot/data/{signal|target}/{asset}/{function}/YYYY-MM-DD.csv structure.
                             If False, uses legacy trade_store/stock_data structure.
        """
        self.use_new_structure = use_new_structure
        self.chatbot_data_dir = Path(CHATBOT_DATA_DIR)
        self.signal_data_dir = Path(CHATBOT_SIGNAL_DIR)
        self.target_data_dir = Path(CHATBOT_TARGET_DIR)
        self.stock_data_dir = Path(STOCK_DATA_DIR)
        
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available ticker/asset symbols from both signal and target folders.
        
        Returns:
            List of ticker symbols
        """
        try:
            if self.use_new_structure:
                # Get folder names from both signal/ and target/
                tickers = set()
                
                if self.signal_data_dir.exists():
                    signal_tickers = [d.name for d in self.signal_data_dir.iterdir() if d.is_dir()]
                    tickers.update(signal_tickers)
                
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
        Get list of available function names from both signal and target folders.
        
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
                # Get functions for specific ticker from both signal and target
                for base_dir in [self.signal_data_dir, self.target_data_dir]:
                    if base_dir.exists():
                        ticker_dir = base_dir / ticker
                        if ticker_dir.exists():
                            function_dirs = [d.name for d in ticker_dir.iterdir() if d.is_dir()]
                            functions.update(function_dirs)
                return sorted(list(functions))
            else:
                # Get all unique functions across all tickers from both signal and target
                for base_dir in [self.signal_data_dir, self.target_data_dir]:
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
        Get list of available dates for a specific ticker and function from both signal and target.
        
        Args:
            ticker: Ticker/asset symbol
            function: Function name (optional). If None, gets dates across all functions.
            
        Returns:
            List of date strings in YYYY-MM-DD format
        """
        try:
            dates = set()
            
            # Check both signal and target directories
            for base_dir in [self.signal_data_dir, self.target_data_dir]:
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
        functions: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data from new structure: chatbot/data/{signal|target}/{asset}/{function}/YYYY-MM-DD.csv
        Loads from BOTH signal and target folders automatically.
        
        Args:
            tickers: List of ticker/asset symbols
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            dedup_columns: Columns to use for deduplication (placeholder)
            functions: List of function names to filter (None = all functions)
            
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
                
                # Load data for each function from BOTH signal and target folders
                all_dfs = []
                for function_name in functions_to_load:
                    # Load from both signal and target directories
                    for base_dir in [self.signal_data_dir, self.target_data_dir]:
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
                        
                        # Determine data type based on directory
                        data_type = "signal" if base_dir == self.signal_data_dir else "target"
                        
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
                
                # Sort by date if Date column exists
                if 'Date' in combined_df.columns:
                    combined_df = combined_df.sort_values('Date')
                
                # Limit rows if too many
                if len(combined_df) > MAX_ROWS_TO_INCLUDE:
                    logger.info(f"Limiting {ticker} data from {len(combined_df)} to {MAX_ROWS_TO_INCLUDE} rows")
                    # Sample evenly across the data
                    step = len(combined_df) // MAX_ROWS_TO_INCLUDE
                    combined_df = combined_df.iloc[::step][:MAX_ROWS_TO_INCLUDE]
                
                result[ticker] = combined_df
                logger.info(f"Loaded {len(combined_df)} rows for {ticker} (from {total_files_loaded} files)")
                
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
        functions: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for specified tickers, functions, and date range.
        Loads from BOTH signal and target folders automatically.
        Uses new structure if available, falls back to legacy.
        
        Args:
            tickers: List of ticker/asset symbols
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            dedup_columns: Columns to use for deduplication
            functions: List of function names to filter (None = all functions)
            
        Returns:
            Dictionary mapping ticker to DataFrame (includes DataType column: 'signal' or 'target')
        """
        if self.use_new_structure:
            return self.load_stock_data_new_structure(
                tickers, from_date, to_date, dedup_columns, functions
            )
        else:
            return self.load_stock_data_legacy(tickers, from_date, to_date)
    
    def get_available_manus_files(self) -> List[str]:
        """
        Get list of available Manus data files.
        
        Returns:
            List of file names (without .csv extension)
        """
        try:
            csv_files = list(self.manus_data_dir.glob("*.csv"))
            files = [f.stem for f in csv_files]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error getting available Manus files: {e}")
            return []
    
    def load_manus_data(self, file_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load Manus chat data files.
        
        Args:
            file_names: List of file names to load (without .csv). If None, loads all.
            
        Returns:
            Dictionary mapping file name to DataFrame
        """
        result = {}
        
        if file_names is None:
            file_names = self.get_available_manus_files()
        
        for file_name in file_names:
            try:
                file_path = self.manus_data_dir / f"{file_name}.csv"
                
                if not file_path.exists():
                    logger.warning(f"Manus data file not found: {file_name}")
                    continue
                
                df = pd.read_csv(file_path, encoding=CSV_ENCODING)
                
                # Limit rows if too many
                if len(df) > MAX_ROWS_TO_INCLUDE:
                    logger.info(f"Limiting {file_name} data from {len(df)} to {MAX_ROWS_TO_INCLUDE} rows")
                    df = df.head(MAX_ROWS_TO_INCLUDE)
                
                result[file_name] = df
                logger.info(f"Loaded {len(df)} rows from {file_name}")
                
            except Exception as e:
                logger.error(f"Error loading Manus data {file_name}: {e}")
                continue
        
        return result
    
    def format_data_for_prompt(
        self,
        stock_data: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Format loaded data into a text representation for GPT-4o prompt.
        
        Args:
            stock_data: Dictionary of stock DataFrames
            
        Returns:
            Formatted string representation of the data
        """
        formatted_parts = []
        
        # Format stock data
        if stock_data:
            formatted_parts.append("=== TRADING DATA ===\n")
            for ticker, df in stock_data.items():
                formatted_parts.append(f"\n--- {ticker} ---")
                
                if 'Date' in df.columns:
                    formatted_parts.append(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
                
                formatted_parts.append(f"Number of Records: {len(df)}")
                formatted_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                
                # Add summary statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    formatted_parts.append("\nSummary Statistics:")
                    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                        formatted_parts.append(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}")
                
                # Add sample of data (first 5 and last 5 rows)
                formatted_parts.append("\nData Sample (first 5 and last 5 rows):")
                if len(df) <= 10:
                    sample_df = df
                else:
                    sample_df = pd.concat([df.head(5), df.tail(5)])
                formatted_parts.append(sample_df.to_string(index=False))
                formatted_parts.append("\n")
        
        return "\n".join(formatted_parts)
    
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
