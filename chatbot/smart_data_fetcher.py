"""
Smart data fetcher that retrieves only the required columns from CSV files.
Fetches data based on asset name, function, date, and selected columns.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime

from .config import (
    CHATBOT_ENTRY_DIR,
    CHATBOT_EXIT_DIR,
    CHATBOT_TARGET_DIR,
    CHATBOT_BREADTH_DIR,
    CSV_ENCODING,
    DATE_FORMAT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartDataFetcher:
    """
    Fetches only the required columns from data files based on:
    - Signal type (entry/exit/target/breadth)
    - Asset name (ticker)
    - Function name (trading strategy)
    - Date
    - Required columns
    """
    
    def __init__(self):
        """Initialize the smart data fetcher."""
        self.entry_dir = Path(CHATBOT_ENTRY_DIR)
        self.exit_dir = Path(CHATBOT_EXIT_DIR)
        self.target_dir = Path(CHATBOT_TARGET_DIR)
        self.breadth_dir = Path(CHATBOT_BREADTH_DIR)
    
    def fetch_data(
        self,
        signal_types: List[str],
        required_columns: List[str],
        assets: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit_rows: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from specified signal types with only the required columns.
        
        Args:
            signal_types: List of signal types to fetch from (entry, exit, target, breadth)
            required_columns: List of column names to fetch
            assets: Optional list of asset/ticker names to filter by
            functions: Optional list of function names to filter by
            from_date: Optional start date (YYYY-MM-DD)
            to_date: Optional end date (YYYY-MM-DD)
            limit_rows: Optional limit on number of rows per signal type
            
        Returns:
            Dictionary mapping signal_type to DataFrame with fetched data
            {
                "entry": DataFrame with required columns,
                "exit": DataFrame with required columns,
                ...
            }
        """
        result = {}
        
        for signal_type in signal_types:
            try:
                if signal_type == "breadth":
                    df = self._fetch_breadth_data(
                        required_columns=required_columns,
                        from_date=from_date,
                        to_date=to_date,
                        limit_rows=limit_rows
                    )
                else:
                    df = self._fetch_signal_type_data(
                        signal_type=signal_type,
                        required_columns=required_columns,
                        assets=assets,
                        functions=functions,
                        from_date=from_date,
                        to_date=to_date,
                        limit_rows=limit_rows
                    )
                
                if not df.empty:
                    result[signal_type] = df
                    logger.info(f"Fetched {len(df)} rows from {signal_type}")
                else:
                    logger.warning(f"No data fetched from {signal_type}")
            
            except Exception as e:
                logger.error(f"Error fetching data from {signal_type}: {e}")
        
        return result
    
    def _fetch_signal_type_data(
        self,
        signal_type: str,
        required_columns: List[str],
        assets: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch data from a signal type (entry/exit/target).
        
        Args:
            signal_type: One of "entry", "exit", "target"
            required_columns: List of column names to fetch
            assets: Optional list of assets to filter by
            functions: Optional list of functions to filter by
            from_date: Optional start date
            to_date: Optional end date
            limit_rows: Optional row limit
            
        Returns:
            DataFrame with fetched data
        """
        # Get base directory for signal type
        base_dir = self._get_signal_type_dir(signal_type)
        if not base_dir.exists():
            logger.warning(f"Directory does not exist: {base_dir}")
            return pd.DataFrame()
        
        all_data = []
        
        # Iterate through asset directories
        for asset_dir in base_dir.iterdir():
            if not asset_dir.is_dir():
                continue
            
            asset_name = asset_dir.name
            
            # Filter by assets if specified
            if assets and asset_name not in assets:
                continue
            
            # Iterate through function directories
            for function_dir in asset_dir.iterdir():
                if not function_dir.is_dir():
                    continue
                
                function_name = function_dir.name
                
                # Filter by functions if specified
                if functions and function_name not in functions:
                    continue
                
                # Get CSV files in date range
                csv_files = self._get_csv_files_in_range(
                    function_dir,
                    from_date,
                    to_date
                )
                
                # Read data from CSV files
                for csv_file in csv_files:
                    try:
                        df = self._read_csv_with_columns(csv_file, required_columns)
                        
                        if not df.empty:
                            # Add metadata columns
                            df['_signal_type'] = signal_type
                            df['_asset'] = asset_name
                            df['_function'] = function_name
                            df['_date'] = csv_file.stem  # filename is the date
                            
                            all_data.append(df)
                    
                    except Exception as e:
                        logger.error(f"Error reading {csv_file}: {e}")
        
        # Combine all data
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Apply row limit if specified
        if limit_rows and len(combined_df) > limit_rows:
            combined_df = combined_df.head(limit_rows)
        
        return combined_df
    
    def _fetch_breadth_data(
        self,
        required_columns: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch breadth data (no assets/functions, just date-based files).
        
        Args:
            required_columns: List of column names to fetch
            from_date: Optional start date
            to_date: Optional end date
            limit_rows: Optional row limit
            
        Returns:
            DataFrame with fetched data
        """
        if not self.breadth_dir.exists():
            logger.warning(f"Breadth directory does not exist: {self.breadth_dir}")
            return pd.DataFrame()
        
        # Get CSV files in date range
        csv_files = self._get_csv_files_in_range(
            self.breadth_dir,
            from_date,
            to_date
        )
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = self._read_csv_with_columns(csv_file, required_columns)
                
                if not df.empty:
                    # Add metadata
                    df['_signal_type'] = 'breadth'
                    df['_date'] = csv_file.stem
                    all_data.append(df)
            
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
        
        # Combine all data
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Apply row limit if specified
        if limit_rows and len(combined_df) > limit_rows:
            combined_df = combined_df.head(limit_rows)
        
        return combined_df
    
    def _get_signal_type_dir(self, signal_type: str) -> Path:
        """Get the base directory for a signal type."""
        if signal_type == "entry":
            return self.entry_dir
        elif signal_type == "exit":
            return self.exit_dir
        elif signal_type == "target":
            return self.target_dir
        elif signal_type == "breadth":
            return self.breadth_dir
        else:
            raise ValueError(f"Invalid signal type: {signal_type}")
    
    def _get_csv_files_in_range(
        self,
        directory: Path,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Path]:
        """
        Get CSV files in a directory within the specified date range.
        
        Args:
            directory: Directory to search
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of CSV file paths
        """
        csv_files = sorted(list(directory.glob("*.csv")))
        
        if not from_date and not to_date:
            return csv_files
        
        # Filter by date range
        filtered_files = []
        
        for csv_file in csv_files:
            file_date_str = csv_file.stem
            
            # Try to parse the date from filename
            try:
                file_date = datetime.strptime(file_date_str, DATE_FORMAT)
                
                # Check if within range
                if from_date:
                    from_date_obj = datetime.strptime(from_date, DATE_FORMAT)
                    if file_date < from_date_obj:
                        continue
                
                if to_date:
                    to_date_obj = datetime.strptime(to_date, DATE_FORMAT)
                    if file_date > to_date_obj:
                        continue
                
                filtered_files.append(csv_file)
            
            except ValueError:
                # If filename is not a date, skip it
                logger.warning(f"Could not parse date from filename: {csv_file.name}")
                continue
        
        return filtered_files
    
    def _read_csv_with_columns(
        self,
        csv_file: Path,
        required_columns: List[str]
    ) -> pd.DataFrame:
        """
        Read a CSV file and return only the required columns.
        Uses flexible matching: exact match, partial match, or semantic match.
        
        Args:
            csv_file: Path to CSV file
            required_columns: List of column names to read
            
        Returns:
            DataFrame with only the required columns
        """
        try:
            # First read just the header to see what columns are available
            df_header = pd.read_csv(csv_file, nrows=0, encoding=CSV_ENCODING)
            available_columns = df_header.columns.tolist()
            
            # Find which required columns exist in this file using flexible matching
            columns_to_read = self._match_columns_flexibly(required_columns, available_columns)
            
            if not columns_to_read:
                logger.debug(f"None of the required columns found in {csv_file.name}")
                return pd.DataFrame()
            
            # Read only the required columns
            df = pd.read_csv(csv_file, usecols=columns_to_read, encoding=CSV_ENCODING)
            
            return df
        
        except Exception as e:
            logger.error(f"Error reading CSV {csv_file}: {e}")
            return pd.DataFrame()
    
    def _match_columns_flexibly(
        self,
        required_columns: List[str],
        available_columns: List[str]
    ) -> List[str]:
        """
        Match required columns to available columns using flexible matching.
        
        Matching strategies:
        1. Exact match (case-insensitive)
        2. Partial match (column contains required keyword)
        3. Semantic match (e.g., "target" matches "target_1", "target_price", etc.)
        
        Args:
            required_columns: List of column names requested
            available_columns: List of column names in the CSV
            
        Returns:
            List of actual column names to read from CSV
        """
        matched_columns = []
        
        for req_col in required_columns:
            req_col_lower = req_col.lower().strip()
            
            # Strategy 1: Exact match (case-insensitive)
            for avail_col in available_columns:
                if avail_col.lower().strip() == req_col_lower:
                    if avail_col not in matched_columns:
                        matched_columns.append(avail_col)
                    break
            else:
                # Strategy 2: Partial match - check if required column is in available column
                # OR if available column contains the required column keyword
                matched = False
                for avail_col in available_columns:
                    avail_col_lower = avail_col.lower().strip()
                    
                    # Check if they share common keywords
                    if (req_col_lower in avail_col_lower or 
                        avail_col_lower in req_col_lower or
                        self._are_semantically_related(req_col_lower, avail_col_lower)):
                        if avail_col not in matched_columns:
                            matched_columns.append(avail_col)
                            matched = True
                
                if not matched:
                    logger.debug(f"Could not match required column '{req_col}' to any available columns")
        
        return matched_columns
    
    def _are_semantically_related(self, col1: str, col2: str) -> bool:
        """
        Check if two column names are semantically related.
        
        Examples:
        - "target" matches "target_1", "target_price", "target_reached"
        - "entry" matches "entry_date", "entry_price", "entry_signal"
        - "performance" matches "current_performance", "performance_pct"
        
        Args:
            col1: First column name (lowercase)
            col2: Second column name (lowercase)
            
        Returns:
            True if semantically related
        """
        # Define keyword groups that are semantically related
        related_groups = [
            {'target', 'target_1', 'target_2', 'target_3', 'target_price', 'target_reached', 'target_hit'},
            {'entry', 'entry_date', 'entry_price', 'entry_signal', 'entry_time'},
            {'exit', 'exit_date', 'exit_price', 'exit_signal', 'exit_time'},
            {'performance', 'current_performance', 'performance_pct', 'perf', 'pnl'},
            {'price', 'current_price', 'close_price', 'open_price', 'close', 'open'},
            {'date', 'signal_date', 'entry_date', 'exit_date', 'timestamp'},
            {'signal', 'signal_type', 'signal_date', 'signal_strength'},
            {'volume', 'vol', 'avg_volume', 'volume_ratio'},
            {'rsi', 'rsi_14', 'rsi_value'},
            {'macd', 'macd_line', 'macd_signal', 'macd_hist'},
            {'bollinger', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_width'},
            {'stochastic', 'stoch', 'stoch_k', 'stoch_d'},
            {'divergence', 'div', 'bullish_div', 'bearish_div'},
            {'trend', 'trendline', 'uptrend', 'downtrend'},
        ]
        
        # Check if both columns belong to the same semantic group
        for group in related_groups:
            # Check if any word from col1 or col2 is in this group
            col1_words = set(col1.replace('_', ' ').split())
            col2_words = set(col2.replace('_', ' ').split())
            
            if (any(word in group for word in col1_words) and 
                any(word in group for word in col2_words)):
                return True
        
        return False
    
    def get_data_summary(
        self,
        signal_type: str,
        asset: Optional[str] = None,
        function: Optional[str] = None
    ) -> Dict:
        """
        Get summary information about available data.
        
        Args:
            signal_type: One of "entry", "exit", "target", "breadth"
            asset: Optional asset name
            function: Optional function name
            
        Returns:
            Dictionary with summary info (available dates, row counts, etc.)
        """
        base_dir = self._get_signal_type_dir(signal_type)
        
        if signal_type == "breadth":
            csv_files = list(base_dir.glob("*.csv"))
            return {
                "signal_type": signal_type,
                "num_files": len(csv_files),
                "dates": [f.stem for f in sorted(csv_files)]
            }
        
        # For entry/exit/target
        if asset and function:
            function_dir = base_dir / asset / function
            if function_dir.exists():
                csv_files = list(function_dir.glob("*.csv"))
                return {
                    "signal_type": signal_type,
                    "asset": asset,
                    "function": function,
                    "num_files": len(csv_files),
                    "dates": [f.stem for f in sorted(csv_files)]
                }
        
        return {"signal_type": signal_type, "error": "Invalid parameters"}


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = SmartDataFetcher()
    
    # Test 1: Fetch entry data for TSM
    print("\n" + "="*60)
    print("TEST 1: Fetch entry data for TSM")
    print("="*60)
    
    result = fetcher.fetch_data(
        signal_types=["entry"],
        required_columns=["Symbol", "Signal", "Current Mark to Market and Holding Period"],
        assets=["TSM"],
        from_date="2025-10-14",
        to_date="2025-10-14"
    )
    
    if "entry" in result:
        print(f"\nFetched {len(result['entry'])} rows")
        print(result['entry'].head())
    
    # Test 2: Fetch breadth data
    print("\n" + "="*60)
    print("TEST 2: Fetch breadth data")
    print("="*60)
    
    result = fetcher.fetch_data(
        signal_types=["breadth"],
        required_columns=["Function", "Bullish Asset vs Total Asset (%)", "Date"],
        from_date="2025-10-14"
    )
    
    if "breadth" in result:
        print(f"\nFetched {len(result['breadth'])} rows")
        print(result['breadth'].head())
