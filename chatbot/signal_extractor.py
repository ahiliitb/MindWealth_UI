"""
Signal extractor to identify signals mentioned in AI responses and fetch complete signal data.
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime

from .smart_data_fetcher import SmartDataFetcher
from .config import (
    CHATBOT_ENTRY_DIR,
    CHATBOT_EXIT_DIR, 
    CHATBOT_TARGET_DIR,
    CHATBOT_BREADTH_DIR,
    DATE_FORMAT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalExtractor:
    """
    Extracts signal information from AI responses and fetches complete signal data with all columns.
    """
    
    def __init__(self):
        """Initialize the signal extractor."""
        self.smart_data_fetcher = SmartDataFetcher()
        
    def extract_full_signal_tables(
        self,
        ai_response: str,
        fetched_data: Dict[str, pd.DataFrame] = None,
        query_params: Dict = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract complete signal tables from AI response by identifying used signals and fetching full data.
        
        Args:
            ai_response: The AI's response text
            fetched_data: Data that was used to generate the response (for reference)
            query_params: Query parameters used (assets, functions, date range, etc.)
            
        Returns:
            Dictionary mapping signal types to complete DataFrames with all columns
        """
        logger.info("Extracting full signal tables from AI response...")
        
        # Step 1: Identify which signals were actually used/mentioned
        used_signals = self._identify_used_signals(ai_response, fetched_data)
        
        # Step 2: For each signal type that was used, fetch complete data
        signal_tables = {}
        
        if not used_signals:
            logger.info("No signals identified in response")
            return signal_tables
            
        # Get query parameters for data fetching
        assets = query_params.get('assets', []) if query_params else []
        functions = query_params.get('functions', []) if query_params else []
        from_date = query_params.get('from_date') if query_params else None
        to_date = query_params.get('to_date') if query_params else None
        
        # Step 3: Fetch complete data for each signal type mentioned
        for signal_type in used_signals.get('signal_types', set()):
            try:
                logger.info(f"Fetching complete {signal_type} data...")
                
                # First, get a sample to determine available columns
                sample_data = self.smart_data_fetcher.fetch_data(
                    signal_types=[signal_type],
                    required_columns=["Function"],  # Minimal request to get column info
                    assets=assets[:1] if assets else None,  # Just one asset for sample
                    functions=functions[:1] if functions else None,  # Just one function for sample
                    from_date=from_date,
                    to_date=to_date,
                    limit_rows=1
                )
                
                # Fetch ALL available columns from CSV files in their original order
                # This preserves the exact structure as it appears in the CSV files
                logger.info(f"Fetching ALL columns for {signal_type} to preserve original CSV structure")
                
                # Fetch complete data with ALL columns (no column filtering)
                complete_data = self.smart_data_fetcher.fetch_data(
                    signal_types=[signal_type],
                    required_columns=None,  # Fetch ALL columns
                    assets=assets,
                    functions=functions,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if signal_type in complete_data and not complete_data[signal_type].empty:
                    # Filter to only the signals that were actually used
                    filtered_data = self._filter_used_signals(
                        complete_data[signal_type], 
                        used_signals,
                        used_signals.get('signal_keys', [])
                    )
                    
                    if not filtered_data.empty:
                        # Remove metadata columns to show only original CSV structure
                        cleaned_data = filtered_data.copy()
                        metadata_columns = ['_signal_type', '_asset', '_function', '_date']
                        for col in metadata_columns:
                            if col in cleaned_data.columns:
                                cleaned_data = cleaned_data.drop(columns=[col])
                        
                        signal_tables[signal_type] = cleaned_data
                        logger.info(f"Added {len(cleaned_data)} {signal_type} signals to table (showing only original CSV columns)")
                
            except Exception as e:
                logger.error(f"Error fetching complete {signal_type} data: {e}")
                
        logger.info(f"Extracted {len(signal_tables)} signal type tables")
        return signal_tables
    
    def _identify_used_signals(
        self,
        ai_response: str,
        fetched_data: Dict[str, pd.DataFrame] = None
    ) -> Dict[str, Set]:
        """
        Extract the 4 key signal identifiers: function, symbol, interval, signal date.
        These will be used as keys to fetch complete signal data.
        
        IMPORTANT: Only include signal types that are actually mentioned in the AI response,
        not all signal types from fetched_data.
        
        Args:
            ai_response: The AI response text
            fetched_data: The data that was fetched for the response
            
        Returns:
            Dictionary with sets of used signal_types, symbols, functions, dates, intervals
        """
        used_signals = {
            'signal_types': set(),
            'symbols': set(),
            'functions': set(),
            'dates': set(),
            'intervals': set(),
            'signal_keys': []  # List of (function, symbol, interval, date) tuples
        }
        
        # Strategy 1: Parse AI response to determine which signal types are actually mentioned
        # This is the PRIMARY source of truth for what should be included
        signal_type_patterns = {
            'entry': r'\b(entry|entries|entering|long|short|buy|buying)\b',
            'exit': r'\b(exit|exits|exiting|close|closing|sell|selling)\b',
            'target': r'\b(target|targets|profit|take.?profit)\b',
            'breadth': r'\b(breadth|market.?breadth|sector)\b'
        }
        
        mentioned_signal_types = set()
        for signal_type, pattern in signal_type_patterns.items():
            if re.search(pattern, ai_response, re.IGNORECASE):
                mentioned_signal_types.add(signal_type)
                logger.info(f"Found {signal_type} signals mentioned in AI response")
        
        # Strategy 2: Extract specific signals mentioned in GPT's response
        # Parse GPT response to find specific signals it referenced
        specific_signals_mentioned = self._extract_specific_signals_from_response(ai_response)
        
        if specific_signals_mentioned:
            # Use only the specific signals GPT mentioned
            used_signals['signal_keys'] = specific_signals_mentioned
            for function, symbol, interval, signal_date in specific_signals_mentioned:
                used_signals['functions'].add(function)
                used_signals['symbols'].add(symbol)
                used_signals['intervals'].add(interval)
                used_signals['dates'].add(signal_date)
            
            # Determine signal types from mentioned signal types OR default to 'entry'
            # This ensures tables are generated for the specific signals
            if mentioned_signal_types:
                used_signals['signal_types'].update(mentioned_signal_types)
            else:
                # Default to 'entry' when specific signals are mentioned but no explicit signal type
                used_signals['signal_types'].add('entry')
            
            logger.info(f"Found {len(specific_signals_mentioned)} specific signals mentioned in GPT response")
            logger.info(f"Will generate tables for signal types: {used_signals['signal_types']}")
            
            # When we have specific signals, we should ONLY process those - skip the fallback data processing
            return used_signals
        
        elif fetched_data:
            # Fallback: If no specific signals found, extract from available data
            # If specific signal types were mentioned, use only those
            # If no signal types mentioned, use ALL available signal types (default behavior)
            signal_types_to_process = mentioned_signal_types if mentioned_signal_types else set(fetched_data.keys())
            
            for signal_type, df in fetched_data.items():
                # Process signal types that were mentioned OR all types if none specifically mentioned
                if signal_type in signal_types_to_process and not df.empty:
                    used_signals['signal_types'].add(signal_type)
                    logger.info(f"Extracting key signal data from {signal_type}")
                    
                    # Extract the 4 key pieces from each row
                    for _, row in df.iterrows():
                        signal_key = self._extract_signal_key(row)
                        if signal_key:
                            function, symbol, interval, signal_date = signal_key
                            used_signals['functions'].add(function)
                            used_signals['symbols'].add(symbol)
                            used_signals['intervals'].add(interval)
                            used_signals['dates'].add(signal_date)
                            used_signals['signal_keys'].append(signal_key)
        
        # Strategy 3: Parse AI response for additional signal components
        
        # Extract stock symbols (3-5 uppercase letters)
        symbol_matches = re.findall(r'\b[A-Z]{2,5}\b', ai_response)
        for symbol in symbol_matches:
            # Filter out common words that might match the pattern
            if symbol not in ['USD', 'API', 'AI', 'CEO', 'IPO', 'ETF', 'NYSE', 'NASDAQ', 'SEC']:
                used_signals['symbols'].add(symbol)
        
        # Extract function mentions (common trading functions)
        function_patterns = [
            r'FRACTAL\s+TRACK',
            r'BOLLINGER\s+BAND',
            r'MATRIX\s+DIVERGENCE',
            r'BREAKOUT\s+MATRIX',
            r'TRENDLINE',
            r'FIBONACCI',
            r'RSI',
            r'MACD',
            r'STOCHASTIC'
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, ai_response, re.IGNORECASE)
            for match in matches:
                used_signals['functions'].add(match.upper())
        
        # Extract dates (YYYY-MM-DD format)
        date_matches = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', ai_response)
        for date in date_matches:
            used_signals['dates'].add(date)
        
        # Extract intervals (1D, 4H, 1H, etc.)
        interval_matches = re.findall(r'\b\d+[DHM]\b', ai_response, re.IGNORECASE)
        for interval in interval_matches:
            used_signals['intervals'].add(interval.upper())
        
        logger.info(f"Identified signal keys - Types: {used_signals['signal_types']}, "
                   f"Symbols: {len(used_signals['symbols'])}, "
                   f"Functions: {used_signals['functions']}, "
                   f"Intervals: {used_signals['intervals']}, "
                   f"Dates: {len(used_signals['dates'])}, "
                   f"Signal Keys: {len(used_signals['signal_keys'])}")
        
        return used_signals

    def _extract_specific_signals_from_response(self, ai_response: str) -> List[Tuple[str, str, str, str]]:
        """
        Parse GPT's response to extract specific signals it mentioned.
        
        Primary Strategy: Look for SIGNAL_KEYS JSON format
        Fallback: Parse text patterns for signal mentions
        
        Returns:
            List of (function, symbol, interval, signal_date) tuples for signals specifically mentioned
        """
        specific_signals = []
        
        # Strategy 1: Look for SIGNAL_KEYS JSON format (new primary method)
        signal_keys_pattern = r'SIGNAL_KEYS:\s*\[(.*?)\]'
        signal_keys_match = re.search(signal_keys_pattern, ai_response, re.DOTALL)
        
        if signal_keys_match:
            try:
                import json
                # Extract the JSON array content
                keys_content = signal_keys_match.group(1).strip()
                
                # Try to parse as JSON array
                # First, wrap it properly if needed
                if not keys_content.startswith('['):
                    keys_content = f'[{keys_content}]'
                else:
                    keys_content = f'[{keys_content}]'
                
                # Parse each signal key object
                signal_objects = json.loads(keys_content)
                
                for signal_obj in signal_objects:
                    if isinstance(signal_obj, dict) and all(key in signal_obj for key in ['function', 'symbol', 'interval', 'signal_date']):
                        function = signal_obj['function'].strip()
                        symbol = signal_obj['symbol'].strip()
                        interval = signal_obj['interval'].strip()
                        signal_date = signal_obj['signal_date'].strip()
                        
                        specific_signals.append((function, symbol.upper(), interval, signal_date))
                
                logger.info(f"Successfully parsed {len(specific_signals)} signals from SIGNAL_KEYS format")
                
                if specific_signals:
                    return specific_signals
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse SIGNAL_KEYS JSON: {e}")
                # Continue to fallback strategies
        
        # Strategy 2: Fallback - Look for table-like structures or formatted signal lists
        # Pattern: SYMBOL, Signal/Function, Date patterns
        symbol_signal_patterns = [
            # Match: "AAPL, Long, 2025-10-16" or "AAPL FRACTAL TRACK 2025-10-16"
            r'([A-Z]{2,5})[,\s]+([A-Z\s]+)[,\s]+(\d{4}-\d{2}-\d{2})',
            # Match: "AAPL FRACTAL TRACK" (without date)
            r'([A-Z]{2,5})\s+(FRACTAL\s+TRACK|BOLLINGER\s+BAND|MATRIX\s+DIVERGENCE|BREAKOUT\s+MATRIX)',
            # Match: Function Symbol Date
            r'(FRACTAL\s+TRACK|BOLLINGER\s+BAND|MATRIX\s+DIVERGENCE|BREAKOUT\s+MATRIX)\s+([A-Z]{2,5})\s+(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in symbol_signal_patterns:
            matches = re.findall(pattern, ai_response, re.IGNORECASE)
            for match in matches:
                if len(match) == 3:
                    symbol, function_or_signal, date = match
                    # Clean up function name
                    function = function_or_signal.upper().strip()
                    if function in ['LONG', 'SHORT', 'BUY', 'SELL']:
                        function = 'FRACTAL TRACK'  # Default function for basic signals
                    
                    specific_signals.append((function, symbol.upper(), 'Daily', date))
                elif len(match) == 2:
                    symbol, function = match
                    specific_signals.append((function.upper(), symbol.upper(), 'Daily', 'Unknown'))
        
        # Strategy 3: Look for numbered lists or bullet points mentioning specific signals
        # Pattern: "1. AAPL" or "- MSFT" followed by signal details
        list_pattern = r'(?:^\s*[\d\-\*\+]\s*\.?\s*|^[\-\*\+]\s*)([A-Z]{2,5})'
        list_matches = re.findall(list_pattern, ai_response, re.MULTILINE)
        
        # For each symbol found in lists, try to find associated function/date in nearby text
        for symbol in list_matches:
            # Look for function mentions near this symbol (within 200 chars)
            symbol_pos = ai_response.find(symbol)
            if symbol_pos != -1:
                nearby_text = ai_response[max(0, symbol_pos-100):symbol_pos+100]
                
                # Find function in nearby text
                function_matches = re.findall(r'(FRACTAL\s+TRACK|BOLLINGER\s+BAND|MATRIX\s+DIVERGENCE|BREAKOUT\s+MATRIX)', nearby_text, re.IGNORECASE)
                date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', nearby_text)
                
                function = function_matches[0].upper() if function_matches else 'FRACTAL TRACK'
                date = date_matches[0] if date_matches else 'Unknown'
                
                specific_signals.append((function, symbol.upper(), 'Daily', date))
        
        # Strategy 4: Look for explicit mentions of "top N" signals
        # If GPT mentions "top 5" or similar, try to extract exactly those signals
        top_n_pattern = r'(?:top|best)\s+(\d+)'
        top_n_matches = re.findall(top_n_pattern, ai_response, re.IGNORECASE)
        
        if top_n_matches and specific_signals:
            # Limit to the number mentioned (e.g., "top 5" -> limit to 5 signals)
            max_signals = int(top_n_matches[0])
            specific_signals = specific_signals[:max_signals]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_signals = []
        for signal in specific_signals:
            if signal not in seen:
                seen.add(signal)
                unique_signals.append(signal)
        
        logger.info(f"Extracted {len(unique_signals)} specific signals from GPT response using fallback patterns")
        return unique_signals

    def _extract_signal_key(self, row: pd.Series) -> Optional[Tuple[str, str, str, str]]:
        """
        Extract the 4 key signal identifiers from a data row: function, symbol, interval, signal date.
        
        Args:
            row: DataFrame row containing signal data
            
        Returns:
            Tuple of (function, symbol, interval, signal_date) or None
        """
        try:
            function = None
            symbol = None
            interval = None
            signal_date = None
            
            # Extract Function
            if 'Function' in row:
                function = str(row['Function']).strip()
            elif '_function' in row:
                function = str(row['_function']).strip()
            
            # Extract Symbol from various possible columns
            if 'Symbol, Signal, Signal Date/Price[$]' in row:
                # Parse combined column: "AAPL, Long, 2025-10-14 (Price: 247.66)"
                combined = str(row['Symbol, Signal, Signal Date/Price[$]'])
                parts = combined.split(',')
                if len(parts) >= 1:
                    symbol = parts[0].strip()
                if len(parts) >= 3:
                    # Extract date from "2025-10-14 (Price: 247.66)"
                    date_part = parts[2].strip()
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_part)
                    if date_match:
                        signal_date = date_match.group(1)
            
            # Try other symbol columns
            if not symbol:
                for col in ['Symbol', '_asset']:
                    if col in row and pd.notna(row[col]):
                        symbol = str(row[col]).strip()
                        break
            
            # Extract date if not found
            if not signal_date and '_date' in row:
                signal_date = str(row['_date']).strip()
            
            # Extract Interval
            if 'Interval, Confirmation Status' in row:
                interval_data = str(row['Interval, Confirmation Status'])
                interval_match = re.search(r'^([^,]+)', interval_data)
                if interval_match:
                    interval = interval_match.group(1).strip()
            
            # Default interval if not found
            if not interval:
                interval = "1D"  # Default to daily
            
            # Only return if we have at least function and symbol
            if function and symbol:
                return (function, symbol, interval or "1D", signal_date or "Unknown")
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting signal key from row: {e}")
            return None
    
    def _filter_used_signals(
        self,
        complete_df: pd.DataFrame,
        used_signals: Dict[str, Set],
        signal_keys: List[Tuple[str, str, str, str]] = None
    ) -> pd.DataFrame:
        """
        Filter complete signal data using the 4 key signal identifiers as precise filters.
        
        Args:
            complete_df: Complete signal DataFrame with all data
            used_signals: Dictionary with sets of used signal components
            signal_keys: List of (function, symbol, interval, date) tuples for precise matching
            
        Returns:
            Filtered DataFrame containing only used signals
        """
        if complete_df.empty:
            return complete_df
        
        # Start with all data
        filtered_df = complete_df.copy()
        
        # Strategy 1: Use precise signal keys if available (most accurate)
        if signal_keys:
            logger.info(f"Using {len(signal_keys)} precise signal keys for filtering")
            precise_matches = []
            
            for function, symbol, interval, signal_date in signal_keys:
                # Find rows that match this specific signal key
                match_mask = pd.Series([True] * len(complete_df), index=complete_df.index)
                
                # Match function
                if 'Function' in complete_df.columns:
                    match_mask &= complete_df['Function'].astype(str).str.upper() == function.upper()
                elif '_function' in complete_df.columns:
                    match_mask &= complete_df['_function'].astype(str).str.upper() == function.upper()
                
                # Match symbol
                symbol_matched = False
                for col in complete_df.columns:
                    if 'symbol' in col.lower() or 'asset' in col.lower():
                        symbol_mask = complete_df[col].astype(str).str.upper().str.contains(symbol.upper())
                        if symbol_mask.any():
                            match_mask &= symbol_mask
                            symbol_matched = True
                            break
                
                if not symbol_matched:
                    continue  # Skip if symbol not found
                
                # Match date if specific date provided
                if signal_date and signal_date != "Unknown":
                    date_matched = False
                    for col in complete_df.columns:
                        if 'date' in col.lower():
                            date_mask = complete_df[col].astype(str).str.contains(signal_date)
                            if date_mask.any():
                                match_mask &= date_mask
                                date_matched = True
                                break
                
                # Collect matching rows
                matches = complete_df[match_mask]
                if not matches.empty:
                    precise_matches.append(matches)
                    logger.info(f"Found {len(matches)} matches for {function}/{symbol}/{interval}/{signal_date}")
            
            if precise_matches:
                filtered_df = pd.concat(precise_matches).drop_duplicates()
                logger.info(f"Precise filtering: {len(filtered_df)} rows found using signal keys")
                return filtered_df
        
        # Strategy 2: Fallback to broader filtering using sets
        used_symbols = used_signals.get('symbols', set())
        used_functions = used_signals.get('functions', set())
        used_dates = used_signals.get('dates', set())
        used_intervals = used_signals.get('intervals', set())
        
        # Filter by symbols if we have specific ones mentioned
        if used_symbols:
            symbol_mask = False
            for col in complete_df.columns:
                if 'symbol' in col.lower() or 'asset' in col.lower():
                    for symbol in used_symbols:
                        col_mask = complete_df[col].astype(str).str.upper().str.contains(symbol.upper())
                        symbol_mask = symbol_mask | col_mask
            
            if symbol_mask.any():
                filtered_df = filtered_df[symbol_mask]
                logger.info(f"Filtered by symbols, {len(filtered_df)} rows remaining")
        
        # Filter by functions if we have specific ones mentioned
        if used_functions:
            function_mask = False
            for col in complete_df.columns:
                if 'function' in col.lower():
                    for function in used_functions:
                        col_mask = complete_df[col].astype(str).str.upper().str.contains(function.upper())
                        function_mask = function_mask | col_mask
            
            if function_mask.any():
                filtered_df = filtered_df[function_mask]
                logger.info(f"Filtered by functions, {len(filtered_df)} rows remaining")
        
        # Filter by dates if we have specific ones mentioned
        if used_dates and len(filtered_df) > 20:
            date_mask = False
            for col in complete_df.columns:
                if 'date' in col.lower():
                    for date in used_dates:
                        col_mask = complete_df[col].astype(str).str.contains(date)
                        date_mask = date_mask | col_mask
            
            if date_mask.any():
                filtered_df = filtered_df[date_mask]
                logger.info(f"Filtered by dates, {len(filtered_df)} rows remaining")
        
        # If still too many results, take the most recent ones
        if len(filtered_df) > 50:
            filtered_df = filtered_df.head(50)
            logger.info(f"Limited to 50 most recent signals")
        
        return filtered_df
    
    def _extract_signal_info_from_row(self, row: pd.Series, signal_type: str) -> Optional[Dict]:
        """
        Extract signal information from a DataFrame row.
        
        Args:
            row: DataFrame row containing signal data
            signal_type: Type of signal (entry, exit, target, breadth)
            
        Returns:
            Dictionary with signal information or None
        """
        try:
            signal_info = {
                'Signal_Type': signal_type.upper()
            }
            
            # Extract function name
            if 'Function' in row:
                signal_info['Function'] = row['Function']
            elif '_function' in row:
                signal_info['Function'] = row['_function']
            
            # Extract symbol from various possible columns
            symbol = None
            if 'Symbol, Signal, Signal Date/Price[$]' in row:
                # Parse combined column: "AAPL, Long, 2025-10-14 (Price: 247.66)"
                combined = str(row['Symbol, Signal, Signal Date/Price[$]'])
                parts = combined.split(',')
                if len(parts) >= 1:
                    symbol = parts[0].strip()
                if len(parts) >= 2:
                    signal_info['Signal_Direction'] = parts[1].strip()
                if len(parts) >= 3:
                    # Extract date and price from "2025-10-14 (Price: 247.66)"
                    date_price = parts[2].strip()
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_price)
                    if date_match:
                        signal_info['Signal_Date'] = date_match.group(1)
                    price_match = re.search(r'Price:\s*(\d+\.?\d*)', date_price)
                    if price_match:
                        signal_info['Price'] = f"${price_match.group(1)}"
            
            # Try other possible symbol columns
            if not symbol:
                for col in ['Symbol', '_asset']:
                    if col in row and pd.notna(row[col]):
                        symbol = str(row[col]).strip()
                        break
            
            if symbol:
                signal_info['Symbol'] = symbol
            
            # Extract interval information
            if 'Interval, Confirmation Status' in row:
                interval_data = str(row['Interval, Confirmation Status'])
                interval_match = re.search(r'^([^,]+)', interval_data)
                if interval_match:
                    signal_info['Interval'] = interval_match.group(1).strip()
            
            # Extract status information
            if 'Exit Signal Date/Price[$]' in row:
                exit_info = str(row['Exit Signal Date/Price[$]'])
                if 'No Exit Yet' in exit_info:
                    signal_info['Status'] = 'Active'
                else:
                    signal_info['Status'] = 'Closed'
            else:
                signal_info['Status'] = 'Unknown'
            
            # Only return if we have at least symbol and function
            if signal_info.get('Symbol') and signal_info.get('Function'):
                return signal_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting signal info from row: {e}")
            return None
    
    def _parse_signals_from_text(self, text: str) -> List[Dict]:
        """
        Parse AI response text to extract mentioned signals (fallback method).
        
        Args:
            text: AI response text
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        try:
            # Look for common patterns in AI responses
            # Pattern 1: "AAPL entry signal" or "MSFT exit signal"
            pattern1 = re.finditer(r'([A-Z]{1,5})\s+(entry|exit|target)\s+signal', text, re.IGNORECASE)
            for match in pattern1:
                signals.append({
                    'Symbol': match.group(1),
                    'Signal_Type': match.group(2).upper(),
                    'Status': 'Mentioned'
                })
            
            # Pattern 2: "FRACTAL TRACK for AAPL"
            pattern2 = re.finditer(r'([A-Z\s]+)\s+for\s+([A-Z]{1,5})', text)
            for match in pattern2:
                function = match.group(1).strip()
                if any(keyword in function.upper() for keyword in ['FRACTAL', 'BAND', 'MATRIX', 'TRACK']):
                    signals.append({
                        'Symbol': match.group(2),
                        'Function': function,
                        'Status': 'Mentioned'
                    })
            
            # Remove duplicates
            seen = set()
            unique_signals = []
            for signal in signals:
                key = (signal.get('Symbol', ''), signal.get('Function', ''), signal.get('Signal_Type', ''))
                if key not in seen:
                    seen.add(key)
                    unique_signals.append(signal)
            
            return unique_signals
            
        except Exception as e:
            logger.error(f"Error parsing signals from text: {e}")
            return []
    
    def extract_signals_from_response(
        self,
        ai_response: str,
        fetched_data: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Legacy method for backward compatibility - returns simplified signal info.
        For full tables with all columns, use extract_full_signal_tables() instead.
        """
        signals = []
        
        # Strategy 1: Use fetched_data if available (most reliable)
        if fetched_data:
            for signal_type, df in fetched_data.items():
                if df.empty:
                    continue
                
                for _, row in df.iterrows():
                    signal_info = self._extract_signal_info_from_row(row, signal_type)
                    if signal_info:
                        signals.append(signal_info)
        
        # Strategy 2: Parse AI response for signal mentions (fallback)
        if not signals:
            signals = self._parse_signals_from_text(ai_response)
        
        # Convert to DataFrame
        if signals:
            df = pd.DataFrame(signals)
            # Ensure consistent column order
            column_order = ['Symbol', 'Function', 'Signal_Type', 'Signal_Direction', 'Signal_Date', 'Price', 'Interval', 'Status']
            # Only include columns that exist
            existing_cols = [col for col in column_order if col in df.columns]
            return df[existing_cols]
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Symbol', 'Function', 'Signal_Type', 'Signal_Direction', 'Signal_Date', 'Price', 'Status'])