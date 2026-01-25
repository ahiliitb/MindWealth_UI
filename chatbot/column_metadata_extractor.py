"""
Column metadata extractor for analyzing available columns across all signal types.
Scans the data directory structure and extracts column names from CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional
import logging
import json
from collections import defaultdict

from .config import (
    CHATBOT_ENTRY_DIR,
    CHATBOT_EXIT_DIR,
    CHATBOT_TARGET_DIR,
    CHATBOT_BREADTH_DIR,
    CSV_ENCODING
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnMetadataExtractor:
    """
    Extracts and caches column metadata from all signal types.
    Provides information about available columns for each signal type and function.
    """
    
    def __init__(self):
        """Initialize the column metadata extractor."""
        self.entry_dir = Path(CHATBOT_ENTRY_DIR)
        self.exit_dir = Path(CHATBOT_EXIT_DIR)
        self.target_dir = Path(CHATBOT_TARGET_DIR)
        self.breadth_dir = Path(CHATBOT_BREADTH_DIR)
        
        # Cache for column metadata
        self._metadata_cache: Optional[Dict] = None
        
    def extract_all_metadata(self, force_refresh: bool = False) -> Dict:
        """
        Extract metadata for all signal types.
        
        Args:
            force_refresh: If True, refresh the cache even if it exists
            
        Returns:
            Dictionary with structure:
            {
                "entry": {
                    "TRENDPULSE": ["Column1", "Column2", ...],
                    "FRACTAL TRACK": [...]
                },
                "exit": {...},
                "portfolio_target_achieved": {...},
                "breadth": [...] (no functions, just columns)
            }
        """
        if self._metadata_cache is not None and not force_refresh:
            return self._metadata_cache
        
        logger.info("Extracting column metadata from all signal types...")
        
        metadata = {
            "entry": self._extract_signal_type_metadata(self.entry_dir),
            "exit": self._extract_signal_type_metadata(self.exit_dir),
            "portfolio_target_achieved": self._extract_signal_type_metadata(self.target_dir),
            "breadth": self._extract_breadth_metadata()
        }
        
        self._metadata_cache = metadata
        logger.info("Column metadata extraction complete")
        
        return metadata
    
    def _extract_signal_type_metadata(self, base_dir: Path) -> Dict[str, List[str]]:
        """
        Extract column metadata for a signal type (entry/exit/portfolio_target_achieved).
        
        Args:
            base_dir: Base directory for the signal type (e.g., entry_dir)
            
        Returns:
            Dictionary mapping function names to list of columns
        """
        if not base_dir.exists():
            logger.warning(f"Directory does not exist: {base_dir}")
            return {}
        
        function_columns = {}
        
        try:
            # Iterate through asset directories
            for asset_dir in base_dir.iterdir():
                if not asset_dir.is_dir():
                    continue
                
                # Iterate through function directories
                for function_dir in asset_dir.iterdir():
                    if not function_dir.is_dir():
                        continue
                    
                    function_name = function_dir.name
                    
                    # Skip if we already have columns for this function
                    if function_name in function_columns:
                        continue
                    
                    # Find the first CSV file in this function directory
                    csv_files = list(function_dir.glob("*.csv"))
                    if not csv_files:
                        continue
                    
                    # Read column names from the first CSV file
                    try:
                        df = pd.read_csv(csv_files[0], nrows=0, encoding=CSV_ENCODING)
                        columns = df.columns.tolist()
                        function_columns[function_name] = columns
                        logger.info(f"Extracted {len(columns)} columns for function: {function_name}")
                    except Exception as e:
                        logger.error(f"Error reading CSV {csv_files[0]}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting metadata from {base_dir}: {e}")
        
        return function_columns
    
    def _extract_breadth_metadata(self) -> List[str]:
        """
        Extract column metadata for breadth signals.
        Breadth signals don't have functions/assets, just date-based files.
        
        Returns:
            List of column names
        """
        if not self.breadth_dir.exists():
            logger.warning(f"Breadth directory does not exist: {self.breadth_dir}")
            return []
        
        try:
            # Find the first CSV file in breadth directory
            csv_files = list(self.breadth_dir.glob("*.csv"))
            if not csv_files:
                logger.warning("No CSV files found in breadth directory")
                return []
            
            # Read column names from the first CSV file
            df = pd.read_csv(csv_files[0], nrows=0, encoding=CSV_ENCODING)
            columns = df.columns.tolist()
            logger.info(f"Extracted {len(columns)} columns for breadth signals")
            return columns
        
        except Exception as e:
            logger.error(f"Error extracting breadth metadata: {e}")
            return []
    
    def get_all_unique_columns(self) -> List[str]:
        """
        Get all unique column names across all signal types and functions.
        
        Returns:
            Sorted list of unique column names
        """
        metadata = self.extract_all_metadata()
        all_columns = set()
        
        # Add columns from entry, exit, portfolio_target_achieved
        for signal_type in ["entry", "exit", "portfolio_target_achieved"]:
            for function_name, columns in metadata[signal_type].items():
                all_columns.update(columns)
        
        # Add breadth columns
        all_columns.update(metadata["breadth"])
        
        return sorted(list(all_columns))
    
    def get_columns_for_signal_type(self, signal_type: str) -> Dict[str, List[str]]:
        """
        Get all columns for a specific signal type.
        
        Args:
            signal_type: One of "entry", "exit", "portfolio_target_achieved", "breadth"
            
        Returns:
            For entry/exit/portfolio_target_achieved: Dict mapping function names to columns
            For breadth: Dict with single key "breadth" mapping to columns
        """
        metadata = self.extract_all_metadata()
        
        if signal_type not in metadata:
            logger.warning(f"Invalid signal type: {signal_type}")
            return {}
        
        if signal_type == "breadth":
            return {"breadth": metadata["breadth"]}
        
        return metadata[signal_type]
    
    def get_columns_for_function(
        self, 
        signal_type: str, 
        function_name: str
    ) -> List[str]:
        """
        Get columns for a specific function within a signal type.
        
        Args:
            signal_type: One of "entry", "exit", "portfolio_target_achieved", "breadth"
            function_name: Name of the function (e.g., "TRENDPULSE")
            
        Returns:
            List of column names
        """
        metadata = self.extract_all_metadata()
        
        if signal_type == "breadth":
            return metadata["breadth"]
        
        if signal_type not in metadata:
            return []
        
        return metadata[signal_type].get(function_name, [])
    
    def save_metadata_to_json(self, output_path: Path) -> None:
        """
        Save extracted metadata to a JSON file for inspection.
        
        Args:
            output_path: Path where to save the JSON file
        """
        metadata = self.extract_all_metadata()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to JSON: {e}")
    
    def get_metadata_summary(self) -> Dict:
        """
        Get a summary of the metadata (counts, etc.).
        
        Returns:
            Dictionary with summary statistics
        """
        metadata = self.extract_all_metadata()
        
        summary = {
            "entry": {
                "num_functions": len(metadata["entry"]),
                "functions": list(metadata["entry"].keys()),
                "total_unique_columns": len(set(col for cols in metadata["entry"].values() for col in cols))
            },
            "exit": {
                "num_functions": len(metadata["exit"]),
                "functions": list(metadata["exit"].keys()),
                "total_unique_columns": len(set(col for cols in metadata["exit"].values() for col in cols))
            },
            "portfolio_target_achieved": {
                "num_functions": len(metadata["portfolio_target_achieved"]),
                "functions": list(metadata["portfolio_target_achieved"].keys()),
                "total_unique_columns": len(set(col for cols in metadata["portfolio_target_achieved"].values() for col in cols))
            },
            "breadth": {
                "num_columns": len(metadata["breadth"]),
                "columns": metadata["breadth"]
            },
            "total_unique_columns_all": len(self.get_all_unique_columns())
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the metadata."""
        summary = self.get_metadata_summary()
        
        print("\n" + "="*60)
        print("COLUMN METADATA SUMMARY")
        print("="*60)
        
        for signal_type in ["entry", "exit", "portfolio_target_achieved"]:
            print(f"\n{signal_type.upper()}:")
            print(f"  Functions: {summary[signal_type]['num_functions']}")
            print(f"  Unique columns: {summary[signal_type]['total_unique_columns']}")
            print(f"  Function names: {', '.join(summary[signal_type]['functions'])}")
        
        print(f"\nBREADTH:")
        print(f"  Columns: {summary['breadth']['num_columns']}")
        print(f"  Column names: {', '.join(summary['breadth']['columns'])}")
        
        print(f"\nTOTAL UNIQUE COLUMNS ACROSS ALL TYPES: {summary['total_unique_columns_all']}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test the extractor
    extractor = ColumnMetadataExtractor()
    extractor.print_summary()
    
    # Save to JSON
    output_path = Path(__file__).parent / "column_metadata.json"
    extractor.save_metadata_to_json(output_path)
    print(f"Metadata saved to: {output_path}")
