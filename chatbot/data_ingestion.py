"""
Data ingestion script for adding new signals to the consolidated CSV system.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_data_fetcher import SmartDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionManager:
    """
    Manages ingestion of new signal data into the consolidated CSV system.
    """

    def __init__(self):
        self.fetcher = SmartDataFetcher(use_consolidated_csvs=True)

    def ingest_entry_signals(self, signals_data: pd.DataFrame) -> bool:
        """
        Ingest new entry signals into the consolidated entry.csv.

        Args:
            signals_data: DataFrame with entry signal data. Should include:
                - symbol: Asset symbol
                - signal_type: 'Long' or 'Short'
                - asset_name: Asset name (same as symbol)
                - function: Trading function/strategy
                - interval: Time interval (Daily, Weekly, Monthly, Quarterly)
                - Signal Open Price: Price at which signal was generated
                - Plus any other signal-specific columns

        Returns:
            True if successful
        """
        if signals_data.empty:
            logger.warning("No entry signals to ingest")
            return False

        # Ensure required columns are present
        required_cols = ['symbol', 'signal_type', 'asset_name', 'function', 'interval', 'Signal Open Price']
        missing_cols = [col for col in required_cols if col not in signals_data.columns]

        if missing_cols:
            logger.error(f"Missing required columns for entry signals: {missing_cols}")
            return False

        # Add signal_type_meta column
        signals_data = signals_data.copy()
        signals_data['signal_type_meta'] = 'entry'

        success = self.fetcher.add_data_to_consolidated_csv('entry', signals_data, deduplicate=True)

        if success:
            logger.info(f"✅ Successfully ingested {len(signals_data)} entry signals")
        else:
            logger.error("❌ Failed to ingest entry signals")

        return success

    def ingest_exit_signals(self, signals_data: pd.DataFrame) -> bool:
        """
        Ingest new exit signals into the consolidated exit.csv.

        Args:
            signals_data: DataFrame with exit signal data. Should include:
                - symbol: Asset symbol
                - signal_type: 'Long' or 'Short'
                - asset_name: Asset name (same as symbol)
                - function: Trading function/strategy
                - interval: Time interval (Daily, Weekly, Monthly, Quarterly)
                - Signal Open Price: Price at which signal was generated
                - Plus any other signal-specific columns

        Returns:
            True if successful
        """
        if signals_data.empty:
            logger.warning("No exit signals to ingest")
            return False

        # Ensure required columns are present
        required_cols = ['symbol', 'signal_type', 'asset_name', 'function', 'interval', 'Signal Open Price']
        missing_cols = [col for col in required_cols if col not in signals_data.columns]

        if missing_cols:
            logger.error(f"Missing required columns for exit signals: {missing_cols}")
            return False

        # Add signal_type_meta column
        signals_data = signals_data.copy()
        signals_data['signal_type_meta'] = 'exit'

        success = self.fetcher.add_data_to_consolidated_csv('exit', signals_data, deduplicate=True)

        if success:
            logger.info(f"✅ Successfully ingested {len(signals_data)} exit signals")
        else:
            logger.error("❌ Failed to ingest exit signals")

        return success

    def ingest_portfolio_target_signals(self, signals_data: pd.DataFrame) -> bool:
        """
        Ingest new portfolio target achieved signals into the consolidated portfolio_target_achieved.csv.

        Args:
            signals_data: DataFrame with portfolio target signal data. Should include:
                - symbol: Asset symbol
                - signal_type: 'Long' or 'Short'
                - asset_name: Asset name (same as symbol)
                - function: Trading function/strategy
                - interval: Time interval (Daily, Weekly, Monthly, Quarterly)
                - Signal Open Price: Price at which signal was generated
                - Plus any other signal-specific columns

        Returns:
            True if successful
        """
        if signals_data.empty:
            logger.warning("No portfolio_target_achieved signals to ingest")
            return False

        # Ensure required columns are present
        required_cols = ['symbol', 'signal_type', 'asset_name', 'function', 'interval', 'Signal Open Price']
        missing_cols = [col for col in required_cols if col not in signals_data.columns]

        if missing_cols:
            logger.error(f"Missing required columns for portfolio_target_achieved signals: {missing_cols}")
            return False

        # Add signal_type_meta column
        signals_data = signals_data.copy()
        signals_data['signal_type_meta'] = 'portfolio_target_achieved'

        success = self.fetcher.add_data_to_consolidated_csv('portfolio_target_achieved', signals_data, deduplicate=True)

        if success:
            logger.info(f"✅ Successfully ingested {len(signals_data)} portfolio_target_achieved signals")
        else:
            logger.error("❌ Failed to ingest portfolio_target_achieved signals")

        return success

    def ingest_breadth_signals(self, signals_data: pd.DataFrame) -> bool:
        """
        Ingest new breadth signals into the consolidated breadth.csv.

        Args:
            signals_data: DataFrame with breadth signal data. Should include:
                - date: Date (YYYY-MM-DD)
                - Function: Trading function/strategy
                - Plus any other breadth-specific columns

        Returns:
            True if successful
        """
        if signals_data.empty:
            logger.warning("No breadth signals to ingest")
            return False

        # Ensure required columns are present
        required_cols = ['date', 'Function']
        missing_cols = [col for col in required_cols if col not in signals_data.columns]

        if missing_cols:
            logger.error(f"Missing required columns for breadth signals: {missing_cols}")
            return False

        # Add signal_type_meta column
        signals_data = signals_data.copy()
        signals_data['signal_type_meta'] = 'breadth'

        success = self.fetcher.add_data_to_consolidated_csv('breadth', signals_data, deduplicate=True)

        if success:
            logger.info(f"✅ Successfully ingested {len(signals_data)} breadth signals")
        else:
            logger.error("❌ Failed to ingest breadth signals")

        return success

    def get_data_summary(self) -> Dict:
        """
        Get summary of data in consolidated CSVs.

        Returns:
            Dictionary with summary information
        """
        summary = {}

        for signal_type in ['entry', 'exit', 'portfolio_target_achieved', 'breadth']:
            csv_path = self.fetcher._get_consolidated_csv_path(signal_type)

            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, nrows=0)  # Just read headers
                    summary[signal_type] = {
                        'exists': True,
                        'columns': len(df.columns),
                        'column_names': list(df.columns)
                    }
                except Exception as e:
                    summary[signal_type] = {
                        'exists': True,
                        'error': str(e)
                    }
            else:
                summary[signal_type] = {
                    'exists': False
                }

        return summary


def main():
    """Example usage of the data ingestion manager."""
    manager = DataIngestionManager()

    # Print data summary
    print("=== Consolidated CSV Data Summary ===")
    summary = manager.get_data_summary()

    for signal_type, info in summary.items():
        print(f"\n{signal_type.upper()}:")
        if info['exists']:
            if 'error' in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Columns: {info['columns']}")
                print(f"  Column names: {info['column_names'][:5]}..." if len(info['column_names']) > 5 else f"  Column names: {info['column_names']}")
        else:
            print("  File does not exist")

    print("\n=== Data Ingestion Example ===")
    print("To ingest new data, create a DataFrame with the required columns and call:")
    print("- manager.ingest_entry_signals(df)")
    print("- manager.ingest_exit_signals(df)")
    print("- manager.ingest_portfolio_target_signals(df)")
    print("- manager.ingest_breadth_signals(df)")


if __name__ == "__main__":
    main()