"""
Update Signal Open Price column in consolidated CSV files with real prices from local stock data.
Fetches open prices based on the interval (Daily/Weekly/Monthly/Quarterly) for each signal.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalOpenPriceUpdater:
    """
    Updates Signal Open Price column with real prices from local stock data based on interval.
    """

    def __init__(self, stock_data_dir: Path = None):
        """Initialize the updater."""
        if stock_data_dir is None:
            # Default to trade_store/stock_data relative to chatbot directory
            stock_data_dir = Path(__file__).resolve().parent.parent / "trade_store" / "stock_data"
        self.stock_data_dir = Path(stock_data_dir)
        self.price_cache = {}  # Cache to avoid repeated file reads

    def extract_signal_date(self, signal_text: str) -> Optional[str]:
        """
        Extract signal date from the signal text.
        Format: "SYMBOL, Long/Short, YYYY-MM-DD (Price: XX.XX)"
        """
        try:
            # Find the date pattern YYYY-MM-DD
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', str(signal_text))
            if date_match:
                return date_match.group(1)
        except Exception as e:
            logger.warning(f"Could not extract date from: {signal_text} - {e}")
        return None

    def extract_interval(self, interval_text: str) -> str:
        """
        Extract interval from the interval text.
        Format: "Daily, is CONFIRMED..." or "Monthly, is CONFIRMED..."
        """
        interval_str = str(interval_text).lower()
        if 'daily' in interval_str:
            return 'Daily'
        elif 'weekly' in interval_str:
            return 'Weekly'
        elif 'monthly' in interval_str:
            return 'Monthly'
        elif 'quarterly' in interval_str:
            return 'Quarterly'
        else:
            return 'Daily'  # Default fallback

    def get_open_price(self, symbol: str, signal_date: str, interval: str) -> Optional[float]:
        """
        Get the open price for a symbol at the given date and interval from local stock data.
        """
        cache_key = f"{symbol}_{signal_date}_{interval}"

        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        try:
            # Parse the signal date
            signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')

            # Load the stock data CSV
            stock_file = self.stock_data_dir / f"{symbol}.csv"
            if not stock_file.exists():
                logger.warning(f"Stock data file not found: {stock_file}")
                return None

            # Read the stock data
            stock_data = pd.read_csv(stock_file)
            if stock_data.empty:
                logger.warning(f"Empty stock data file: {stock_file}")
                return None

            # Ensure Date column is datetime
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.sort_values('Date')

            # Filter data based on interval
            if interval == 'Daily':
                # Find exact date match
                date_matches = stock_data[stock_data['Date'].dt.date == signal_dt.date()]
                if not date_matches.empty:
                    open_price = date_matches.iloc[0]['Open']
                    self.price_cache[cache_key] = open_price
                    return open_price

            elif interval == 'Weekly':
                # Find the week containing the signal date
                week_start = signal_dt - timedelta(days=signal_dt.weekday())  # Monday of the week
                week_end = week_start + timedelta(days=6)  # Sunday of the week

                week_data = stock_data[
                    (stock_data['Date'].dt.date >= week_start.date()) &
                    (stock_data['Date'].dt.date <= week_end.date())
                ]

                if not week_data.empty:
                    # Use the Open price of the first trading day in the week
                    open_price = week_data.iloc[0]['Open']
                    self.price_cache[cache_key] = open_price
                    return open_price

            elif interval == 'Monthly':
                # Find data for the target month
                target_year = signal_dt.year
                target_month = signal_dt.month

                month_data = stock_data[
                    (stock_data['Date'].dt.year == target_year) &
                    (stock_data['Date'].dt.month == target_month)
                ]

                if not month_data.empty:
                    # Use the Open price of the first trading day in the month
                    open_price = month_data.iloc[0]['Open']
                    self.price_cache[cache_key] = open_price
                    return open_price

            elif interval == 'Quarterly':
                # Find data for the target quarter
                target_year = signal_dt.year
                target_quarter = (signal_dt.month - 1) // 3 + 1

                quarter_data = stock_data[
                    (stock_data['Date'].dt.year == target_year) &
                    (((stock_data['Date'].dt.month - 1) // 3 + 1) == target_quarter)
                ]

                if not quarter_data.empty:
                    # Use the Open price of the first trading day in the quarter
                    open_price = quarter_data.iloc[0]['Open']
                    self.price_cache[cache_key] = open_price
                    return open_price

            logger.warning(f"Could not find {interval} open price for {symbol} on {signal_date}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol} {signal_date} {interval}: {e}")
            return None

    def update_csv_file(self, csv_path: Path, max_updates: int = None) -> int:
        """
        Update the Signal Open Price column in a CSV file.
        """
        if not csv_path.exists():
            logger.warning(f"CSV file does not exist: {csv_path}")
            return 0

        logger.info(f"📁 Processing {csv_path.name}...")

        try:
            # Read the CSV
            df = pd.read_csv(csv_path, encoding='utf-8')
            original_count = len(df)

            if df.empty:
                logger.warning(f"CSV file is empty: {csv_path}")
                return 0

            # Check if Signal Open Price column exists
            if 'Signal Open Price' not in df.columns:
                logger.warning(f"'Signal Open Price' column not found in {csv_path}")
                return 0

            updated_count = 0

            # Process each row
            for idx, row in df.iterrows():
                if max_updates and updated_count >= max_updates:
                    break

                try:
                    # Extract symbol from the signal text
                    signal_text = str(row.get('Symbol, Signal, Signal Date/Price[$]', ''))
                    if not signal_text or ',' not in signal_text:
                        continue

                    # Extract symbol (first part before comma)
                    symbol = signal_text.split(',')[0].strip()
                    if not symbol:
                        continue

                    # Extract signal date
                    signal_date = self.extract_signal_date(signal_text)
                    if not signal_date:
                        continue

                    # Extract interval
                    interval_text = str(row.get('Interval, Confirmation Status', ''))
                    interval = self.extract_interval(interval_text)

                    # Get real open price
                    real_open_price = self.get_open_price(symbol, signal_date, interval)

                    if real_open_price is not None:
                        # Update the Signal Open Price column
                        df.at[idx, 'Signal Open Price'] = real_open_price
                        updated_count += 1

                        if updated_count % 50 == 0:
                            logger.info(f"✅ Updated {updated_count} rows in {csv_path.name}...")

                    # Small delay to avoid rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {csv_path.name}: {e}")
                    continue

            # Save the updated CSV
            if updated_count > 0:
                df.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"💾 Saved {csv_path.name} with {updated_count} updated prices")

            return updated_count

        except Exception as e:
            logger.error(f"Error processing {csv_path}: {e}")
            return 0

    def update_all_csvs(self, data_dir: Path, max_updates_per_file: int = None, start_file: str = None) -> Dict[str, int]:
        """
        Update all consolidated CSV files.
        """
        results = {}

        csv_files = [
            ('entry', data_dir / 'entry.csv'),
            ('exit', data_dir / 'exit.csv'),
            ('portfolio_target_achieved', data_dir / 'portfolio_target_achieved.csv')
        ]

        start_processing = start_file is None
        for name, csv_path in csv_files:
            if not start_processing:
                if name == start_file:
                    start_processing = True
                else:
                    continue

            logger.info(f"\n🔄 Updating {name}.csv...")
            updated_count = self.update_csv_file(csv_path, max_updates_per_file)
            results[name] = updated_count

            # Clear cache between files to save memory
            self.price_cache.clear()

        return results


def main():
    """Main function to update signal open prices."""
    logger.info("🚀 Starting Signal Open Price Updates")
    logger.info("=" * 60)

    updater = SignalOpenPriceUpdater()
    data_dir = Path("data")

    # Update all CSV files
    results = updater.update_all_csvs(data_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 UPDATE SUMMARY:")
    total_updated = 0
    for file_type, count in results.items():
        logger.info(f"  {file_type}.csv: {count} prices updated")
        total_updated += count

    logger.info(f"\n✅ Total: {total_updated} signal open prices updated with real Yahoo Finance data")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()