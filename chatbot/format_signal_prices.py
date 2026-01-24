"""
Format Signal Open Price column to show only 4 significant digits after decimal.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_price_column(df, column_name):
    """
    Format the price column to show exactly 4 decimal places as strings.
    """
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found")
        return df

    def format_price(price):
        try:
            # Convert to float and format to exactly 4 decimal places as string
            if pd.isna(price) or price == '':
                return price
            formatted_price = f"{float(price):.4f}"
            return formatted_price
        except (ValueError, TypeError):
            logger.warning(f"Could not format price: {price}")
            return price

    df[column_name] = df[column_name].apply(format_price)
    return df


def update_csv_file(csv_path):
    """
    Update the Signal Open Price column in a CSV file to 4 decimal places.
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

        # Format the prices
        df = format_price_column(df, 'Signal Open Price')

        # Save the updated CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"💾 Saved {csv_path.name} with formatted prices")

        return original_count

    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        return 0


def update_all_csvs(data_dir: Path):
    """
    Update all consolidated CSV files.
    """
    results = {}

    csv_files = [
        ('entry', data_dir / 'entry.csv'),
        ('exit', data_dir / 'exit.csv'),
        ('portfolio_target_achieved', data_dir / 'portfolio_target_achieved.csv')
    ]

    for name, csv_path in csv_files:
        logger.info(f"\n🔄 Updating {name}.csv...")
        count = update_csv_file(csv_path)
        results[name] = count

    return results


def main():
    """Main function to format signal prices."""
    logger.info("🚀 Starting Signal Price Formatting")
    logger.info("=" * 50)

    data_dir = Path("data")

    # Update all CSV files
    results = update_all_csvs(data_dir)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 FORMATTING SUMMARY:")
    total_processed = 0
    for file_type, count in results.items():
        logger.info(f"  {file_type}.csv: {count} records processed")
        total_processed += count

    logger.info(f"\n✅ Total: {total_processed} signal prices formatted to 4 decimal places")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()