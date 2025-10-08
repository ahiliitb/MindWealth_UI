"""
Parsers for performance analysis CSV files
"""

from .base_parsers import parse_performance_csv


def parse_latest_performance(df):
    """Parse latest_performance.csv"""
    return parse_performance_csv(df)


def parse_forward_backtesting(df):
    """Parse forward_backtesting.csv"""
    return parse_performance_csv(df)

