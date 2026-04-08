"""
Parsers for performance analysis CSV files
"""

from .base_parsers import parse_performance_csv


def parse_combined_performance_report(df):
    """Parse combined_performance_report.csv"""
    return parse_performance_csv(df)

