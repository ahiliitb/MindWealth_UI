"""
Page creation functions for the dashboard
"""

from .dashboard import create_top_signals_dashboard
from .analysis_page import create_analysis_page
from .performance_page import create_performance_summary_page
from .breadth_page import create_breadth_page
from .text_file_page import create_text_file_page

__all__ = [
    'create_top_signals_dashboard',
    'create_analysis_page',
    'create_performance_summary_page',
    'create_breadth_page',
    'create_text_file_page'
]

