"""
Text file page for displaying Claude output
"""

import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime

from constant import (
    GPT_SIGNALS_REPORT_TXT_PATH_US,
    GPT_SIGNALS_REPORT_CSV_PATH_US,
)
from ..components.cards import create_summary_cards, create_strategy_cards
from ..utils.file_discovery import extract_date_from_filename


def find_latest_gpt_file(base_path, extension='txt'):
    """
    Find the most recent Claude report file (dated or non-dated).
    Returns the file path and extracted date if found.
    """
    # Get directory and base filename
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    
    # Pattern to match: YYYY-MM-DD_claude_signals_report.ext or claude_signals_report.ext
    pattern_txt = os.path.join(dir_path, f"*_claude_signals_report.{extension}")
    pattern_base = os.path.join(dir_path, f"claude_signals_report.{extension}")
    
    # Find all matching files
    dated_files = glob.glob(pattern_txt)
    base_files = glob.glob(pattern_base)
    
    all_files = dated_files + base_files
    
    if not all_files:
        return None, None
    
    # Sort files by date (most recent first)
    def get_file_date(file_path):
        filename = os.path.basename(file_path)
        date_obj = extract_date_from_filename(filename)
        return date_obj if date_obj else datetime.min
    
    # Get the most recent file
    latest_file = max(all_files, key=get_file_date)
    file_date = extract_date_from_filename(os.path.basename(latest_file))
    
    return latest_file, file_date


def create_text_file_page():
    """Create a page to display Claude Signals report: text first, then cards + table"""
    # Display data fetch datetime
    from ..utils.helpers import display_data_fetch_info
    display_data_fetch_info(location="sidebar")
    
    st.title("🤖 Claude Signals Report")
    
    # Find the latest Claude files
    txt_file, txt_date = find_latest_gpt_file(GPT_SIGNALS_REPORT_TXT_PATH_US, 'txt')
    csv_file, csv_date = find_latest_gpt_file(GPT_SIGNALS_REPORT_CSV_PATH_US, 'csv')
    
    # Display date if found (prefer CSV date if both exist, otherwise use TXT date)
    report_date = csv_date if csv_date else txt_date
    if report_date:
        formatted_date = report_date.strftime('%B %d, %Y')
        st.markdown(f"**📅 Report Date: {formatted_date} at 5:00 PM EST**")
    
    st.markdown("---")
    
    # 1) Text output
    st.markdown("### 📝 Claude Analysis (Text)")
    txt_path = txt_file if txt_file else GPT_SIGNALS_REPORT_TXT_PATH_US
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            st.text_area("Report Text:", content, height=600, key="claude_signals_text")
    except FileNotFoundError:
        st.warning(f"Text report not found: {txt_path}")
    except Exception as e:
        st.error(f"Error reading text report: {str(e)}")
    
    st.markdown("---")
    
    # 2) CSV output: show strategy cards first, then table
    st.markdown("### 📊 Claude Signals (Cards + Table)")
    csv_path = csv_file if csv_file else GPT_SIGNALS_REPORT_CSV_PATH_US
    if os.path.exists(csv_path):
        try:
            # Read raw CSV first
            raw_df = pd.read_csv(csv_path)
            
            if raw_df.empty:
                st.info("No data available in Claude signals CSV.")
                return
            
            # Claude Signals CSV has the same structure as outstanding_signal.csv
            # Use parse_detailed_signal_csv parser directly
            from ..parsers.base_parsers import parse_detailed_signal_csv
            parsed_df = parse_detailed_signal_csv(raw_df)
            
            if parsed_df.empty:
                st.info("No data could be parsed from Claude signals CSV.")
                return
            
            # Strategy cards - use parsed data which has Raw_Data column
            # Summary cards if possible
            required_summary_cols = {'Win_Rate', 'Num_Trades', 'Strategy_CAGR', 'Strategy_Sharpe'}
            has_summary_cols = required_summary_cols.issubset(set(parsed_df.columns))
            
            if has_summary_cols:
                create_summary_cards(parsed_df)
                st.markdown("---")
            
            # Strategy cards with function-specific column logic
            create_strategy_cards(parsed_df, page_name="Claude Signals", tab_context="claude_signals")
            st.markdown("---")
            
            # Detail table - exclude function-specific columns (same logic as other pages)
            st.markdown("### 📋 Detailed Data Table (Original CSV Format)")
            
            # Columns to exclude from detail table (only show in strategy cards if not "No Information")
            columns_to_exclude = [
                'Sigmashell, Success Rate of Past Analysis [%]',
                'Divergence observed with, Signal Type',
                'Maxima Broken Date/Price[$]',
                'Track Level/Price($), Price on Latest Trading day vs Track Level, Signal Type',
                'Reference Upmove or Downmove start Date/Price($), end Date/Price($)',
                '% Change in Price on Latest Trading day vs Price on Trendpulse Breakout day/Earliest Unconfirmed Signal day/Confirmed Signal day'
            ]
            
            # Remove excluded columns if they exist
            columns_to_display = [col for col in raw_df.columns if col not in columns_to_exclude]
            filtered_raw_df = raw_df[columns_to_display]
            
            # Reorder columns: Symbol/Signal first, Exit Signal second, Function third
            from ..utils.helpers import reorder_dataframe_columns, find_column_by_keywords
            filtered_raw_df = reorder_dataframe_columns(filtered_raw_df)
            
            # Find Symbol and Exit Signal columns for pinning
            symbol_col = find_column_by_keywords(filtered_raw_df.columns, ['Symbol, Signal', 'Symbol'])
            if not symbol_col:
                for col in filtered_raw_df.columns:
                    if 'Symbol' in col and 'Signal' in col and 'Exit' not in col:
                        symbol_col = col
                        break
            exit_col = find_column_by_keywords(filtered_raw_df.columns, ['Exit Signal Date', 'Exit Signal', 'Exit'])
            
            # Display with better formatting, pinning, and autosize for ALL columns
            column_config = {}
            for col in filtered_raw_df.columns:
                # Pin Symbol and Exit Signal columns
                if col == symbol_col or col == exit_col:
                    column_config[col] = st.column_config.TextColumn(
                        col,
                        help=f"Original CSV column: {col}",
                        pinned="left"
                        # No width parameter = autosize
                    )
                else:
                    column_config[col] = st.column_config.TextColumn(
                        col,
                        help=f"Original CSV column: {col}"
                        # No width parameter = autosize
                    )
            
            st.dataframe(
                filtered_raw_df,
                use_container_width=True,
                height=600,
                column_config=column_config
            )
        except Exception as e:
            st.error(f"Error reading CSV report: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning(f"CSV report not found: {csv_path}")

