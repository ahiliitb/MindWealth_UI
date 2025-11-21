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
            raw_df = pd.read_csv(csv_path)
            if raw_df.empty:
                st.info("No data available in Claude signals CSV.")
                return
            
            # Try to use the same card components as other pages if expected columns exist
            required_summary_cols = {'Win_Rate', 'Num_Trades', 'Strategy_CAGR', 'Strategy_Sharpe'}
            required_card_cols = {'Function', 'Symbol'}
            has_summary_cols = required_summary_cols.issubset(set(raw_df.columns))
            has_card_base = required_card_cols.issubset(set(raw_df.columns))
            
            df_for_cards = None
            if has_card_base:
                df_for_cards = raw_df.copy()
                # Ensure columns expected by cards exist; if missing, add safe defaults
                for col, default in [
                    ('Win_Rate', 0.0),
                    ('Num_Trades', 0),
                    ('Strategy_CAGR', 0.0),
                    ('Strategy_Sharpe', 0.0),
                    ('Interval', 'Unknown'),
                    ('Signal_Type', 'Unknown'),
                    ('Signal_Date', 'Unknown'),
                ]:
                    if col not in df_for_cards.columns:
                        df_for_cards[col] = default
                
                # Provide a minimal Raw_Data dict per row to satisfy cards rendering fallbacks
                if 'Raw_Data' not in df_for_cards.columns:
                    df_for_cards['Raw_Data'] = [{} for _ in range(len(df_for_cards))]
            
            # Strategy cards
            if df_for_cards is not None:
                # Summary cards if possible
                if has_summary_cols:
                    create_summary_cards(df_for_cards)
                    st.markdown("---")
                
                create_strategy_cards(df_for_cards, page_name="Claude Signals", tab_context="claude_signals")
                st.markdown("---")
            else:
                st.info("Columns required for strategy cards not found. Showing table only.")
            
            # Raw table (same style as other pages' original CSV table)
            st.markdown("### 📋 Detailed Data Table (Original CSV)")
            st.dataframe(
                raw_df,
                use_container_width=True,
                height=600,
            )
        except Exception as e:
            st.error(f"Error reading CSV report: {str(e)}")
    else:
        st.warning(f"CSV report not found: {csv_path}")

