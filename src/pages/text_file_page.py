"""
Text file page for displaying GPT output
"""

import streamlit as st
import pandas as pd
import os

from constant import (
    GPT_SIGNALS_REPORT_TXT_PATH_US,
    GPT_SIGNALS_REPORT_CSV_PATH_US,
)
from ..components.cards import create_summary_cards, create_strategy_cards


def create_text_file_page():
    """Create a page to display GPT Signals report: text first, then cards + table"""
    st.title("🤖 GPT Signals Report")
    st.markdown("---")
    
    # 1) Text output
    st.markdown("### 📝 GPT Analysis (Text)")
    try:
        with open(GPT_SIGNALS_REPORT_TXT_PATH_US, 'r', encoding='utf-8') as file:
            content = file.read()
            st.text_area("Report Text:", content, height=400, key="gpt_signals_text")
    except FileNotFoundError:
        st.warning(f"Text report not found: {GPT_SIGNALS_REPORT_TXT_PATH_US}")
    except Exception as e:
        st.error(f"Error reading text report: {str(e)}")
    
    st.markdown("---")
    
    # 2) CSV output: show strategy cards first, then table
    st.markdown("### 📊 GPT Signals (Cards + Table)")
    if os.path.exists(GPT_SIGNALS_REPORT_CSV_PATH_US):
        try:
            raw_df = pd.read_csv(GPT_SIGNALS_REPORT_CSV_PATH_US)
            if raw_df.empty:
                st.info("No data available in GPT signals CSV.")
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
                
                create_strategy_cards(df_for_cards, page_name="GPT Signals", tab_context="gpt_signals")
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
        st.warning(f"CSV report not found: {GPT_SIGNALS_REPORT_CSV_PATH_US}")

