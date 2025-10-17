"""
Text file page for displaying GPT output
"""

import streamlit as st

from constant import BOX_CLAUDE_OUTPUT_TXT_PATH


def create_text_file_page():
    """Create a page to display GPT output"""
    st.title("🤖 AI Output")
    st.markdown("---")
    
    # Display GPT Output directly without tabs
    st.markdown("### 🤖 GPT Output")
    try:
        with open(BOX_CLAUDE_OUTPUT_TXT_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
            st.text_area("File Content:", content, height=600, key="gpt_output")
    except FileNotFoundError:
        st.error(f"File not found: {BOX_CLAUDE_OUTPUT_TXT_PATH}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

