"""
Text file page for displaying Claude output
"""

import streamlit as st

from constant import CLAUDE_OUTPUT_TXT_PATH, BOX_CLAUDE_OUTPUT_TXT_PATH


def create_text_file_page():
    """Create a page to display text files with tabs"""
    st.title("📄 Claude Output")
    st.markdown("---")
    
    # Create two tabs for the text files
    tab1, tab2 = st.tabs(["Claude Output", "GPT Output"])
    
    # Claude Output tab
    with tab1:
        st.markdown("### 📝 Claude Output")
        try:
            with open(CLAUDE_OUTPUT_TXT_PATH, 'r', encoding='utf-8') as file:
                content = file.read()
                st.text_area("File Content:", content, height=600, key="claude_output")
        except FileNotFoundError:
            st.error(f"File not found: {CLAUDE_OUTPUT_TXT_PATH}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Box Claude Output tab
    with tab2:
        st.markdown("### 🤖 GPT Output")
        try:
            with open(BOX_CLAUDE_OUTPUT_TXT_PATH, 'r', encoding='utf-8') as file:
                content = file.read()
                st.text_area("File Content:", content, height=600, key="box_claude_output")
        except FileNotFoundError:
            st.error(f"File not found: {BOX_CLAUDE_OUTPUT_TXT_PATH}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

