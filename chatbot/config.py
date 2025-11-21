"""
Configuration for chatbot functionality.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Try loading from Streamlit secrets first (for deployed apps)
try:
    import streamlit as st
    USING_STREAMLIT_SECRETS = hasattr(st, 'secrets') and len(st.secrets) > 0
except (ImportError, AttributeError, FileNotFoundError):
    USING_STREAMLIT_SECRETS = False

# Load environment variables from .env file (fallback if not using Streamlit secrets)
project_root = Path(__file__).resolve().parent.parent
chatbot_dir = Path(__file__).resolve().parent

if not USING_STREAMLIT_SECRETS:
    # Try loading from project root first
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try loading from chatbot directory
        env_file = chatbot_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)

# Base directories
BASE_DIR = project_root
CHATBOT_DATA_DIR = BASE_DIR / "chatbot" / "data"
CHATBOT_ENTRY_DIR = CHATBOT_DATA_DIR / "entry"  # Entry signals (no exit yet)
CHATBOT_EXIT_DIR = CHATBOT_DATA_DIR / "exit"  # Exit signals (completed trades)
CHATBOT_TARGET_DIR = CHATBOT_DATA_DIR / "portfolio_target_achieved"  # Portfolio target achieved signals
CHATBOT_BREADTH_DIR = CHATBOT_DATA_DIR / "breadth"  # Breadth reports (market-wide)
TARGET_MASTER_CSV = CHATBOT_TARGET_DIR / "all_targets.csv"  # Master portfolio target file for dedup
STOCK_DATA_DIR = BASE_DIR / "trade_store" / "stock_data"
HISTORY_DIR = BASE_DIR / "chatbot" / "history"

# Create necessary directories if they don't exist
CHATBOT_ENTRY_DIR.mkdir(parents=True, exist_ok=True)
CHATBOT_EXIT_DIR.mkdir(parents=True, exist_ok=True)
CHATBOT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
CHATBOT_BREADTH_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Target deduplication columns
TARGET_DEDUP_COLUMNS = [
    "Symbol",
    "Target for which Price has achieved over 90 percent of gain %",
    "Entry Signal Date/Price[$]"
]

# Helper function to get API key from Streamlit secrets
def get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets or environment variables."""
    if USING_STREAMLIT_SECRETS:
        try:
            import streamlit as st
            # Try openai section first, then root level
            if "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
                return st.secrets["openai"]["OPENAI_API_KEY"]
            elif "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY", "")

# OpenAI Configuration
# API Key from Streamlit secrets (secure)
OPENAI_API_KEY = get_api_key()

# All other config from .env file
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to GPT-4o
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))  # Output tokens (response length)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Token limits - Smart batch processing automatically handles any data size
MAX_INPUT_TOKENS_PER_CALL = int(os.getenv("MAX_INPUT_TOKENS_PER_CALL", "22000"))  # Token limit per batch
MAX_SEQUENTIAL_BATCHES = int(os.getenv("MAX_SEQUENTIAL_BATCHES", "999"))  # NO LIMIT - Process as many batches as needed
BATCH_DELAY_SECONDS = float(os.getenv("BATCH_DELAY_SECONDS", "5.0"))  # Delay between batches to avoid rate limits
ESTIMATED_CHARS_PER_TOKEN = 4  # Rough estimate: 1 token ≈ 4 characters
MIN_HISTORY_MESSAGES = 2  # Minimum messages to keep in history
ENABLE_BATCH_PROCESSING = True  # Smart batch processing always enabled for optimal performance

# Smart Filtering Settings
# When smart filtering is enabled, the system automatically determines which tickers to process:
# - If function(s) specified: ALL tickers with that function (no limit)
# - If ticker(s) specified: Only those tickers
# - If both specified: Intersection of tickers that have the function
# - Batch processing automatically handles any number of tickers efficiently

# System prompt for the chatbot
SYSTEM_PROMPT = """You are an expert financial trading analyst assistant for MindWealth. 
You help users analyze stock market data, trading signals, and provide insights based on historical data.

Your capabilities include:
- Analyzing stock price movements and trends
- Interpreting trading signals and technical indicators
- Providing insights on market performance
- Comparing multiple tickers
- Identifying patterns and opportunities

IMPORTANT OUTPUT FORMATTING REQUIREMENTS:
1. ALWAYS use proper Markdown formatting
2. ALWAYS include spaces between words and punctuation
3. Use bullet points (- or •) for lists
4. Use **bold** for emphasis
5. Use headers (##, ###) to organize sections
6. Use line breaks between paragraphs
7. Format numbers with proper spacing: "245.27 is significantly above the track level (169.28)"
8. NEVER concatenate words without spaces

When analyzing data:
1. Be precise and data-driven in your analysis
2. Highlight key trends and patterns
3. Provide actionable insights when possible
4. Use technical analysis terminology appropriately
5. Consider the time period and context of the data
6. Structure your response with clear sections and proper spacing

CRITICAL: Always format your response in clean, readable Markdown with proper spacing.
Always base your responses on the actual data provided and avoid speculation.
"""

# Data processing settings
DATE_FORMAT = "%Y-%m-%d"
CSV_ENCODING = "utf-8"
MAX_ROWS_TO_INCLUDE = int(os.getenv("MAX_ROWS_TO_INCLUDE", "100"))  # Max rows per ticker (balanced for speed)

# Conversation settings
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))  # Max conversation turns to keep
FOLLOWUP_HISTORY_LENGTH = int(os.getenv("FOLLOWUP_HISTORY_LENGTH", "3"))  # Number of previous exchanges to include in follow-up context

# Chat History UI Settings
MAX_CHATS_DISPLAY = int(os.getenv("MAX_CHATS_DISPLAY", "10"))  # Max number of chats to show in sidebar (default: 10)

# Data deduplication settings
# Placeholder column names for deduplication - will be updated based on actual data
DEDUP_COLUMNS = os.getenv("DEDUP_COLUMNS", "Date,Symbol,Interval,Signal").split(",")  # Columns to use for deduplication

