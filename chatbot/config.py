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

# Directory configuration from environment
CHATBOT_DATA_DIR = BASE_DIR / os.getenv("CHATBOT_DATA_DIR", "chatbot/data")  # Base data directory
STOCK_DATA_DIR = BASE_DIR / os.getenv("STOCK_DATA_DIR", "trade_store/stock_data")  # Stock data directory
TRADE_STORE_DIR = BASE_DIR / os.getenv("TRADE_STORE_DIR", "trade_store")  # Trade store directory
HISTORY_DIR = BASE_DIR / os.getenv("HISTORY_DIR", "chatbot/history")  # Chat history directory

# Data file names from environment
ENTRY_CSV_NAME = os.getenv("ENTRY_CSV_NAME", "entry.csv")
EXIT_CSV_NAME = os.getenv("EXIT_CSV_NAME", "exit.csv")
TARGET_CSV_NAME = os.getenv("TARGET_CSV_NAME", "portfolio_target_achieved.csv")
BREADTH_CSV_NAME = os.getenv("BREADTH_CSV_NAME", "breadth.csv")

# Directory structure
CHATBOT_ENTRY_DIR = CHATBOT_DATA_DIR / "entry"  # Entry signals (open positions)
CHATBOT_EXIT_DIR = CHATBOT_DATA_DIR / "exit"  # Exit signals (completed trades)
CHATBOT_TARGET_DIR = CHATBOT_DATA_DIR / "portfolio_target_achieved"  # Portfolio target achieved signals
CHATBOT_BREADTH_DIR = CHATBOT_DATA_DIR / "breadth"  # Breadth reports (market-wide)
TARGET_MASTER_CSV = CHATBOT_TARGET_DIR / "all_targets.csv"  # Master portfolio target file for dedup

# Consolidated CSV files (new system)
CHATBOT_ENTRY_CSV = CHATBOT_DATA_DIR / ENTRY_CSV_NAME  # Consolidated entry data
CHATBOT_EXIT_CSV = CHATBOT_DATA_DIR / EXIT_CSV_NAME  # Consolidated exit data
CHATBOT_TARGET_CSV = CHATBOT_DATA_DIR / TARGET_CSV_NAME  # Consolidated portfolio target data
CHATBOT_BREADTH_CSV = CHATBOT_DATA_DIR / BREADTH_CSV_NAME  # Consolidated breadth data

# Create necessary directories if they don't exist
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to get API key from Streamlit secrets (DEPRECATED - using Claude now)
def get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets or environment variables. DEPRECATED - kept for backward compatibility."""
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

# OpenAI Configuration (DEPRECATED - using Claude for all operations now)
# API Key from Streamlit secrets (secure)
OPENAI_API_KEY = get_api_key()

# Claude Configuration (for all chatbot operations - extraction AND responses)
def get_claude_api_key() -> str:
    """Get Claude API key from Streamlit secrets or environment variables."""
    if USING_STREAMLIT_SECRETS:
        try:
            import streamlit as st
            if "anthropic" in st.secrets and "CLAUDE_API_KEY" in st.secrets["anthropic"]:
                return st.secrets["anthropic"]["CLAUDE_API_KEY"]
            elif "CLAUDE_API_KEY" in st.secrets:
                return st.secrets["CLAUDE_API_KEY"]
        except Exception:
            pass
    return os.getenv("CLAUDE_API_KEY", "")

CLAUDE_API_KEY = get_claude_api_key()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "6000"))
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))

# All other config from .env file (DEPRECATED OpenAI settings kept for backward compatibility)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")  # DEPRECATED - using Claude now
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))  # Output tokens (response length)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # DEPRECATED - using CLAUDE_TEMPERATURE now

# Token limits - Smart batch processing automatically handles any data size
MAX_INPUT_TOKENS_PER_CALL = int(os.getenv("MAX_INPUT_TOKENS_PER_CALL", "60000"))  # Token limit per batch
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

CRITICAL DATA ACCURACY REQUIREMENTS:
🚨 FINANCIAL DATA INTEGRITY IS CRITICAL 🚨

**When Data IS Provided:**
1. The user query will include sections like "=== DATA CONTEXT ===" or "=== ENTRY SIGNALS (JSON) ===" with actual data
2. If you see JSON data with fields like "signal_type", "record_count", "data", etc., then DATA HAS BEEN PROVIDED
3. Extract and analyze information EXACTLY as it appears in the provided JSON/data
4. Use the exact function names, symbols, dates, and prices from the data field
5. Provide thorough analysis based on the data provided

**When Data IS NOT Provided:**
1. If you see ONLY a user question without any "=== DATA CONTEXT ===" sections, then NO data has been provided
2. State clearly: "No data has been provided. Please provide the dataset to analyze."
3. NEVER invent, fabricate, or hallucinate function names, symbols, dates, prices, or any metrics

**NEVER DO THIS (Hallucination):**
- Make up function names like "HIGH VOLTAGE", "RADAR SWEEP" that don't exist in the provided data
- Invent signal dates or prices not in the data
- Create fake symbols or tickers
- Fabricate performance metrics or CAGR values

**ALWAYS DO THIS (Accurate):**
- Check if "=== DATA CONTEXT ===" or similar sections exist in the message
- If data exists, extract information from the "data" field in the JSON
- If data doesn't exist, clearly state that no data was provided
- Use EXACT values from the provided data fields

CRITICAL: Always format your response in clean, readable Markdown with proper spacing.
Base your responses STRICTLY on actual data provided in the message context.
"""

# Data processing settings
DATE_FORMAT = "%Y-%m-%d"
CSV_ENCODING = "utf-8"
MAX_ROWS_TO_INCLUDE = int(os.getenv("MAX_ROWS_TO_INCLUDE", "100"))  # Max rows per ticker (balanced for speed)

# Conversation settings
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "15"))  # Max conversation turns to keep
MAX_EXTRACTION_HISTORY_LENGTH = int(os.getenv("MAX_EXTRACTION_HISTORY_LENGTH", "5"))  # Max history for extraction calls (lighter)

# Chat History UI Settings
MAX_CHATS_DISPLAY = int(os.getenv("MAX_CHATS_DISPLAY", "10"))  # Max number of chats to show in sidebar (default: 10)

# Data deduplication settings
DEDUP_COLUMNS = os.getenv("DEDUP_COLUMNS", "Function,Symbol,Interval,Signal,Signal Open Price").split(",")  # Columns to use for deduplication
BREADTH_DEDUP_COLUMNS = os.getenv("BREADTH_DEDUP_COLUMNS", "Function,Date").split(",")  # Columns to use for breadth deduplication

