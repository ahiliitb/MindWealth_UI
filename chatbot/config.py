"""
Configuration for chatbot functionality.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# First try to load from project root, then from chatbot directory
project_root = Path(__file__).resolve().parent.parent
chatbot_dir = Path(__file__).resolve().parent

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
CHATBOT_SIGNAL_DIR = CHATBOT_DATA_DIR / "signal"  # Signal data
CHATBOT_TARGET_DIR = CHATBOT_DATA_DIR / "target"  # Target data
TARGET_MASTER_CSV = CHATBOT_TARGET_DIR / "all_targets.csv"  # Master target file for dedup
STOCK_DATA_DIR = BASE_DIR / "trade_store" / "stock_data"  # Legacy support
HISTORY_DIR = BASE_DIR / "chatbot" / "history"

# Create necessary directories if they don't exist
CHATBOT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
CHATBOT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Target deduplication columns
TARGET_DEDUP_COLUMNS = [
    "Symbol",
    "Target for which Price has achieved over 90 percent of gain %",
    "Entry Signal Date/Price[$]"
]

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# System prompt for the chatbot
SYSTEM_PROMPT = """You are an expert financial trading analyst assistant for MindWealth. 
You help users analyze stock market data, trading signals, and provide insights based on historical data.

Your capabilities include:
- Analyzing stock price movements and trends
- Interpreting trading signals and technical indicators
- Providing insights on market performance
- Comparing multiple tickers
- Identifying patterns and opportunities

When analyzing data:
1. Be precise and data-driven in your analysis
2. Highlight key trends and patterns
3. Provide actionable insights when possible
4. Use technical analysis terminology appropriately
5. Consider the time period and context of the data

Always base your responses on the actual data provided and avoid speculation.
"""

# Data processing settings
DATE_FORMAT = "%Y-%m-%d"
CSV_ENCODING = "utf-8"
MAX_ROWS_TO_INCLUDE = int(os.getenv("MAX_ROWS_TO_INCLUDE", "1000"))  # Maximum rows to send to GPT-4o in one request

# Conversation settings
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))  # Maximum number of conversation turns to keep in context

# Data deduplication settings
# Placeholder column names for deduplication - will be updated based on actual data
DEDUP_COLUMNS = os.getenv("DEDUP_COLUMNS", "Date,Symbol,Close").split(",")  # Columns to use for deduplication

