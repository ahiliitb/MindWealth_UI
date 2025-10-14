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
CHATBOT_BREADTH_DIR = CHATBOT_DATA_DIR / "breadth"  # Breadth reports (market-wide)
TARGET_MASTER_CSV = CHATBOT_TARGET_DIR / "all_targets.csv"  # Master target file for dedup
STOCK_DATA_DIR = BASE_DIR / "trade_store" / "stock_data"  # Legacy support
HISTORY_DIR = BASE_DIR / "chatbot" / "history"

# Create necessary directories if they don't exist
CHATBOT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
CHATBOT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
CHATBOT_BREADTH_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Target deduplication columns
TARGET_DEDUP_COLUMNS = [
    "Symbol",
    "Target for which Price has achieved over 90 percent of gain %",
    "Entry Signal Date/Price[$]"
]

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")  # GPT-4 Turbo has 128K context window
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))  # Output tokens (response length)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Token limits - Optimized for SPEED (single calls by default)
# Org limited to 30K TPM - we stay under this for fast responses
MAX_INPUT_TOKENS_PER_CALL = int(os.getenv("MAX_INPUT_TOKENS_PER_CALL", "22000"))  # Safe single-call limit
MAX_TICKERS_PER_QUERY = int(os.getenv("MAX_TICKERS_PER_QUERY", "15"))  # Limit tickers for fast response
MAX_SEQUENTIAL_BATCHES = int(os.getenv("MAX_SEQUENTIAL_BATCHES", "5"))  # Reduced batches (only for large queries)
BATCH_DELAY_SECONDS = float(os.getenv("BATCH_DELAY_SECONDS", "65.0"))  # Delay between batches
ESTIMATED_CHARS_PER_TOKEN = 4  # Rough estimate: 1 token ≈ 4 characters
MIN_HISTORY_MESSAGES = 2  # Minimum messages to keep
ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "false").lower() == "true"  # DISABLED by default for speed

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

# Data deduplication settings
# Placeholder column names for deduplication - will be updated based on actual data
DEDUP_COLUMNS = os.getenv("DEDUP_COLUMNS", "Date,Symbol,Close").split(",")  # Columns to use for deduplication

