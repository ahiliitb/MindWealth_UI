# MindWealth UI - Modular Code Structure

This directory contains the refactored, modular codebase for the MindWealth Trading Strategy Analysis application.

## 📁 Directory Structure

```
src/
├── __init__.py                 # Package initialization
├── parsers/                    # CSV data parsing modules
│   ├── __init__.py
│   ├── base_parsers.py        # Base parser functions (parse_signal_csv, parse_detailed_signal_csv, parse_performance_csv)
│   ├── signal_parsers.py      # Individual strategy parsers (Band Matrix, DeltaDrift, Fractal Track, etc.)
│   ├── advanced_parsers.py    # Advanced parsers (outstanding signals, breadth, target signals)
│   └── performance_parsers.py # Performance data parsers
│
├── components/                 # UI components
│   ├── __init__.py
│   ├── cards.py               # Card components for displaying strategy info
│   └── charts.py              # Interactive chart components (candlestick charts)
│
├── pages/                      # Page creation functions
│   ├── __init__.py
│   ├── dashboard.py           # Main dashboard page
│   ├── analysis_page.py       # General analysis page for CSV data
│   ├── performance_page.py    # Performance summary page
│   ├── breadth_page.py        # Breadth analysis page
│   └── text_file_page.py      # Text file display page (Claude output)
│
└── utils/                      # Utility functions
    ├── __init__.py
    ├── data_loader.py         # Data loading utilities (CSV and stock data)
    ├── file_discovery.py      # File discovery and CSV structure detection
    └── helpers.py             # Helper utility functions

```

## 🔧 Module Descriptions

### Parsers (`src/parsers/`)

**Purpose:** Parse different CSV file formats and convert them to standardized DataFrames.

#### `base_parsers.py`
- `parse_signal_csv(df, function_name)` - Generic parser for standard signal CSV files
- `parse_detailed_signal_csv(df)` - Parser for CSV files with Function column
- `parse_performance_csv(df)` - Parser for performance analysis CSV files

#### `signal_parsers.py`
- `parse_bollinger_band(df)` - Band Matrix strategy
- `parse_distance(df)` - DeltaDrift strategy
- `parse_fib_ret(df)` - Fractal Track strategy
- `parse_general_divergence(df)` - BaselineDiverge strategy
- `parse_new_high(df)` - Altitude Alpha strategy
- `parse_stochastic_divergence(df)` - Oscillator Delta strategy
- `parse_sigma(df)` - SigmaShell strategy
- `parse_sentiment(df)` - PulseGauge strategy (with special quoted field handling)
- `parse_trendline(df)` - TrendPulse strategy

#### `advanced_parsers.py`
- `parse_outstanding_signal(df)` - Outstanding signals
- `parse_outstanding_exit_signal(df)` - Outstanding exit signals
- `parse_new_signal(df)` - New signals
- `parse_target_signals(df, page_name)` - Target signals with detailed information
- `parse_breadth(df)` - Signal breadth indicator data

#### `performance_parsers.py`
- `parse_latest_performance(df)` - Latest performance data
- `parse_forward_backtesting(df)` - Forward backtesting performance data

### Components (`src/components/`)

**Purpose:** Reusable UI components for displaying data.

#### `cards.py`
- `create_summary_cards(df)` - Display summary metrics (win rate, trades, CAGR, Sharpe)
- `create_strategy_cards(df, page_name)` - Display strategy cards with pagination
- `display_strategy_cards_page(df, page_name)` - Display individual strategy card details
- `create_performance_summary_cards(df)` - Performance summary metrics
- `create_performance_cards(df)` - Performance analysis cards
- `display_performance_cards_page(df)` - Individual performance card details
- `create_breadth_summary_cards(df)` - Breadth analysis summary
- `create_breadth_cards(df)` - Individual breadth analysis cards

#### `charts.py`
- `create_interactive_chart(row_data, raw_data)` - Interactive candlestick chart with:
  - Real OHLC data from CSV files
  - Reference upmove/downmove lines
  - Buy/sell signal markers
  - Crosshair tracking
  - Full-width responsive design

### Pages (`src/pages/`)

**Purpose:** Page-level components that organize and display complete views.

#### `dashboard.py`
- `create_top_signals_dashboard()` - Main dashboard with strategy overview

#### `analysis_page.py`
- `create_analysis_page(data_file, page_title)` - General analysis page with:
  - Signal type tabs (All/Long/Short)
  - Interval tabs (Daily/Weekly/Monthly/Quarterly/Yearly)
  - Filters (functions, symbols, win rate)
  - Strategy cards and data tables

#### `performance_page.py`
- `create_performance_summary_page(data_file, page_title)` - Performance metrics page with:
  - Strategy filters
  - Win rate filters
  - Performance cards
  - Original CSV data tables

#### `breadth_page.py`
- `create_breadth_page(data_file, page_title)` - Breadth analysis page with:
  - Breadth summary cards
  - Individual strategy breadth analysis
  - Bullish asset and signal percentages

#### `text_file_page.py`
- `create_text_file_page()` - Display Claude output text files

### Utils (`src/utils/`)

**Purpose:** Utility functions for data loading, file discovery, and helpers.

#### `data_loader.py`
- `load_data_from_file(file_path, page_name)` - Load and parse CSV files with caching
- `load_stock_data_file(symbol, start_date, end_date, interval)` - Load stock data from CSV with interval conversion

#### `file_discovery.py`
- `discover_csv_files()` - Discover all CSV files in trade_store/US directory
- `detect_csv_structure(file_path)` - Detect CSV file type based on filename

#### `helpers.py`
- `find_column_by_keywords(columns, keywords)` - Find column by keyword matching

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

### Importing Modules

```python
# Import parsers
from src.parsers import parse_bollinger_band, parse_distance

# Import components
from src.components import create_summary_cards, create_interactive_chart

# Import pages
from src.pages import create_top_signals_dashboard, create_analysis_page

# Import utilities
from src.utils import load_data_from_file, discover_csv_files
```

## 📝 Adding New Functionality

### Adding a New CSV Parser

1. Determine if it fits an existing pattern (signal, performance, or breadth)
2. Add parser function to appropriate file in `src/parsers/`
3. Register the parser in `src/parsers/__init__.py`
4. Add file mapping in `src/utils/file_discovery.py`

### Adding a New UI Component

1. Create component function in `src/components/cards.py` or create new file
2. Export function in `src/components/__init__.py`
3. Use component in relevant page

### Adding a New Page

1. Create page function in `src/pages/` directory
2. Export function in `src/pages/__init__.py`
3. Import and use in `app.py`

## 🔄 Migration Notes

- **Original file:** `app.py` (2800+ lines) → backed up as `app_backup.py`
- **New file:** `app.py` (120 lines) → imports from modular structure
- **All functionality preserved:** No features were removed during refactoring
- **Backward compatible:** Uses same `constant.py` and data files

## 🧪 Testing

To verify all modules load correctly:

```bash
cd /Users/ahilkhaniitb/work/mindwealth/MindWealth_UI
streamlit run app.py
```

## 📦 Benefits of Modular Structure

1. **Better Organization:** Related code grouped together
2. **Easier Navigation:** Find specific functionality quickly
3. **Reusability:** Components can be reused across pages
4. **Maintainability:** Changes isolated to specific modules
5. **Testability:** Individual modules can be tested independently
6. **Scalability:** Easy to add new strategies and pages
7. **Team Collaboration:** Multiple developers can work on different modules
8. **Code Clarity:** Smaller, focused files are easier to understand

