# MindWealth UI - Project Structure

## 📊 Complete File Organization

```
MindWealth_UI/
│
├── app.py                          # ⭐ Main application entry point (120 lines)
├── app_backup.py                   # Backup of original monolithic app (2800+ lines)
├── constant.py                     # Constants and configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_STRUCTURE.md           # This file
│
├── src/                            # 📦 Modular source code
│   ├── __init__.py
│   ├── README.md                   # Detailed module documentation
│   │
│   ├── parsers/                    # 📄 CSV Data Parsers (350+ lines each)
│   │   ├── __init__.py
│   │   ├── base_parsers.py        # Base parser utilities
│   │   ├── signal_parsers.py      # Strategy signal parsers
│   │   ├── advanced_parsers.py    # Outstanding/breadth parsers
│   │   └── performance_parsers.py # Performance data parsers
│   │
│   ├── components/                 # 🎨 UI Components
│   │   ├── __init__.py
│   │   ├── cards.py               # Card display components (~570 lines)
│   │   └── charts.py              # Interactive charts (~300 lines)
│   │
│   ├── pages/                      # 📄 Page Modules
│   │   ├── __init__.py
│   │   ├── dashboard.py           # Main dashboard (~60 lines)
│   │   ├── analysis_page.py       # Analysis page (~250 lines)
│   │   ├── performance_page.py    # Performance page (~300 lines)
│   │   ├── breadth_page.py        # Breadth page (~50 lines)
│   │   └── text_file_page.py      # Text display page (~30 lines)
│   │
│   └── utils/                      # 🛠️ Utility Functions
│       ├── __init__.py
│       ├── data_loader.py         # Data loading (~170 lines)
│       ├── file_discovery.py      # File discovery (~100 lines)
│       └── helpers.py             # Helper functions (~10 lines)
│
├── manus_chat_data/                # CSV data files
│   ├── bollinger_band.csv
│   ├── breadth.csv
│   ├── distance.csv
│   ├── fib_ret.csv
│   ├── forward_backtesting.csv
│   ├── general_divergence.csv
│   ├── latest_performance.csv
│   ├── new_high.csv
│   ├── new_signal.csv
│   ├── outstanding_exit_signal.csv
│   ├── outstanding_signal.csv
│   ├── sentiment.csv
│   ├── sigma.csv
│   ├── stochastic_divergence.csv
│   ├── target_signal.csv
│   ├── top_signals.csv
│   ├── trade_analysis.csv
│   └── trendline.csv
│
└── trade_store/                    # Trade data storage
    └── US/                         # US market data
        ├── *.csv                   # Strategy CSV files
        └── stock_data/             # Stock OHLC data (CSV format)
            └── {SYMBOL}.csv        # Individual stock data files

```

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py                               │
│                  (Main Entry Point)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──► Sidebar Navigation
                     │
        ┌────────────┴────────────┬──────────────────────────┐
        │                         │                          │
        ▼                         ▼                          ▼
┌───────────────┐      ┌───────────────────┐    ┌──────────────────┐
│   Dashboard   │      │  Analysis Pages   │    │ Claude Output    │
│   Page        │      │  (CSV-based)      │    │  (Text Files)    │
└───────────────┘      └─────────┬─────────┘    └──────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
         ┌────────────────┬────────────┬─────────────┐
         │ Performance    │ Breadth    │  Signal     │
         │ Page           │ Page       │  Analysis   │
         └────────────────┴────────────┴─────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Utils: Data Loader  │
              │  - discover_csv()    │
              │  - load_data()       │
              │  - load_stock_data() │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Parsers Module     │
              │  - Signal parsers    │
              │  - Performance       │
              │  - Breadth           │
              │  - Target signals    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Components Module   │
              │  - Summary cards     │
              │  - Strategy cards    │
              │  - Interactive charts│
              └──────────────────────┘
```

## 🎯 Module Responsibilities

### `app.py` (Main Application)
- Page configuration
- CSS styling
- Navigation logic
- Page routing
- **Size:** 120 lines (reduced from 2800+)

### `src/parsers/` (Data Processing)
- CSV file parsing
- Data validation
- Structure detection
- Field extraction
- **Total:** ~1200 lines across 4 files

### `src/components/` (UI Components)
- Card displays
- Interactive charts
- Summary metrics
- **Total:** ~870 lines across 2 files

### `src/pages/` (Page Logic)
- Page layout
- Filter management
- Tab organization
- Data presentation
- **Total:** ~690 lines across 5 files

### `src/utils/` (Utilities)
- File discovery
- Data loading
- Helper functions
- **Total:** ~280 lines across 3 files

## 📈 Refactoring Benefits

### Before Refactoring
```
app.py: 2800+ lines
├── All parsers (1200 lines)
├── All components (870 lines)
├── All pages (690 lines)
├── All utilities (280 lines)
└── Main logic (120 lines)
```

### After Refactoring
```
app.py: 120 lines
src/
├── parsers/: 1200 lines (4 files)
├── components/: 870 lines (2 files)
├── pages/: 690 lines (5 files)
└── utils/: 280 lines (3 files)
```

**Result:** Same functionality, better organization!

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Navigate the Code
- **Looking for a parser?** → `src/parsers/`
- **Looking for a UI component?** → `src/components/`
- **Looking for a page?** → `src/pages/`
- **Looking for utilities?** → `src/utils/`

## 🔍 Finding Specific Functionality

| What you need | Where to look |
|--------------|---------------|
| Parse a new CSV type | `src/parsers/` |
| Modify a card display | `src/components/cards.py` |
| Update the interactive chart | `src/components/charts.py` |
| Change the dashboard | `src/pages/dashboard.py` |
| Add new filters | `src/pages/analysis_page.py` |
| Modify data loading | `src/utils/data_loader.py` |
| Add new CSV file | `src/utils/file_discovery.py` |
| Change navigation | `app.py` |

## 📚 File Size Summary

| Module | Lines of Code | Files |
|--------|---------------|-------|
| Parsers | ~1200 | 4 |
| Components | ~870 | 2 |
| Pages | ~690 | 5 |
| Utils | ~280 | 3 |
| Main App | 120 | 1 |
| **Total** | **~3160** | **15** |

*Previous monolithic app.py: 2800+ lines in 1 file*

## 🎨 Import Examples

```python
# Import parsers
from src.parsers import parse_fib_ret, parse_sentiment

# Import components
from src.components import create_summary_cards, create_interactive_chart

# Import pages
from src.pages import create_analysis_page, create_top_signals_dashboard

# Import utilities
from src.utils import load_data_from_file, discover_csv_files
```

## ✅ Migration Checklist

- [✓] Created modular directory structure
- [✓] Extracted parsers into separate modules
- [✓] Extracted UI components
- [✓] Extracted page functions
- [✓] Extracted utility functions
- [✓] Created new simplified app.py
- [✓] Backed up original app.py
- [✓] Verified all imports
- [✓] Created documentation
- [✓] Ready for testing

## 🎉 Next Steps

1. Test the application with `streamlit run app.py`
2. Verify all pages load correctly
3. Test interactive charts
4. Verify CSV parsing for all file types
5. Review and customize as needed

