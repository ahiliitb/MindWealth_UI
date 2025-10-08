# 🚀 Quick Start Guide

## Running the Application

```bash
streamlit run app.py
```

## 📁 Finding What You Need

### I want to...

#### Add a new CSV parser
1. Open `src/parsers/signal_parsers.py`
2. Add your parser function
3. Register it in `src/parsers/__init__.py`
4. Add file mapping in `src/utils/file_discovery.py`

#### Modify a card display
1. Open `src/components/cards.py`
2. Find the card function (e.g., `create_summary_cards`)
3. Make your changes

#### Update the interactive chart
1. Open `src/components/charts.py`
2. Modify `create_interactive_chart` function

#### Add a new page
1. Create file in `src/pages/` (e.g., `my_page.py`)
2. Export in `src/pages/__init__.py`
3. Import and add route in `app.py`

#### Change data loading
1. Open `src/utils/data_loader.py`
2. Modify `load_data_from_file` or `load_stock_data_file`

#### Add a new CSV file type
1. Add parser in `src/parsers/`
2. Update `detect_csv_structure` in `src/utils/file_discovery.py`
3. Add to parser mapping in `src/utils/data_loader.py`

## 📦 Module Quick Reference

```
src/
├── parsers/          # CSV file parsing
│   ├── base_parsers.py        # Common patterns
│   ├── signal_parsers.py      # Strategy parsers
│   ├── advanced_parsers.py    # Special parsers
│   └── performance_parsers.py # Performance data
│
├── components/       # UI components
│   ├── cards.py      # All card displays
│   └── charts.py     # Interactive charts
│
├── pages/           # Page logic
│   ├── dashboard.py          # Main dashboard
│   ├── analysis_page.py      # Analysis pages
│   ├── performance_page.py   # Performance pages
│   ├── breadth_page.py       # Breadth analysis
│   └── text_file_page.py     # Text display
│
└── utils/           # Utilities
    ├── data_loader.py        # Data loading
    ├── file_discovery.py     # File discovery
    └── helpers.py            # Helper functions
```

## 🔍 Common Tasks

### Task 1: Add a New Strategy
```python
# In src/parsers/signal_parsers.py
def parse_my_strategy(df):
    """Parse my_strategy.csv"""
    return parse_signal_csv(df, 'My Strategy Name')

# In src/utils/file_discovery.py (detect_csv_structure)
file_mapping = {
    # ... existing mappings ...
    'my_strategy.csv': 'my_strategy',
}

# In src/utils/data_loader.py (load_data_from_file)
parser_mapping = {
    # ... existing mappings ...
    'my_strategy': parse_my_strategy,
}
```

### Task 2: Modify Card Display
```python
# In src/components/cards.py
def create_summary_cards(df):
    # Modify existing code here
    pass
```

### Task 3: Add New Metric
```python
# In src/components/cards.py
with col5:  # Add a new column
    new_metric = df['New_Field'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{new_metric:.1f}</p>
        <p class="metric-label">New Metric</p>
    </div>
    """, unsafe_allow_html=True)
```

## 🧪 Testing

### Check Syntax
```bash
python3 -m py_compile app.py
```

### Test Imports
```python
from src.parsers import parse_fib_ret
from src.components import create_summary_cards
from src.pages import create_analysis_page
from src.utils import load_data_from_file
```

### Run Application
```bash
streamlit run app.py
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `PROJECT_STRUCTURE.md` | Complete project structure |
| `src/README.md` | Detailed module documentation |
| `REFACTORING_SUMMARY.md` | Migration guide |
| `QUICK_START.md` | This file |

## 🆘 Troubleshooting

### Import Errors
- Check `__init__.py` files are present in all directories
- Verify import paths match module structure
- Run from project root directory

### Missing Data
- Check CSV files exist in `trade_store/US/`
- Verify file names match mappings in `file_discovery.py`

### Syntax Errors
- Run `python3 -m py_compile <file>` to check syntax
- Check indentation (Python uses spaces/tabs consistently)

### Revert to Original
```bash
mv app.py app_new.py
mv app_backup.py app.py
```

## 💡 Tips

1. **Always test** after making changes
2. **Use the backup** (`app_backup.py`) as reference
3. **Follow the existing patterns** in each module
4. **Keep functions small** and focused
5. **Add comments** for complex logic
6. **Update documentation** when adding features

## 🎯 Next Steps

1. ✅ Run the application
2. ✅ Test all pages
3. ✅ Verify CSV parsing
4. ✅ Test interactive charts
5. ✅ Make your first enhancement!

---

**Happy coding! 🚀**

For more details, see `PROJECT_STRUCTURE.md` and `src/README.md`

