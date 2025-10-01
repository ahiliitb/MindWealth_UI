import streamlit as st
import pandas as pd
import re
from constant import *

# Set page config
st.set_page_config(
    page_title="Trading Strategy Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    .strategy-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .strategy-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .positive {
        color: #00C851;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .neutral {
        color: #ffbb33;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def create_summary_cards(df):
    """Create summary metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_win_rate = df['Win_Rate'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_win_rate:.1f}%</p>
            <p class="metric-label">Average Win Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_trades = len(df['Num_Trades'])
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_trades}</p>
            <p class="metric-label">Total Trades</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_cagr = df['Strategy_CAGR'].mean()
        color_class = "positive" if avg_cagr > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {color_class}">{avg_cagr:.1f}%</p>
            <p class="metric-label">Average Strategy CAGR</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_sharpe = df['Strategy_Sharpe'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_sharpe:.2f}</p>
            <p class="metric-label">Average Sharpe Ratio</p>
        </div>
        """, unsafe_allow_html=True)

def create_strategy_cards(df):
    """Create individual strategy cards with pagination for large datasets"""
    st.markdown("### 📊 Strategy Performance Cards")
    st.markdown("Click on any card to see important trade details")
    
    total_signals = len(df)
    
    if total_signals == 0:
        st.warning("No signals match the current filters.")
        return
    
    # Display total count
    st.markdown(f"**Total Signals: {total_signals}**")
    
    # Pagination settings for strategy cards - 30 per tab for Signal Analysis
    cards_per_page = 30
    total_pages = (total_signals + cards_per_page - 1) // cards_per_page
    
    # Create pagination if there are many signals
    if total_signals <= cards_per_page:
        # If 30 or fewer signals, show all in one view
        display_strategy_cards_page(df)
    else:
        # Create tabs for pagination
        # Generate tab labels
        tab_labels = []
        for i in range(total_pages):
            start_idx = i * cards_per_page + 1
            end_idx = min((i + 1) * cards_per_page, total_signals)
            tab_labels.append(f"#{start_idx}-{end_idx}")
        
        # Create tabs dynamically
        if total_pages <= 8:  # Limit to 8 tabs to avoid overcrowding
            # If 8 or fewer pages, create all tabs at once
            tabs = st.tabs(tab_labels)
            for i, tab in enumerate(tabs):
                with tab:
                    start_idx = i * cards_per_page
                    end_idx = min((i + 1) * cards_per_page, total_signals)
                    page_df = df.iloc[start_idx:end_idx]
                    display_strategy_cards_page(page_df)
        else:
            # If more than 8 pages, use selectbox for navigation
            st.markdown("**Navigate to page:**")
            selected_page = st.selectbox(
                "Choose page:",
                options=list(range(1, total_pages + 1)),
                format_func=lambda x: f"Page {x} (#{(x-1)*cards_per_page + 1}-{min(x*cards_per_page, total_signals)})",
                key="strategy_cards_page_selector"
            )
            
            # Display selected page
            start_idx = (selected_page - 1) * cards_per_page
            end_idx = min(selected_page * cards_per_page, total_signals)
            page_df = df.iloc[start_idx:end_idx]
            
            st.markdown(f"**Showing signals {start_idx + 1} to {end_idx} of {total_signals}**")
            display_strategy_cards_page(page_df)

def display_strategy_cards_page(df):
    """Display strategy cards for a given page"""
    if len(df) == 0:
        st.warning("No data to display on this page.")
        return
    
    # Use Streamlit's container with height parameter for scrolling
    with st.container(height=600):  # Fixed height container that will scroll
        for idx, row in df.iterrows():
            # Create expandable card
            with st.expander(f"🔍 {row['Function']} - {row['Symbol']} ({row['Win_Rate']:.1f}% Win Rate)", expanded=False):
                st.markdown("**📋 Key Trade Information**")
                
                # Get raw data for specific fields
                raw_data = row['Raw_Data']
                
                # Create three columns for better layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🎯 Trade Details**")
                    st.write(f"**Symbol:** {row['Symbol']}")
                    st.write(f"**Function:** {row['Function']}")
                    st.write(f"**Interval:** {raw_data.get('Interval, Confirmation Status', 'N/A').split(',')[0] if ',' in str(raw_data.get('Interval, Confirmation Status', '')) else raw_data.get('Interval, Confirmation Status', 'N/A')}")
                    # Extract signal date and price properly
                    signal_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', 'N/A')
                    if 'Price:' in str(signal_info):
                        # Extract date (between the last comma and the opening parenthesis)
                        parts = str(signal_info).split(',')
                        if len(parts) >= 3:
                            signal_type = parts[1].strip()  # Get signal type (Long or Short)
                            date_part = parts[2].strip().split('(')[0].strip()  # Get date before (Price:
                            price_part = str(signal_info).split('(Price:')[1].replace(')', '').strip()  # Get price
                            st.write(f"**Signal:** {signal_type}")
                            st.write(f"**Signal Date:** {date_part}")
                            st.write(f"**Signal Price:** ${price_part}")
                        else:
                            st.write(f"**Signal Date & Price:** {signal_info}")
                    else:
                        st.write(f"**Signal Date & Price:** {signal_info}")
                    st.write(f"**Exit Date & Price:** {raw_data.get('Exit Signal Date/Price[$]', 'N/A')}")
                    st.write(f"**Win Rate:** {row['Win_Rate']:.1f}%")
                    
                with col2:
                    st.markdown("**📊 Status & Performance**")
                    st.write(f"**Confirmation Status:** {raw_data.get('Interval, Confirmation Status', 'N/A').split(',')[1].strip() if ',' in str(raw_data.get('Interval, Confirmation Status', '')) else 'N/A'}")
                    st.write(f"**Current MTM:** {raw_data.get('Current Mark to Market and Holding Period', 'N/A')}")
                    st.write(f"**Strategy CAGR:** {row['Strategy_CAGR']:.2f}%")
                    st.write(f"**Buy & Hold CAGR:** {row['Buy_Hold_CAGR']:.2f}%")
                    st.write(f"**Strategy Sharpe:** {row['Strategy_Sharpe']:.2f}")
                    st.write(f"**Buy & Hold Sharpe:** {row['Buy_Hold_Sharpe']:.2f}")
                    
                with col3:
                    st.markdown("**⚠️ Risk & Timing**")
                    st.write(f"**Cancellation Level/Date:** {raw_data.get('Cancellation Level/Date', 'N/A')}")
                    
                    # Extract average holding period from the complex string
                    holding_period_info = raw_data.get('Backtested Holding Period(Win Trades) (days) (Max./Min./Avg.)', 'N/A')
                    if '/' in str(holding_period_info):
                        avg_holding = str(holding_period_info).split('/')[-1].strip()
                        st.write(f"**Avg Holding Period:** {avg_holding} days")
                    else:
                        st.write(f"**Avg Holding Period:** N/A")
                    
                    # Extract average backtested return
                    returns_info = raw_data.get('Backtested Returns(Win Trades) [%] (Best/Worst/Avg)', 'N/A')
                    if '/' in str(returns_info):
                        avg_return = str(returns_info).split('/')[-1].strip()
                        st.write(f"**Avg Backtested Return:** {avg_return}")
                    else:
                        st.write(f"**Avg Backtested Return:** N/A")



def create_top_signals_dashboard():
    """Create the dashboard page"""
    st.title("📊 Trading Strategy Dashboard")
    st.markdown("---")
    
    st.info("Welcome to the Trading Strategy Analysis Dashboard! Use the navigation menu to explore different strategy analysis pages.")
    
    # Display overview of available strategies
    st.markdown("### 🎯 Available Strategy Analysis Pages")
    
    # Get list of available CSV files
    csv_files = discover_csv_files()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Function Strategies:**")
        strategy_pages = [
            "Band Matrix", "DeltaDrift", "Fractal Track", "BaselineDiverge",
            "Altitude Alpha", "Oscillator Delta", "SigmaShell", "PulseGauge",
            "TrendPulse", "Signal Breadth Indicator (SBI)"
        ]
        for page in strategy_pages:
            if page in csv_files:
                st.markdown(f"✅ {page}")
            else:
                st.markdown(f"❌ {page} (No data)")
    
    with col2:
        st.markdown("**Signal & Performance Pages:**")
        signal_pages = [
            "Outstanding Signals", "Outstanding Target", "Outstanding Signals Exit",
            "New Signals", "Latest Performance", "Forward Testing Performance"
        ]
        for page in signal_pages:
            if page in csv_files:
                st.markdown(f"✅ {page}")
            else:
                st.markdown(f"❌ {page} (No data)")
    
    st.markdown("### 📄 Additional Features")
    st.markdown("• **Claude Output**: View text file outputs")
    st.markdown("• **Dynamic Analysis**: Each page provides detailed strategy analysis with filters and visualizations")
    st.markdown("• **Real-time Data**: All data is loaded dynamically from CSV files")



def find_column_by_keywords(columns, keywords):
    """Find a column name that contains any of the keywords"""
    for col in columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    return None

def detect_csv_structure(file_path):
    """Detect the structure and type of CSV file"""
    try:
        df = pd.read_csv(file_path, nrows=1)  # Read only header
        columns = df.columns.tolist()
        
        # Check for different CSV types based on column patterns
        if any('Symbol, Signal' in col for col in columns):
            if 'Function' in columns:
                return 'detailed_signals'  # Like outstanding_signal.csv, top_signals.csv
            else:
                return 'basic_signals'  # Like bollinger_band.csv, Distance.csv
        elif 'Strategy' in columns and 'Interval' in columns:
            return 'performance_summary'  # Like latest_performance.csv, forward_backtesting.csv
        elif 'Function' in columns and 'Bullish Asset vs Total Asset' in str(columns):
            return 'breadth_data'  # Like breadth.csv
        elif 'Function' in columns and len(columns) < 10:
            return 'simple_signals'  # Simple structure
        else:
            return 'unknown'
    except Exception as e:
        st.error(f"Error detecting CSV structure for {file_path}: {str(e)}")
        return 'unknown'

def parse_detailed_signals(df):
    """Parse detailed signals CSV (like outstanding_signal.csv, top_signals.csv)"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Parse symbol and signal info
        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', '')
        symbol_match = re.search(r'([^,]+),\s*([^,]+),\s*([^(]+)\(Price:\s*([^)]+)\)', str(symbol_info))
        
        if symbol_match:
            symbol = symbol_match.group(1).strip()
            signal_type = symbol_match.group(2).strip()
            signal_date = symbol_match.group(3).strip()
            try:
                signal_price = float(symbol_match.group(4).strip())
            except:
                signal_price = 0
        else:
            symbol, signal_type, signal_date, signal_price = "Unknown", "Unknown", "Unknown", 0
        
        # Parse win rate and number of trades
        win_rate_info = row.get('Win Rate [%], History Tested, Number of Trades', '')
        win_rate_match = re.search(r'([0-9.]+)%.*?([0-9]+)$', str(win_rate_info))
        
        if win_rate_match:
            try:
                win_rate = float(win_rate_match.group(1))
                num_trades = int(win_rate_match.group(2))
            except:
                win_rate, num_trades = 0, 0
        else:
            win_rate, num_trades = 0, 0
        
        # Parse CAGR
        strategy_cagr = 0
        buy_hold_cagr = 0
        if 'Backtested Strategy CAGR [%]' in row:
            try:
                strategy_cagr = float(str(row['Backtested Strategy CAGR [%]']).replace('%', ''))
            except:
                strategy_cagr = 0
        if 'CAGR of Buy and Hold [%]' in row:
            try:
                buy_hold_cagr = float(str(row['CAGR of Buy and Hold [%]']).replace('%', ''))
            except:
                buy_hold_cagr = 0
        
        # Parse Sharpe ratios
        strategy_sharpe = 0
        buy_hold_sharpe = 0
        if 'Backtested Strategy Sharpe Ratio' in row:
            try:
                strategy_sharpe = float(row['Backtested Strategy Sharpe Ratio'])
            except:
                strategy_sharpe = 0
        if 'Sharpe Ratio of Buy and Hold' in row:
            try:
                buy_hold_sharpe = float(row['Sharpe Ratio of Buy and Hold'])
            except:
                buy_hold_sharpe = 0
        
        # Parse max loss and drawdown
        max_loss_info = row.get('Backtested Max Loss [%], Max Drawdown [%]', '')
        max_loss_match = re.search(r'([0-9.]+)%,\s*([0-9.]+)%', str(max_loss_info))
        
        if max_loss_match:
            try:
                max_loss = float(max_loss_match.group(1))
                max_drawdown = float(max_loss_match.group(2))
            except:
                max_loss, max_drawdown = 0, 0
        else:
            max_loss, max_drawdown = 0, 0
        
        # Parse returns
        returns_info = row.get('Backtested Returns(Win Trades) [%] (Best/Worst/Avg)', '')
        returns_match = re.search(r'([0-9.]+)%/([0-9.]+)%/([0-9.]+)%', str(returns_info))
        
        if returns_match:
            try:
                best_return = float(returns_match.group(1))
                worst_return = float(returns_match.group(2))
                avg_return = float(returns_match.group(3))
            except:
                best_return, worst_return, avg_return = 0, 0, 0
        else:
            best_return, worst_return, avg_return = 0, 0, 0
        
        processed_data.append({
            'Function': row.get('Function', 'Unknown'),
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'CAGR_Difference': strategy_cagr - buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Max_Loss': max_loss,
            'Max_Drawdown': max_drawdown,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_basic_signals(df, page_name="Unknown"):
    """Parse basic signals CSV (like bollinger_band.csv, Distance.csv)"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Parse symbol and signal info
        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', '')
        symbol_match = re.search(r'([^,]+),\s*([^,]+),\s*([^(]+)\(Price:\s*([^)]+)\)', str(symbol_info))
        
        if symbol_match:
            symbol = symbol_match.group(1).strip()
            signal_type = symbol_match.group(2).strip()
            signal_date = symbol_match.group(3).strip()
            try:
                signal_price = float(symbol_match.group(4).strip())
            except:
                signal_price = 0
        else:
            symbol, signal_type, signal_date, signal_price = "Unknown", "Unknown", "Unknown", 0
        
        # Parse win rate and number of trades
        win_rate_info = row.get('Win Rate [%], History Tested, Number of Trades', '')
        win_rate_match = re.search(r'([0-9.]+)%.*?([0-9]+)$', str(win_rate_info))
        
        if win_rate_match:
            try:
                win_rate = float(win_rate_match.group(1))
                num_trades = int(win_rate_match.group(2))
            except:
                win_rate, num_trades = 0, 0
        else:
            win_rate, num_trades = 0, 0
        
        # Parse CAGR
        strategy_cagr = 0
        buy_hold_cagr = 0
        if 'Backtested Strategy CAGR [%]' in row:
            try:
                strategy_cagr = float(str(row['Backtested Strategy CAGR [%]']).replace('%', ''))
            except:
                strategy_cagr = 0
        if 'CAGR of Buy and Hold [%]' in row:
            try:
                buy_hold_cagr = float(str(row['CAGR of Buy and Hold [%]']).replace('%', ''))
            except:
                buy_hold_cagr = 0
        
        # Parse Sharpe ratios
        strategy_sharpe = 0
        buy_hold_sharpe = 0
        if 'Backtested Strategy Sharpe Ratio' in row:
            try:
                strategy_sharpe = float(row['Backtested Strategy Sharpe Ratio'])
            except:
                strategy_sharpe = 0
        if 'Sharpe Ratio of Buy and Hold' in row:
            try:
                buy_hold_sharpe = float(row['Sharpe Ratio of Buy and Hold'])
            except:
                buy_hold_sharpe = 0
        
        # Parse max loss and drawdown
        max_loss_info = row.get('Backtested Max Loss [%], Max Drawdown [%]', '')
        max_loss_match = re.search(r'([0-9.]+)%,\s*([0-9.]+)%', str(max_loss_info))
        
        if max_loss_match:
            try:
                max_loss = float(max_loss_match.group(1))
                max_drawdown = float(max_loss_match.group(2))
            except:
                max_loss, max_drawdown = 0, 0
        else:
            max_loss, max_drawdown = 0, 0
        
        # Parse returns
        returns_info = row.get('Backtested Returns(Win Trades) [%] (Best/Worst/Avg)', '')
        returns_match = re.search(r'([0-9.]+)%/([0-9.]+)%/([0-9.]+)%', str(returns_info))
        
        if returns_match:
            try:
                best_return = float(returns_match.group(1))
                worst_return = float(returns_match.group(2))
                avg_return = float(returns_match.group(3))
            except:
                best_return, worst_return, avg_return = 0, 0, 0
        else:
            best_return, worst_return, avg_return = 0, 0, 0
        
        processed_data.append({
            'Function': page_name,  # Use page name as function name
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'CAGR_Difference': strategy_cagr - buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Max_Loss': max_loss,
            'Max_Drawdown': max_drawdown,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_performance_summary(df, page_name="Unknown"):
    """Parse performance summary CSV (like latest_performance.csv, forward_backtesting.csv)"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Extract win rate from percentage string
        win_rate_str = str(row.get('Win Percentage', '0%')).replace('%', '')
        try:
            win_rate = float(win_rate_str)
        except:
            win_rate = 0
        
        # Extract number of trades
        try:
            num_trades = int(row.get('Total Analysed Trades', 0))
        except:
            num_trades = 0
        
        # Extract average profit
        profit_str = str(row.get('Profit [%] (Best/Worst/Avg.)', '0/0/0'))
        profit_match = re.search(r'([0-9.-]+)%/([0-9.-]+)%/([0-9.-]+)%', profit_str)
        
        if profit_match:
            try:
                best_return = float(profit_match.group(1))
                worst_return = float(profit_match.group(2))
                avg_return = float(profit_match.group(3))
            except:
                best_return, worst_return, avg_return = 0, 0, 0
        else:
            best_return, worst_return, avg_return = 0, 0, 0
        
        # Extract holding period information
        holding_period_str = str(row.get('Holding Period (days) (Max./Min./Avg.)', '0/0/0'))
        holding_match = re.search(r'([0-9.]+) days/([0-9.]+) days/([0-9.]+) days', holding_period_str)
        
        if holding_match:
            try:
                max_holding = float(holding_match.group(1))
                min_holding = float(holding_match.group(2))
                avg_holding = float(holding_match.group(3))
            except:
                max_holding, min_holding, avg_holding = 0, 0, 0
        else:
            max_holding, min_holding, avg_holding = 0, 0, 0
        
        # Extract average backtested win rate
        try:
            avg_backtested_win_rate = float(str(row.get('Avg Backtested Win Rate [%]', '0%')).replace('%', ''))
        except:
            avg_backtested_win_rate = 0
        
        # Extract average backtested holding period
        try:
            avg_backtested_holding = float(str(row.get('Avg Backtested Holding Period (days)', '0')).replace(' days', ''))
        except:
            avg_backtested_holding = 0
        
        processed_data.append({
            'Strategy': row.get('Strategy', 'Unknown'),
            'Interval': row.get('Interval', 'Unknown'),
            'Signal_Type': row.get('Signal Type', 'Unknown'),
            'Total_Trades': num_trades,
            'Win_Percentage': win_rate,
            'Max_Holding_Days': max_holding,
            'Min_Holding_Days': min_holding,
            'Avg_Holding_Days': avg_holding,
            'Best_Profit': best_return,
            'Worst_Profit': worst_return,
            'Avg_Profit': avg_return,
            'Avg_Backtested_Win_Rate': avg_backtested_win_rate,
            'Avg_Backtested_Holding_Days': avg_backtested_holding,
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_breadth_data(df, page_name="Unknown"):
    """Parse breadth CSV (like breadth.csv)"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Extract function name
        function = row.get('Function', 'Unknown')
        
        # Extract bullish asset percentage
        bullish_asset_str = str(row.get('Bullish Asset vs Total Asset (%).', '0%')).replace('%', '')
        try:
            bullish_asset_pct = float(bullish_asset_str)
        except:
            bullish_asset_pct = 0
        
        # Extract bullish signal percentage
        bullish_signal_str = str(row.get('Bullish Signal vs Total Signal (%)', '0%')).replace('%', '')
        try:
            bullish_signal_pct = float(bullish_signal_str)
        except:
            bullish_signal_pct = 0
        
        processed_data.append({
            'Function': function,
            'Bullish_Asset_Percentage': bullish_asset_pct,
            'Bullish_Signal_Percentage': bullish_signal_pct,
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

@st.cache_data
def load_data_from_file(file_path, page_name="Unknown"):
    """Load and process trading data from any CSV file with dynamic structure detection"""
    try:
        # Detect CSV structure
        csv_type = detect_csv_structure(file_path)
        
        # Load the full CSV
        df = pd.read_csv(file_path)
        
        if df.empty:
            st.warning(f"No data found in {file_path}")
            return pd.DataFrame()
        
        # Parse based on detected structure
        if csv_type == 'detailed_signals':
            return parse_detailed_signals(df)
        elif csv_type == 'basic_signals':
            return parse_basic_signals(df, page_name)
        elif csv_type == 'performance_summary':
            return parse_performance_summary(df, page_name)
        elif csv_type == 'breadth_data':
            return parse_breadth_data(df, page_name)
        else:
            # Fallback to basic parsing for unknown structures
            st.warning(f"Unknown CSV structure for {file_path}, using basic parsing")
            return parse_basic_signals(df, page_name)
            
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame()

def create_performance_summary_cards(df):
    """Create summary metric cards for performance data"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_win_rate = df['Win_Percentage'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_win_rate:.1f}%</p>
            <p class="metric-label">Average Win Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_trades = df['Total_Trades'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_trades}</p>
            <p class="metric-label">Total Trades</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_profit = df['Avg_Profit'].mean()
        color_class = "positive" if avg_profit > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {color_class}">{avg_profit:.1f}%</p>
            <p class="metric-label">Average Profit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_holding = df['Avg_Holding_Days'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_holding:.0f}</p>
            <p class="metric-label">Avg Holding Days</p>
        </div>
        """, unsafe_allow_html=True)

def create_performance_cards(df):
    """Create individual performance cards with pagination for large datasets"""
    st.markdown("### 📊 Performance Analysis Cards")
    st.markdown("Click on any card to see detailed performance metrics")
    
    total_records = len(df)
    
    if total_records == 0:
        st.warning("No performance data matches the current filters.")
        return
    
    # Display total count
    st.markdown(f"**Total Records: {total_records}**")
    
    # Pagination settings for performance cards - 30 per tab
    cards_per_page = 30
    total_pages = (total_records + cards_per_page - 1) // cards_per_page
    
    # Create pagination if there are many records
    if total_records <= cards_per_page:
        # If 30 or fewer records, show all in one view
        display_performance_cards_page(df)
    else:
        # Create tabs for pagination
        # Generate tab labels
        tab_labels = []
        for i in range(total_pages):
            start_idx = i * cards_per_page + 1
            end_idx = min((i + 1) * cards_per_page, total_records)
            tab_labels.append(f"#{start_idx}-{end_idx}")
        
        # Create tabs dynamically
        if total_pages <= 8:  # Limit to 8 tabs to avoid overcrowding
            # If 8 or fewer pages, create all tabs at once
            tabs = st.tabs(tab_labels)
            for i, tab in enumerate(tabs):
                with tab:
                    start_idx = i * cards_per_page
                    end_idx = min((i + 1) * cards_per_page, total_records)
                    page_df = df.iloc[start_idx:end_idx]
                    display_performance_cards_page(page_df)
        else:
            # If more than 8 pages, use selectbox for navigation
            st.markdown("**Navigate to page:**")
            selected_page = st.selectbox(
                "Choose page:",
                options=list(range(1, total_pages + 1)),
                format_func=lambda x: f"Page {x} (#{(x-1)*cards_per_page + 1}-{min(x*cards_per_page, total_records)})",
                key="performance_cards_page_selector"
            )
            
            # Display selected page
            start_idx = (selected_page - 1) * cards_per_page
            end_idx = min(selected_page * cards_per_page, total_records)
            page_df = df.iloc[start_idx:end_idx]
            
            st.markdown(f"**Showing records {start_idx + 1} to {end_idx} of {total_records}**")
            display_performance_cards_page(page_df)

def display_performance_cards_page(df):
    """Display performance cards for a given page"""
    if len(df) == 0:
        st.warning("No data to display on this page.")
        return
    
    # Use Streamlit's container with height parameter for scrolling
    with st.container(height=600):  # Fixed height container that will scroll
        for idx, row in df.iterrows():
            # Create expandable card
            with st.expander(f"📊 {row['Strategy']} - {row['Interval']} ({row['Win_Percentage']:.1f}% Win Rate)", expanded=False):
                st.markdown("**📋 Performance Metrics**")
                
                # Create three columns for better layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🎯 Strategy Details**")
                    st.write(f"**Strategy:** {row['Strategy']}")
                    st.write(f"**Interval:** {row['Interval']}")
                    st.write(f"**Signal Type:** {row['Signal_Type']}")
                    st.write(f"**Total Trades:** {row['Total_Trades']}")
                    st.write(f"**Win Percentage:** {row['Win_Percentage']:.1f}%")
                    
                with col2:
                    st.markdown("**📊 Profit Analysis**")
                    st.write(f"**Best Profit:** {row['Best_Profit']:.1f}%")
                    st.write(f"**Worst Profit:** {row['Worst_Profit']:.1f}%")
                    st.write(f"**Average Profit:** {row['Avg_Profit']:.1f}%")
                    st.write(f"**Avg Backtested Win Rate:** {row['Avg_Backtested_Win_Rate']:.1f}%")
                    
                with col3:
                    st.markdown("**⏱️ Holding Period Analysis**")
                    st.write(f"**Max Holding Days:** {row['Max_Holding_Days']:.0f}")
                    st.write(f"**Min Holding Days:** {row['Min_Holding_Days']:.0f}")
                    st.write(f"**Avg Holding Days:** {row['Avg_Holding_Days']:.0f}")
                    st.write(f"**Avg Backtested Holding:** {row['Avg_Backtested_Holding_Days']:.0f} days")


def create_text_file_page():
    """Create a page to display text files with tabs"""
    st.title("📄 Claude Output")
    st.markdown("---")
    
    # Create two tabs for the text files
    tab1, tab2 = st.tabs(["Claude Output", "Box Claude Output"])
    
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
        st.markdown("### 📦 Box Claude Output")
        try:
            with open(BOX_CLAUDE_OUTPUT_TXT_PATH, 'r', encoding='utf-8') as file:
                content = file.read()
                st.text_area("File Content:", content, height=600, key="box_claude_output")
        except FileNotFoundError:
            st.error(f"File not found: {BOX_CLAUDE_OUTPUT_TXT_PATH}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def create_performance_summary_page(data_file, page_title):
    """Create a performance summary page for CSV files with different structure"""
    st.title(f"📊 {page_title}")
    st.markdown("---")
    
    # Load data from the specific file
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Create main tabs for signal types
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 ALL Signal Types", "📈 Long Signals", "📉 Short Signals"])
    
    # Sidebar filters for performance data
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Strategy filter
    st.sidebar.markdown("**Strategies:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("All", key=f"select_all_strategies_{page_title}", help="Select all strategies", use_container_width=True):
            st.session_state[f'selected_strategies_{page_title}'] = list(df['Strategy'].unique())
    with col2:
        if st.button("None", key=f"deselect_all_strategies_{page_title}", help="Deselect all strategies", use_container_width=True):
            st.session_state[f'selected_strategies_{page_title}'] = []
    
    # Initialize session state for strategies
    if f'selected_strategies_{page_title}' not in st.session_state:
        st.session_state[f'selected_strategies_{page_title}'] = list(df['Strategy'].unique())
    
    # Display strategy selection status
    if len(st.session_state[f'selected_strategies_{page_title}']) == len(df['Strategy'].unique()):
        st.sidebar.markdown("*All strategies selected*")
    elif len(st.session_state[f'selected_strategies_{page_title}']) == 0:
        st.sidebar.markdown("*No strategies selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state[f'selected_strategies_{page_title}'])} of {len(df['Strategy'].unique())} selected*")
    
    with st.sidebar.expander("Select Strategies", expanded=False):
        strategies = st.multiselect(
            "",
            options=df['Strategy'].unique(),
            default=st.session_state[f'selected_strategies_{page_title}'],
            key=f"strategies_multiselect_{page_title}",
            label_visibility="collapsed"
        )
    
    # Update session state
    st.session_state[f'selected_strategies_{page_title}'] = strategies
    
    
    # Win rate filter
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=0,
        help="Minimum win rate threshold",
        key=f"win_rate_slider_{page_title}"
    )
    
    def display_performance_content(filtered_df, tab_name, signal_type_filter=None):
        """Display performance content for each tab"""
        if filtered_df.empty:
            st.warning(f"No {tab_name} data matches the selected filters. Please adjust your filters.")
            return
        
        # Performance summary cards
        st.markdown(f"### 🎯 Performance Summary - {tab_name}")
        create_performance_summary_cards(filtered_df)
        
        st.markdown("---")
        
        # Performance cards
        create_performance_cards(filtered_df)
        
        st.markdown("---")
        
        # Data table - Original CSV format
        st.markdown(f"### 📋 Detailed Data Table - {tab_name} (Original CSV Format)")
        
        # Create a dataframe with original CSV data
        csv_data = []
        for _, row in filtered_df.iterrows():
            csv_data.append(row['Raw_Data'])
        
        if csv_data:
            original_df = pd.DataFrame(csv_data)
            
            # Display with better formatting
            st.dataframe(
                original_df,
                use_container_width=True,
                height=600,
                column_config={
                    col: st.column_config.TextColumn(
                        col,
                        width="medium",
                        help=f"Original CSV column: {col}"
                    ) for col in original_df.columns
                }
            )
    
    # ALL Signal Types
    with main_tab1:
        # Create interval tabs for ALL signal types
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = df[
                (df['Strategy'].isin(strategies)) &
                (df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "ALL Intervals", "ALL")
        
        # Daily
        with interval_tab2:
            daily_df = df[df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Strategy'].isin(strategies)) &
                (daily_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Daily", "ALL")
        
        # Weekly
        with interval_tab3:
            weekly_df = df[df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Strategy'].isin(strategies)) &
                (weekly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Weekly", "ALL")
        
        # Monthly
        with interval_tab4:
            monthly_df = df[df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Strategy'].isin(strategies)) &
                (monthly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Monthly", "ALL")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = df[df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Strategy'].isin(strategies)) &
                (quarterly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Quarterly", "ALL")
        
        # Yearly
        with interval_tab6:
            yearly_df = df[df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Strategy'].isin(strategies)) &
                (yearly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Yearly", "ALL")
    
    # Long Signals
    with main_tab2:
        # Create interval tabs for Long signals
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = df[
                (df['Strategy'].isin(strategies)) &
                (df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "ALL Intervals", "Long")
        
        # Daily
        with interval_tab2:
            daily_df = df[df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Strategy'].isin(strategies)) &
                (daily_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (daily_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Daily", "Long")
        
        # Weekly
        with interval_tab3:
            weekly_df = df[df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Strategy'].isin(strategies)) &
                (weekly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (weekly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Weekly", "Long")
        
        # Monthly
        with interval_tab4:
            monthly_df = df[df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Strategy'].isin(strategies)) &
                (monthly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (monthly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Monthly", "Long")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = df[df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Strategy'].isin(strategies)) &
                (quarterly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (quarterly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Quarterly", "Long")
        
        # Yearly
        with interval_tab6:
            yearly_df = df[df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Strategy'].isin(strategies)) &
                (yearly_df['Signal_Type'].str.contains('Long', case=False, na=False)) &
                (yearly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Yearly", "Long")
    
    # Short Signals
    with main_tab3:
        # Create interval tabs for Short signals
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = df[
                (df['Strategy'].isin(strategies)) &
                (df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "ALL Intervals", "Short")
        
        # Daily
        with interval_tab2:
            daily_df = df[df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Strategy'].isin(strategies)) &
                (daily_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (daily_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Daily", "Short")
        
        # Weekly
        with interval_tab3:
            weekly_df = df[df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Strategy'].isin(strategies)) &
                (weekly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (weekly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Weekly", "Short")
        
        # Monthly
        with interval_tab4:
            monthly_df = df[df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Strategy'].isin(strategies)) &
                (monthly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (monthly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Monthly", "Short")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = df[df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Strategy'].isin(strategies)) &
                (quarterly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (quarterly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Quarterly", "Short")
        
        # Yearly
        with interval_tab6:
            yearly_df = df[df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Strategy'].isin(strategies)) &
                (yearly_df['Signal_Type'].str.contains('Short', case=False, na=False)) &
                (yearly_df['Win_Percentage'] >= min_win_rate)
            ]
            display_performance_content(filtered_df, "Yearly", "Short")

def create_breadth_summary_cards(df):
    """Create summary metric cards for breadth data"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_bullish_assets = df['Bullish_Asset_Percentage'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Bullish Assets</h3>
            <h2>{avg_bullish_assets:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_bullish_signals = df['Bullish_Signal_Percentage'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Bullish Signals</h3>
            <h2>{avg_bullish_signals:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_strategies = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Strategies</h3>
            <h2>{total_strategies}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Find the strategy with highest bullish assets
        best_asset_strategy = df.loc[df['Bullish_Asset_Percentage'].idxmax(), 'Function']
        best_asset_pct = df['Bullish_Asset_Percentage'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Asset Breadth</h3>
            <h2>{best_asset_pct:.1f}%</h2>
            <p>{best_asset_strategy}</p>
        </div>
        """, unsafe_allow_html=True)

def create_breadth_cards(df):
    """Create individual breadth analysis cards"""
    # Create cards in a 2-column layout
    cols = st.columns(2)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        with cols[idx % 2]:
            with st.expander(f"📊 {row['Function']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Bullish Assets",
                        f"{row['Bullish_Asset_Percentage']:.1f}%",
                        help="Percentage of bullish assets vs total assets"
                    )
                
                with col2:
                    st.metric(
                        "Bullish Signals",
                        f"{row['Bullish_Signal_Percentage']:.1f}%",
                        help="Percentage of bullish signals vs total signals"
                    )
                
                # Breadth strength indicator
                avg_breadth = (row['Bullish_Asset_Percentage'] + row['Bullish_Signal_Percentage']) / 2
                
                if avg_breadth >= 70:
                    breadth_status = "🟢 Strong"
                    breadth_color = "green"
                elif avg_breadth >= 40:
                    breadth_status = "🟡 Moderate"
                    breadth_color = "orange"
                else:
                    breadth_status = "🔴 Weak"
                    breadth_color = "red"
                
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <h4 style="color: {breadth_color};">{breadth_status}</h4>
                    <p>Average Breadth: {avg_breadth:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)


def create_breadth_page(data_file, page_title):
    """Create a specialized page for breadth data"""
    st.title(f"📊 {page_title}")
    st.markdown("---")
    
    # Load data from the specific file
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Breadth summary cards
    st.markdown("### 🎯 Market Breadth Summary")
    create_breadth_summary_cards(df)
    
    st.markdown("---")
    
    # Breadth analysis cards
    st.markdown("### 📈 Strategy Breadth Analysis")
    create_breadth_cards(df)
    
    st.markdown("---")
    
    # Data table - Original CSV format
    st.markdown("### 📋 Detailed Data Table (Original CSV Format)")
    
    # Create a dataframe with original CSV data
    csv_data = []
    for _, row in df.iterrows():
        csv_data.append(row['Raw_Data'])
    
    if csv_data:
        original_df = pd.DataFrame(csv_data)
        
        # Display with better formatting
        st.dataframe(
            original_df,
            use_container_width=True,
            height=400,
            column_config={
                col: st.column_config.TextColumn(
                    col,
                    width="medium",
                    help=f"Original CSV column: {col}"
                ) for col in original_df.columns
            }
        )

def create_analysis_page(data_file, page_title):
    """Create an analysis page similar to Signal Analysis for any CSV file"""
    st.title(f"📈 {page_title}")
    st.markdown("---")
    
    # Load data from the specific file
    df = load_data_from_file(f'{data_file}', page_title)
    
    if df.empty:
        st.warning(f"No data available for {page_title}")
        return
    
    # Check if this is a performance summary page (after processing)
    if 'Strategy' in df.columns and 'Interval' in df.columns and 'Total_Trades' in df.columns:
        create_performance_summary_page(data_file, page_title)
        return
    
    # Check if this is a breadth data page (after processing)
    if 'Function' in df.columns and 'Bullish_Asset_Percentage' in df.columns and 'Bullish_Signal_Percentage' in df.columns:
        create_breadth_page(data_file, page_title)
        return
    
    # Add interval and position type extraction
    def extract_interval(row):
        interval_info = row['Raw_Data'].get('Interval, Confirmation Status', 'Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = str(interval_info).strip()
        return interval
    
    def extract_position_type(row):
        signal_info = row['Raw_Data'].get('Symbol, Signal, Signal Date/Price[$]', '')
        if 'Long' in str(signal_info):
            return 'Long'
        elif 'Short' in str(signal_info):
            return 'Short'
        else:
            return 'Unknown'
    
    df['Interval'] = df.apply(extract_interval, axis=1)
    df['Position_Type'] = df.apply(extract_position_type, axis=1)
    
    # Create main tabs for position types
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 ALL Positions", "📈 Long Positions", "📉 Short Positions"])
    
    # Sidebar filters (same as Signal Analysis)
    st.sidebar.markdown("#### 🔍 Filters")
    
    # Function filter with select all/none buttons
    st.sidebar.markdown("**Functions:**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("All", key=f"select_all_functions_{page_title}", help="Select all functions", use_container_width=True):
            st.session_state[f'selected_functions_{page_title}'] = list(df['Function'].unique())
    with col2:
        if st.button("None", key=f"deselect_all_functions_{page_title}", help="Deselect all functions", use_container_width=True):
            st.session_state[f'selected_functions_{page_title}'] = []
    
    # Initialize session state for functions
    if f'selected_functions_{page_title}' not in st.session_state:
        st.session_state[f'selected_functions_{page_title}'] = list(df['Function'].unique())
    
    # Display function selection status
    if len(st.session_state[f'selected_functions_{page_title}']) == len(df['Function'].unique()):
        st.sidebar.markdown("*All functions selected*")
    elif len(st.session_state[f'selected_functions_{page_title}']) == 0:
        st.sidebar.markdown("*No functions selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state[f'selected_functions_{page_title}'])} of {len(df['Function'].unique())} selected*")
    
    with st.sidebar.expander("Select Functions", expanded=False):
        functions = st.multiselect(
            "",
            options=df['Function'].unique(),
            default=st.session_state[f'selected_functions_{page_title}'],
            key=f"functions_multiselect_{page_title}",
            label_visibility="collapsed"
        )
    
    # Update session state
    st.session_state[f'selected_functions_{page_title}'] = functions
    
    # Symbol filter with select all/none buttons
    st.sidebar.markdown("**Symbols:**")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("All", key=f"select_all_symbols_{page_title}", help="Select all symbols", use_container_width=True):
            st.session_state[f'selected_symbols_{page_title}'] = list(df['Symbol'].unique())
    with col4:
        if st.button("None", key=f"deselect_all_symbols_{page_title}", help="Deselect all symbols", use_container_width=True):
            st.session_state[f'selected_symbols_{page_title}'] = []
    
    # Initialize session state for symbols
    if f'selected_symbols_{page_title}' not in st.session_state:
        st.session_state[f'selected_symbols_{page_title}'] = list(df['Symbol'].unique())
    
    # Display symbol selection status
    if len(st.session_state[f'selected_symbols_{page_title}']) == len(df['Symbol'].unique()):
        st.sidebar.markdown("*All symbols selected*")
    elif len(st.session_state[f'selected_symbols_{page_title}']) == 0:
        st.sidebar.markdown("*No symbols selected*")
    else:
        st.sidebar.markdown(f"*{len(st.session_state[f'selected_symbols_{page_title}'])} of {len(df['Symbol'].unique())} selected*")
    
    with st.sidebar.expander("Select Symbols", expanded=False):
        symbols = st.multiselect(
            "",
            options=df['Symbol'].unique(),
            default=st.session_state[f'selected_symbols_{page_title}'],
            key=f"symbols_multiselect_{page_title}",
            label_visibility="collapsed"
        )
    
    # Update session state
    st.session_state[f'selected_symbols_{page_title}'] = symbols
    
    # Win rate filter
    min_win_rate = st.sidebar.slider(
        "Min Win Rate (%)",
        min_value=0,
        max_value=100,
        value=0,
        help="Minimum win rate threshold",
        key=f"win_rate_slider_{page_title}"
    )
    
    # Use the same display_interval_tabs function but with unique keys
    def display_interval_tabs_for_page(position_df, position_name):
        """Display interval tabs within each position tab for this page"""
        # Create interval sub-tabs
        interval_tab1, interval_tab2, interval_tab3, interval_tab4, interval_tab5, interval_tab6 = st.tabs([
            "📊 ALL", "📅 Daily", "📆 Weekly", "📈 Monthly", "📋 Quarterly", "📊 Yearly"
        ])
        
        def display_tab_content(filtered_df, tab_name):
            """Display content for each tab"""
            if filtered_df.empty:
                st.warning(f"No {tab_name} data matches the selected filters. Please adjust your filters.")
                return
            
            # Create unique key prefix for charts
            chart_key = f"{page_title.lower().replace(' ', '_')}_{position_name.lower().replace(' ', '_')}_{tab_name.lower().replace(' ', '_')}"
            
            # Summary cards
            st.markdown(f"### 🎯 Key Performance Metrics - {position_name} {tab_name}")
            create_summary_cards(filtered_df)
            
            st.markdown("---")
            
            # Strategy cards
            create_strategy_cards(filtered_df)
            
            st.markdown("---")
            
            # Data table - Original CSV format
            st.markdown(f"### 📋 Detailed Data Table - {position_name} {tab_name} (Original CSV Format)")
            
            # Create a dataframe with original CSV data
            csv_data = []
            for _, row in filtered_df.iterrows():
                csv_data.append(row['Raw_Data'])
            
            if csv_data:
                original_df = pd.DataFrame(csv_data)
                
                # Display with better formatting
                st.dataframe(
                    original_df,
                    use_container_width=True,
                    height=600,
                    column_config={
                        col: st.column_config.TextColumn(
                            col,
                            width="medium",
                            help=f"Original CSV column: {col}"
                        ) for col in original_df.columns
                    }
                )
        
        # ALL Intervals
        with interval_tab1:
            filtered_df = position_df[
                (position_df['Function'].isin(functions)) &
                (position_df['Symbol'].isin(symbols)) &
                (position_df['Win_Rate'] >= min_win_rate)
            ]
            display_tab_content(filtered_df, "ALL Intervals")
        
        # Daily
        with interval_tab2:
            daily_df = position_df[position_df['Interval'].str.contains('Daily', case=False, na=False)]
            filtered_df = daily_df[
                (daily_df['Function'].isin(functions)) &
                (daily_df['Symbol'].isin(symbols)) &
                (daily_df['Win_Rate'] >= min_win_rate)
            ]
            display_tab_content(filtered_df, "Daily")
        
        # Weekly
        with interval_tab3:
            weekly_df = position_df[position_df['Interval'].str.contains('Weekly', case=False, na=False)]
            filtered_df = weekly_df[
                (weekly_df['Function'].isin(functions)) &
                (weekly_df['Symbol'].isin(symbols)) &
                (weekly_df['Win_Rate'] >= min_win_rate)
            ]
            display_tab_content(filtered_df, "Weekly")
        
        # Monthly
        with interval_tab4:
            monthly_df = position_df[position_df['Interval'].str.contains('Monthly', case=False, na=False)]
            filtered_df = monthly_df[
                (monthly_df['Function'].isin(functions)) &
                (monthly_df['Symbol'].isin(symbols)) &
                (monthly_df['Win_Rate'] >= min_win_rate)
            ]
            display_tab_content(filtered_df, "Monthly")
        
        # Quarterly
        with interval_tab5:
            quarterly_df = position_df[position_df['Interval'].str.contains('Quarterly', case=False, na=False)]
            filtered_df = quarterly_df[
                (quarterly_df['Function'].isin(functions)) &
                (quarterly_df['Symbol'].isin(symbols)) &
                (quarterly_df['Win_Rate'] >= min_win_rate)
            ]
            display_tab_content(filtered_df, "Quarterly")
        
        # Yearly
        with interval_tab6:
            yearly_df = position_df[position_df['Interval'].str.contains('Yearly', case=False, na=False)]
            filtered_df = yearly_df[
                (yearly_df['Function'].isin(functions)) &
                (yearly_df['Symbol'].isin(symbols)) &
                (yearly_df['Win_Rate'] >= min_win_rate)
            ]
            display_tab_content(filtered_df, "Yearly")
    
    # ALL Positions Tab
    with main_tab1:
        display_interval_tabs_for_page(df, "ALL Positions")
    
    # Long Positions Tab
    with main_tab2:
        long_df = df[df['Position_Type'] == 'Long']
        display_interval_tabs_for_page(long_df, "Long Positions")
    
    # Short Positions Tab
    with main_tab3:
        short_df = df[df['Position_Type'] == 'Short']
        display_interval_tabs_for_page(short_df, "Short Positions")

def discover_csv_files():
    """Dynamically discover all CSV files in the trade_store/US directory"""
    import os
    import glob
    
    # Define the specific order for page names
    ordered_pages = [
        'Band Matrix',
        'DeltaDrift', 
        'Fractal Track',
        'BaselineDiverge',
        'Altitude Alpha',
        'Oscillator Delta',
        'SigmaShell',
        'PulseGauge',
        'TrendPulse',
        'Signal Breadth Indicator (SBI)',
        'Outstanding Signals',
        'Outstanding Target',
        'Outstanding Signals Exit',
        'New Signals',
        'Latest Performance',
        'Forward Testing Performance'
    ]
    
    # Map file names to model function names
    name_mapping = {
        'Bollinger Band': 'Band Matrix',
        'Distance': 'DeltaDrift',
        'Fib Ret': 'Fractal Track',
        'General Divergence': 'BaselineDiverge',
        'New High': 'Altitude Alpha',
        'Stochastic Divergence': 'Oscillator Delta',
        'Sigma': 'SigmaShell',
        'Sentiment': 'PulseGauge',
        'Trendline': 'TrendPulse',
        'Breadth': 'Signal Breadth Indicator (SBI)',
        'Outstanding Signal': 'Outstanding Signals',
        'Outstanding Exit Signal': 'Outstanding Signals Exit',
        'New Signal': 'New Signals',
        'Latest Performance': 'Latest Performance',
        'Forward Testing': 'Forward Testing Performance',
        'Target Signal': 'Outstanding Target'
    }
    
    csv_files = {}
    trade_store_path = "./trade_store/US"
    
    if os.path.exists(trade_store_path):
        # Find all CSV files
        csv_pattern = os.path.join(trade_store_path, "*.csv")
        csv_file_paths = glob.glob(csv_pattern)
        
        # Create a mapping from filename to filepath
        file_mapping = {}
        for file_path in csv_file_paths:
            filename = os.path.basename(file_path)
            # Remove .csv extension and create a readable name
            name = filename.replace('.csv', '').replace('_', ' ').replace('-', ' ').title()
            file_mapping[name] = file_path
        
        # Add files in the specified order
        for page_name in ordered_pages:
            # Find the corresponding file
            for original_name, mapped_name in name_mapping.items():
                if mapped_name == page_name and original_name in file_mapping:
                    csv_files[page_name] = file_mapping[original_name]
                    break
    
    return csv_files

def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("**Select Page**")
    
    # Dynamically discover CSV files
    csv_files = discover_csv_files()
    
    # Define all available pages in the correct order
    page_options = {
        "Dashboard": None,
        "Claude Output": "text_files",
    }
    
    # Add CSV files in the specified order
    page_options.update(csv_files)
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page:",
        list(page_options.keys()),
        key="page_selector"
    )
    
    # Display selected page
    if page == "Dashboard":
        create_top_signals_dashboard()
    elif page == "Claude Output":
        create_text_file_page()
    else:
        # Create analysis page for CSV files
        csv_file = page_options[page]
        if csv_file and csv_file != "text_files":
            create_analysis_page(csv_file, page)
        else:
            st.error(f"No data file found for {page}")

if __name__ == "__main__":
    main()