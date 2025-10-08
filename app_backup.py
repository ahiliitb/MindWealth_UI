import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
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

def create_strategy_cards(df, page_name="Unknown"):
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
    
    # Create tabs for pagination - always use tabs instead of dropdown
    if total_signals <= cards_per_page:
        # If all signals fit in one page, just display them
        display_strategy_cards_page(df, page_name)
    else:
        # Generate tab labels
        tab_labels = []
        for i in range(total_pages):
            start_idx = i * cards_per_page + 1
            end_idx = min((i + 1) * cards_per_page, total_signals)
            tab_labels.append(f"#{start_idx}-{end_idx}")
        
        
        # Create tabs for all pages
        tabs = st.tabs(tab_labels)
        for i, tab in enumerate(tabs):
            with tab:
                start_idx = i * cards_per_page
                end_idx = min((i + 1) * cards_per_page, total_signals)
                page_df = df.iloc[start_idx:end_idx]
                st.markdown(f"**Showing signals {start_idx + 1} to {end_idx} of {total_signals}**")
                display_strategy_cards_page(page_df, page_name)

def display_strategy_cards_page(df, page_name="Unknown"):
    """Display strategy cards for a given page"""
    if len(df) == 0:
        st.warning("No data to display on this page.")
        return
    
    # Display strategy cards without height constraint for full-width charts
    for idx, row in df.iterrows():
        # Create expandable card
        with st.expander(f"🔍 {row['Function']} - {row['Symbol']} ({row['Win_Rate']:.1f}% Win Rate)", expanded=False):
            st.markdown("**📋 Key Trade Information**")
            
            # Get raw data for specific fields
            raw_data = row['Raw_Data']

            # Add interactive chart button for Fib-Ret page and FRACTAL TRACK functions
            show_chart = False
            if page_name == "Fractal Track":
                show_chart = True
            elif 'Function' in row and row['Function'] == 'FRACTAL TRACK':
                show_chart = True
            
            if show_chart:
                if st.button(f"📊 View Interactive Chart", key=f"chart_{idx}"):
                    create_interactive_chart(row, raw_data)
            
            
            # Create three columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🎯 Trade Details**")
                st.write(f"**Symbol:** {row['Symbol']}")
                st.write(f"**Function:** {row['Function']}")
                
                # Handle different data structures
                if 'Interval' in row and row['Interval'] != 'Unknown':
                    st.write(f"**Interval:** {row['Interval']}")
                else:
                    # Fallback to raw data parsing
                    interval_info = raw_data.get('Interval, Confirmation Status', 'N/A')
                    if ',' in str(interval_info):
                        st.write(f"**Interval:** {str(interval_info).split(',')[0]}")
                    else:
                        st.write(f"**Interval:** {interval_info}")
                
                # Handle signal information - check if we have parsed data or need to parse raw data
                if 'Signal_Type' in row and row['Signal_Type'] != 'Unknown':
                    st.write(f"**Signal:** {row['Signal_Type']}")
                    if 'Signal_Date' in row and row['Signal_Date'] != 'Unknown':
                        st.write(f"**Signal Date:** {row['Signal_Date']}")
                    if 'Signal_Price' in row and row['Signal_Price'] != 0:
                        st.write(f"**Signal Price:** ${row['Signal_Price']:.4f}")
                else:
                    # Fallback to raw data parsing
                    # Fallback to raw data parsing
                    signal_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', 'N/A')
                    if 'Price:' in str(signal_info):
                        parts = str(signal_info).split(',')
                        if len(parts) >= 3:
                            signal_type = parts[1].strip()
                            date_part = parts[2].strip().split('(')[0].strip()
                            price_part = str(signal_info).split('(Price:')[1].replace(')', '').strip()
                            st.write(f"**Signal:** {signal_type}")
                            st.write(f"**Signal Date:** {date_part}")
                            st.write(f"**Signal Price:** ${price_part}")
                        else:
                            st.write(f"**Signal Date & Price:** {signal_info}")
                    else:
                        st.write(f"**Signal Date & Price:** {signal_info}")
                # Handle exit information
                if 'Entry_Date' in row and row['Entry_Date'] != 'Unknown':
                    st.write(f"**Entry Date:** {row['Entry_Date']}")
                    if 'Entry_Price' in row and row['Entry_Price'] != 0:
                        st.write(f"**Entry Price:** ${row['Entry_Price']:.4f}")
                else:
                    st.write(f"**Exit Date & Price:** {raw_data.get('Exit Signal Date/Price[$]', 'N/A')}")
                
                    st.write(f"**Win Rate:** {row['Win_Rate']:.1f}%")
                    
            with col2:
                st.markdown("**📊 Status & Performance**")
                
                # Handle confirmation status
                if 'Interval, Confirmation Status' in raw_data:
                    conf_status = raw_data.get('Interval, Confirmation Status', 'N/A')
                    if ',' in str(conf_status):
                        st.write(f"**Confirmation Status:** {str(conf_status).split(',')[1].strip()}")
                    else:
                        st.write(f"**Confirmation Status:** N/A")
                else:
                    st.write(f"**Confirmation Status:** N/A")
                
                # Handle current status
                if 'Current_Date' in row and row['Current_Date'] != 'Unknown':
                    st.write(f"**Current Date:** {row['Current_Date']}")
                    if 'Current_Price' in row and row['Current_Price'] != 0:
                        st.write(f"**Current Price:** ${row['Current_Price']:.4f}")
                else:
                    st.write(f"**Current MTM:** {raw_data.get('Current Mark to Market and Holding Period', 'N/A')}")
                
                # Handle performance metrics
                if 'Strategy_CAGR' in row:
                    st.write(f"**Strategy CAGR:** {row['Strategy_CAGR']:.2f}%")
                if 'Buy_Hold_CAGR' in row:
                    st.write(f"**Buy & Hold CAGR:** {row['Buy_Hold_CAGR']:.2f}%")
                if 'Strategy_Sharpe' in row:
                    st.write(f"**Strategy Sharpe:** {row['Strategy_Sharpe']:.2f}")
                if 'Buy_Hold_Sharpe' in row:
                    st.write(f"**Buy & Hold Sharpe:** {row['Buy_Hold_Sharpe']:.2f}")
                    
                # Handle gain information for target signals
                if 'Gain_Percentage' in row and row['Gain_Percentage'] != 0:
                    st.write(f"**Gain:** {row['Gain_Percentage']:.2f}%")
                if 'Holding_Days' in row and row['Holding_Days'] != 0:
                    st.write(f"**Holding Days:** {row['Holding_Days']} days")
                    
            with col3:
                st.markdown("**⚠️ Risk & Timing**")
                st.write(f"**Cancellation Level/Date:** {raw_data.get('Cancellation Level/Date', 'N/A')}")
                    
                # Handle target information for target signals
                if 'Target_Price' in row and row['Target_Price'] != 0:
                    st.write(f"**Target Price:** ${row['Target_Price']:.4f}")
                    if 'Target_Type' in row and row['Target_Type'] != 'Unknown':
                        st.write(f"**Target Type:** {row['Target_Type']}")
                
                if 'Next_Targets' in row and row['Next_Targets'] != 'N/A':
                    st.write(f"**Next Targets:** {row['Next_Targets']}")
                    
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

                # Handle exit prices for target signals
                if 'Exit_Prices' in row and row['Exit_Prices'] != 'N/A':
                    st.write(f"**Exit Prices:** {row['Exit_Prices']}")
                
                # Handle reference upmove/downmove for fractal track and outstanding signals
                reference_upmove = raw_data.get('Reference Upmove or Downmove start Date/Price($), end Date/Price($)', 'N/A')
                if reference_upmove and reference_upmove != 'N/A' and reference_upmove != 'No Information':
                    st.write(f"**Reference Upmove/Downmove:** {reference_upmove}")
                
                # Handle track level/price for fractal track and outstanding signals
                track_level_full = raw_data.get('Track Level/Price($), Price on Latest Trading day vs Track Level, Signal Type', 'N/A')
                
                if track_level_full and track_level_full != 'N/A' and track_level_full != 'No Information':
                    # Parse the track level data (format: "23.66% (Price: 73.2031), 5.8% above, Upmove Bounce Back")
                    try:
                        parts = track_level_full.split(', ')
                        if len(parts) >= 3:
                            track_level = parts[0].strip()  # "23.66% (Price: 73.2031)"
                            signal_type = parts[2].strip()  # "Upmove Bounce Back"
                            
                            st.write(f"**Track Level:** {track_level}")
                            st.write(f"**Signal Type:** {signal_type}")
                        else:
                            st.write(f"**Track Level/Price:** {track_level_full}")
                    except:
                        st.write(f"**Track Level/Price:** {track_level_full}")
                
                # Handle separate signal type field if it exists
                signal_type_separate = raw_data.get('Signal Type', 'N/A')
                if signal_type_separate and signal_type_separate != 'N/A' and signal_type_separate != 'No Information':
                    if 'Signal Type' not in track_level_full:  # Only show if not already displayed
                        st.write(f"**Signal Type:** {signal_type_separate}")
                


def load_stock_data_file(symbol, start_date, end_date, interval='Daily'):
    """Load and process stock data from CSV file for a given symbol"""
    import os
    
    # Map interval to pandas frequency
    INTERVAL_LETTER_DICT = {
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Yearly': 'Y'
    }
    
    # Construct the file path - now using CSV files
    csv_file_path = f'./trade_store/stock_data/{symbol}.csv'
    
    if not os.path.exists(csv_file_path):
        return None
    
    try:
        # Read CSV file
        # Assuming CSV format with columns: Date, Open, High, Low, Close, Volume
        df = pd.read_csv(csv_file_path)
        
        # Check if CSV is empty
        if df.empty:
            return None
        
        # Convert Date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'Date'}, inplace=True)
        else:
            st.error(f"No 'Date' column found in {csv_file_path}")
            return None
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Ensure we have required OHLC columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in {csv_file_path}: {missing_cols}")
            return None
        
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Convert to required interval if not Daily
        if interval != 'Daily':
            agg_dict = {
                'Open': 'first',
                'Close': 'last',
                'High': 'max',
                'Low': 'min'
            }
            
            # Add Volume to aggregation if it exists
            if 'Volume' in df.columns:
                agg_dict['Volume'] = 'sum'
            
            df = df.groupby(pd.Grouper(freq=INTERVAL_LETTER_DICT[interval])).agg(agg_dict)
            df = df.dropna()
        
        df.reset_index(inplace=True)
        return df
        
    except Exception as e:
        st.error(f"Error reading CSV file for {symbol}: {str(e)}")
        return None


def create_interactive_chart(row_data, raw_data):
    """Create an interactive candlestick chart for Fib-Ret data"""
    try:
        # Extract symbol from the data
        symbol_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', 'Unknown')
        if ',' in str(symbol_info):
            symbol = str(symbol_info).split(',')[0].strip().replace('"', '')
        else:
            symbol = "Unknown"
        
        # Extract signal information
        signal_date = None
        signal_price = None
        signal_type = None
        
        if 'Price:' in str(symbol_info):
            try:
                # Parse "CVS, Long, 2025-10-02 (Price: 77.45)"
                parts = str(symbol_info).split(',')
                if len(parts) >= 3:
                    signal_type = parts[1].strip()
                    date_price_part = parts[2].strip()
                    if '(' in date_price_part and ')' in date_price_part:
                        date_part = date_price_part.split('(')[0].strip()
                        price_part = date_price_part.split('(Price:')[1].replace(')', '').strip()
                        signal_date = date_part
                        signal_price = float(price_part)
            except:
                pass
        
        # Extract reference upmove information
        reference_upmove = raw_data.get('Reference Upmove or Downmove start Date/Price($), end Date/Price($)', 'N/A')
        upmove_start_date = None
        upmove_start_price = None
        upmove_end_date = None
        upmove_end_price = None
        
        if reference_upmove and reference_upmove != 'N/A' and reference_upmove != 'No Information':
            try:
                # Parse "2025-07-24 (Price: 58.5), 2025-10-02 (Price: 77.76)"
                if ',' in str(reference_upmove):
                    parts = str(reference_upmove).split(',')
                    if len(parts) >= 2:
                        start_part = parts[0].strip()
                        end_part = parts[1].strip()
                        
                        # Parse start date and price
                        if '(' in start_part and ')' in start_part:
                            start_date = start_part.split('(')[0].strip()
                            start_price = start_part.split('(Price:')[1].replace(')', '').strip()
                            upmove_start_date = start_date
                            upmove_start_price = float(start_price)
                        
                        # Parse end date and price
                        if '(' in end_part and ')' in end_part:
                            end_date = end_part.split('(')[0].strip()
                            end_price = end_part.split('(Price:')[1].replace(')', '').strip()
                            upmove_end_date = end_date
                            upmove_end_price = float(end_price)
            except:
                pass
        
        # Extract interval from the data
        interval_info = raw_data.get('Interval, Confirmation Status', 'Daily, Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = 'Daily'
        
        # Calculate date range: 23 candles back from upmove start date to today
        start_date_for_data = None
        end_date_for_data = datetime.now()
        
        if upmove_start_date:
            try:
                upmove_start_dt = datetime.strptime(upmove_start_date, '%Y-%m-%d')
                # Calculate 23 candles back based on interval
                if 'Daily' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=23)
                elif 'Weekly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(weeks=23)
                elif 'Monthly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=23*30)  # Approximate
                elif 'Quarterly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=23*90)  # Approximate
                elif 'Yearly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=23*365)  # Approximate
                else:
                    start_date_for_data = upmove_start_dt - timedelta(days=23)
            except:
                start_date_for_data = datetime.now() - timedelta(days=365)  # Default to 1 year back
        else:
            # If no upmove date, default to showing last 100 candles
            if 'Daily' in interval:
                start_date_for_data = datetime.now() - timedelta(days=100)
            elif 'Weekly' in interval:
                start_date_for_data = datetime.now() - timedelta(weeks=100)
            else:
                start_date_for_data = datetime.now() - timedelta(days=365)
        
        # Load real data from CSV file
        df_ohlc = load_stock_data_file(symbol, start_date_for_data, end_date_for_data, interval)
        
        if df_ohlc is None or df_ohlc.empty:
            st.warning(f"No CSV data available for {symbol}. Using mock data for demonstration.")
            # Fallback to mock data if CSV file not available
            dates = pd.date_range(start=start_date_for_data, end=end_date_for_data, freq='D')
            base_price = signal_price if signal_price else 50
            
            np.random.seed(42)
            price_data = []
            current_price = base_price
            
            for i, date in enumerate(dates):
                trend = 0.0001 * np.sin(i / 30)
                volatility = 0.02
                change = np.random.normal(trend, volatility)
                current_price *= (1 + change)
                
                daily_vol = abs(np.random.normal(0, 0.015))
                high = current_price * (1 + daily_vol)
                low = current_price * (1 - daily_vol)
                open_price = current_price * (1 + np.random.normal(0, 0.005))
                close_price = current_price * (1 + np.random.normal(0, 0.008))
                
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                price_data.append({
                    'Date': date,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price
                })
            
            df_ohlc = pd.DataFrame(price_data)
        
        # Create the candlestick chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df_ohlc['Date'],
            open=df_ohlc['Open'],
            high=df_ohlc['High'],
            low=df_ohlc['Low'],
            close=df_ohlc['Close'],
            name=symbol,
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Add reference upmove line if data is available
        if upmove_start_date and upmove_end_date and upmove_start_price and upmove_end_price:
            try:
                start_dt = datetime.strptime(upmove_start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(upmove_end_date, '%Y-%m-%d')
                
                fig.add_trace(go.Scatter(
                    x=[start_dt, end_dt],
                    y=[upmove_start_price, upmove_end_price],
                    mode='lines',
                    line=dict(dash='dash', color='blue', width=2),
                    name='Reference Upmove',
                    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
            except:
                pass
        
        # Add buy/sell signals
        if signal_date and signal_price:
            try:
                signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
                
                # Determine if it's a buy or sell signal
                if signal_type and 'Long' in signal_type:
                    marker_color = 'green'
                    marker_symbol = 'triangle-up'
                    signal_name = 'Buy Signal'
                else:
                    marker_color = 'red'
                    marker_symbol = 'triangle-down'
                    signal_name = 'Sell Signal'
                
                fig.add_trace(go.Scatter(
                    x=[signal_dt],
                    y=[signal_price],
                    mode='markers',
                    marker=dict(
                        color=marker_color,
                        size=15,
                        symbol=marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=signal_name,
                    hovertemplate=f'Signal: {signal_type}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
            except:
                pass
        
        # Update layout for better appearance and full width
        fig.update_layout(
            title=dict(
                text=f'{symbol} - Interactive Chart ({interval})',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1f77b4')
            ),
            xaxis=dict(
                title=dict(text='Date', font=dict(size=14)),
                tickfont=dict(size=11),
                gridcolor='#f0f0f0',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='#999999',
                spikethickness=1
            ),
            yaxis=dict(
                title=dict(text='Price ($)', font=dict(size=14)),
                tickfont=dict(size=11),
                gridcolor='#f0f0f0',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='#999999',
                spikethickness=1
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=700,
            margin=dict(l=50, r=10, t=100, b=60),
            plot_bgcolor='white',
            paper_bgcolor='#fafafa',
            xaxis_rangeslider_visible=False,
            autosize=True,
            width=None  # Remove any fixed width
        )
        
        # Display the chart with full width - use container width
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True  # Enable responsive behavior
            }
        )
        
    except Exception as e:
        st.error(f"Error creating interactive chart: {str(e)}")
        st.info(f"Chart shows data from 23 candles before the reference upmove start date to today. Interval: {interval if 'interval' in locals() else 'Daily'}")


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
    """Detect the structure and type of CSV file based on filename"""
    import os
    
    filename = os.path.basename(file_path)
    
    # Map filenames to their specific parsers
    file_mapping = {
        'bollinger_band.csv': 'bollinger_band',
        'Distance.csv': 'distance',
        'Fib-Ret.csv': 'fib_ret',
        'General-Divergence.csv': 'general_divergence',
        'new_high.csv': 'new_high',
        'Stochastic-Divergence.csv': 'stochastic_divergence',
        'sigma.csv': 'sigma',
        'sentiment.csv': 'sentiment',
        'Trendline.csv': 'trendline',
        'breadth.csv': 'breadth',
        'outstanding_signal.csv': 'outstanding_signal',
        'outstanding_exit_signal.csv': 'outstanding_exit_signal',
        'new_signal.csv': 'new_signal',
        'target_signal.csv': 'target_signal',
        'latest_performance.csv': 'latest_performance',
        'forward_backtesting.csv': 'forward_backtesting'
    }
    
    return file_mapping.get(filename, 'unknown')

# Individual CSV Parsers

def parse_bollinger_band(df):
    """Parse bollinger_band.csv"""
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
        
        processed_data.append({
            'Function': 'Band Matrix',
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_distance(df):
    """Parse Distance.csv"""
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
            'Function': 'DeltaDrift',
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_fib_ret(df):
    """Parse Fib-Ret.csv"""
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
            'Function': 'Fractal Track',
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_general_divergence(df):
    """Parse General-Divergence.csv"""
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
            'Function': 'BaselineDiverge',
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

# Continue with remaining parsers - I'll add them in a more efficient way
def parse_new_high(df):
    """Parse new_high.csv"""
    return parse_signal_csv(df, 'Altitude Alpha')

def parse_stochastic_divergence(df):
    """Parse Stochastic-Divergence.csv"""
    return parse_signal_csv(df, 'Oscillator Delta')

def parse_sigma(df):
    """Parse sigma.csv"""
    return parse_signal_csv(df, 'SigmaShell')

def parse_sentiment(df):
    """Parse sentiment.csv with specific handling for quoted first column"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Parse symbol and signal info - handle quoted first column
        symbol_info = row.get('Symbol, Signal, Signal Date/Price[$]', '')
        # Remove quotes and parse
        symbol_info_clean = str(symbol_info).strip('"')
        symbol_match = re.search(r'([^,]+),\s*([^,]+),\s*([^(]+)\(Price:\s*([^)]+)\)', symbol_info_clean)
        
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
            'Function': 'PulseGauge',
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_trendline(df):
    """Parse Trendline.csv"""
    return parse_signal_csv(df, 'TrendPulse')

def parse_outstanding_signal(df):
    """Parse outstanding_signal.csv"""
    return parse_detailed_signal_csv(df)

def parse_outstanding_exit_signal(df):
    """Parse outstanding_exit_signal.csv"""
    return parse_detailed_signal_csv(df)

def parse_new_signal(df):
    """Parse new_signal.csv"""
    return parse_detailed_signal_csv(df)

def parse_latest_performance(df):
    """Parse latest_performance.csv"""
    return parse_performance_csv(df)

def parse_forward_backtesting(df):
    """Parse forward_backtesting.csv"""
    return parse_performance_csv(df)

def parse_breadth(df):
    """Parse breadth.csv"""
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

# Helper functions for common parsing patterns
def parse_signal_csv(df, function_name):
    """Parse signal CSV files with common structure"""
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
            'Function': function_name,
            'Symbol': symbol,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_detailed_signal_csv(df):
    """Parse detailed signal CSV files with Function column"""
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
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Exit_Status': row.get('Exit Signal Date/Price[$]', 'N/A'),
            'Current_MTM': row.get('Current Mark to Market and Holding Period', 'N/A'),
            'Confirmation_Status': row.get('Interval, Confirmation Status', 'N/A'),
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

def parse_performance_csv(df):
    """Parse performance CSV files"""
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


def parse_target_signals(df, page_name="Unknown"):
    """Parse target signals CSV (target_signal.csv)"""
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
        win_rate_info = row.get('Number of Trades/Historic Win Rate [%]', '')
        win_rate_match = re.search(r'([0-9]+)/([0-9.]+)%', str(win_rate_info))
        
        if win_rate_match:
            try:
                num_trades = int(win_rate_match.group(1))
                win_rate = float(win_rate_match.group(2))
            except:
                win_rate, num_trades = 0, 0
        else:
            win_rate, num_trades = 0, 0
        
        # Parse current trading date and price
        current_info = row.get('Current Trading Date/Price[$]', '')
        current_match = re.search(r'([^(]+)\(Price:\s*([^)]+)\)', str(current_info))
        if current_match:
            current_date = current_match.group(1).strip()
            try:
                current_price = float(current_match.group(2).strip())
            except:
                current_price = 0
        else:
            current_date, current_price = "Unknown", 0
        
        # Parse entry signal date and price
        entry_info = row.get('Entry Signal Date/Price[$]', '')
        entry_match = re.search(r'([^(]+)\(Price:\s*([^)]+)\)', str(entry_info))
        
        if entry_match:
            entry_date = entry_match.group(1).strip()
            try:
                entry_price = float(entry_match.group(2).strip())
            except:
                entry_price = 0
        else:
            entry_date, entry_price = "Unknown", 0
        
        # Parse gain and holding period
        gain_info = row.get('% Gain, Holding Period (days)', '')
        gain_match = re.search(r'([0-9.]+)%,\s*([0-9]+)\s*days', str(gain_info))
        
        if gain_match:
            try:
                gain_pct = float(gain_match.group(1))
                holding_days = int(gain_match.group(2))
            except:
                gain_pct, holding_days = 0, 0
        else:
            gain_pct, holding_days = 0, 0
        
        # Parse backtested returns
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
        
        # Parse interval and function
        interval = row.get('Interval', 'Unknown')
        function = row.get('Function', 'Unknown')
        
        # Parse target information
        target_info = row.get('Target for which Price has achieved over 90 percent of gain %', '')
        target_price = 0
        target_type = "Unknown"
        
        if '(' in str(target_info) and ')' in str(target_info):
            # Extract price and type from format like "0.8118 (Historic Rise or Fall to Pivot)"
            target_match = re.search(r'([0-9.]+)\s*\(([^)]+)\)', str(target_info))
            if target_match:
                try:
                    target_price = float(target_match.group(1))
                    target_type = target_match.group(2).strip()
                except:
                    target_price, target_type = 0, "Unknown"
        
        # Parse next targets
        next_targets = row.get('Next Two Target % from Latest Trading Price', 'N/A')
        
        # Parse remaining potential exit prices
        exit_prices = row.get('Remaining Potential Exit Prices [$]', 'N/A')
        
        # Calculate performance metrics (simplified)
        strategy_cagr = 0
        buy_hold_cagr = 0
        strategy_sharpe = 0
        buy_hold_sharpe = 0
        
        # Try to extract from performance data if available
        performance_info = row.get('Latest Past 6 Months Performance[%]/No. of Analysed Trades/Avg Holding Period (days) (Across ALL Assets)', '')
        if performance_info and '/' in str(performance_info):
            try:
                perf_parts = str(performance_info).split('/')
                if len(perf_parts) >= 1:
                    # Use performance percentage as a rough CAGR estimate
                    strategy_cagr = float(perf_parts[0].replace('%', ''))
            except:
                pass
        processed_data.append({
            'Symbol': symbol,
            'Function': function,
            'Signal_Type': signal_type,
            'Signal_Date': signal_date,
            'Signal_Price': signal_price,
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Current_Date': current_date,
            'Current_Price': current_price,
            'Gain_Percentage': gain_pct,
            'Holding_Days': holding_days,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Best_Return': best_return,
            'Worst_Return': worst_return,
            'Avg_Return': avg_return,
            'Target_Price': target_price,
            'Target_Type': target_type,
            'Next_Targets': next_targets,
            'Exit_Prices': exit_prices,
            'Interval': interval,
            'Strategy_CAGR': strategy_cagr,
            'Buy_Hold_CAGR': buy_hold_cagr,
            'Strategy_Sharpe': strategy_sharpe,
            'Buy_Hold_Sharpe': buy_hold_sharpe,
            'Raw_Data': row.to_dict()
        })
    
    return pd.DataFrame(processed_data)

@st.cache_data
def load_data_from_file(file_path, page_name="Unknown"):
    """Load and process trading data from any CSV file with specific parsers"""
    try:
        # Detect CSV structure
        csv_type = detect_csv_structure(file_path)
        
        # Load the full CSV with special handling for sentiment.csv
        if 'sentiment.csv' in file_path:
            # Handle sentiment.csv with complex column names - use manual parsing
            try:
                import csv
                import io
                
                # Read the file and manually parse it
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use csv.Sniffer to detect the dialect
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(content[:1000])
                
                # Parse the CSV manually
                reader = csv.reader(io.StringIO(content), dialect=dialect)
                rows = list(reader)
                
                if len(rows) < 2:
                    st.warning(f"No data found in {file_path}")
                    return pd.DataFrame()
                
                # Create DataFrame from parsed rows
                df = pd.DataFrame(rows[1:], columns=rows[0])
                
            except Exception as e:
                st.error(f"Error parsing sentiment.csv: {str(e)}")
                return pd.DataFrame()
        else:
            df = pd.read_csv(file_path)
        
        if df.empty:
            st.warning(f"No data found in {file_path}")
            return pd.DataFrame()
        
        # Parse based on detected structure using specific parsers
        parser_mapping = {
            'bollinger_band': parse_bollinger_band,
            'distance': parse_distance,
            'fib_ret': parse_fib_ret,
            'general_divergence': parse_general_divergence,
            'new_high': parse_new_high,
            'stochastic_divergence': parse_stochastic_divergence,
            'sigma': parse_sigma,
            'sentiment': parse_sentiment,
            'trendline': parse_trendline,
            'breadth': parse_breadth,
            'outstanding_signal': parse_outstanding_signal,
            'outstanding_exit_signal': parse_outstanding_exit_signal,
            'new_signal': parse_new_signal,
            'target_signal': parse_target_signals,
            'latest_performance': parse_latest_performance,
            'forward_backtesting': parse_forward_backtesting
        }
        
        if csv_type in parser_mapping:
            return parser_mapping[csv_type](df)
        else:
            # Fallback to basic parsing for unknown structures
            st.warning(f"Unknown CSV structure for {file_path}, using basic parsing")
            return parse_signal_csv(df, page_name)
            
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
            create_strategy_cards(filtered_df, page_title)
            
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
        'bollinger_band.csv': 'Band Matrix',
        'Distance.csv': 'DeltaDrift',
        'Fib-Ret.csv': 'Fractal Track',
        'General-Divergence.csv': 'BaselineDiverge',
        'new_high.csv': 'Altitude Alpha',
        'Stochastic-Divergence.csv': 'Oscillator Delta',
        'sigma.csv': 'SigmaShell',
        'sentiment.csv': 'PulseGauge',
        'Trendline.csv': 'TrendPulse',
        'breadth.csv': 'Signal Breadth Indicator (SBI)',
        'outstanding_signal.csv': 'Outstanding Signals',
        'outstanding_exit_signal.csv': 'Outstanding Signals Exit',
        'new_signal.csv': 'New Signals',
        'latest_performance.csv': 'Latest Performance',
        'forward_backtesting.csv': 'Forward Testing Performance',
        'target_signal.csv': 'Outstanding Target'
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
            file_mapping[filename] = file_path
        
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