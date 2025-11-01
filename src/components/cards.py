"""
UI Card Components for displaying trading strategy information
"""

from curses import raw
import streamlit as st
import pandas as pd

from .charts import create_interactive_chart, create_divergence_chart, create_bollinger_band_chart, create_outstanding_signal_chart, create_outstanding_exit_signal_chart


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


def create_strategy_cards(df, page_name="Unknown", tab_context=""):
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
        display_strategy_cards_page(df, page_name, tab_context)
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
                # Add pagination context to make keys unique across pagination tabs
                pagination_context = f"{tab_context}_page{i}"

                display_strategy_cards_page(page_df, page_name, pagination_context)


def display_strategy_cards_page(df, page_name="Unknown", tab_context=""):
    """Display strategy cards for a given page with scrollable container"""
    if len(df) == 0:
        st.warning("No data to display on this page.")
        return

    # Add custom CSS for scrollable container
    st.markdown("""
    <style>
    /* Custom scrollbar styling for strategy cards */
    .stContainer {
        max-height: 70vh;
        overflow-y: auto;
        overflow-x: hidden;
    }
    .stContainer::-webkit-scrollbar {
        width: 12px;
    }
    .stContainer::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
        margin: 5px;
    }
    .stContainer::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    .stContainer::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create scrollable container for cards
    with st.container(height=1000, border=True):
        # Display strategy cards in scrollable area
        for card_num, (idx, row) in enumerate(df.iterrows()):
            # Get raw data for extracting expander info
            raw_data = row['Raw_Data']

            # Extract interval
            interval_display = "Unknown"
            if 'Interval' in row and row['Interval'] != 'Unknown':
                interval_display = row['Interval']
                if interval_display == 'Unknown':
                    interval_display = raw_data.get("Interval", "Unknown")

            else:
                interval_info = raw_data.get('Interval, Confirmation Status', 'Unknown')
                if ',' in str(interval_info):
                    interval_display = str(interval_info).split(',')[0].strip()
                else:
                    interval_display = str(interval_info).strip()

            
            # Extract signal type (Long/Short)
            signal_type_display = "Unknown"
            if 'Signal_Type' in row and row['Signal_Type'] != 'Unknown':
                signal_type_display = row['Signal_Type']
            else:
                signal_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', '')
                if 'Long' in str(signal_info):
                    signal_type_display = 'Long'
                elif 'Short' in str(signal_info):
                    signal_type_display = 'Short'
            
            # Extract signal date
            signal_date_display = "Unknown"
            if 'Signal_Date' in row and row['Signal_Date'] != 'Unknown':
                signal_date_display = row['Signal_Date']
            else:
                signal_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', '')
                if 'Price:' in str(signal_info):
                    parts = str(signal_info).split(',')
                    if len(parts) >= 3:
                        date_part = parts[2].strip().split('(')[0].strip()
                        signal_date_display = date_part
            
            # Create expandable card with new title format
            expander_title = f"🔍 {row['Function']} - {row['Symbol']} | {interval_display} | {signal_type_display} | {signal_date_display}"
            
            with st.expander(expander_title, expanded=False):
                st.markdown("**📋 Key Trade Information**")

                # Add interactive chart button for all strategy pages with functions
                show_chart = False
                # List of pages that should have charts
                chart_enabled_pages = [
                    "Band Matrix", "DeltaDrift", "Fractal Track", "BaselineDiverge",
                    "Altitude Alpha", "Oscillator Delta", "SigmaShell", "PulseGauge",
                    "TrendPulse", "Outstanding Signals", "Outstanding Signals Exit", "New Signals"
                ]
                
                if page_name in chart_enabled_pages:
                    show_chart = True
                elif 'Function' in row and row['Function'] == 'FRACTAL TRACK':
                    show_chart = True
                
                if show_chart:
                    # Use a unique key with hash of all row data PLUS tab context to ensure absolute uniqueness
                    # tab_context includes the interval tab (ALL/Daily/Weekly) making keys unique across tabs
                    import hashlib
                    # Create a unique identifier from multiple data points INCLUDING tab context
                    unique_str = f"{page_name}_{tab_context}_{card_num}_{row['Symbol']}_{signal_date_display}_{interval_display}_{signal_type_display}_{idx}"
                    # Add hash to make it even more unique (in case of exact duplicates)
                    unique_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
                    chart_key = f"chart_{unique_hash}_{card_num}"
                    if st.button(f"📊 View Interactive Chart", key=chart_key):
                        # Route to appropriate chart based on page type
                        if page_name == 'Outstanding Signals Exit':
                            # Fetch original signal data and display chart with exit marker
                            create_outstanding_exit_signal_chart(row, raw_data)
                        elif page_name in ['Outstanding Signals', 'New Signals']:
                            # For New Signals and Outstanding Signals, fetch original data from function CSVs
                            create_outstanding_signal_chart(row, raw_data)
                        elif page_name == 'Oscillator Delta':
                            # Divergence chart with divergence line
                            create_divergence_chart(row, raw_data)
                        elif page_name == 'Band Matrix':
                            # Bollinger Bands chart with BB overlay
                            create_bollinger_band_chart(row, raw_data)
                        elif page_name in ['Fractal Track', 'TrendPulse']:
                            # Interactive chart with reference lines (Fibonacci for Fractal Track, TrendPulse line for TrendPulse)
                            create_interactive_chart(row, raw_data)
                        else:
                            # Simple candlestick chart with buy/sell marker for all other pages
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
                    
                    # Add divergence information for Oscillator Delta and Outstanding Signals
                    if page_name in ['Oscillator Delta', 'Outstanding Signals']:
                        divergence_info = raw_data.get('Divergence Start/End (Date and Price [$])', '')
                        if divergence_info and '/' in str(divergence_info):
                            try:
                                parts = str(divergence_info).split('/')
                                if len(parts) >= 2:
                                    start_part = parts[0].strip()
                                    end_part = parts[1].strip()
                                    
                                    # Extract start date and price
                                    if '(' in start_part and ')' in start_part:
                                        start_date = start_part.split('(')[0].strip()
                                        start_price = start_part.split('(')[1].split(')')[0].replace('Price: ', '')
                                        st.write(f"**Divergence Start:** {start_date} (${start_price})")
                                    
                                    # Extract end date and price
                                    if '(' in end_part and ')' in end_part:
                                        end_date = end_part.split('(')[0].strip()
                                        end_price = end_part.split('(')[1].split(')')[0].replace('Price: ', '')
                                        st.write(f"**Divergence End:** {end_date} (${end_price})")
                            except:
                                pass
                    
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


