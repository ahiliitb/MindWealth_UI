"""
Interactive chart components for trading strategies
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from ..utils.data_loader import load_stock_data_file


def calculate_retracement_levels(ret_high_low, high_or_low_diff, is_uptrend):
    """
    Calculate Fibonacci retracement levels
    
    Args:
        ret_high_low: Reference high (for uptrend) or low (for downtrend)
        high_or_low_diff: Difference between high and low
        is_uptrend: True for uptrend (Long), False for downtrend (Short)
    
    Returns:
        Dictionary with level percentages as keys and price levels as values
    """
    levels = [23.6, 38.2, 50, 61.8, 78.6, 88.6]
    retracement_prices = {}
    
    if is_uptrend:
        # For uptrend (Long): retracement goes down from the high
        for level in levels:
            price_level = ret_high_low - (level / 100) * high_or_low_diff
            retracement_prices[level] = price_level
    else:
        # For downtrend (Short): retracement goes up from the low
        for level in levels:
            price_level = ret_high_low + (level / 100) * high_or_low_diff
            retracement_prices[level] = price_level
    
    return retracement_prices


def create_horizontal_chart(symbol: str, interval: str, horizontal_price: float):
    """Create a simple candlestick chart with a horizontal reference line.

    Uses `load_stock_data_file` to fetch OHLC for the symbol and interval.
    """
    try:
        # Determine back window based on interval, mirroring the referenced logic
        # Daily: 4 years; Weekly: 15 years; Otherwise: full history (IPO to date)
        end_date_for_data = datetime.now()
        start_date_for_data = None
        if interval and 'Daily' in interval:
            start_date_for_data = end_date_for_data - timedelta(days=365 * 4)
        elif interval and 'Weekly' in interval:
            start_date_for_data = end_date_for_data - timedelta(days=365 * 15)
        else:
            # Load full history by passing None (approximation for IPO date)
            start_date_for_data = None

        df = load_stock_data_file(symbol, start_date_for_data, end_date_for_data, interval if interval else 'Daily')
        if df is None or df.empty:
            st.warning(f"No price data available for {symbol} ({interval}).")
            return

        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='green',
            decreasing_line_color='red'
        ))

        # Add horizontal reference line
        if horizontal_price is not None:
            try:
                horizontal_price_float = float(horizontal_price)
                # Use Scatter for better control over the horizontal line
                fig.add_trace(go.Scatter(
                    x=[df['Date'].min(), df['Date'].max()],
                    y=[horizontal_price_float, horizontal_price_float],
                    mode='lines',
                    line=dict(color='orange', width=3, dash='dash'),
                    name=f'Horizontal: {horizontal_price_float:.4f}',
                    hovertemplate=f'Horizontal Level: ${horizontal_price_float:.4f}<extra></extra>'
                ))
            except (ValueError, TypeError):
                pass

        # Update layout for better appearance (similar to other charts)
        fig.update_layout(
            title=dict(
                text=f'{symbol} - Horizontal Analysis ({interval})',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1f77b4')
            ),
            xaxis=dict(
                title=dict(text='Date', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                tickangle=0,
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='#999999',
                spikethickness=1
            ),
            yaxis=dict(
                title=dict(text='Price ($)', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True,
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
            margin=dict(l=65, r=20, t=90, b=70),
            plot_bgcolor='white',
            paper_bgcolor='#fafafa',
            xaxis_rangeslider_visible=False,
            autosize=True,
            width=None
        )

        # Display the chart with full width
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
        )
    except Exception as e:
        st.error(f"Error rendering horizontal chart for {symbol}: {str(e)}")

def create_interactive_chart(row_data, raw_data, exit_date=None, exit_price=None, exit_signal_type=None):
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
        
        # Auto-extract exit date and price from CSV if not provided (universal exit marker support)
        if not exit_date or not exit_price:
            exit_info = raw_data.get('Exit Signal Date/Price[$]', 'No Exit Yet')
            if exit_info and str(exit_info) != 'No Exit Yet':
                # Parse exit date and price (format: "DATE (Price: X)" or "DATE (Price: X) (Today)")
                exit_str = str(exit_info)
                if '(Price:' in exit_str:
                    try:
                        parts = exit_str.split('(Price:')
                        exit_date = parts[0].strip()
                        # Extract price - handle both "X)" and "X) (Today)"
                        price_part = parts[1].split(')')[0].strip()
                        exit_price = float(price_part)
                        exit_signal_type = signal_type  # Use same signal type
                    except:
                        pass
        
        # Extract reference upmove information (for Fractal Track)
        reference_upmove = raw_data.get('Reference Upmove or Downmove start Date/Price($), end Date/Price($)', 'N/A')
        upmove_start_date = None
        upmove_start_price = None
        upmove_end_date = None
        upmove_end_price = None
        
        # Check if this is TrendPulse data
        is_trendpulse = False
        trendpulse_start_end = raw_data.get('TrendPulse Start/End (Date and Price($))', 'N/A')
        if trendpulse_start_end and trendpulse_start_end != 'N/A':
            is_trendpulse = True
            reference_upmove = trendpulse_start_end  # Use TrendPulse data instead
        
        if reference_upmove and reference_upmove != 'N/A' and reference_upmove != 'No Information':
            try:
                # TrendPulse format: "2020-10-31 (Price: 126.9444)/2025-10-08 (Price: 221.2778)"
                # Fractal Track format: "2025-07-24 (Price: 58.5), 2025-10-02 (Price: 77.76)"
                
                # Determine separator based on format
                separator = '/' if '/' in str(reference_upmove) and ')/' in str(reference_upmove) else ','
                
                if separator in str(reference_upmove):
                    parts = str(reference_upmove).split(separator)
                    if len(parts) >= 2:
                        start_part = parts[0].strip()
                        end_part = parts[1].strip()
                        
                        # Parse start date and price
                        if '(' in start_part and ')' in start_part:
                            start_date = start_part.split('(')[0].strip()
                            # Handle both "Price:" and "Price " formats
                            if 'Price:' in start_part:
                                start_price = start_part.split('(Price:')[1].replace(')', '').strip()
                            else:
                                start_price = start_part.split('(Price ')[1].replace(')', '').strip()
                            upmove_start_date = start_date
                            upmove_start_price = float(start_price)
                        
                        # Parse end date and price
                        if '(' in end_part and ')' in end_part:
                            end_date = end_part.split('(')[0].strip()
                            # Handle both "Price:" and "Price " formats
                            if 'Price:' in end_part:
                                end_price = end_part.split('(Price:')[1].replace(')', '').strip()
                            else:
                                end_price = end_part.split('(Price ')[1].replace(')', '').strip()
                            upmove_end_date = end_date
                            upmove_end_price = float(end_price)
                
            except Exception as e:
                pass
        
        # Extract interval from the data
        interval_info = raw_data.get('Interval, Confirmation Status', 'Daily, Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = 'Daily'
        
        # Calculate date range: 200 candles back from upmove start date to today
        start_date_for_data = None
        end_date_for_data = datetime.now()
        
        if upmove_start_date:
            try:
                upmove_start_dt = datetime.strptime(upmove_start_date, '%Y-%m-%d')
                # Calculate 200 candles back based on interval
                if 'Daily' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=200)
                elif 'Weekly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(weeks=200)
                elif 'Monthly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=200*30)  # Approximate
                elif 'Quarterly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=200*90)  # Approximate
                elif 'Yearly' in interval:
                    start_date_for_data = upmove_start_dt - timedelta(days=200*365)  # Approximate
                else:
                    start_date_for_data = upmove_start_dt - timedelta(days=200)
            except:
                start_date_for_data = datetime.now() - timedelta(days=365)  # Default to 1 year back
        else:
            # If no upmove date, default to showing last 200 candles
            if 'Daily' in interval:
                start_date_for_data = datetime.now() - timedelta(days=200)
            elif 'Weekly' in interval:
                start_date_for_data = datetime.now() - timedelta(weeks=200)
            elif 'Monthly' in interval:
                start_date_for_data = datetime.now() - timedelta(days=200*30)  # Approximate
            elif 'Quarterly' in interval:
                start_date_for_data = datetime.now() - timedelta(days=200*90)  # Approximate
            elif 'Yearly' in interval:
                start_date_for_data = datetime.now() - timedelta(days=200*365)  # Approximate
            else:
                start_date_for_data = datetime.now() - timedelta(days=365)
        
        # Load real data from CSV file
        df_ohlc = load_stock_data_file(symbol, start_date_for_data, end_date_for_data, interval)
        
        if df_ohlc is None or df_ohlc.empty:
            st.error(f"❌ No price data available for {symbol}. Please ensure the stock data file exists in trade_store/stock_data/{symbol}.csv")
            st.info("💡 To resolve this issue:\n"
                   f"1. Check if the file `trade_store/stock_data/{symbol}.csv` exists\n"
                   "2. Ensure the file contains valid OHLC data with Date, Open, High, Low, Close columns\n"
                   "3. Verify the date range requested is within the available data")
            return
        
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
                
                # Determine line style based on chart type
                if is_trendpulse:
                    line_style = dict(dash='solid', color='blue', width=4)  # Bold solid line for TrendPulse
                    line_name = 'TrendPulse Line'
                else:
                    line_style = dict(dash='dash', color='blue', width=2)  # Dashed line for Fractal Track
                    line_name = 'Reference Upmove'
                
                fig.add_trace(go.Scatter(
                    x=[start_dt, end_dt],
                    y=[upmove_start_price, upmove_end_price],
                    mode='lines',
                    line=line_style,
                    name=line_name,
                    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Calculate and add Fibonacci retracement levels (only for Fractal Track, not TrendPulse)
                is_uptrend = signal_type and 'Long' in signal_type if signal_type else True
                
                # Only add retracement levels for Fractal Track
                if not is_trendpulse:
                    high_or_low_diff = abs(upmove_end_price - upmove_start_price)
                    
                    # Determine the reference high/low based on trend
                    if is_uptrend:
                        ret_high_low = upmove_end_price  # High point for uptrend
                    else:
                        ret_high_low = upmove_start_price  # Low point for downtrend
                    
                    # Calculate retracement levels
                    retracement_levels = calculate_retracement_levels(ret_high_low, high_or_low_diff, is_uptrend)
                    
                    # Define colors for different retracement levels
                    level_colors = {
                        23.6: '#00FF00',  # Bright green
                        38.2: '#32CD32',  # Lime green
                        50.0: '#FFD700',  # Gold
                        61.8: '#FFA500',  # Orange
                        78.6: '#FF4500',  # Orange red
                        88.6: '#FF0000'   # Red
                    }
                    
                    # Get the date range for horizontal lines
                    if not df_ohlc.empty:
                        chart_start_date = df_ohlc['Date'].min()
                        chart_end_date = df_ohlc['Date'].max()
                    else:
                        chart_start_date = start_dt
                        chart_end_date = datetime.now()
                    
                    # Add horizontal lines for each retracement level
                    for level, price in retracement_levels.items():
                        fig.add_trace(go.Scatter(
                            x=[chart_start_date, chart_end_date],
                            y=[price, price],
                            mode='lines',
                            line=dict(
                                dash='dot',
                                color=level_colors.get(level, '#808080'),
                                width=1.5
                            ),
                            name=f'Fib {level}%',
                            hovertemplate=f'Fibonacci {level}%<br>Price: $%{{y:.2f}}<extra></extra>',
                            showlegend=True
                        ))
                        
                        # Add text annotation for the level
                        fig.add_annotation(
                            x=chart_end_date,
                            y=price,
                            text=f'{level}%',
                            showarrow=False,
                            xanchor='left',
                            font=dict(size=10, color=level_colors.get(level, '#808080')),
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor=level_colors.get(level, '#808080'),
                            borderwidth=1
                        )
                
            except Exception as e:
                st.warning(f"Could not calculate retracement levels: {str(e)}")
        
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
                        size=23 if (exit_date and exit_price) else 30,
                        symbol=marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=signal_name,
                    hovertemplate=f'Signal: {signal_type}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
                
            except Exception as e:
                pass
        
        # Add EXIT marker if provided (for Outstanding Exit Signals)
        if exit_date and exit_price:
            try:
                exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
                
                # Exit marker is opposite color of entry
                if exit_signal_type and 'Long' in exit_signal_type:
                    exit_marker_color = 'red'
                    exit_marker_symbol = 'triangle-down'
                else:
                    exit_marker_color = 'green'
                    exit_marker_symbol = 'triangle-up'
                
                fig.add_trace(go.Scatter(
                    x=[exit_dt],
                    y=[exit_price],
                    mode='markers',
                    marker=dict(
                        color=exit_marker_color,
                        size=23,
                        symbol=exit_marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name='Exit',
                    hovertemplate=f'Exit<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
                
            except Exception as e:
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
                title=dict(text='Date', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                tickangle=0,
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='#999999',
                spikethickness=1
            ),
            yaxis=dict(
                title=dict(text='Price ($)', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True,
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
            margin=dict(l=65, r=20, t=90, b=70),
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
        st.info(f"Chart shows data from 200 candles before the reference upmove start date to today. Interval: {interval if 'interval' in locals() else 'Daily'}")


def create_divergence_chart(row, raw_data, exit_date=None, exit_price=None, exit_signal_type=None):
    """
    Create a specialized chart for divergence analysis (Stochastic Divergence, General Divergence)
    Shows divergence lines and signal points with different colors for long/short signals
    """
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
        
        # Auto-extract exit date and price from CSV if not provided (universal exit marker support)
        if not exit_date or not exit_price:
            exit_info = raw_data.get('Exit Signal Date/Price[$]', 'No Exit Yet')
            if exit_info and str(exit_info) != 'No Exit Yet':
                # Parse exit date and price (format: "DATE (Price: X)" or "DATE (Price: X) (Today)")
                exit_str = str(exit_info)
                if '(Price:' in exit_str:
                    try:
                        parts = exit_str.split('(Price:')
                        exit_date = parts[0].strip()
                        # Extract price - handle both "X)" and "X) (Today)"
                        price_part = parts[1].split(')')[0].strip()
                        exit_price = float(price_part)
                        exit_signal_type = signal_type  # Use same signal type
                    except:
                        pass
        
        # Extract divergence start/end information from processed data
        # The data has been processed by the parser, so we need to access the raw CSV data
        divergence_start_date = None
        divergence_start_price = None
        divergence_end_date = None
        divergence_end_price = None
        
        # Try to get divergence info from the raw data
        if raw_data:
            # Look for divergence info in the raw CSV row
            divergence_info = None
            for key, value in raw_data.items():
                if 'Divergence Start/End' in str(key):
                    divergence_info = str(value)
                    break
            
            if divergence_info and '/' in divergence_info:
                try:
                    # Parse divergence start/end info - format: "2025-09-10 (Price: 54.37)/2025-09-30 (Price: 55.49)"
                    parts = divergence_info.split('/')
                    if len(parts) >= 2:
                        start_part = parts[0].strip()
                        end_part = parts[1].strip()
                        
                        # Extract start date and price
                        if '(' in start_part and ')' in start_part:
                            start_date = start_part.split('(')[0].strip()
                            start_price_str = start_part.split('(')[1].split(')')[0].replace('Price: ', '')
                            divergence_start_date = start_date
                            divergence_start_price = float(start_price_str)
                        
                        # Extract end date and price
                        if '(' in end_part and ')' in end_part:
                            end_date = end_part.split('(')[0].strip()
                            end_price_str = end_part.split('(')[1].split(')')[0].replace('Price: ', '')
                            divergence_end_date = end_date
                            divergence_end_price = float(end_price_str)
                                                        
                except Exception as e:
                    st.warning(f"Could not parse divergence data: {e}")
                    st.info(f"Raw divergence info: {divergence_info}")
            else:
                st.warning("No divergence information found in the data")
                st.info(f"Available keys: {list(raw_data.keys()) if raw_data else 'No raw data'}")
        else:
            st.warning("No raw data available for divergence extraction")
        
        # Load real stock data from CSV files (same as Fractal Track/Fib-Ret)
        from datetime import datetime, timedelta
        
        # Calculate date range: 200 candles back from divergence start date to today
        start_date_for_data = None
        end_date_for_data = datetime.now()
        
        if divergence_start_date:
            try:
                divergence_start_dt = datetime.strptime(divergence_start_date, '%Y-%m-%d')
                # Calculate 200 days back from divergence start date
                start_date_for_data = divergence_start_dt - timedelta(days=200)
            except:
                start_date_for_data = datetime.now() - timedelta(days=365)  # Default to 1 year back
        else:
            # If no divergence date, default to showing last 200 days
            start_date_for_data = datetime.now() - timedelta(days=200)
        
        # Load real data from CSV file (same as other charts)
        df_ohlc = load_stock_data_file(symbol, start_date_for_data, end_date_for_data, 'Daily')
        
        if df_ohlc is None or df_ohlc.empty:
            st.error(f"❌ No price data available for {symbol}. Please ensure the stock data file exists in trade_store/stock_data/{symbol}.csv")
            st.info("💡 To resolve this issue:\n"
                   f"1. Check if the file `trade_store/stock_data/{symbol}.csv` exists\n"
                   "2. Ensure the file contains valid OHLC data with Date, Open, High, Low, Close columns\n"
                   "3. Verify the date range requested is within the available data")
            return
        
        # Create the chart
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
        
        # Add divergence line if we have the data
        if divergence_start_date and divergence_end_date and divergence_start_price and divergence_end_price:
            try:
                start_dt = datetime.strptime(divergence_start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(divergence_end_date, '%Y-%m-%d')
                
                # Determine line color based on signal type
                line_color = 'blue' if 'Long' in signal_type else 'red'
                line_width = 4  # Bold line
                
                fig.add_trace(go.Scatter(
                    x=[start_dt, end_dt],
                    y=[divergence_start_price, divergence_end_price],
                    mode='lines',
                    line=dict(color=line_color, width=line_width),
                    name=f'Divergence Line ({signal_type})',
                    hovertemplate=f'Divergence: {signal_type}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
            except:
                pass
        
        # Add signal point
        if signal_date and signal_price:
            try:
                # Extract signal date
                signal_date_str = signal_date.split('(')[0].strip()
                signal_dt = datetime.strptime(signal_date_str, '%Y-%m-%d')
                
                # Determine marker color and symbol based on signal type
                if 'Long' in signal_type:
                    marker_color = 'green'
                    marker_symbol = 'triangle-up'
                else:
                    marker_color = 'red'
                    marker_symbol = 'triangle-down'
                
                fig.add_trace(go.Scatter(
                    x=[signal_dt],
                    y=[signal_price],
                    mode='markers',
                    marker=dict(
                        size=23 if (exit_date and exit_price) else 30,
                        color=marker_color,
                        symbol=marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=f'Signal Point ({signal_type})',
                    hovertemplate=f'Signal: {signal_type}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
            except:
                pass
        
        # Add EXIT marker if available (universal support)
        if exit_date and exit_price:
            try:
                exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
                
                # Exit marker is opposite color of entry
                if exit_signal_type and 'Long' in exit_signal_type:
                    exit_marker_color = 'red'
                    exit_marker_symbol = 'triangle-down'
                else:
                    exit_marker_color = 'green'
                    exit_marker_symbol = 'triangle-up'
                
                fig.add_trace(go.Scatter(
                    x=[exit_dt],
                    y=[exit_price],
                    mode='markers',
                    marker=dict(
                        size=23,
                        color=exit_marker_color,
                        symbol=exit_marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name='Exit',
                    hovertemplate=f'Exit<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
            except:
                pass
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} - Divergence Analysis ({signal_type})',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1f77b4')
            ),
            xaxis=dict(
                title=dict(text='Date', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                tickangle=0,
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True
            ),
            yaxis=dict(
                title=dict(text='Price ($)', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True
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
            margin=dict(l=65, r=20, t=90, b=70),
            plot_bgcolor='white',
            paper_bgcolor='#fafafa',
            xaxis_rangeslider_visible=False,
            autosize=True,
            width=None
        )
        
        # Display the chart
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
        )
        
    except Exception as e:
        st.error(f"Error creating divergence chart: {str(e)}")
        st.info("Divergence chart shows the divergence pattern and signal points for analysis.")


def create_bollinger_band_chart(row, raw_data, exit_date=None, exit_price=None, exit_signal_type=None):
    """
    Create a specialized chart for Bollinger Bands analysis
    Shows Bollinger Bands (20-period) with buy/sell markers
    """
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
                # Parse "GOOG, Long, 2025-10-08 (Price: 247.13)"
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
        
        # Auto-extract exit date and price from CSV if not provided (universal exit marker support)
        if not exit_date or not exit_price:
            exit_info = raw_data.get('Exit Signal Date/Price[$]', 'No Exit Yet')
            if exit_info and str(exit_info) != 'No Exit Yet':
                # Parse exit date and price (format: "DATE (Price: X)" or "DATE (Price: X) (Today)")
                exit_str = str(exit_info)
                if '(Price:' in exit_str:
                    try:
                        parts = exit_str.split('(Price:')
                        exit_date = parts[0].strip()
                        # Extract price - handle both "X)" and "X) (Today)"
                        price_part = parts[1].split(')')[0].strip()
                        exit_price = float(price_part)
                        exit_signal_type = signal_type  # Use same signal type
                    except:
                        pass
        
        # Extract interval from the data
        interval_info = raw_data.get('Interval, Confirmation Status', 'Daily, Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = 'Daily'
        
        # Calculate date range: 200 candles back from signal date to today
        start_date_for_data = None
        end_date_for_data = datetime.now()
        
        if signal_date:
            try:
                signal_dt = datetime.strptime(signal_date, '%Y-%m-%d')
                # Calculate 200 candles back based on interval
                if 'Daily' in interval:
                    start_date_for_data = signal_dt - timedelta(days=200)
                elif 'Weekly' in interval:
                    start_date_for_data = signal_dt - timedelta(weeks=200)
                elif 'Monthly' in interval:
                    start_date_for_data = signal_dt - timedelta(days=200*30)
                elif 'Quarterly' in interval:
                    start_date_for_data = signal_dt - timedelta(days=200*90)
                elif 'Yearly' in interval:
                    start_date_for_data = signal_dt - timedelta(days=200*365)
                else:
                    start_date_for_data = signal_dt - timedelta(days=200)
            except:
                start_date_for_data = datetime.now() - timedelta(days=365)
        else:
            start_date_for_data = datetime.now() - timedelta(days=200)
        
        # Load real data from CSV file
        df_ohlc = load_stock_data_file(symbol, start_date_for_data, end_date_for_data, interval)
        
        if df_ohlc is None or df_ohlc.empty:
            st.warning(f"No CSV data available for {symbol}. Cannot display Bollinger Bands chart.")
            return
        
        # Calculate Bollinger Bands (20-period, 2 standard deviations)
        period = 20
        df_ohlc['SMA'] = df_ohlc['Close'].rolling(window=period).mean()
        df_ohlc['STD'] = df_ohlc['Close'].rolling(window=period).std()
        df_ohlc['Upper_Band'] = df_ohlc['SMA'] + (2 * df_ohlc['STD'])
        df_ohlc['Lower_Band'] = df_ohlc['SMA'] - (2 * df_ohlc['STD'])
        
        # Create the chart
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
        
        # Add Bollinger Bands
        # Upper Band
        fig.add_trace(go.Scatter(
            x=df_ohlc['Date'],
            y=df_ohlc['Upper_Band'],
            mode='lines',
            line=dict(color='rgba(250, 128, 114, 0.8)', width=2),
            name='Upper Band (20, 2σ)',
            hovertemplate='Upper Band<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Middle Band (SMA)
        fig.add_trace(go.Scatter(
            x=df_ohlc['Date'],
            y=df_ohlc['SMA'],
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0.8)', width=2, dash='dot'),
            name='Middle Band (SMA 20)',
            hovertemplate='SMA 20<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Lower Band
        fig.add_trace(go.Scatter(
            x=df_ohlc['Date'],
            y=df_ohlc['Lower_Band'],
            mode='lines',
            line=dict(color='rgba(135, 206, 250, 0.8)', width=2),
            name='Lower Band (20, 2σ)',
            hovertemplate='Lower Band<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add buy/sell signal marker
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
                        size=23 if (exit_date and exit_price) else 30,
                        symbol=marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=signal_name,
                    hovertemplate=f'Signal: {signal_type}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
            except Exception as e:
                pass
        
        # Add EXIT marker if provided (for Outstanding Exit Signals)
        if exit_date and exit_price:
            try:
                exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
                
                # Exit marker is opposite color of entry
                if exit_signal_type and 'Long' in exit_signal_type:
                    exit_marker_color = 'red'
                    exit_marker_symbol = 'triangle-down'
                else:
                    exit_marker_color = 'green'
                    exit_marker_symbol = 'triangle-up'
                
                fig.add_trace(go.Scatter(
                    x=[exit_dt],
                    y=[exit_price],
                    mode='markers',
                    marker=dict(
                        color=exit_marker_color,
                        size=23,
                        symbol=exit_marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name='Exit',
                    hovertemplate=f'Exit<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ))
            except Exception as e:
                pass
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{symbol} - Bollinger Bands Chart ({interval}) - Period: 20, StdDev: 2',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1f77b4')
            ),
            xaxis=dict(
                title=dict(text='Date', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                tickangle=0,
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='#999999',
                spikethickness=1
            ),
            yaxis=dict(
                title=dict(text='Price ($)', font=dict(size=16, color='#111')),
                tickfont=dict(size=13, color='#111'),
                gridcolor='#e9e9e9',
                gridwidth=1,
                showgrid=True,
                zeroline=False,
                showline=True,
                linewidth=1.5,
                linecolor='#333',
                mirror=True,
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
            margin=dict(l=65, r=20, t=90, b=70),
            plot_bgcolor='white',
            paper_bgcolor='#fafafa',
            xaxis_rangeslider_visible=False,
            autosize=True,
            width=None
        )
        
        # Display the chart
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
        )
        
    except Exception as e:
        st.error(f"Error creating Bollinger Bands chart: {str(e)}")
        st.info("Bollinger Bands chart shows price action with 20-period Bollinger Bands overlay.")


def fetch_original_signal_data(function, symbol, signal_date, interval, signal_type):
    """
    Fetch original signal data from individual strategy CSV files
    Used by Outstanding Signals page to get the complete signal data
    """
    import pandas as pd
    import csv
    import os
    import glob
    from datetime import datetime
    
    # Map function names to CSV file names (case-insensitive)
    function_to_file = {
        'FRACTAL TRACK': 'Fib-Ret.csv',
        'BAND MATRIX': 'bollinger_band.csv',
        'DELTADRIFT': 'Distance.csv',
        'BASELINEDIVERGE': 'General-Divergence.csv',
        'ALTITUDE ALPHA': 'new_high.csv',
        'OSCILLATOR DELTA': 'Stochastic-Divergence.csv',
        'SIGMASHELL': 'sigma.csv',
        'PULSEGAUGE': 'sentiment.csv',
        'TRENDPULSE': 'Trendline.csv'
    }
    
    # Normalize function name to uppercase for matching
    function_upper = str(function).upper().strip()
    csv_file = function_to_file.get(function_upper)
    if not csv_file:
        return None
    
    # Get the project root directory (two levels up from src/components/charts.py)
    # __file__ = /path/to/project/src/components/charts.py
    # dirname(__file__) = /path/to/project/src/components
    # dirname(dirname(__file__)) = /path/to/project/src
    # dirname(dirname(dirname(__file__))) = /path/to/project (project root)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))  # Go up 2 levels: components -> src -> project root
    trade_store_dir = os.path.join(project_root, 'trade_store', 'US')
    
    # Handle date-prefixed files: search for both dated and non-dated versions
    # Pattern: YYYY-MM-DD_Fib-Ret.csv or Fib-Ret.csv
    non_dated_path = os.path.join(trade_store_dir, csv_file)  # Non-dated
    dated_pattern = os.path.join(trade_store_dir, f'*_{csv_file}')  # Dated pattern
    
    # Find all matching files
    matching_files = []
    
    # Check for non-dated version
    if os.path.exists(non_dated_path):
        matching_files.append((None, non_dated_path))  # (date, filepath)
    
    # Find dated versions using glob pattern
    for file_path in glob.glob(dated_pattern):
        filename = os.path.basename(file_path)
        # Extract date from filename (YYYY-MM-DD_filename.csv)
        # Verify the suffix matches our target file (case-insensitive)
        if filename.lower().endswith(f'_{csv_file.lower()}'):
            date_match = filename.split('_', 1)
            if len(date_match) == 2:
                try:
                    file_date = datetime.strptime(date_match[0], '%Y-%m-%d')
                    matching_files.append((file_date, file_path))
                except ValueError:
                    pass
    
    # Prefer the most recent dated version, fallback to non-dated
    if not matching_files:
        st.error(f"File not found: {csv_file} in {trade_store_dir}")
        return None
    
    # Sort: dated files first (most recent first), then non-dated files
    # Separate dated and non-dated files
    dated_files = [(date, path) for date, path in matching_files if date is not None]
    non_dated_files = [(date, path) for date, path in matching_files if date is None]
    
    # Sort dated files by date descending (most recent first)
    dated_files.sort(key=lambda x: x[0], reverse=True)
    
    # Combine: dated files first, then non-dated
    sorted_files = dated_files + non_dated_files
    file_path = sorted_files[0][1]
    
    # Ensure file_path is absolute
    file_path = os.path.abspath(file_path)
    
    try:
        # Read the CSV file
        with open(file_path, 'r', encoding='utf-8') as f:
            # Use csv.Sniffer to detect dialect
            sample = f.read(4096)
            f.seek(0)
            
            # Read CSV
            df = pd.read_csv(f, skipinitialspace=True)
        
        # Search for matching signal
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            
            # Get symbol info from the row
            symbol_info = row_dict.get('Symbol, Signal, Signal Date/Price[$]', '')
            
            # Check if this row matches our criteria
            if symbol in str(symbol_info) and signal_date in str(symbol_info):
                # Also check signal type and interval if available
                if signal_type in str(symbol_info):
                    interval_info = row_dict.get('Interval, Confirmation Status', '')
                    if interval in str(interval_info) or interval == 'Daily':
                        # Found matching signal, return the raw data
                        return row_dict
        
        return None
    except Exception as e:
        st.error(f"Error fetching original signal data: {str(e)}")
        return None


def create_outstanding_signal_chart(row, raw_data):
    """
    Create chart for Outstanding Signals page by fetching original signal data
    """
    try:
        # Extract function, symbol, signal date, interval, and signal type from outstanding signals data
        function = raw_data.get('Function', 'Unknown')
        symbol_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', '')
        
        # Parse symbol info
        symbol = None
        signal_type = None
        signal_date = None
        
        if ',' in str(symbol_info):
            parts = str(symbol_info).split(',')
            if len(parts) >= 3:
                symbol = parts[0].strip().replace('"', '')
                signal_type = parts[1].strip()
                date_part = parts[2].strip()
                if '(' in date_part:
                    signal_date = date_part.split('(')[0].strip()
        
        # Extract interval
        interval_info = raw_data.get('Interval, Confirmation Status', 'Daily, Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = 'Daily'
        
        # Fetch original signal data from individual CSV
        original_data = fetch_original_signal_data(function, symbol, signal_date, interval, signal_type)
        
        if original_data:
            
            # Create a dummy row with the necessary fields for the chart functions
            # The chart functions expect row.get() to work, so we'll create a compatible structure
            original_row = pd.Series(original_data)
            
            # Route to appropriate chart based on function (using uppercase for matching)
            function_upper = str(function).upper().strip()
            
            if function_upper == 'OSCILLATOR DELTA':
                create_divergence_chart(original_row, original_data)
            elif function_upper == 'BAND MATRIX':
                create_bollinger_band_chart(original_row, original_data)
            elif function_upper in ['FRACTAL TRACK']:
                create_interactive_chart(original_row, original_data)
            elif function_upper == 'TRENDPULSE':
                create_interactive_chart(original_row, original_data)
            else:
                # For all other functions, use simple interactive chart
                create_interactive_chart(original_row, original_data)
        else:
            # Fallback to using outstanding signals data
            create_interactive_chart(row, raw_data)
        
    except Exception as e:
        st.error(f"Error creating outstanding signal chart: {str(e)}")
        # Fallback to simple chart
        create_interactive_chart(row, raw_data)


def create_outstanding_exit_signal_chart(row, raw_data):
    """
    Create chart for Outstanding Exit Signals page by fetching original signal data
    Extension of create_outstanding_signal_chart with exit marker added
    """
    try:
        # Extract function, symbol, signal date, interval, and signal type from outstanding exit signals data
        function = raw_data.get('Function', 'Unknown')
        symbol_info = raw_data.get('Symbol, Signal, Signal Date/Price[$]', '')
        
        # Parse symbol info (same as Outstanding Signals)
        symbol = None
        signal_type = None
        signal_date = None
        
        if ',' in str(symbol_info):
            parts = str(symbol_info).split(',')
            if len(parts) >= 3:
                symbol = parts[0].strip().replace('"', '')
                signal_type = parts[1].strip()
                date_part = parts[2].strip()
                if '(' in date_part:
                    signal_date = date_part.split('(')[0].strip()
        
        # Extract interval (same as Outstanding Signals)
        interval_info = raw_data.get('Interval, Confirmation Status', 'Daily, Unknown')
        if ',' in str(interval_info):
            interval = str(interval_info).split(',')[0].strip()
        else:
            interval = 'Daily'
        
        # Extract exit date and price from Outstanding Exit Signal CSV
        exit_info = raw_data.get('Exit Signal Date/Price[$]', 'No Exit Yet')
        exit_date = None
        exit_price = None
        
        if exit_info and str(exit_info) != 'No Exit Yet':
            # Parse exit date and price (format: "DATE (Price: X)" or "DATE (Price: X) (Today)")
            exit_str = str(exit_info)
            if '(Price:' in exit_str:
                parts = exit_str.split('(Price:')
                exit_date = parts[0].strip()
                # Extract price - handle both "X)" and "X) (Today)"
                price_part = parts[1].split(')')[0].strip()
                try:
                    exit_price = float(price_part)
                except:
                    pass
        
        # Fetch original signal data from individual CSV (same as Outstanding Signals)
        original_data = fetch_original_signal_data(function, symbol, signal_date, interval, signal_type)
        
        if original_data:
            # Create a Series for compatibility with chart functions
            original_row = pd.Series(original_data)
            
            # Add exit marker to the chart by modifying the figure after creation
            # We'll need to modify the existing chart functions to accept optional exit parameters
            # For now, route to appropriate chart and then add exit marker info
            
            function_upper = str(function).upper().strip()
            
            # Display exit information at the top
            if exit_date and exit_price:
                st.markdown(f"**🔴 Exit Signal:** {exit_date} @ ${exit_price:.4f}")
                
                # Calculate profit/loss if we have signal price
                signal_price = original_data.get('Signal_Price', None)
                if signal_price:
                    if signal_type == 'Long':
                        profit_pct = ((exit_price - signal_price) / signal_price) * 100
                    else:
                        profit_pct = ((signal_price - exit_price) / signal_price) * 100
                    profit_color = "green" if profit_pct > 0 else "red"
                    st.markdown(f"**Profit/Loss:** <span style='color: {profit_color}; font-weight: bold;'>{profit_pct:+.2f}%</span>", unsafe_allow_html=True)
            
            # Route to appropriate chart based on function (same as Outstanding Signals)
            if function_upper == 'OSCILLATOR DELTA':
                create_divergence_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type)
            elif function_upper == 'BAND MATRIX':
                create_bollinger_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type)
            elif function_upper in ['FRACTAL TRACK']:
                create_interactive_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type)
            elif function_upper == 'TRENDPULSE':
                create_interactive_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type)
            else:
                # For all other functions, use simple interactive chart with exit
                create_interactive_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type)
        else:
            # Fallback
            st.warning(f"Could not fetch original signal data for {function} - {symbol}")
            
    except Exception as e:
        st.error(f"Error creating outstanding exit signal chart: {str(e)}")


def create_divergence_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type):
    """Extension of create_divergence_chart with exit marker on chart"""
    # We need to recreate the divergence chart logic but add exit marker before displaying
    # For simplicity, call base chart and then add annotation about exit
    # The proper way would be to modify the figure, but since it's already displayed, we show exit info
    
    # Import here to avoid circular dependency
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    
    try:
        # Extract data (same logic as create_divergence_chart)
        symbol_info = original_data.get('Symbol, Signal, Signal Date/Price[$]', 'Unknown')
        if ',' in str(symbol_info):
            symbol = str(symbol_info).split(',')[0].strip().replace('"', '')
        else:
            symbol = "Unknown"
        
        signal_date = None
        signal_price = None
        signal_type_parsed = signal_type
        
        if 'Price:' in str(symbol_info):
            try:
                parts = str(symbol_info).split(',')
                if len(parts) >= 3:
                    signal_type_parsed = parts[1].strip()
                    date_price_part = parts[2].strip()
                    if '(' in date_price_part and ')' in date_price_part:
                        date_part = date_price_part.split('(')[0].strip()
                        price_part = date_price_part.split('(Price:')[1].replace(')', '').strip()
                        signal_date = date_part
                        signal_price = float(price_part)
            except:
                pass
        
        # Extract divergence info
        divergence_start_date = None
        divergence_start_price = None
        divergence_end_date = None
        divergence_end_price = None
        
        if original_data:
            divergence_info = None
            for key, value in original_data.items():
                if 'Divergence Start/End' in str(key):
                    divergence_info = str(value)
                    break
            
            if divergence_info and '/' in divergence_info:
                try:
                    parts = divergence_info.split('/')
                    if len(parts) >= 2:
                        start_part = parts[0].strip()
                        end_part = parts[1].strip()
                        
                        if '(' in start_part and ')' in start_part:
                            start_date = start_part.split('(')[0].strip()
                            start_price_str = start_part.split('(')[1].split(')')[0].replace('Price: ', '')
                            divergence_start_date = start_date
                            divergence_start_price = float(start_price_str)
                        
                        if '(' in end_part and ')' in end_part:
                            end_date = end_part.split('(')[0].strip()
                            end_price_str = end_part.split('(')[1].split(')')[0].replace('Price: ', '')
                            divergence_end_date = end_date
                            divergence_end_price = float(end_price_str)
                except:
                    pass
        
        # Load stock data
        start_date_for_data = None
        end_date_for_data = datetime.now()
        
        if divergence_start_date:
            try:
                divergence_start_dt = datetime.strptime(divergence_start_date, '%Y-%m-%d')
                start_date_for_data = divergence_start_dt - timedelta(days=200)
            except:
                start_date_for_data = datetime.now() - timedelta(days=365)
        else:
            start_date_for_data = datetime.now() - timedelta(days=200)
        
        df_ohlc = load_stock_data_file(symbol, start_date_for_data, end_date_for_data, 'Daily')
        
        if df_ohlc is None or df_ohlc.empty:
            st.warning(f"No data available for {symbol}")
            return
        
        # Create the chart
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df_ohlc['Date'] if 'Date' in df_ohlc.columns else df_ohlc.index,
            open=df_ohlc['Open'],
            high=df_ohlc['High'],
            low=df_ohlc['Low'],
            close=df_ohlc['Close'],
            name=symbol,
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Add divergence line
        if divergence_start_date and divergence_end_date and divergence_start_price and divergence_end_price:
            try:
                start_dt = datetime.strptime(divergence_start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(divergence_end_date, '%Y-%m-%d')
                
                line_color = 'blue' if 'Long' in signal_type_parsed else 'red'
                
                fig.add_trace(go.Scatter(
                    x=[start_dt, end_dt],
                    y=[divergence_start_price, divergence_end_price],
                    mode='lines',
                    line=dict(color=line_color, width=4),
                    name=f'Divergence Line ({signal_type_parsed})'
                ))
            except:
                pass
        
        # Add entry signal marker
        if signal_date and signal_price:
            try:
                signal_date_str = signal_date.split('(')[0].strip()
                signal_dt = datetime.strptime(signal_date_str, '%Y-%m-%d')
                
                marker_color = 'green' if 'Long' in signal_type_parsed else 'red'
                marker_symbol = 'triangle-up' if 'Long' in signal_type_parsed else 'triangle-down'
                
                fig.add_trace(go.Scatter(
                    x=[signal_dt],
                    y=[signal_price],
                    mode='markers',
                    marker=dict(
                        size=23,
                        color=marker_color,
                        symbol=marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=f'Entry ({signal_type_parsed})'
                ))
            except:
                pass
        
        # Add EXIT marker (NEW!)
        if exit_date and exit_price:
            try:
                exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
                
                # Exit marker is opposite color of entry
                exit_marker_color = 'red' if 'Long' in signal_type_parsed else 'green'
                exit_marker_symbol = 'triangle-down' if 'Long' in signal_type_parsed else 'triangle-up'
                
                fig.add_trace(go.Scatter(
                    x=[exit_dt],
                    y=[exit_price],
                    mode='markers',
                    marker=dict(
                        size=23,
                        color=exit_marker_color,
                        symbol=exit_marker_symbol,
                        line=dict(width=2, color='white')
                    ),
                    name='Exit'
                ))
            except:
                pass
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Divergence Analysis with Exit',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating divergence chart with exit: {str(e)}")
        # Fallback to base chart
        create_divergence_chart(original_row, original_data)


def create_bollinger_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type):
    """Extension of create_bollinger_band_chart with exit marker on chart"""
    # Call base chart WITH exit parameters to add exit marker on the chart
    create_bollinger_band_chart(original_row, original_data, exit_date, exit_price, signal_type)


def create_interactive_chart_with_exit(original_row, original_data, exit_date, exit_price, signal_type):
    """Extension of create_interactive_chart with exit marker on chart"""
    # Call base chart WITH exit parameters to add exit marker on the chart
    create_interactive_chart(original_row, original_data, exit_date, exit_price, signal_type)

