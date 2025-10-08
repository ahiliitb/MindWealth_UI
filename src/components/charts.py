"""
Interactive chart components for trading strategies
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from ..utils.data_loader import load_stock_data_file


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

