#!/usr/bin/env python3
"""Test script to generate breadth indicator plot locally without sending email."""

import pandas as pd
import plotly.graph_objects as go
from constant import BREADTH_SIGNAL_STORE_CSV_PATH_US

def create_breadth_indicator_plot_from_csv(csv_path: str):
    """Create a single Plotly figure with two lines from breadth CSV.

    - The CSV has two types of rows:
      1. Data rows with percentages (first column has %, second and third have data)
      2. Function name rows (like "All Function Combined")
    - We need to find sections between "All Function Combined" markers
    - Title: "Bullish Signal Breadth Indicator (SBI) plot for All function"
    """
    try:
        # Read CSV with headers, don't use any column as index
        df = pd.read_csv(csv_path, index_col=False)
        
        print(f"Total rows in CSV: {len(df)}")
        
        # Filter to keep ONLY "All Function Combined" rows
        data_rows = df[df['Function'] == 'All Function Combined'].copy()
        
        if len(data_rows) == 0:
            print("No 'All Function Combined' rows found in CSV")
            return None
        
        print(f"Found {len(data_rows)} 'All Function Combined' observations")
        
        # Convert percentage strings like "12.34%" -> float 12.34
        def pct_to_float(val):
            if pd.isna(val) or val == '' or val == ' ':
                return 0.0
            s = str(val).strip().replace('%', '').replace(',', '')
            if s == '' or s == 'nan':
                return 0.0
            try:
                return float(s)
            except:
                return 0.0
        
        # Extract data from the correct columns
        y1 = [pct_to_float(val) for val in data_rows['Bullish Asset vs Total Asset (%)'].tolist()]
        y2 = [pct_to_float(val) for val in data_rows['Bullish Signal vs Total Signal (%)'].tolist()]
        
        # Use dates as x-axis if Date column exists and has non-empty values
        if 'Date' in data_rows.columns:
            dates = data_rows['Date'].tolist()
            # Check if we have any actual dates (non-empty values)
            has_dates = any(str(d).strip() != '' and not pd.isna(d) for d in dates)
            if has_dates:
                # Create x-axis mixing observation numbers for empty dates and actual dates
                x = []
                obs_counter = 1
                for d in dates:
                    if str(d).strip() == '' or pd.isna(d):
                        x.append(f"Obs {obs_counter}")
                        obs_counter += 1
                    else:
                        x.append(str(d))
            else:
                # All dates are empty, use observation numbers
                x = list(range(1, len(data_rows) + 1))
        else:
            x = list(range(1, len(data_rows) + 1))
        
        print(f"\nPlot data summary:")
        print(f"Number of points: {len(x)}")
        print(f"Y1 range: [{min(y1):.2f}, {max(y1):.2f}]")
        print(f"Y2 range: [{min(y2):.2f}, {max(y2):.2f}]")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y1, 
            mode='lines', 
            name='Bullish Asset vs Total Asset (%)',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y2, 
            mode='lines', 
            name='Bullish Signal vs Total Signal (%)',
            line=dict(color='#ff7f0e', width=2)
        ))

        # Determine x-axis title based on whether we have dates or not
        if 'Date' in data_rows.columns:
            dates = data_rows['Date'].tolist()
            has_dates = any(str(d).strip() != '' and not pd.isna(d) for d in dates)
            xaxis_title = 'Date / Observation' if has_dates else 'Observation'
        else:
            xaxis_title = 'Observation'
        
        fig.update_layout(
            title='Bullish Signal Breadth Indicator (SBI) plot for All function',
            xaxis_title=xaxis_title,
            yaxis_title='Percentage (%)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=70, b=40, r=20, l=60),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )

        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print(f"Reading CSV from: {BREADTH_SIGNAL_STORE_CSV_PATH_US}")
    
    # Create the plot
    fig = create_breadth_indicator_plot_from_csv(BREADTH_SIGNAL_STORE_CSV_PATH_US)
    
    if fig:
        print("\n✓ Plot created successfully!")
        
        # Save as PNG file
        output_png = "breadth_indicator_test_plot.png"
        try:
            fig.write_image(output_png, width=1200, height=600)
            print(f"✓ Plot saved to: {output_png}")
        except Exception as e:
            print(f"Error saving PNG: {e}")
            print("Note: You may need to install kaleido: pip install kaleido")
    else:
        print("\n✗ Failed to create plot")

