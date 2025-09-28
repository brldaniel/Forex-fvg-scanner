# requirements.txt
"""
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
yfinance==0.2.18
plotly==5.15.0
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Forex FVG Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FVGScanner:
    def __init__(self):
        self.major_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'
        ]
        self.timeframes = {
            'Monthly': '1mo',
            'Weekly': '1wk', 
            'Daily': '1d',
            '4Hour': '60m'
        }
        self.pair_names = {
            'EURUSD=X': 'EUR/USD',
            'GBPUSD=X': 'GBP/USD', 
            'USDJPY=X': 'USD/JPY',
            'USDCHF=X': 'USD/CHF',
            'AUDUSD=X': 'AUD/USD',
            'USDCAD=X': 'USD/CAD',
            'NZDUSD=X': 'NZD/USD'
        }
        
    def detect_fvg(self, df, lookback_periods=100):
        """Detect Fair Value Gaps (3-candle pattern)"""
        fvgs = []
        
        for i in range(2, min(len(df), lookback_periods)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1] 
            candle3 = df.iloc[i]
            
            candle1_high = max(candle1['Open'], candle1['Close'])
            candle1_low = min(candle1['Open'], candle1['Close'])
            candle3_high = max(candle3['Open'], candle3['Close'])
            candle3_low = min(candle3['Open'], candle3['Close'])
            
            # Bullish FVG: Candle3 low > Candle1 high
            if candle3_low > candle1_high:
                fvg_high = candle3_low
                fvg_low = candle1_high
                fvg_type = "Bullish"
                fvgs.append({
                    'date': df.index[i],
                    'type': fvg_type,
                    'high': fvg_high,
                    'low': fvg_low,
                    'size_pips': abs(fvg_high - fvg_low) * 10000,
                    'candle_position': i
                })
            
            # Bearish FVG: Candle3 high < Candle1 low  
            elif candle3_high < candle1_low:
                fvg_high = candle1_low
                fvg_low = candle3_high
                fvg_type = "Bearish"
                fvgs.append({
                    'date': df.index[i],
                    'type': fvg_type,
                    'high': fvg_high,
                    'low': fvg_low,
                    'size_pips': abs(fvg_high - fvg_low) * 10000,
                    'candle_position': i
                })
        
        return fvgs
    
    def get_price_data(self, symbol, timeframe, periods=100):
        """Fetch price data for given symbol and timeframe"""
        try:
            ticker = yf.Ticker(symbol)
            
            if timeframe == '1mo':
                data = ticker.history(period='5y', interval=timeframe)
            elif timeframe == '1wk':
                data = ticker.history(period='2y', interval=timeframe)
            elif timeframe == '1d':
                data = ticker.history(period='1y', interval=timeframe)
            else:  # 60m
                data = ticker.history(period='60d', interval=timeframe)
            
            return data.tail(periods)
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def scan_pair(self, pair):
        """Scan a single pair across timeframes"""
        for tf_name, tf_code in self.timeframes.items():
            data = self.get_price_data(pair, tf_code)
            if data is None or len(data) < 3:
                continue
                
            fvgs = self.detect_fvg(data)
            
            if fvgs:
                latest_fvg = max(fvgs, key=lambda x: x['date'])
                
                result = {
                    'pair': self.pair_names.get(pair, pair),
                    'symbol': pair,
                    'timeframe': tf_name,
                    'fvg_type': latest_fvg['type'],
                    'date_detected': latest_fvg['date'].strftime('%Y-%m-%d'),
                    'fvg_high': round(latest_fvg['high'], 5),
                    'fvg_low': round(latest_fvg['low'], 5),
                    'size_pips': round(latest_fvg['size_pips'], 1),
                    'age_candles': len(data) - latest_fvg['candle_position']
                }
                return result
        
        return None

    def get_detailed_analysis(self, pair, timeframe):
        """Get detailed FVG analysis for a specific pair and timeframe"""
        tf_code = self.timeframes[timeframe]
        data = self.get_price_data(pair, tf_code, periods=200)
        
        if data is None or len(data) < 3:
            return None, None
            
        fvgs = self.detect_fvg(data, lookback_periods=200)
        return data, fvgs

def create_fvg_chart(data, fvgs, pair_name, timeframe):
    """Create an interactive candlestick chart with FVG highlighting"""
    fig = go.Figure()
    
    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add FVG zones
    for i, fvg in enumerate(fvgs[-5:]):  # Show last 5 FVGs
        color = 'rgba(0, 255, 0, 0.3)' if fvg['type'] == 'Bullish' else 'rgba(255, 0, 0, 0.3)'
        
        fig.add_trace(go.Scatter(
            x=[fvg['date'], fvg['date']],
            y=[fvg['low'], fvg['high']],
            mode='lines',
            line=dict(width=15, color=color),
            name=f"{fvg['type']} FVG",
            hoverinfo='text',
            hovertext=f"{fvg['type']} FVG<br>High: {fvg['high']:.5f}<br>Low: {fvg['low']:.5f}<br>Size: {fvg['size_pips']:.1f} pips"
        ))
    
    fig.update_layout(
        title=f"{pair_name} - {timeframe} Timeframe with FVG Zones",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=500,
        showlegend=True
    )
    
    return fig

def main():
    # Initialize scanner
    scanner = FVGScanner()
    
    # Sidebar
    st.sidebar.title("🎯 FVG Scanner Settings")
    
    # Pair selection
    selected_pairs = st.sidebar.multiselect(
        "Select Forex Pairs:",
        options=scanner.major_pairs,
        default=scanner.major_pairs,
        format_func=lambda x: scanner.pair_names.get(x, x)
    )
    
    # Custom pair input
    st.sidebar.subheader("Add Custom Pair")
    custom_pair = st.sidebar.text_input("Enter symbol (e.g., EURGBP=X):")
    if st.sidebar.button("Add Pair") and custom_pair:
        if custom_pair not in scanner.major_pairs:
            scanner.major_pairs.append(custom_pair)
            scanner.pair_names[custom_pair] = custom_pair.replace('=X', '')
            st.sidebar.success(f"Added {custom_pair}")
    
    # Lookback settings
    lookback = st.sidebar.slider("Lookback Period (candles):", 50, 200, 100)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About FVG")
    st.sidebar.info(
        "**Fair Value Gap (FVG)** is a 3-candle pattern where:\n"
        "- **Bullish**: Candle3 low > Candle1 high\n"
        "- **Bearish**: Candle3 high < Candle1 low\n"
        "The app scans from Monthly → Weekly → Daily → 4H timeframes."
    )
    
    # Main content
    st.title("📊 Forex Fair Value Gap Scanner")
    st.markdown("Scanning for 3-candle Fair Value Gaps across major forex pairs")
    
    # Scan button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start FVG Scan", type="primary", use_container_width=True):
            with st.spinner("Scanning forex pairs..."):
                results = []
                progress_bar = st.progress(0)
                
                for i, pair in enumerate(selected_pairs):
                    result = scanner.scan_pair(pair)
                    if result:
                        results.append(result)
                    progress_bar.progress((i + 1) / len(selected_pairs))
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} FVG patterns!")
                    
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(results)
                    
                    # Color coding for FVG types
                    def color_fvg_type(val):
                        color = 'lightgreen' if val == 'Bullish' else 'lightcoral'
                        return f'background-color: {color}'
                    
                    # Display results table
                    display_df = df[[
                        'pair', 'timeframe', 'fvg_type', 'date_detected', 
                        'fvg_high', 'fvg_low', 'size_pips', 'age_candles'
                    ]].copy()
                    
                    st.subheader("🎯 FVG Scan Results")
                    styled_df = display_df.style.applymap(color_fvg_type, subset=['fvg_type'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total FVGs Found", len(results))
                    with col2:
                        bullish_count = len(df[df['fvg_type'] == 'Bullish'])
                        st.metric("Bullish FVGs", bullish_count)
                    with col3:
                        bearish_count = len(df[df['fvg_type'] == 'Bearish'])
                        st.metric("Bearish FVGs", bearish_count)
                    with col4:
                        avg_size = df['size_pips'].mean()
                        st.metric("Avg Size (pips)", f"{avg_size:.1f}")
                    
                    # Detailed analysis section
                    st.subheader("🔍 Detailed Analysis")
                    
                    selected_pair = st.selectbox(
                        "Select pair for detailed analysis:",
                        options=df['symbol'].unique(),
                        format_func=lambda x: scanner.pair_names.get(x, x)
                    )
                    
                    selected_timeframe = st.selectbox(
                        "Select timeframe:",
                        options=list(scanner.timeframes.keys())
                    )
                    
                    if st.button("Show Detailed Chart"):
                        with st.spinner("Generating detailed analysis..."):
                            data, fvgs = scanner.get_detailed_analysis(selected_pair, selected_timeframe)
                            
                            if data is not None and fvgs:
                                st.success(f"Found {len(fvgs)} FVG patterns on {selected_timeframe}")
                                
                                # Display chart
                                pair_name = scanner.pair_names.get(selected_pair, selected_pair)
                                fig = create_fvg_chart(data, fvgs, pair_name, selected_timeframe)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # FVG details table
                                st.subheader("FVG Details")
                                fvg_details = []
                                for fvg in fvgs[-10:]:  # Last 10 FVGs
                                    fvg_details.append({
                                        'Date': fvg['date'].strftime('%Y-%m-%d'),
                                        'Type': fvg['type'],
                                        'High': fvg['high'],
                                        'Low': fvg['low'],
                                        'Size (pips)': fvg['size_pips'],
                                        'Bars Ago': len(data) - fvg['candle_position']
                                    })
                                
                                fvg_df = pd.DataFrame(fvg_details)
                                st.dataframe(fvg_df.style.applymap(color_fvg_type, subset=['Type']), 
                                           use_container_width=True)
                            else:
                                st.warning(f"No FVG patterns found for {selected_pair} on {selected_timeframe}")
                    
                    # Export results
                    st.subheader("💾 Export Results")
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"fvg_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    st.warning("No FVG patterns found in the selected pairs!")
        
        # Instructions
        st.markdown("---")
        st.subheader("🎯 How to Use")
        st.markdown("""
        1. **Select pairs** to scan in the sidebar
        2. **Click 'Start FVG Scan'** to run the analysis
        3. **View results** in the table below
        4. **Use detailed analysis** to see charts for specific pairs
        5. **Download results** as CSV for further analysis
        
        **Timeframe Hierarchy**: Monthly → Weekly → Daily → 4H
        """)

if __name__ == "__main__":
    main()
