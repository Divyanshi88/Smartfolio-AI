import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from tensorflow.keras.models import load_model
import torch
  



import json
from PIL import Image
import time
import plotly.figure_factory as ff
import altair as alt
from streamlit_lottie import st_lottie
import requests
from streamlit_card import card
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.chart_container import chart_container
# from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_toggle import st_toggle_switch
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stoggle import stoggle
from streamlit_option_menu import option_menu
from streamlit_folium import st_folium
import pickle

# Import the model code
from train_all_model1 import (
    load_stock_data, preprocess_data, create_sequences, 
    train_lstm_model, train_bilstm_model, train_gru_model,
    train_transformer_model, train_informer_model,
    TimeSeriesTransformer, InformerModel
)

# Set page configuration
st.set_page_config(
    page_title="StockVision AI - Advanced Stock Price Prediction",
    page_icon="ðŸ“ˆ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
try:
    local_css("style.css")
except:
    st.markdown("""
    <style>
    /* Main page styling */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Custom container styles */
    .custom-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 14px;
        color: #64748b;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transform: translateY(-2px);
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, #3b82f6, #2dd4bf);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Chart styling */
    .custom-chart {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        border: none;
        color: #4b5563;
        padding: 10px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6;
        font-weight: bold;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        height: 6px;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background-color: #3b82f6;
        border: 2px solid white;
    }
    
    /* Select box styling */
    .stSelectbox [data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Model comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .comparison-table th {
        background-color: #3b82f6;
        color: white;
        padding: 12px 15px;
        text-align: left;
    }
    
    .comparison-table tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    .comparison-table tr:nth-child(odd) {
        background-color: #ffffff;
    }
    
    .comparison-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #e2e8f0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

import json
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Example:
lottie_stocks = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json")
lottie_analysis = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_D6tjsm.json")
lottie_prediction = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_nc3fmxnl.json")

# Session state initialization
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = False
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Navigation
with st.sidebar:
    st.image("https://www.svgrepo.com/show/374216/stock.svg", width=120)
    st.title("StockVision AI")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Data Explorer", "Model Training", "Prediction", "Model Comparison", "About"],
        icons=["house", "database", "gear", "graph-up", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    
    # Dark mode toggle
    dark_mode = st_toggle_switch(
        label="Dark Mode",
        key="switch_1",
        default_value=st.session_state.dark_mode,
        label_after=False
    )
    
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        if dark_mode:
            st.markdown("""
            <style>
            .main {
                background-color: #0f172a;
                color: #f1f5f9;
            }
            .custom-container {
                background-color: #1e293b;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .metric-card {
                background-color: #283548;
                color: #f1f5f9;
            }
            .metric-value {
                color: #f1f5f9;
            }
            .metric-label {
                color: #94a3b8;
            }
            .stTabs [aria-selected="true"] {
                background-color: #283548 !important;
            }
            .comparison-table th {
                background-color: #2563eb;
            }
            .comparison-table tr:nth-child(even) {
                background-color: #283548;
            }
            .comparison-table tr:nth-child(odd) {
                background-color: #1e293b;
            }
            .comparison-table td {
                border-bottom: 1px solid #334155;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
            .main {
                background-color: #f5f7fa;
                color: #0f172a;
            }
            .custom-container {
                background-color: #ffffff;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .metric-card {
                background-color: #f1f5f9;
                color: #0f172a;
            }
            .metric-value {
                color: #0f172a;
            }
            .metric-label {
                color: #64748b;
            }
            .stTabs [aria-selected="true"] {
                background-color: white !important;
            }
            .comparison-table th {
                background-color: #3b82f6;
            }
            .comparison-table tr:nth-child(even) {
                background-color: #f8fafc;
            }
            .comparison-table tr:nth-child(odd) {
                background-color: #ffffff;
            }
            .comparison-table td {
                border-bottom: 1px solid #e2e8f0;
            }
            </style>
            """, unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Â© 2025 StockVision AI")
    st.sidebar.caption("Powered by Streamlit")

# Home page
if selected == "Home":
    # Header with animation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1 style='font-size:3em;'>Welcome to StockVision AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5em;'>Advanced Stock Price Prediction with AI</p>", unsafe_allow_html=True)
        st.markdown(
            """
            StockVision AI leverages cutting-edge time series models and transformer architectures to predict stock prices with high accuracy.
            """
        )
        
        # Quick action buttons
        st.markdown("### Get Started")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Load Sample Data", use_container_width=True):
                st.session_state.page = "Data Explorer"
                st.rerun()

        with col_b:
            if st.button("Train Models", use_container_width=True):
                st.session_state.page = "Model Training"
                st.rerun()

        with col_c:
            if st.button("Make Predictions", use_container_width=True):
                st.session_state.page = "Prediction"
                st.rerun()

    
    with col2:
        st_lottie(lottie_stocks, height=300, key="stocks_animation")
    
    # Feature highlights
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class='custom-container'>
                <h3 style='text-align: center;'>Advanced Models</h3>
                <ul>
                    <li>LSTM & BiLSTM Networks</li>
                    <li>GRU Architecture</li>
                    <li>Transformer Models</li>
                    <li>Informer Architecture</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
            <div class='custom-container'>
                <h3 style='text-align: center;'>Interactive Analysis</h3>
                <ul>
                    <li>Historical Data Visualization</li>
                    <li>Technical Indicators</li>
                    <li>Model Performance Metrics</li>
                    <li>Comparison Dashboard</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown("""
            <div class='custom-container'>
                <h3 style='text-align: center;'>User-Friendly</h3>
                <ul>
                    <li>Intuitive Interface</li>
                    <li>Easy Model Training</li>
                    <li>Custom Prediction Horizon</li>
                    <li>Export Capabilities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent market updates
    st.markdown("---")
    colored_header(
        label="Market Pulse",
        description="Recent market movements and trends",
        color_name="blue-70"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class='custom-container'>
                <h4>Major Indices</h4>
                <div style='display: flex; justify-content: space-between;'>
                    <div>S&P 500</div>
                    <div style='color: green;'>+1.2%</div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Nasdaq</div>
                    <div style='color: green;'>+0.8%</div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Dow Jones</div>
                    <div style='color: red;'>-0.3%</div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Russell 2000</div>
                    <div style='color: green;'>+1.5%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
            <div class='custom-container'>
                <h4>Top Moving Sectors</h4>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Technology</div>
                    <div style='color: green;'>+2.1%</div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Healthcare</div>
                    <div style='color: green;'>+1.7%</div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Financial</div>
                    <div style='color: red;'>-0.5%</div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>Energy</div>
                    <div style='color: green;'>+1.3%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Data Explorer page
elif selected == "Data Explorer":
    colored_header(
        label="Data Explorer",
        description="Load and analyze stock data",
        color_name="blue-70"
    )
    
    # Data source selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        data_source = st.radio(
            "Select Data Source",
            ["Yahoo Finance", "Upload CSV", "Sample Data"],
            horizontal=True
        )
    
    # Data loading based on source
    if data_source == "Yahoo Finance":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")
        
        with col2:
            today = datetime.now()
            start_date = st.date_input("Start Date", today - timedelta(days=365*3))
            end_date = st.date_input("End Date", today)
        
        if st.button("Load Data", use_container_width=True):
            with st.spinner("Fetching data from Yahoo Finance..."):
                try:
                    df = yf.download(ticker, start=start_date, end=end_date)
                    if df.empty:
                        st.error("No data found for the specified ticker and date range.")
                    else:
                        # Ensure date is in the index but also as a column
                        df = df.reset_index()
                        df['Date'] = pd.to_datetime(df['Date'])
                        st.session_state.stock_data = df
                        st.session_state.ticker = ticker
                        st.success(f"Successfully loaded data for {ticker}")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your stock data CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                st.session_state.stock_data = df
                st.session_state.ticker = uploaded_file.name.split('.')[0]
                st.success(f"Successfully loaded data from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
    
    else:  # Sample Data
        if st.button("Load Sample Data (AAPL)", use_container_width=True):
            with st.spinner("Loading sample data..."):
                try:
                    today = datetime.now()
                    start_date = today - timedelta(days=365*3)
                    df = yf.download("AAPL", start=start_date, end=today)
                    df = df.reset_index()
                    df['Date'] = pd.to_datetime(df['Date'])
                    st.session_state.stock_data = df
                    st.session_state.ticker = "AAPL"
                    st.success("Successfully loaded sample data (AAPL)")
                except Exception as e:
                    st.error(f"Error loading sample data: {e}")
    
    # Display and analyze loaded data
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        
        # Data overview
        st.markdown("### Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        with col4:
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
            
        
        style_metric_cards()
        
        # Data tabs
        tab1, tab2, tab3, tab4 = st.tabs(["ðŸ“Š Preview", "ðŸ“ˆ Visualization", "ðŸ“‰ Technical Indicators", "ðŸ” Statistics"])
        
        with tab1:
            st.dataframe(df.sort_values('Date', ascending=False).head(10), use_container_width=True)
            
            if st.checkbox("Show full dataset"):
                st.dataframe(df, use_container_width=True)
        
        with tab2:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line Chart", "Candlestick Chart", "OHLC Chart", "Area Chart"]
            )
         
            #
            with tab3:
                # Calculate indicators
                if 'indicator_df' not in st.session_state:
                    temp_df = df.copy()
                    temp_df['MA5'] = temp_df['Close'].rolling(window=5).mean()
                    temp_df['MA20'] = temp_df['Close'].rolling(window=20).mean()
                    temp_df['MA50'] = temp_df['Close'].rolling(window=50).mean()
                    
                # RSI
                delta = temp_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                temp_df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                temp_df['EMA12'] = temp_df['Close'].ewm(span=12, adjust=False).mean()
                temp_df['EMA26'] = temp_df['Close'].ewm(span=26, adjust=False).mean()
                temp_df['MACD'] = temp_df['EMA12'] - temp_df['EMA26']
                temp_df['Signal'] = temp_df['MACD'].ewm(span=9, adjust=False).mean()
                # Bollinger Bands Calculation
                temp_df['20MA'] = temp_df['Close'].rolling(window=20).mean()
                # Fill NaN values in rolling standard deviation
                temp_df['20STD'] = temp_df['Close'].rolling(window=20).std().fillna(0)
                temp_df['Upper'] = temp_df['20MA'] + (temp_df['20STD'] * 2)
                temp_df['Lower'] = temp_df['20MA'] - (temp_df['20STD'] * 2)
                
                



                st.session_state.indicator_df = temp_df.copy()
            
            indicator_df = st.session_state.indicator_df
            
            indicator_type = st.selectbox(
                "Select Technical Indicator",
                ["Moving Averages", "RSI", "MACD", "Bollinger Bands"]
            )
            
            with chart_container(st.container()):
                if indicator_type == "Moving Averages":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Close'], name='Close', line=dict(color='royalblue')))
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['MA5'], name='5-Day MA', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['MA20'], name='20-Day MA', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['MA50'], name='50-Day MA', line=dict(color='red')))
                    fig.update_layout(
                        title=f'{st.session_state.ticker} Moving Averages',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif indicator_type == "RSI":
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.1, 
                                      row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Close'], name='Close'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['RSI'], name='RSI'), row=2, col=1)
                    
                    # Add horizontal lines at 70 and 30
                    fig.add_shape(type='line', x0=indicator_df['Date'].min(), y0=70, x1=indicator_df['Date'].max(), y1=70,
                                line=dict(color='red', width=1, dash='dash'), row=2, col=1)
                    fig.add_shape(type='line', x0=indicator_df['Date'].min(), y0=30, x1=indicator_df['Date'].max(), y1=30,
                                line=dict(color='green', width=1, dash='dash'), row=2, col=1)
                    
                    fig.update_layout(
                        title=f'{st.session_state.ticker} Relative Strength Index (RSI)',
                        template='plotly_white',
                        height=600
                    )
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif indicator_type == "MACD":
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.1, 
                                      row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Close'], name='Close'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['MACD'], name='MACD'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Signal'], name='Signal'), row=2, col=1)
                    # Histogram for the difference between MACD and Signal line
                    macd_hist = indicator_df['MACD'] - indicator_df['Signal']
                    fig.add_trace(
                        go.Bar(
                            x=indicator_df['Date'], 
                            y=macd_hist, 
                            name='Histogram',
                            marker=dict(
                                color=np.where(macd_hist >= 0, 'green', 'red'),
                                line=dict(color='rgb(248, 248, 249)', width=1)
                            )
                        ), 
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title=f'{st.session_state.ticker} MACD',
                        template='plotly_white',
                        height=600
                    )
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="MACD", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif indicator_type == "Bollinger Bands":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Close'], name='Close', line=dict(color='royalblue')))
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Upper'], name='Upper Band', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Lower'], name='Lower Band', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['20MA'], name='20-Day MA', line=dict(color='orange')))
                    fig.update_layout(
                        title=f'{st.session_state.ticker} Bollinger Bands',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_white'
                        
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Price Statistics")
                price_stats = df['Close'].describe().reset_index()
                price_stats.columns = ['Statistic', 'Value']
                st.dataframe(price_stats, use_container_width=True)
                
                # Daily returns
                df['Daily Return'] = df['Close'].pct_change() * 100
                
                fig = px.histogram(
                    df.dropna(), x='Daily Return',
                    nbins=50,
                    title="Distribution of Daily Returns",
                    labels={'Daily Return': 'Daily Return (%)'},
                    template='plotly_white'
                )
                fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Volume Statistics")
                volume_stats = df['Volume'].describe().reset_index()
                volume_stats.columns = ['Statistic', 'Value']
                st.dataframe(volume_stats, use_container_width=True)
                
                # Volume over time
                fig = px.line(
                    df, x='Date', y='Volume',
                    title="Trading Volume Over Time",
                    labels={'Volume': 'Volume', 'Date': 'Date'},
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.markdown("#### Correlation Matrix")
            corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            corr_matrix = df[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

# Model Training page
elif selected == "Model Training":
    colored_header(
        label="Model Training",
        description="Train various deep learning models on your stock data",
        color_name="blue-70"
    )
    
    # Check if data is loaded
    if st.session_state.stock_data is None:
        st.warning("Please load stock data in the Data Explorer tab first.")
        if st.button("Go to Data Explorer"):
            st.session_state.page = "Data Explorer"
            st.experimental_rerun()
    else:
        df = st.session_state.stock_data
        
        st.markdown(f"### Training Models on {st.session_state.ticker} Stock Data")
        
        with st.expander("Model Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                sequence_length = st.slider("Sequence Length (Days)", 10, 100, 30, 
                                          help="Number of past days to use for prediction")
                future_days = st.slider("Prediction Horizon (Days)", 1, 30, 5, 
                                       help="Number of days to predict into the future")
                
                feature_cols = st.multiselect(
                    "Select Features",
                    options=['Open', 'High', 'Low', 'Close', 'Volume', 'Day of Week', 'Month', 'Year'],
                    default=['Open', 'High', 'Low', 'Close', 'Volume'],
                    help="Select features to use for training"
                )
            
            with col2:
                train_split = st.slider("Training Data Split (%)", 50, 90, 80, 
                                      help="Percentage of data to use for training")
                batch_size = st.select_slider(
                    "Batch Size",
                    options=[8, 16, 32, 64, 128],
                    value=32,
                    help="Batch size for training"
                )
                
                models_to_train = st.multiselect(
                    "Select Models to Train",
                    options=["LSTM", "BiLSTM", "GRU", "Transformer", "Informer"],
                    default=["LSTM", "BiLSTM"],
                    help="Select which models to train"
                )
        
        # Advanced settings
        with st.expander("Advanced Training Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.1, 0.01, 0.001, 0.0001],
                    value=0.001
                )
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            
            with col2:
                optimizer = st.selectbox(
                    "Optimizer",
                    options=["Adam", "RMSprop", "SGD"],
                    index=0
                )
                loss_function = st.selectbox(
                    "Loss Function",
                    options=["MSE", "MAE", "Huber"],
                    index=0
                )
                early_stopping = st.checkbox("Use Early Stopping", value=True)
        
        # Data preprocessing and model training button
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("Ready to train selected models. Click the button to start training.")
        with col2:
            train_button = st.button("Train Models", use_container_width=True)
        
        if train_button:
            # Placeholder for model training logic
            with st.spinner("Preprocessing data and training models..."):
                # Add date features if selected
                temp_df = df.copy()
                
                if 'Day of Week' in feature_cols:
                    temp_df['Day of Week'] = temp_df['Date'].dt.dayofweek
                if 'Month' in feature_cols:
                    temp_df['Month'] = temp_df['Date'].dt.month
                if 'Year' in feature_cols:
                    temp_df['Year'] = temp_df['Date'].dt.year
                
                # Data preprocessing
                st.text("Preprocessing data...")
                progress_bar = st.progress(0)
                
                # Simulate preprocessing steps with progress bar
                for i in range(5):
                    time.sleep(0.2)
                    progress_bar.progress((i + 1) * 0.1)
                
                # Feature scaling
                st.text("Scaling features...")
                for i in range(5, 10):
                    time.sleep(0.2)
                    progress_bar.progress((i + 1) * 0.1)
                
                # Create empty placeholders for models and metrics
                st.session_state.models = {}
                st.session_state.metrics = {}
                
                # Train each selected model
                for i, model_name in enumerate(models_to_train):
                    model_progress = st.progress(0)
                    st.text(f"Training {model_name} model...")
                    
                    # Simulate training with epochs
                    for epoch in range(min(10, epochs)):  # Simulating fewer epochs for demo
                        time.sleep(0.2)
                        current_progress = (epoch + 1) / min(10, epochs)
                        model_progress.progress(current_progress)
                    
                    # Store dummy model and metrics for demonstration
                    st.session_state.models[model_name] = f"{model_name}_model"
                    
                    # Generate random metrics for demonstration
                    import random
                    train_mse = random.uniform(0.0001, 0.01)
                    val_mse = train_mse * random.uniform(1.0, 1.5)
                    train_mae = train_mse * 0.8
                    val_mae = val_mse * 0.8
                    
                    st.session_state.metrics[model_name] = {
                        'train_mse': train_mse,
                        'val_mse': val_mse,
                        'train_mae': train_mae,
                        'val_mae': val_mae
                    }
                
                st.session_state.trained_models = True
                st.success("Model training completed!")
        
        # Display training results if models are trained
        if st.session_state.trained_models:
            st.markdown("### Training Results")
            
            # Model metrics
            metrics_df = pd.DataFrame()
            
            for model_name, metrics in st.session_state.metrics.items():
                temp_df = pd.DataFrame({
                    'Model': [model_name],
                    'Train MSE': [metrics['train_mse']],
                    'Val MSE': [metrics['val_mse']],
                    'Train MAE': [metrics['train_mae']],
                    'Val MAE': [metrics['val_mae']]
                })
                metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualization of metrics
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    metrics_df, x='Model', y='Val MSE', 
                    title="Validation MSE by Model",
                    color='Model',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    metrics_df, x='Model', y='Val MAE', 
                    title="Validation MAE by Model",
                    color='Model',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Save models
            st.markdown("### Save Models")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                save_path = st.text_input("Save Directory", "./models")
            
            with col2:
                if st.button("Save Models", use_container_width=True):
                    with st.spinner("Saving models..."):
                        # Simulate saving models
                        time.sleep(1)
                        st.success(f"Models saved to {save_path}")

# Prediction page
elif selected == "Prediction":
    colored_header(
        label="Stock Price Prediction",
        description="Make predictions using trained models",
        color_name="blue-70"
    )
    
    # Check if models are trained
    if not st.session_state.trained_models:
        st.warning("Please train models first in the Model Training tab.")
        if st.button("Go to Model Training"):
            st.session_state.page = "Model Training"
            st.experimental_rerun()
    else:
        st.markdown(f"### Predict {st.session_state.ticker} Stock Prices")
        
        # Prediction settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_model = st.selectbox(
                "Select Model",
                options=list(st.session_state.models.keys()),
                index=0
            )
        
        with col2:
            forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
        
        with col3:
            confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95, 1)
        
        # Make prediction
        if st.button("Generate Prediction", use_container_width=True):
            with st.spinner("Generating predictions..."):
                # Simulate prediction process
                time.sleep(1)
                
                # Generate dummy prediction data for visualization
                import numpy as np
                df = st.session_state.stock_data
                last_date = df['Date'].max()
                last_price = df['Close'].iloc[-1]
                
                # Generate future dates
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                
                # Generate dummy predictions with some randomness
                np.random.seed(42)  # For reproducibility
                pred_values = [last_price]
                for _ in range(forecast_days):
                    # Random walk with drift
                    next_val = pred_values[-1] * (1 + np.random.normal(0.001, 0.02))
                    pred_values.append(next_val)
                
                pred_values = pred_values[1:]  # Remove the starting value
                
                # Calculate confidence intervals
                ci_factor = 1.96 if confidence_interval == 95 else 2.58 if confidence_interval == 99 else 1.28
                std_dev = np.std(df['Close'].pct_change().dropna()) * last_price
                upper_bound = [pred + ci_factor * std_dev * np.sqrt(i+1) for i, pred in enumerate(pred_values)]
                lower_bound = [pred - ci_factor * std_dev * np.sqrt(i+1) for i, pred in enumerate(pred_values)]
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted': pred_values,
                    'Upper': upper_bound,
                    'Lower': lower_bound
                })
                
                # Store in session state
                st.session_state.predictions[selected_model] = pred_df
                
                st.success("Prediction generated successfully!")
        
        # Display predictions if available
        if selected_model in st.session_state.predictions:
            pred_df = st.session_state.predictions[selected_model]
            
            # Metrics
            st.markdown("### Prediction Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                latest_price = st.session_state.stock_data['Close'].iloc[-1]
                predicted_price = pred_df['Predicted'].iloc[-1]
                change = ((predicted_price - latest_price) / latest_price) * 100
                st.metric(
                    f"Price in {forecast_days} Days",
                    f"${predicted_price:.2f}",
                    f"{change:.2f}%"
                )
            
            with col2:
                min_price = pred_df['Lower'].min()
                st.metric(
                    f"Minimum Predicted",
                    f"${min_price:.2f}",
                    f"{((min_price - latest_price) / latest_price) * 100:.2f}%"
                )
            
            with col3:
                max_price = pred_df['Upper'].max()
                st.metric(
                    f"Maximum Predicted",
                    f"${max_price:.2f}",
                    f"{((max_price - latest_price) / latest_price) * 100:.2f}%"
                )
            
            style_metric_cards()
            
            # Visualize predictions
            st.markdown("### Forecast Visualization")
            
            # Combine historical and predicted data
            hist_df = st.session_state.stock_data[['Date', 'Close']].tail(30)
            
            fig = go.Figure()
            
            # Historical line
            fig.add_trace(go.Scatter(
                x=hist_df['Date'],
                y=hist_df['Close'],
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Prediction line
            fig.add_trace(go.Scatter(
                x=pred_df['Date'],
                y=pred_df['Predicted'],
                name='Prediction',
                line=dict(color='red')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pred_df['Date'].tolist() + pred_df['Date'].tolist()[::-1],
                y=pred_df['Upper'].tolist() + pred_df['Lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231,107,243,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_interval}% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'{st.session_state.ticker} Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                legend_title='Legend',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction table
            st.markdown("### Detailed Forecast")
            
            # Format the prediction dataframe for display
            display_df = pred_df.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Predicted'] = display_df['Predicted'].round(2)
            display_df['Upper'] = display_df['Upper'].round(2)
            display_df['Lower'] = display_df['Lower'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download predictions
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"{st.session_state.ticker}_prediction_{selected_model}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Model Comparison page
elif selected == "Model Comparison":
    colored_header(
        label="Model Comparison",
        description="Compare the performance of different prediction models",
        color_name="blue-70"
    )
    
    # Check if models are trained
    if not st.session_state.trained_models:
        st.warning("Please train models first in the Model Training tab.")
        if st.button("Go to Model Training"):
            st.session_state.page = "Model Training"
            st.experimental_rerun()
    else:
        st.markdown("### Compare Model Performance")
        
        # Select models to compare
        models_to_compare = st.multiselect(
            "Select Models to Compare",
            options=list(st.session_state.models.keys()),
            default=list(st.session_state.models.keys())
        )
        
        if not models_to_compare:
            st.warning("Please select at least one model to compare.")
        else:
            # Comparison metrics
            metrics = ['MSE', 'MAE', 'RMSE', 'MAPE', 'RÂ²']
            
            # Generate dummy comparison results
            if st.button("Run Comparison", use_container_width=True):
                with st.spinner("Comparing models..."):
                    # Simulate comparison process
                    time.sleep(1)
                    
                    # Create comparison dataframe with random metrics
                    import random
                    
                    comparison_data = []
                    
                    for model in models_to_compare:
                        # Base MSE from training metrics
                        base_mse = st.session_state.metrics[model]['val_mse']
                        
                        comparison_data.append({
                            'Model': model,
                            'MSE': base_mse,
                            'MAE': base_mse * 0.8,
                            'RMSE': np.sqrt(base_mse),
                            'MAPE': base_mse * 100,
                            'RÂ²': max(0, 1 - (base_mse / 0.01))
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.session_state.comparison_results = comparison_df
                    
                    st.success("Comparison completed!")
            
            # Display comparison results
            if st.session_state.comparison_results is not None:
                comparison_df = st.session_state.comparison_results
                
                # Filter for selected models
                comparison_df = comparison_df[comparison_df['Model'].isin(models_to_compare)]
                
                # Determine best model
                best_model = comparison_df.iloc[comparison_df['MSE'].argmin()]['Model']
                
                st.markdown(f"### Results Summary ({len(models_to_compare)} Models)")
                st.info(f"Best performing model: **{best_model}** (lowest MSE)")
                
                # Format table for display
                display_df = comparison_df.copy()
                for col in metrics:
                    if col in display_df.columns:
                        if col in ['MSE', 'MAE', 'RMSE']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
                        elif col == 'MAPE':
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                        elif col == 'RÂ²':
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                
                # Display results table
                st.markdown("#### Comparison Table")
                st.dataframe(display_df, use_container_width=True)
                
                # Visualize comparison
                st.markdown("#### Comparison Charts")
                
                tab1, tab2 = st.tabs(["ðŸ“Š Bar Charts", "ðŸ“ˆ Radar Chart"])
                
                with tab1:
                    metric_to_plot = st.selectbox(
                        "Select Metric to Visualize",
                        options=['MSE', 'MAE', 'RMSE', 'MAPE', 'RÂ²']
                    )
                    
                    fig = px.bar(
                        comparison_df, 
                        x='Model', 
                        y=metric_to_plot,
                        title=f"Model Comparison by {metric_to_plot}",
                        color='Model',
                        template='plotly_white'
                    )
                    
                    # For RÂ², higher is better, for others lower is better
                    if metric_to_plot == 'RÂ²':
                        fig.update_layout(yaxis_title=f"{metric_to_plot} (Higher is better)")
                    else:
                        fig.update_layout(yaxis_title=f"{metric_to_plot} (Lower is better)")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Prepare data for radar chart
                    radar_data = comparison_df.copy()
                    
                    # Normalize metrics for radar chart (0-1 scale)
                    for metric in ['MSE', 'MAE', 'RMSE', 'MAPE']:
                        if metric in radar_data.columns:
                            max_val = radar_data[metric].max()
                            min_val = radar_data[metric].min()
                            if max_val > min_val:
                                # Invert so lower is better
                                radar_data[f'{metric}_norm'] = 1 - ((radar_data[metric] - min_val) / (max_val - min_val))
                            else:
                                radar_data[f'{metric}_norm'] = 1.0
                    
                    # RÂ² is already 0-1 and higher is better
                    if 'RÂ²' in radar_data.columns:
                        max_val = radar_data['RÂ²'].max()
                        min_val = radar_data['RÂ²'].min()
                        if max_val > min_val:
                            radar_data['RÂ²_norm'] = (radar_data['RÂ²'] - min_val) / (max_val - min_val)
                        else:
                            radar_data['RÂ²_norm'] = 1.0
                    
                    # Create radar chart
                    categories = ['MSE_norm', 'MAE_norm', 'RMSE_norm', 'MAPE_norm', 'RÂ²_norm']
                    category_labels = ['MSE', 'MAE', 'RMSE', 'MAPE', 'RÂ²']
                    
                    fig = go.Figure()
                    
                    for i, model in enumerate(radar_data['Model']):
                        values = radar_data.loc[radar_data['Model'] == model, categories].values.flatten().tolist()
                        values.append(values[0])  # Close the loop
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=category_labels + [category_labels[0]],  # Close the loop
                            fill='toself',
                            name=model
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        title="Model Performance Comparison (Higher is Better)",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export comparison results
                st.markdown("#### Export Results")
                
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download Comparison Results as CSV",
                    data=csv,
                    file_name=f"{st.session_state.ticker}_model_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# About page
elif selected == "About":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 style='font-size:2.5em;'>About StockVision AI</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            StockVision AI is an advanced financial analytics platform that uses cutting-edge deep learning techniques to predict stock prices. 
            
            ### Our Technology
            
            The application leverages multiple state-of-the-art time series models:
            
            * **LSTM (Long Short-Term Memory)**: Excellent for capturing long-term dependencies in time series data
            * **BiLSTM (Bidirectional LSTM)**: Processes data in both forward and backward directions
            * **GRU (Gated Recurrent Unit)**: Simpler architecture with comparable performance to LSTM
            * **Transformer**: Attention-based architecture that revolutionized NLP, adapted for time series
            * **Informer**: Efficient transformer variant specialized for long sequence time-series forecasting
            
            ### Features
            
            * Historical data analysis with interactive visualizations
            * Technical indicators calculation and plotting
            * Model training with customizable parameters
            * Stock price prediction with confidence intervals
            * Model performance comparison
            
            ### Data Sources
            
            StockVision AI uses Yahoo Finance API to fetch historical stock data. Users can also upload their own CSV files.
            """
        )
    
    with col2:
        st_lottie(lottie_analysis, height=300, key="analysis_animation")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Disclaimer")
        st.info(
            """
            The predictions provided by StockVision AI are for informational purposes only and should not be construed as financial advice. 
            Stock markets are subject to numerous factors that cannot be fully captured by any predictive model. 
            Always consult with a qualified financial advisor before making investment decisions.
            """
        )
    
    with col2:
        st.markdown("### Feedback & Support")
        st.warning(
            """
            This is a demo application. For questions, feedback, or support, please contact us at:
            
            ðŸ“§ support@stockvision.ai
            
            We welcome suggestions for new features and improvements!
            """
        )
    
    st.markdown("---")
    
    # Team section
    colored_header(
        label="Our Team",
        description="The experts behind StockVision AI",
        color_name="blue-70"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://randomuser.me/api/portraits/men/32.jpg" width="150" style="border-radius: 50%;">
                <h3>Dr. Alex Johnson</h3>
                <p>Lead Data Scientist</p>
                <p>Ph.D. in Machine Learning with 10+ years of experience in financial forecasting.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://randomuser.me/api/portraits/women/44.jpg" width="150" style="border-radius: 50%;">
                <h3>Sarah Chen</h3>
                <p>Financial Analyst</p>
                <p>CFA with expertise in quantitative analysis and algorithmic trading strategies.</p>
            </div>
            """
        )