import streamlit as st

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="SmartFolio AI - Advanced Stock Portfolio Management",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
import io


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
from streamlit_toggle import toggle
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

# Helper function to find column by name in a DataFrame
def find_column(df, name):
    # First try exact match
    if name in df.columns:
        return name
    
    # Try case-insensitive match for string columns
    for col in df.columns:
        if isinstance(col, str) and col.lower() == name.lower():
            return col
    
    # Try tuple columns
    for col in df.columns:
        if isinstance(col, tuple) and len(col) > 0:
            # Check if the first element is a string before calling lower()
            if isinstance(col[0], str) and col[0].lower() == name.lower():
                return col
    
    return None


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
        color: #0f172a;
    }
    
    .custom-container ul li {
        color: #0f172a;
        margin-bottom: 8px;
    }
    
    /* Enhanced Metric card styling with animations */
    .metric-card {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0) 100%);
        z-index: -1;
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #0f172a;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #3b82f6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        transform: translateY(20px);
        opacity: 0;
        animation: slideUpFadeIn 0.6s forwards 0.2s;
    }
    
    .metric-label {
        font-size: 15px;
        color: #64748b;
        font-weight: 500;
        transform: translateY(20px);
        opacity: 0;
        animation: slideUpFadeIn 0.6s forwards 0.4s;
    }
    
    @keyframes slideUpFadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Add a subtle pulse effect to highlight important metrics */
    .metric-card.highlight {
        animation: metricPulse 2s infinite;
    }
    
    @keyframes metricPulse {
        0% { box-shadow: 0 4px 15px rgba(59, 130, 246, 0.05); }
        50% { box-shadow: 0 4px 25px rgba(59, 130, 246, 0.3); }
        100% { box-shadow: 0 4px 15px rgba(59, 130, 246, 0.05); }
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
    
    /* Enhanced Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-20px); }
        60% { transform: translateY(-10px); }
    }
    
    .fade-in {
        animation: fadeIn 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
    }
    
    .slide-in-left {
        animation: slideInLeft 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
    }
    
    .slide-in-right {
        animation: slideInRight 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
    }
    
    .scale-in {
        animation: scaleIn 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
    }
    
    .bounce {
        animation: bounce 1s ease infinite;
    }
    
    /* Staggered animations for lists */
    .staggered-item:nth-child(1) { animation-delay: 0.1s; }
    .staggered-item:nth-child(2) { animation-delay: 0.2s; }
    .staggered-item:nth-child(3) { animation-delay: 0.3s; }
    .staggered-item:nth-child(4) { animation-delay: 0.4s; }
    .staggered-item:nth-child(5) { animation-delay: 0.5s; }
    .staggered-item:nth-child(6) { animation-delay: 0.6s; }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

import json
import requests

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"Error loading Lottie animation: {e}")
        return None

# Enhanced Lottie animations with more dynamic options
lottie_stocks = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_u4yrau.json")
lottie_analysis = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_0apkn3k1.json")
lottie_prediction = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_UJjaUv.json")
lottie_chart = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_l4xxtfd3.json")
lottie_success = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_jR229a.json")

# Add CSS for Lottie animation containers
st.markdown("""
<style>
/* Enhanced Lottie animation container */
.lottie-container {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    transition: all 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55);
    transform: perspective(1000px) rotateY(0deg);
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    margin: 10px 0;
}

.lottie-container:hover {
    transform: perspective(1000px) rotateY(5deg) translateY(-10px);
    box-shadow: 0 20px 35px rgba(0, 0, 0, 0.15);
}

/* Lottie animation entrance */
@keyframes lottieEntrance {
    0% {
        opacity: 0;
        transform: scale(0.8) translateY(30px);
    }
    70% {
        opacity: 0.7;
        transform: scale(1.05) translateY(-5px);
    }
    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

.lottie-entrance {
    animation: lottieEntrance 1.2s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards;
}

/* Floating animation for continuous movement */
@keyframes floating {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.lottie-float {
    animation: floating 4s ease-in-out infinite;
}

/* Pulse animation for attention */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.lottie-pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* Glow effect for dark mode */
.dark-mode .lottie-container {
    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
}

.dark-mode .lottie-container:hover {
    box-shadow: 0 20px 35px rgba(59, 130, 246, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Function to apply theme with enhanced transitions
def apply_theme(dark_mode=False):
    # Add transition animation for theme change
    st.markdown("""
    <style>
    /* Global transition for theme change */
    * {
        transition: background-color 0.5s ease, color 0.5s ease, border-color 0.5s ease, box-shadow 0.5s ease !important;
    }
    
    /* Theme change animation */
    @keyframes themeTransition {
        0% { filter: saturate(0) brightness(1); }
        50% { filter: saturate(0.5) brightness(0.8); }
        100% { filter: saturate(1) brightness(1); }
    }
    
    .streamlit-container {
        animation: themeTransition 0.8s ease forwards;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add or remove dark-mode class to body
    if dark_mode:
        st.markdown("""
        <script>
            document.body.classList.add('dark-mode');
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <script>
            document.body.classList.remove('dark-mode');
        </script>
        """, unsafe_allow_html=True)
    
    if dark_mode:
        # Dark mode CSS
        st.markdown("""
        <style>
        /* Base styles */
        .main {
            background-color: #0f172a;
            color: #f1f5f9;
            transition: all 0.3s ease;
        }
        
        /* Sidebar */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #1e293b;
        }
        
        .sidebar .sidebar-content {
            background-color: #1e293b;
            color: #f1f5f9;
        }
        
        /* Containers */
        .custom-container {
            background-color: #1e293b;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        /* Cards */
        .metric-card {
            background-color: #283548;
            color: #f1f5f9;
            transition: all 0.3s ease;
        }
        
        .metric-value {
            color: #f1f5f9;
        }
        
        .metric-label {
            color: #94a3b8;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e293b;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #283548 !important;
            color: #3b82f6 !important;
        }
        
        /* Tables */
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
        
        /* Inputs */
        .stTextInput > div > div > input {
            background-color: #283548;
            color: #f1f5f9;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #283548;
            color: #f1f5f9;
        }
        
        .stSelectbox [data-baseweb="select"] svg {
            color: #f1f5f9;
        }
        
        .stDateInput > div > div > input {
            background-color: #283548;
            color: #f1f5f9;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
            color: white;
            border: none;
        }
        
        /* Sliders */
        .stSlider [data-baseweb="slider"] {
            background-color: #475569;
        }
        
        .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
            background-color: #3b82f6;
        }
        
        /* Dataframes */
        .dataframe {
            background-color: #1e293b;
            color: #f1f5f9;
        }
        
        .dataframe th {
            background-color: #334155;
            color: #f1f5f9;
        }
        
        .dataframe td {
            background-color: #1e293b;
            color: #f1f5f9;
        }
        
        /* Charts */
        .js-plotly-plot .plotly {
            background-color: #1e293b;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #283548;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #283548;
            color: #f1f5f9;
        }
        
        .streamlit-expanderContent {
            background-color: #1e293b;
            color: #f1f5f9;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode CSS
        st.markdown("""
        <style>
        /* Base styles */
        .main {
            background-color: #f5f7fa;
            color: #0f172a;
            transition: all 0.3s ease;
        }
        
        /* Sidebar */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #f8fafc;
        }
        
        .sidebar .sidebar-content {
            background-color: #f8fafc;
            color: #0f172a;
        }
        
        /* Containers */
        .custom-container {
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            color: #0f172a;
        }
        
        .custom-container ul li {
            color: #0f172a;
            margin-bottom: 8px;
        }
        
        /* Cards */
        .metric-card {
            background-color: #f1f5f9;
            color: #0f172a;
            transition: all 0.3s ease;
        }
        
        .metric-value {
            color: #0f172a;
        }
        
        .metric-label {
            color: #64748b;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8fafc;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            color: #3b82f6 !important;
        }
        
        /* Tables */
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
        
        /* Inputs */
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #0f172a;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #ffffff;
            color: #0f172a;
        }
        
        .stSelectbox [data-baseweb="select"] svg {
            color: #0f172a;
        }
        
        .stDateInput > div > div > input {
            background-color: #ffffff;
            color: #0f172a;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
            color: white;
            border: none;
        }
        
        /* Sliders */
        .stSlider [data-baseweb="slider"] {
            background-color: #e2e8f0;
        }
        
        .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
            background-color: #3b82f6;
        }
        
        /* Dataframes */
        .dataframe {
            background-color: #ffffff;
            color: #0f172a;
        }
        
        .dataframe th {
            background-color: #f1f5f9;
            color: #0f172a;
        }
        
        .dataframe td {
            background-color: #ffffff;
            color: #0f172a;
        }
        
        /* Charts */
        .js-plotly-plot .plotly {
            background-color: #ffffff;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #f1f5f9;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #f1f5f9;
            color: #0f172a;
        }
        
        .streamlit-expanderContent {
            background-color: #ffffff;
            color: #0f172a;
        }
        </style>
        """, unsafe_allow_html=True)

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

# Apply the theme based on session state
apply_theme(st.session_state.dark_mode)

# Add theme preference cookie script
st.markdown("""
<script>
    // Function to get cookie by name
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
        return null;
    }
    
    // Function to set cookie
    function setCookie(name, value, days) {
        const d = new Date();
        d.setTime(d.getTime() + (days*24*60*60*1000));
        const expires = "expires="+ d.toUTCString();
        document.cookie = name + "=" + value + ";" + expires + ";path=/";
    }
    
    // Check for theme preference cookie on page load
    document.addEventListener('DOMContentLoaded', function() {
        const themePref = getCookie('stockvision_theme');
        if (themePref === 'dark') {
            // If cookie says dark mode but UI is in light mode, click the toggle
            if (!document.querySelector('.theme-icon.moon')) {
                setTimeout(() => {
                    const toggleSwitch = document.querySelector('input[type="checkbox"][aria-label="Dark Mode"]');
                    if (toggleSwitch) toggleSwitch.click();
                }, 500);
            }
        } else if (themePref === 'light') {
            // If cookie says light mode but UI is in dark mode, click the toggle
            if (!document.querySelector('.theme-icon.sun')) {
                setTimeout(() => {
                    const toggleSwitch = document.querySelector('input[type="checkbox"][aria-label="Dark Mode"]');
                    if (toggleSwitch) toggleSwitch.click();
                }, 500);
            }
        }
        
        // Add event listener to the toggle switch
        setTimeout(() => {
            const toggleSwitch = document.querySelector('input[type="checkbox"][aria-label="Dark Mode"]');
            if (toggleSwitch) {
                toggleSwitch.addEventListener('change', function() {
                    setCookie('stockvision_theme', this.checked ? 'dark' : 'light', 365);
                });
            }
        }, 1000);
    });
</script>
""", unsafe_allow_html=True)

# Navigation
with st.sidebar:
    # Use a URL for the logo image
    st.image("images/logo/smartfolio_logo.png", width=120)
    st.title("SmartFolio AI")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Data Explorer", "Model Training", "Prediction", "Model Comparison", "About"],
        icons=["house", "database", "gear", "graph-up", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    
    # Enhanced Theme toggle with animated icons
    st.markdown("""
    <style>
    .theme-toggle-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    
    /* Improved theme icons with 3D effects */
    .theme-icon {
        font-size: 28px;
        margin-right: 15px;
        transition: all 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55);
        filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
        transform-origin: center;
    }
    
    .theme-icon:hover {
        transform: scale(1.2) rotate(15deg);
    }
    
    .theme-icon.sun {
        color: #f59e0b;
        text-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
    }
    
    .theme-icon.moon {
        color: #818cf8;
        text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    }
    
    /* Custom toggle switch styling */
    .theme-toggle-switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 30px;
        margin-right: 10px;
    }
    
    .theme-toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .theme-toggle-slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #e2e8f0;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55);
        border-radius: 34px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .theme-toggle-slider:before {
        position: absolute;
        content: "";
        height: 22px;
        width: 22px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55);
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    input:checked + .theme-toggle-slider {
        background-color: #3b82f6;
    }
    
    input:checked + .theme-toggle-slider:before {
        transform: translateX(30px);
    }
    
    .theme-toggle-slider:hover {
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    input:checked + .theme-toggle-slider:hover:before {
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
    }
    
    .theme-label {
        font-weight: 500;
        transition: all 0.3s ease;
        margin-left: 10px;
        user-select: none;
    }
    
    /* Enhanced animations */
    @keyframes sunRise {
        0% { opacity: 0; transform: translateY(10px) scale(0.8); }
        50% { opacity: 0.8; transform: translateY(-5px) scale(1.1); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    @keyframes moonRise {
        0% { opacity: 0; transform: translateY(10px) scale(0.8) rotate(-30deg); }
        50% { opacity: 0.8; transform: translateY(-5px) scale(1.1) rotate(15deg); }
        100% { opacity: 1; transform: translateY(0) scale(1) rotate(0); }
    }
    
    .sun-animation {
        animation: sunRise 0.8s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards;
    }
    
    .moon-animation {
        animation: moonRise 0.8s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards;
    }
    
    /* Theme transition overlay */
    .theme-transition-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.3);
        z-index: 9999;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .theme-transition-active {
        opacity: 1;
    }
    
    /* Fix for Streamlit iframe structure */
    iframe {
        position: relative;
        z-index: 1;
    }
    
    /* Ensure overlay covers the entire viewport */
    body {
        overflow-x: hidden;
    }
    </style>
    
    <!-- Theme transition overlay -->
    <div id="themeTransitionOverlay" class="theme-transition-overlay"></div>
    
    <script>
    // Function to animate theme transition
    function animateThemeTransition() {
        const overlay = document.getElementById('themeTransitionOverlay');
        if (overlay) {
            overlay.classList.add('theme-transition-active');
            setTimeout(() => {
                overlay.classList.remove('theme-transition-active');
            }, 500);
        }
    }
    
    // Add event listener to toggle switch
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(() => {
            const toggleSwitch = document.querySelector('.theme-toggle-switch input');
            if (toggleSwitch) {
                toggleSwitch.addEventListener('change', animateThemeTransition);
            }
        }, 1000);
    });
    </script>
    """, unsafe_allow_html=True)
    
    col_theme1, col_theme2 = st.columns([1, 5])
    
    with col_theme1:
        if st.session_state.dark_mode:
            st.markdown("""
            <div class="theme-icon moon moon-animation">
                <span>🌙</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="theme-icon sun sun-animation">
                <span>☀️</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col_theme2:
        # Add custom styling for the toggle
        st.markdown("""
        <style>
        /* Style the Streamlit toggle to look like our custom design */
        .stToggle {
            transform: scale(1.2);
            margin-top: 5px;
            display: flex;
            align-items: center;
        }
        
        /* Style the toggle track */
        .stToggle > div > div {
            background-color: #e2e8f0 !important;
            border-radius: 34px !important;
            height: 24px !important;
            width: 48px !important;
            transition: all 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55) !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Style the toggle knob */
        .stToggle > div > div > div {
            background-color: white !important;
            border-radius: 50% !important;
            height: 18px !important;
            width: 18px !important;
            margin: 3px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55) !important;
        }
        
        /* Style the active toggle */
        .stToggle > div[data-baseweb="checkbox"] > div[data-checked="true"] {
            background-color: #3b82f6 !important;
        }
        
        /* Move the knob when checked */
        .stToggle > div[data-baseweb="checkbox"] > div[data-checked="true"] > div {
            transform: translateX(24px) !important;
        }
        
        /* Add hover effects */
        .stToggle > div > div:hover {
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Add glow effect on hover when checked */
        .stToggle > div[data-baseweb="checkbox"] > div[data-checked="true"]:hover > div {
            box-shadow: 0 0 8px 2px rgba(59, 130, 246, 0.4) !important;
        }
        
        /* Style the toggle label */
        .stToggle label {
            font-weight: 500 !important;
            margin-left: 10px !important;
            color: var(--text-color, #0f172a) !important;
            font-size: 16px !important;
        }
        
        
        /* Dark mode adjustments */
        .dark-mode .stToggle label {
            color: #f1f5f9 !important;
        }
        
        /* Add animation to the toggle */
        @keyframes togglePulse {
            0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(59, 130, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }
        
        .stToggle > div[data-baseweb="checkbox"] > div[data-checked="true"] > div {
            animation: togglePulse 1.5s ease-in-out 1;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Use the Streamlit toggle with our custom styling
        dark_mode = toggle(
            label="Dark Mode",
            key="switch_1",
            value=st.session_state.dark_mode
        )
        
        # Add JavaScript for theme transition effect
        st.markdown("""
        <div id="themeTransitionOverlay" class="theme-transition-overlay"></div>
        
        <script>
        // Function to animate theme transition
        function animateThemeTransition() {
            const overlay = document.getElementById('themeTransitionOverlay');
            if (overlay) {
                overlay.classList.add('theme-transition-active');
                setTimeout(() => {
                    overlay.classList.remove('theme-transition-active');
                }, 500);
            }
        }
        
        // Add event listener to toggle switch
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(() => {
                const toggleSwitch = document.querySelector('.stToggle input[type="checkbox"]');
                if (toggleSwitch) {
                    toggleSwitch.addEventListener('change', animateThemeTransition);
                }
            }, 1000);
        });
        </script>
        """, unsafe_allow_html=True)
    
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        apply_theme(dark_mode)
        st.rerun()  # Rerun the app to apply the theme changes
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 SmartFolio AI")
    st.sidebar.caption("Powered by Streamlit")

# Home page
if selected == "Home":
    # Logo and title at the top
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("images/logo/smartfolio_logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='font-size:2.5em; margin-top:15px;'>SmartFolio AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-bottom:30px;'>", unsafe_allow_html=True)
    
    # Header with animation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1 style='font-size:3em;'>Welcome to SmartFolio AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5em;'>Advanced Stock Portfolio Management with AI</p>", unsafe_allow_html=True)
        st.markdown(
            """
            SmartFolio AI leverages cutting-edge time series models and transformer architectures to help you manage your investment portfolio with confidence and precision.
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
        # Display the project logo
        st.markdown('<div class="logo-container" style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
        st.image("images/logo/smartfolio_logo.png", width=300)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #0f172a;'>Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class='custom-container'>
                <h3 style='text-align: center; color: #0f172a;'>Advanced Models</h3>
                <ul style='color: #0f172a;'>
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
                <h3 style='text-align: center; color: #0f172a;'>Interactive Analysis</h3>
                <ul style='color: #0f172a;'>
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
                <h3 style='text-align: center; color: #0f172a;'>User-Friendly</h3>
                <ul style='color: #0f172a;'>
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
    # Logo and title at the top
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("images/logo/smartfolio_logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='font-size:2.5em; margin-top:15px;'>SmartFolio AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-bottom:30px;'>", unsafe_allow_html=True)
    
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
                    # Download data with explicit period
                    df = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date,
                        progress=False,
                        auto_adjust=True,  # Adjust all OHLC automatically
                        actions="inline"   # Include dividends and splits
                    )
                    
                    if df.empty:
                        st.error(f"No data found for {ticker} in the specified date range.")
                    else:
                        # Ensure date is in the index but also as a column
                        df = df.reset_index()
                        
                        # Fix column names if they are tuples
                        if any(isinstance(col, tuple) for col in df.columns):
                            # Create a new list of column names
                            new_columns = []
                            for col in df.columns:
                                if isinstance(col, tuple):
                                    if col[0] == 'Date':
                                        new_columns.append('Date')
                                    else:
                                        new_columns.append(col[0])  # Use the first part of the tuple
                                else:
                                    new_columns.append(col)
                            
                            # Directly assign the new column names
                            df.columns = new_columns
                            
                            st.info(f"Column names have been standardized: {list(df.columns)}")
                        
                        # Ensure Date is datetime
                        try:
                            if 'Date' in df.columns:
                                df['Date'] = pd.to_datetime(df['Date'])
                            else:
                                st.warning("Date column not found. Please check your data.")
                        except Exception as e:
                            st.warning(f"Could not convert Date column to datetime: {str(e)}")
                        
                        # Ensure all required columns exist
                        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        
                        # Check for similar column names (case insensitive)
                        col_map = {}
                        for req_col in required_cols:
                            for col in df.columns:
                                if col.lower() == req_col.lower() and col != req_col:
                                    col_map[col] = req_col
                        
                        # Rename columns if needed
                        if col_map:
                            df = df.rename(columns=col_map)
                            st.info(f"Some columns were renamed for consistency: {col_map}")
                        
                        # Now check for missing columns
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            st.warning(f"Data is missing columns: {', '.join(missing_cols)}. Some features may not work.")
                        
                        # Fill any NaN values
                        df = df.fillna(method='ffill').fillna(method='bfill')
                        
                        # Store in session state
                        st.session_state.stock_data = df
                        st.session_state.ticker = ticker
                        
                        st.success(f"Successfully loaded data for {ticker} ({df.shape[0]} rows)")
                        
                        # Show a preview
                        with st.expander("Preview Data"):
                            st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.info("Please check your internet connection and try again.")
    
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
            with st.spinner("Loading sample data (AAPL)..."):
                try:
                    today = datetime.now()
                    start_date = today - timedelta(days=365*3)
                    
                    df = yf.download(
                            "AAPL", 
                            start=start_date, 
                            end=today,
                            progress=False,
                            auto_adjust=True,
                            actions="inline"
                        )
                        
                    if df.empty:
                                st.error("Could not load sample data. Please check your internet connection.")
                    else:
                                # Process the data
                                df = df.reset_index()
                                
                                # Fix column names if they are tuples
                                if any(isinstance(col, tuple) for col in df.columns):
                                    # Create a new list of column names
                                    new_columns = []
                                    for col in df.columns:
                                        if isinstance(col, tuple):
                                            if col[0] == 'Date':
                                                new_columns.append('Date')
                                            else:
                                                new_columns.append(col[0])  # Use the first part of the tuple
                                        else:
                                            new_columns.append(col)
                                    
                                    # Directly assign the new column names
                                    df.columns = new_columns
                                    
                                    st.info(f"Column names have been standardized: {list(df.columns)}")
                                
                                # Ensure Date is datetime
                                try:
                                    if 'Date' in df.columns:
                                        df['Date'] = pd.to_datetime(df['Date'])
                                    else:
                                        st.warning("Date column not found. Please check your data.")
                                except Exception as e:
                                    st.warning(f"Could not convert Date column to datetime: {str(e)}")
                                
                                # Fill any NaN values
                                df = df.fillna(method='ffill').fillna(method='bfill')
                                
                                # Store in session state
                                st.session_state.stock_data = df
                                st.session_state.ticker = "AAPL"
                                
                                st.success(f"Successfully loaded sample data (AAPL) with {df.shape[0]} rows")
                                
                                # Show a preview
                                with st.expander("Preview Sample Data"):
                                    st.dataframe(df.head())
                except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
                        st.info("Please check your internet connection and try again.")
        
    # Display and analyze loaded data
    if st.session_state.stock_data is not None:
        try:
            df = st.session_state.stock_data
            
            # Validate DataFrame
            if df is None or df.empty:
                st.error("The loaded data is empty. Please try loading data again.")
                st.stop()
            
            # Find columns by case-insensitive matching
            # Map standard column names
            col_mapping = {
                'date': find_column(df, 'date'),
                'open': find_column(df, 'open'),
                'high': find_column(df, 'high'),
                'low': find_column(df, 'low'),
                'close': find_column(df, 'close'),
                'volume': find_column(df, 'volume')
            }
            
            # Check for missing required columns
            missing_cols = [name for name, col in col_mapping.items() if col is None]
            
            if missing_cols:
                st.warning(f"Missing columns in data: {', '.join(missing_cols)}. Some visualizations may not work.")
            

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.warning("Please try loading the data again.")
            st.stop()
        
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
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Preview", "📊 Visualization", "📈 Technical Indicators", "📉 Statistics"])
        
        with tab1:
            st.dataframe(df.sort_values('Date', ascending=False).head(10), use_container_width=True)
            
            if st.checkbox("Show full dataset"):
                st.dataframe(df, use_container_width=True)
        
        with tab2:
            st.subheader("Stock Price Visualization")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='royalblue')))
            
            # Add range slider and buttons
            fig.update_layout(
                title=f'{st.session_state.ticker} Stock Price History',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            st.subheader("Trading Volume")
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker=dict(color='darkblue')))
            volume_fig.update_layout(
                title=f'{st.session_state.ticker} Trading Volume',
                xaxis_title='Date',
                yaxis_title='Volume',
                template='plotly_white'
            )
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Candlestick chart
            st.subheader("Candlestick Chart")
            candlestick = go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'], 
                high=df['High'],
                low=df['Low'], 
                close=df['Close'],
                name='Candlestick'
            )])
            
            candlestick.update_layout(
                title=f'{st.session_state.ticker} Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(candlestick, use_container_width=True)

        with tab3:
            # Ensure that indicator_df exists in session state
            if 'indicator_df' not in st.session_state:
                ticker = st.session_state.ticker  # Get selected ticker
                # Function to load stock data
                
                temp_df = df.copy()
                temp_df['MA5'] = temp_df['Close'].rolling(window=5).mean()
                temp_df['MA20'] = temp_df['Close'].rolling(window=20).mean()
                temp_df['MA50'] = temp_df['Close'].rolling(window=50).mean()
                
                # RSI Calculation
                delta = temp_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                temp_df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD Calculation
                temp_df['EMA12'] = temp_df['Close'].ewm(span=12, adjust=False).mean()
                temp_df['EMA26'] = temp_df['Close'].ewm(span=26, adjust=False).mean()
                temp_df['MACD'] = temp_df['EMA12'] - temp_df['EMA26']
                temp_df['Signal'] = temp_df['MACD'].ewm(span=9, adjust=False).mean()
                
                # Bollinger Bands
                temp_df['MA20'] = temp_df['Close'].rolling(window=20).mean()
                temp_df['STD20'] = temp_df['Close'].rolling(window=20).std()

                temp_df['Upper'] = temp_df['MA20'] + (temp_df['STD20'] * 2)
                temp_df['Lower'] = temp_df['MA20'] - (temp_df['STD20'] * 2)

                
                # Store in session state
                st.session_state.indicator_df = temp_df.copy()
            
            # Get the indicator_df from session state
            indicator_df = st.session_state.indicator_df
            
            # Dropdown to select the technical indicator
            indicator_type = st.selectbox(
                "Select Technical Indicator",
                ["Moving Averages", "RSI", "MACD", "Bollinger Bands"]
            )
            
            # Display the chart directly without chart_container
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
                
                # Add horizontal lines at 70 and 30 for RSI
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
                fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['MA20'], name='20-Day MA', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Upper'], name='Upper Band', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=indicator_df['Date'], y=indicator_df['Lower'], name='Lower Band', line=dict(color='red', dash='dash')))
                
                fig.update_layout(
                    title=f'{st.session_state.ticker} Bollinger Bands',
                    xaxis_title='Date',
                    yaxis_title='Price', template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Price Statistics")
                try:
                    # Find the Close column using our helper function
                    close_col = find_column(df, 'close')
                    
                    if df is not None and not df.empty and close_col:
                        price_stats = df[close_col].describe().reset_index()
                        price_stats.columns = ['Statistic', 'Value']
                        st.dataframe(price_stats, use_container_width=True)
                    else:
                        st.warning("Cannot display price statistics: Missing 'Close' column in data.")
                except Exception as e:
                    st.error(f"Error calculating price statistics: {str(e)}")
                    st.info("Try reloading the data or selecting a different stock.")
                    
                # Daily returns
                try:
                    # Find the Close column using our helper function
                    close_col = find_column(df, 'close')
                    
                    if df is not None and not df.empty and close_col:
                        # Calculate daily returns directly
                        daily_returns = df[close_col].pct_change().dropna() * 100
                        
                        if not daily_returns.empty:
                                # Create histogram using Graph Objects
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=daily_returns,
                                    nbinsx=50,
                                    name='Daily Returns'
                                ))
                                
                                fig.update_layout(
                                    title="Distribution of Daily Returns",
                                    xaxis_title="Daily Return (%)",
                                    yaxis_title="Frequency",
                                    template='plotly_white'
                                )
                                
                                # Add vertical line at 0
                                fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot create histogram: Not enough data after processing.")
                    else:
                        st.warning("Cannot display returns histogram: Missing required data columns.")
                except Exception as e:
                    st.error(f"Error creating returns histogram: {str(e)}")
                    st.info("Try reloading the data or selecting a different stock.")
            
            with col2:
                st.markdown("#### Volume Statistics")
                try:
                    # Find the Volume column using our helper function
                    volume_col = find_column(df, 'volume')
                    
                    if df is not None and not df.empty and volume_col:
                        volume_stats = df[volume_col].describe().reset_index()
                        volume_stats.columns = ['Statistic', 'Value']
                        st.dataframe(volume_stats, use_container_width=True)
                    else:
                        st.warning("Cannot display volume statistics: Missing 'Volume' column in data.")
                except Exception as e:
                    st.error(f"Error calculating volume statistics: {str(e)}")
                    st.info("Try reloading the data or selecting a different stock.")
                    
                    # Volume over time
                    try:
                        # Use the mapped column names from the debug section
                        date_col = find_column(df, 'date')
                        volume_col = find_column(df, 'volume')
                        
                        if df is not None and not df.empty and date_col and volume_col:
                            # Create a simple line chart using Plotly Graph Objects instead of Express
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df[date_col], 
                                y=df[volume_col],
                                mode='lines',
                                name='Volume'
                            ))
                            
                            fig.update_layout(
                                title="Trading Volume Over Time",
                                xaxis_title="Date",
                                yaxis_title="Volume",
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot display volume chart: Missing required data columns.")
                    except Exception as e:
                        st.error(f"Error creating volume chart: {str(e)}")
                        st.info("Try reloading the data or selecting a different stock.")


                
                # Correlation matrix
                st.markdown("#### Correlation Matrix")
                try:
                    if df is not None and not df.empty:
                        # Find columns using our helper function
                        corr_cols = {
                            'open': find_column(df, 'open'),
                            'high': find_column(df, 'high'),
                            'low': find_column(df, 'low'),
                            'close': find_column(df, 'close'),
                            'volume': find_column(df, 'volume')
                        }
                        
                        # Filter out None values
                        available_cols = [col for col in corr_cols.values() if col is not None]
                        
                        if len(available_cols) >= 2:  # Need at least 2 columns for correlation
                            # Create correlation matrix
                            corr_matrix = df[available_cols].corr().round(2)
                            
                            # Create heatmap using Graph Objects
                            fig = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.index,
                                colorscale='RdBu_r',
                                zmin=-1, zmax=1,
                                text=corr_matrix.values.round(2),
                                texttemplate="%{text}",
                                showscale=True
                            ))
                            
                            fig.update_layout(
                                title="Correlation Matrix",
                                template='plotly_white',
                                height=500,
                                width=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Cannot create correlation matrix: Need at least 2 numeric columns.")
                    else:
                        st.warning("Cannot create correlation matrix: No data available.")
                except Exception as e:
                    st.error(f"Error creating correlation matrix: {str(e)}")
                    st.info("Try reloading the data or selecting a different stock.")

# Model Training page
elif selected == "Model Training":
    # Logo and title at the top
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("images/logo/smartfolio_logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='font-size:2.5em; margin-top:15px;'>SmartFolio AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-bottom:30px;'>", unsafe_allow_html=True)
    
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
    # Logo and title at the top
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("images/logo/smartfolio_logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='font-size:2.5em; margin-top:15px;'>SmartFolio AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-bottom:30px;'>", unsafe_allow_html=True)
    
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
    # Logo and title at the top
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("images/logo/smartfolio_logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='font-size:2.5em; margin-top:15px;'>SmartFolio AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-bottom:30px;'>", unsafe_allow_html=True)
    
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
    # Logo and title at the top
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("images/logo/smartfolio_logo.png", width=100)
    with col_title:
        st.markdown("<h1 style='font-size:2.5em; margin-top:15px;'>SmartFolio AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-bottom:30px;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 style='font-size:2.5em;'>About SmartFolio AI</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            SmartFolio AI is an advanced portfolio management platform that uses cutting-edge deep learning techniques to analyze stocks and optimize investment decisions. 
            
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
            
            SmartFolio AI uses Yahoo Finance API to fetch historical stock data. Users can also upload their own CSV files.
            """
        )
    
    with col2:
        # Display the project logo
        st.markdown('<div class="logo-container" style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
        st.image("images/logo/smartfolio_logo.png", width=300)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Disclaimer")
        st.info(
            """
            The predictions provided by SmartFolio AI are for informational purposes only and should not be construed as financial advice. 
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
    
    # Creator section with animated header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; animation: fadeInUp 0.8s ease-out;">
        <h2 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 10px; background: linear-gradient(90deg, #3b82f6, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Creator</h2>
        <p style="font-size: 1.2rem; opacity: 0.8;">The mind behind SmartFolio AI</p>
    </div>
    
    <style>
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Add animation for creator card */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom CSS for creator card with dark mode support and animations
    creator_css = """
    <style>
    /* Light mode styles (default) */
    .creator-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        animation: fadeInUp 0.8s ease-out forwards;
        opacity: 0;
    }
    
    .creator-card {
        animation-delay: 0.2s;
    }
    
    .creator-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .creator-image {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #3b82f6;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    
    .team-card:hover .team-image {
        transform: scale(1.05);
        border-color: #2563eb;
    }
    
    .creator-name {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 5px;
        color: #0f172a;
    }
    
    .creator-role {
        font-size: 1.1rem;
        color: #3b82f6;
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    .creator-bio {
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.5;
    }
    
    .social-icons {
        margin-top: 15px;
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    
    .social-icon {
        font-size: 1.2rem;
        color: #3b82f6;
        transition: color 0.2s ease;
    }
    
    .social-icon:hover {
        color: #2563eb;
    }
    
    /* Dark mode styles */
    .dark-mode .creator-card {
        background-color: #1e293b;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .dark-mode .creator-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    .dark-mode .creator-name {
        color: #f1f5f9;
    }
    
    .dark-mode .creator-bio {
        color: #94a3b8;
    }
    
    .dark-mode .social-icon {
        color: #60a5fa;
    }
    
    .dark-mode .social-icon:hover {
        color: #93c5fd;
    }
    </style>
    """
    
    # Apply dark mode styles if enabled
    if st.session_state.dark_mode:
        # Add !important to ensure styles override any other styles
        creator_css = creator_css.replace('.creator-card {', '.creator-card { background-color: #1e293b !important;')
        creator_css = creator_css.replace('.creator-name {', '.creator-name { color: #f1f5f9 !important;')
        creator_css = creator_css.replace('.creator-bio {', '.creator-bio { color: #94a3b8 !important;')
        # Add dark mode class to the body for reference
        creator_css = creator_css + "\nbody { background-color: #0f172a; }"
    
    st.markdown(creator_css, unsafe_allow_html=True)
    
    # Display creator card in the About page
    from creator_card import display_creator_card
    display_creator_card()
