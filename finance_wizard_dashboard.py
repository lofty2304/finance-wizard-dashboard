import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import requests
import yfinance as yf

# --- Streamlit UI ---
st.set_page_config(page_title="Finance Wizard (Deployed)", layout="centered")
st.title("🧙 Finance Wizard: Stock & NAV Predictor + Trend Reversals")

asset_type = st.selectbox("Choose Asset Type", ["Stock", "Mutual Fund"])
symbol_or_slug = st.text_input("Enter Ticker Symbol (e.g. INFY.NS) or Groww Slug (e.g. axis-small-cap-direct-growth)")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)

hotkey = st.selectbox("Prediction Strategy", [
    "🔮 W - Technical Price Prediction",
    "🧠 ML - Machine Learning-Based Forecast",
    "🌐 TA - Trend Analysis",
    "🔀 TE - Trend Reversal Detection"
])

# --- Function: Get Stock Price using yfinance ---
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="60d")
    df = df.reset_index()
    return df[["date", "price"]].sort_values("date")
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("Please enter a valid input.")
    else:
        st.write("🔄 Fetching data...")
        if asset_type == "Stock":
            df = get_stock_price(symbol_or_slug)
            label = f"{symbol_or_slug.upper()} Stock"
        else:
            df = get_nav_data(symbol_or_slug)
            label = f"{symbol_or_slug.replace('-', ' ').title()} NAV"

        # New debug info
        if df is None or df.empty:
            st.error("❌ No data found. Check your input format.")
            st.write("📦 Raw DataFrame: ", df)
        else:
            st.success("✅ Data successfully fetched.")
            st.write("📊 Sample Data:", df.tail())
            analyze_and_predict(df, days_ahead, label)

