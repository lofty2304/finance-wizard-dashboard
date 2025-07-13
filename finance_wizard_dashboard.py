import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import yfinance as yf
import requests

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
    if df.empty:
        return None
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df.rename(columns={"Date": "date", "Close": "price"}, inplace=True)
    return df

# --- Function: Get NAV from Groww API ---
def get_nav_data(slug):
    try:
        url = f"https://groww.in/v1/api/mf/v1/scheme/{slug}"
        res = requests.get(url)
        data = res.json()
        navs = data["navHistory"]
        df = pd.DataFrame(navs)
        df["date"] = pd.to_datetime(df["navDate"])
        df["price"] = df["nav"]
        return df[["date", "price"]].sort_values("date")
    except:
        return None

# --- Analysis + Prediction Function ---
def analyze_and_predict(df, days_ahead, label):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values

    model = LinearRegression().fit(X, y)
    pred_index = df["day_index"].max() + days_ahead
    pred_price = model.predict([[pred_index]])[0]

    # Confidence interval
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_dev = np.std(residuals)
    upper = pred_price + 1.96 * std_dev
    lower = pred_price - 1.96 * std_dev

    # Trend reversal
    df["MA5"] = df["price"].rolling(window=5).mean()
    reversal = False
    if len(df["MA5"].dropna()) >= 3:
        slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
        slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
        if np.sign(slope1) != np.sign(slope2):
            reversal = True

    # Chart
    st.subheader(f"📊 {label} — Price Forecast")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price", marker="o")
    ax.plot(df["date"], df["MA5"], label="MA-5", linestyle="--", color="orange")
    ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead),
                pred_price, yerr=1.96 * std_dev, fmt='ro', label="Predicted ±95% CI")
    ax.legend()
    st.pyplot(fig)

    # Output
    st.success(f"🔮 Predicted in {days_ahead} days: ₹{round(pred_price,2)}")
    st.info(f"📈 95% Confidence Interval: ₹{round(lower,2)} – ₹{round(upper,2)}")
    if reversal:
        st.error("⚠️ Potential Trend Reversal Detected!")
    else:
        st.success("✅ Trend appears stable.")

# --- Main Logic ---
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("Please enter a valid input.")
    else:
        if asset_type == "Stock":
            df = get_stock_price(symbol_or_slug)
            label = f"{symbol_or_slug.upper()} Stock"
        else:
            df = get_nav_data(symbol_or_slug)
            label = f"{symbol_or_slug.replace('-', ' ').title()} NAV"

        if df is None or df.empty:
            st.error("❌ Failed to fetch data. Check the symbol or try again later.")
            st.write("📦 Raw Output:", df)
        else:
            st.success("✅ Data successfully fetched.")
            st.write("📊 Sample Data:", df.tail())
            analyze_and_predict(df, days_ahead, label)
