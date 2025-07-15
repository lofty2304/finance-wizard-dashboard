import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import r2_score
import joblib
import os
from datetime import datetime
import yfinance as yf
import requests
import openai
import finnhub
import ta

# --- API KEYS ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# --- Cache directory ---
CACHE_DIR = "cached_models"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- UI CONFIG ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")

# --- Sidebar options ---
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 7-Day Forecast", value=True)

strategy = st.selectbox("🧠 Strategy", [
    "🔮 W - Predict One Stock",
    "🧠 ML - Polynomial Forecast",
    "⚖️ MC - ML Model Comparison",
    "🧐 MD - ML Model Explanation",
    "📉 D - Downside Risk",
    "🕵️ S - Stock Deep Dive",
    "📉 TI - Technical Indicator Forecast",
    "↔️ TC - Compare Indicators",
    "💡 TD - Explain Indicators",
    "⚠️ TE - Indicator Limitations",
    "🟢 SA - Optimistic Scenario",
    "🟡 SC - Conservative Scenario",
    "🔴 SD - Pessimistic Scenario",
    "⚖️ SE - Extreme Shock"
])

# --- User Input ---
fund_name_input = st.text_input(
    "Enter Fund Name or Ticker (e.g. AAPL, INFY.NS, Axis Global Equity Alpha Fund)",
    "NBCC.NS"
)

# --- Resolve Fund Name to Ticker using GPT ---
def resolve_fund_name_to_ticker(name):
    if "." in name and len(name) < 10:
        return name.upper()
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Convert Indian mutual fund names to Yahoo Finance-compatible tickers. Reply only with the ticker symbol (no explanation)."},
                {"role": "user", "content": f"What is the Yahoo Finance ticker for {name}?"}
            ]
        )
        ticker = res["choices"][0]["message"]["content"].strip().upper()
        if len(ticker) < 2:
            return name.upper()
        return ticker
    except:
        return name.upper()

resolved = resolve_fund_name_to_ticker(fund_name_input)
st.caption(f"🧠 Resolved Ticker: **{resolved}** (via AI)")

days_ahead = st.slider("📆 Days Ahead to Forecast", 1, 30, 7)

# --- Fetch NAV from Multiple Sources ---
@st.cache_data(ttl=300)
def get_live_nav(symbol):
    try:
        info = yf.Ticker(symbol).info
        nav = info.get("navPrice") or info.get("regularMarketPrice")
        if nav and nav > 0:
            return round(nav, 2), "Yahoo Finance"
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=5).text
        for line in txt.splitlines():
            if symbol.upper() in line:
                val = float(line.split(";")[-1])
                if val > 0:
                    return round(val, 2), "AMFI India"
    except: pass
    return None, "Unavailable"

# --- Get Historical Price or Fallback to Dummy NAV Chart ---
def get_stock_price(symbol, live_nav):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        if not df.empty:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["date", "price"]
            return df, "Yahoo Finance"
    except: pass
    if live_nav:
        dates = pd.date_range(end=datetime.today(), periods=90)
        df = pd.DataFrame({"date": dates, "price": [live_nav] * 90})
        return df, "NAV-based Dummy Data"
    return None, "Unavailable"
def predict_price(df, days_ahead):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    future_index = X[-1][0] + days_ahead
    pred = model.predict([[future_index]])[0]
    std_dev = np.std(y - model.predict(X))
    return pred, pred - 1.96 * std_dev, pred + 1.96 * std_dev

def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score financial sentiment from -1 to 1."},
                {"role": "user", "content": text}
            ]
        )
        return float(res["choices"][0]["message"]["content"].strip())
    except:
        return 0

def analyze_and_predict(df, strategy_code, days_ahead, symbol, live_nav, nav_source):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    df["MA5"] = df["price"].rolling(5).mean()
    df["Live_NAV"] = live_nav
    st.caption(f"📊 NAV Source: {nav_source}")

    if strategy_code == "W":
        pred, low, high = predict_price(df, days_ahead)
        st.metric("Prediction", f"₹{round(pred, 2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")
        st.dataframe(df.tail(7))

    elif strategy_code == "ML":
        model_poly = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        try:
            df_p = df.rename(columns={"date": "ds", "price": "y"})
            prophet = Prophet().fit(df_p)
            future = prophet.make_future_dataframe(periods=days_ahead)
            forecast = prophet.predict(future)
            yhat = round(forecast.iloc[-1]["yhat"], 2)
        except:
            yhat = "N/A"
        st.metric("Polynomial", round(model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0], 2))
        st.metric("Random Forest", round(rf.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("XGBoost", round(xgb_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("Prophet", yhat)

    elif strategy_code == "D":
        pred, _, _ = predict_price(df, days_ahead)
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.warning(f"Downside: ₹{round(downside, 2)} | Volatility: {round(vol*100, 2)}%")

    elif strategy_code == "S":
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail())
        score = get_sentiment_score(symbol)
        st.metric("Sentiment Score", round(score, 2))

    elif strategy_code == "TI":
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["price"])
        df["BB_H"] = bb.bollinger_hband()
        df["BB_L"] = bb.bollinger_lband()
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail())

    # Add additional strategies here...

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], linestyle="--", label="MA5")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)

# --- Run Button ---
if st.button("Run Strategy"):
    nav, nav_source = get_live_nav(resolved)
    df, fetch_source = get_stock_price(resolved, nav)
    if df is None or df.empty:
        st.error("❌ No data found. Try another fund or stock.")
    else:
        strategy_code = strategy.split("-")[0].strip().split()[-1]
        analyze_and_predict(df, strategy_code, days_ahead, resolved, nav, nav_source)
