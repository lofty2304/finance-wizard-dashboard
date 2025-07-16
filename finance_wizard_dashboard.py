# --- Integrated Finance Wizard Master File (Part 1/2) ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import r2_score
import ta
import yfinance as yf
import openai
import requests
from datetime import datetime, timedelta
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# --- API Keys ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- UI Setup ---
st_autorefresh(interval=600000, key="auto_refresh")
st.set_page_config(page_title="Finance Wizard", layout="wide")
st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://i.imgur.com/1N6y4WQ.png" width="80"/>
        <div style="margin-left: 10px;">
            <h1 style="margin: 0;">🧙 Finance Wizard Lucia</h1>
            <p style="margin: 0;">AI-Powered Forecasts, NAV Insights & Global Sentiment</p>
        </div>
    </div>
""", unsafe_allow_html=True)
st.caption("🧡 Empowering clients with 22 years of trust and transparency.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    days_ahead = st.slider("📅 Days Ahead to Forecast", 1, 30, 7)
    show_r2 = st.checkbox("📊 Show R² Scores", value=True)
    simulate_runs = st.slider("🔁 Multi-run Simulation Count", 1, 20, 1)
    st.markdown("Each strategy uses this value to forecast.")

strategy = st.selectbox("📌 Choose Forecasting Strategy", [
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
    "📈 BK - Backtest vs Benchmark",
    "💹 SIP - SIP Return Engine",
    "🧠 DL - LSTM Forecasting",
    "📊 PC - Profit vs SIP Comparison",
    "🟢 SA - Optimistic Scenario",
    "🟡 SC - Conservative Scenario",
    "🔴 SD - Pessimistic Scenario",
    "⚖️ SE - Extreme Shock"
])

st.markdown("## 🔎 Enter Stock / Fund Symbol")
user_input = st.text_input("Ticker / Fund / AMFI Code", "NBCC.NS")

def resolve_ticker(name):
    if "." in name and len(name) < 10:
        return name.upper()
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Convert Indian mutual fund names to Yahoo tickers."},
                {"role": "user", "content": f"What is the ticker for {name}?"}
            ]
        )
        return res["choices"][0]["message"]["content"].strip().upper()
    except:
        return name.upper()

resolved = resolve_ticker(user_input)
st.caption(f"🧾 Resolved Ticker: **{resolved}**")
# --- Data Utilities ---
@st.cache_data(ttl=300)
def get_live_nav(ticker):
    try:
        info = yf.Ticker(ticker).info
        nav = info.get("navPrice") or info.get("regularMarketPrice")
        if nav: return round(nav, 2), "Yahoo Finance"
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt").text
        for line in txt.splitlines():
            if ticker.upper() in line:
                val = float(line.split(";")[-1])
                return round(val, 2), "AMFI India"
    except: pass
    return None, "Unavailable"

def get_stock_price(symbol, fallback_nav):
    try:
        df = yf.Ticker(symbol).history(period="180d")
        if not df.empty:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["date", "price"]
            return df
    except: pass
    if fallback_nav:
        dates = pd.date_range(end=datetime.today(), periods=90)
        df = pd.DataFrame({"date": dates, "price": [fallback_nav] * 90})
        return df
    return None

def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Score financial sentiment from -1 to 1."},
                      {"role": "user", "content": text}]
        )
        return float(res["choices"][0]["message"]["content"].strip())
    except: return 0

def fetch_news_sentiment(symbol):
    try:
        url = "https://newsapi.org/v2/everything"
        res = requests.get(url, params={
            "q": symbol, "language": "en", "sortBy": "publishedAt",
            "apiKey": st.secrets["NEWS_API_KEY"]
        }).json()
        articles = res.get("articles", [])[:5]
        texts = [a["title"] + " " + a.get("description", "") for a in articles]
        scores = [get_sentiment_score(txt) for txt in texts]
        return round(np.mean(scores), 2) if scores else 0
    except: return 0

# --- LSTM Model ---
def lstm_forecast(df, days_ahead):
    data = df["price"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(data_scaled) - days_ahead):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i + days_ahead - 1])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=False, input_shape=(60, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    test_input = data_scaled[-60:].reshape(1, 60, 1)
    pred_scaled = model.predict(test_input)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    return round(pred, 2)

# --- SIP Return Logic ---
def run_sip_backtest(df):
    sip_amount = 1000
    dates = df["date"]
    invested = len(dates) * sip_amount
    units = [sip_amount / p if p else 0 for p in df["price"]]
    final_price = df["price"].iloc[-1]
    value = sum(units) * final_price
    return round(invested, 2), round(value, 2), round((value - invested) / invested * 100, 2)

# --- Strategy Execution ---
def run_strategy(code, df, days_ahead, nav, nav_source, resolved, show_r2=True):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["MA5"] = df["price"].rolling(5).mean()

    def plot_main_graph(extra=None):
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["price"], label="Price")
        ax.plot(df["date"], df["MA5"], label="MA5")
        if extra is not None:
            future_dates = [df["date"].iloc[-1] + timedelta(days=i+1) for i in range(len(extra))]
            ax.plot(future_dates, extra, label="Forecast", linestyle="--")
        ax.legend(); ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    if code == "DL":
        st.subheader("🧠 LSTM Forecasting")
        pred = lstm_forecast(df, days_ahead)
        st.metric("🔮 LSTM Prediction", f"₹{pred}")
        plot_main_graph()

    elif code == "SIP":
        st.subheader("💹 SIP Return Analysis")
        invested, value, return_pct = run_sip_backtest(df)
        st.metric("📥 Total Invested", f"₹{invested}")
        st.metric("📈 Final Value", f"₹{value}")
        st.metric("💰 Return (%)", f"{return_pct}%")
        plot_main_graph()

    elif code == "BK":
        st.subheader("📈 Backtest Engine")
        invested, value, return_pct = run_sip_backtest(df)
        st.markdown(f"Backtest return on ₹1000 monthly: **₹{value}** from ₹{invested} → **{return_pct}%**")
        plot_main_graph()

    elif code == "PC":
        st.subheader("📊 SIP vs LSTM Comparison")
        pred = lstm_forecast(df, days_ahead)
        invested, value, ret = run_sip_backtest(df)
        st.metric("SIP Final Value", f"₹{value}")
        st.metric("LSTM Predicted Price", f"₹{pred}")
        st.markdown("LSTM is point forecast. SIP is periodic. Use both.")

    elif code in ["SA", "SC", "SD", "SE"]:
        st.subheader(f"🔁 Scenario Simulation: {code}")
        scenario_map = {"SA": 1.02, "SC": 1.00, "SD": 0.98, "SE": 0.95}
        growth = scenario_map.get(code, 1.0)
        prices = [df["price"].iloc[-1] * (growth ** i) for i in range(days_ahead)]
        plot_main_graph(extra=prices)
        st.metric("Scenario Growth Factor", f"{growth}x")

    elif code == "TD":
        st.subheader("💡 Indicator Explanation")
        st.markdown("""
        - **RSI**: Relative Strength Index (momentum, overbought/sold)
        - **MACD**: Trend strength using two EMAs
        - **MA5**: Moving Average over 5 days (short trend)
        """)

    elif code == "TE":
        st.subheader("⚠️ Indicator Limitations")
        st.markdown("""
        - RSI may give false signals in strong trends
        - MACD lags behind price
        - MA5 is sensitive to noise
        """)

    elif code == "MD":
        st.subheader("🧐 Model Explanation")
        st.markdown("""
        - **Linear Regression**: Simple trend line
        - **Polynomial**: Captures non-linear curves
        - **Random Forest**: Ensemble of decision trees
        - **XGBoost**: Gradient boosting model
        - **Prophet**: Facebook's time series model
        - **LSTM**: Deep learning for sequential data
        """)

    elif code == "TI":
        st.subheader("📉 Technical Indicator Forecast")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        st.line_chart(df.set_index("date")[["price", "RSI"]])

    elif code == "TC":
        st.subheader("↔️ Compare Indicators")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["price"]).macd()
        st.line_chart(df.set_index("date")[["RSI", "MACD"]])

    elif code == "S":
        st.subheader("🕵️ Deep Dive")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["price"]).macd()
        df["Signal"] = ta.trend.MACD(df["price"]).macd_signal()
        score = fetch_news_sentiment(resolved)
        st.metric("📰 News Sentiment", f"{score}")
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail())
        plot_main_graph()

    else:
        st.warning("⚠️ Strategy not implemented yet or invalid code.")

# --- Main Run Button ---
if st.button("🚀 Run Strategy"):
    nav, nav_source = get_live_nav(resolved)
    df = get_stock_price(resolved, nav)
    if df is None or df.empty:
        st.error("❌ No data found. Try another stock or fund.")
    else:
        strategy_code = strategy.split("-")[0].split()[-1].strip()
        for i in range(simulate_runs):
            if simulate_runs > 1:
                st.markdown(f"### Simulation {i+1}/{simulate_runs}")
            run_strategy(strategy_code, df.copy(), days_ahead, nav, nav_source, resolved, show_r2)
