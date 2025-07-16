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
import ta
import yfinance as yf
import openai
import requests
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense

# --- API Key ---
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

# --- Data Functions ---
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
        return pd.DataFrame({"date": dates, "price": [fallback_nav] * 90})
    return None

# --- Forecast Models ---
def linear_forecast(df, days_ahead):
    model = LinearRegression()
    model.fit(df[["day_index"]], df["price"])
    future_days = np.arange(df["day_index"].max()+1, df["day_index"].max()+1 + days_ahead).reshape(-1, 1)
    pred = model.predict(future_days)
    return pred.tolist(), model.score(df[["day_index"]], df["price"])

def polynomial_forecast(df, days_ahead, degree=3):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(df[["day_index"]])
    model = LinearRegression()
    model.fit(X_poly, df["price"])
    future_days = np.arange(df["day_index"].max()+1, df["day_index"].max()+1 + days_ahead).reshape(-1, 1)
    pred = model.predict(poly.transform(future_days))
    return pred.tolist(), model.score(X_poly, df["price"])

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
    model.add(LSTM(50, activation='relu', input_shape=(60, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    test_input = data_scaled[-60:].reshape(1, 60, 1)
    pred = model.predict(test_input)
    return round(scaler.inverse_transform(pred)[0][0], 2)

# --- Utility Plots & Calculations ---
def plot_main_graph(df, forecast=None, title="Forecast"):
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Historical Price")
    if "MA5" in df.columns:
        ax.plot(df["date"], df["MA5"], label="MA5")
    if forecast is not None:
        last_date = df["date"].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
        ax.plot(future_dates, forecast, label=title, linestyle="--")
    ax.legend(); ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

def calculate_cagr(start_value, end_value, periods):
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return 0.0
    return round(((end_value / start_value) ** (1 / (periods / 12))) - 1, 4) * 100

def simulate_sip(df, sip_amount=1000, months=12):
    df = df.copy().iloc[-months:]
    if df.empty: return 0, 0, 0
    invested = sip_amount * months
    units = [sip_amount / p if p else 0 for p in df["price"]]
    final_value = sum(units) * df["price"].iloc[-1]
    cagr = calculate_cagr(sip_amount, final_value/months, months)
    return round(invested, 2), round(final_value, 2), round(cagr, 2)

def get_benchmark_df(days):
    try:
        df = yf.Ticker("^NSEI").history(period=f"{days}d").reset_index()
        df = df[["Date", "Close"]]
        df.columns = ["date", "price"]
        return df
    except:
        return None

# --- Strategy Handler ---
def run_strategy(code, df, days_ahead, nav, nav_source, resolved, show_r2=True):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["MA5"] = df["price"].rolling(5).mean()

    if code == "PC":
        st.subheader("📊 SIP vs LSTM Forecast Comparison")
        monthly_sip = st.number_input("💰 Monthly SIP (₹)", min_value=100, value=1000, step=100)
        sip_months = st.slider("📆 SIP Term (Months)", 3, 60, 12)
        invested, value, cagr = simulate_sip(df, monthly_sip, sip_months)
        lstm_pred = lstm_forecast(df, days_ahead)
        plot_main_graph(df)
        st.metric("Total SIP Invested", f"₹{invested}")
        st.metric("Final SIP Value", f"₹{value}")
        st.metric("SIP CAGR", f"{cagr}%")
        st.metric("LSTM Predicted Price", f"₹{lstm_pred}")
        st.markdown("""
        - **SIP logic** assumes monthly fixed investment over selected term.  
        - **LSTM** is a one-shot forecast for future price.
        """)

    elif code == "BK":
        st.subheader("📈 Backtest vs Benchmark")
        benchmark = get_benchmark_df(180)
        if benchmark is None:
            st.error("⚠️ Could not fetch benchmark data.")
            return
        df["norm"] = df["price"] / df["price"].iloc[0]
        benchmark["norm"] = benchmark["price"] / benchmark["price"].iloc[0]
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["norm"], label=resolved)
        ax.plot(benchmark["date"], benchmark["norm"], label="NIFTY 50")
        ax.legend(); ax.set_title("Performance Comparison"); st.pyplot(fig)
        asset_ret = df["price"].iloc[-1] / df["price"].iloc[0] - 1
        nifty_ret = benchmark["price"].iloc[-1] / benchmark["price"].iloc[0] - 1
        asset_cagr = calculate_cagr(df["price"].iloc[0], df["price"].iloc[-1], 6)
        nifty_cagr = calculate_cagr(benchmark["price"].iloc[0], benchmark["price"].iloc[-1], 6)
        st.metric(f"{resolved} Return", f"{round(asset_ret*100,2)}%")
        st.metric("NIFTY Return", f"{round(nifty_ret*100,2)}%")
        st.metric(f"{resolved} CAGR", f"{round(asset_cagr,2)}%")
        st.metric("NIFTY CAGR", f"{round(nifty_cagr,2)}%")

# --- Run Trigger ---
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
