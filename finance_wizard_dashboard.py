# --- Finance Wizard: Final Integrated Master Code (Unified Version) ---
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
# --- Data Fetching Functions ---
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

# --- Sentiment Analysis ---
@st.cache_data(ttl=600)
def get_sentiment_score(query):
    try:
        news_api_key = st.secrets["NEWS_API_KEY"]
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=5&sortBy=publishedAt&apiKey={news_api_key}"
        data = requests.get(url).json()
        headlines = [article["title"] for article in data["articles"]]
        prompt = " ".join(headlines) + "\n\nClassify sentiment as Positive, Neutral, or Negative with score from 0 (neg) to 1 (pos)."
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial sentiment classifier."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = res["choices"][0]["message"]["content"]
        score = 0.5
        if "positive" in answer.lower(): score = 0.7
        elif "negative" in answer.lower(): score = 0.3
        return round(score, 2), answer.strip()
    except:
        return 0.5, "Neutral (fallback)"

# --- Forecast Models ---
def random_forest_forecast(df, days_ahead):
    model = RandomForestRegressor()
    model.fit(df[["day_index"]], df["price"])
    future_days = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days_ahead).reshape(-1, 1)
    pred = model.predict(future_days)
    return pred.tolist(), model.score(df[["day_index"]], df["price"])

def xgboost_forecast(df, days_ahead):
    model = xgb.XGBRegressor()
    model.fit(df[["day_index"]], df["price"])
    future_days = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days_ahead).reshape(-1, 1)
    pred = model.predict(future_days)
    return pred.tolist(), model.score(df[["day_index"]], df["price"])

def prophet_forecast(df, days_ahead):
    prophet_df = df[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    return forecast.tail(days_ahead)["yhat"].tolist(), model

# --- Scenario Logic ---
def adjust_forecast(base_forecast, sentiment_score, scenario_code):
    multiplier = {
        "SA": 1.05 + sentiment_score * 0.1,
        "SC": 1.02 + sentiment_score * 0.05,
        "SD": 0.98 - sentiment_score * 0.05,
        "SE": 0.90 - sentiment_score * 0.1
    }.get(scenario_code, 1)
    return [round(p * multiplier, 2) for p in base_forecast]

# --- Strategy Handler ---
def run_strategy(code, df, days_ahead, nav, nav_source, resolved, show_r2=True):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["MA5"] = df["price"].rolling(5).mean()

    if code in ["SA", "SC", "SD", "SE"]:
        st.subheader("🧭 Scenario Forecasting")
        sentiment_score, raw = get_sentiment_score(resolved)
        base_pred, _ = linear_forecast(df, days_ahead)
        adjusted = adjust_forecast(base_pred, sentiment_score, code)
        plot_main_graph(df, adjusted, title=f"{code} Scenario Forecast")
        st.metric("🔮 Sentiment Score", f"{sentiment_score} / 1.0")
        st.caption(f"📰 {raw}")
        return

    elif code == "W":
        st.subheader("🔮 Forecast with Prophet Model")
        pred, model = prophet_forecast(df, days_ahead)
        plot_main_graph(df, pred, title="Prophet Forecast")
        st.caption(f"Forecast from {df['date'].iloc[-1].date()} to {(df['date'].iloc[-1] + timedelta(days=days_ahead)).date()}.")

    elif code == "ML":
        st.subheader("🧠 Polynomial Forecast (Degree 3)")
        pred, r2 = polynomial_forecast(df, days_ahead)
        plot_main_graph(df, pred, title="Polynomial Forecast")
        if show_r2: st.metric("R² Score", round(r2, 4))

    elif code == "MC":
        st.subheader("⚖️ Model Comparison (Last Point)")
        pred_lr, r2_lr = linear_forecast(df, days_ahead)
        pred_poly, r2_poly = polynomial_forecast(df, days_ahead)
        pred_rf, r2_rf = random_forest_forecast(df, days_ahead)
        pred_xgb, r2_xgb = xgboost_forecast(df, days_ahead)
        st.metric("Linear", f"{pred_lr[-1]:.2f} | R²: {r2_lr:.2f}")
        st.metric("Poly", f"{pred_poly[-1]:.2f} | R²: {r2_poly:.2f}")
        st.metric("RF", f"{pred_rf[-1]:.2f} | R²: {r2_rf:.2f}")
        st.metric("XGB", f"{pred_xgb[-1]:.2f} | R²: {r2_xgb:.2f}")

    elif code == "DL":
        st.subheader("🧠 LSTM Deep Learning Forecast")
        price = lstm_forecast(df, days_ahead)
        st.metric("Predicted Price", f"₹{price}")
        plot_main_graph(df)

    elif code == "SIP":
        st.subheader("💹 SIP Return Engine")
        monthly_sip = st.number_input("💰 Monthly SIP (₹)", min_value=100, value=1000, step=100)
        term = st.slider("📆 SIP Term (Months)", 3, 60, 12)
        invested, value, cagr = simulate_sip(df, monthly_sip, term)
        st.metric("Total Invested", f"₹{invested}")
        st.metric("Final Value", f"₹{value}")
        st.metric("CAGR", f"{cagr}%")
        plot_main_graph(df)

    elif code == "BK":
        benchmark = get_benchmark_df(180)
        if benchmark is None:
            st.error("⚠️ Benchmark data not found.")
            return
        df["norm"] = df["price"] / df["price"].iloc[0]
        benchmark["norm"] = benchmark["price"] / benchmark["price"].iloc[0]
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["norm"], label=resolved)
        ax.plot(benchmark["date"], benchmark["norm"], label="NIFTY 50")
        ax.legend(); st.pyplot(fig)

    elif code == "PC":
        st.subheader("📊 SIP vs LSTM Comparison")
        amt = st.number_input("SIP (₹)", 100, 10000, 1000, 100)
        term = st.slider("Term (Months)", 3, 60, 12)
        invested, value, cagr = simulate_sip(df, amt, term)
        lstm_pred = lstm_forecast(df, days_ahead)
        st.metric("SIP Final", f"₹{value}"); st.metric("CAGR", f"{cagr}%")
        st.metric("LSTM Forecast", f"₹{lstm_pred}")
        plot_main_graph(df)

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
