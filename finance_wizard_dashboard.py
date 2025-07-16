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
import ta
import yfinance as yf
import openai
import finnhub
import requests
import os
from datetime import datetime

# --- API Keys ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# --- Auto-refresh sentiment every 10 minutes ---
st_autorefresh(interval=600000, key="auto_refresh")  # 10 minutes

# --- Page Config ---
st.set_page_config(page_title="Finance Wizard", layout="centered")

# --- Header ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("wizard_couple.png.JPG", width=100)
with col2:
    st.markdown("# 🧙 Finance Wizard Lucia")
    st.markdown("**AI-driven market intelligence, forecasts & NAV sync**")
st.caption("🧡 22 years of love, trust & analysis.")

# --- Sidebar options ---
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    st.markdown("📅 Forecast Horizon (Days):")
    days_ahead = st.slider("Days", 1, 30, 7)

# --- Strategy Dropdown ---
strategy = st.selectbox("Choose Strategy", [
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

# --- Ticker input ---
st.markdown("### 🔎 Enter Fund or Stock Symbol")
user_input = st.text_input("Ticker / Fund / AMFI Code", "NBCC.NS")

# --- Ticker Resolver (using OpenAI if ambiguous) ---
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
st.caption(f"Resolved Ticker: **{resolved}**")

# --- Get live NAV ---
@st.cache_data(ttl=300)
def get_live_nav(ticker):
    try:
        info = yf.Ticker(ticker).info
        nav = info.get("navPrice") or info.get("regularMarketPrice")
        if nav:
            return round(nav, 2), "Yahoo Finance"
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=5).text
        for line in txt.splitlines():
            if ticker.upper() in line:
                val = float(line.split(";")[-1])
                return round(val, 2), "AMFI India"
    except: pass
    return None, "Unavailable"

# --- Get 90-day price data ---
def get_stock_price(symbol, fallback_nav):
    try:
        df = yf.Ticker(symbol).history(period="90d")
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
# --- Forecast helper ---
def predict_price(df, days_ahead):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    future_index = X[-1][0] + days_ahead
    pred = model.predict([[future_index]])[0]
    std = np.std(y - model.predict(X))
    return pred, pred - 1.96 * std, pred + 1.96 * std

# --- Sentiment score via OpenAI ---
def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score financial sentiment from -1 (very negative) to 1 (very positive)."},
                {"role": "user", "content": text}
            ]
        )
        return float(res["choices"][0]["message"]["content"].strip())
    except:
        return 0

def fetch_news_sentiment(symbol):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": st.secrets["NEWS_API_KEY"]
        }
        res = requests.get(url, params=params).json()
        articles = res.get("articles", [])[:5]
        texts = [a["title"] + " " + a.get("description", "") for a in articles]
        scores = [get_sentiment_score(txt) for txt in texts]
        return round(np.mean(scores), 2) if scores else 0
    except:
        return 0

# --- Main Logic Handler ---
def run_strategy(strategy_code, df, days_ahead, nav, source):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["MA5"] = df["price"].rolling(5).mean()
    df["Live_NAV"] = nav

    st.caption(f"📊 Live NAV Source: {source}")

    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    if strategy_code == "W":
        pred, low, high = predict_price(df, days_ahead)
        st.metric("Forecast Price", f"₹{round(pred, 2)}")
        st.info(f"95% Confidence Interval: ₹{round(low)} – ₹{round(high)}")

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

    elif strategy_code == "MC":
        st.subheader("📈 Model Comparison (ML vs Prophet)")

        lin = LinearRegression().fit(X, y)
        poly_model = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)

        try:
            prop_df = df.rename(columns={"date": "ds", "price": "y"})
            prop = Prophet().fit(prop_df)
            forecast = prop.predict(prop.make_future_dataframe(periods=days_ahead))
            r2_prophet = r2_score(y, forecast["yhat"][:len(y)])
            pred_prophet = forecast.iloc[-1]["yhat"]
        except:
            r2_prophet = None
            pred_prophet = "N/A"

        preds = {
            "Linear": lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": poly_model.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xgb_model.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": pred_prophet
        }

        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Prediction"]))

        if show_r2:
            r2s = {
                "Linear": r2_score(y, lin.predict(X)),
                "Polynomial": r2_score(y, poly_model.predict(X_poly)),
                "Random Forest": r2_score(y, rf.predict(X)),
                "XGBoost": r2_score(y, xgb_model.predict(X)),
                "Prophet": r2_prophet
            }
            st.subheader("📊 R² Scores")
            for k, v in r2s.items():
                if v is not None:
                    st.write(f"{k}: {round(v, 4)}")

        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], lin.predict(X), label="Linear")
        ax.plot(df["date"], poly_model.predict(X_poly), label="Polynomial")
        ax.plot(df["date"], rf.predict(X), label="Random Forest")
        ax.plot(df["date"], xgb_model.predict(X), label="XGBoost")
        if r2_prophet is not None:
            ax.plot(df["date"], forecast["yhat"][:len(y)], label="Prophet")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    elif strategy_code == "MD":
        st.subheader("📘 Model Explanation")
        st.markdown("R² score and residual standard deviation help evaluate model fit.")
        model_poly = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)

        try:
            df_prop = df.rename(columns={"date": "ds", "price": "y"})
            prop = Prophet().fit(df_prop)
            forecast = prop.predict(prop.make_future_dataframe(periods=0))
            yhat_prophet = forecast["yhat"]
        except:
            yhat_prophet = None

        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual", linewidth=2)
        ax.plot(df["date"], model_poly.predict(X_poly), label="Polynomial")
        ax.plot(df["date"], rf.predict(X), label="Random Forest")
        ax.plot(df["date"], xgb_model.predict(X), label="XGBoost")
        if yhat_prophet is not None:
            ax.plot(df["date"], yhat_prophet[:len(y)], label="Prophet")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    elif strategy_code == "D":
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        pred, _, _ = predict_price(df, days_ahead)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.warning(f"📉 Downside Risk: ₹{round(downside, 2)} | Volatility: {round(vol*100, 2)}%")

    elif strategy_code == "S":
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail())
        score = fetch_news_sentiment(resolved)
        st.metric("📣 News Sentiment Score", round(score, 2))

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price", linewidth=2)
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

# --- Run Button ---
if st.button("Run Strategy"):
    nav, nav_src = get_live_nav(resolved)
    df = get_stock_price(resolved, nav)
    if df is None or df.empty:
        st.error("❌ No data found. Try another stock or fund.")
    else:
        strategy_code = strategy.split("-")[-1].strip()
        run_strategy(strategy_code, df, days_ahead, nav, nav_src)
