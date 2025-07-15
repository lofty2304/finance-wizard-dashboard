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

# --- Input Field ---
fund_name_input = st.text_input(
    "Enter Fund Name or Ticker (e.g. AAPL, INFY.NS, Axis Global Equity Alpha Fund)",
    "NBCC.NS"
)

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

resolved_ticker = resolve_fund_name_to_ticker(fund_name_input)
st.caption(f"🧠 Resolved Ticker: **{resolved_ticker}** via AI")

# --- Live NAV Multi-tiered ---
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

# --- Stock or Fund Price Fetch ---
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
    future_index = df["day_index"].max() + days_ahead
    pred_price = model.predict([[future_index]])[0]
    std_dev = np.std(y - model.predict(X))
    return pred_price, pred_price - 1.96 * std_dev, pred_price + 1.96 * std_dev

def analyze_and_predict(df, strategy_code, days_ahead, symbol, nav_source):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    df["MA5"] = df["price"].rolling(5).mean()
    st.caption(f"📊 Data Source: **{nav_source}**")

    if strategy_code == "W":
        st.markdown("**🔮 Predict One Stock** using linear regression.")
        pred, low, high = predict_price(df, days_ahead)
        st.metric("Prediction", f"₹{round(pred,2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")
        st.dataframe(df.tail(10))

    elif strategy_code == "ML":
        st.markdown("**🧠 Polynomial + Ensemble ML Forecast**")
        poly_model = LinearRegression().fit(X_poly, y)
        rf_model = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        try:
            prophet_df = df.rename(columns={"date": "ds", "price": "y"})
            prophet_model = Prophet().fit(prophet_df)
            forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=days_ahead))
            prophet_pred = forecast.iloc[-1]["yhat"]
        except:
            prophet_pred = "N/A"
        st.metric("Polynomial", round(poly_model.predict(poly.transform([[X[-1][0] + days_ahead]]))[0], 2))
        st.metric("Random Forest", round(rf_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("XGBoost", round(xgb_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("Prophet", prophet_pred)

    elif strategy_code == "MC":
        st.markdown("**⚖️ Model Comparison** with R² Scores")
        lin = LinearRegression().fit(X, y)
        poly_model = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        try:
            prop_df = df.rename(columns={"date": "ds", "price": "y"})
            prop = Prophet().fit(prop_df)
            future = prop.make_future_dataframe(periods=days_ahead)
            forecast = prop.predict(future)
            pred_prophet = forecast["yhat"].iloc[-1]
            r2_prophet = r2_score(y, forecast["yhat"][:len(y)])
        except:
            pred_prophet = "N/A"
            r2_prophet = None

        preds = {
            "Linear": lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": poly_model.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xgb_model.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": pred_prophet
        }

        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Prediction"]))

        if show_r2:
            st.subheader("📊 R² Scores")
            r2s = {
                "Linear": r2_score(y, lin.predict(X)),
                "Polynomial": r2_score(y, poly_model.predict(X_poly)),
                "Random Forest": r2_score(y, rf.predict(X)),
                "XGBoost": r2_score(y, xgb_model.predict(X)),
                "Prophet": r2_prophet
            }
            for model, score in r2s.items():
                if score is not None:
                    st.write(f"{model}: {round(score, 4)}")

    elif strategy_code == "S":
        st.markdown("**🕵️ Deep Dive** with sentiment + indicators")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal"]].tail(10))
        sent_score = get_sentiment_score(symbol)
        st.metric("Sentiment Score", round(sent_score, 2))
        st.caption("Sentiment score: -1 (bearish) → +1 (bullish)")

    elif strategy_code == "D":
        pred, _, _ = predict_price(df, days_ahead)
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.warning(f"📉 Estimated Downside: ₹{round(downside,2)} | Volatility: {round(vol*100,2)}%")
        st.caption("Downside may exceed current price if volatility is high.")

    elif strategy_code == "TI":
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["price"])
        df["BB_H"] = bb.bollinger_hband()
        df["BB_L"] = bb.bollinger_lband()
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail(10))
        st.caption("RSI: overbought > 70, oversold < 30 | BB: volatility envelope")

    elif strategy_code == "TD":
        st.markdown("""
        **💡 Indicator Definitions**  
        - **RSI**: Strength of trend  
        - **MACD**: Momentum reversal  
        - **BB**: Volatility  
        - **EMA**: Smoothed MA  
        - **MA5**: Short-term average
        """)

    elif strategy_code == "TE":
        st.warning("⚠️ Indicators can lag. Combine with ML + fundamentals.")

    # Forecast line
    if plot_future:
        X_future = np.array([[i] for i in range(X[-1][0] + 1, X[-1][0] + 8)])
        dates_future = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=7)
        y_poly = LinearRegression().fit(X_poly, y).predict(poly.transform(X_future))
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["price"], label="Price")
        ax.plot(dates_future, y_poly, label="7-Day Forecast", linestyle="--")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)

# --- Run Button ---
if st.button("Run Strategy"):
    if not fund_name_input:
        st.warning("Enter a stock or mutual fund name")
    else:
        resolved_symbol = resolve_fund_name_to_ticker(fund_name_input)
        live_nav, nav_source = get_live_nav(resolved_symbol)
        df, fetch_source = get_stock_price(resolved_symbol, live_nav)
        if df is None or df.empty:
            st.error("❌ No data found. Try another ticker.")
        else:
            analyze_and_predict(df, strategy.split("-")[0].strip().split()[-1], days_ahead, resolved_symbol, fetch_source)
