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

def model_cache_key(symbol, model_name, days):
    return os.path.join(CACHE_DIR, f"{symbol}_{model_name}_{days}.joblib")

# --- Auto-refresh ---
st_autorefresh(interval=600000, key="auto-refresh")

# --- NewsAPI Sentiment ---
def fetch_news_sentiment():
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "q": "markets OR inflation OR interest rates OR oil OR war OR economy",
            "language": "en",
            "category": "business",
            "apiKey": st.secrets["NEWS_API_KEY"]
        }
        response = requests.get(url, params=params)
        articles = response.json().get("articles", [])[:5]
        total_score = 0
        for article in articles:
            title = article["title"]
            summary = article.get("description", "")
            full_text = title + " " + summary
            score = get_sentiment_score(full_text)
            total_score += score
        return round(total_score / max(1, len(articles)), 2)
    except Exception as e:
        st.warning(f"NewsAPI sentiment failed: {e}")
        return 0

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

# --- Display Sentiment ---
st.subheader("📰 Live News Sentiment (Updated Every 10 Minutes)")
news_sentiment = fetch_news_sentiment()
if news_sentiment > 0.3:
    st.success(f"📈 Positive Sentiment: {news_sentiment}")
elif news_sentiment < -0.3:
    st.error(f"📉 Negative Sentiment: {news_sentiment}")
else:
    st.info(f"⚖️ Neutral Sentiment: {news_sentiment}")

# --- UI CONFIG ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")

# Sidebar switches
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 30-Day Forecast", value=False)

strategy = st.selectbox("🧠 Strategy", [
    "🔮 W - Predict One Stock",
    "🏦 A - Compare Stocks",
    "🕵️ S - Stock Deep Dive",
    "📉 D - Downside Risk",
    "🧠 ML - Polynomial Forecast",
    "⚖️ MC - ML Model Comparison",
    "🧐 MD - ML Model Explanation",
    "❓ ME - ML Uncertainty Analysis",
    "📉 TI - Technical Indicator Forecast",
    "↔️ TC - Compare Indicators",
    "💡 TD - Explain Indicators",
    "⚠️ TE - Indicator Limitations",
    "🟢 SA - Optimistic Scenario",
    "🟡 SC - Conservative Scenario",
    "🔴 SD - Pessimistic Scenario",
    "⚖️ SE - Extreme Shock"
])

symbol_input = st.text_input("Enter Stock Ticker(s)", "INFY.NS")
days_ahead = st.slider("Days Ahead to Forecast", 1, 30, 7)

# --- DATA FETCH ---
def get_stock_price(symbol):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["date", "price"]
        return df
    except:
        return None

def predict_price(df, days_ahead):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    future_index = df["day_index"].max() + days_ahead
    pred_price = model.predict([[future_index]])[0]
    std_dev = np.std(y - model.predict(X))
    return pred_price, pred_price - 1.96 * std_dev, pred_price + 1.96 * std_dev

# --- ANALYSIS ---
def analyze_and_predict(df, strategy_code, days_ahead, symbol):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    df["MA5"] = df["price"].rolling(5).mean()
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    st.subheader(f"📊 Strategy: {strategy_code}")

    if strategy_code == "W":
        pred, low, high = predict_price(df, days_ahead)
        st.success(f"🔮 Prediction for {symbol} in {days_ahead} days: ₹{round(pred,2)}")
        st.info(f"📈 Confidence Interval: ₹{round(low)} – ₹{round(high)}")

    elif strategy_code == "ML":
        model_poly = LinearRegression().fit(X_poly, y)
        pred = model_poly.predict(poly.transform([[df["day_index"].max() + days_ahead]]))[0]
        st.success(f"🧠 Polynomial ML Prediction: ₹{round(pred, 2)}")

        rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)
        st.success(f"🌲 Random Forest: ₹{round(rf_model.predict([[X[-1][0] + days_ahead]])[0], 2)}")

        xgb_model = xgb.XGBRegressor().fit(X, y)
        st.success(f"🚀 XGBoost: ₹{round(xgb_model.predict([[X[-1][0] + days_ahead]])[0], 2)}")

        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        prophet_model = Prophet().fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        forecast = prophet_model.predict(future)
        st.success(f"🔮 Prophet: ₹{round(forecast.iloc[-1]['yhat'], 2)}")

    elif strategy_code == "MC":
        model_lin = LinearRegression().fit(X, y)
        model_poly = LinearRegression().fit(X_poly, y)
        rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        prophet_model = Prophet().fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        forecast = prophet_model.predict(future)

        predictions = {
            "Linear": model_lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf_model.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xgb_model.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": forecast.iloc[-1]["yhat"]
        }

        st.dataframe(pd.DataFrame(predictions.items(), columns=["Model", "Prediction"]))
        st.bar_chart(pd.DataFrame(predictions.values(), index=predictions.keys(), columns=["Prediction"]))

        if show_r2:
            r2s = {
                "Linear": r2_score(y, model_lin.predict(X)),
                "Polynomial": r2_score(y, model_poly.predict(X_poly)),
                "Random Forest": r2_score(y, rf_model.predict(X)),
                "XGBoost": r2_score(y, xgb_model.predict(X)),
                "Prophet": r2_score(y, forecast["yhat"][:len(y)])
            }
            st.subheader("📊 R² Scores")
            for model, score in r2s.items():
                st.write(f"{model}: {round(score, 4)}")

        if plot_future:
            future_days = 30
            X_future = np.array([[i] for i in range(X[-1][0] + 1, X[-1][0] + future_days + 1)])
            dates_future = pd.date_range(start=df["date"].max(), periods=future_days + 1, freq="D")[1:]
            y_poly_future = model_poly.predict(poly.transform(X_future))

            fig, ax = plt.subplots()
            ax.plot(df["date"], y, label="Actual")
            ax.plot(dates_future, y_poly_future, label="Polynomial Forecast", linestyle="--")
            ax.legend()
            st.pyplot(fig)

    # Other strategies like "S", "D", "TI", etc. remain the same...

    # Chart
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    ax.legend()
    st.pyplot(fig)

# --- EXECUTION ---
if st.button("Run Strategy"):
    strategy_code = strategy.split("-")[0].strip().split()[-1]
    if strategy_code == "A":
        tickers = [s.strip().upper() for s in symbol_input.split(",")]
        results = []
        for t in tickers:
            df = get_stock_price(t)
            if df is not None and not df.empty:
                pred, _, _ = predict_price(df, days_ahead)
                results.append((t, round(pred, 2)))
        if results:
            df_comp = pd.DataFrame(results, columns=["Ticker", "Prediction"])
            st.write("🏦 Multi-Stock Forecast")
            st.dataframe(df_comp)
            st.bar_chart(df_comp.set_index("Ticker"))
        else:
            st.error("No valid data.")
    else:
        df = get_stock_price(symbol_input)
        if df is None or df.empty:
            st.error("No data found.")
        else:
            analyze_and_predict(df, strategy_code, days_ahead, symbol_input.upper())
