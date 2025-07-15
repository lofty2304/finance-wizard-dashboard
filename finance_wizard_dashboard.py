import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import yfinance as yf
import requests
import openai
import finnhub
import ta
rom sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
import joblib
import os

# --- API KEYS ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
# --- Auto-Refresh Every 10 Minutes ---
st_autorefresh(interval=600000, key="auto-refresh")

# --- NewsAPI Fetch + OpenAI Sentiment ---
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

# --- Display Sentiment Status ---
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

def get_sentiment_score(text):
    # --- Model Caching ---
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
import joblib
import os

CACHE_DIR = "cached_models"
os.makedirs(CACHE_DIR, exist_ok=True)

def model_cache_key(symbol, model_name, days):
    return os.path.join(CACHE_DIR, f"{symbol}_{model_name}_{days}.joblib")

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

# --- ANALYSIS LOGIC ---
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

    elif strategy_code == "S":
        analyze_and_predict(df, "W", days_ahead, symbol)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        st.write(df[["date", "price", "RSI", "MACD", "Signal"]].tail())
        sent_score = get_sentiment_score(symbol)
        st.info(f"🧠 Sentiment Score: {round(sent_score, 2)}")

    elif strategy_code == "D":
        pred, _, _ = predict_price(df, days_ahead)
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.warning(f"📉 Estimated Downside: ₹{round(downside,2)} | Volatility: {round(vol*100,2)}%")

        elif strategy_code == "ML":
        # 🧠 Base Polynomial
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(poly.transform([[df["day_index"].max() + days_ahead]]))[0]
        st.success(f"🧠 Polynomial ML Prediction: ₹{round(pred, 2)}")

        # 🌲 Random Forest
        rf_key = model_cache_key(symbol, "RF", days_ahead)
        if os.path.exists(rf_key):
            rf_model = joblib.load(rf_key)
        else:
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            joblib.dump(rf_model, rf_key)
        rf_pred = rf_model.predict([[df["day_index"].max() + days_ahead]])[0]
        st.success(f"🌲 Random Forest Prediction: ₹{round(rf_pred, 2)}")

        # 🚀 XGBoost
        xgb_key = model_cache_key(symbol, "XGB", days_ahead)
        if os.path.exists(xgb_key):
            xgb_model = joblib.load(xgb_key)
        else:
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X, y)
            joblib.dump(xgb_model, xgb_key)
        xgb_pred = xgb_model.predict([[df["day_index"].max() + days_ahead]])[0]
        st.success(f"🚀 XGBoost Prediction: ₹{round(xgb_pred, 2)}")

        # 🔮 Prophet
        prophet_key = model_cache_key(symbol, "PROPHET", days_ahead)
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        if os.path.exists(prophet_key):
            prophet_model = joblib.load(prophet_key)
        else:
            prophet_model = Prophet()
            prophet_model.fit(prophet_df)
            joblib.dump(prophet_model, prophet_key)
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        forecast = prophet_model.predict(future)
        prophet_price = forecast.iloc[-1]["yhat"]
        st.success(f"🔮 Prophet Forecast: ₹{round(prophet_price, 2)}")


    e    elif strategy_code == "MC":
        st.info("⚖️ Comparing predictions from multiple ML models...")

        future_index = df["day_index"].max() + days_ahead

        # Linear
        model_lin = LinearRegression().fit(X, y)
        pred_lin = model_lin.predict([[future_index]])[0]

        # Polynomial
        model_poly = LinearRegression().fit(X_poly, y)
        pred_poly = model_poly.predict(poly.transform([[future_index]]))[0]

        # Random Forest
        rf_key = model_cache_key(symbol, "RF", days_ahead)
        if os.path.exists(rf_key):
            rf_model = joblib.load(rf_key)
        else:
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            joblib.dump(rf_model, rf_key)
        pred_rf = rf_model.predict([[future_index]])[0]

        # XGBoost
        xgb_key = model_cache_key(symbol, "XGB", days_ahead)
        if os.path.exists(xgb_key):
            xgb_model = joblib.load(xgb_key)
        else:
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X, y)
            joblib.dump(xgb_model, xgb_key)
        pred_xgb = xgb_model.predict([[future_index]])[0]

        # Prophet
        prophet_key = model_cache_key(symbol, "PROPHET", days_ahead)
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        if os.path.exists(prophet_key):
            prophet_model = joblib.load(prophet_key)
        else:
            prophet_model = Prophet()
            prophet_model.fit(prophet_df)
            joblib.dump(prophet_model, prophet_key)
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        forecast = prophet_model.predict(future)
        pred_prophet = forecast.iloc[-1]["yhat"]

        predictions = {
            "Linear": pred_lin,
            "Polynomial": pred_poly,
            "Random Forest": pred_rf,
            "XGBoost": pred_xgb,
            "Prophet": pred_prophet
        }

        df_preds = pd.DataFrame(predictions.items(), columns=["Model", "Prediction"])
        st.dataframe(df_preds)
        st.bar_chart(df_preds.set_index("Model"))


        elif strategy_code == "MD":
        st.info("🧐 Plotting Actual vs Predicted values for ML models...")

        # Actual timeline
        dates = df["date"]
        preds = {}

        # Polynomial
        model_poly = LinearRegression().fit(X_poly, y)
        preds["Polynomial"] = model_poly.predict(X_poly)

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)
        preds["Random Forest"] = rf_model.predict(X)

        # XGBoost
        xgb_model = xgb.XGBRegressor().fit(X, y)
        preds["XGBoost"] = xgb_model.predict(X)

        # Prophet
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        prophet_model = Prophet().fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=0)
        forecast = prophet_model.predict(future)
        preds["Prophet"] = forecast["yhat"].values

        # Plot all
        fig, ax = plt.subplots()
        ax.plot(dates, y, label="Actual", linewidth=2)
        for model_name, yhat in preds.items():
            ax.plot(dates, yhat, label=model_name)
        ax.legend()
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Residual summary
        st.subheader("Residual Std Dev (Prediction Error):")
        for name, yhat in preds.items():
            error = np.std(y - yhat)
            st.write(f"{name}: ±₹{round(error, 2)}")


    elif strategy_code == "ME":
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        std_dev = np.std(y - pred)
        st.warning(f"❓ ML Std Deviation: {round(std_dev, 2)} | Risk of overfitting if degree too high")

    elif strategy_code == "TI":
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["price"])
        df["BB_H"] = bb.bollinger_hband()
        df["BB_L"] = bb.bollinger_lband()
        st.write(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail())

    elif strategy_code == "TC":
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        st.write(df[["date", "RSI", "EMA20"]].tail())

    elif strategy_code == "TD":
        st.markdown("""
        💡 RSI = Overbought/Oversold  
        💡 MACD = Momentum shift  
        💡 BB = Volatility breakouts  
        """)

    elif strategy_code == "TE":
        st.warning("⚠️ Technical indicators may lag. Combine with sentiment and fundamentals.")

    elif strategy_code in ["SA", "SC", "SD", "SE"]:
        base_pred, _, _ = predict_price(df, days_ahead)
        multiplier = {"SA": 1.10, "SC": 1.02, "SD": 0.95, "SE": 0.80}[strategy_code]
        pred_price = base_pred * multiplier
        st.success(f"{strategy_code} adjusted forecast: ₹{round(pred_price, 2)}")

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
