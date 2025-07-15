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

CACHE_DIR = "cached_models"
os.makedirs(CACHE_DIR, exist_ok=True)

def model_cache_key(symbol, model_name, days):
    return os.path.join(CACHE_DIR, f"{symbol}_{model_name}_{days}.joblib")

# --- Auto-refresh every 10 mins
st_autorefresh(interval=600000, key="auto-refresh")

# --- NewsAPI + Sentiment ---
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
            full_text = article["title"] + " " + article.get("description", "")
            score = get_sentiment_score(full_text)
            total_score += score
        return round(total_score / max(1, len(articles)), 2)
    except Exception as e:
        st.warning(f"News sentiment fetch failed: {e}")
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

# --- News Sentiment Display ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")
st.subheader("📰 Market Sentiment (Updated Every 10 Minutes)")

news_sentiment = fetch_news_sentiment()
if news_sentiment > 0.3:
    st.success(f"📈 Positive Sentiment: {news_sentiment}")
elif news_sentiment < -0.3:
    st.error(f"📉 Negative Sentiment: {news_sentiment}")
else:
    st.info(f"⚖️ Neutral Sentiment: {news_sentiment}")
# --- Sidebar switches ---
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 30-Day Forecast", value=False)

# --- Strategy Select ---
strategy = st.selectbox("🧠 Select Strategy", [
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

# --- User ticker input ---
symbol_input = st.text_input("📌 Enter Stock or Mutual Fund Ticker", "AAPL").upper()
days_ahead = st.slider("⏳ Days Ahead to Forecast", 1, 30, 7)

# --- Data Fetch with NAV Fallback ---
def get_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="90d")
        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["date", "price"]

        # Fallback for Mutual Fund NAV if possible
        if df["price"].isnull().all() and "navPrice" in ticker.info:
            nav = ticker.info.get("navPrice")
            df = pd.DataFrame({
                "date": pd.date_range(end=datetime.today(), periods=90),
                "price": [nav] * 90
            })
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
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
def analyze_and_predict(df, strategy_code, days_ahead, symbol):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    df["MA5"] = df["price"].rolling(5).mean()

    st.subheader(f"📊 Strategy: {strategy_code}")

    if strategy_code == "W":
        st.markdown("""
        **🔮 Strategy W: Predict One Stock**  
        Uses a basic linear regression model over recent prices to forecast near-term price movement.  
        Best used for fast directional guidance when data is limited.
        """)
        st.dataframe(df[["date", "price"]].tail(10))
        pred, low, high = predict_price(df, days_ahead)
        st.success(f"Forecast in {days_ahead} days: ₹{round(pred, 2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")

    elif strategy_code == "ML":
        st.markdown("""
        **🧠 ML Forecast (Polynomial, RF, XGBoost, Prophet)**  
        Applies four machine learning models to forecast: Polynomial regression, Random Forest, XGBoost, and Prophet (time series).
        """)
        model_poly = LinearRegression().fit(X_poly, y)
        rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        prophet_model = Prophet().fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        forecast = prophet_model.predict(future)

        st.metric("Polynomial", f"₹{round(model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0], 2)}")
        st.metric("Random Forest", f"₹{round(rf_model.predict([[X[-1][0] + days_ahead]])[0], 2)}")
        st.metric("XGBoost", f"₹{round(xgb_model.predict([[X[-1][0] + days_ahead]])[0], 2)}")
        st.metric("Prophet", f"₹{round(forecast.iloc[-1]['yhat'], 2)}")

    elif strategy_code == "MC":
        st.markdown("""
        **⚖️ Strategy MC: Model Comparison**  
        Evaluates five models side by side (Linear, Polynomial, RF, XGBoost, Prophet) and compares both forecasted values and R² scores.
        """)
        model_lin = LinearRegression().fit(X, y)
        model_poly = LinearRegression().fit(X_poly, y)
        rf_model = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        prophet_model = Prophet().fit(prophet_df)
        forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=days_ahead))

        preds = {
            "Linear": model_lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf_model.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xgb_model.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": forecast["yhat"].iloc[-1]
        }

        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Forecast"]))
        st.bar_chart(pd.DataFrame(preds.values(), index=preds.keys(), columns=["Price"]))

        if show_r2:
            scores = {
                "Linear": r2_score(y, model_lin.predict(X)),
                "Polynomial": r2_score(y, model_poly.predict(X_poly)),
                "Random Forest": r2_score(y, rf_model.predict(X)),
                "XGBoost": r2_score(y, xgb_model.predict(X)),
                "Prophet": r2_score(y, forecast["yhat"][:len(y)])
            }
            st.subheader("📈 R² Scores")
            for k, v in scores.items():
                st.write(f"{k}: {round(v, 4)}")

    elif strategy_code == "D":
        st.markdown("""
        **📉 Strategy D: Downside Risk**  
        Calculates price volatility and potential price drop at 95% confidence. Helps estimate worst-case scenario in short term.
        """)
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        pred, _, _ = predict_price(df, days_ahead)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.metric("Volatility", f"{round(vol*100, 2)}%")
        st.metric("Estimated Downside", f"₹{round(downside, 2)}")

        # Optional pie visualization
        fig, ax = plt.subplots()
        ax.pie([downside, df["price"].iloc[-1]], labels=["Downside", "Current"], autopct="%1.1f%%")
        st.pyplot(fig)
    elif strategy_code == "TI":
        st.markdown("""
        **📉 Strategy TI: Technical Indicator Forecast**  
        Computes indicators like RSI, MACD, Bollinger Bands to help detect overbought/oversold conditions and momentum shifts.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["price"])
        df["BB_H"] = bb.bollinger_hband()
        df["BB_L"] = bb.bollinger_lband()

        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail(10))

        fig, ax = plt.subplots()
        ax.plot(df["date"], df["price"], label="Price")
        ax.plot(df["date"], df["BB_H"], linestyle="--", color="red", label="Upper BB")
        ax.plot(df["date"], df["BB_L"], linestyle="--", color="green", label="Lower BB")
        ax.legend()
        ax.set_title("Bollinger Bands & Price")
        st.pyplot(fig)

    elif strategy_code == "TC":
        st.markdown("""
        **↔️ Strategy TC: Compare Indicators**  
        Compares RSI (momentum) and EMA (trend smoothing) to identify potential entry/exit points.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()

        st.dataframe(df[["date", "RSI", "EMA20"]].tail(10))
        st.line_chart(df[["RSI", "EMA20"]])

    elif strategy_code in ["SA", "SC", "SD", "SE"]:
        multipliers = {
            "SA": 1.10,  # +10%
            "SC": 1.02,  # +2%
            "SD": 0.95,  # -5%
            "SE": 0.80   # -20%
        }
        labels = {
            "SA": "🟢 Optimistic",
            "SC": "🟡 Conservative",
            "SD": "🔴 Pessimistic",
            "SE": "⚖️ Extreme Shock"
        }

        st.markdown(f"""
        **{labels[strategy_code]} Scenario**  
        Adjusts the base linear forecast by a scenario multiplier to simulate extreme or ideal market conditions.
        """)
        base_pred, _, _ = predict_price(df, days_ahead)
        pred_price = base_pred * multipliers[strategy_code]
        st.metric("Scenario Forecast", f"₹{round(pred_price, 2)}")

        scenario_data = {
            "Base": base_pred,
            labels[strategy_code]: pred_price
        }
        st.bar_chart(pd.DataFrame(scenario_data.values(), index=scenario_data.keys(), columns=["Price"]))
    elif strategy_code == "TD":
        st.markdown("""
        **💡 Strategy TD: Explain Indicators**  
        Glossary of technical indicators:
        - **RSI**: Momentum indicator. Above 70 = Overbought. Below 30 = Oversold.
        - **MACD**: Measures trend momentum by comparing two moving averages.
        - **Bollinger Bands**: Detect volatility and potential breakouts.
        - **EMA**: Smooths price to reveal longer-term trend.
        """)

    elif strategy_code == "TE":
        st.warning("""
        **⚠️ Strategy TE: Technical Indicator Limitations**  
        Indicators are derived from past prices. They **lag** real-time market moves and should not be used in isolation.
        Combine with:
        - Volume
        - Fundamental trends
        - News + Sentiment
        """)

    elif strategy_code == "MD":
        st.markdown("""
        **🧐 Strategy MD: Model Explanation**  
        Visualizes model residuals (prediction error) to assess fit quality and variance.
        """)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        residuals = y - pred
        r2 = model.score(X_poly, y)
        st.write("R² Score:", round(r2, 4))

        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], pred, linestyle="--", label="Predicted")
        ax.legend()
        st.pyplot(fig)

    elif strategy_code == "ME":
        st.markdown("""
        **❓ Strategy ME: ML Uncertainty**  
        Measures model prediction uncertainty by standard deviation of residuals.
        Helps assess how much risk or variation exists in model output.
        """)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        std_dev = np.std(y - pred)
        st.warning(f"Standard Deviation (Prediction Error): ±₹{round(std_dev, 2)}")

    # --- Always Plot Final Price Chart ---
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)
# --- EXECUTION ---
if st.button("Run Strategy"):
    strategy_code = strategy.split("-")[0].strip().split()[-1]
    df = get_stock_price(symbol_input)
    if df is None or df.empty:
        st.error("❌ No data found for that ticker. Please check the symbol or try another.")
    else:
        analyze_and_predict(df, strategy_code, days_ahead, symbol_input)
