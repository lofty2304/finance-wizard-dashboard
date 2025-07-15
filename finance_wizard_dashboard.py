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

# --- Cache Directory ---
CACHE_DIR = "cached_models"
os.makedirs(CACHE_DIR, exist_ok=True)

def model_cache_key(symbol, model_name, days):
    return os.path.join(CACHE_DIR, f"{symbol}_{model_name}_{days}.joblib")

# --- Auto-Refresh ---
st_autorefresh(interval=600000, key="auto-refresh")

# --- NAV Fallback ---
def get_latest_nav(symbol):
    try:
        ticker = yf.Ticker(symbol)
        nav = ticker.info.get("navPrice")
        if nav: return nav
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt").text
        for line in txt.splitlines():
            if symbol.upper() in line:
                return float(line.split(";")[-1])
    except: pass
    fallback = {"NBCC": 114.90}
    return fallback.get(symbol.upper(), None)

# --- News Sentiment ---
def fetch_news_sentiment():
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "q": "markets OR inflation OR interest rates OR oil OR war OR economy",
            "language": "en",
            "category": "business",
            "apiKey": st.secrets["NEWS_API_KEY"]
        }
        r = requests.get(url, params=params)
        articles = r.json().get("articles", [])[:5]
        score = 0
        for a in articles:
            text = a["title"] + " " + a.get("description", "")
            score += get_sentiment_score(text)
        return round(score / max(1, len(articles)), 2)
    except Exception as e:
        st.warning(f"Sentiment fetch failed: {e}")
        return 0

def get_sentiment_score(text):
    try:
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score financial sentiment from -1 to 1."},
                {"role": "user", "content": text}
            ]
        )
        return float(r["choices"][0]["message"]["content"].strip())
    except:
        return 0

# --- Display Sentiment ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")

st.subheader("📰 Global News Sentiment (Updated Every 10 Minutes)")
sentiment = fetch_news_sentiment()
if sentiment > 0.3:
    st.success(f"📈 Positive: {sentiment}")
elif sentiment < -0.3:
    st.error(f"📉 Negative: {sentiment}")
else:
    st.info(f"⚖️ Neutral: {sentiment}")

# --- UI Inputs ---
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 7-Day Forecast", value=True)

strategy = st.selectbox("📊 Choose Strategy", [
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

symbol_input = st.text_input("📌 Enter Ticker", "INFY.NS").upper()
days_ahead = st.slider("⏳ Days Ahead to Forecast", 1, 30, 7)

# --- Fetch Stock or NAV Data ---
def get_stock_price(symbol):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        if df.empty:
            nav = get_latest_nav(symbol)
            if nav:
                dates = pd.date_range(end=datetime.today(), periods=90)
                return pd.DataFrame({"date": dates, "price": [nav] * 90})
            return None
        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["date", "price"]
        return df
    except:
        return None

def predict_price(df, days):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    future_idx = df["day_index"].max() + days
    pred = model.predict([[future_idx]])[0]
    std = np.std(y - model.predict(X))
    return pred, pred - 1.96 * std, pred + 1.96 * std
def analyze_and_predict(df, strategy_code, days_ahead, symbol):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    df["MA5"] = df["price"].rolling(5).mean()

    st.subheader(f"📊 Strategy: {strategy_code}")

    # Shared forecast block for 7-day future
    def plot_forecast_curve(model, model_type="Polynomial"):
        if not plot_future:
            return
        future_days = 7
        X_future = np.array([[i] for i in range(X[-1][0] + 1, X[-1][0] + future_days + 1)])
        dates_future = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=future_days)
        if model_type == "Polynomial":
            y_future = model.predict(poly.transform(X_future))
        else:
            y_future = model.predict(X_future)

        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(dates_future, y_future, label="7-Day Forecast", linestyle="--")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)

    if strategy_code == "W":
        st.markdown("""
        **🔮 Predict One Stock**  
        Uses a linear regression model over the last 90 days to predict future price movement.
        Useful for fast forecasting when data is limited.
        """)
        st.dataframe(df[["date", "price"]].tail(10))
        pred, low, high = predict_price(df, days_ahead)
        st.success(f"Prediction in {days_ahead} days: ₹{round(pred,2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")

    elif strategy_code == "ML":
        st.markdown("""
        **🧠 ML Polynomial Forecast**  
        Applies Polynomial regression, Random Forest, XGBoost and Prophet to generate forecasts.
        Ideal for pattern recognition in price data.
        """)
        model_poly = LinearRegression().fit(X_poly, y)
        rf_model = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        prophet_df = df.rename(columns={"date": "ds", "price": "y"})
        prophet_model = Prophet().fit(prophet_df)
        forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=days_ahead))

        st.metric("Polynomial", round(model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0], 2))
        st.metric("Random Forest", round(rf_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("XGBoost", round(xgb_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("Prophet", round(forecast.iloc[-1]["yhat"], 2))

        plot_forecast_curve(model_poly)

    elif strategy_code == "MC":
        st.markdown("""
        **⚖️ ML Model Comparison**  
        Compares five models: Linear, Polynomial, Random Forest, XGBoost, and Prophet.
        R² scores help evaluate fit accuracy.
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

        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Prediction"]))
        st.bar_chart(pd.DataFrame(preds.values(), index=preds.keys(), columns=["Price"]))

        if show_r2:
            r2s = {
                "Linear": r2_score(y, model_lin.predict(X)),
                "Polynomial": r2_score(y, model_poly.predict(X_poly)),
                "Random Forest": r2_score(y, rf_model.predict(X)),
                "XGBoost": r2_score(y, xgb_model.predict(X)),
                "Prophet": r2_score(y, forecast["yhat"][:len(y)])
            }
            st.subheader("📈 R² Scores")
            for model, score in r2s.items():
                st.write(f"{model}: {round(score, 4)}")

        plot_forecast_curve(model_poly)

    elif strategy_code == "D":
        st.markdown("""
        **📉 Downside Risk**  
        Uses historical volatility to estimate possible loss with 95% confidence over the next period.
        """)
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        pred, _, _ = predict_price(df, days_ahead)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.metric("Volatility", f"{round(vol*100, 2)}%")
        st.metric("Estimated Downside", f"₹{round(downside, 2)}")
        fig, ax = plt.subplots()
        ax.pie([downside, df["price"].iloc[-1]], labels=["Downside", "Current"], autopct="%1.1f%%")
        st.pyplot(fig)

    elif strategy_code in ["SA", "SC", "SD", "SE"]:
        multipliers = {
            "SA": 1.10, "SC": 1.02, "SD": 0.95, "SE": 0.80
        }
        labels = {
            "SA": "🟢 Optimistic", "SC": "🟡 Conservative",
            "SD": "🔴 Pessimistic", "SE": "⚖️ Extreme Shock"
        }
        base_pred, _, _ = predict_price(df, days_ahead)
        pred_price = base_pred * multipliers[strategy_code]
        st.markdown(f"""
        **{labels[strategy_code]} Scenario**  
        Simulates outcome under stress or optimism with a price multiplier.
        """)
        st.metric("Scenario Forecast", f"₹{round(pred_price, 2)}")
        st.bar_chart(pd.DataFrame([base_pred, pred_price], index=["Base", labels[strategy_code]], columns=["Price"]))

    elif strategy_code == "TD":
        st.markdown("""
        **💡 Explain Indicators**  
        - RSI: Relative Strength Index (momentum)
        - MACD: Trend/momentum crossover
        - Bollinger Bands: Volatility breakouts
        - EMA: Smooth trend direction
        - MA5: 5-day moving average (shows recent trend)
        """)

    elif strategy_code == "TE":
        st.warning("""
        **⚠️ Limitations of Technical Indicators**  
        - Lagging indicators
        - No volume or macro consideration
        - Should be used with sentiment + fundamentals
        """)
    elif strategy_code == "TI":
        st.markdown("""
        **📉 Technical Indicator Forecast**  
        - RSI: Detect overbought/oversold zones  
        - MACD: Momentum shifts  
        - Bollinger Bands: Volatility and mean-reversion
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
        ax.plot(df["date"], df["BB_H"], linestyle="--", color="green", label="Upper BB")
        ax.plot(df["date"], df["BB_L"], linestyle="--", color="red", label="Lower BB")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)
        st.caption("📘 MA5 = 5-Day Moving Average")

    elif strategy_code == "TC":
        st.markdown("""
        **↔️ Compare Indicators**  
        - **RSI**: Detect momentum extremes  
        - **EMA20**: Identify smoothed price trend
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        st.dataframe(df[["date", "RSI", "EMA20"]].tail(10))
        st.line_chart(df.set_index("date")[["RSI", "EMA20"]])

    elif strategy_code == "S":
        st.markdown("""
        **🕵️ Stock Deep Dive**  
        Combines RSI, MACD, MA5 and Sentiment Score for holistic insight into a stock's health.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        df["MA5"] = df["price"].rolling(5).mean()
        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail(10))
        sent_score = get_sentiment_score(symbol)
        st.info(f"🧠 Sentiment Score: {round(sent_score, 2)}")

    elif strategy_code == "MD":
        st.markdown("""
        **🧐 Model Explanation**  
        Visualize how well Polynomial model fits actual data.
        """)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], pred, label="Predicted", linestyle="--")
        ax.legend()
        st.pyplot(fig)
        r2 = model.score(X_poly, y)
        st.metric("R² Score", round(r2, 4))

    elif strategy_code == "ME":
        st.markdown("""
        **❓ ML Uncertainty**  
        Shows the variation between model predictions and actual values to estimate risk of misfit.
        """)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        std_dev = np.std(y - pred)
        st.warning(f"Prediction Std Deviation: ±₹{round(std_dev, 2)}")

    # Final price chart for all
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.set_title(f"{symbol} | Price and 5-Day Moving Average")
    st.pyplot(fig)

# --- EXECUTION ---
if st.button("Run Strategy"):
    strategy_code = strategy.split("-")[0].strip().split()[-1]
    df = get_stock_price(symbol_input)
    if df is None or df.empty:
        st.error("❌ No data found. Please check the symbol or try another.")
    else:
        analyze_and_predict(df, strategy_code, days_ahead, symbol_input)
