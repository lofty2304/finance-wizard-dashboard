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

# --- Auto-refresh every 10 minutes ---
st_autorefresh(interval=600000, key="auto-refresh")

# --- Cache Directory ---
CACHE_DIR = "cached_models"
os.makedirs(CACHE_DIR, exist_ok=True)

def model_cache_key(symbol, model_name, days):
    return os.path.join(CACHE_DIR, f"{symbol}_{model_name}_{days}.joblib")

# --- Tiered NAV Fetcher ---
@st.cache_data(ttl=300)
def get_live_nav(symbol):
    # Tier 1: Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        nav = ticker.info.get("navPrice") or ticker.info.get("regularMarketPrice")
        if nav and nav > 0:
            return round(nav, 2)
    except: pass
    # Tier 2: AMFI TXT
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=5).text
        for line in txt.splitlines():
            if symbol.upper() in line:
                val = float(line.split(";")[-1])
                if val > 0:
                    return round(val, 2)
    except: pass
    # Tier 3: RapidAPI / Yahoo proxy
    try:
        rapid_url = "https://yfapi.net/v6/finance/quote"
        headers = {"x-api-key": st.secrets.get("RAPIDAPI_KEY")}
        params = {"symbols": symbol}
        r = requests.get(rapid_url, headers=headers, params=params, timeout=5)
        quote = r.json()["quoteResponse"]["result"][0]
        price = quote.get("regularMarketPrice")
        if price and price > 0:
            return round(price, 2)
    except: pass
    # Tier 4: Manual fallback
    fallback = {"NBCC.NS": 114.90, "INFY.NS": 1470.75, "HDFCMF.NS": 45.52}
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
        total_score = 0
        for a in articles:
            text = a["title"] + " " + a.get("description", "")
            total_score += get_sentiment_score(text)
        return round(total_score / max(1, len(articles)), 2)
    except Exception as e:
        st.warning(f"🛑 Sentiment API failed. Error: {e}")
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

# --- UI Setup ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")

# --- Global Sentiment ---
st.subheader("🌐 Global Sentiment Score (Every 10 Min)")
sentiment = fetch_news_sentiment()
if sentiment > 0.3:
    st.success(f"📈 Positive: {sentiment}")
elif sentiment < -0.3:
    st.error(f"📉 Negative: {sentiment}")
else:
    st.info(f"⚖️ Neutral: {sentiment}")

# --- Sidebar Toggles ---
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 7-Day Forecast", value=True)

# --- Strategy & Input ---
strategy = st.selectbox("📊 Strategy", [
    "🔮 W - Predict One Stock",
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

symbol_input = st.text_input("Enter Stock/Mutual Fund Ticker", "NBCC.NS").upper()
days_ahead = st.slider("Forecast Horizon (Days)", 1, 30, 7)
# --- Price Data Fetch ---
def get_stock_price(symbol, live_nav):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        if df.empty and live_nav:
            dates = pd.date_range(end=datetime.today(), periods=90)
            df = pd.DataFrame({"date": dates, "price": [live_nav] * 90})
        else:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["date", "price"]
        df["Live_NAV/Price"] = live_nav
        return df
    except:
        return None

# --- Predict Price with Confidence Interval ---
def predict_price(df, days):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    future_idx = df["day_index"].max() + days
    pred = model.predict([[future_idx]])[0]
    std = np.std(y - model.predict(X))
    return pred, pred - 1.96 * std, pred + 1.96 * std

# --- Analyze and Predict Strategy ---
def analyze_and_predict(df, strategy_code, days_ahead, symbol, live_nav):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    df["MA5"] = df["price"].rolling(5).mean()
    df["Live_NAV/Price"] = live_nav  # Inject again just in case

    st.subheader(f"📊 Strategy: {strategy_code}")

    # Helper: Forecast plot
    def plot_forecast(model, model_type="Polynomial"):
        if not plot_future: return
        future_days = 7
        X_future = np.array([[i] for i in range(X[-1][0] + 1, X[-1][0] + future_days + 1)])
        dates_future = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=future_days)
        y_future = model.predict(poly.transform(X_future)) if model_type == "Polynomial" else model.predict(X_future)
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(dates_future, y_future, label=f"{model_type} 7-Day Forecast", linestyle="--")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)

    if strategy_code == "W":
        st.markdown("**🔮 Predict One Stock**\nUses linear regression to forecast stock or NAV.")
        pred, low, high = predict_price(df, days_ahead)
        st.metric("Prediction", f"₹{round(pred,2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")
        st.dataframe(df[["date", "price", "Live_NAV/Price", "MA5"]].tail(10))

    elif strategy_code == "ML":
        st.markdown("**🧠 ML Polynomial Forecast**\nApplies multiple models: Polynomial, RF, XGBoost, Prophet.")
        model_poly = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xg = xgb.XGBRegressor().fit(X, y)
        pred_poly = model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0]
        pred_rf = rf.predict([[X[-1][0] + days_ahead]])[0]
        pred_xg = xg.predict([[X[-1][0] + days_ahead]])[0]

        prophet_df = df.rename(columns={"date": "ds", "price": "y"}).dropna()
        prophet_valid = prophet_df["y"].nunique() > 1
        try:
            if prophet_valid:
                prophet_model = Prophet().fit(prophet_df)
                future = prophet_model.make_future_dataframe(periods=days_ahead)
                forecast = prophet_model.predict(future)
                pred_prophet = forecast.iloc[-1]["yhat"]
            else:
                pred_prophet = "N/A"
        except:
            pred_prophet = "N/A"

        st.metric("Polynomial", round(pred_poly, 2))
        st.metric("Random Forest", round(pred_rf, 2))
        st.metric("XGBoost", round(pred_xg, 2))
        st.metric("Prophet", pred_prophet)
        st.dataframe(df[["date", "price", "Live_NAV/Price", "MA5"]].tail(10))
        plot_forecast(model_poly)

    elif strategy_code == "MC":
        st.markdown("**⚖️ Compare ML Models**\nForecasts using 5 models and shows R² fit scores.")
        model_lin = LinearRegression().fit(X, y)
        model_poly = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xg = xgb.XGBRegressor().fit(X, y)
        try:
            prophet_df = df.rename(columns={"date": "ds", "price": "y"}).dropna()
            prophet_model = Prophet().fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=days_ahead)
            forecast = prophet_model.predict(future)
            pred_prophet = forecast["yhat"].iloc[-1]
            r2_prophet = r2_score(y, forecast["yhat"][:len(y)])
        except:
            pred_prophet = "N/A"
            r2_prophet = None

        predictions = {
            "Linear": model_lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xg.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": pred_prophet
        }

        st.dataframe(pd.DataFrame(predictions.items(), columns=["Model", "Prediction"]))
        if show_r2:
            r2s = {
                "Linear": r2_score(y, model_lin.predict(X)),
                "Polynomial": r2_score(y, model_poly.predict(X_poly)),
                "Random Forest": r2_score(y, rf.predict(X)),
                "XGBoost": r2_score(y, xg.predict(X)),
                "Prophet": r2_prophet
            }
            st.subheader("📊 R² Scores")
            for m, score in r2s.items():
                if score is not None:
                    st.write(f"{m}: {round(score, 4)}")
        plot_forecast(model_poly)

    elif strategy_code == "D":
        st.markdown("**📉 Downside Risk**\nEstimates lower bound using market volatility.")
        pred, _, _ = predict_price(df, days_ahead)
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.metric("Predicted Price", round(pred, 2))
        st.metric("Estimated Downside", round(downside, 2))
        st.metric("Volatility", f"{round(vol*100, 2)}%")
        if downside > df["price"].iloc[-1]:
            st.info("⚠️ Forecast trend is upward despite downside logic.")

    elif strategy_code in ["SA", "SC", "SD", "SE"]:
        labels = {
            "SA": "🟢 Optimistic",
            "SC": "🟡 Conservative",
            "SD": "🔴 Pessimistic",
            "SE": "⚠️ Extreme Shock"
        }
        multipliers = {"SA": 1.10, "SC": 1.02, "SD": 0.95, "SE": 0.80}
        adjusted = live_nav * multipliers[strategy_code]
        st.markdown(f"**{labels[strategy_code]} Scenario**\nBased on multiplier and latest NAV.")
        st.metric("Scenario Price", f"₹{round(adjusted, 2)}")
        st.bar_chart(pd.DataFrame([live_nav, adjusted], index=["Current", labels[strategy_code]], columns=["Price"]))
    elif strategy_code == "S":
        st.markdown("""
        **🕵️ Stock Deep Dive**  
        Combines short-term technicals with AI-powered sentiment scoring.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        df["Live_NAV/Price"] = live_nav
        st.dataframe(df[["date", "price", "Live_NAV/Price", "RSI", "MACD", "Signal", "MA5"]].tail(10))
        sent_score = get_sentiment_score(symbol)
        st.metric("Sentiment Score", round(sent_score, 2))
        st.markdown("""
        **💬 Sentiment Score Meaning**  
        - **+1** = Extremely Positive  
        - **0** = Neutral  
        - **–1** = Extremely Negative  
        Powered by OpenAI sentiment analysis from financial news.
        """)

    elif strategy_code == "TI":
        st.markdown("""
        **📉 Technical Indicator Forecast**  
        Evaluates short-term market momentum and volatility.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["price"])
        df["BB_H"] = bb.bollinger_hband()
        df["BB_L"] = bb.bollinger_lband()
        df["Live_NAV/Price"] = live_nav
        st.dataframe(df[["date", "price", "Live_NAV/Price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail(10))
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["price"], label="Price")
        ax.plot(df["date"], df["BB_H"], linestyle="--", color="green", label="Upper BB")
        ax.plot(df["date"], df["BB_L"], linestyle="--", color="red", label="Lower BB")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)

    elif strategy_code == "TC":
        st.markdown("""
        **↔️ Compare Indicators**  
        - **RSI**: Measures price strength and momentum  
        - **EMA20**: Exponential moving average over 20 days  
        Helps assess short-term vs smoothed price action.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        df["Live_NAV/Price"] = live_nav
        st.dataframe(df[["date", "RSI", "EMA20", "Live_NAV/Price"]].tail(10))
        st.line_chart(df.set_index("date")[["RSI", "EMA20"]])

    elif strategy_code == "MD":
        st.markdown("""
        **🧐 Model Explanation**  
        Shows how well the Polynomial model fits the actual data.
        R² Score indicates how much variance in price is captured by the model.
        """)
        model_poly = LinearRegression().fit(X_poly, y)
        yhat = model_poly.predict(X_poly)
        r2 = r2_score(y, yhat)
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], yhat, label="Model Fit", linestyle="--")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)
        st.metric("R² Score", round(r2, 4))
        st.caption("R² closer to 1 = better predictive accuracy.")

    elif strategy_code == "ME":
        st.markdown("""
        **❓ ML Uncertainty Analysis**  
        Estimates prediction deviation and potential volatility.
        """)
        model_poly = LinearRegression().fit(X_poly, y)
        residuals = y - model_poly.predict(X_poly)
        std_dev = np.std(residuals)
        st.warning(f"Prediction Std Dev: ± ₹{round(std_dev, 2)}")
        st.caption("Higher std dev means more uncertain forecasts.")

    elif strategy_code == "TD":
        st.markdown("""
        **💡 Indicator Glossary**  
        - **RSI**: Measures price overbought/oversold state  
        - **MACD**: Trend change detection via EMAs  
        - **BB**: Bollinger Bands = price volatility envelope  
        - **EMA20**: 20-day smoothed trend  
        - **MA5**: Simple 5-day moving average  
        """)

    elif strategy_code == "TE":
        st.markdown("""
        **⚠️ Indicator Limitations**  
        - Technical indicators are based on past prices  
        - They may lag during high-volatility moves  
        - Must be paired with ML & sentiment for reliable signals  
        """)

    # --- Universal Price Chart ---
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)

# --- EXECUTION TRIGGER ---
if st.button("Run Strategy"):
    if not symbol_input:
        st.warning("Please enter a stock or mutual fund ticker.")
    else:
        live_nav = get_live_nav(symbol_input)
        df = get_stock_price(symbol_input, live_nav)
        if df is None or df.empty:
            st.error("No data available.")
        else:
            analyze_and_predict(df, strategy.split("-")[0].strip().split()[-1], days_ahead, symbol_input, live_nav)
