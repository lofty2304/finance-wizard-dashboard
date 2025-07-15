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
        nav = yf.Ticker(symbol).info.get("navPrice")
        if nav and nav > 0:
            return nav
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt").text
        for line in txt.splitlines():
            if symbol.upper() in line:
                return float(line.split(";")[-1])
    except: pass
    manual_fallback = {"NBCC.NS": 114.90}
    return manual_fallback.get(symbol.upper(), None)

# --- Sentiment Fetch ---
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
        st.warning(f"🛑 Sentiment API failed. Defaulting to 0. Error: {e}")
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

# --- Header and UI ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")

st.subheader("📰 Global Market Sentiment (Updates Every 10 Minutes)")
sentiment = fetch_news_sentiment()
if sentiment > 0.3:
    st.success(f"📈 Positive Sentiment: {sentiment}")
elif sentiment < -0.3:
    st.error(f"📉 Negative Sentiment: {sentiment}")
else:
    st.info(f"⚖️ Neutral Sentiment: {sentiment}")

# --- UI Options ---
with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 7-Day Forecast", value=True)

strategy = st.selectbox("📊 Choose Strategy", [
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

symbol_input = st.text_input("Enter Stock or Fund Ticker", "NBCC.NS").upper()
days_ahead = st.slider("Days Ahead to Forecast", 1, 30, 7)

# --- Data Loader ---
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
    df["MA5"] = df["price"].rolling(5).mean()
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    st.subheader(f"📊 Strategy: {strategy_code}")

    def plot_forecast(model, model_type="Polynomial"):
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
        Uses linear regression on the past 90 days to project the price forward.  
        Suitable for short-term forecasts based on existing trends.
        """)
        pred, low, high = predict_price(df, days_ahead)
        st.metric("Forecast", f"₹{round(pred,2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")

    elif strategy_code == "ML":
        st.markdown("""
        **🧠 Polynomial Forecast (ML)**  
        Applies Polynomial Regression, Random Forest, XGBoost, and Prophet for machine-learned forecasting.  
        Returns a composite view of different model predictions.
        """)
        model_poly = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xg = xgb.XGBRegressor().fit(X, y)

        prophet_df = df.rename(columns={"date": "ds", "price": "y"}).dropna()
        prophet_df = prophet_df[~prophet_df["ds"].duplicated()]
        if len(prophet_df) > 10 and prophet_df["y"].nunique() > 1:
            prophet_model = Prophet().fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=days_ahead)
            forecast = prophet_model.predict(future)
            prophet_value = forecast["yhat"].iloc[-1]
        else:
            prophet_value = "N/A"
            st.warning("📉 Not enough variance for Prophet prediction.")

        st.metric("Polynomial", round(model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0], 2))
        st.metric("Random Forest", round(rf.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("XGBoost", round(xg.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("Prophet", prophet_value)

        plot_forecast(model_poly)

    elif strategy_code == "MC":
        st.markdown("""
        **⚖️ ML Model Comparison**  
        Compares Linear, Polynomial, Random Forest, XGBoost, and Prophet predictions side-by-side.  
        R² Scores reveal how well models fit the current trend.
        """)
        model_lin = LinearRegression().fit(X, y)
        model_poly = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xg = xgb.XGBRegressor().fit(X, y)

        prophet_df = df.rename(columns={"date": "ds", "price": "y"}).dropna()
        prophet_df = prophet_df[~prophet_df["ds"].duplicated()]
        if len(prophet_df) > 10 and prophet_df["y"].nunique() > 1:
            prophet_model = Prophet().fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=days_ahead)
            forecast = prophet_model.predict(future)
            prophet_pred = forecast["yhat"].iloc[-1]
            r2_prophet = r2_score(y, forecast["yhat"][:len(y)])
        else:
            prophet_pred = "N/A"
            r2_prophet = None
            st.warning("📉 Prophet disabled due to data constraints.")

        preds = {
            "Linear": model_lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xg.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": prophet_pred
        }

        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Prediction"]))
        if show_r2:
            r2s = {
                "Linear": r2_score(y, model_lin.predict(X)),
                "Polynomial": r2_score(y, model_poly.predict(X_poly)),
                "Random Forest": r2_score(y, rf.predict(X)),
                "XGBoost": r2_score(y, xg.predict(X)),
                "Prophet": r2_prophet
            }
            st.subheader("📊 R² Scores (Goodness of Fit)")
            for model, score in r2s.items():
                if score is not None:
                    st.write(f"{model}: {round(score, 4)} — Measures how well the model fits past price behavior. 1.0 = perfect fit.")
        plot_forecast(model_poly)

    elif strategy_code == "D":
        st.markdown("""
        **📉 Downside Risk**  
        Uses predicted price and market volatility to estimate 95% downside scenario.
        """)
        pred, _, _ = predict_price(df, days_ahead)
        vol = np.std(df["price"].pct_change().dropna())
        current = df["price"].iloc[-1]
        downside = pred - 1.96 * vol * current

        st.metric("Predicted Price", f"₹{round(pred, 2)}")
        st.metric("Current Price", f"₹{round(current, 2)}")
        st.metric("Volatility", f"{round(vol*100, 2)}%")
        st.metric("Downside", f"₹{round(downside, 2)}")
        if downside > current:
            st.info("⚠️ Downside is still above current price due to upward trend in prediction.")
        else:
            st.warning("📉 Potential drop below current market value.")

    elif strategy_code in ["SA", "SC", "SD", "SE"]:
        labels = {
            "SA": "🟢 Optimistic",
            "SC": "🟡 Conservative",
            "SD": "🔴 Pessimistic",
            "SE": "⚠️ Extreme Shock"
        }
        multipliers = {"SA": 1.10, "SC": 1.02, "SD": 0.95, "SE": 0.80}
        base_price = get_latest_nav(symbol) or df["price"].iloc[-1]
        forecast = base_price * multipliers[strategy_code]
        st.markdown(f"""
        **{labels[strategy_code]} Scenario**  
        Based on adjusted multiplier of real NAV or current price (₹{round(base_price,2)}).
        """)
        st.metric("Scenario Forecast", f"₹{round(forecast, 2)}")
        st.bar_chart(pd.DataFrame([base_price, forecast], index=["Base", labels[strategy_code]], columns=["Price"]))
    elif strategy_code == "TI":
        st.markdown("""
        **📉 Technical Indicator Forecast**  
        Uses 3 classic indicators:  
        - **RSI** (Relative Strength Index): Momentum; overbought >70, oversold <30  
        - **MACD**: Trend momentum and crossovers  
        - **Bollinger Bands**: Price volatility envelopes  
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

    elif strategy_code == "TC":
        st.markdown("""
        **↔️ Compare Indicators**  
        - **RSI**: Measures strength of price action.  
        - **EMA20**: Exponential Moving Average of past 20 days.  
        Helps identify **momentum direction**.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        st.dataframe(df[["date", "RSI", "EMA20"]].tail(10))
        st.line_chart(df.set_index("date")[["RSI", "EMA20"]])

    elif strategy_code == "S":
        st.markdown("""
        **🕵️ Stock Deep Dive**  
        Combines short-term technicals with AI sentiment scoring.
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        df["MA5"] = df["price"].rolling(5).mean()

        st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail(10))

        sent_score = get_sentiment_score(symbol)
        st.metric("Sentiment Score", round(sent_score, 2))
        st.markdown("""
        **🧠 Sentiment Score Meaning:**  
        - **+1** = Very positive news flow  
        - **0** = Neutral  
        - **–1** = Very negative financial sentiment  
        Based on recent business and macro headlines.
        """)

    elif strategy_code == "MD":
        st.markdown("""
        **🧐 Model Explanation**  
        Fits Polynomial model to historical price, and shows how closely it matches.  
        Helps visualize **fit quality**.  
        **R² Score** explains how much of the price movement is captured by the model.
        """)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], pred, label="Polynomial Fit", linestyle="--")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)
        st.metric("R² Score", round(model.score(X_poly, y), 4))
        st.caption("Higher R² = Better model fit to actual price history.")

    elif strategy_code == "ME":
        st.markdown("""
        **❓ ML Model Uncertainty**  
        Calculates prediction **error deviation**, giving insight into how risky a forecast might be.
        """)
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        std_dev = np.std(y - pred)
        st.warning(f"Prediction Standard Deviation: ± ₹{round(std_dev, 2)}")

    elif strategy_code == "TD":
        st.markdown("""
        **💡 Indicator Definitions**  
        - **RSI**: Measures momentum (scale 0–100)  
        - **MACD**: Shows trend reversals  
        - **Bollinger Bands**: Price range vs. volatility  
        - **EMA**: Smooth average trend  
        - **MA5**: 5-day Simple Moving Average  
        Use indicators in **combination**, not isolation.
        """)

    elif strategy_code == "TE":
        st.markdown("""
        **⚠️ Limitations of Indicators**  
        - All indicators are **lagging**  
        - May mislead during high-volatility periods  
        - Ignore fundamentals or volume  
        Use them with sentiment, macro, and machine learning!
        """)

    # Final price chart
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5 (5-Day Avg)", linestyle="--")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)

# --- EXECUTION ---
if st.button("Run Strategy"):
    strategy_code = strategy.split("-")[0].strip().split()[-1]
    df = get_stock_price(symbol_input)
    if df is None or df.empty:
        st.error("❌ No data found. Check the ticker or symbol.")
    else:
        analyze_and_predict(df, strategy_code, days_ahead, symbol_input)
