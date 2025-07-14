import streamlit as st
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
import ta  # ✅ New

# --- API Keys ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# --- UI ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Full Market Intelligence")

asset_type = st.selectbox("Choose Asset Type", ["Stock", "Mutual Fund"])
symbol_or_slug = st.text_input("Enter Ticker (e.g. INFY.NS) or Fund Slug")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)

hotkey = st.selectbox("Strategy", [
    "🔮 W - Linear Regression",
    "🧠 ML - Polynomial ML",
    "🌐 TA - Trend (MA)",
    "🔀 TE - Reversal Only",
    "🟢 SA - Optimistic Scenario",
    "🟡 SC - Conservative Scenario",
    "🔴 SD - Pessimistic Scenario",
    "⚖️ SE - Extreme Shock",
    "📉 TI - Technical Indicator Forecast",
    "↔️ TC - Compare Indicators",
    "💡 TD - Explain Indicators",
    "⚠️ TE - Indicator Risk Assessment"
])

# --- Data Fetch ---
def get_stock_price(symbol):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["date", "price"]
        return df
    except: return None

def get_nav_data(slug):
    try:
        url = f"https://groww.in/v1/api/mf/v1/scheme/{slug}"
        res = requests.get(url)
        df = pd.DataFrame(res.json()["navHistory"])
        df["date"] = pd.to_datetime(df["navDate"])
        df["price"] = df["nav"]
        return df[["date", "price"]]
    except: return None

# --- Sentiment ---
def get_news_sentiment(symbol):
    try:
        news = finnhub_client.company_news(symbol, _from="2024-01-01", to=datetime.now().strftime("%Y-%m-%d"))
        records = []
        for n in news[:10]:
            score = get_sentiment_score(n["headline"] + " " + n.get("summary", ""))
            records.append({"datetime": datetime.fromtimestamp(n["datetime"]), "score": score})
        return pd.DataFrame(records)
    except: return pd.DataFrame()

def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score financial sentiment from -1 to 1."},
                {"role": "user", "content": text}
            ])
        return float(res["choices"][0]["message"]["content"].strip())
    except: return 0

# --- Main Analysis Logic ---
def analyze_and_predict(df, days_ahead, label, strategy):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values

    pred_price, lower, upper, std_dev = None, None, None, None
    reversal = False

    # Always calculate MA5
    df["MA5"] = df["price"].rolling(5).mean()

    try:
        if strategy == "W":
            model = LinearRegression().fit(X, y)
            future_index = df["day_index"].max() + days_ahead
            pred_price = model.predict([[future_index]])[0]
            std_dev = np.std(y - model.predict(X))

        elif strategy == "ML":
            X_scaled = X / X.max()
            poly = PolynomialFeatures(3)
            X_poly = poly.fit_transform(X_scaled)
            model = LinearRegression().fit(X_poly, y)
            pred_price = model.predict(poly.transform([[(df["day_index"].max() + days_ahead) / X.max()]]))[0]
            std_dev = np.std(y - model.predict(X_poly))

        elif strategy == "TA":
            pred_price = df["price"].rolling(5).mean().iloc[-1]
            std_dev = np.std(df["price"].diff().dropna())

        elif strategy == "TE":
            if len(df["MA5"].dropna()) >= 3:
                s1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
                s2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
                if np.sign(s1) != np.sign(s2): reversal = True

        elif strategy in ["SA", "SC", "SD", "SE"]:
            model = LinearRegression().fit(X, y)
            base_pred = model.predict([[df["day_index"].max() + days_ahead]])[0]
            if strategy == "SA":
                pred_price = base_pred * 1.10
                lower, upper = pred_price * 0.98, pred_price * 1.20
            elif strategy == "SC":
                pred_price = base_pred * 1.02
                lower, upper = pred_price * 0.98, pred_price * 1.05
            elif strategy == "SD":
                pred_price = base_pred * 0.95
                lower, upper = pred_price * 0.90, pred_price * 1.00
            elif strategy == "SE":
                pred_price = base_pred * 0.80
                lower, upper = pred_price * 0.75, pred_price * 0.85

        elif strategy == "TI":
            df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
            macd = ta.trend.MACD(df["price"])
            df["MACD"] = macd.macd()
            df["Signal"] = macd.macd_signal()
            bb = ta.volatility.BollingerBands(df["price"])
            df["BB_H"] = bb.bollinger_hband()
            df["BB_L"] = bb.bollinger_lband()

            signal = "🔁 Neutral"
            if df["RSI"].iloc[-1] < 30 and df["MACD"].iloc[-1] > df["Signal"].iloc[-1]:
                signal = "📈 Buy Signal"
            elif df["RSI"].iloc[-1] > 70 and df["MACD"].iloc[-1] < df["Signal"].iloc[-1]:
                signal = "📉 Sell Signal"
            st.info(f"TI Signal: {signal}")
            st.write(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail())

        elif strategy == "TC":
            df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
            df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
            df["BB_H"] = ta.volatility.BollingerBands(df["price"]).bollinger_hband()
            df["BB_L"] = ta.volatility.BollingerBands(df["price"]).bollinger_lband()
            st.write(df[["date", "RSI", "EMA20", "BB_H", "BB_L"]].tail())

        elif strategy == "TD":
            st.markdown("""
            ### 💡 Indicator Explanations:
            - **RSI**: Measures overbought (>70) or oversold (<30).
            - **MACD**: Detects momentum shifts via EMA crossovers.
            - **Bollinger Bands**: Show volatility compression and breakouts.
            - **EMA**: Reacts faster to recent price than SMA.
            """)

        elif strategy == "TE":
            st.warning("""
            ⚠️ Limitations:
            - Technicals often lag price action.
            - News shocks override indicators.
            - Works best with confirmation and volume.
            """)

        # Confidence Interval
        if pred_price and not lower:
            lower = pred_price - 1.96 * std_dev
            upper = pred_price + 1.96 * std_dev

    except Exception as e:
        st.error(f"Strategy Error: {e}")
        return

    # --- Chart ---
    st.subheader("📉 Price Chart + MA5")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    if strategy not in ["TE", "TI", "TC", "TD"] and pred_price:
        ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead),
                    pred_price, yerr=1.96 * std_dev, fmt='ro', label="Prediction")
    ax.legend()
    st.pyplot(fig)

    # --- Output ---
    if strategy not in ["TE", "TI", "TC", "TD"] and pred_price:
        st.success(f"Prediction in {days_ahead} days: ₹{round(pred_price, 2)}")
        st.info(f"CI: ₹{round(lower, 2)} – ₹{round(upper, 2)}")

    if strategy == "TE" and reversal:
        st.error("⚠️ Trend reversal detected!")

# --- Run ---
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("Please enter a valid input.")
    else:
        df = get_stock_price(symbol_or_slug) if asset_type == "Stock" else get_nav_data(symbol_or_slug)
        label = f"{symbol_or_slug.upper()} Stock" if asset_type == "Stock" else f"{symbol_or_slug.title()} NAV"
        strategy_code = hotkey.split("-")[0].strip().split()[-1]

        if df is None or df.empty:
            st.error("❌ Data not found.")
        else:
            st.write(df.tail())
            analyze_and_predict(df, days_ahead, label, strategy_code)
