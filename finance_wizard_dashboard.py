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
import ta

# --- API Keys ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# --- UI ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Stock Forecasting Dashboard")

strategy = st.selectbox("📌 Select Strategy", [
    "🔮 W - Predict One Stock",
    "🏦 A - Compare Stocks",
    "🕵️ S - Stock Deep Dive",
    "📉 D - Downside Risk",
    "📉 TI - Technical Indicator Forecast"
])

symbol_input = st.text_input("Enter Ticker(s)", "INFY.NS")
days_ahead = st.slider("Days Ahead", 1, 30, 7)

# --- Helpers ---
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

def plot_chart(df, label, pred_price=None, lower=None, upper=None):
    df["MA5"] = df["price"].rolling(5).mean()
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    if pred_price:
        ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead),
                    pred_price, yerr=[[pred_price - lower], [upper - pred_price]],
                    fmt='ro', label="Prediction")
    ax.legend()
    st.pyplot(fig)

def get_sentiment(symbol):
    try:
        news = finnhub_client.company_news(symbol, _from="2024-01-01", to=datetime.now().strftime("%Y-%m-%d"))
        scores = []
        for n in news[:5]:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Score from -1 (bad) to 1 (good)"},
                    {"role": "user", "content": n["headline"]}
                ])
            scores.append(float(res["choices"][0]["message"]["content"].strip()))
        return np.mean(scores)
    except:
        return 0
# --- Strategy Logic ---
if st.button("Run Analysis"):
    if not symbol_input:
        st.error("❌ Please enter at least one ticker.")
    else:
        symbols = [s.strip().upper() for s in symbol_input.split(",")]

        if strategy.startswith("🔮 W"):
            symbol = symbols[0]
            df = get_stock_price(symbol)
            if df is not None:
                pred, low, high = predict_price(df, days_ahead)
                st.success(f"🔮 {symbol} predicted price in {days_ahead} days: ₹{round(pred,2)}")
                st.info(f"📈 Confidence Interval: ₹{round(low,2)} – ₹{round(high,2)}")
                plot_chart(df, symbol, pred, low, high)
            else:
                st.error(f"Data not found for {symbol}")

        elif strategy.startswith("🏦 A"):
            results = []
            for symbol in symbols:
                df = get_stock_price(symbol)
                if df is not None:
                    pred, _, _ = predict_price(df, days_ahead)
                    results.append((symbol, round(pred, 2)))
            if results:
                st.subheader("🏦 Comparison of Predicted Prices")
                comp_df = pd.DataFrame(results, columns=["Symbol", f"Predicted Price in {days_ahead} days"])
                st.dataframe(comp_df)
                st.bar_chart(comp_df.set_index("Symbol"))
            else:
                st.error("No valid data for any symbol.")

        elif strategy.startswith("🕵️ S"):
            symbol = symbols[0]
            df = get_stock_price(symbol)
            if df is not None:
                st.write("📉 Basic Forecast:")
                pred, low, high = predict_price(df, days_ahead)
                st.success(f"Prediction: ₹{round(pred, 2)} | CI: ₹{round(low)} – ₹{round(high)}")

                st.write("📊 Technical Indicators:")
                df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
                macd = ta.trend.MACD(df["price"])
                df["MACD"] = macd.macd()
                df["Signal"] = macd.macd_signal()
                bb = ta.volatility.BollingerBands(df["price"])
                df["BB_H"] = bb.bollinger_hband()
                df["BB_L"] = bb.bollinger_lband()
                st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail())

                sentiment_score = get_sentiment(symbol)
                st.info(f"🧠 News Sentiment Score: {round(sentiment_score, 2)}")
                plot_chart(df, symbol, pred, low, high)
            else:
                st.error("❌ Data not found.")

        elif strategy.startswith("📉 D"):
            symbol = symbols[0]
            df = get_stock_price(symbol)
            if df is not None:
                df["returns"] = df["price"].pct_change()
                volatility = np.std(df["returns"].dropna())
                pred, _, _ = predict_price(df, days_ahead)
                downside = pred - (1.96 * volatility * df["price"].iloc[-1])
                st.warning(f"📉 Estimated downside risk: ₹{round(downside, 2)}")
                st.info(f"🔄 Historical volatility: {round(volatility*100, 2)}%")
                plot_chart(df, symbol, pred, downside, pred)
            else:
                st.error("❌ Ticker not found.")

        elif strategy.startswith("📉 TI"):
            symbol = symbols[0]
            df = get_stock_price(symbol)
            if df is not None:
                df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
                macd = ta.trend.MACD(df["price"])
                df["MACD"] = macd.macd()
                df["Signal"] = macd.macd_signal()
                bb = ta.volatility.BollingerBands(df["price"])
                df["BB_H"] = bb.bollinger_hband()
                df["BB_L"] = bb.bollinger_lband()
                st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "BB_H", "BB_L"]].tail())
                plot_chart(df, symbol)
            else:
                st.error("❌ Data not found.")
