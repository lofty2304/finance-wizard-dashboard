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

# --- API KEYS ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

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
        model = LinearRegression().fit(X_poly, y)
        pred_price = model.predict(poly.transform([[df["day_index"].max() + days_ahead]]))[0]
        st.success(f"🧠 Polynomial ML Prediction: ₹{round(pred_price, 2)}")

    elif strategy_code == "MC":
        model_lin = LinearRegression().fit(X, y)
        pred_lin = model_lin.predict([[df["day_index"].max() + days_ahead]])[0]
        model_poly = LinearRegression().fit(X_poly, y)
        pred_poly = model_poly.predict(poly.transform([[df["day_index"].max() + days_ahead]]))[0]
        st.write(f"Linear: ₹{round(pred_lin, 2)} | Poly: ₹{round(pred_poly, 2)}")

    elif strategy_code == "MD":
        model = LinearRegression().fit(X_poly, y)
        pred = model.predict(X_poly)
        residuals = y - pred
        st.write("R² Score:", model.score(X_poly, y))
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(df["date"], pred, label="Predicted", linestyle="--")
        ax.legend()
        st.pyplot(fig)

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
