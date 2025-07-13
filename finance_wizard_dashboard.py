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
import os
import finnhub

# --- Load Secrets from Environment ---
openai.api_key = os.getenv("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")
finnhub_client = finnhub.Client(api_key=os.getenv("d1psdrpr01qku4u4dhngd1psdrpr01qku4u4dho0"))

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Finance Wizard (Deployed)", layout="centered")
st.title("🧙 Finance Wizard: Forecast + Sentiment AI")

asset_type = st.selectbox("Choose Asset Type", ["Stock", "Mutual Fund"])
symbol_or_slug = st.text_input("Enter Ticker Symbol (e.g. INFY.NS) or Groww Slug (e.g. axis-small-cap-direct-growth)")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)

hotkey = st.selectbox("Prediction Strategy", [
    "🔮 W - Technical Price Prediction",
    "🧠 ML - Machine Learning-Based Forecast",
    "🌐 TA - Trend Analysis",
    "🔀 TE - Trend Reversal Detection"
])

# --- Data Loaders ---
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="60d")
    if df.empty: return None
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df.rename(columns={"Date": "date", "Close": "price"}, inplace=True)
    return df

def get_nav_data(slug):
    try:
        url = f"https://groww.in/v1/api/mf/v1/scheme/{slug}"
        res = requests.get(url)
        data = res.json()
        navs = data["navHistory"]
        df = pd.DataFrame(navs)
        df["date"] = pd.to_datetime(df["navDate"])
        df["price"] = df["nav"]
        return df[["date", "price"]].sort_values("date")
    except:
        return None

# --- Sentiment Analysis Functions ---
def get_sentiment_score(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyzer. Rate sentiment from -1 (very negative) to 1 (very positive). Respond with only a number."},
                {"role": "user", "content": text}
            ]
        )
        return float(response['choices'][0]['message']['content'].strip())
    except:
        return 0

def get_news_sentiment(symbol):
    try:
        news = finnhub_client.company_news(symbol, _from="2024-01-01", to=datetime.now().strftime("%Y-%m-%d"))
        news = news[:10]
        scores = []
        timestamps = []
        for article in news:
            text = f"{article['headline']}\n{article.get('summary', '')}"
            score = get_sentiment_score(text)
            scores.append(score)
            timestamps.append(datetime.fromtimestamp(article['datetime']))
        df = pd.DataFrame({"datetime": timestamps, "score": scores})
        return df
    except:
        return pd.DataFrame()

# --- Prediction Logic ---
def analyze_and_predict(df, days_ahead, label, hotkey_code):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values

    pred_price, lower, upper = None, None, None
    reversal = False

    if hotkey_code == "W":
        model = LinearRegression().fit(X, y)
        future_index = df["day_index"].max() + days_ahead
        pred_price = model.predict([[future_index]])[0]
        residuals = y - model.predict(X)

    elif hotkey_code == "ML":
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        future_index = df["day_index"].max() + days_ahead
        pred_price = model.predict(poly.transform([[future_index]]))[0]
        residuals = y - model.predict(X_poly)

    elif hotkey_code == "TA":
        pred_price = df["price"].rolling(window=5).mean().iloc[-1]
        residuals = df["price"].diff().dropna()

    elif hotkey_code == "TE":
        df["MA5"] = df["price"].rolling(window=5).mean()
        if len(df["MA5"].dropna()) >= 3:
            slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
            slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
            if np.sign(slope1) != np.sign(slope2):
                reversal = True

    if pred_price is not None:
        std_dev = np.std(residuals)
        upper = pred_price + 1.96 * std_dev
        lower = pred_price - 1.96 * std_dev

    df["MA5"] = df["price"].rolling(window=5).mean()
    if not reversal and len(df["MA5"].dropna()) >= 3:
        slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
        slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
        if np.sign(slope1) != np.sign(slope2):
            reversal = True

    st.subheader(f"📊 {label} — {hotkey_code} Strategy")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price", marker="o")
    ax.plot(df["date"], df["MA5"], label="MA-5", linestyle="--")
    if pred_price:
        ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead),
                    pred_price, yerr=1.96 * std_dev, fmt='ro', label="Prediction ±95%")
    ax.legend()
    st.pyplot(fig)

    if pred_price:
        st.success(f"🔮 Prediction ({days_ahead} days): ₹{round(pred_price,2)}")
        st.info(f"📈 95% Confidence: ₹{round(lower,2)} – ₹{round(upper,2)}")

    if reversal:
        st.warning("⚠️ Possible Trend Reversal Detected!")

# --- Main Execution ---
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("Please enter a valid input.")
    else:
        if asset_type == "Stock":
            df = get_stock_price(symbol_or_slug)
            label = f"{symbol_or_slug.upper()} Stock"
        else:
            df = get_nav_data(symbol_or_slug)
            label = f"{symbol_or_slug.replace('-', ' ').title()} NAV"

        if df is None or df.empty:
            st.error("❌ Data fetch failed.")
        else:
            st.success("✅ Data loaded.")
            st.write(df.tail())
            hotkey_code = hotkey.split()[0].strip("🔮🧠🌐🔀")
            analyze_and_predict(df, days_ahead, label, hotkey_code)

            if asset_type == "Stock":
                sentiment_df = get_news_sentiment(symbol_or_slug)
                if not sentiment_df.empty:
                    st.subheader("🧠 News Sentiment Over Time")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(sentiment_df["datetime"], sentiment_df["score"], marker="o")
                    ax2.axhline(0, color='gray', linestyle='--')
                    ax2.set_ylabel("Sentiment Score")
                    st.pyplot(fig2)

                    avg_sent = sentiment_df["score"].mean()
                    st.info(f"🧠 Avg Sentiment: {round(avg_sent,2)}")
                    if avg_sent < -0.3:
                        st.error("🚨 Negative Sentiment Detected — Review Carefully!")
