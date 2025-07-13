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

# --- Load API keys ---
openai.api_key = os.getenv("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")
finnhub_client = finnhub.Client(api_key=st.secrets["d1psdrpr01qku4u4dhngd1psdrpr01qku4u4dho0"])

# --- UI ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Forecast + Sentiment AI")

asset_type = st.selectbox("Choose Asset Type", ["Stock", "Mutual Fund"])
symbol_or_slug = st.text_input("Enter Ticker (e.g. INFY.NS) or Groww Slug")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)
hotkey = st.selectbox("Prediction Strategy", [
    "🔮 W - Technical Price Prediction",
    "🧠 ML - Machine Learning-Based Forecast",
    "🌐 TA - Trend Analysis",
    "🔀 TE - Trend Reversal Detection"
])

# --- Data fetchers ---
def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="60d")
        if df.empty:
            return None
        df = df.reset_index()
        df = df[["Date", "Close"]]
        df.rename(columns={"Date": "date", "Close": "price"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Stock data fetch failed: {e}")
        return None

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
    except Exception as e:
        st.error(f"❌ NAV data fetch failed: {e}")
        return None

# --- Sentiment
def get_sentiment_score(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a financial sentiment scorer. Return a score from -1 to 1."},
                {"role": "user", "content": text}
            ]
        )
        score = float(response['choices'][0]['message']['content'].strip())
        return score
    except Exception as e:
        st.warning(f"⚠️ OpenAI failed: {e}")
        return 0

def get_news_sentiment(symbol):
    try:
        news = finnhub_client.company_news(symbol, _from="2024-01-01", to=datetime.now().strftime("%Y-%m-%d"))
        news = news[:10]
        scores, times = [], []
        for item in news:
            txt = f"{item['headline']}\n{item.get('summary', '')}"
            score = get_sentiment_score(txt)
            scores.append(score)
            times.append(datetime.fromtimestamp(item['datetime']))
        return pd.DataFrame({"datetime": times, "score": scores})
    except Exception as e:
        st.warning(f"⚠️ Finnhub failed: {e}")
        return pd.DataFrame()

# --- Forecast logic
def analyze_and_predict(df, days_ahead, label, strategy):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values

    pred_price, lower, upper = None, None, None
    reversal = False

    st.write(f"🧪 Running Strategy: `{strategy}` on {label}")
    st.write("📏 Data points:", len(df))

    try:
        if strategy == "W":
            model = LinearRegression().fit(X, y)
            future_index = df["day_index"].max() + days_ahead
            pred_price = model.predict([[future_index]])[0]
            std_dev = np.std(y - model.predict(X))

        elif strategy == "ML":
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            future_index = df["day_index"].max() + days_ahead
            pred_price = model.predict(poly.transform([[future_index]]))[0]
            std_dev = np.std(y - model.predict(X_poly))

        elif strategy == "TA":
            pred_price = df["price"].rolling(window=5).mean().iloc[-1]
            std_dev = np.std(df["price"].diff().dropna())

        elif strategy == "TE":
            df["MA5"] = df["price"].rolling(5).mean()
            if len(df["MA5"].dropna()) >= 3:
                slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
                slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
                if np.sign(slope1) != np.sign(slope2):
                    reversal = True
            pred_price = None

        if pred_price is not None:
            lower = pred_price - 1.96 * std_dev
            upper = pred_price + 1.96 * std_dev

    except Exception as e:
        st.error(f"❌ Model failed: {e}")
        return

    # Add MA5 and reversal signal
    df["MA5"] = df["price"].rolling(5).mean()
    if not reversal and len(df["MA5"].dropna()) >= 3:
        slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
        slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
        if np.sign(slope1) != np.sign(slope2):
            reversal = True

    # Plot
    st.subheader("📊 Forecast Chart")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    if pred_price:
        ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead),
                    pred_price, yerr=1.96*std_dev, fmt='ro', label="Prediction")
    ax.legend()
    st.pyplot(fig)

    if pred_price:
        st.success(f"🔮 Predicted: ₹{round(pred_price,2)} (in {days_ahead} days)")
        st.info(f"📈 95% CI: ₹{round(lower,2)} – ₹{round(upper,2)}")

    if reversal:
        st.warning("⚠️ Possible trend reversal!")

# --- RUN ---
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("Please enter a valid input.")
    else:
        if asset_type == "Stock":
            df = get_stock_price(symbol_or_slug)
            label = f"{symbol_or_slug.upper()} Stock"
        else:
            df = get_nav_data(symbol_or_slug)
            label = f"{symbol_or_slug.title()} NAV"

        if df is None or df.empty:
            st.error("❌ No data available.")
        else:
            st.write("✅ Data loaded:", df.tail())
            code = hotkey.split()[0].strip("🔮🧠🌐🔀")
            analyze_and_predict(df, days_ahead, label, code)

            # Sentiment
            if asset_type == "Stock":
                sent_df = get_news_sentiment(symbol_or_slug)
                if not sent_df.empty:
                    avg_score = sent_df["score"].mean()
                    st.subheader("🧠 News Sentiment Over Time")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(sent_df["datetime"], sent_df["score"], marker='o')
                    ax2.axhline(0, color='gray', linestyle='--')
                    ax2.set_ylabel("Sentiment Score")
                    st.pyplot(fig2)
                    st.info(f"🧠 Avg News Sentiment: {round(avg_score,2)}")
                    if avg_score < -0.3:
                        st.error("🚨 Negative news sentiment detected.")
