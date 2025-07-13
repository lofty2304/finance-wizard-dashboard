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

# ✅ Load secure API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# ✅ UI
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Prediction + Sentiment")

asset_type = st.selectbox("Choose Asset Type", ["Stock", "Mutual Fund"])
symbol_or_slug = st.text_input("Enter Ticker (e.g. INFY.NS or axis-small-cap-direct-growth)")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)
hotkey = st.selectbox("Strategy", ["🔮 W - Technical", "🧠 ML - Machine Learning", "🌐 TA - Trend", "🔀 TE - Reversal"])

# ✅ Data fetchers
def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="60d")
        if df.empty: return None
        df = df.reset_index()
        df = df[["Date", "Close"]]
        df.rename(columns={"Date": "date", "Close": "price"}, inplace=True)
        return df
    except:
        return None

def get_nav_data(slug):
    try:
        url = f"https://groww.in/v1/api/mf/v1/scheme/{slug}"
        res = requests.get(url)
        navs = res.json()["navHistory"]
        df = pd.DataFrame(navs)
        df["date"] = pd.to_datetime(df["navDate"])
        df["price"] = df["nav"]
        return df[["date", "price"]].sort_values("date")
    except:
        return None

# ✅ Sentiment functions
def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score this financial headline from -1 to 1."},
                {"role": "user", "content": text}
            ]
        )
        return float(res["choices"][0]["message"]["content"].strip())
    except:
        return 0

def get_news_sentiment(symbol):
    try:
        news = finnhub_client.company_news(symbol, _from="2024-01-01", to=datetime.now().strftime("%Y-%m-%d"))
        news = news[:10]
        data = []
        for article in news:
            score = get_sentiment_score(article["headline"] + " " + article.get("summary", ""))
            data.append({"datetime": datetime.fromtimestamp(article["datetime"]), "score": score})
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# ✅ Prediction logic
def analyze_and_predict(df, days_ahead, label, code):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    pred_price, lower, upper = None, None, None
    reversal = False

    try:
        if code == "W":
            model = LinearRegression().fit(X, y)
            pred_price = model.predict([[df["day_index"].max() + days_ahead]])[0]
            std_dev = np.std(y - model.predict(X))
        elif code == "ML":
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            pred_price = model.predict(poly.transform([[df["day_index"].max() + days_ahead]]))[0]
            std_dev = np.std(y - model.predict(X_poly))
        elif code == "TA":
            pred_price = df["price"].rolling(5).mean().iloc[-1]
            std_dev = np.std(df["price"].diff().dropna())
        elif code == "TE":
            df["MA5"] = df["price"].rolling(5).mean()
            if len(df["MA5"].dropna()) >= 3:
                slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
                slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
                if np.sign(slope1) != np.sign(slope2):
                    reversal = True

        if pred_price:
            lower = pred_price - 1.96 * std_dev
            upper = pred_price + 1.96 * std_dev

    except Exception as e:
        st.error(f"❌ Model failed: {e}")
        return

    # Trend Reversal
    df["MA5"] = df["price"].rolling(5).mean()
    if not reversal and len(df["MA5"].dropna()) >= 3:
        slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
        slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
        if np.sign(slope1) != np.sign(slope2):
            reversal = True

    # Plot chart
    st.subheader("📈 Price Forecast")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    if pred_price:
        ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead), pred_price,
                    yerr=1.96*std_dev, fmt='ro', label="Prediction")
    ax.legend()
    st.pyplot(fig)

    # Output
    if pred_price:
        st.success(f"🔮 Predicted Price in {days_ahead} days: ₹{round(pred_price,2)}")
        st.info(f"📊 95% CI: ₹{round(lower,2)} – ₹{round(upper,2)}")
    if reversal:
        st.warning("⚠️ Trend Reversal Detected!")

# ✅ Main button
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("Please enter a valid input.")
    else:
        label = f"{symbol_or_slug.upper()} Stock" if asset_type == "Stock" else f"{symbol_or_slug.title()} NAV"
        df = get_stock_price(symbol_or_slug) if asset_type == "Stock" else get_nav_data(symbol_or_slug)

        if df is None or df.empty:
            st.error("❌ Data not available.")
        else:
            st.write("📊 Data Preview", df.tail())
            code = hotkey.split()[0].strip("🔮🧠🌐🔀")
            analyze_and_predict(df, days_ahead, label, code)

            if asset_type == "Stock":
                sent_df = get_news_sentiment(symbol_or_slug)
                st.write("🧪 News data:", sent_df.head())
                if not sent_df.empty:
                    avg = sent_df["score"].mean()
                    st.subheader("🧠 News Sentiment")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(sent_df["datetime"], sent_df["score"], marker="o")
                    ax2.axhline(0, color="gray", linestyle="--")
                    ax2.set_ylabel("Sentiment Score")
                    st.pyplot(fig2)
                    st.info(f"🧠 Avg Sentiment: {round(avg,2)}")
                    if avg < -0.3:
                        st.error("🚨 Negative News Sentiment Detected!")
