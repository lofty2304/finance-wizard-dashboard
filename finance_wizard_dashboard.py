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

# --- API keys ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# --- UI ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Stock & Mutual Fund Intelligence")

asset_type = st.selectbox("Choose Asset Type", ["Stock", "Mutual Fund"])
symbol_or_slug = st.text_input("Enter Symbol (e.g. INFY.NS) or Slug (e.g. axis-small-cap-direct-growth)")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)
hotkey = st.selectbox("Strategy", [
    "🔮 W - Linear Regression",
    "🧠 ML - Polynomial ML",
    "🌐 TA - Trend (MA)",
    "🔀 TE - Reversal Only"
])

# --- Data loaders ---
def get_stock_price(symbol):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        if df.empty: return None
        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["date", "price"]
        return df
    except:
        return None

def get_nav_data(slug):
    try:
        url = f"https://groww.in/v1/api/mf/v1/scheme/{slug}"
        res = requests.get(url)
        df = pd.DataFrame(res.json()["navHistory"])
        df["date"] = pd.to_datetime(df["navDate"])
        df["price"] = df["nav"]
        return df[["date", "price"]].sort_values("date")
    except:
        return None

# --- Sentiment ---
def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score this financial headline from -1 (very negative) to 1 (very positive)."},
                {"role": "user", "content": text}
            ]
        )
        return float(res["choices"][0]["message"]["content"].strip())
    except:
        return 0

def get_news_sentiment(symbol):
    try:
        news = finnhub_client.company_news(symbol, _from="2024-01-01", to=datetime.now().strftime("%Y-%m-%d"))
        records = []
        for n in news[:10]:
            score = get_sentiment_score(n["headline"] + " " + n.get("summary", ""))
            records.append({"datetime": datetime.fromtimestamp(n["datetime"]), "score": score})
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

# --- Core prediction logic ---
def analyze_and_predict(df, days_ahead, label, strategy):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values

    pred_price, lower, upper, std_dev = None, None, None, None
    reversal = False
    st.write(f"🧪 Strategy Used: {strategy}")

    try:
        if strategy == "W":
            model = LinearRegression().fit(X, y)
            future_index = df["day_index"].max() + days_ahead
            pred_price = model.predict([[future_index]])[0]
            residuals = y - model.predict(X)
            std_dev = np.std(residuals)

        elif strategy == "ML":
            # Normalized index improves polynomial prediction for longer terms
            X_scaled = X / X.max()
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X_scaled)
            model = LinearRegression().fit(X_poly, y)
            future_index = (df["day_index"].max() + days_ahead) / X.max()
            pred_price = model.predict(poly.transform([[future_index]]))[0]
            residuals = y - model.predict(X_poly)
            std_dev = np.std(residuals)

        elif strategy == "TA":
            pred_price = df["price"].rolling(window=5).mean().iloc[-1]
            std_dev = np.std(df["price"].diff().dropna())

        elif strategy == "TE":
            df["MA5"] = df["price"].rolling(window=5).mean()
            if len(df["MA5"].dropna()) >= 3:
                slope1 = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
                slope2 = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
                if np.sign(slope1) != np.sign(slope2):
                    reversal = True
            pred_price, lower, upper = None, None, None

        if pred_price is not None and std_dev is not None:
            lower = pred_price - 1.96 * std_dev
            upper = pred_price + 1.96 * std_dev

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return

    # Always compute MA5 for chart
    df["MA5"] = df["price"].rolling(5).mean()

    # --- Chart display ---
    st.subheader("📉 Price Chart + MA5")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], label="MA5", linestyle="--")
    if strategy != "TE" and pred_price:
        ax.errorbar(df["date"].max() + pd.Timedelta(days=days_ahead),
                    pred_price, yerr=1.96 * std_dev, fmt='ro', label="Prediction")
    ax.legend()
    st.pyplot(fig)

    # --- Output ---
    if strategy != "TE" and pred_price:
        st.success(f"🔮 Prediction ({days_ahead} days): ₹{round(pred_price, 2)}")
        st.info(f"📈 95% CI: ₹{round(lower, 2)} – ₹{round(upper, 2)}")

    if strategy == "TE":
        if reversal:
            st.error("⚠️ Trend reversal detected!")
        else:
            st.success("✅ No trend reversal detected.")

# --- Main run ---
if st.button("Run Prediction"):
    if not symbol_or_slug:
        st.error("❌ Please enter a valid input.")
    else:
        df = get_stock_price(symbol_or_slug) if asset_type == "Stock" else get_nav_data(symbol_or_slug)
        label = f"{symbol_or_slug.upper()} Stock" if asset_type == "Stock" else f"{symbol_or_slug.title()} NAV"

        if df is None or df.empty:
            st.error("❌ Data not available or ticker/slug is invalid.")
        else:
            st.write("✅ Data Preview", df.tail())
            strategy_code = hotkey.split("-")[0].strip().split()[-1]
            analyze_and_predict(df, days_ahead, label, strategy_code)

            if asset_type == "Stock":
                sent_df = get_news_sentiment(symbol_or_slug)
                if not sent_df.empty:
                    avg = sent_df["score"].mean()
                    st.subheader("🧠 News Sentiment")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(sent_df["datetime"], sent_df["score"], marker="o")
                    ax2.axhline(0, linestyle="--", color="gray")
                    st.pyplot(fig2)
                    st.info(f"🧠 Avg Sentiment Score: {round(avg, 2)}")
                    if avg < -0.3:
                        st.error("🚨 News sentiment is strongly negative!")
