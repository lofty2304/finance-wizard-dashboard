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
import ta
import yfinance as yf
import openai
import requests
from datetime import datetime, timedelta
import os

# === API KEYS ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === AUTO-REFRESH SENTIMENT EVERY 10 MINUTES ===
st_autorefresh(interval=600000, key="auto_refresh")

# === PAGE CONFIG ===
st.set_page_config(page_title="Finance Wizard", layout="wide")

# === HEADER ===
st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://i.imgur.com/1N6y4WQ.png" width="80"/>
        <div style="margin-left: 10px;">
            <h1 style="margin: 0;">🧙 Finance Wizard Lucia</h1>
            <p style="margin: 0;">AI-Powered Forecasts, NAV Insights & Global Sentiment</p>
        </div>
    </div>
""", unsafe_allow_html=True)
st.caption("🧡 Empowering clients with 22 years of trust and transparency.")

# === SIDEBAR CONFIG ===
# === SIDEBAR CONFIG ===
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    days_ahead = st.slider("📅 Days Ahead to Forecast", 1, 30, 7)
    show_r2 = st.checkbox("📊 Show R² Scores", value=True)
    plot_future = st.checkbox("📈 Show Forecast Overlay on Chart", value=True)  # <-- NEW LINE
    st.markdown("Each strategy will use this value to forecast into the future.")

# === STRATEGY SELECTOR ===
strategy = st.selectbox("📌 Choose Forecasting Strategy", [
    "🔮 W - Predict One Stock",
    "🧠 ML - Polynomial Forecast",
    "⚖️ MC - ML Model Comparison",
    "🧐 MD - ML Model Explanation",
    "📉 D - Downside Risk",
    "🕵️ S - Stock Deep Dive",
    "📉 TI - Technical Indicator Forecast",
    "↔️ TC - Compare Indicators",
    "💡 TD - Explain Indicators",
    "⚠️ TE - Indicator Limitations",
    "🟢 SA - Optimistic Scenario",
    "🟡 SC - Conservative Scenario",
    "🔴 SD - Pessimistic Scenario",
    "⚖️ SE - Extreme Shock"
])

# === TICKER INPUT ===
st.markdown("## 🔎 Enter Stock / Fund Symbol")
user_input = st.text_input("Ticker / Fund / AMFI Code", "NBCC.NS")

# === RESOLVE TICKER ===
def resolve_ticker(name):
    if "." in name and len(name) < 10:
        return name.upper()
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Convert Indian mutual fund names to Yahoo tickers."},
                {"role": "user", "content": f"What is the ticker for {name}?"}
            ]
        )
        return res["choices"][0]["message"]["content"].strip().upper()
    except:
        return name.upper()

resolved = resolve_ticker(user_input)
st.caption(f"🧾 Resolved Ticker: **{resolved}**")

# === LIVE NAV FETCHER ===
@st.cache_data(ttl=300)
def get_live_nav(ticker):
    try:
        info = yf.Ticker(ticker).info
        nav = info.get("navPrice") or info.get("regularMarketPrice")
        if nav:
            return round(nav, 2), "Yahoo Finance"
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=5).text
        for line in txt.splitlines():
            if ticker.upper() in line:
                val = float(line.split(";")[-1])
                return round(val, 2), "AMFI India"
    except: pass
    return None, "Unavailable"

# === 90-DAY HISTORICAL PRICES ===
def get_stock_price(symbol, fallback_nav):
    try:
        df = yf.Ticker(symbol).history(period="90d")
        if not df.empty:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["date", "price"]
            return df
    except: pass
    if fallback_nav:
        dates = pd.date_range(end=datetime.today(), periods=90)
        df = pd.DataFrame({"date": dates, "price": [fallback_nav] * 90})
        return df
    return None

# === SENTIMENT FETCHER ===
def get_sentiment_score(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Score financial sentiment from -1 (very negative) to 1 (very positive)."},
                {"role": "user", "content": text}
            ]
        )
        return float(res["choices"][0]["message"]["content"].strip())
    except:
        return 0

def fetch_news_sentiment(symbol):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": st.secrets["NEWS_API_KEY"]
        }
        res = requests.get(url, params=params).json()
        articles = res.get("articles", [])[:5]
        texts = [a["title"] + " " + a.get("description", "") for a in articles]
        scores = [get_sentiment_score(txt) for txt in texts]
        return round(np.mean(scores), 2) if scores else 0
    except:
        return 0

# === FORECAST HELPER (used by strategy handlers) ===
def predict_price(df, days_ahead):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    future_index = X[-1][0] + days_ahead
    pred = model.predict([[future_index]])[0]
    std = np.std(y - model.predict(X))
    return pred, pred - 1.96 * std, pred + 1.96 * std
# --- Strategy Execution Logic ---
def run_strategy(code, df, days_ahead, nav, nav_source, plot_future=True):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["MA5"] = df["price"].rolling(5).mean()
    df["Live_NAV"] = nav

    start_dt = df["date"].min().strftime("%d-%m-%Y")
    end_dt = df["date"].max().strftime("%d-%m-%Y")
    forecast_dt = (df["date"].max() + timedelta(days=days_ahead)).strftime("%d-%m-%Y")

    st.caption(f"📆 Data range: {start_dt} to {end_dt} | Forecast till: {forecast_dt}")
    st.caption(f"📈 Live NAV (Source: {nav_source}) → ₹{nav if nav else 'N/A'}")

    # Base features
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    # Shared chart
    def plot_main_graph(forecast_overlay=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["date"], df["price"], label="Price", linewidth=2)
        ax.plot(df["date"], df["MA5"], label="MA5 (5-day moving avg)", linestyle="--")
        if forecast_overlay:
            future_x = df["day_index"].iloc[-1] + np.arange(1, days_ahead + 1)
            future_dates = [df["date"].iloc[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
            ax.plot(future_dates, forecast_overlay, label="Forecast", linestyle=":", color="orange")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    if code == "W":
        st.subheader("🔮 Forecast: One Stock")
        pred, low, high = predict_price(df, days_ahead)
        st.metric("📊 Predicted Price", f"₹{round(pred, 2)}")
        st.info(f"95% Confidence Interval: ₹{round(low, 2)} – ₹{round(high, 2)}")
        st.markdown("**Explanation**: This uses a simple linear model to forecast the next closing price. The 95% confidence interval shows potential price spread.")
        if plot_future:
            forecast_line = [pred] * days_ahead
            plot_main_graph(forecast_overlay=forecast_line)
        else:
            plot_main_graph()

    elif code == "ML":
        st.subheader("🧠 Machine Learning Forecasts")
        poly_model = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor(n_estimators=100).fit(X, y)
        xgb_model = xgb.XGBRegressor(n_estimators=100).fit(X, y)

        # Prophet setup
        try:
            df_prophet = df.rename(columns={"date": "ds", "price": "y"})
            prophet = Prophet().fit(df_prophet)
            future_df = prophet.make_future_dataframe(periods=days_ahead)
            forecast_df = prophet.predict(future_df)
            prophet_pred = forecast_df.iloc[-1]["yhat"]
        except Exception as e:
            prophet_pred = "N/A"

        preds = {
            "Polynomial": poly_model.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xgb_model.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": prophet_pred
        }

        for name, val in preds.items():
            st.metric(name, f"₹{round(val, 2) if isinstance(val, float) else val}")

        st.markdown("**Explanation**: Each ML model has its own way of fitting trends. Polynomial fits curve-based logic, Random Forest/XGBoost use decision trees, and Prophet uses seasonality and trends.")

        if plot_future and isinstance(prophet_pred, float):
            overlay = forecast_df.iloc[-days_ahead:]["yhat"].values
            plot_main_graph(forecast_overlay=overlay)
        else:
            plot_main_graph()

       elif code == "MC":
        st.subheader("⚖️ Model Comparison")
        models = {
            "Linear": LinearRegression().fit(X, y),
            "Polynomial": LinearRegression().fit(X_poly, y),
            "Random Forest": RandomForestRegressor().fit(X, y),
            "XGBoost": xgb.XGBRegressor().fit(X, y)
        }

        df_prophet = df.rename(columns={"date": "ds", "price": "y"})
        try:
            prophet = Prophet().fit(df_prophet)
            future = prophet.make_future_dataframe(periods=days_ahead)
            forecast = prophet.predict(future)
            prophet_r2 = r2_score(y, forecast["yhat"][:len(y)])
            prophet_forecast = forecast.iloc[-1]["yhat"]
        except:
            prophet_r2 = None
            prophet_forecast = "N/A"

        preds = {
            k: model.predict(poly.transform([[X[-1][0] + days_ahead]]) if "Poly" in k else [[X[-1][0] + days_ahead]])[0]
            for k, model in models.items()
        }
        preds["Prophet"] = prophet_forecast

        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Prediction (₹)"]))

        if show_r2:
            st.markdown("### 📈 R² Score")
            r2_vals = {
                k: r2_score(y, model.predict(X_poly if "Poly" in k else X))
                for k, model in models.items()
            }
            r2_vals["Prophet"] = prophet_r2
            st.write(pd.DataFrame(r2_vals.items(), columns=["Model", "R²"]))

        st.markdown("**Explanation**: Higher R² = better fit. This table compares accuracy and predicted value across models.")

        if plot_future:
            plot_main_graph()
        else:
            plot_main_graph()

    elif code == "D":
        st.subheader("📉 Downside Risk")
        returns = df["price"].pct_change().dropna()
        vol = np.std(returns)
            pred, _, _ = predict_price(df, days_ahead)
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
            st.warning(f"📉 Downside: ₹{round(downside, 2)} | Volatility: {round(vol * 100, 2)}%")
    st.markdown("**Explanation**: Downside = projected dip based on current volatility. NAV mismatch may occur if prices are illiquid or lagging.")
    if plot_future:
        downside_line = [downside] * days_ahead
        plot_main_graph(forecast_overlay=downside_line)
    else:
        plot_main_graph()


    elif code == "S":
        st.subheader("🕵️ Deep Stock Dive")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        macd = ta.trend.MACD(df["price"])
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        score = 0
        try:
            news = requests.get("https://newsapi.org/v2/everything", params={
                "q": resolved,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": st.secrets["NEWS_API_KEY"]
            }).json()
            top_titles = [a["title"] for a in news["articles"][:5]]
            scores = [get_sentiment_score(title) for title in top_titles]
            score = np.mean(scores) if scores else 0
        except:
            pass
            st.metric("🧠 News Sentiment", round(score, 2))
    st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail())
    st.markdown("**Explanation**: RSI indicates momentum, MACD shows trend shift, sentiment derived from news headlines.")
    if plot_future:
        plot_main_graph()
    else:
        plot_main_graph()

   # === RUN STRATEGY ON BUTTON CLICK (FINAL STEP) ===
if resolved and st.button("🚀 Run Forecast"):
    df_price = get_stock_price(resolved, get_live_nav(resolved)[0])
    if df_price is not None:
        nav, nav_source = get_live_nav(resolved)
        run_strategy(
            strategy.split()[0],
            df_price,
            days_ahead,
            nav,
            nav_source,
            plot_future
        )
    else:
        st.error("❌ Could not fetch historical prices. Check ticker or connection.")

