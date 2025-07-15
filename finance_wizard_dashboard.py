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

# --- Cache directory ---
CACHE_DIR = "cached_models"
os.makedirs(CACHE_DIR, exist_ok=True)

def model_cache_key(symbol, model_name, days):
    return os.path.join(CACHE_DIR, f"{symbol}_{model_name}_{days}.joblib")

# --- Auto-refresh ---
st_autorefresh(interval=600000, key="auto-refresh")

# --- News Sentiment Fetch ---
def fetch_news_sentiment():
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "q": "markets OR inflation OR interest rates OR oil OR war OR economy",
            "language": "en",
            "category": "business",
            "apiKey": st.secrets["NEWS_API_KEY"]
        }
        response = requests.get(url, params=params)
        articles = response.json().get("articles", [])[:5]
        total_score = 0
        for article in articles:
            title = article["title"]
            summary = article.get("description", "")
            full_text = title + " " + summary
            score = get_sentiment_score(full_text)
            total_score += score
        return round(total_score / max(1, len(articles)), 2)
    except Exception as e:
        st.warning(f"News sentiment fetch failed: {e}")
        return 0

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

# --- Display Sentiment ---
st.subheader("📰 Live News Sentiment (Updates Every 10 Minutes)")
news_sentiment = fetch_news_sentiment()
if news_sentiment > 0.3:
    st.success(f"📈 Positive Sentiment: {news_sentiment}")
elif news_sentiment < -0.3:
    st.error(f"📉 Negative Sentiment: {news_sentiment}")
else:
    st.info(f"⚖️ Neutral Sentiment: {news_sentiment}")

# --- UI CONFIG ---
st.set_page_config(page_title="Finance Wizard", layout="centered")
st.title("🧙 Finance Wizard: Intelligent Market Dashboard")

with st.sidebar:
    show_r2 = st.checkbox("Show R² Scores", value=True)
    plot_future = st.checkbox("Plot 7-Day Forecast", value=True)

strategy = st.selectbox("🧠 Strategy", [
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

# --- 🧠 INPUT FIELD: Fund Name or Ticker ---
fund_name_input = st.text_input(
    "Enter Fund Name or Ticker (e.g. AAPL, INFY.NS, Axis Global Equity Alpha Fund)",
    "NBCC.NS"
)

def resolve_fund_name_to_ticker(fund_name):
    if "." in fund_name and len(fund_name) < 10:
        return fund_name.upper()
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Convert mutual fund or stock names into Yahoo Finance-compatible tickers. Reply only with the ticker symbol (no explanation)."},
                {"role": "user", "content": f"What is the Yahoo Finance ticker for {fund_name}?"}
            ]
        )
        return res["choices"][0]["message"]["content"].strip().upper()
    except:
        return fund_name.upper()

resolved_ticker = resolve_fund_name_to_ticker(fund_name_input)
st.caption(f"🧠 Resolved Ticker: **{resolved_ticker}** (via AI). Please verify manually if needed.")

days_ahead = st.slider("📆 Days Ahead to Forecast", 1, 30, 7)
# --- 🧮 Live NAV Fetcher (Multi-Tiered) ---
@st.cache_data(ttl=300)
def get_live_nav(symbol):
    try:
        info = yf.Ticker(symbol).info
        nav = info.get("navPrice") or info.get("regularMarketPrice")
        if nav and nav > 0:
            return round(nav, 2)
    except: pass
    try:
        txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=5).text
        for line in txt.splitlines():
            if symbol.upper() in line:
                val = float(line.split(";")[-1])
                if val > 0:
                    return round(val, 2)
    except: pass
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
    fallback = {"NBCC.NS": 114.90, "INFY.NS": 1470.75}
    return fallback.get(symbol.upper(), None)

# --- 📈 Price Fetch ---
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

def predict_price(df, days_ahead):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict([[X[-1][0] + days_ahead]])[0]
    std = np.std(y - model.predict(X))
    return pred, pred - 1.96 * std, pred + 1.96 * std

# --- 🧠 Strategy Core ---
def analyze_and_predict(df, strategy_code, days_ahead, symbol, live_nav):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_index"]].values
    y = df["price"].values
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    df["MA5"] = df["price"].rolling(5).mean()
    df["Live_NAV/Price"] = live_nav
    st.subheader(f"📊 Strategy: {strategy_code}")

    def plot_forecast(model, model_type="Polynomial"):
        if not plot_future: return
        X_future = np.array([[i] for i in range(X[-1][0] + 1, X[-1][0] + 8)])
        dates_future = pd.date_range(start=df["date"].max() + pd.Timedelta(days=1), periods=7)
        y_future = model.predict(poly.transform(X_future))
        fig, ax = plt.subplots()
        ax.plot(df["date"], y, label="Actual")
        ax.plot(dates_future, y_future, label=f"{model_type} 7-Day Forecast", linestyle="--")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        st.pyplot(fig)

    if strategy_code == "W":
        st.markdown("**🔮 Predict One Stock**\nUses linear regression to estimate future value.")
        pred, low, high = predict_price(df, days_ahead)
        st.metric("Predicted Price", f"₹{round(pred,2)}")
        st.info(f"Confidence Interval: ₹{round(low)} – ₹{round(high)}")
        st.dataframe(df[["date", "price", "Live_NAV/Price", "MA5"]].tail(10))

    elif strategy_code == "ML":
        st.markdown("**🧠 Polynomial Forecast**\nMultiple ML models for deeper price learning.")
        model_poly = LinearRegression().fit(X_poly, y)
        rf_model = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        try:
            prophet_df = df.rename(columns={"date": "ds", "price": "y"})
            prophet_model = Prophet().fit(prophet_df)
            forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=days_ahead))
            pred_prophet = forecast.iloc[-1]["yhat"]
        except:
            pred_prophet = "N/A"
        st.metric("Polynomial", round(model_poly.predict(poly.transform([[X[-1][0] + days_ahead]]))[0], 2))
        st.metric("Random Forest", round(rf_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("XGBoost", round(xgb_model.predict([[X[-1][0] + days_ahead]])[0], 2))
        st.metric("Prophet", pred_prophet)
        plot_forecast(model_poly)

    elif strategy_code == "MC":
        st.markdown("**⚖️ Model Comparison**\nCompare all models with R² Scores.")
        lin = LinearRegression().fit(X, y)
        poly_model = LinearRegression().fit(X_poly, y)
        rf = RandomForestRegressor().fit(X, y)
        xgb_model = xgb.XGBRegressor().fit(X, y)
        try:
            prop_df = df.rename(columns={"date": "ds", "price": "y"})
            prop = Prophet().fit(prop_df)
            future = prop.make_future_dataframe(periods=days_ahead)
            forecast = prop.predict(future)
            pred_prophet = forecast["yhat"].iloc[-1]
            r2_prophet = r2_score(y, forecast["yhat"][:len(y)])
        except:
            pred_prophet = "N/A"
            r2_prophet = None

        preds = {
            "Linear": lin.predict([[X[-1][0] + days_ahead]])[0],
            "Polynomial": poly_model.predict(poly.transform([[X[-1][0] + days_ahead]]))[0],
            "Random Forest": rf.predict([[X[-1][0] + days_ahead]])[0],
            "XGBoost": xgb_model.predict([[X[-1][0] + days_ahead]])[0],
            "Prophet": pred_prophet
        }
        st.dataframe(pd.DataFrame(preds.items(), columns=["Model", "Prediction"]))
        if show_r2:
            r2s = {
                "Linear": r2_score(y, lin.predict(X)),
                "Polynomial": r2_score(y, poly_model.predict(X_poly)),
                "Random Forest": r2_score(y, rf.predict(X)),
                "XGBoost": r2_score(y, xgb_model.predict(X)),
                "Prophet": r2_prophet
            }
            st.subheader("📊 R² Scores")
            for model, score in r2s.items():
                if score is not None:
                    st.write(f"{model}: {round(score, 4)}")
        plot_forecast(poly_model)

    elif strategy_code in ["SA", "SC", "SD", "SE"]:
        multipliers = {"SA": 1.10, "SC": 1.02, "SD": 0.95, "SE": 0.80}
        scenario = multipliers[strategy_code] * live_nav
        label = {"SA": "Optimistic", "SC": "Conservative", "SD": "Pessimistic", "SE": "Shock"}
        st.markdown(f"**📉 {label[strategy_code]} Forecast**\nMultiplier-based scenario modeling.")
        st.metric(f"{label[strategy_code]} NAV", round(scenario, 2))
        st.bar_chart(pd.DataFrame([live_nav, scenario], index=["Current", label[strategy_code]], columns=["Price"]))
    elif strategy_code == "S":
        st.markdown("""
        **🕵️ Stock Deep Dive**  
        Combines price trends, technicals, and AI-powered sentiment analysis.
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
        Powered by GPT-based financial news analysis.
        """)

    elif strategy_code == "D":
        st.markdown("**📉 Downside Risk**\nEstimates worst-case value using volatility + ML.")
        pred, _, _ = predict_price(df, days_ahead)
        vol = np.std(df["price"].pct_change().dropna())
        downside = pred - 1.96 * vol * df["price"].iloc[-1]
        st.warning(f"⚠️ Estimated Downside: ₹{round(downside, 2)} (Volatility: {round(vol*100, 2)}%)")
        st.caption("📌 Downside may exceed current price if volatility is high or trend is upward.")

    elif strategy_code == "TI":
        st.markdown("**📉 Technical Indicator Forecast**")
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
        - **RSI** (Relative Strength Index): momentum  
        - **EMA20** (Exponential MA): trend direction
        """)
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["EMA20"] = ta.trend.EMAIndicator(df["price"], 20).ema_indicator()
        st.dataframe(df[["date", "RSI", "EMA20", "Live_NAV/Price"]].tail(10))
        st.line_chart(df.set_index("date")[["RSI", "EMA20"]])

    elif strategy_code == "TD":
        st.markdown("""
        **💡 Indicator Definitions**  
        - **RSI**: Strength of price movement  
        - **MACD**: Crossover momentum detection  
        - **BB (Bollinger Bands)**: Volatility bounds  
        - **EMA**: Smoothed moving average  
        - **MA5**: 5-day average (used for short trend)
        """)

    elif strategy_code == "TE":
        st.markdown("""
        **⚠️ Technical Limitations**  
        - Indicators lag in high-volatility environments  
        - Not reliable alone — combine with sentiment + ML  
        - Always validate with fundamentals
        """)

    # --- 📊 Universal Price Chart ---
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"], label="Price")
    ax.plot(df["date"], df["MA5"], linestyle="--", label="MA5")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)

# --- 🚀 EXECUTION TRIGGER ---
if st.button("Run Strategy"):
    if not fund_name_input:
        st.warning("Please enter a fund name or stock ticker.")
    else:
        resolved_symbol = resolve_fund_name_to_ticker(fund_name_input)
        live_nav = get_live_nav(resolved_symbol)
        df = get_stock_price(resolved_symbol, live_nav)
        if df is None or df.empty:
            st.error("No data found. Try another ticker or fund.")
        else:
            analyze_and_predict(df, strategy.split("-")[0].strip().split()[-1], days_ahead, resolved_symbol, live_nav)
