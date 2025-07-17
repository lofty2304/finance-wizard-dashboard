# --- Part 1: Imports, API & Utilities ---
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import ta, yfinance as yf, openai, requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]

# UI config
st_autorefresh(interval=600000, key="auto_refresh")
st.set_page_config(page_title="Finance Wizard", layout="wide")
# --- Part 2: Sidebar + Sentiment + Fetchers ---
st.sidebar.markdown("### ⚙️ Settings")
days_ahead = st.sidebar.slider("📅 Days Ahead Forecast", 1, 30, 7)
show_r2 = st.sidebar.checkbox("📊 Show R² Score", True)
simulate_runs = st.sidebar.slider("🔁 Number of Runs", 1, 5, 1)

strategy = st.selectbox("📌 Choose Strategy", [
    "W - Prophet", "ML - Polynomial", "MC - Model Comparison", "MD - Model Explanation",
    "D - Downside Risk", "S - Deep Dive", "TI - Technical", "TC - Compare Ind.",
    "TD - Explain Ind.", "TE - Limitations", "BK - Backtest vs Benchmark", 
    "SIP - SIP Engine", "DL - LSTM", "PC - SIP vs LSTM", "SA - Scenario", 
    "SC - Scenario", "SD - Scenario", "SE - Scenario"
])
resolved = st.text_input("Ticker / AMFI Code", "NBCC.NS").upper()
st.caption(f"🔍 Live Ticker: **{resolved}**")

# Sentiment fetching
@st.cache_data(ttl=600)
def get_sentiment(q):
    key = st.secrets["NEWS_API_KEY"]
    resp = requests.get(f"https://newsapi.org/v2/everything?q={q}&pageSize=3&sortBy=publishedAt&apiKey={key}").json()
    heads = [a["title"] for a in resp.get("articles", [])]
    prompt = " ".join(heads) + "\n\nSentiment 0–1:"
    ans = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are a financial sentiment model."},
                  {"role":"user","content":prompt}]
    )["choices"][0]["message"]["content"].lower()
    return (0.7 if "positive" in ans else 0.3 if "negative" in ans else 0.5), ans
# --- Part 3: Forecasting & Simulation Utilities ---
@st.cache_data(ttl=300)
def get_live_nav(tick):
    info = yf.Ticker(tick).info
    return round(info.get("navPrice", info.get("regularMarketPrice", 0)), 2)

def get_price_df(tick):
    nav = get_live_nav(tick)
    df = yf.Ticker(tick).history(period="180d").reset_index()[["Date","Close"]]
    df.columns = ["date","price"]
    return df if not df.empty else pd.DataFrame({
        "date": pd.date_range(end=datetime.today(), periods=90),
        "price": [nav]*90
    })

def calc_cagr(start, end, months):
    return round(((end/start)**(12/months) - 1)*100, 2)

def simulate_sip(df, amt, months):
    dfm = df.tail(months)
    if dfm.empty: return 0, 0, 0
    units = [amt/p for p in dfm["price"] if p>0]
    fv = sum(units) * dfm["price"].iloc[-1]
    return amt*months, round(fv,2), calc_cagr(amt, fv/months, months)

def linear_forecast(df, days):
    X = df["day_index"].values.reshape(-1,1); y = df["price"]
    m = LinearRegression().fit(X, y)
    fx = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1)
    return m.predict(fx).tolist(), m.score(X,y)

def polynomial_forecast(df, days):
    deg = PolynomialFeatures(3)
    Xp = deg.fit_transform(df[["day_index"]])
    m = LinearRegression().fit(Xp, df["price"])
    fx = deg.transform(np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1))
    return m.predict(fx).tolist(), m.score(Xp, df["price"])

def random_forest_forecast(df, days):
    X = df["day_index"].values.reshape(-1,1)
    m = RandomForestRegressor().fit(X, df["price"])
    fx = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1)
    return m.predict(fx).tolist(), m.score(X, df["price"])

def xgboost_forecast(df, days):
    X = df["day_index"].values.reshape(-1,1)
    m = xgb.XGBRegressor().fit(X, df["price"])
    fx = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1)
    return m.predict(fx).tolist(), m.score(X, df["price"])

def prophet_forecast(df, days):
    pdf = df.rename(columns={"date":"ds","price":"y"})
    pdf["ds"] = pdf["ds"].dt.tz_localize(None)
    m = Prophet().fit(pdf)
    fc = m.predict(m.make_future_dataframe(periods=days))["yhat"].tail(days).tolist()
    return fc, m

def lstm_forecast(df, days):
    data = df["price"].values.reshape(-1,1)
    scaler = MinMaxScaler().fit(data); sd = scaler.transform(data)
    X,y = [], []
    for i in range(60, len(sd)-days):
        X.append(sd[i-60:i]); y.append(sd[i+days-1])
    model = Sequential([
        LSTM(50, activation="relu", input_shape=(60,1)),
        Dense(1)
    ])
    model.compile("adam", "mse"); model.fit(np.array(X), np.array(y), epochs=10, batch_size=16, verbose=0)
    pred = model.predict(sd[-60:].reshape(1,60,1))
    return round(scaler.inverse_transform(pred)[0][0],2)

def get_benchmark_df(days):
    df = yf.Ticker("^NSEI").history(period=f"{days}d").reset_index()[["Date","Close"]]
    df.columns = ["date","price"]
    return df

def plot_main_graph(df, forecast=None, title=""):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df["date"], df["price"], label="Historical")
    if forecast:
        start = df["date"].iloc[-1]
        ax.plot([start + timedelta(i+1) for i in range(len(forecast))], forecast, "--", label=title)
    ax.legend(); st.pyplot(fig)
# --- Part 4: Strategy Handler & Execution ---
def adjust_forecast(forecast, sentiment, code):
    mapf = {"SA":1.05, "SC":1.02, "SD":0.98, "SE":0.95}
    base = mapf.get(code, 1)
    return [round(p * (base + (sentiment-0.5)*0.1),2) for p in forecast]

def run_strategy(code, df, nav):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    sentiment, raw = get_sentiment(resolved)
    st.metric("📌 Live NAV", f"₹{nav}")
    st.metric("📰 Sentiment Score", sentiment)

    # Forecast logic
    if code in ["SA","SC","SD","SE"]:
        pred, _ = linear_forecast(df, days_ahead)
        adj = adjust_forecast(pred, sentiment, code)
        plot_main_graph(df, adj, f"{code} Scenario Forecast")
        st.caption(f"Scenario adjusted by sentiment. News: {raw[:120]}…")
    elif code=="W":
        st.subheader("🔮 Prophet Forecast")
        fc,_ = prophet_forecast(df, days_ahead)
        plot_main_graph(df, fc, "Prophet Forecast")
        st.markdown(f"Forecast from {df.date.iloc[-1].date()} to {(df.date.iloc[-1]+timedelta(days_ahead)).date()}")
    elif code=="ML":
        st.subheader("📐 Polynomial Forecast")
        fc,r2 = polynomial_forecast(df, days_ahead)
        plot_main_graph(df, fc)
        st.metric("R² Score", round(r2,3))
    elif code=="MC":
        st.subheader("🔍 Model Comparison")
        models = {
            "Linear": linear_forecast,
            "Poly": polynomial_forecast,
            "RF": random_forest_forecast,
            "XGB": xgboost_forecast
        }
        results = {}
        for name, fn in models.items():
            fc,r = fn(df, days_ahead)
            results[name] = f"{fc[-1]:.2f} (R²={r:.2f})"
        st.json(results)
        plot_main_graph(df, polynomial_forecast(df,days_ahead)[0], "Polynomial Forecast")
    elif code=="MD":
        st.subheader("🧩 Model Explanation")
        st.markdown("""
        - **Linear**: fits linear trend  
        - **Polynomial**: catches curves  
        - **RF/XGB**: non-linear trees  
        - **Prophet**: seasonality & trend  
        - **LSTM**: sequence/deep learning
        """)
        fc,_ = prophet_forecast(df, days_ahead)
        plot_main_graph(df, fc, "Prophet Example")
    elif code=="D":
        st.subheader("⚠️ Downside Risk")
        vol = df["price"].pct_change().std() * 100
        ds = df["price"].mean() - 1.5 * df["price"].std()
        st.metric("Volatility", f"{vol:.2f}%")
        st.metric("Potential Downside", f"₹{ds:.2f}")
        plot_main_graph(df)
    elif code=="S":
        st.subheader("🔍 Stock Deep Dive")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["price"]).macd()
        df["Signal"] = ta.trend.MACD(df["price"]).macd_signal()
        st.dataframe(df.tail(5)[["date","price","RSI","MACD","Signal","MA5"]])
        fc,_ = linear_forecast(df, days_ahead)
        plot_main_graph(df, fc, "Next Forecast")
        st.markdown("Shows latest RSI/MACD and 1-step linear forecast for guidance.")
    elif code=="TI":
        st.subheader("📊 Technical Indicator (RSI)")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        st.line_chart(df.set_index("date")[["price","RSI"]])
        st.markdown("RSI >70 = overbought; <30 = oversold.")
    elif code=="TC":
        st.subheader("🔁 RSI & MACD Overview")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["price"]).macd()
        st.line_chart(df.set_index("date")[["RSI","MACD"]])
    elif code=="TD":
        st.subheader("💡 Indicators – Explained")
        st.markdown("- **RSI**: momentum\n- **MACD**: trend signals\n- **MA5**: short-term smoothing")
    elif code=="TE":
        st.subheader("⚠️ Indicator Caveats")
        st.markdown("- RSI: false signals\n- MACD: lagging\n- MA5: whipsaws")
    elif code=="BK":
        st.subheader("📊 Backtest vs NIFTY 50")
        bench = get_benchmark_df(180)
        df["norm"] = df["price"]/df["price"].iloc[0]
        bench["norm"] = bench["price"]/bench["price"].iloc[0]
        plt.figure(figsize=(8,4))
        plt.plot(df["date"], df["norm"], label=resolved)
        plt.plot(bench["date"], bench["norm"], label="NIFTY")
        st.pyplot(plt)
        st.metric(f"{resolved} Return", f"{(df['norm'].iloc[-1]-1)*100:.2f}%")
        st.metric("NIFTY Return", f"{(bench['norm'].iloc[-1]-1)*100:.2f}%")
    elif code=="SIP":
        st.subheader("💹 SIP Engine")
        amt = st.sidebar.number_input("Monthly SIP (₹)", 100, 10000, 1000, step=100)
        term = st.sidebar.slider("SIP Term (months)", 3, 60, 12)
        inv,fv,cagr = simulate_sip(df, amt, term)
        st.metric("Invested", f"₹{inv}")
        st.metric("Final Value", f"₹{fv}")
        st.metric("CAGR", f"{cagr}%")
        st.markdown("Units = invest ÷ price; Final value = units × latest price")
        plot_main_graph(df)
    elif code=="DL":
        st.subheader("🧠 LSTM Forecast")
        fc = lstm_forecast(df, days_ahead)
        st.metric("LSTM Predicted", f"₹{fc}")
        plot_main_graph(df, [fc], "LSTM Next-Day Forecast")
    elif code=="PC":
        st.subheader("✅ SIP vs LSTM")
        amt = st.sidebar.number_input("Monthly SIP (₹)", 100, 10000, 1000, step=100, key="pc_amt")
        term = st.sidebar.slider("Term (months)", 3, 60, 12, key="pc_term")
        inv,fv,cagr = simulate_sip(df, amt, term)
        lstmp = lstm_forecast(df, days_ahead)
        st.metric("SIP ➤ Final", f"₹{fv} ({cagr}% CAGR)")
        st.metric("LSTM ➤ Forecast", f"₹{lstmp}")
        plot_main_graph(df, [lstmp], "LSTM vs SIP Value")
# Execution Trigger
if st.button("🚀 Run Strategy"):
    df = get_price_df(resolved)
    nav = get_live_nav(resolved)
    for i in range(simulate_runs):
        if simulate_runs > 1:
            st.markdown(f"### Run {i+1}/{simulate_runs}")
        run_strategy(strategy.split()[0], df.copy(), nav)
