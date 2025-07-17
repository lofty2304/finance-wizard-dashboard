# --- Finance Wizard: Master Code — Part 1 ---

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
import ta, yfinance as yf, openai, requests
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense

# API & UI Setup
openai.api_key = st.secrets["OPENAI_API_KEY"]
st_autorefresh(interval=600000, key="auto_refresh")
st.set_page_config(page_title="Finance Wizard", layout="wide")
st.sidebar.markdown("### ⚙️ Settings")
days_ahead = st.sidebar.slider("📅 Days Ahead", 1, 30, 7)
show_r2 = st.sidebar.checkbox("📊 Show R²", True)
simulate_runs = st.sidebar.slider("🔁 Simulations", 1, 20, 1)
# Layout, Ticker Input & Sentiment Function

strategy = st.selectbox("📌 Choose Strategy", [
    "W - One Stock",
    "ML - Polynomial",
    "MC - Model Comparison",
    "MD - Model Explanation",
    "D - Downside Risk",
    "S - Stock Deep Dive",
    "TI - Tech Indicators",
    "TC - Compare Indicators",
    "TD - Explain Indicators",
    "TE - Limitations",
    "BK - Backtest vs Benchmark",
    "SIP - SIP Engine",
    "DL - LSTM Forecast",
    "PC - SIP vs LSTM",
    "SA SC SD SE - Scenarios"
])

resolved = st.text_input("Ticker / AMFI", "NBCC.NS").upper()
st.caption(f"🔍 Resolved Ticker: **{resolved}**")

@st.cache_data(ttl=600)
def get_sentiment_score(q):
    try:
        key = st.secrets["NEWS_API_KEY"]
        url = f"https://newsapi.org/v2/everything?q={q}&language=en&pageSize=3&sortBy=publishedAt&apiKey={key}"
        data = requests.get(url).json()
        hl = [a["title"] for a in data.get("articles", [])]
        prompt = " ".join(hl) + "\nSentiment (0–1):"
        ans = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"Sentiment classifier"},
                      {"role":"user","content":prompt}]
        )["choices"][0]["message"]["content"]
        score = 0.5 + (0.2 if "positive" in ans.lower() else -0.2 if "negative" in ans.lower() else 0)
        return max(0, min(1, round(score,2))), ans
    except:
        return 0.5, "Neutral"
# Fetch & Forecast Functions

@st.cache_data(ttl=300)
def get_live_nav(tick):
    info = yf.Ticker(tick).info
    return round(info.get("navPrice", info.get("regularMarketPrice", 0)), 2), "Yahoo"

def get_stock_price(tick, fallback_nav):
    df = yf.Ticker(tick).history(period="180d").reset_index()[["Date","Close"]]
    df.columns = ["date","price"]
    return df if not df.empty else pd.DataFrame({"date":pd.date_range(end=datetime.today(), periods=90),"price":[fallback_nav]*90})

def linear_forecast(df, days):
    X = df["day_index"].values.reshape(-1,1)
    y = df["price"].values
    m = LinearRegression().fit(X,y)
    fX = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1)
    return m.predict(fX).tolist(), m.score(X,y)

def polynomial_forecast(df, days):
    deg = PolynomialFeatures(3)
    Xp = deg.fit_transform(df[["day_index"]])
    y = df["price"]
    m = LinearRegression().fit(Xp,y)
    fX = deg.transform(np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1))
    return m.predict(fX).tolist(), m.score(Xp,y)

def random_forest_forecast(df, days):
    X = df["day_index"].values.reshape(-1,1)
    y = df["price"]
    m = RandomForestRegressor().fit(X,y)
    fX = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1)
    return m.predict(fX).tolist(), m.score(X,y)

def xgboost_forecast(df, days):
    X = df["day_index"].values.reshape(-1,1)
    y = df["price"]
    m = xgb.XGBRegressor().fit(X,y)
    fX = np.arange(df["day_index"].max()+1, df["day_index"].max()+1+days).reshape(-1,1)
    return m.predict(fX).tolist(), m.score(X,y)
# Prophet with timezone fix
def prophet_forecast(df, days):
    pdf = df.rename(columns={"date":"ds","price":"y"})
    pdf["ds"] = pdf["ds"].dt.tz_localize(None)
    m = Prophet().fit(pdf)
    future = m.make_future_dataframe(periods=days)
    fc = m.predict(future)["yhat"].tail(days).tolist()
    return fc, m

# LSTM forecast
def lstm_forecast(df, days):
    data = df["price"].values.reshape(-1,1)
    scaler = MinMaxScaler().fit(data)
    sdata = scaler.transform(data)
    X,y = [],[]
    for i in range(60, len(sdata)-days):
        X.append(sdata[i-60:i])
        y.append(sdata[i+days-1])
    X,y = np.array(X), np.array(y)
    model = Sequential([LSTM(50, activation='relu', input_shape=(60,1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y, epochs=10, batch_size=16, verbose=0)
    pred = model.predict(sdata[-60:].reshape(1,60,1))
    return round(scaler.inverse_transform(pred)[0][0],2)
# SIP & Benchmark utilities
def calculate_cagr(start,end,months):
    return round(((end/start)**(12/months)-1)*100,2)

def simulate_sip(df, amt, months):
    df1 = df.iloc[-months:]
    if df1.empty: return 0,0,0
    invested = amt * months
    units = [amt/price for price in df1["price"]]
    fv = sum(units) * df1["price"].iloc[-1]
    cagr = calculate_cagr(amt, fv/months, months)
    return invested, round(fv,2), cagr

def get_benchmark_df(days):
    df = yf.Ticker("^NSEI").history(period=f"{days}d").reset_index()[["Date","Close"]]
    df.columns = ["date","price"]
    return df
# Plot utility
def plot_main_graph(df, forecast=None, title="Forecast"):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df["date"], df["price"], label="Historical")
    if "MA5" in df.columns:
        ax.plot(df["date"], df["MA5"], label="MA5")
    if forecast:
        start = df["date"].iloc[-1]
        ax.plot([start + timedelta(i+1) for i in range(len(forecast))], forecast, linestyle="--", label=title)
    ax.legend(); ax.set_title(title)
    st.pyplot(fig)
def adjust_forecast(base, score, code):
    factors = {"SA":1.05,"SC":1.02,"SD":0.98,"SE":0.95}
    return [round(p * (factors[code] + (score-0.5)*0.1),2) for p in base]

def run_strategy(code, df, days_ahead, nav, _nav_src, ticker, show_r2):
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["MA5"] = df["price"].rolling(5).mean()

    sentiment, raw = get_sentiment_score(ticker)

    if code in ["SA","SC","SD","SE"]:
        st.subheader(f"📈 Scenario Forecast — {code}")
        base, _ = linear_forecast(df, days_ahead)
        adj = adjust_forecast(base, sentiment, code)
        plot_main_graph(df, adj, f"{code} Scenario")
        st.metric("Sentiment", sentiment)
        st.caption(raw)
        return

    if code=="W":
        st.subheader("🔮 Prophet Forecast")
        fc,_ = prophet_forecast(df, days_ahead)
        plot_main_graph(df, fc, "Prophet Forecast")
        st.markdown(f"Forecast from {df['date'].iloc[-1].strftime('%Y-%m-%d')} to {(df['date'].iloc[-1]+timedelta(days=days_ahead)).strftime('%Y-%m-%d')}.")

    elif code=="ML":
        st.subheader("📐 Polynomial Forecast")
        fc,r2 = polynomial_forecast(df, days_ahead)
        plot_main_graph(df, fc)
        st.metric("R²", round(r2,3))

    elif code=="MC":
        st.subheader("🔧 Model Comparison")
        methods = [linear_forecast, polynomial_forecast, random_forest_forecast, xgboost_forecast]
        names = ["Linear","Poly","RF","XGB"]
        metrics = {}
        for fn,name in zip(methods,names):
            fc,r2 = fn(df,days_ahead)
            metrics[name] = (fc[-1], round(r2,3))
        st.json({n:f"{v[0]:.2f} (R²={v[1]})" for n,v in metrics.items()})
        plot_main_graph(df, polynomial_forecast(df,days_ahead)[0], "Polynomial Forecast")

    elif code=="MD":
        st.subheader("🧠 ML Model Explanation")
        st.markdown("""
        - **Linear** = trend line  
        - **Polynomial** = non-linear fit  
        - **RF / XGB** = tree ensembles  
        Props: use best based on R²
        """)
        fc,_ = prophet_forecast(df, days_ahead)
        plot_main_graph(df, fc, "Explanation Forecast")

    elif code=="D":
        st.subheader("⚠️ Downside Risk")
        vol = df["price"].pct_change().std() *100
        ds = df["price"].mean() - 1.5*df["price"].std()
        st.metric("Volatility", f"{vol:.2f}%")
        st.metric("Downside Price", f"₹{ds:.2f}")
        plot_main_graph(df)

    elif code=="S":
        st.subheader("🕵️ Stock Deep Dive")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["price"]).macd()
        df["Signal"] = ta.trend.MACD(df["price"]).macd_signal()
        st.dataframe(df.tail(5)[["date","price","RSI","MACD","Signal","MA5"]])
        plot_main_graph(df)

    elif code=="TI":
        st.subheader("📉 Technical Indicators")
        df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()
        st.line_chart(df.set_index("date")[["price","RSI"]])
        st.markdown("RSI >70 overbought, <30 oversold")

    elif code=="TC":
        st.subheader("🔁 RSI & MACD")
        df["RSI"], df["MACD"] = ta.momentum.RSIIndicator(df["price"]).rsi(), ta.trend.MACD(df["price"]).macd()
        st.line_chart(df.set_index("date")[["RSI","MACD"]])

    elif code=="TD":
        st.subheader("💡 Indicators Explained")
        st.markdown("- RSI = momentum\n- MACD = trend shifts\n- MA5 = smoothing")

    elif code=="TE":
        st.subheader("⚠️ Indicator Limitations")
        st.markdown("- RSI lags\n- MACD false signals\n- MA5 prone to whipsaws")

    elif code=="BK":
        st.subheader("📊 Backtest vs NIFTY")
        bench = get_benchmark_df(180)
        df["norm"], bench["norm"] = df["price"]/df["price"].iloc[0], bench["price"]/bench["price"].iloc[0]
        plt.figure(figsize=(8,4))
        plt.plot(df["date"], df["norm"], label=resolved)
        plt.plot(bench["date"], bench["norm"], label="NIFTY")
        plt.legend(); st.pyplot(plt); st.write("")
        st.metric(f"{resolved} Return", f"{(df['norm'].iloc[-1]-1)*100:.2f}%")
        st.metric("NIFTY Return", f"{(bench['norm'].iloc[-1]-1)*100:.2f}%")

    elif code=="SIP":
        st.subheader("💹 SIP Engine")
        amt = st.number_input("Monthly SIP (₹)",100,10000,1000,100)
        term = st.slider("Term (months)",3,60,12)
        inv,fv,cagr = simulate_sip(df,amt,term)
        st.metric("Invested", f"₹{inv}")
        st.metric("Final Value", f"₹{fv}")
        st.metric("CAGR", f"{cagr}%")
        st.markdown("Formula: invest/month ÷ price = units; final × price")
        plot_main_graph(df)

    elif code=="DL":
        st.subheader("🧠 LSTM Forecast")
        price = lstm_forecast(df, days_ahead)
        st.metric("Predicted Price", f"₹{price}")
        plot_main_graph(df)

    elif code=="PC":
        st.subheader("📊 SIP vs LSTM")
        amt = st.number_input("SIP (₹)",100,10000,1000,100)
        term = st.slider("Term (months)",3,60,12)
        inv,fv,cagr = simulate_sip(df,amt,term)
        lstm_p = lstm_forecast(df,days_ahead)
        st.metric("SIP Final", f"₹{fv} ({cagr}% CAGR)")
        st.metric("LSTM Forecast", f"₹{lstm_p}")
        plot_main_graph(df)
# Execution
if st.button("🚀 Run Strategy"):
    nav,_ = get_live_nav(resolved)
    df = get_stock_price(resolved, nav)
    if df is None or df.empty:
        st.error("Data not found.")
    else:
        for i in range(simulate_runs):
            if simulate_runs>1: st.markdown(f"### Run {i+1}/{simulate_runs}")
            run_strategy(strategy.split()[0], df.copy(), days_ahead, nav, None, resolved, show_r2)
