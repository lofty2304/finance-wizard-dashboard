import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error
import ta
import yfinance as yf
import openai
import requests
from datetime import datetime, timedelta
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

--- API Keys ---

openai.api_key = st.secrets["OPENAI_API_KEY"]

--- UI Setup ---

st_autorefresh(interval=600000, key="auto_refresh")
st.set_page_config(page_title="Finance Wizard", layout="wide")
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

--- Sidebar ---

with st.sidebar:
st.markdown("### ⚙️ Settings")
days_ahead = st.slider("📅 Days Ahead to Forecast", 1, 30, 7)
show_r2 = st.checkbox("📊 Show R² Scores", value=True)
simulate_runs = st.slider("🔁 Multi-run Simulation Count", 1, 20, 1)
st.markdown("Each strategy uses this value to forecast.")

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
"📈 BK - Backtest vs Benchmark",
"💹 SIP - SIP Return Engine",
"🧠 DL - LSTM Forecasting",
"📊 PC - Profit vs SIP Comparison",
"🟢 SA - Optimistic Scenario",
"🟡 SC - Conservative Scenario",
"🔴 SD - Pessimistic Scenario",
"⚖️ SE - Extreme Shock"
])

st.markdown("## 🔎 Enter Stock / Fund Symbol")
user_input = st.text_input("Ticker / Fund / AMFI Code", "NBCC.NS")

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
st.caption(f"🧾 Resolved Ticker: {resolved}")

--- Data Utilities ---

@st.cache_data(ttl=300)
def get_live_nav(ticker):
try:
info = yf.Ticker(ticker).info
nav = info.get("navPrice") or info.get("regularMarketPrice")
if nav: return round(nav, 2), "Yahoo Finance"
except: pass
try:
txt = requests.get("https://www.amfiindia.com/spages/NAVAll.txt").text
for line in txt.splitlines():
if ticker.upper() in line:
val = float(line.split(";")[-1])
return round(val, 2), "AMFI India"
except: pass
return None, "Unavailable"

def get_stock_price(symbol, fallback_nav):
try:
df = yf.Ticker(symbol).history(period="180d")
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

--- Strategy Functions ---

def linear_forecast(df, days_ahead):
model = LinearRegression()
model.fit(df[["day_index"]], df["price"])
future_days = np.arange(df["day_index"].max() + 1, df["day_index"].max() + days_ahead + 1).reshape(-1, 1)
pred = model.predict(future_days)
return pred.tolist(), model.score(df[["day_index"]], df["price"])

def polynomial_forecast(df, days_ahead, degree=3):
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(df[["day_index"]])
model = LinearRegression()
model.fit(X_poly, df["price"])
future_days = np.arange(df["day_index"].max() + 1, df["day_index"].max() + days_ahead + 1).reshape(-1, 1)
pred = model.predict(poly.transform(future_days))
return pred.tolist(), model.score(X_poly, df["price"])

def random_forest_forecast(df, days_ahead):
model = RandomForestRegressor(n_estimators=100)
model.fit(df[["day_index"]], df["price"])
future_days = np.arange(df["day_index"].max() + 1, df["day_index"].max() + days_ahead + 1).reshape(-1, 1)
pred = model.predict(future_days)
return pred.tolist(), model.score(df[["day_index"]], df["price"])

def xgboost_forecast(df, days_ahead):
model = xgb.XGBRegressor()
model.fit(df[["day_index"]], df["price"])
future_days = np.arange(df["day_index"].max() + 1, df["day_index"].max() + days_ahead + 1).reshape(-1, 1)
pred = model.predict(future_days)
return pred.tolist(), model.score(df[["day_index"]], df["price"])

def prophet_forecast(df, days_ahead):
df_p = df[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
model = Prophet()
model.fit(df_p)
future = model.make_future_dataframe(periods=days_ahead)
forecast = model.predict(future)
future_data = forecast.tail(days_ahead)["yhat"].values.tolist()
return future_data, model

--- LSTM Model ---

def lstm_forecast(df, days_ahead):
data = df["price"].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = [], []  
for i in range(60, len(data_scaled) - days_ahead):  
    X.append(data_scaled[i-60:i])  
    y.append(data_scaled[i + days_ahead - 1])  
X, y = np.array(X), np.array(y)  

model = Sequential()  
model.add(LSTM(50, activation='relu', return_sequences=False, input_shape=(60, 1)))  
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mse')  
model.fit(X, y, epochs=10, batch_size=16, verbose=0)  

test_input = data_scaled[-60:].reshape(1, 60, 1)  
pred_scaled = model.predict(test_input)  
pred = scaler.inverse_transform(pred_scaled)[0][0]  
return round(pred, 2)

--- SIP Logic ---

def run_sip_backtest(df):
sip_amount = 1000
dates = df["date"]
invested = len(dates) * sip_amount
units = [sip_amount / p if p else 0 for p in df["price"]]
final_price = df["price"].iloc[-1]
value = sum(units) * final_price
return round(invested, 2), round(value, 2), round((value - invested) / invested * 100, 2)

--- Plot ---

def plot_main_graph(df, forecast=None, title="Forecast"):
fig, ax = plt.subplots()
ax.plot(df["date"], df["price"], label="Historical Price")
if "MA5" in df.columns:
ax.plot(df["date"], df["MA5"], label="MA5")
if forecast is not None:
last_date = df["date"].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
ax.plot(future_dates, forecast, label=title, linestyle="--")
ax.legend(); ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

--- Run Strategy ---

def run_strategy(code, df, days_ahead, nav, nav_source, resolved, show_r2=True):
df["day_index"] = (df["date"] - df["date"].min()).dt.days
df["MA5"] = df["price"].rolling(5).mean()

if code == "W":  
    st.subheader("🔮 Predict One Stock (Linear Forecast)")  
    forecast, score = linear_forecast(df, days_ahead)  
    plot_main_graph(df, forecast, "Linear Forecast")  
    st.metric("Forecasted Price", f"₹{round(forecast[-1], 2)}")  
    if show_r2: st.markdown(f"**R² Score:** {round(score, 4)}")  
    st.markdown("This strategy uses a basic linear regression model to estimate future prices. It assumes trend continuity.")  

elif code == "ML":  
    st.subheader("🧠 Polynomial Forecast")  
    forecast, score = polynomial_forecast(df, days_ahead)  
    plot_main_graph(df, forecast, "Polynomial Forecast")  
    st.metric("Forecasted Price", f"₹{round(forecast[-1], 2)}")  
    if show_r2: st.markdown(f"**R² Score:** {round(score, 4)}")  
    st.markdown("Polynomial regression captures non-linear trends. Degree=3 fits curves better than simple linear.")  

elif code == "MC":  
    st.subheader("⚖️ Model Comparison: Linear vs XGBoost vs Random Forest")  
    lin_pred, lin_r2 = linear_forecast(df, days_ahead)  
    xgb_pred, xgb_r2 = xgboost_forecast(df, days_ahead)  
    rf_pred, rf_r2 = random_forest_forecast(df, days_ahead)  
    plot_main_graph(df, xgb_pred, "XGBoost Forecast")  
    st.markdown(f"- Linear R²: {round(lin_r2,4)}\n- XGBoost R²: {round(xgb_r2,4)}\n- RF R²: {round(rf_r2,4)}")  
    st.markdown("This compares 3 ML models to estimate accuracy. Higher R² → better fit to past data.")  

elif code == "D":  
    st.subheader("📉 Downside Risk Estimation")  
    volatility = df["price"].pct_change().std() * 100  
    downside = df["price"].mean() - df["price"].std() * 1.5  
    st.metric("📊 Volatility", f"{round(volatility, 2)}%")  
    st.metric("⚠️ Downside Risk Price", f"₹{round(downside, 2)}")  
    st.markdown("Downside price = Mean - 1.5×Std Dev. High volatility increases risk of large drop.")  
    plot_main_graph(df)  

elif code in ["SA", "SC", "SD", "SE"]:  
    st.subheader("🔁 Scenario Forecasting")  
    map_factor = {"SA": 1.02, "SC": 1.00, "SD": 0.98, "SE": 0.95}  
    label_map = {  
        "SA": "Optimistic Scenario (2% growth/day)",  
        "SC": "Conservative (flat trend)",  
        "SD": "Pessimistic (2% drop/day)",  
        "SE": "Extreme Shock (5% drop/day)"  
    }  
    growth = map_factor[code]  
    forecast = [df["price"].iloc[-1] * (growth ** i) for i in range(days_ahead)]  
    plot_main_graph(df, forecast, label_map[code])  
    st.metric("Scenario Growth Factor", f"{growth}x")  
    st.markdown(f"This is a **{label_map[code]}** assumption-based simulation. Not model-driven but helps stress test possible futures.")  

elif code == "DL":  
    st.subheader("🧠 LSTM Forecasting (Deep Learning)")  
    pred = lstm_forecast(df, days_ahead)  
    plot_main_graph(df)  
    st.metric("LSTM Predicted Price", f"₹{pred}")  
    st.markdown("LSTM is trained on sequential data (60 days). It's good at capturing short-term temporal patterns.")  

elif code == "SIP":  
  
    st.subheader("💹 SIP Investment Simulator")  
    sip_input = st.number_input("Monthly SIP Amount (₹)", min_value=100, value=1000, step=100)  
    invested = len(df) * sip_input  
    units = [sip_input / p if p else 0 for p in df["price"]]  
    total_units = sum(units)  
    final_value = total_units * df["price"].iloc[-1]  
    ret = round((final_value - invested) / invested * 100, 2)  

    st.metric("Invested", f"₹{invested}")  
    st.metric("Value", f"₹{round(final_value, 2)}")  
    st.metric("Return (%)", f"{ret}%")  
    plot_main_graph(df)  
    st.markdown("""  
    **Formula**: Units Bought = SIP Amount / Price at time    
    Final Value = Sum of Units × Final Price    
    Return % = (Final Value - Invested) / Invested × 100  
    """)  


elif code == "PC":  
    st.subheader("📊 SIP vs LSTM Comparison")  
    lstm_pred = lstm_forecast(df, days_ahead)  
    invested, value, ret = run_sip_backtest(df)  
    plot_main_graph(df)  
    st.metric("SIP Value", f"₹{value}")  
    st.metric("LSTM Predicted Price", f"₹{lstm_pred}")  
    st.markdown("SIP = periodic investing | LSTM = one-time price prediction.\nCompare stability vs prediction.")  

elif code == "TI":  
    st.subheader("📉 Technical Indicator Forecast")  
    df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()  
    st.line_chart(df.set_index("date")[["price", "RSI"]])  
    st.markdown("**RSI (Relative Strength Index)** shows momentum. Above 70 = overbought. Below 30 = oversold.")  

elif code == "TC":  
    st.subheader("↔️ Compare Technical Indicators")  
    df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()  
    df["MACD"] = ta.trend.MACD(df["price"]).macd()  
    st.line_chart(df.set_index("date")[["RSI", "MACD"]])  
    st.markdown("- **MACD** shows trend change.\n- **RSI** shows overbought/oversold behavior.")  

elif code == "TD":  
    st.subheader("💡 Technical Indicator Explanation")  
    st.markdown("""  
    - **RSI**: Measures speed/strength of price movements.  
    - **MACD**: Shows convergence/divergence of two EMAs.  
    - **MA5**: 5-day moving average to smooth short-term noise.  
    """)  

elif code == "TE":  
    st.subheader("⚠️ Indicator Limitations")  
    st.markdown("""  
    - **RSI**: False signals in trending markets.  
    - **MACD**: Lags behind rapid price changes.  
    - **MA5**: Too short. Prone to whipsaws.  
    """)  

elif code == "MD":  
    st.subheader("🧐 Model Explanation")  
    st.markdown("""  
    | Model | Strength | Limitation |  
    |-------|----------|------------|  
    | Linear | Simple, interpretable | Can't handle curves |  
    | Polynomial | Captures curves | Overfitting risk |  
    | Random Forest | Handles noise | Slower |  
    | XGBoost | Very accurate | Needs tuning |  
    | Prophet | Good for seasonality | Complex |  
    | LSTM | Deep learning | Needs data, GPU |  
    """)  

elif code == "S":  
    st.subheader("🕵️ Stock Deep Dive")  
    df["RSI"] = ta.momentum.RSIIndicator(df["price"]).rsi()  
    df["MACD"] = ta.trend.MACD(df["price"]).macd()  
    df["Signal"] = ta.trend.MACD(df["price"]).macd_signal()  
    st.dataframe(df[["date", "price", "RSI", "MACD", "Signal", "MA5"]].tail())  
    plot_main_graph(df)  
    st.markdown("**RSI, MACD, and Signal line** give technical view.\nUse in combination for better decisions.")  

else:  
    st.warning("⚠️ Strategy not implemented yet or invalid code.")

--- Run Button ---

if st.button("🚀 Run Strategy"):
nav, nav_source = get_live_nav(resolved)
df = get_stock_price(resolved, nav)
if df is None or df.empty:
st.error("❌ No data found. Try another stock or fund.")
else:
strategy_code = strategy.split("-")[0].split()[-1].strip()
for i in range(simulate_runs):
if simulate_runs > 1:
st.markdown(f"### Simulation {i+1}/{simulate_runs}")
run_strategy(strategy_code, df.copy(), days_ahead, nav, nav_source, resolved, show_r2)
