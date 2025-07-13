import streamlit as st
from moneycontrolPy import MC
import pandas as pd
import numpy as np
from datetime import datetime
import os
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Initialize Moneycontrol wrapper
mc = MC()

# Page setup
st.set_page_config(page_title="Finance Wizard Pro Dashboard", layout="centered")
st.title("🧙 Finance Wizard: Price Predictor + Reversal Alerts")

# Input UI
asset_type = st.selectbox("Choose Asset Type", ["mutual_fund", "stock"])
slug_or_symbol = st.text_input("Enter Symbol or Slug (e.g. INFY or hdfc-mid-cap-opportunities-direct-plan-growth)")
days_ahead = st.slider("Days ahead to predict", 1, 30, 7)

hotkey = st.selectbox("Prediction Type (Hotkey)", [
    "🔮 W - Technical Price Prediction",
    "🧠 ML - Machine Learning-Based Forecast",
    "🌐 TA - Trend Analysis",
    "🔀 TE - Trend Reversal Detection"
])

if st.button("Predict Now"):
    if not slug_or_symbol:
        st.error("Please enter a valid symbol or slug.")
    else:
        # Fetch data from Moneycontrol
        try:
            if asset_type == "mutual_fund":
                data = mc.getMutualFund(slug_or_symbol)
                price = float(data['nav'])
                asset_name = slug_or_symbol.replace("-", " ").title()
            else:
                data = mc.getQuotes(slug_or_symbol)
                price = float(data['price'])
                asset_name = data['companyName']
        except:
            st.error("❌ Failed to fetch data. Please check symbol or slug.")
            st.stop()

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = f"{slug_or_symbol}_{asset_type}_log.csv"

        # Log today's price
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "price"])
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([today, price])

        # Load historical data
        df = pd.read_csv(log_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["day_index"] = (df["date"] - df["date"].min()).dt.days

        # ML Model: Linear Regression
        X = df[["day_index"]].values
        y = df["price"].values
        model = LinearRegression().fit(X, y)
        future_index = df["day_index"].max() + days_ahead
        pred_price = model.predict(np.array([[future_index]]))[0]

        # Confidence interval (±1.96 * stddev of residuals)
        y_pred = model.predict(X)
        residuals = y - y_pred
        std_dev = np.std(residuals)
        upper_band = pred_price + 1.96 * std_dev
        lower_band = pred_price - 1.96 * std_dev

        # Trend reversal detection (MA slope)
        df["MA5"] = df["price"].rolling(window=5).mean()
        reversal = False
        if len(df) >= 6:
            recent_slope = df["MA5"].iloc[-1] - df["MA5"].iloc[-2]
            prev_slope = df["MA5"].iloc[-2] - df["MA5"].iloc[-3]
            if np.sign(recent_slope) != np.sign(prev_slope):
                reversal = True

        # Plotting
        st.subheader(f"📊 {asset_name} Trend & Prediction")
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["price"], label="Actual Price", marker='o')
        ax.plot(df["date"], df["MA5"], label="MA-5 Trend", linestyle="--", color="orange")
        ax.errorbar(datetime.now() + pd.to_timedelta(days_ahead, unit="d"),
                    pred_price, yerr=1.96*std_dev, fmt='ro', label="Predicted ±95% CI")
        ax.legend()
        st.pyplot(fig)

        # Results
        st.success(f"🔮 Predicted Price in {days_ahead} days: ₹{round(pred_price,2)}")
        st.info(f"📈 95% Confidence Interval: ₹{round(lower_band,2)} – ₹{round(upper_band,2)}")
        st.write(f"🧠 Prediction Method: {hotkey}")

        if reversal:
            st.error("⚠️ Potential Trend Reversal Detected!")
        else:
            st.success("✅ No reversal trend detected — current momentum stable.")
