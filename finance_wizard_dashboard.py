import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import openai
from datetime import datetime, timedelta
import finnhub
import ta

# Set up API keys from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])

# Cache functions for performance optimization
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_data(ticker, start_date, end_date, source="yahoo"):
    """Fetch financial data from Yahoo Finance or Finnhub."""
    try:
        if source == "yahoo":
            data = yf.download(ticker, start=start_date, end=end_date)
        elif source == "finnhub":
            start_ts = int(datetime.strptime(str(start_date), "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(str(end_date), "%Y-%m-%d").timestamp())
            data = pd.DataFrame(finnhub_client.stock_candles(ticker, "D", start_ts, end_ts))
            data['Date'] = pd.to_datetime(data['t'], unit='s')
            data = data.set_index('Date')[['c']].rename(columns={'c': 'Close'})
        if data.empty:
            raise ValueError("No data found for the given ticker.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_resource
def load_nn_model():
    """Load the pre-trained TensorFlow neural network model."""
    # Adjust path based on your project structure, e.g., /app/models/nn_model.h5 in Docker
    return tf.keras.models.load_model('models/nn_model.h5')

# Sentiment analysis using Finnhub
def get_sentiment(ticker):
    """Fetch sentiment score from Finnhub news."""
    try:
        news = finnhub_client.company_news(ticker, _from=str(datetime.today() - timedelta(days=30)), to=str(datetime.today()))
        if not news:
            return 0.0
        sentiment_scores = [item.get('sentiment', 0) for item in news if 'sentiment' in item]
        return np.mean(sentiment_scores) if sentiment_scores else 0.0
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return 0.0

# AI-generated insights using OpenAI
def generate_insight(ticker, data_summary, prediction):
    """Generate insights using OpenAI based on data and forecast."""
    prompt = f"Provide a concise insight on the forecast for {ticker}. Historical data summary: {data_summary}. Predicted price in 30 days: {prediction:.2f}."
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Insight generation failed: {e}"

# Main app
st.title("🧙 Finance Wizard Dashboard")
st.markdown("A powerful tool for financial forecasting and analysis.")

# Sidebar for user inputs
st.sidebar.header("📊 Settings")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())
data_source = st.sidebar.selectbox("Data Source", ["Yahoo Finance", "Finnhub"])
model_choice = st.sidebar.selectbox("Forecasting Model", ["Linear Regression", "Prophet", "Neural Network"])
show_sentiment = st.sidebar.checkbox("Show Sentiment Analysis")
show_indicators = st.sidebar.multiselect("Technical Indicators", ["SMA (20)", "RSI (14)"])

# Input validation
if not ticker:
    st.error("Please enter a valid ticker symbol.")
else:
    # Fetch data
    with st.spinner(f"Fetching data for {ticker} from {data_source}..."):
        source_key = "yahoo" if data_source == "Yahoo Finance" else "finnhub"
        data = fetch_data(ticker, start_date, end_date, source=source_key)

    if data is not None:
        st.success(f"Data loaded successfully for {ticker}")

        # Prepare data for modeling
        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']  # For Prophet compatibility

        # Add technical indicators if selected
        if show_indicators:
            if "SMA (20)" in show_indicators:
                data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            if "RSI (14)" in show_indicators:
                data['RSI_14'] = ta.momentum.rsi(data['Close'], window=14)

        # Forecasting based on model choice
        with st.spinner(f"Running {model_choice} model..."):
            if model_choice == "Prophet":
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(df)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                prediction = forecast['yhat'].iloc[-1]
                data_summary = f"Last close: {df['y'].iloc[-1]:.2f}, Avg. close (last 30 days): {df['y'].tail(30).mean():.2f}"

                # Interactive Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='orange', dash='dot')))
                for indicator in show_indicators:
                    if indicator == "SMA (20)":
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA (20)', line=dict(color='green')))
                    if indicator == "RSI (14)":
                        fig.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], name='RSI (14)', line=dict(color='purple'), yaxis='y2'))
                fig.update_layout(
                    title=f"{ticker} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100]),
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            elif model_choice == "Linear Regression":
                df['days'] = (df['ds'] - df['ds'].min()).dt.days
                X = df[['days']]
                y = df['y']
                model = LinearRegression().fit(X, y)
                future_days = np.array([df['days'].max() + i for i in range(1, 31)]).reshape(-1, 1)
                forecast = model.predict(future_days)
                prediction = forecast[-1]
                data_summary = f"Last close: {df['y'].iloc[-1]:.2f}, Avg. close (last 30 days): {df['y'].tail(30).mean():.2f}"
                future_dates = [df['ds'].max() + timedelta(days=i) for i in range(1, 31)]

                # Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast, name='Forecast', line=dict(color='orange', dash='dot')))
                for indicator in show_indicators:
                    if indicator == "SMA (20)":
                        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA (20)', line=dict(color='green')))
                    if indicator == "RSI (14)":
                        fig.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], name='RSI (14)', line=dict(color='purple'), yaxis='y2'))
                fig.update_layout(
                    title=f"{ticker} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100]),
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            elif model_choice == "Neural Network":
                model = load_nn_model()
                sequence_length = 60  # Adjust based on your model
                if len(df) >= sequence_length:
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df[['y']])
                    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
                    prediction_scaled = model.predict(last_sequence)[0][0]
                    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
                    data_summary = f"Last close: {df['y'].iloc[-1]:.2f}, Avg. close (last 30 days): {df['y'].tail(30).mean():.2f}"
                    st.write(f"Predicted price in 30 days: {prediction:.2f}")
                else:
                    st.error("Insufficient data for Neural Network prediction.")
                    prediction, data_summary = None, None

        # Display sentiment analysis
        if show_sentiment:
            sentiment = get_sentiment(ticker)
            st.metric("Sentiment Score", f"{sentiment:.2f}", delta=None)

        # Generate and display AI insight
        if prediction and data_summary:
            insight = generate_insight(ticker, data_summary, prediction)
            st.subheader("🤖 AI Insight")
            st.write(insight)

    else:
        st.error("Failed to fetch data. Please check the ticker symbol or try again.")

# Disclaimer
st.markdown("---")
st.markdown("**Disclaimer**: Forecasts and insights are for informational purposes only and do not constitute financial advice.")        
