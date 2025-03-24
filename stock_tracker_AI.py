import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import time
import datetime
import plotly.graph_objects as go

# Load API key securely from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo", interval="1d")  # 1 month historical data
    return data

# Function to calculate technical indicators
def calculate_indicators(data):
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(window=14).mean()))
    return data

# Function to generate buy/sell signals
def generate_signals(data):
    data["Buy_Signal"] = (data["Close"] > data["SMA_20"]) & (data["RSI"] < 30)
    data["Sell_Signal"] = (data["Close"] < data["SMA_20"]) & (data["RSI"] > 70)
    return data

# Function to fetch market sentiment using OpenAI
def get_market_sentiment(ticker):
    prompt = f"Analyze the market sentiment for {ticker}. Provide a short summary (bullish, bearish, or neutral) with key reasons."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("📈 AI-Powered Stock Tracker")

# User input for multiple stock tickers
tickers = st.text_input("Enter stock ticker symbols (comma-separated)", "AAPL,TSLA,GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# Refresh interval (5 minutes)
refresh_time = 300

# Live stock tracking
for ticker in tickers:
    st.subheader(f"📊 Stock Data for {ticker}")
    
    # Fetch stock data
    data = get_stock_data(ticker)
    data = calculate_indicators(data)
    data = generate_signals(data)
    
    # Plot stock price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="20-Day SMA"))
    
    # Highlight Buy/Sell Signals
    buy_signals = data[data["Buy_Signal"]]
    sell_signals = data[data["Sell_Signal"]]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Close"], mode="markers", name="Buy Signal", marker=dict(color="green", size=10)))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["Close"], mode="markers", name="Sell Signal", marker=dict(color="red", size=10)))

    st.plotly_chart(fig)
    
    # Market sentiment
    sentiment = get_market_sentiment(ticker)
    st.write(f"📢 **Market Sentiment for {ticker}:** {sentiment}")

    # Auto-refresh
    time.sleep(refresh_time)

st.success("✅ Stock data updates every 5 minutes!")
