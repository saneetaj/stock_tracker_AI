import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import datetime
import time
import plotly.graph_objects as go

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

# Initialize OpenAI client
#client = openai.OpenAI(api_key=openai_api_key)
client = openai #instead of openai.OpenAI
openai.api_key = openai_api_key

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

# Function to fetch market sentiment using OpenAI (optimized for rate limits)
import time
import openai

def get_market_sentiment(tickers):
    sentiments = {}
    for ticker in tickers:
        attempt = 1
        while attempt <= 5:  # Retry up to 5 times
            try:
                prompt = f"Analyze the market sentiment for {ticker}. Provide a short summary (bullish, bearish, or neutral) with key reasons."
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                # Log response headers to check rate limit status
                print(f"Response Headers for {ticker}: {response.headers}")
                sentiments[ticker] = response.choices[0].message.content
                break  # Exit retry loop if successful
            except openai.RateLimitError:
                if attempt < 5:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit error for {ticker}, retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    sentiments[ticker] = "⚠️ Rate limit reached. Try again later."
                    break
            except Exception as e:
                sentiments[ticker] = f"⚠️ Error: {e}"
                break
        time.sleep(1)  # Small delay between tickers
 
    return sentiments
    
# Streamlit UI
st.title("📈 AI-Powered Stock Tracker")
st.sidebar.subheader(f"📢 Sentiment for {ticker}")

# User input for multiple stock tickers
tickers = st.text_input("Enter stock ticker symbols (comma-separated)", "AAPL,TSLA,GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# "Analyze" button to trigger stock tracking
if st.button("🔍 Analyze"):
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

        # Fetch market sentiment (reduces API calls)
        #sentiment = get_market_sentiment(ticker)
        #st.write(f"📢 **Market Sentiment for {ticker}:** {sentiment}")
        sentiments = get_market_sentiment(ticker)
        if ticker in sentiments:
            st.sidebar.write(sentiments[ticker])

    # Auto-refresh logic
    st.success("✅ Stock data updates every 5 minutes!")
    time.sleep(300)  # Refresh every 5 minutes
