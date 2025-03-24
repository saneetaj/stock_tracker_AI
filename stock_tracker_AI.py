import streamlit as st
import openai
import finnhub
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import requests
import logging

# Load OpenAI API key and Finnhub API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
finnhub_api_key = st.secrets["finnhub_api_key"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=finnhub_api_key)

# Function to fetch stock data from Finnhub (for the latest close price)
def get_stock_data(ticker):
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and 'c' in data:
        return data
    else:
        return None

# Function to fetch intraday data (1-minute resolution) for buy/sell signals during market hours
def get_intraday_data(ticker):
    now = datetime.datetime.now()
    start_time = int((now - datetime.timedelta(days=1)).timestamp())  # Yesterday's timestamp
    end_time = int(now.timestamp())  # Current timestamp

    # Fetch 1-minute candlestick data
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=1&from={start_time}&to={end_time}&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and 'c' in data:
        return data
    else:
        return None

# Function to calculate technical indicators on intraday data (e.g., SMA, RSI)
def calculate_indicators(data):
    data["SMA_20"] = data["Close"].rolling(window=20).mean()  # 20-minute Simple Moving Average
    data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(window=14).mean()))  # RSI calculation
    return data

# Function to generate buy/sell signals based on SMA and RSI
def generate_signals(data):
    data["Buy_Signal"] = (data["Close"] > data["SMA_20"]) & (data["RSI"] < 30)
    data["Sell_Signal"] = (data["Close"] < data["SMA_20"]) & (data["RSI"] > 70)
    return data

# Function to fetch stock news from Finnhub
def get_stock_news(ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-03-01&to=2025-03-24&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and data:
        news_articles = []
        for article in data[:3]:  # Limit to top 3 articles
            title = article['headline']
            url = article['url']
            news_articles.append(f"• {title}: {url}")
        return "\n".join(news_articles)
    else:
        return f"⚠️ No news available for {ticker}."

# Function to fetch market sentiment using OpenAI
def get_market_sentiment(tickers):
    sentiments = {}
    rate_limit_error_flag = False

    for ticker in tickers:
        news_data = get_stock_news(ticker)
        st.sidebar.write(news_data)
        attempt = 1

        while attempt <= 5:
            try:
                prompt = f"Analyze the market sentiment for {ticker} in the below news. :\n{news_data}\nProvide a brief summary (bullish, bearish, or neutral) with key reasons. Strictly limit the summary to 250 words max."

                response = client.chat.completions.create(
                    model="gpt-4-turbo", 
                    messages=[
                        {"role": "system", "content": "You are a financial news analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300
                )
                sentiments[ticker] = response.choices[0].message.content.strip()
                break

            except openai.OpenAIError as e:
                if not rate_limit_error_flag:
                    sentiments['error'] = "⚠️ Rate limit reached. Try again later."
                    rate_limit_error_flag = True
                if attempt < 5:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    sentiments[ticker] = "⚠️ Rate limit reached. Try again later."
                    break

        time.sleep(2)

    return sentiments

# Streamlit UI
st.title("📈 AI-Powered Stock Tracker")

# User input for multiple stock tickers
tickers = st.text_input("Enter stock ticker symbol", "AAPL")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# "Analyze" button to trigger stock tracking
if st.button("🔍 Analyze"):
    sentiments = get_market_sentiment(tickers)
    
    # Display sentiment info in the sidebar only if it exists
    for ticker in tickers:
        if ticker in sentiments and sentiments[ticker] != "⚠️ Rate limit reached. Try again later.":
            st.sidebar.subheader(f"📢 Sentiment for {ticker}")
            st.sidebar.write(sentiments[ticker])
        elif ticker in sentiments:
            st.sidebar.subheader(f"📢 Sentiment for {ticker}")
            st.sidebar.write(sentiments[ticker])

        st.subheader(f"📊 Stock Data for {ticker}")

        # Fetch today's intraday performance data
        intraday_data = get_intraday_data(ticker)
        if intraday_data is None:
            st.write(f"⚠️ No intraday data available for {ticker}")
            continue

        # Convert to DataFrame
        df_intraday = pd.DataFrame({
            'Timestamp': [datetime.datetime.fromtimestamp(ts) for ts in intraday_data['t']],
            'Close': intraday_data['c'],
            'High': intraday_data['h'],
            'Low': intraday_data['l'],
            'Open': intraday_data['o'],
            'Volume': intraday_data['v'],
        })

        # Calculate technical indicators for buy/sell signals
        df_intraday = calculate_indicators(df_intraday)
        df_intraday = generate_signals(df_intraday)

        # Generate and display buy/sell signals
        buy_signals = df_intraday[df_intraday["Buy_Signal"]]
        sell_signals = df_intraday[df_intraday["Sell_Signal"]]
        
        st.subheader(f"Buy/Sell Signals for {ticker}")
        if not buy_signals.empty:
            st.write(f"💡 Buy Signals at: {buy_signals['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()}")
        if not sell_signals.empty:
            st.write(f"🚨 Sell Signals at: {sell_signals['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()}")

        # Plot the intraday price data and highlight buy/sell signals
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df_intraday['Timestamp'],
                                     open=df_intraday['Open'], high=df_intraday['High'],
                                     low=df_intraday['Low'], close=df_intraday['Close'],
                                     name="Intraday Candlestick"))

        # Highlight Buy/Sell Signals
        fig.add_trace(go.Scatter(x=buy_signals['Timestamp'], y=buy_signals["Close"], mode="markers", 
                                 name="Buy Signal", marker=dict(color="green", size=10)))
        fig.add_trace(go.Scatter(x=sell_signals['Timestamp'], y=sell_signals["Close"], mode="markers", 
                                 name="Sell Signal", marker=dict(color="red", size=10)))

        st.plotly_chart(fig)

    # Auto-refresh logic (refresh every 5 minutes)
    st.success("✅ Stock data updates every 5 minutes!")
    time.sleep(300)  # Refresh every 5 minutes
