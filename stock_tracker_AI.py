import streamlit as st
import openai
import finnhub
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import requests

# Load API keys from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
finnhub_api_key = st.secrets["finnhub_api_key"]

# Initialize OpenAI and Finnhub clients
client = openai.OpenAI(api_key=openai_api_key)
finnhub_client = finnhub.Client(api_key=finnhub_api_key)

# Function to fetch latest stock data
def get_stock_data(ticker):
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_api_key}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

# Function to fetch intraday stock data
def get_intraday_data(ticker):
    now = datetime.datetime.now()
    start_time = int((now - datetime.timedelta(days=1)).timestamp())
    end_time = int(now.timestamp())
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=1&from={start_time}&to={end_time}&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()
    return data if response.status_code == 200 and 'c' in data else get_stock_data(ticker)

# Function to calculate technical indicators
def calculate_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean()))
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["MACD"] = df["EMA_9"] - df["EMA_50"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

# Function to generate buy/sell signals
def generate_signals(df):
    df["Buy_Signal"] = (df["Close"] > df["SMA_20"]) & (df["RSI"] < 30)
    df["Sell_Signal"] = (df["Close"] < df["SMA_20"]) & (df["RSI"] > 70)
    df["Buy_Signal_EMA"] = df["EMA_9"] > df["EMA_50"]
    df["Sell_Signal_EMA"] = df["EMA_9"] < df["EMA_50"]
    df["Buy_Signal_MACD"] = df["MACD"] > df["MACD_Signal"]
    df["Sell_Signal_MACD"] = df["MACD"] < df["MACD_Signal"]
    df["Buy_Signal_Combined"] = df["Buy_Signal"] & df["Buy_Signal_EMA"] & df["Buy_Signal_MACD"]
    df["Sell_Signal_Combined"] = df["Sell_Signal"] | df["Sell_Signal_EMA"] | df["Sell_Signal_MACD"]
    return df

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
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content": "You are a financial news analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400
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
tickers = st.text_input("Enter stock ticker symbol", "AAPL").strip().upper()

if st.button("🔍 Analyze"):
    sentiment = get_market_sentiment(tickers)
    st.sidebar.subheader(f"📢 Sentiment for {tickers}")
    st.sidebar.write(sentiment)
    
    st.subheader(f"📊 Stock Data for {tickers}")
    data = get_intraday_data(tickers)
    if data:
        df = pd.DataFrame({
            'Date': [datetime.datetime.now()],
            'Close': [data['c']],
            'High': [data['h']],
            'Low': [data['l']],
        })
        df = calculate_indicators(df)
        df = generate_signals(df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df["Close"], mode="lines", name="Close Price"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df["SMA_20"], mode="lines", name="20-Day SMA"))
        buy_signals = df[df["Buy_Signal_Combined"]]
        sell_signals = df[df["Sell_Signal_Combined"]]
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals["Close"], mode="markers", name="Buy Signal", marker=dict(color="green", size=10)))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals["Close"], mode="markers", name="Sell Signal", marker=dict(color="red", size=10)))
        st.plotly_chart(fig)
    else:
        st.write(f"⚠️ No data available for {tickers}")
    
   # Auto-refresh logic
    st.success("✅ Stock data updates every 5 minutes!")
    time.sleep(300)  # Refresh every 5 minutes
