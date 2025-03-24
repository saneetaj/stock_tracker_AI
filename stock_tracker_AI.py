import streamlit as st
import openai
import finnhub
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import logging

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
finnhub_api_key = st.secrets["finnhub_api_key"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=finnhub_api_key)

# Function to fetch stock data from Finnhub
def get_stock_data(ticker):
    try:
        # Fetch historical data for the past month (30 days)
        data = finnhub_client.stock_candles(ticker, 'D', int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp()), int(datetime.datetime.now().timestamp()))
        if data['s'] != 'ok':
            return pd.DataFrame()  # Return empty dataframe if the request fails
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df.set_index('t', inplace=True)
        df = df[['c', 'h', 'l', 'o', 'v']]  # Select close, high, low, open, volume columns
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        return df
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

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

# Function to fetch stock news from Finnhub
def get_stock_news(ticker):
    try:
        news = finnhub_client.company_news(ticker, _from=datetime.datetime.now() - datetime.timedelta(days=7), to=datetime.datetime.now())
        if not news:
            return f"No recent news found for {ticker}."
        news_articles = []
        for article in news[:3]:  # Limit to top 3 articles
            title = article['headline']
            url = article['url']
            news_articles.append(f"• {title}: {url}")
        return "\n".join(news_articles)
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return f"⚠️ Could not fetch news for {ticker}."

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
                    max_tokens=200
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

        # Fetch stock data
        data = get_stock_data(ticker)
        if data.empty:
            st.write(f"⚠️ No data available for {ticker}")
            continue

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

    # Auto-refresh logic
    st.success("✅ Stock data updates every 5 minutes!")
    time.sleep(300)  # Refresh every 5 minutes
