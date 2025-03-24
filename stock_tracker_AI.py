import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import requests
import feedparser

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

# Initialize OpenAI client
#openai.api_key = openai_api_key
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

# Function to fetch stock news from Yahoo
import requests
from bs4 import BeautifulSoup

def get_stock_news(ticker):
    """
    Fetches the top 3 news articles for a given stock ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        str: A string containing the formatted news articles, or an error message.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}  # Prevent blocking
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not fetch news for {ticker}: {e}"

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="js-stream-content", limit=3)  # Get top 3 articles
    except Exception as e:
        return f"⚠️ Error parsing HTML for {ticker}: {e}"

    if not articles:
        return f"No recent news found for {ticker}."

    news_articles = []
    for article in articles:
        try:
            title_tag = article.find("h3")
            if title_tag and title_tag.a:
                title = title_tag.text.strip()
                link = f"https://finance.yahoo.com{title_tag.a['href']}"
                news_articles.append(f"• {title}: {link}")  # Improved formatting
        except Exception as e:
            return f"⚠️ Error processing article for {ticker}: {e}" # error for individual articles

    return "\n".join(news_articles)


# Function to fetch market sentiment using OpenAI
def get_market_sentiment(tickers):
    sentiments = {}
    rate_limit_error_flag = False  # Flag to track if rate limit error has occurred

    for ticker in tickers:
        news_data = get_stock_news(ticker)  # Fetch news for each ticker
        st.sidebar.write(news_data)
        attempt = 1

        while attempt <= 5:  # Retry up to 5 times
            try:
                prompt = f"In 200-250 words analyze the market sentiment for {ticker} in the following news:\n{news_data}\nProvide a short summary (bullish, bearish, or neutral) with key reasons."

                response = client.chat.completions.create(
                    model="gpt-4-turbo",  # Ensure the correct model
                    messages=[
                        {"role": "system", "content": "You are a financial news analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100
                )
                sentiments[ticker] = response.choices[0].message.content.strip()
                break  # Exit retry loop if successful

            except openai.OpenAIError as e:  # Catch OpenAI API errors
                if not rate_limit_error_flag:
                    sentiments['error'] = "⚠️ Rate limit reached. Try again later."
                    rate_limit_error_flag = True  # Only show the error once
                if attempt < 5:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    sentiments[ticker] = "⚠️ Rate limit reached. Try again later."
                    break

        time.sleep(2)  # Small delay between tickers
    
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

