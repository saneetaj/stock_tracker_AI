import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import openai
import time
from datetime import datetime
import plotly.graph_objects as go

# Set OpenAI API Key
openai.api_key = st.secrets["openai_api_key"]

# Function to fetch stock data
def get_stock_data(ticker, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

# Function to calculate technical indicators
def add_technical_indicators(df):
    if df.empty:
        return df
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"])
    return df

# Function to analyze market sentiment using OpenAI
def get_market_sentiment(ticker):
    prompt = f"""Analyze the latest financial news, blogs, and forums about {ticker}. Summarize the market sentiment in 200-300 words and classify it as Buy, Sell, or Neutral."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a financial analyst."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# Function to generate buy/sell signals
def generate_signals(df, sentiment):
    df["Buy_Signal"] = (df["RSI"] < 30) & (df["MACD"] > df["MACD_Signal"]) & (df["ADX"] > 25)
    df["Sell_Signal"] = (df["RSI"] > 70) & (df["MACD"] < df["MACD_Signal"]) & (df["ADX"] > 25)
    sentiment_signal = "Buy" if "Buy" in sentiment else "Sell" if "Sell" in sentiment else "Neutral"
    return df, sentiment_signal

# Streamlit UI
st.title("📈 Multi-Stock Tracker with AI Market Sentiment")

tickers = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA, MSFT):", "AAPL,TSLA,MSFT")
tickers_list = [ticker.strip().upper() for ticker in tickers.split(",")]

if st.button("Analyze"):
    for ticker in tickers_list:
        df = get_stock_data(ticker)
        df = add_technical_indicators(df)
        sentiment = get_market_sentiment(ticker)
        df, sentiment_signal = generate_signals(df, sentiment)

        # Plot stock chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name="Candlesticks"))
        st.subheader(f"Stock: {ticker}")
        st.plotly_chart(fig)

        # Display signals and sentiment
        latest_price = df['Close'].iloc[-1]
        if df["Buy_Signal"].iloc[-1]:
            st.write(f"🔹 **Buy Signal** at ${latest_price:.2f}")
        elif df["Sell_Signal"].iloc[-1]:
            st.write(f"🔻 **Sell Signal** at ${latest_price:.2f}")
        else:
            st.write("⚪ **No Strong Signal**")

        st.write(f"**Market Sentiment:** {sentiment_signal}")
        st.text(sentiment)

        time.sleep(1)
