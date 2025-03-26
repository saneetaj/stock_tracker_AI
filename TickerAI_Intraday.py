import streamlit as st
import openai
import pandas as pd
import numpy as np
import datetime
import time
import plotly.graph_objects as go
import requests
import logging
import asyncio
from typing import Optional, List
import pytz
from streamlit_autorefresh import st_autorefresh

# Alpaca Imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca_trade_api.rest import REST

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys from Streamlit secrets
try:
    openai_api_key = st.secrets["openai_api_key"]
    alpaca_api_key = st.secrets["alpaca_api_key"]
    alpaca_secret_key = st.secrets["alpaca_secret_key"]
    finnhub_api_key = st.secrets["finnhub_api_key"]
except KeyError as e:
    st.error(f"Missing API key in Streamlit secrets: {e}")
    logging.error(f"Missing API key in Streamlit secrets: {e}")
    st.stop()

# Initialize OpenAI client
try:
    openai_client = openai.OpenAI(api_key=openai_api_key)
except openai.OpenAIError as e:
    st.error(f"Error initializing OpenAI client: {e}")
    logging.error(f"Error initializing OpenAI client: {e}")
    st.stop()

# Initialize Alpaca data and trade clients
historical_client = None
live_stream = None
try:
    historical_client = StockHistoricalDataClient(
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
    )
    live_stream = StockDataStream(
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
    )
except Exception as e:
    st.error(f"Error initializing Alpaca data client: {e}")
    logging.error(f"Error initializing Alpaca data client: {e}")
    # Allow historical data even if live stream fails.

try:
    trade_client = REST(key_id=alpaca_api_key, secret_key=alpaca_secret_key)
except Exception as e:
    st.error(f"Error initializing Alpaca trade client: {e}")
    logging.error(f"Error initializing Alpaca trade client: {e}")
    st.stop()

# Initialize Finnhub client
try:
    import finnhub
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
except Exception as e:
    st.error(f"Error initializing Finnhub client: {e}")
    logging.error(f"Error initializing Finnhub client: {e}")
    st.stop()


# -------------------------
# Indicator Calculation Functions

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_intraday_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates intraday-focused indicators:
      - SMA_20 (20-bar simple moving average)
      - RSI (7-period)
      - Bollinger Bands (20-period)
    """
    if "Date" not in data.columns:
        st.error("Data is missing the 'Date' column.")
        return pd.DataFrame()
    # Ensure Date is datetime and convert to US/Eastern
    data["Date"] = pd.to_datetime(data["Date"])
    # Assume raw data is in UTC; localize then convert:
    if data["Date"].dt.tz is None:
        data["Date"] = data["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    else:
        data["Date"] = data["Date"].dt.tz_convert("US/Eastern")
    data.sort_values("Date", inplace=True)

    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["RSI"] = compute_rsi(data["Close"], period=7)
    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    data["BB_Std"] = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + 2 * data["BB_Std"]
    data["BB_Lower"] = data["BB_Middle"] - 2 * data["BB_Std"]

    return data

# -------------------------
# Intraday Signal Generation

def generate_intraday_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates simple intraday buy/sell signals:
      - Buy if RSI < 30 OR Close < Lower Bollinger Band
      - Sell if RSI > 70 OR Close > Upper Bollinger Band
    """
    data = data.copy()
    data["Buy_Signal"] = (data["RSI"] < 30) | (data["Close"] < data["BB_Lower"])
    data["Sell_Signal"] = (data["RSI"] > 70) | (data["Close"] > data["BB_Upper"])
    return data

# -------------------------
# Intraday Backtesting

def backtest_intraday_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Backtests the intraday strategy:
      - Enter long on next bar after a Buy signal.
      - Exit on next bar after a Sell signal.
    Calculates cumulative returns versus a buy-and-hold baseline.
    """
    df = data.copy()
    if "Date" not in df.columns:
        st.error("Data is missing the 'Date' column.")
        return pd.DataFrame()

    df = calculate_intraday_indicators(df)
    df.sort_values("Date", inplace=True)
    df = df.reset_index(drop=True)
    df = generate_intraday_signals(df)
    df.set_index("Date", inplace=True)

    # Create a Signal column (1 for buy, 0 for sell)
    df["Signal"] = 0
    df.loc[df["Buy_Signal"], "Signal"] = 1
    df.loc[df["Sell_Signal"], "Signal"] = 0

    # Forward-fill position (simulate entering next bar)
    df["Position"] = df["Signal"].replace(to_replace=0, method='ffill').shift(1).fillna(0)

    df["Market_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Position"]
    df["Cum_Market_Return"] = (1 + df["Market_Return"]).cumprod()
    df["Cum_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()

    return df

# -------------------------
# Data Retrieval for Intraday

def get_intraday_stock_data(ticker: str, days: int = 1) -> Optional[pd.DataFrame]:
    """
    Fetches intraday 1-minute bars for the past 'days' from Alpaca.
    After fetching, converts timestamps from UTC to US/Eastern and filters to regular market hours.
    """
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            start=start_date,
            end=end_date,
            timeframe=TimeFrame.Minute,  # 1-minute bars
            feed="iex"
        )
        if historical_client is not None:
            bars = historical_client.get_stock_bars(request_params)
        else:
            return None

        if bars:
            bars_list = bars[ticker]
            df = pd.DataFrame([
                {
                    'Date': bar.timestamp,
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume,
                }
                for bar in bars_list
            ])
            # Convert Date column to datetime and then to US/Eastern
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is None:
                df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            else:
                df["Date"] = df["Date"].dt.tz_convert("US/Eastern")
            # Filter to regular market hours: 9:30 AM to 4:00 PM ET
            start_time = datetime.time(9, 30)
            end_time = datetime.time(16, 0)
            df = df[(df["Date"].dt.time >= start_time) & (df["Date"].dt.time <= end_time)]
            return df
        else:
            st.error(f"⚠️ No intraday data found for {ticker}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
        logging.error(f"Error fetching intraday data for {ticker}: {e}")
        return None

# -------------------------
# News and Sentiment Functions (optional)

def get_stock_news(ticker: str) -> str:
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-03-01&to=2025-03-24&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200 and data:
        news_articles = []
        for article in data[:3]:
            title = article['headline']
            article_url = article['url']
            news_articles.append(f"• {title}: {article_url}")
        return "\n".join(news_articles)
    else:
        return f"⚠️ No news available for {ticker}."

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_sentiment(tickers: List[str]) -> dict:
    sentiments = {}
    rate_limit_error_flag = False
    for ticker in tickers:
        news_data = get_stock_news(ticker)
        attempt = 1
        while attempt <= 5:
            try:
                prompt = f"Analyze the market sentiment for {ticker} using the news below:\n{news_data}\nProvide a brief summary (bullish, bearish, or neutral) with key reasons. Limit to 250 words."
                response = openai_client.chat.completions.create(
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

# -------------------------
# Streamlit UI

async def main():
    st.title("📈 Intraday Ticker AI")
    tickers_input = st.text_input("Enter stock ticker symbol(s), separated by commas", "AAPL, MSFT", key="tickers_input")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        
    if st.button("🔍 Analyze"):
        sentiments = get_market_sentiment(tickers)

        for ticker in tickers:
            if ticker in sentiments:
                st.sidebar.subheader(f"📢 Sentiment for {ticker}")
                st.sidebar.write(sentiments[ticker])
            st.subheader(f"📊 Intraday Stock Data for {ticker}")
            intraday_data = get_intraday_stock_data(ticker, days=1)
            if intraday_data is None or intraday_data.empty:
                st.write(f"⚠️ No intraday data available for {ticker}")
                continue

            processed_data = calculate_intraday_indicators(intraday_data)
            processed_data = generate_intraday_signals(processed_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=processed_data['Date'],
                y=processed_data['Close'],
                mode='lines',
                name='Close Price'
            ))
            buy_signals = processed_data[processed_data["Buy_Signal"] == True]
            sell_signals = processed_data[processed_data["Sell_Signal"] == True]
            fig.add_trace(go.Scatter(
                x=buy_signals['Date'],
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=10),
                name='Buy Signal'
            ))
            fig.add_trace(go.Scatter(
                x=sell_signals['Date'],
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=10),
                name='Sell Signal'
            ))
            fig.update_layout(
                title=f"{ticker} Intraday Chart (1-minute) with Buy/Sell Signals (US/Eastern)",
                xaxis_title="Time (US/Eastern)",
                yaxis_title="Price",
                legend_title="Signals"
            )
            st.plotly_chart(fig, use_container_width=True)

            # -------------------------
            # Intraday Backtest
            st.subheader(f"📈 Intraday Backtest for {ticker}")
            bt_results = backtest_intraday_strategy(intraday_data)
            if not bt_results.empty:
                bt_fig = go.Figure()
                bt_fig.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results['Cum_Market_Return'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Buy & Hold: %{y:.2f}<extra></extra>'
                ))
                bt_fig.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results['Cum_Strategy_Return'],
                    mode='lines',
                    name='Intraday Strategy',
                    line=dict(color='orange', width=2),
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Strategy: %{y:.2f}<extra></extra>'
                ))
                bt_fig.update_layout(
                    title=f"Intraday Strategy vs. Buy & Hold: {ticker}",
                    xaxis_title="Time (US/Eastern)",
                    yaxis_title="Cumulative Return",
                    legend_title="Strategy",
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(bt_fig, use_container_width=True)
            else:
                st.write("⚠️ No backtest data available.")

if __name__ == "__main__":
    asyncio.run(main())
    # Refresh the stock quotes every 5 minutes (300000 ms) without refreshing the cached news.
    st_autorefresh(interval=120000, limit=0, key="intraday_autorefresh")
