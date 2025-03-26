import streamlit as st
import openai
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import requests
import logging
import subprocess
import sys
import finnhub
import asyncio
from typing import Optional, List
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

# Initialize Alpaca data client
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
    # Allow historical data to be fetched even if live stream fails.

# Initialize Alpaca trade client (for trading actions)
try:
    trade_client = REST(key_id=alpaca_api_key, secret_key=alpaca_secret_key)
except Exception as e:
    st.error(f"Error initializing Alpaca trade client: {e}")
    logging.error(f"Error initializing Alpaca trade client: {e}")
    st.stop()

# Initialize Finnhub client
try:
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
except Exception as e:
    st.error(f"Error initializing Finnhub client: {e}")
    logging.error(f"Error initializing Finnhub client: {e}")
    st.stop()

# -------------------------
# Utility Functions

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using average gain and loss.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------------
# Data Retrieval Functions

def get_historical_stock_data(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            start=start_date,
            end=end_date,
            timeframe=TimeFrame.Day,
            feed="iex"
        )
        if historical_client is not None:
            bars = historical_client.get_stock_bars(request_params)
        else:
            return None

        if bars:
            bars_list = bars[ticker]  # Access bars for the specific ticker
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
            return df
        else:
            st.error(f"⚠️ No historical data found for {ticker}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching historical data for {ticker}: {e}")
        logging.error(f"Error fetching historical data for {ticker}: {e}")
        return None

async def get_intraday_data(ticker: str) -> Optional[pd.DataFrame]:
    try:
        data_list: List[dict] = []

        async def stock_data_handler(data: dict):
            data_list.append(data)

        if live_stream is None:
            st.error("⚠️ Live stream is not initialized. Intraday data is unavailable.")
            logging.error("Live stream is not initialized. Intraday data is unavailable.")
            return None

        try:
            if live_stream is not None:
                await live_stream.subscribe_bars(stock_data_handler, ticker)
                await asyncio.sleep(10)
                await live_stream.unsubscribe_bars(stock_data_handler, ticker)
                await live_stream.close()
            else:
                return None
        except Exception as stream_error:
            logging.error(f"Error with live stream: {stream_error}")
            return None

        if data_list:
            df = pd.DataFrame([
                {
                    'Date': item['timestamp'],
                    'Open': item['open'],
                    'High': item['high'],
                    'Low': item['low'],
                    'Close': item['close'],
                    'Volume': item['volume'],
                }
                for item in data_list
            ])
            return df
        else:
            st.error(f"⚠️ No intraday data found for {ticker}")
            logging.error("No intraday data found for {ticker}")
            return None

    except Exception as e:
        st.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
        logging.error(f"Error fetching intraday data for {ticker}: {e}")
        return None

# -------------------------
def compute_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes the Average Directional Index (ADX) to gauge trend strength.
    """
    # Calculate True Range (TR)
    high_low = data["High"] - data["Low"]
    high_prev_close = (data["High"] - data["Close"].shift()).abs()
    low_prev_close = (data["Low"] - data["Close"].shift()).abs()
    tr = high_low.combine(high_prev_close, max).combine(low_prev_close, max)
    
    # Calculate directional movements
    up_move = data["High"] - data["High"].shift()
    down_move = data["Low"].shift() - data["Low"]
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smooth the TR, +DM, and -DM using a rolling sum (simple smoothing)
    tr_smooth = tr.rolling(window=period, min_periods=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).sum()
    
    # Calculate the directional indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Compute the Directional Index (DX)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    # ADX is the rolling average of DX
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx

def compute_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Computes the Commodity Channel Index (CCI).
    """
    # Typical Price
    tp = (data["High"] + data["Low"] + data["Close"]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(lambda x: pd.Series(x).mad())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various technical indicators:
      - 50-day and 200-day SMAs
      - RSI (14)
      - MACD (12,26) and its signal line (9)
      - Bollinger Bands (20-day)
      - Stochastic Oscillator (14,3)
      - ADX (14)
      - CCI (20)
    """
    try:
        data["Date"] = pd.to_datetime(data["Date"])
        data.sort_values("Date", inplace=True)
        # SMAs for trend
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        # RSI
        data["RSI"] = compute_rsi(data["Close"], period=14)
        # MACD
        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        # Bollinger Bands (20-day)
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        data["BB_Std"] = data["Close"].rolling(window=20).std()
        data["BB_Upper"] = data["BB_Middle"] + 2 * data["BB_Std"]
        data["BB_Lower"] = data["BB_Middle"] - 2 * data["BB_Std"]
        # Stochastic Oscillator
        data["Stoch_High"] = data["High"].rolling(window=14).max()
        data["Stoch_Low"] = data["Low"].rolling(window=14).min()
        data["Stochastic_K"] = 100 * ((data["Close"] - data["Stoch_Low"]) / (data["Stoch_High"] - data["Stoch_Low"]))
        data["Stochastic_D"] = data["Stochastic_K"].rolling(window=3).mean()
        # ADX
        data["ADX"] = compute_adx(data, period=14)
        # CCI
        data["CCI"] = compute_cci(data, period=20)
        return data
    except Exception as e:
        st.error(f"Error in calculate_indicators: {e}")
        logging.error(f"Error in calculate_indicators: {e}")
        return pd.DataFrame()

def generate_combined_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusted combined signals using additional indicators.
    
    BUY conditions (in a bullish trend):
      - Trend: SMA_50 > SMA_200 and ADX > 25 (indicates a strong trend)
      - Price pullback: Close is at least 2% below SMA_50 (i.e. Close < 0.98 * SMA_50)
      - RSI: Below 50 (suggesting a pullback)
      - MACD: MACD > MACD_Signal (momentum confirmation)
      - CCI: Below -100 (indicating oversold conditions)
      
    SELL conditions:
      - Either trend reversal: SMA_50 < SMA_200 or ADX < 25
      - Or in a bullish trend, when price rallies above SMA_50 by at least 2% and RSI > 50.
    """
    data = data.copy()
    
    # Initialize signals
    data["Buy_Signal_Combined"] = False
    data["Sell_Signal_Combined"] = False
    
    # Bullish trend condition: SMA_50 > SMA_200 and ADX > 25
    bullish_trend = (data["SMA_50"] > data["SMA_200"]) & (data["ADX"] > 25)
    
    # Buy condition: In bullish trend, price pullback, low RSI, MACD confirmation, and oversold CCI.
    buy_condition = bullish_trend & \
                    (data["Close"] < 0.98 * data["SMA_50"]) & \
                    (data["RSI"] < 50) & \
                    (data["MACD"] > data["MACD_Signal"]) & \
                    (data["CCI"] < -100)
    
    # Sell condition: Either trend reversal or in bullish trend when price rallies above SMA_50 and RSI > 50.
    trend_reversal = (data["SMA_50"] < data["SMA_200"]) | (data["ADX"] < 25)
    rally_condition = bullish_trend & (data["Close"] > 1.02 * data["SMA_50"]) & (data["RSI"] > 50)
    sell_condition = trend_reversal | rally_condition
    
    data.loc[buy_condition, "Buy_Signal_Combined"] = True
    data.loc[sell_condition, "Sell_Signal_Combined"] = True
    
    return data

def backtest_combined_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Backtests the adjusted strategy:
      - Positions are taken on the next day's open (simulated via a shifted signal).
      - We assume that when a buy signal is triggered, the position is entered and held until a sell signal.
      - Cumulative returns for the strategy and a buy & hold benchmark are calculated.
    """
    df = data.copy().set_index("Date")
    # Generate signals on the data
    df = generate_combined_signals(df.reset_index()).set_index("Date")
    
    # Create a "Position" series:
    # We assume that a buy signal starts a long position and a sell signal ends it.
    df["Signal"] = 0
    df.loc[df["Buy_Signal_Combined"], "Signal"] = 1
    df.loc[df["Sell_Signal_Combined"], "Signal"] = 0
    # Use forward-fill to simulate holding the position until an exit signal.
    df["Position"] = df["Signal"].replace(to_replace=0, method='ffill').shift(1).fillna(0)
    
    # Compute returns
    df["Market_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Position"]
    df["Cum_Market_Return"] = (1 + df["Market_Return"]).cumprod()
    df["Cum_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()
    
    return df
# -------------------------
# News and Sentiment Functions

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

def get_market_sentiment(tickers):
    sentiments = {}
    rate_limit_error_flag = False
    for ticker in tickers:
        news_data = get_stock_news(ticker)
        attempt = 1
        while attempt <= 5:
            try:
                prompt = f"Analyze the market sentiment for {ticker} in the below news:\n{news_data}\nProvide a brief summary (bullish, bearish, or neutral) with key reasons. Limit the summary to 250 words max."
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
    st.title("📈 Ticker AI")
    tickers_input = st.text_input("Enter stock ticker symbol(s), separated by commas", "AAPL, MSFT, GOOG", key="tickers_input")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    if st.button("🔍 Analyze"):
        sentiments = get_market_sentiment(tickers)
        for ticker in tickers:
            if ticker in sentiments:
                st.sidebar.subheader(f"📢 Sentiment for {ticker}")
                st.sidebar.write(sentiments[ticker])
            st.subheader(f"📊 Stock Data for {ticker}")
            # Get data (prefer intraday if available; otherwise, use historical)
            intraday_data = await get_intraday_data(ticker)
            historical_data = get_historical_stock_data(ticker)
            data_to_use = intraday_data if intraday_data is not None else historical_data
            if data_to_use is None:
                st.write(f"⚠️ No data available for {ticker}")
                continue

            # Calculate indicators and generate combined signals
            processed_data = calculate_indicators(data_to_use)
            processed_data = generate_combined_signals(processed_data)

            # Plot the stock chart with combined signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Close'],
                                     mode='lines', name='Close Price', showlegend=True))
            # Mark buy signals where combined signal is True
            buy_signals = processed_data[processed_data["Buy_Signal_Combined"] == True]
            # Mark sell signals where combined sell signal is True (if implemented)
            sell_signals = processed_data[processed_data["Sell_Signal_Combined"] == True]
            fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'],
                                     mode='markers', marker=dict(color='green', symbol='triangle-up', size=12),
                                     name='Buy Signal', showlegend=True))
            fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'],
                                     mode='markers', marker=dict(color='red', symbol='triangle-down', size=12),
                                     name='Sell Signal', showlegend=True))
            fig.update_layout(title=f"{ticker} Stock Chart with Combined Signals",
                              xaxis_title="Date",
                              yaxis_title="Price",
                              legend_title="Signals")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(processed_data.tail())
            st.success("✅ Stock data and chart updates every 5 minutes!")
            
            # -------------------------
            # Backtesting the Combined Strategy on Historical Data
            st.subheader(f"📈 Backtest: Combined Indicator Strategy for {ticker}")
            backtest_data = calculate_indicators(historical_data)
            backtest_data = generate_combined_signals(backtest_data)
            bt_results = backtest_combined_strategy(backtest_data)
            if not bt_results.empty:
                bt_fig = go.Figure()
                bt_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Cum_Market_Return'],
                                            mode='lines', name='Buy & Hold', showlegend=True))
                bt_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Cum_Strategy_Return'],
                                            mode='lines', name='Combined Strategy', showlegend=True))
                bt_fig.update_layout(title=f"Cumulative Returns: {ticker}",
                                     xaxis_title="Date",
                                     yaxis_title="Cumulative Return",
                                     legend_title="Strategy")
                st.plotly_chart(bt_fig, use_container_width=True)
            else:
                st.write("⚠️ Backtest data not available.")

if __name__ == "__main__":
    asyncio.run(main())
