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
    historical_client = StockHistoricalDataClient(  # Removed feed here
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
    )
    live_stream = StockDataStream(
        api_key=alpaca_api_key,
        secret_key=alpaca_secret_key,
    )
    #st.write(f"live_stream after init: {live_stream}")
except Exception as e:
    st.error(f"Error initializing Alpaca data client: {e}")
    logging.error(f"Error initializing Alpaca data client: {e}")
    # Don't stop here, allow historical data to be fetched if live stream fails.

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
    st.error(f"Error initializing Finnhub trade client: {e}")
    logging.error(f"Error initializing Finnhub trade client: {e}")
    st.stop()


# Function to fetch historical stock data from Alpaca
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

        #st.write(f"bars type: {type(bars)}")  # Print type of bars
        #st.write(f"bars: {bars}")

        if bars:
            #  bars_list = list(bars.values())[0] # Removed .values()
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
        logging.error(f"⚠️ Error fetching historical data for {ticker}: {e}")
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
            #st.write(f"live_stream before subscribe: {live_stream}")
            # await live_stream.subscribe_bars(stock_data_handler, ticker, data_feed='iex') # Removed data_feed
            if live_stream is not None: # Check if live_stream is still valid
                await live_stream.subscribe_bars(stock_data_handler, ticker)
                await asyncio.sleep(10)
                await live_stream.unsubscribe_bars(stock_data_handler, ticker)
                await live_stream.close()
            else:
                return None
        except Exception as stream_error:
            #st.error(f"⚠️ Error with live stream: {stream_error}")
            logging.error(f"⚠️ Error with live stream: {stream_error}")
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
            logging.error(f"No intraday data found for {ticker}")
            return None

    except Exception as e:
        st.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
        logging.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
        return None


# Function to calculate technical indicators
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the given stock data.

    Args:
        data (pd.DataFrame): A DataFrame containing stock data with 'Close' prices.

    Returns:
        pd.DataFrame: The DataFrame with added technical indicators.
    """
    try:
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(window=21).mean()))
        # Exponential Moving Average (EMA)
        # Shorter period EMA crossing above longer period EMA can be a Buy signal (bullish trend)
        # EMA crossing below can be a Sell signal (bearish trend)
        data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

        # MACD (12-26-9)
        # MACD crossing below the signal line can be a Sell signal (bearish crossover)
        data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()  # 12-day EMA
        data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()  # 26-day EMA
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands (20-period)
        # Price touching or going below the lower Bollinger Band suggests an oversold condition (Buy signal)
        # Price touching or going above the upper Bollinger Band suggests an overbought condition (Sell signal)
        data["Bollinger_Middle"] = data["Close"].rolling(window=20).mean()
        data["Bollinger_Upper"] = data["Bollinger_Middle"] + 2 * data["Close"].rolling(window=20).std()
        data["Bollinger_Lower"] = data["Bollinger_Middle"] - 2 * data["Close"].rolling(window=20).std()

        # Stochastic Oscillator
        # Stochastic K value below 20 suggests oversold (potential Buy signal)
        # Stochastic K value above 80 suggests overbought (potential Sell signal)
        high_14 = data["High"].rolling(window=14).max()
        low_14 = data["Low"].rolling(window=14).min()
        data["Stochastic_K"] = (data["Close"] - low_14) / (high_14 - low_14) * 100
        data["Stochastic_D"] = data["Stochastic_K"].rolling(window=3).mean()

        # ATR (Average True Range)
        # ATR measures market volatility, but is not directly used in buy/sell signals in this case
        # It's often used to adjust stop-loss levels based on the volatility of the stock
        data["High-Low"] = data["High"] - data["Low"]
        data["High-Prev Close"] = abs(data["High"] - data["Close"].shift(1))
        data["Low-Prev Close"] = abs(data["Low"] - data["Close"].shift(1))
        data["TR"] = data[["High-Low", "High-Prev Close", "Low-Prev Close"]].max(axis=1)
        data["ATR"] = data["TR"].rolling(window=14).mean()
        return data
    except Exception as e:
        st.error(f"Error in calculate_indicators: {e}")
        logging.error(f"Error in calculate_indicators: {e}")
        return pd.DataFrame()  # Return empty dataframe


# Function to generate buy/sell signals
def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates buy/sell signals based on technical indicators.

    Args:
        data (pd.DataFrame): A DataFrame containing stock data with technical indicators.

    Returns:
        pd.DataFrame: The DataFrame with added buy/sell signal columns.
    """
    try:
        # Basic Buy/Sell Signals based on SMA and RSI
        data["Buy_Signal"] = (
            (data["Close"] > data["SMA_50"]) &  # Price above SMA
            (data["RSI"] < 40) &  # RSI is not too high (less than 40 instead of 30)
            (data["MACD"] > data["MACD_Signal"]) # MACD above signal
        )
        data["Sell_Signal"] = (
            (data["Close"] < data["SMA_50"]) &  # Price below SMA
            (data["RSI"] > 60) & # RSI is not too low (greater than 60 instead of 70)
            (data["MACD"] < data["MACD_Signal"]) # MACD below signal
        )

        # Additional signals based on EMA, MACD, Bollinger Bands, and Stochastic Oscillator
        data["Buy_Signal_EMA"] = data["EMA_9"] > data["EMA_50"]  # Buy if short-term EMA is above long-term EMA
        data["Sell_Signal_EMA"] = data["EMA_9"] < data["EMA_50"]  # Sell if short-term EMA is below long-term EMA

        data["Buy_Signal_MACD"] = data["MACD"] > data["MACD_Signal"]  # Buy if MACD is above the signal line
        data["Sell_Signal_MACD"] = data["MACD"] < data["MACD_Signal"]  # Sell if MACD is below the signal line

        data["Buy_Signal_BB"] = data["Close"] < data[
            "Bollinger_Lower"
        ]  # Buy if price is below the lower Bollinger Band
        data["Sell_Signal_BB"] = data["Close"] > data[
            "Bollinger_Upper"
        ]  # Sell if price is above the upper Bollinger Band

        data["Buy_Signal_Stochastic"] = (data["Stochastic_K"] < 20) & (
            data["Stochastic_K"] > data["Stochastic_D"]
        )  # Buy if Stochastic K is below 20 and above D
        data["Sell_Signal_Stochastic"] = (data["Stochastic_K"] > 80) & (
            data["Stochastic_K"] < data["Stochastic_D"]
        )  # Sell if Stochastic K is above 80 and below D
        return data
    except Exception as e:
        st.error(f"Error in generate_signals: {e}")
        logging.error(f"Error in generate_signals: {e}")
        return pd.DataFrame()  # return empty dataframe


# Function to combine all buy/sell signals
def combine_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all buy/sell signals into a single buy/sell decision.

    Args:
        data (pd.DataFrame): A DataFrame containing individual buy/sell signals.

    Returns:
        pd.DataFrame: The DataFrame with combined buy/sell signal columns.
    """
    try:
        # Initialize combined signals to False
        data["Buy_Signal_Combined"] = False
        data["Sell_Signal_Combined"] = False

        # Define weights for each signal
        weights = {
            "SMA_RSI": 0.5,  # Increased weight
            "EMA": 0.1,
            "MACD": 0.3, # Increased weight
            "BB": 0.05, # decreased weight
            "Stochastic": 0.05, # decreased weight
        }

        # Calculate a combined score for buy and sell signals
        data["Buy_Score"] = (
            (data["Buy_Signal"] & (data["RSI"] < 30)).astype(int) * weights["SMA_RSI"] + # Combining SMA and RSI for Buy
            data["Buy_Signal_EMA"].astype(int) * weights["EMA"] +
            data["Buy_Signal_MACD"].astype(int) * weights["MACD"] +
            data["Buy_Signal_BB"].astype(int) * weights["BB"] +
            data["Buy_Signal_Stochastic"].astype(int) * weights["Stochastic"]
        )
        data["Sell_Score"] = (
            (data["Sell_Signal"] & (data["RSI"] > 70)).astype(int) * weights["SMA_RSI"] + # Combining SMA and RSI for Sell
            data["Sell_Signal_EMA"].astype(int) * weights["EMA"] +
            data["Sell_Signal_MACD"].astype(int) * weights["MACD"] +
            data["Sell_Signal_BB"].astype(int) * weights["BB"] +
            data["Sell_Signal_Stochastic"].astype(int) * weights["Stochastic"]
        )

        # Generate combined buy/sell signals based on the scores
        data["Buy_Signal_Combined"] = data["Buy_Score"] >= 0.7  # Buy if the combined score is high
        data["Sell_Signal_Combined"] = data["Sell_Score"] >= 0.45 # Sell if combined score is high

        return data
    except Exception as e:
        st.error(f"Error in combine_signals: {e}")
        logging.error(f"Error in combine_signals: {e}")
        return pd.DataFrame()  # return empty dataframe

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
        attempt = 1

        while attempt <= 5:
            try:
                prompt = f"Analyze the market sentiment for {ticker} in the below news. :\n{news_data}\nProvide a brief summary (bullish, bearish, or neutral) with key reasons. Strictly limit the summary to 250 words max."

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


# Streamlit UI
async def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("📈 Ticker AI")

    tickers_input = st.text_input("Enter stock ticker symbol(s), separated by commas", "AAPL, MSFT, GOOG", key="tickers_input") # Added key
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    if st.button("🔍 Analyze"):
        sentiments = get_market_sentiment(tickers)

        for ticker in tickers:
            if ticker in sentiments:
                st.sidebar.subheader(f"📢 Sentiment for {ticker}")
                st.sidebar.write(sentiments[ticker])

            st.subheader(f"📊 Stock Data for {ticker}")

            # Fetch stock data
            intraday_data = await get_intraday_data(ticker)
            historical_data = get_historical_stock_data(ticker)

            data_to_use = intraday_data if intraday_data is not None else historical_data

            if data_to_use is None:
                st.write(f"⚠️ No data available for {ticker}")
                continue

            processed_data = calculate_indicators(data_to_use)
            processed_data = generate_signals(processed_data)
            processed_data = combine_signals(processed_data)

            # Create the line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Close'],
                                     mode='lines',
                                     name='Close Price',showlegend=True)) # Added name for close price

            # Add buy/sell signals to the chart
            buy_signals = processed_data[processed_data['Buy_Signal_Combined'] == True]
            sell_signals = processed_data[processed_data['Sell_Signal_Combined'] == True]

            #Plotting Buy Signals
            fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'],
                                     mode='markers',
                                     marker=dict(color='green', symbol='triangle-up', size=12),
                                     name='Buy Signals',showlegend=True)) # Added name

            #Plotting Sell Signals
            fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'],
                                     mode='markers',
                                     marker=dict(color='red', symbol='triangle-down', size=12),
                                     name='Sell Signals',showlegend=True)) # Added name

            # Update layout for better visualization
            fig.update_layout(title=f"{ticker} Stock Chart with Buy/Sell Signals",
                              xaxis_title="Date",
                              yaxis_title="Price",
                              legend_title="Signals")

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # Display the most recent data
            st.dataframe(processed_data.tail())

            st.success("✅ Stock data and chart updates every 5 minutes!")



if __name__ == "__main__":
    asyncio.run(main())
