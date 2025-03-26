import streamlit as st
import openai
import pandas as pd
import datetime
import time
import plotly.graph_objects as go
import requests
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.news import NewsDataClient
from typing import Optional, List

# Load API keys from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
alpaca_api_key = st.secrets["alpaca_api_key"]
alpaca_secret_key = st.secrets["alpaca_secret_key"]

#Alpaca Endpoint: https://api.alpaca.markets

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Initialize Alpaca data client
historical_client = StockHistoricalDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
live_client = StockDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
news_client = NewsDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)


# Function to fetch historical stock data from Alpaca
def get_historical_stock_data(ticker: str, days: int = 50) -> Optional[pd.DataFrame]:
    """
    Fetches historical stock data from Alpaca.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        days (int): The number of days of historical data to retrieve.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the historical stock data,
                          or None if an error occurs.
    """
    try:
        # Calculate the start date
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        # Create the request object
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            start=start_date,
            end=end_date,
            timeframe=TimeFrame.Day,  # Use daily timeframe
        )

        # Get the bars
        bars = historical_client.get_stock_bars(request_params)

        # Convert to DataFrame
        if bars:
            bars_list = list(bars.values())[0]  # Get bars for the first symbol
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
            st.error(f"⚠️ Error: No historical data found for {ticker} from Alpaca.")
            return None

    except Exception as e:
        st.error(f"⚠️ Error fetching historical data for {ticker}: {e}")
        return None



# Function to get intraday stock data from Alpaca
def get_intraday_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetches intraday stock data from Alpaca.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the intraday stock data,
                          or None if an error occurs.
    """
    try:
        # Calculate the start and end dates.  Alpaca uses bars for intraday.
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=1)

        # Create the request object
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            start=start_date,
            end=end_date,
            timeframe=TimeFrame.Minute60,  # 60-minute interval
        )

        # Get the bars
        bars = live_client.get_stock_bars(request_params)

        # Convert to DataFrame
        if bars:
            bars_list = list(bars.values())[0]
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
            return df.sort_values(by='Date')
        else:
            st.error(f"⚠️ Error: No intraday data found for {ticker} from Alpaca.")
            return None

    except Exception as e:
        st.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
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
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(window=14).mean()))
    # Exponential Moving Average (EMA)
    # Shorter period EMA crossing above longer period EMA can be a Buy signal (bullish trend)
    # EMA crossing below can be a Sell signal (bearish trend)
    data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

    # MACD (12-26-9)
    # MACD crossing above the signal line can be a Buy signal (bullish crossover)
    # MACD crossing below the signal line can be a Sell signal (bearish crossover)
    data["MACD"] = data["EMA_9"] - data["EMA_50"]
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



# Function to generate buy/sell signals
def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates buy/sell signals based on technical indicators.

    Args:
        data (pd.DataFrame): A DataFrame containing stock data with technical indicators.

    Returns:
        pd.DataFrame: The DataFrame with added buy/sell signal columns.
    """
    # Basic Buy/Sell Signals based on SMA and RSI
    data["Buy_Signal"] = (data["Close"] > data["SMA_20"]) & (data["RSI"] < 30)  # Buy if price above SMA and RSI is oversold
    data["Sell_Signal"] = (data["Close"] < data["SMA_20"]) & (data["RSI"] > 70)  # Sell if price below SMA and RSI is overbought

    # Additional signals based on EMA, MACD, Bollinger Bands, and Stochastic Oscillator
    data["Buy_Signal_EMA"] = data["EMA_9"] > data["EMA_50"]  # Buy if short-term EMA is above long-term EMA
    data["Sell_Signal_EMA"] = data["EMA_9"] < data["EMA_50"]  # Sell if short-term EMA is below long-term EMA

    data["Buy_Signal_MACD"] = data["MACD"] > data["MACD_Signal"]  # Buy if MACD is above the signal line
    data["Sell_Signal_MACD"] = data["MACD"] < data["MACD_Signal"]  # Sell if MACD is below the signal line

    data["Buy_Signal_BB"] = data["Close"] < data["Bollinger_Lower"]  # Buy if price is below the lower Bollinger Band
    data["Sell_Signal_BB"] = data["Close"] > data["Bollinger_Upper"]  # Sell if price is above the upper Bollinger Band

    data["Buy_Signal_Stochastic"] = (data["Stochastic_K"] < 20) & (
        data["Stochastic_K"] > data["Stochastic_D"])  # Buy if Stochastic K is below 20 and above D
    data["Sell_Signal_Stochastic"] = (data["Stochastic_K"] > 80) & (
        data["Stochastic_K"] < data["Stochastic_D"])  # Sell if Stochastic K is above 80 and below D

    return data



# Function to combine all buy/sell signals
def combine_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all buy/sell signals into a single buy/sell decision.

    Args:
        data (pd.DataFrame): A DataFrame containing individual buy/sell signals.

    Returns:
        pd.DataFrame: The DataFrame with combined buy/sell signal columns.
    """
    # Combining all the buy signals into one: All conditions must be true for a Buy
    data["Buy_Signal_Combined"] = (
        data["Buy_Signal"]
        & data["Buy_Signal_EMA"]
        & data["Buy_Signal_MACD"]
        & data["Buy_Signal_BB"]
        & data["Buy_Signal_Stochastic"]
    )

    # Combining all the sell signals into one: Any of the conditions being true will trigger a Sell
    data["Sell_Signal_Combined"] = (
        data["Sell_Signal"]
        | data["Sell_Signal_EMA"]
        | data["Sell_Signal_MACD"]
        | data["Sell_Signal_BB"]
        | data["Sell_Signal_Stochastic"]
    )

    return data



def get_stock_news(ticker: str) -> str:
    """
    Fetches stock news from Alpaca.

    Args:
        ticker (str): The ticker symbol for the company.

    Returns:
        str: A formatted string containing news headlines and URLs, or a message
             indicating no news is available.
    """
    try:
        news = news_client.get_news(symbol=ticker)
        if news:
            news_articles = []
            for article in news[:3]:  # Limit to top 3 articles
                title = article.headline
                url = article.url
                news_articles.append(f"• {title}: {url}")
            return "\n".join(news_articles)
        else:
            return f"⚠️ No news available for {ticker}."
    except Exception as e:
        return f"⚠️ Error fetching news for {ticker}: {e}"



# Function to fetch market sentiment using OpenAI
def get_market_sentiment(tickers: List[str]) -> dict:
    """
    Fetches market sentiment for a list of tickers using OpenAI's GPT-4.

    Args:
        tickers (List[str]): A list of stock ticker symbols.

    Returns:
        dict: A dictionary where keys are tickers and values are sentiment summaries,
              or an error message if the OpenAI API request fails.
    """
    sentiments = {}
    rate_limit_error_flag = False

    for ticker in tickers:
        news_data = get_stock_news(ticker)
        attempt = 1

        while attempt <= 5:
            try:
                prompt = f"Analyze the market sentiment for {ticker} in the below news. :\n{news_data}\nProvide a brief summary (bullish, bearish, or neutral) with key reasons. Strictly limit the summary to 250 words max."

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial news analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400
                )
                sentiments[ticker] = response.choices[0].message.content.strip()
                break  # Exit the retry loop if successful

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
                    break  # Exit the retry loop after max attempts

        time.sleep(2)  # Add a delay to stay within API rate limits

    return sentiments



# Streamlit UI
def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("📈 Ticker AI")

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
            intraday_data = get_intraday_data(ticker)
            historical_data = get_historical_stock_data(ticker)

            # Determine which data to use (intraday if available, otherwise historical)
            data_to_use = intraday_data if intraday_data is not None else historical_data

            if data_to_use is None:
                st.write(f"⚠️ No data available for {ticker}")
                continue  # Move to the next ticker

            df = calculate_indicators(data_to_use)  # use the dataframe
            if len(df) < 20:  # check dataframe length.
                st.write(f"⚠️ Not enough data points to calculate indicators for {ticker}")
                continue
            df = generate_signals(df)  # Generate buy/sell signals
            df = combine_signals(df)  # Combine all signals

            # Plot stock price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df["Close"], mode="lines", name="Close Price"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df["SMA_20"], mode="lines", name="20-Day SMA"))

            # Highlight Buy/Sell Signals
            buy_signals = df[df["Buy_Signal_Combined"]]
            sell_signals = df[df["Sell_Signal_Combined"]]
            fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals["Close"], mode="markers",
                                     name="Buy Signal", marker=dict(color="green", size=10)))
            fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals["Close"], mode="markers",
                                     name="Sell Signal", marker=dict(color="red", size=10)))

            st.plotly_chart(fig)

        # Auto-refresh logic
        st.success("✅ Stock data updates every 5 minutes!")
        time.sleep(300)  # Refresh every 5 minutes



if __name__ == "__main__":
    main()
