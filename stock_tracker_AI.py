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

# Function to fetch historical stock data from Finnhub
def get_historical_stock_data(ticker, days=50):
    now = datetime.datetime.now()
    end_time = int(now.timestamp())
    start_time = int((now - datetime.timedelta(days=days)).timestamp())

    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={start_time}&to={end_time}&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and 'c' in data:
        df = pd.DataFrame({
            'Date': [datetime.datetime.fromtimestamp(t) for t in data['t']],
            'Close': data['c'],
            'High': data['h'],
            'Low': data['l'],
        })
        return df
    else:
        st.write(f"⚠️ Error fetching historical data for {ticker}. Status code: {response.status_code}. Response: {data}")
        return None

# Function to get intraday stock data (last available data when markets are closed)
def get_intraday_data(ticker):
    now = datetime.datetime.now()
    start_time = int((now - datetime.timedelta(days=1)).timestamp())
    end_time = int(now.timestamp())

    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=1&from={start_time}&to={end_time}&token={finnhub_api_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and 'c' in data:
        if not data['c']:
            st.write(f"No intraday data available for {ticker}. Market might be closed. Using historical data.")
            return get_historical_stock_data(ticker) #get historical data
        else:
            df = pd.DataFrame({
                'Date': [datetime.datetime.fromtimestamp(t) for t in data['t']],
                'Close': data['c'],
                'High': data['h'],
                'Low': data['l'],
            })
            return df
    elif 'error' in data:
        st.write(f"⚠️ Error fetching intraday data for {ticker}: {data['error']}\nMarkets might be closed and/or paid version of Finnhub required. Using historical data.")
        return get_historical_stock_data(ticker) #get historical data
    else:
        st.write(f"Error fetching data for {ticker}. Response: {data}")
        return None

# Function to calculate technical indicators
def calculate_indicators(data):
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
def generate_signals(data):
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

    data["Buy_Signal_Stochastic"] = (data["Stochastic_K"] < 20) & (data["Stochastic_K"] > data["Stochastic_D"])  # Buy if Stochastic K is below 20 and above D
    data["Sell_Signal_Stochastic"] = (data["Stochastic_K"] > 80) & (data["Stochastic_K"] < data["Stochastic_D"])  # Sell if Stochastic K is above 80 and below D

    return data

# Function to combine all buy/sell signals
def combine_signals(data):
    # Combining all the buy signals into one: All conditions must be true for a Buy
    data["Buy_Signal_Combined"] = (
        data["Buy_Signal"] &
        data["Buy_Signal_EMA"] &
        data["Buy_Signal_MACD"] &
        data["Buy_Signal_BB"] &
        data["Buy_Signal_Stochastic"]
    )

    # Combining all the sell signals into one: Any of the conditions being true will trigger a Sell
    data["Sell_Signal_Combined"] = (
        data["Sell_Signal"] |
        data["Sell_Signal_EMA"] |
        data["Sell_Signal_MACD"] |
        data["Sell_Signal_BB"] |
        data["Sell_Signal_Stochastic"]
    )

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
        data = get_intraday_data(ticker)
        if data is None:
            st.write(f"⚠️ No data available for {ticker}")
            continue

        df = calculate_indicators(data) # use the dataframe returned from get_intraday_data
        if len(df) < 20: #check dataframe length.
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
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals["Close"], mode="markers", name="Buy Signal", marker=dict(color="green", size=10)))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals["Close"], mode="markers", name="Sell Signal", marker=dict(color="red", size=10)))

        st.plotly_chart(fig)

    # Auto-refresh logic
    st.success("✅ Stock data updates every 5 minutes!")
    time.sleep(300)  # Refresh every 5 minutes
