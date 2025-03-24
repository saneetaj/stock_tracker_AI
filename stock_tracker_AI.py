import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import time
import plotly.graph_objects as go

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]
openai.api_key = openai_api_key  # Initialize here, once.

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo", interval="1d")  # 1 month historical data
        return data
    except Exception as e:
        st.error(f"Failed to fetch stock data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error to avoid crashing

# Function to calculate technical indicators
def calculate_indicators(data):
    if not data.empty: # Check if the dataframe is empty
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(window=14).mean()))
    return data

# Function to generate buy/sell signals
def generate_signals(data):
    if not data.empty: # Check if the dataframe is empty
        data["Buy_Signal"] = (data["Close"] > data["SMA_20"]) & (data["RSI"] < 30)
        data["Sell_Signal"] = (data["Close"] < data["SMA_20"]) & (data["RSI"] > 70)
    return data

# Function to fetch market sentiment using OpenAI (optimized for rate limits)
def get_market_sentiment(tickers):
    sentiments = {}
    rate_limit_error = False

    for ticker in tickers:
        attempt = 0
        while attempt < 5:
            attempt += 1
            try:
                prompt = f"Analyze the market sentiment for {ticker}. Provide a short summary (bullish, bearish, or neutral) with key reasons."
                response = openai.Completion.create(
                    model="gpt-3.5-turbo",  # Correct model name.  Important!
                    prompt=prompt,
                    max_tokens=100,
                )
                # Check for 'choices' and extract text safely
                if response.choices and len(response.choices) > 0:
                    sentiments[ticker] = response.choices[0].text.strip()
                else:
                    sentiments[ticker] = "No sentiment data available."
                break  # Exit the loop on success
            except openai.error.RateLimitError as e:
                rate_limit_error = True
                wait_time = 2 ** attempt  # Exponential backoff (1, 2, 4, 8, 16 seconds)
                st.warning(f"Rate limit exceeded for {ticker}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except openai.error.APIError as e: # Catch other API errors
                st.error(f"OpenAI API Error for {ticker}: {e}")
                sentiments[ticker] = f"Error: {e}"
                break # Exit loop for this ticker
            except Exception as e:  # Catch all other exceptions
                st.error(f"Error processing {ticker}: {e}")
                sentiments[ticker] = f"Error: {e}"
                break  # Exit the loop on other errors

        if attempt >= 5:
            sentiments[ticker] = "Failed to get sentiment after multiple retries."

        time.sleep(2)  # Add a delay, even after successful calls, and outside the retry loop

    if rate_limit_error:
        st.warning("⚠️  OpenAI rate limit was encountered.  Consider reducing the number of tickers or running less frequently.")
    return sentiments
# Streamlit UI
st.title("📈 AI-Powered Stock Tracker")

# User input for multiple stock tickers
tickers = st.text_input("Enter stock ticker symbol", "AAPL, MSFT, GOOG").upper()
tickers = [ticker.strip() for ticker in tickers.split(",")]


# "Analyze" button to trigger stock tracking
if st.button("🔍 Analyze"):
    # Fetch and display sentiments
    sentiments = get_market_sentiment(tickers)

    # --- Display of Stock Data and Sentiment ---
    for ticker in tickers:
        st.subheader(f"📊 Stock Data for {ticker}")
        data = get_stock_data(ticker)

        if data.empty:
            st.warning(f"No stock data available for {ticker}.")
            continue  # Skip to the next ticker

        data = calculate_indicators(data)
        data = generate_signals(data)

        # Plot stock price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
        if "SMA_20" in data:
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="20-Day SMA"))

        # Highlight Buy/Sell Signals
        if "Buy_Signal" in data and "Sell_Signal" in data:
            buy_signals = data[data["Buy_Signal"]]
            sell_signals = data[data["Sell_Signal"]]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Close"], mode="markers", name="Buy Signal", marker=dict(color="green", size=10)))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["Close"], mode="markers", name="Sell Signal", marker=dict(color="red", size=10)))

        st.plotly_chart(fig)

        # Display Sentiment in Streamlit
        st.subheader(f"📢 Market Sentiment for {ticker}")
        if ticker in sentiments:
            st.write(sentiments[ticker])
        else:
            st.write("No sentiment data.")

    # Auto-refresh -  moved outside the button
    st.success("✅ Stock data updates every 5 minutes!")
    time.sleep(300)



'''
import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import datetime
import time
import plotly.graph_objects as go

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

# Initialize OpenAI client
openai.api_key = openai_api_key

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

# Function to fetch market sentiment using OpenAI (optimized for rate limits)
def get_market_sentiment(tickers):
    sentiments = {}
    rate_limit_error_flag = False  # Flag to track if rate limit error has occurred

    for ticker in tickers:
        attempt = 1
        while attempt <= 5:  # Retry up to 5 times
            try:
                prompt = f"Analyze the market sentiment for {ticker}. Provide a short summary (bullish, bearish, or neutral) with key reasons."
                response = openai.Completion.create(
                    model="gpt-3.5-turbo",  # Use the correct model name
                    prompt=prompt,
                    max_tokens=100
                )
                sentiments[ticker] = response['choices'][0]['text'].strip()  # Fetch the text from the response
                break  # Exit retry loop if successful
            except Exception as e:  # Catch all exceptions (no more `openai.error` needed)
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
    '''

