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

# For auto-refresh
from streamlit_autorefresh import st_autorefresh

# Alpaca Imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca_trade_api.rest import REST

# Neural Network / Scaling Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Changed to INFO for more detail

# ------------------------------------
# Load API keys from Streamlit secrets
# (Keep existing code for loading keys)
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
# (Keep existing code)
try:
    openai_client = openai.OpenAI(api_key=openai_api_key)
except openai.OpenAIError as e:
    st.error(f"Error initializing OpenAI client: {e}")
    logging.error(f"Error initializing OpenAI client: {e}")
    st.stop()


# Initialize Alpaca data and trade clients
# (Keep existing code)
historical_client = None
live_stream = None
try:
    historical_client = StockHistoricalDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
    # live_stream = StockDataStream(api_key=alpaca_api_key, secret_key=alpaca_secret_key) # Live stream setup can be complex
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
# (Keep existing code)
try:
    import finnhub
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
except Exception as e:
    st.error(f"Error initializing Finnhub client: {e}")
    logging.error(f"Error initializing Finnhub client: {e}")
    st.stop()

# ------------------------------------
# Indicator Calculation Functions for Intraday

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series: # Changed default to 14
    """Calculates RSI with specified period."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Exponential Moving Average for RSI calculation (common practice)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_intraday_indicators(data: pd.DataFrame,
                                  short_ema_period: int = 9,
                                  long_ema_period: int = 21,
                                  rsi_period: int = 14,
                                  bb_period: int = 20) -> pd.DataFrame:
    """
    Calculates intraday-focused indicators needed for the strategy.
    Handles timezone conversion to US/Eastern.
    """
    df = data.copy()
    if "Date" not in df.columns:
        st.error("Data is missing the 'Date' column.")
        logging.error("Data is missing the 'Date' column during indicator calculation.")
        return pd.DataFrame()

    # Ensure 'Date' is datetime and timezone-aware (US/Eastern)
    try:
        df["Date"] = pd.to_datetime(df["Date"])
        if df["Date"].dt.tz is None:
            df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        else:
            df["Date"] = df["Date"].dt.tz_convert("US/Eastern")
    except Exception as e:
        st.error(f"Error processing 'Date' column: {e}")
        logging.error(f"Error processing 'Date' column: {e}")
        return pd.DataFrame()

    df.sort_values("Date", inplace=True)

    # Calculate EMAs
    df[f"EMA_{short_ema_period}"] = df["Close"].ewm(span=short_ema_period, adjust=False).mean()
    df[f"EMA_{long_ema_period}"] = df["Close"].ewm(span=long_ema_period, adjust=False).mean()

    # Calculate RSI
    df["RSI"] = compute_rsi(df["Close"], period=rsi_period)

    # Calculate Bollinger Bands (optional, but keep for potential future use/plotting)
    df["BB_Middle"] = df["Close"].rolling(window=bb_period).mean()
    df["BB_Std"] = df["Close"].rolling(window=bb_period).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]

    # Add Day identifier for daily signal limits
    df['Day'] = df['Date'].dt.date

    logging.info(f"Calculated indicators for data ending {df['Date'].iloc[-1]}")
    return df

# ------------------------------------
# NEW Day Trading Signal Generation: EMA Pullback Strategy

def generate_daytrade_pullback_signals(data: pd.DataFrame,
                                       short_ema_period: int = 9,
                                       long_ema_period: int = 21,
                                       rsi_period: int = 14,
                                       rsi_overbought: int = 70, # Threshold to avoid buying
                                       rsi_oversold: int = 30   # Threshold to avoid selling
                                       ) -> pd.DataFrame:
    """
    Generates day trading signals based on EMA pullbacks in the direction of the trend.
    Limits signals to the first valid buy and first valid sell per day.

    Assumes 'EMA_X', 'EMA_Y', 'RSI', 'Low', 'High', 'Close', 'Day' columns exist.
    """
    df = data.copy()
    required_cols = [f"EMA_{short_ema_period}", f"EMA_{long_ema_period}", "RSI", "Low", "High", "Close", "Day"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns for signal generation: Check {required_cols}")
        logging.error(f"Missing required columns for signal generation: {required_cols}")
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        return df

    # Conditions for Buy Signal
    buy_trend_ok = (df[f"EMA_{short_ema_period}"] > df[f"EMA_{long_ema_period}"]) & (df["Close"] > df[f"EMA_{long_ema_period}"])
    buy_pullback = (df["Low"] <= df[f"EMA_{short_ema_period}"]) & (df["Close"] > df[f"EMA_{short_ema_period}"]) # Price touches or dips below short EMA then closes above
    buy_rsi_ok = df["RSI"] < rsi_overbought

    df["Buy_Signal_Raw"] = buy_trend_ok & buy_pullback & buy_rsi_ok

    # Conditions for Sell Signal (Exit Long / Enter Short)
    sell_trend_ok = (df[f"EMA_{short_ema_period}"] < df[f"EMA_{long_ema_period}"]) & (df["Close"] < df[f"EMA_{long_ema_period}"])
    sell_pullback = (df["High"] >= df[f"EMA_{short_ema_period}"]) & (df["Close"] < df[f"EMA_{short_ema_period}"]) # Price touches or pops above short EMA then closes below
    sell_rsi_ok = df["RSI"] > rsi_oversold

    df["Sell_Signal_Raw"] = sell_trend_ok & sell_pullback & sell_rsi_ok

    # --- Limit Signals to First Per Day ---
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False

    # Group by day and find the index of the first raw buy/sell signal
    first_buy_indices = df[df['Buy_Signal_Raw']].groupby('Day').head(1).index
    first_sell_indices = df[df['Sell_Signal_Raw']].groupby('Day').head(1).index

    # Set the final signals only at these first occurrences
    df.loc[first_buy_indices, 'Buy_Signal'] = True
    df.loc[first_sell_indices, 'Sell_Signal'] = True

    # Clean up intermediate columns
    df.drop(columns=['Buy_Signal_Raw', 'Sell_Signal_Raw'], inplace=True)

    logging.info(f"Generated pullback signals. Buys: {df['Buy_Signal'].sum()}, Sells: {df['Sell_Signal'].sum()}")
    return df


# ------------------------------------
# Intraday Backtesting (MODIFIED for Day Trading)

def backtest_intraday_strategy(data: pd.DataFrame,
                               short_ema_period: int = 9,
                               long_ema_period: int = 21,
                               rsi_period: int = 14
                               ) -> pd.DataFrame:
    """
    Performs a simplified backtest of the intraday strategy.
    MODIFIED: Includes end-of-day position closing.
    """
    df = data.copy()
    if "Date" not in df.columns:
        st.error("Data is missing the 'Date' column for backtesting.")
        logging.error("Data is missing the 'Date' column for backtesting.")
        return pd.DataFrame()

    # 1. Calculate Indicators
    df = calculate_intraday_indicators(df, short_ema_period, long_ema_period, rsi_period)
    if df.empty: return pd.DataFrame() # Stop if indicator calculation failed

    # 2. Generate Signals using the NEW function
    df = generate_daytrade_pullback_signals(df, short_ema_period, long_ema_period, rsi_period)

    # Ensure 'Date' is the index for time-based operations
    if not isinstance(df.index, pd.DatetimeIndex):
         if "Date" in df.columns:
             df.set_index("Date", inplace=True)
         else:
             st.error("Cannot set DatetimeIndex for backtesting.")
             logging.error("Cannot set DatetimeIndex for backtesting.")
             return pd.DataFrame()

    # --- 3. Simulate Positions (Day Trading Logic) ---
    df["Position"] = 0 # 1 for long, -1 for short (optional), 0 for flat
    df["Trade_Action"] = 0 # 1 for Buy, -1 for Sell, 0 for Hold/Close

    current_position = 0
    last_trade_day = None

    # Define market close time (e.g., 3:59 PM Eastern)
    market_close_time = datetime.time(15, 59)

    for i in range(len(df)):
        current_time = df.index[i].time()
        current_day = df.index[i].date()

        # --- End-of-Day Exit ---
        # If it's the last bar of the day OR time is at/after market close time, flatten position
        is_last_bar_of_day = (i == len(df) - 1) or (current_day != df.index[i+1].date())
        force_exit = (current_time >= market_close_time) or is_last_bar_of_day

        if force_exit and current_position != 0:
            df.iloc[i, df.columns.get_loc("Position")] = 0 # Flatten position
            df.iloc[i, df.columns.get_loc("Trade_Action")] = -current_position # Record closing action
            current_position = 0
            # logging.info(f"{df.index[i]}: Forced EOD exit.")
            continue # Skip regular signal logic on forced exit bar

        # --- Regular Signal Logic ---
        # Reset trade tracking for a new day
        if last_trade_day != current_day:
            last_trade_day = current_day

        # Check for Buy Signal
        if df.iloc[i]["Buy_Signal"] and current_position == 0:
            df.iloc[i, df.columns.get_loc("Position")] = 1
            df.iloc[i, df.columns.get_loc("Trade_Action")] = 1 # Record Buy
            current_position = 1
            # logging.info(f"{df.index[i]}: Entered LONG.")

        # Check for Sell Signal (Exit Long or Enter Short - simplified to exit long here)
        elif df.iloc[i]["Sell_Signal"] and current_position == 1:
            df.iloc[i, df.columns.get_loc("Position")] = 0
            df.iloc[i, df.columns.get_loc("Trade_Action")] = -1 # Record Sell
            current_position = 0
            # logging.info(f"{df.index[i]}: Exited LONG.")

        # If no signal or already in position, hold
        elif current_position != 0:
             df.iloc[i, df.columns.get_loc("Position")] = current_position

    # --- 4. Calculate Returns ---
    # Shift position by 1 to apply it to the *next* bar's return (trade on signal, realize P/L on next bar)
    df["Position_Shifted"] = df["Position"].shift(1).fillna(0)
    df["Market_Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Market_Return"] * df["Position_Shifted"]

    # Calculate cumulative returns
    df["Cum_Market_Return"] = (1 + df["Market_Return"]).cumprod()
    df["Cum_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()

    logging.info("Backtesting complete.")
    return df

# ------------------------------------
# Data Retrieval for Intraday Data
# (Keep existing get_intraday_stock_data function, ensure it returns 'Date' column)
def get_intraday_stock_data(ticker: str, days: int = 1) -> Optional[pd.DataFrame]:
    """Fetches 1-minute intraday bars from Alpaca for the specified number of days."""
    try:
        # Use pytz to handle timezone correctly, especially around DST changes
        eastern = pytz.timezone('US/Eastern')
        end_date_aware = datetime.datetime.now(eastern)
        # Go back slightly more than 'days' to ensure we capture the start of the first day
        start_date_aware = end_date_aware - datetime.timedelta(days=days + 1)

        # Format dates for Alpaca API (RFC3339 or YYYY-MM-DD)
        # Using timezone-aware start/end is generally safer with Alpaca v2
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            start=start_date_aware,
            end=end_date_aware,
            timeframe=TimeFrame.Minute, # 1-minute bars
            # feed="iex" # IEX is free but limited, SIP provides full market data (requires subscription)
            feed="sip" # Use SIP if available/subscribed, otherwise fallback to iex
        )
        logging.info(f"Requesting intraday data for {ticker} from {start_date_aware} to {end_date_aware}")

        if historical_client is not None:
            bars = historical_client.get_stock_bars(request_params)
        else:
            st.error("Alpaca historical client not initialized.")
            logging.error("Alpaca historical client not initialized.")
            return None

        if bars and ticker in bars.df.index.get_level_values('symbol').unique():
            # Access data using .df multi-index
            df = bars.df.loc[ticker].copy() # Get DataFrame for the specific ticker

            # Rename columns to match expected format (lowercase)
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume', 'timestamp': 'Date'
            }, inplace=True)

            # Ensure 'Date' column exists after rename
            if 'Date' not in df.columns:
                 df['Date'] = df.index # If timestamp was the index

            # Convert index/Date to US/Eastern timezone
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is None:
                df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            else:
                df["Date"] = df["Date"].dt.tz_convert("US/Eastern")

            # Filter for standard market hours (9:30 AM to 4:00 PM Eastern)
            start_time = datetime.time(9, 30)
            end_time = datetime.time(16, 0)
            df = df[(df["Date"].dt.time >= start_time) & (df["Date"].dt.time <= end_time)]

            # Filter for the requested number of *trading days*
            unique_days = df['Date'].dt.normalize().unique()
            if len(unique_days) > days:
                 first_valid_day = unique_days[-days]
                 df = df[df['Date'].dt.normalize() >= first_valid_day]

            logging.info(f"Successfully fetched {len(df)} intraday bars for {ticker}")
            return df.reset_index(drop=True) # Reset index for easier processing later
        else:
            st.warning(f"⚠️ No intraday data found for {ticker} in the response.")
            logging.warning(f"No intraday data found for {ticker} in the response.")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
        logging.exception(f"Error fetching intraday data for {ticker}:") # Log full traceback
        return None


# ------------------------------------
# Historical Data Retrieval (for NN Forecasting)
# (Keep existing get_historical_stock_data function)
def get_historical_stock_data(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetches daily historical bars for the past 'days' from Alpaca."""
    try:
        eastern = pytz.timezone('US/Eastern')
        end_date_aware = datetime.datetime.now(eastern)
        start_date_aware = end_date_aware - datetime.timedelta(days=days + 2) # Go back a bit extra

        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            start=start_date_aware,
            end=end_date_aware,
            timeframe=TimeFrame.Day,
            feed="sip" # Or "iex"
        )
        logging.info(f"Requesting daily data for {ticker} from {start_date_aware} to {end_date_aware}")

        if historical_client is not None:
            bars = historical_client.get_stock_bars(request_params)
        else:
            st.error("Alpaca historical client not initialized.")
            logging.error("Alpaca historical client not initialized.")
            return None

        if bars and ticker in bars.df.index.get_level_values('symbol').unique():
            df = bars.df.loc[ticker].copy()
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume', 'timestamp': 'Date'
            }, inplace=True)
            if 'Date' not in df.columns: df['Date'] = df.index

            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is None:
                df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            else:
                df["Date"] = df["Date"].dt.tz_convert("US/Eastern")

            df.sort_values("Date", inplace=True)
            # Ensure we have approximately the right number of days after filtering weekends etc.
            df = df.last(f'{days}D') # Keep data from the last N calendar days
            logging.info(f"Successfully fetched {len(df)} daily bars for {ticker}")
            return df.reset_index(drop=True)
        else:
            st.warning(f"⚠️ No historical daily data found for {ticker}")
            logging.warning(f"No historical daily data found for {ticker}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching historical data for {ticker}: {e}")
        logging.exception(f"Error fetching historical data for {ticker}:")
        return None


# ------------------------------------
# Neural Network Forecasting Functions
# (Keep existing prepare_data, build_lstm_model, predict_stock_with_lstm functions)
def prepare_data(series: np.ndarray, window_size: int) -> (np.ndarray, np.ndarray):
    """Convert time series to sequences for training."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def predict_stock_with_lstm(data: pd.DataFrame, window_size: int = 20, future_steps: int = 5) -> pd.DataFrame:
    """
    Trains a simple LSTM model on historical daily closing prices and forecasts future_steps days.
    Returns a DataFrame with forecasted dates and predicted closing prices.
    """
    df = data.copy()
    if 'Date' not in df.columns:
        st.error("Missing 'Date' column for LSTM prediction.")
        logging.error("Missing 'Date' column for LSTM prediction.")
        return pd.DataFrame()

    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    if 'Close' not in df.columns:
        st.error("Missing 'Close' column for LSTM prediction.")
        logging.error("Missing 'Close' column for LSTM prediction.")
        return pd.DataFrame()

    close_prices = df["Close"].values.reshape(-1, 1)

    # Handle potential NaNs before scaling
    if np.isnan(close_prices).any():
        st.warning("NaN values found in 'Close' prices before LSTM training. Filling with forward fill.")
        logging.warning("NaN values found in 'Close' prices before LSTM training. Filling with forward fill.")
        close_prices = pd.Series(close_prices.flatten()).ffill().values.reshape(-1, 1)
        if np.isnan(close_prices).any(): # If still NaNs after ffill (e.g., at the start)
             st.error("NaN values persist after ffill. Cannot train LSTM.")
             logging.error("NaN values persist after ffill. Cannot train LSTM.")
             return pd.DataFrame()


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    if len(scaled_data) <= window_size:
        st.error(f"Not enough historical data ({len(scaled_data)} points) to train the LSTM model with window size {window_size}.")
        logging.error(f"Not enough historical data for LSTM: {len(scaled_data)} <= {window_size}")
        return pd.DataFrame()

    X, y = prepare_data(scaled_data, window_size)
    if X.shape[0] == 0:
         st.error("Could not create training sequences (X, y) for LSTM.")
         logging.error("Could not create training sequences (X, y) for LSTM.")
         return pd.DataFrame()

    X = X.reshape((X.shape[0], X.shape[1], 1)) # Reshape for LSTM input

    try:
        model = build_lstm_model((X.shape[1], 1))
        # Consider adding validation split or early stopping for better training
        model.fit(X, y, epochs=10, batch_size=16, verbose=0) # verbose=0 to avoid printing progress in Streamlit

        last_window = scaled_data[-window_size:]
        predictions_scaled = []
        current_input = last_window.reshape(1, window_size, 1)

        for _ in range(future_steps):
            pred = model.predict(current_input, verbose=0)
            predictions_scaled.append(pred[0, 0])
            # Prepare next input: remove first step, append prediction
            current_input = np.append(current_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        # Inverse transform the scaled predictions
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

        # Generate future dates (business days only)
        last_date = df.index[-1]
        forecast_dates = []
        current_date = last_date
        while len(forecast_dates) < future_steps:
            current_date += datetime.timedelta(days=1)
            # Ensure the generated date is timezone-aware (using the source timezone)
            if current_date.weekday() < 5: # Monday to Friday
                forecast_dates.append(current_date)

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted_Close": predictions
        })
        # Ensure forecast dates are also US/Eastern
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.tz_convert("US/Eastern")

        logging.info(f"LSTM forecast generated successfully for {future_steps} steps.")
        return forecast_df

    except Exception as e:
        st.error(f"Error during LSTM prediction: {e}")
        logging.exception("Error during LSTM prediction:")
        return pd.DataFrame()


# ------------------------------------
# Finnhub News and ChatGPT Summarization Functions
# (Keep existing get_stock_news and get_market_sentiment functions)
def get_stock_news(ticker: str) -> str:
    """
    Retrieves the top 3 news articles for the given ticker from Finnhub.
    """
    # Use Finnhub client if initialized
    if finnhub_client:
        try:
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)
            news = finnhub_client.company_news(ticker, _from=yesterday.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'))
            if news:
                news_articles = []
                for article in news[:3]: # Limit to top 3
                    title = article.get('headline', 'No Title')
                    article_url = article.get('url', '')
                    # Truncate long titles
                    title = (title[:100] + '...') if len(title) > 100 else title
                    news_articles.append(f"• [{title}]({article_url})") # Format as Markdown link
                return "\n".join(news_articles) if news_articles else f"ℹ️ No recent news found for {ticker}."
            else:
                 return f"ℹ️ No recent news found for {ticker}."
        except Exception as e:
            logging.error(f"Error fetching news from Finnhub for {ticker}: {e}")
            return f"⚠️ Error fetching news for {ticker}."
    else:
        # Fallback to direct request if client failed (less ideal)
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')}&to={datetime.datetime.now().strftime('%Y-%m-%d')}&token={finnhub_api_key}"
        try:
            response = requests.get(url, timeout=10) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if data:
                news_articles = []
                for article in data[:3]:
                    title = article.get('headline', 'No Title')
                    article_url = article.get('url', '')
                    title = (title[:100] + '...') if len(title) > 100 else title
                    news_articles.append(f"• [{title}]({article_url})")
                return "\n".join(news_articles) if news_articles else f"ℹ️ No recent news found for {ticker}."
            else:
                return f"ℹ️ No recent news found for {ticker}."
        except requests.exceptions.RequestException as e:
             logging.error(f"Error fetching news via direct request for {ticker}: {e}")
             return f"⚠️ Error fetching news for {ticker}."


# Use st.cache_data for caching OpenAI results
@st.cache_data(ttl=1800) # Cache for 30 minutes
def get_market_sentiment(_tickers: List[str]) -> dict: # Prefix internal variable with _
    """
    For each ticker, retrieves news from Finnhub and then uses ChatGPT (via OpenAI)
    to generate a brief market sentiment summary. Handles rate limits.
    """
    sentiments = {}
    # Use a frozenset for caching based on tickers, regardless of order
    # tickers_key = frozenset(_tickers) # Caching key - st.cache_data handles this automatically

    rate_limit_message = "⚠️ OpenAI rate limit likely reached. Sentiment analysis paused. Please try again later."
    general_error_message = "⚠️ Error during sentiment analysis."
    api_error_occurred = False

    for ticker in _tickers:
        news_data = get_stock_news(ticker)
        if news_data.startswith("⚠️"): # Skip analysis if news fetching failed
            sentiments[ticker] = news_data # Pass on the error message
            continue

        if api_error_occurred: # If rate limit hit for one ticker, skip others in this run
             sentiments[ticker] = rate_limit_message
             continue

        attempt = 1
        max_attempts = 3 # Reduced max attempts
        while attempt <= max_attempts:
            try:
                prompt = (f"Analyze the market sentiment for {ticker} based *only* on the following recent news headlines/links:\n{news_data}\n\n"
                          "Provide a brief summary (1-2 sentences) classifying sentiment as Bullish, Bearish, or Neutral, with the key reason derived *solely* from the provided news. "
                          "If the news is insufficient or mixed, state Neutral.")

                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a concise financial news analyst providing sentiment summaries based *only* on the provided headlines."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100, # Reduced tokens for brevity
                    temperature=0.5 # Lower temperature for more focused output
                )
                sentiments[ticker] = response.choices[0].message.content.strip()
                logging.info(f"Sentiment analysis successful for {ticker} on attempt {attempt}.")
                break # Success, exit retry loop

            except openai.RateLimitError as e:
                logging.warning(f"OpenAI Rate Limit Error for {ticker} on attempt {attempt}: {e}")
                if attempt == max_attempts:
                    sentiments[ticker] = rate_limit_message
                    api_error_occurred = True # Signal to skip remaining tickers
                else:
                    wait_time = 2 ** attempt # Exponential backoff
                    logging.info(f"Waiting {wait_time}s before retrying sentiment analysis for {ticker}.")
                    time.sleep(wait_time)
                attempt += 1

            except openai.OpenAIError as e:
                logging.error(f"OpenAI API Error for {ticker} on attempt {attempt}: {e}")
                if attempt == max_attempts:
                    sentiments[ticker] = general_error_message
                    api_error_occurred = True # Signal general API error
                else:
                    wait_time = 2 ** attempt
                    logging.info(f"Waiting {wait_time}s before retrying sentiment analysis for {ticker}.")
                    time.sleep(wait_time)
                attempt += 1

            except Exception as e: # Catch any other unexpected errors
                 logging.exception(f"Unexpected error during sentiment analysis for {ticker}:")
                 sentiments[ticker] = general_error_message
                 api_error_occurred = True
                 break # Exit loop on unexpected error

        # Short delay between tickers to help avoid rate limits
        if not api_error_occurred:
             time.sleep(1) # 1-second delay

    return sentiments


# ------------------------------------
# Streamlit UI (Modified to use new strategy)
async def main():
    st.set_page_config(layout="wide") # Use wider layout
    st.title("📈 EMA Pullback Day Trading Strategy")

    # --- Sidebar for Inputs and Sentiment ---
    st.sidebar.header("Configuration")
    tickers_input = st.sidebar.text_input("Stock Ticker(s) (comma-separated)", "AAPL,MSFT", key="tickers_input")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    st.sidebar.header("Strategy Parameters")
    short_ema = st.sidebar.slider("Short EMA Period", 5, 20, 9, key="short_ema")
    long_ema = st.sidebar.slider("Long EMA Period", 15, 100, 21, key="long_ema")
    rsi_p = st.sidebar.slider("RSI Period", 7, 30, 14, key="rsi_p")

    # Auto-refresh control
    refresh_interval = st.sidebar.select_slider(
         "Auto-Refresh Interval (minutes)",
         options=[1, 2, 5, 10, 0], # 0 disables refresh
         value=5,
         key="refresh_interval"
    )
    if refresh_interval > 0:
        st_autorefresh(interval=refresh_interval * 60 * 1000, limit=None, key="intraday_autorefresh") # Use limit=None for indefinite refresh
        st.sidebar.caption(f"Auto-refreshing every {refresh_interval} min.")
    else:
         st.sidebar.caption("Auto-refresh disabled.")


    # --- Main Area ---
    if not tickers:
        st.warning("Please enter at least one stock ticker symbol.")
        return # Stop execution if no tickers

    # Display Sentiment in Sidebar (run only once per ticker list change)
    try:
        sentiments = get_market_sentiment(tickers)
        st.sidebar.header("📢 Market Sentiment")
        for ticker in tickers:
            if ticker in sentiments:
                st.sidebar.markdown(f"**{ticker}:**")
                st.sidebar.info(sentiments[ticker])
            else:
                st.sidebar.markdown(f"**{ticker}:** ℹ️ Sentiment not available.")
    except Exception as e:
         st.sidebar.error(f"Error getting sentiment: {e}")
         logging.exception("Error getting sentiment in main UI:")


    # Process each ticker
    for ticker in tickers:
        st.markdown("---") # Separator between tickers
        st.header(f"{ticker} - Intraday Analysis")

        col1, col2 = st.columns(2) # Create columns for charts

        with col1:
            st.subheader("📈 Intraday Chart & Signals")
            # Fetch only 1 day of data for the main intraday chart
            intraday_data = get_intraday_stock_data(ticker, days=1)

            if intraday_data is None or intraday_data.empty:
                st.warning(f"⚠️ No intraday data available for {ticker} for charting.")
            else:
                # Calculate indicators and generate signals for the chart
                processed_data = calculate_intraday_indicators(intraday_data, short_ema, long_ema, rsi_p)
                processed_data = generate_daytrade_pullback_signals(processed_data, short_ema, long_ema, rsi_p)

                # Plot intraday chart with signals
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=processed_data["Date"], y=processed_data["Close"],
                                         mode="lines", name="Close", line=dict(color='blue')))
                # Add EMAs to chart
                fig.add_trace(go.Scatter(x=processed_data["Date"], y=processed_data[f"EMA_{short_ema}"],
                                         mode="lines", name=f"EMA {short_ema}", line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=processed_data["Date"], y=processed_data[f"EMA_{long_ema}"],
                                         mode="lines", name=f"EMA {long_ema}", line=dict(color='purple', width=1)))

                # Plot signals
                buy_signals = processed_data[processed_data["Buy_Signal"]]
                sell_signals = processed_data[processed_data["Sell_Signal"]]
                fig.add_trace(go.Scatter(x=buy_signals["Date"], y=buy_signals["Close"] * 0.998, # Offset slightly below price
                                         mode="markers", marker=dict(color="green", symbol="triangle-up", size=10),
                                         name="Buy Signal", hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=sell_signals["Date"], y=sell_signals["Close"] * 1.002, # Offset slightly above price
                                         mode="markers", marker=dict(color="red", symbol="triangle-down", size=10),
                                         name="Sell Signal", hoverinfo='skip'))

                fig.update_layout(title=f"{ticker} Intraday (1-Min) - EMA Pullback Signals",
                                  xaxis_title="Time (US/Eastern)", yaxis_title="Price",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                  margin=dict(l=20, r=20, t=40, b=20)) # Adjust margins
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Backtest Performance (Simplified)")
            # Fetch slightly more data for backtesting context if needed (e.g., 2 days)
            # Note: Simple backtest uses the same 1-day data here for consistency with chart
            if intraday_data is None or intraday_data.empty:
                 st.warning(f"⚠️ No intraday data available for {ticker} for backtesting.")
            else:
                # Backtest the strategy using the fetched intraday data
                bt_results = backtest_intraday_strategy(intraday_data.copy(), short_ema, long_ema, rsi_p) # Pass params

                if not bt_results.empty and 'Cum_Strategy_Return' in bt_results.columns:
                    bt_fig = go.Figure()
                    bt_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results["Cum_Market_Return"],
                                                mode="lines", name="Buy & Hold",
                                                line=dict(color="gray", width=1.5),
                                                hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Buy & Hold: %{y:.3f}<extra></extra>'))
                    bt_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results["Cum_Strategy_Return"],
                                                mode="lines", name="Strategy",
                                                line=dict(color="green", width=1.5),
                                                hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Strategy: %{y:.3f}<extra></extra>'))
                    bt_fig.update_layout(title=f"Strategy vs. Buy & Hold (1 Day)",
                                         xaxis_title="Time (US/Eastern)", yaxis_title="Cumulative Return",
                                         hovermode="x unified", template="plotly_white",
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                         margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(bt_fig, use_container_width=True)

                    # Display simple performance metrics
                    final_strat_return = (bt_results["Cum_Strategy_Return"].iloc[-1] - 1) * 100
                    final_market_return = (bt_results["Cum_Market_Return"].iloc[-1] - 1) * 100
                    num_trades = bt_results[bt_results['Trade_Action'] != 0]['Trade_Action'].count()
                    st.metric(label="Strategy Return (1 Day)", value=f"{final_strat_return:.2f}%")
                    st.metric(label="Buy & Hold Return (1 Day)", value=f"{final_market_return:.2f}%")
                    st.metric(label="Number of Trades", value=f"{num_trades}")

                else:
                    st.warning("⚠️ Backtest could not be completed.")

        # LSTM Forecasting Section (remains the same)
        st.header(f"{ticker} - Daily Forecast")
        historical_data = get_historical_stock_data(ticker, days=365)
        if historical_data is None or historical_data.empty:
            st.warning(f"⚠️ No historical data available for {ticker} for forecasting.")
        else:
            if len(historical_data) < 50: # Check if enough data for LSTM
                 st.warning(f"⚠️ Insufficient historical data ({len(historical_data)} days) for reliable LSTM forecast.")
            else:
                try:
                    forecast_df = predict_stock_with_lstm(historical_data.copy(), window_size=20, future_steps=5)
                    if not forecast_df.empty:
                        hist_df_plot = historical_data.copy().set_index("Date") # Set index for plotting
                        fig_nn = go.Figure()
                        fig_nn.add_trace(go.Scatter(x=hist_df_plot.index, y=hist_df_plot["Close"],
                                                    mode="lines", name="Historical Close", line=dict(color='lightblue')))
                        fig_nn.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"],
                                                    mode="lines+markers", name="Forecast", line=dict(color='red')))
                        fig_nn.update_layout(title=f"{ticker} Daily Close & 5-Day Forecast (LSTM)",
                                             xaxis_title="Date (US/Eastern)", yaxis_title="Price",
                                             margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_nn, use_container_width=True)
                    else:
                         st.warning("⚠️ LSTM forecast could not be generated.")
                except Exception as e:
                     st.error(f"Error during LSTM prediction display: {e}")
                     logging.exception("Error during LSTM prediction display:")


if __name__ == "__main__":
    # Run the async main function using asyncio
    asyncio.run(main())

