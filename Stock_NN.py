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
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------
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
    historical_client = StockHistoricalDataClient(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
    live_stream = StockDataStream(api_key=alpaca_api_key, secret_key=alpaca_secret_key)
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

# ------------------------------------
# Indicator Calculation Functions for Intraday

def compute_rsi(series: pd.Series, period: int = 7) -> pd.Series:
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
      - SMA_20 (20-bar SMA)
      - RSI (7-period)
      - Bollinger Bands (20-bar)
    """
    if "Date" not in data.columns:
        st.error("Data is missing the 'Date' column.")
        return pd.DataFrame()
    data["Date"] = pd.to_datetime(data["Date"])
    # Assume raw data is in UTC; convert to US/Eastern
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

# ------------------------------------
# Multi-Timeframe Trend Signal Generation

# def generate_intraday_signals_multitimeframe(data: pd.DataFrame,
#                                              trend_threshold_up: float = 0.5,
#                                              trend_threshold_down: float = 0.5) -> pd.DataFrame:
#     """
#     Generates signals based on the trend over multiple timeframes.
    
#     It calculates the percentage change in price over the past 15, 30, 60, and 120 minutes.
#     - If all these timeframes show a positive trend (above trend_threshold_up),
#       the market is considered to be in an uptrend and a sell signal is generated
#       when the price is near a local high (e.g. at or above the upper Bollinger band).
#     - If all timeframes show a negative trend (below -trend_threshold_down),
#       the market is considered to be in a downtrend and a buy signal is generated
#       when the price is near a local low (e.g. at or below the lower Bollinger band).
#     """
#     data = data.copy()
#     # Calculate current price from the most recent data point
#     current_price = data["Close"].iloc[-1]
#     timeframes = [15, 30, 60, 120]  # timeframes in minutes (assume 1 bar = 1 minute)
#     trends = []
#     for tf in timeframes:
#         if len(data) >= tf:
#             price_tf_ago = data["Close"].iloc[-tf]
#             trend_pct = (current_price - price_tf_ago) / price_tf_ago * 100
#             trends.append(trend_pct)
#     # Determine if the trend is consistently up or down
#     uptrend = len(trends) > 0 and all(t >= trend_threshold_up for t in trends)
#     downtrend = len(trends) > 0 and all(t <= -trend_threshold_down for t in trends)
    
#     # Generate signals only if a clear trend exists
#     # For a buy signal: if in a downtrend and the price is near the lower Bollinger band
#     # For a sell signal: if in an uptrend and the price is near the upper Bollinger band
#     data["Buy_Signal"] = downtrend & (data["Close"] <= data["BB_Lower"])
#     data["Sell_Signal"] = uptrend & (data["Close"] >= data["BB_Upper"])
#     return data

def generate_intraday_signals_multitimeframe(data: pd.DataFrame,
                                             trend_threshold_up: float = 0.1,
                                             trend_threshold_down: float = 0.1,
                                             required_fraction: float = 0.75) -> pd.DataFrame:
    """
    Analyzes the trend over multiple timeframes (in minutes) and generates signals.
    
    It calculates the percentage change over 15, 30, 60, and 120 minutes.
    If at least 'required_fraction' (e.g. 75%) of these timeframes show a trend:
      - For an uptrend (sell condition): the percentage change > trend_threshold_up.
      - For a downtrend (buy condition): the percentage change < -trend_threshold_down.
    Then, signals are generated only if the current price is at the extreme of the Bollinger Bands.
    """
    data = data.copy()
    # Get the most recent price from the data.
    current_price = data["Close"].iloc[-1]
    timeframes = [15, 30, 60, 120]
    trend_values = []
    
    # For each timeframe, calculate the percentage change
    for tf in timeframes:
        if len(data) >= tf:
            price_tf_ago = data["Close"].iloc[-tf]
            trend_pct = (current_price - price_tf_ago) / price_tf_ago * 100
            trend_values.append(trend_pct)
    
    # Determine how many timeframes are needed (e.g., 75% of available timeframes)
    num_required = int(np.ceil(len(trend_values) * required_fraction))
    
    # Check if most timeframes indicate an uptrend or downtrend
    uptrend = np.sum(np.array(trend_values) > trend_threshold_up) >= num_required
    downtrend = np.sum(np.array(trend_values) < -trend_threshold_down) >= num_required

    # Generate signals only if a clear trend exists:
    # For a buy: if in a downtrend and the current close is near or below the lower Bollinger band.
    # For a sell: if in an uptrend and the current close is near or above the upper Bollinger band.
    data["Buy_Signal"] = downtrend & (data["Close"] <= data["BB_Lower"])
    data["Sell_Signal"] = uptrend & (data["Close"] >= data["BB_Upper"])
    return data

# ------------------------------------
# Intraday Backtesting
def backtest_intraday_strategy(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if "Date" not in df.columns:
        st.error("Data is missing the 'Date' column.")
        return pd.DataFrame()
    df = calculate_intraday_indicators(df)
    df.sort_values("Date", inplace=True)
    df = df.reset_index(drop=True)
    # Use the multi-timeframe signals
    df = generate_intraday_signals_multitimeframe(df)
    df.set_index("Date", inplace=True)
    df["Signal"] = 0
    df.loc[df["Buy_Signal"], "Signal"] = 1
    df.loc[df["Sell_Signal"], "Signal"] = 0
    # Simulate entering at next bar: forward-fill the position and shift by 1
    df["Position"] = df["Signal"].replace(to_replace=0, method='ffill').shift(1).fillna(0)
    df["Market_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Position"]
    df["Cum_Market_Return"] = (1 + df["Market_Return"]).cumprod()
    df["Cum_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()
    return df

# ------------------------------------
# Data Retrieval for Intraday Data
def get_intraday_stock_data(ticker: str, days: int = 1) -> Optional[pd.DataFrame]:
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
            df = pd.DataFrame([{
                'Date': bar.timestamp,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume,
            } for bar in bars_list])
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is None:
                df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            else:
                df["Date"] = df["Date"].dt.tz_convert("US/Eastern")
            # Filter to regular market hours: 9:30 to 16:00 ET
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

# ------------------------------------
# Historical Data Retrieval (for NN Forecasting)
def get_historical_stock_data(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetches daily historical bars for the past 'days' from Alpaca.
    """
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
            bars_list = bars[ticker]
            df = pd.DataFrame([{
                'Date': bar.timestamp,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume,
            } for bar in bars_list])
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is None:
                df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            else:
                df["Date"] = df["Date"].dt.tz_convert("US/Eastern")
            df.sort_values("Date", inplace=True)
            return df
        else:
            st.error(f"⚠️ No historical data found for {ticker}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching historical data for {ticker}: {e}")
        logging.error(f"Error fetching historical data for {ticker}: {e}")
        return None

# ------------------------------------
# Neural Network Forecasting Functions

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
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    if len(scaled_data) <= window_size:
        st.error("Not enough historical data to train the LSTM model.")
        return pd.DataFrame()
    X, y = prepare_data(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    last_window = scaled_data[-window_size:]
    predictions = []
    current_input = last_window.reshape(1, window_size, 1)
    for _ in range(future_steps):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0, 0])
        # Reshape prediction to (1, 1, 1) and append along axis=1
        current_input = np.append(current_input[:, 1:, :], np.array(pred).reshape(1, 1, 1), axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    forecast_dates = []
    current_date = last_date
    while len(forecast_dates) < future_steps:
        current_date += datetime.timedelta(days=1)
        if current_date.weekday() < 5:
            forecast_dates.append(current_date)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted_Close": predictions
    })
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.tz_convert("US/Eastern")
    return forecast_df

# ------------------------------------
# Streamlit UI
async def main():
    st.title("📈 Trend-Based Strategy with Neural-Network Forecast")
    tickers_input = st.text_input("Enter stock ticker symbol(s), separated by commas", "AAPL", key="tickers_input")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    # Auto-refresh intraday data every 5 minutes
    st_autorefresh(interval=300000, limit=0, key="intraday_autorefresh")
    
    if tickers:
        # (Optional: add sentiment retrieval code here)
        for ticker in tickers:
            st.subheader(f"📊 Intraday Stock Data for {ticker}")
            intraday_data = get_intraday_stock_data(ticker, days=1)
            if intraday_data is None or intraday_data.empty:
                st.write(f"⚠️ No intraday data available for {ticker}")
                continue

            # Calculate indicators on intraday data
            processed_data = calculate_intraday_indicators(intraday_data)
            # Generate signals using multi-timeframe trend analysis.
            # (Default thresholds can be tuned.)
            processed_data = generate_intraday_signals_multitimeframe(processed_data,
                                                                       trend_threshold_up=0.5,
                                                                       trend_threshold_down=0.5)

            # Plot intraday chart with fewer, high-probability signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=processed_data["Date"], y=processed_data["Close"],
                                     mode="lines", name="Close Price"))
            buy_signals = processed_data[processed_data["Buy_Signal"] == True]
            sell_signals = processed_data[processed_data["Sell_Signal"] == True]
            fig.add_trace(go.Scatter(x=buy_signals["Date"], y=buy_signals["Close"],
                                     mode="markers", marker=dict(color="green", symbol="triangle-up", size=12),
                                     name="Buy Signal"))
            fig.add_trace(go.Scatter(x=sell_signals["Date"], y=sell_signals["Close"],
                                     mode="markers", marker=dict(color="red", symbol="triangle-down", size=12),
                                     name="Sell Signal"))
            fig.update_layout(title=f"{ticker} Intraday Chart (Reduced Signals, US/Eastern)",
                              xaxis_title="Time (US/Eastern)", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # Backtest the trend-based strategy
            bt_results = backtest_intraday_strategy(intraday_data)
            if not bt_results.empty:
                bt_fig = go.Figure()
                bt_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results["Cum_Market_Return"],
                                            mode="lines", name="Buy & Hold",
                                            line=dict(color="blue", width=2),
                                            hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Buy & Hold: %{y:.2f}<extra></extra>'))
                bt_fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results["Cum_Strategy_Return"],
                                            mode="lines", name="Strategy",
                                            line=dict(color="orange", width=2),
                                            hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Strategy: %{y:.2f}<extra></extra>'))
                bt_fig.update_layout(title=f"Intraday Strategy vs. Buy & Hold: {ticker}",
                                     xaxis_title="Time (US/Eastern)", yaxis_title="Cumulative Return",
                                     hovermode="x unified", template="plotly_white")
                st.plotly_chart(bt_fig, use_container_width=True)
            else:
                st.write("⚠️ No backtest data available.")

            # LSTM-based Price Forecasting
            st.subheader(f"🔮 Stock Price Forecast for {ticker}")
            historical_data = get_historical_stock_data(ticker, days=365)
            if historical_data is None or historical_data.empty:
                st.write(f"⚠️ No historical data available for {ticker} for forecasting.")
            else:
                forecast_df = predict_stock_with_lstm(historical_data, window_size=20, future_steps=5)
                hist_df = historical_data.copy()
                hist_df.sort_values("Date", inplace=True)
                fig_nn = go.Figure()
                fig_nn.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Close"],
                                            mode="lines", name="Historical Close"))
                fig_nn.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"],
                                            mode="lines+markers", name="Forecast"))
                fig_nn.update_layout(title=f"{ticker} Daily Close and 5-Day Forecast (LSTM)",
                                     xaxis_title="Date (US/Eastern)", yaxis_title="Price")
                st.plotly_chart(fig_nn, use_container_width=True)

# Neural Network Forecasting Function for Stock Prices
def predict_stock_with_lstm(data: pd.DataFrame, window_size: int = 20, future_steps: int = 5) -> pd.DataFrame:
    df = data.copy()
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    if len(scaled_data) <= window_size:
        st.error("Not enough historical data to train the LSTM model.")
        return pd.DataFrame()
    X, y = prepare_data(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    last_window = scaled_data[-window_size:]
    predictions = []
    current_input = last_window.reshape(1, window_size, 1)
    for _ in range(future_steps):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], np.array(pred).reshape(1, 1, 1), axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    forecast_dates = []
    current_date = last_date
    while len(forecast_dates) < future_steps:
        current_date += datetime.timedelta(days=1)
        if current_date.weekday() < 5:
            forecast_dates.append(current_date)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted_Close": predictions})
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.tz_convert("US/Eastern")
    return forecast_df

def prepare_data(series: np.ndarray, window_size: int) -> (np.ndarray, np.ndarray):
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

if __name__ == "__main__":
    asyncio.run(main())
