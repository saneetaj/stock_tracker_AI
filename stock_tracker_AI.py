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
    # Allow historical data to be fetched even if live stream fails.

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

def compute_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_prev_close = (data["High"] - data["Close"].shift()).abs()
    low_prev_close = (data["Low"] - data["Close"].shift()).abs()
    tr = high_low.combine(high_prev_close, max).combine(low_prev_close, max)
    up_move = data["High"] - data["High"].shift()
    down_move = data["Low"].shift() - data["Low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_smooth = tr.rolling(window=period, min_periods=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).sum()
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period, min_periods=period).mean()

def compute_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (data["High"] + data["Low"] + data["Close"]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    def mad_func(x):
        return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(window=period, min_periods=period).apply(mad_func, raw=True)
    return (tp - sma_tp) / (0.015 * mad)

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in data.columns:
        st.error("Data is missing the 'Date' column.")
        return pd.DataFrame()
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values("Date", inplace=True)
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["RSI"] = compute_rsi(data["Close"], period=14)
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    data["BB_Std"] = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + 2 * data["BB_Std"]
    data["BB_Lower"] = data["BB_Middle"] - 2 * data["BB_Std"]
    data["Stoch_High"] = data["High"].rolling(window=14).max()
    data["Stoch_Low"] = data["Low"].rolling(window=14).min()
    data["Stochastic_K"] = 100 * ((data["Close"] - data["Stoch_Low"]) / (data["Stoch_High"] - data["Stoch_Low"]))
    data["Stochastic_D"] = data["Stochastic_K"].rolling(window=3).mean()
    data["ADX"] = compute_adx(data, period=14)
    data["CCI"] = compute_cci(data, period=20)
    return data

# -------------------------
# Buy Low, Sell High Signal Generation and Backtesting

def generate_buy_low_sell_high_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates buy and sell signals using a multi-indicator "buy low, sell high" approach.
    
    Oversold (Buy) conditions:
      - RSI < 30
      - Close < Bollinger Lower Band
      - MACD < MACD_Signal
      - CCI < -100
      - ADX < 20  (non-trending market)
      
    Overbought (Sell) conditions:
      - RSI > 70
      - Close > Bollinger Upper Band
      - MACD > MACD_Signal
      - CCI > 100
      - ADX < 20
      
    If at least 2 oversold conditions are met, a Buy signal is generated.
    If at least 2 overbought conditions are met, a Sell signal is generated.
    """
    data = data.copy()
    required_cols = ["RSI", "BB_Lower", "BB_Upper", "MACD", "MACD_Signal", "CCI", "ADX", "Close"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing columns {missing_cols}. Please recalculate indicators first.")
        return data

    data["cond_RSI_buy"] = data["RSI"] < 30
    data["cond_BB_buy"]  = data["Close"] < data["BB_Lower"]
    data["cond_MACD_buy"] = data["MACD"] < data["MACD_Signal"]
    data["cond_CCI_buy"] = data["CCI"] < -100
    data["cond_ADX_buy"] = data["ADX"] < 20

    data["cond_RSI_sell"] = data["RSI"] > 70
    data["cond_BB_sell"]  = data["Close"] > data["BB_Upper"]
    data["cond_MACD_sell"] = data["MACD"] > data["MACD_Signal"]
    data["cond_CCI_sell"] = data["CCI"] > 100
    data["cond_ADX_sell"] = data["ADX"] < 20

    data["buy_score"] = (data["cond_RSI_buy"].astype(int) +
                         data["cond_BB_buy"].astype(int) +
                         data["cond_MACD_buy"].astype(int) +
                         data["cond_CCI_buy"].astype(int) +
                         data["cond_ADX_buy"].astype(int))
    
    data["sell_score"] = (data["cond_RSI_sell"].astype(int) +
                          data["cond_BB_sell"].astype(int) +
                          data["cond_MACD_sell"].astype(int) +
                          data["cond_CCI_sell"].astype(int) +
                          data["cond_ADX_sell"].astype(int))
    
    data["Buy_Signal"] = data["buy_score"] >= 2
    data["Sell_Signal"] = data["sell_score"] >= 2
    
    return data

def backtest_buy_low_sell_high(data: pd.DataFrame) -> pd.DataFrame:
    """
    Backtests the buy-low, sell-high strategy.
    Positions are simulated as follows:
      - When a Buy signal is generated, go long on the next day.
      - Remain in position until a Sell signal.
    Cumulative returns are calculated for the strategy and compared to a buy-and-hold benchmark.
    """
    df = data.copy()
    if "Date" not in df.columns:
        st.error("Data is missing the 'Date' column.")
        return pd.DataFrame()
    
    df = calculate_indicators(df)
    df.sort_values("Date", inplace=True)
    df = df.reset_index(drop=True)
    df = generate_buy_low_sell_high_signals(df)
    
    df.set_index("Date", inplace=True)
    df["Signal"] = 0
    df.loc[df["Buy_Signal"], "Signal"] = 1
    df.loc[df["Sell_Signal"], "Signal"] = 0
    df["Position"] = df["Signal"].replace(to_replace=0, method='ffill').shift(1).fillna(0)
    
    df["Market_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Position"]
    df["Cum_Market_Return"] = (1 + df["Market_Return"]).cumprod()
    df["Cum_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()
    
    return df

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
            logging.error(f"No intraday data found for {ticker}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching intraday data for {ticker}: {e}")
        logging.error(f"Error fetching intraday data for {ticker}: {e}")
        return None

# -------------------------
# News and Sentiment Functions

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
            # Prefer intraday data if available; otherwise, use historical
            intraday_data = await get_intraday_data(ticker)
            historical_data = get_historical_stock_data(ticker)
            data_to_use = intraday_data if intraday_data is not None else historical_data
            if data_to_use is None or data_to_use.empty:
                st.write(f"⚠️ No data available for {ticker}")
                continue
            
            # Calculate indicators and generate buy low, sell high signals for charting
            processed_data = calculate_indicators(data_to_use)
            processed_data = generate_buy_low_sell_high_signals(processed_data)
            
            # Plot the stock chart with buy/sell signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=processed_data['Date'],
                y=processed_data['Close'],
                mode='lines',
                name='Close Price',
                showlegend=True
            ))
            buy_signals = processed_data[processed_data["Buy_Signal"] == True]
            sell_signals = processed_data[processed_data["Sell_Signal"] == True]
            fig.add_trace(go.Scatter(
                x=buy_signals['Date'],
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(color='green', symbol='triangle-up', size=12),
                name='Buy Signal',
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=sell_signals['Date'],
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=12),
                name='Sell Signal',
                showlegend=True
            ))
            fig.update_layout(
                title=f"{ticker} Stock Chart with Buy Low, Sell High Signals",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Signals"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(processed_data.tail())
            st.success("✅ Stock data and chart updated!")
            
            # -------------------------
            # Backtest the Buy Low, Sell High Strategy on Historical Data
            st.subheader(f"📈 Backtest: Buy Low, Sell High Strategy for {ticker}")
            backtest_data = historical_data  # Use historical data for backtesting
            bt_results = backtest_buy_low_sell_high(backtest_data)
            if not bt_results.empty:
                bt_fig = go.Figure()
                bt_fig.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results['Cum_Market_Return'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='royalblue', width=3),
                    hovertemplate='%{x|%Y-%m-%d}<br>Buy & Hold: %{y:.2f}<extra></extra>'
                ))
                bt_fig.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results['Cum_Strategy_Return'],
                    mode='lines',
                    name='Buy Low, Sell High Strategy',
                    line=dict(color='firebrick', width=3),
                    hovertemplate='%{x|%Y-%m-%d}<br>Strategy: %{y:.2f}<extra></extra>'
                ))
                bt_fig.update_layout(
                    title={
                        'text': f"Cumulative Returns Comparison: {ticker}",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    legend_title="Legend",
                    hovermode="x unified",
                    template="plotly_white"
                )
                bt_fig.add_annotation(
                    text="Blue line: Buy & Hold<br>Red line: Buy Low, Sell High Strategy",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15, showarrow=False,
                    font=dict(size=12)
                )
                st.plotly_chart(bt_fig, use_container_width=True)
            else:
                st.write("⚠️ Backtest data not available.")

if __name__ == "__main__":
    asyncio.run(main())
