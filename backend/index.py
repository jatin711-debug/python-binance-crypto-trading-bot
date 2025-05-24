import tensorflow as tf
import os
import sqlite3
import pandas as pd
import numpy as np
import asyncio
import json
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import Sequential, Input # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from newsapi import NewsApiClient
import logging
import time
from db import init_db
from datetime import datetime

# Enable eager execution if not already enabled
if not tf.executing_eagerly():
    print("Enabling eager execution...")
    tf.compat.v1.enable_eager_execution()

# Load environment variables
load_dotenv()
init_db()

# Binance API credentials
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Binance Client
client = Client(API_KEY, API_SECRET)

# NewsAPI Key
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'YOUR_NEWSAPI_KEY_HERE')
NEWS_SOURCES = 'bbc-news,reuters,bloomberg,cnbc'


# Constants
STOP_LOSS = 0.02  # 2% Stop Loss
TAKE_PROFIT = 0.05  # 5% Take Profit
MAX_POSITION_SIZE = 0.1  # Maximum 10% of account balance per trade

script_directory = os.path.dirname(__file__)
trades_json_path = os.path.join(script_directory, 'trades.json')
log_file_path = os.path.join(script_directory, 'trading_log.txt')
db_path = os.path.join(script_directory, 'trades.db')

def setup_logging():
    """Setup logging configuration."""
    logger = logging.getLogger('trading_bot')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler for general logging
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    # Download VADER lexicon if not already downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        nltk.download('vader_lexicon')
    
    return logger

def initialize_json_log():
    """Initialize the JSON log file with an empty 'data' key."""
    with open(trades_json_path, 'w') as f:
        json.dump({"data": []}, f, indent=4)

def log_trade_to_json(log_data):
    """Log trade data to JSON file."""
    try:
        # Read the existing data
        with open(trades_json_path, 'r') as f:
            json_data = json.load(f)

        # Append new log data to the 'data' key
        json_data["data"].append(log_data)

        # Write the updated data back to the file
        with open(trades_json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        logging.error(f"Error logging trade to JSON: {e}")

def log_trade_to_text(log_data):
    """Log trade data to text file."""
    try:
        logger = logging.getLogger('trading_bot')
        logger.info(f"Trade Executed: {log_data}")
    except Exception as e:
        logging.error(f"Error logging trade to text: {e}")

def log_trade_to_sqlite(log_data):
    """Log trade data to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, current_price, lstm_prediction, 
                                rf_prediction, ensemble_prediction, action, average_sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_data['timestamp'], log_data['current_price'], 
            log_data['lstm_prediction'], log_data['rf_prediction'], 
            log_data['ensemble_prediction'], log_data['action'],
            log_data.get('average_sentiment', 0.0) # Use .get for robustness
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error logging trade to SQLite: {e}")

def fetch_data_in_batches(symbol, interval, batch_size, lookback):
    data = []
    end_time = int(time.time() * 1000)  # Convert to milliseconds
    logger = logging.getLogger('trading_bot')

    for batch in range(batch_size):
        start_time = end_time - (lookback * 60 * 1000)  # Convert lookback to milliseconds
        try:
            data_batch = client.get_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time)
            if not data_batch:
                logger.warning(f"No data received for batch {batch + 1}")
                break
            data.extend(data_batch)
            logger.info(f"Batch {batch + 1}/{batch_size}: Fetched {len(data_batch)} records")
            end_time = start_time
            time.sleep(1)  # To avoid hitting API rate limits
        except Exception as e:
            logger.error(f"Error fetching batch {batch + 1}: {e}")
            break

    return data

async def fetch_historical_data(symbol, interval, lookback):
    """Fetch historical data from Binance with error handling."""
    try:
        loop = asyncio.get_event_loop()
        klines = await loop.run_in_executor(None, fetch_data_in_batches, symbol, interval, 5, lookback)
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        if df.empty:
            raise ValueError("No data received from Binance")
        if df['close'].isnull().any():
            raise ValueError("Missing values in price data")
            
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        raise

def calculate_technical_indicators(df, look_back=60):
    """Calculate technical indicators for the dataset."""
    try:
        # Moving averages
        df['SMA'] = df['close'].rolling(window=look_back).mean()
        df['EMA'] = df['close'].ewm(span=look_back).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=look_back).mean()
        df['BB_upper'] = df['BB_middle'] + (df['close'].rolling(window=look_back).std() * 2)
        df['BB_lower'] = df['BB_middle'] - (df['close'].rolling(window=look_back).std() * 2)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=look_back).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=look_back).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(window=look_back).mean()
        df['Volume_StD'] = df['volume'].rolling(window=look_back).std()
        
        return df.fillna(method='ffill')
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        raise

def prepare_features(df, avg_sentiment=0.0): # Added avg_sentiment with a default
    """Prepare feature set for models, including sentiment score."""
    try:
        features = pd.DataFrame()
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close']/df['close'].shift(1))
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        # Technical indicators
        features['sma'] = df['SMA']
        features['ema'] = df['EMA']
        features['bb_upper'] = df['BB_upper']
        features['bb_lower'] = df['BB_lower']
        features['rsi'] = df['RSI']
        features['macd'] = df['MACD']
        features['macd_signal'] = df['Signal_Line']
        features['volume_sma'] = df['Volume_SMA']
        
        # Additional features
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_to_high'] = df['close'] / df['high']
        features['close_to_low'] = df['close'] / df['low']

        # Add sentiment score as a feature
        # This will add the sentiment score as a constant column for the current feature set
        features['sentiment_score'] = avg_sentiment
        
        return features.fillna(method='ffill')
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        raise

def prepare_lstm_data(features, look_back=60):
    """Prepare data specifically for LSTM model."""
    try:
        # Separate close price for target scaling
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaled = close_scaler.fit_transform(features['close'].values.reshape(-1, 1))
        
        # Scale all features
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(look_back, len(scaled_features)):
            X.append(scaled_features[i-look_back:i])
            y.append(close_scaled[i])
        
        return np.array(X), np.array(y), close_scaler, feature_scaler
    except Exception as e:
        logging.error(f"Error preparing LSTM data: {e}")
        raise

def prepare_rf_data(features, look_back=60):
    """Prepare data specifically for Random Forest model."""
    try:
        X = features.values[look_back:]
        y = features['close'].values[look_back:]
        return X, y
    except Exception as e:
        logging.error(f"Error preparing RF data: {e}")
        raise

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Create LSTM model."""
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=units, return_sequences=True, 
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(dropout_rate),
            LSTM(units=units, return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(dropout_rate),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')
        return model
    except Exception as e:
        logging.error(f"Error creating LSTM model: {e}")
        raise

def create_rf_model():
    """Create Random Forest model."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

def ensemble_predict(lstm_pred, rf_pred, weights=[0.6, 0.4]):
    """Combine predictions from both models."""
    return weights[0] * lstm_pred + weights[1] * rf_pred

def calculate_position_size(account_balance, entry_price, stop_loss_percentage=STOP_LOSS):
    """Calculate position size with risk management."""
    try:
        risk_amount = account_balance * stop_loss_percentage
        position_size = risk_amount / (entry_price * stop_loss_percentage)
        max_position = account_balance * MAX_POSITION_SIZE
        return min(position_size, max_position)
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        raise

def fetch_news_data(symbol):
    """Fetch news data from NewsAPI."""
    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        # Extract base currency for keyword search (e.g., 'BTC' from 'BTCUSDT')
        keyword = symbol.replace('USDT', '').replace('BUSD', '') # Basic symbol to keyword
        headlines = newsapi.get_everything(
            q=keyword,
            sources=NEWS_SOURCES,
            language='en',
            sort_by='publishedAt',
            page_size=20  # Fetch recent 20 articles
        )
        articles = [article['content'] for article in headlines.get('articles', []) if article['content']]
        return articles
    except Exception as e:
        logging.error(f"Error fetching news data for {symbol}: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment of the given text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def get_average_sentiment(symbol):
    """Fetch news and calculate average sentiment for a symbol."""
    articles = fetch_news_data(symbol)
    if not articles:
        return 0.0  # Neutral sentiment if no articles or error

    total_sentiment = 0
    for article_text in articles:
        total_sentiment += analyze_sentiment(article_text)
    
    return total_sentiment / len(articles) if articles else 0.0

async def place_order(symbol, side, quantity, price):
    """Place order on Binance (simulation)."""
    try:
        logging.info(f"Simulated {side} order: {quantity} {symbol} @ {price}")
        # Uncomment for real trading:
        # order = client.create_order(
        #     symbol=symbol,
        #     side=side,
        #     type=Client.ORDER_TYPE_LIMIT,
        #     timeInForce=Client.TIME_IN_FORCE_GTC,
        #     quantity=quantity,
        #     price=str(price)
        # )
        # return order
    except Exception as e:
        logging.error(f"Error placing order: {e}")
        raise

async def main():
    """Main trading loop."""
    logger = setup_logging()
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1MINUTE
    look_back = 60
    lookback = 500
    account_balance = 1000
    
    try:
        # Initialize JSON log
        initialize_json_log()

        # Fetch and process data once before the loop
        df = await fetch_historical_data(symbol, interval, lookback)
        df = calculate_technical_indicators(df, look_back)
        # Fetch initial sentiment to prepare features for the first training run
        initial_avg_sentiment = get_average_sentiment(symbol)
        logger.info(f"Initial average sentiment for {symbol}: {initial_avg_sentiment}")
        features = prepare_features(df, initial_avg_sentiment)
        
        if len(features) < look_back:
            logger.warning("Insufficient data for analysis")
            return
        
        # Prepare data for both models
        X_lstm, y_lstm, close_scaler, _ = prepare_lstm_data(features, look_back)
        X_rf, y_rf = prepare_rf_data(features, look_back)
        
        # Train LSTM model
        lstm_model = create_lstm_model(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
        lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0)
        
        # Train Random Forest model
        rf_model = create_rf_model()
        rf_model.fit(X_rf[:-1], y_rf[:-1])
        
        while True:
            try:
                # Fetch and process new data
                df = await fetch_historical_data(symbol, interval, lookback)
                df = calculate_technical_indicators(df, look_back)
                
                # Get current sentiment score
                avg_sentiment = get_average_sentiment(symbol)
                logger.info(f"Average sentiment for {symbol}: {avg_sentiment}")
                
                features = prepare_features(df, avg_sentiment)
                
                if len(features) < look_back:
                    logger.warning("Insufficient data for analysis")
                    await asyncio.sleep(10)
                    continue
                
                # Prepare data for both models
                X_lstm, y_lstm, _, _ = prepare_lstm_data(features, look_back)
                X_rf, y_rf = prepare_rf_data(features, look_back)
                
                # Predict with LSTM
                lstm_pred = lstm_model.predict(X_lstm[-1:])
                lstm_pred = close_scaler.inverse_transform(lstm_pred.reshape(-1, 1))[0][0]
                
                # Predict with Random Forest
                rf_pred = rf_model.predict(X_rf[-1:])[-1]

                # Generate trading signals
                current_price = df['close'].iloc[-1]
                final_prediction = ensemble_predict(lstm_pred, rf_pred) # This is now the direct prediction
                
                price_change = (final_prediction - current_price) / current_price
                
                # Trading logic
                if price_change > TAKE_PROFIT:
                    action = "BUY"
                elif price_change < -STOP_LOSS:
                    action = "SELL"
                else:
                    action = "HOLD"
                
                # Log trade-related predictions and decisions
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'current_price': float(current_price),
                    'lstm_prediction': float(lstm_pred),
                    'rf_prediction': float(rf_pred),
                    'ensemble_prediction': float(final_prediction), # Prediction used for action
                    'average_sentiment': float(avg_sentiment),     # Logged for observability
                    'action': action
                }
                # Ensure log_trade_to_sqlite is updated if schema changed
                # For now, assuming it can handle extra keys or they are ignored.
                # If schema is strict, this might need adjustment or conditional logging.
                # The previous version of log_trade_to_sqlite only inserted specific keys.
                # Let's assume the dict is passed and the function handles what it needs.
                log_trade_to_sqlite(log_data) 
                log_trade_to_json(log_data)
                log_trade_to_text(log_data)
                
                # Execute trades
                if action != "HOLD":
                    position_size = calculate_position_size(account_balance, current_price)
                    await place_order(symbol, action, position_size, current_price)
                
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                logger.error(f"Error details: {str(e)}")
                await asyncio.sleep(10)
                
    except KeyboardInterrupt:
        logger.info("Bot stopped manually")
    finally:
        logger.info("Cleaning up resources...")

if __name__ == "__main__":
    asyncio.run(main())