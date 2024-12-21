import os
import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
from dotenv import load_dotenv
import logging
import time
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Binance API credentials
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Binance Client
client = Client(API_KEY, API_SECRET, testnet=True)

# Logging setup
logging.basicConfig(filename='trading_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_historical_data(symbol, interval, lookback):
    """Fetch historical data from Binance."""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'trades',
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df[['time', 'close']]

def preprocess_data(data, look_back=60):
    """Preprocess data for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build the LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_price(model, scaler, data, look_back=60):
    """Predict the next price."""
    last_sequence = data[-look_back:].reshape(-1, 1)
    scaled_sequence = scaler.transform(last_sequence)
    X = np.array([scaled_sequence]).reshape(1, look_back, 1)
    predicted_scaled_price = model.predict(X, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)
    return predicted_price[0][0]

def backtest(data, look_back=60):
    """Run backtesting on historical data."""
    X, y, scaler = preprocess_data(data, look_back)
    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    predictions, actual_prices = [], data['close'][look_back:].values
    for i in range(look_back, len(data)):
        predicted_price = predict_next_price(model, scaler, data['close'][:i].values, look_back)
        predictions.append(predicted_price)
    results = pd.DataFrame({'Actual': actual_prices, 'Predicted': predictions})
    # In bot.py, call backtest() for a specific set of data
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1HOUR
    lookback = 500
    data = fetch_historical_data(symbol, interval, lookback)
    results = backtest(data)

    # Visualize the results
    plt.plot(results['Actual'], label='Actual Prices')
    plt.plot(results['Predicted'], label='Predicted Prices')
    plt.legend()
    plt.show()
    return results

def main():
    """Main trading loop."""
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1MINUTE
    look_back, lookback = 60, 500
    while True:
        try:
            data = fetch_historical_data(symbol, interval, lookback)
            X, y, scaler = preprocess_data(data, look_back)
            model = build_lstm_model((X.shape[1], 1))
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            current_price = data['close'].iloc[-1]
            predicted_price = predict_next_price(model, scaler, data['close'].values, look_back)
            logging.info(f"Current Price: {current_price}, Predicted Price: {predicted_price}")
            time.sleep(60)  # Wait before the next prediction
        except KeyboardInterrupt:
            logging.info("Bot stopped manually.")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)


# main()

def test_preprocess_data():
    data = pd.DataFrame({'close': np.linspace(1, 100, 100)})
    X, y, scaler = preprocess_data(data)
    assert X.shape[1] == 60
    assert len(y) == len(data) - 60
    print("Preprocess test passed!")

def test_lstm_model():
    model = build_lstm_model((60, 1))
    assert model.output_shape == (None, 1)
    print("LSTM model test passed!")

def test_backtest():
    data = pd.DataFrame({'close': np.linspace(1, 100, 100)})
    results = backtest(data)
    assert 'Actual' in results.columns
    assert 'Predicted' in results.columns
    print("Backtest test passed!")

# # Run tests
# test_preprocess_data()
# test_lstm_model()
# test_backtest()
