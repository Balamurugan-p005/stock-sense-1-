import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional,
    BatchNormalization, Conv1D, MaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. FETCH STOCK DATA + TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def get_stock_data(stock_symbol, period="5y"):
    data = yf.download(stock_symbol, period=period, auto_adjust=True, progress=False)

    # ── Fix MultiLevel columns from yfinance ──
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove duplicate columns
    data = data.loc[:, ~data.columns.duplicated()]

    # Keep only OHLCV
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # ── Fix for Indian stocks — remove outliers ──
    # Remove rows where price changed more than 20% in one day (bad data)
    data = data[data['Close'].pct_change().abs() < 0.20]

    # Remove zero volume rows (market holiday artifacts)
    data = data[data['Volume'] > 0]

    data.dropna(inplace=True)

    # Ensure all columns are plain 1D float Series
    for col in data.columns:
        if isinstance(data[col], pd.DataFrame):
            data[col] = data[col].iloc[:, 0]
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(inplace=True)

    data = add_technical_indicators(data)
    data.dropna(inplace=True)
    return data


def add_technical_indicators(df):
    """
    Add technical indicators as extra input features to LSTM.
    All Series are squeezed to 1D to avoid MultiIndex assignment errors.
    """
    # Squeeze all base columns to plain 1D Series
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()

    # ── Trend Indicators ──────────────────────
    # Simple Moving Averages
    df['SMA_10']  = close.rolling(10).mean()
    df['SMA_20']  = close.rolling(20).mean()
    df['SMA_50']  = close.rolling(50).mean()

    # Exponential Moving Averages
    df['EMA_12']  = close.ewm(span=12, adjust=False).mean()
    df['EMA_26']  = close.ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df['MACD']        = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # ── Momentum Indicators ───────────────────
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs        = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Rate of Change
    df['ROC'] = close.pct_change(10) * 100

    # ── Volatility Indicators ─────────────────
    # Bollinger Bands
    sma20          = close.rolling(20).mean()
    std20          = close.rolling(20).std()
    bb_upper       = (sma20 + 2 * std20).squeeze()
    bb_lower       = (sma20 - 2 * std20).squeeze()
    bb_width       = (bb_upper - bb_lower).squeeze()
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['BB_Width'] = bb_width
    df['BB_Pos']   = ((close - bb_lower) / (bb_width + 1e-10)).squeeze()

    # ATR (Average True Range)
    tr1           = (high - low).squeeze()
    tr2           = (high - close.shift(1)).abs().squeeze()
    tr3           = (low  - close.shift(1)).abs().squeeze()
    df['ATR']     = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # ── Volume Indicators ─────────────────────
    # On-Balance Volume
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Volume Moving Average
    df['Vol_MA']    = vol.rolling(20).mean()
    df['Vol_Ratio'] = (vol / (df['Vol_MA'].squeeze() + 1e-10)).squeeze()

    # ── Price-derived features ────────────────
    df['Price_Range']  = ((high - low) / (close + 1e-10)).squeeze()
    df['Price_Change'] = close.pct_change().squeeze()
    df['Gap']          = ((df['Open'].squeeze() - close.shift(1)) / (close.shift(1) + 1e-10)).squeeze()

    return df


# ─────────────────────────────────────────────
# 2. BUILD SEQUENCES
# ─────────────────────────────────────────────
def build_sequences(scaled_data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])   # all features
        y.append(scaled_data[i, 0])                     # 0 = Close price
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 3. BUILD MODEL — CNN + Bidirectional LSTM
# ─────────────────────────────────────────────
def build_model(sequence_length, n_features):
    """
    Architecture:
    Conv1D      → extracts local patterns (e.g. 3-day candle patterns)
    Bidirectional LSTM x2 → learns both forward AND backward time dependencies
    BatchNorm   → stabilizes training, prevents overfitting
    Dropout     → regularization
    Dense       → final regression output
    """
    model = Sequential([

        # CNN layer — pattern extractor
        Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(sequence_length, n_features), padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Bidirectional LSTM 1 — learns trends in both directions
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        # Bidirectional LSTM 2 — deeper temporal learning
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.2),

        # Dense output layers
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)   # predicted next close price
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',          # Huber loss — less sensitive to outliers than MSE
        metrics=['mae']
    )
    return model


# ─────────────────────────────────────────────
# 4. TRAIN AND PREDICT
# ─────────────────────────────────────────────
def train_and_predict_lstm(stock_symbol, sequence_length=60):

    # ── Load data with indicators ─────────────
    data = get_stock_data(stock_symbol, period="3y")

    if data.empty or len(data) < sequence_length + 50:
        raise ValueError(
            f"Not enough data. Need at least {sequence_length + 50} trading days. "
            f"Try a longer period or different symbol."
        )

    # ── Feature columns ───────────────────────
    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI', 'ROC',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Pos',
        'ATR', 'OBV', 'Vol_MA', 'Vol_Ratio',
        'Price_Range', 'Price_Change', 'Gap'
    ]

    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in data.columns]
    feature_data = data[feature_cols].values
    n_features   = len(feature_cols)

    # ── Scale all features independently ──────
    scaler       = MinMaxScaler(feature_range=(0, 1))
    close_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled       = scaler.fit_transform(feature_data)
    close_scaler.fit(data[['Close']].values)

    # ── Build sequences ───────────────────────
    X, y = build_sequences(scaled, sequence_length)

    # ── Train / Validation / Test split ───────
    # 75% train | 10% validation | 15% test
    train_end = int(0.75 * len(X))
    val_end   = int(0.85 * len(X))

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    if len(X_test) < 10:
        raise ValueError(
            "Test set too small. Try increasing Historical Period to 3y or 5y, "
            "or reduce LSTM Sequence Length to 45."
        )

    # ── Build and train model ─────────────────
    model = build_model(sequence_length, n_features)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]

    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
        shuffle=False
    )

    # ── Evaluate on test set ──────────────────
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform Close price only
    y_pred   = close_scaler.inverse_transform(y_pred_scaled)
    y_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    # ── Compute metrics ───────────────────────
    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2   = r2_score(y_actual, y_pred)

    # Direction accuracy
    actual_dir = np.sign(np.diff(y_actual.flatten()))
    pred_dir   = np.sign(np.diff(y_pred.flatten()))
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-10))) * 100

    metrics = {
        "MAE"                    : round(float(mae),     4),
        "RMSE"                   : round(float(rmse),    4),
        "R2"                     : round(float(r2),      4),
        "Direction Accuracy (%)" : round(float(dir_acc), 2),
        "MAPE (%)"               : round(float(mape),    2),
    }

    # ── Predict next day ─────────────────────
    last_sequence  = scaled[-sequence_length:].reshape(1, sequence_length, n_features)
    next_scaled    = model.predict(last_sequence, verbose=0)
    predicted_price = close_scaler.inverse_transform(next_scaled)[0][0]

    return (
        round(float(predicted_price), 2),
        data,
        metrics,
        y_actual.flatten().tolist(),
        y_pred.flatten().tolist()
    )