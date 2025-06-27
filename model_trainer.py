# model_trainer.py
import yfinance as yf
import pandas as pd
import numpy as np
from stock_analysis import calculate_sma5, calculate_sma20, calculate_macd, calculate_bollinger_bands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_features(df):
    df = df.copy()
    df['sma5'] = calculate_sma5(df)
    df['sma20'] = calculate_sma20(df)
    df['sma_diff'] = df['sma5'] - df['sma20']
    dif, macd, _ = calculate_macd(df)
    df['dif'] = dif
    df['macd'] = macd
    df['macd_diff'] = df['dif'] - df['macd']
    sma_boll, upper, lower = calculate_bollinger_bands(df)
    df['boll_width'] = upper - lower
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['return'].rolling(window=5).std()
    return df.dropna()

def create_label(df, future_days=5):
    df = df.copy()
    df['future_price'] = df['Close'].shift(-future_days)
    df['label'] = (df['future_price'] > df['Close']).astype(int)  # 漲為1，跌為0
    return df.dropna()

def train_model(symbol='AAPL', period='6mo'):
    print(f"訓練股票模型：{symbol}")
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    df = extract_features(df)
    df = create_label(df)
    
    feature_cols = ['sma_diff', 'macd_diff', 'boll_width', 'volatility']
    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("模型評估報告：")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f'models/{symbol}_trend_model.pkl')
    return model

train_model('AAPL')  