import yfinance as yf
import pandas as pd
import numpy as np
from stock_analysis import calculate_sma5, calculate_sma20, calculate_macd, calculate_bollinger_bands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def extract_features(df
                     
                     ):
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
    df['label'] = (df['future_price'] > df['Close']).astype(int)
    return df.dropna()

def download_and_process(symbol, period='6mo'):
    df = yf.Ticker(symbol).history(period=period)
    df = extract_features(df)
    df = create_label(df)
    df['symbol'] = symbol
    return df

def train_multi_stock_model(stock_list, period='6mo'):
    all_data = []
    for symbol in stock_list:
        try:
            df = download_and_process(symbol, period)
            all_data.append(df)
            print(f"成功加入：{symbol}, 筆數={len(df)}")
        except Exception as e:
            print(f"⚠️ 錯誤：{symbol} → {e}")

    combined_df = pd.concat(all_data, ignore_index=True)
    feature_cols = ['sma_diff', 'macd_diff', 'boll_width', 'volatility']
    X = combined_df[feature_cols]
    y = combined_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📈 模型評估：")
    print(classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stock_trend_model.pkl')
    print("✅ 模型儲存完成：models/stock_trend_model.pkl")

if __name__ == '__main__':
    stock_list = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'NVDA', 'AMZN', '2330.TW']  # 可自定義
    train_multi_stock_model(stock_list)
