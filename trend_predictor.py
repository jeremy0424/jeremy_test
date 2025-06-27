# trend_predictor.py
import joblib
import yfinance as yf
import pandas as pd
from stock_analysis import calculate_sma5, calculate_sma20, calculate_macd, calculate_bollinger_bands
from datetime import timedelta

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

def predict_next_5_days(symbol):
    try:
        model = joblib.load('models/stock_trend_model.pkl')
    except:
        return ["尚未訓練模型"]

    df = yf.Ticker(symbol).history(period='90d')
    if df.empty:
        return [f"{symbol} 資料為空，無法預測"]

    df = extract_features(df)
    if df.empty:
        return [f"{symbol} 技術指標處理後無資料，可能資料不足"]

    last_date = df.index[-1]
    predictions = []
    current_df = df.copy()

    for i in range(1, 6):
        try:
            last_row = current_df.iloc[-1]
            features = last_row[['sma_diff', 'macd_diff', 'boll_width', 'volatility']].values.reshape(1, -1)
            pred = model.predict(features)[0]
            pred_text = "📈 上漲" if pred == 1 else "📉 下跌"
            future_date = last_date + timedelta(days=i)
            predictions.append(f"{future_date.strftime('%Y-%m-%d')}：{pred_text}")

            # 模擬新資料僅保留必要欄位，避免 extract_features 導致清空
            feature_row = pd.DataFrame([{
                'sma_diff': features[0][0],
                'macd_diff': features[0][1],
                'boll_width': features[0][2],
                'volatility': features[0][3],
                'Close': last_row['Close'] * (1.01 if pred == 1 else 0.99)
            }], index=[future_date])

            current_df = pd.concat([current_df, feature_row])

        except Exception as e:
            predictions.append(f"第 {i} 天預測失敗：{e}")

    return predictions
def evaluate_model_accuracy(symbol='AAPL', future_days=5):
    model = joblib.load('models/stock_trend_model.pkl')
    df = yf.Ticker(symbol).history(period='6mo')
    df = extract_features(df)
    df = df.dropna()

    feature_cols = ['sma_diff', 'macd_diff', 'boll_width', 'volatility']
    results = []

    for i in range(len(df) - future_days):
        row = df.iloc[i]
        features = row[feature_cols].values.reshape(1, -1)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]
        future_price = df.iloc[i + future_days]['Close']
        actual = 1 if future_price > row['Close'] else 0
        is_correct = pred == actual

        results.append({
            '日期': df.index[i].strftime('%Y-%m-%d'),
            '預測': '上漲' if pred == 1 else '下跌',
            '機率': round(prob * 100, 1),
            '實際': '上漲' if actual == 1 else '下跌',
            '結果': '✅ 正確' if is_correct else '❌ 錯誤'
        })

    results_df = pd.DataFrame(results)
    acc = round((results_df['結果'] == '✅ 正確').mean() * 100, 2)
    print(f"\n📊 預測準確率：{acc}%")
    return results_df

