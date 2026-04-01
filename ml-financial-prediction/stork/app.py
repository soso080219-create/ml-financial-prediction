# 필요한 라이브러리를 먼저 설치하세요: pip install Flask joblib numpy yfinance tensorflow

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 예측에 사용할 기업 정보
COMPANIES = {
    'LS': {'ticker': '006260.KS', 'model_file': r'C:/2025/minipro/stork/LS_model.h5'},
    'LG': {'ticker': '003550.KS', 'model_file': r'C:/2025/minipro/stork/LG_model.h5'},
    '한화': {'ticker': '000880.KS', 'model_file': r'C:/2025/minipro/stork/Hanwha_model.h5'},
    'KT&G': {'ticker': '033780.KS', 'model_file': r'C:/2025/minipro/stork/KT&G_model.h5'},
    '신한지주': {'ticker': '055550.KS', 'model_file': r'C:/2025/minipro/stork/Shinhan_Financial_Group_model.h5'},
    '삼성전자': {'ticker': '006260.KS', 'model_file': r'C:/2025/minipro/stork/Samsung_model.h5'},
    '현대모비스': {'ticker': '012330.KS', 'model_file': r'C:/2025/minipro/stork/Hyundaemobis_model.h5'},
    '카카오페이': {'ticker': '377300.KS', 'model_file': r'C:/2025/minipro/stork/Kakaopay_model.h5'},
    'POSCO': {'ticker': '005490.KS', 'model_file': r'C:/2025/minipro/stork/Posco_model.h5'},
    'SK이노베이션': {'ticker': '096770.KS', 'model_file': r'C:/2025/minipro/stork/SKinno_model.h5'}
}

# 모든 모델을 로드하여 저장할 딕셔너리
MODELS = {}

# 서버 시작 시 모든 모델 로드
for company_name, info in COMPANIES.items():
    model_path = info['model_file']
    try:
        MODELS[company_name] = load_model(model_path)
        print(f"모델 '{model_path}'이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"경고: '{model_path}' 파일을 찾을 수 없거나 로드에 실패했습니다. model_trainer.py를 사용하여 해당 모델을 먼저 생성하세요. 오류: {e}")
        MODELS[company_name] = None

# 보조 지표 계산 함수
def calculate_technical_indicators(data):
    # 이동평균선 (SMA) - 10일, 20일
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # RSI (상대강도지수)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain.fillna(0) / loss.fillna(0)
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # MACD (이동평균 수렴/확산 지수)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd_series = exp1 - exp2
    data['MACD'] = macd_series
    data['MACD_Signal'] = macd_series.ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # BB (볼린저 밴드)
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    
    data['BBM'] = rolling_mean
    data['BBU'] = rolling_mean + 2 * rolling_std
    data['BBL'] = rolling_mean - 2 * rolling_std

    # OBV (On-Balance Volume)
    obv_data = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['OBV'] = obv_data

    return data

@app.route('/')
def home():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/predict_today', methods=['POST'])
def predict_today():
    """최신 yfinance 데이터를 사용하여 예측을 처리합니다."""
    try:
        data = request.json
        company_name = data.get('company')
        
        if not company_name or company_name not in COMPANIES:
            return jsonify({'error': '유효하지 않은 기업 이름이 제공되었습니다.'}), 400

        model = MODELS.get(company_name)
        if model is None:
            return jsonify({'error': f"'{company_name}'에 대한 모델이 로드되지 않았습니다. 모델 파일을 확인하세요."}), 500

        ticker = COMPANIES[company_name]['ticker']

        # yfinance를 사용하여 최근 30일 데이터를 가져옵니다.
        df = yf.download(ticker, period='30d')
        
        if df.empty:
            return jsonify({'error': f"티커 '{ticker}'에 대한 데이터를 찾을 수 없습니다."}), 500
        
        # 보조 지표를 계산합니다.
        df_processed = calculate_technical_indicators(df)
        
        # NaN 값이 있는 행을 제거합니다.
        df_processed.dropna(inplace=True)
        
        if df_processed.empty:
             return jsonify({'error': "지표 계산 후 예측 가능한 데이터가 없습니다."}), 500

        # 모델 학습에 사용된 특성과 동일한 순서로 데이터를 준비합니다.
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_10', 'SMA_20', 'RSI', 
                    'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BBL', 'BBM', 'BBU', 'OBV']
        
        # 마지막 행의 데이터를 가져와서 모델 입력 형식에 맞게 변환합니다.
        input_data = df_processed.iloc[-1][features].values.reshape(1, -1)
        
        # 모델 학습에 사용된 데이터와 동일하게 스케일링
        # 이전에 사용된 모든 데이터를 사용하여 스케일러를 다시 fit합니다.
        scaler = StandardScaler()
        scaler.fit(df_processed[features])
        input_scaled = scaler.transform(input_data)

        # Keras 모델의 predict() 함수를 사용하여 확률 예측
        prediction_raw = model.predict(input_scaled)[0][0]
        
        # 예측 확률을 기반으로 최종 예측 결정
        prediction = 1 if prediction_raw >= 0.5 else 0
        
        # 결과를 퍼센트로 변환
        up_proba = round(prediction_raw * 100, 2)
        down_proba = round((1 - prediction_raw) * 100, 2)
        
        # 예측 결과를 반환합니다.
        result = "상승" if prediction == 1 else "하락"
        
        return jsonify({
            'prediction': result,
            'up_proba': up_proba,
            'down_proba': down_proba
        })
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
