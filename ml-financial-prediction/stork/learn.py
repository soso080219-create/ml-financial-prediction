# 필요한 라이브러리를 먼저 설치하세요: pip install yfinance scikit-learn pandas tensorflow

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model

# 예측에 사용할 기업 정보와 파일 이름
COMPANIES = {
    '삼성전자': {'ticker': '006260.KS', 'model_file': 'Samsung_model.h5'},
    '현대모비스': {'ticker': '012330.KS', 'model_file': 'Hyundaemobis_model.h5'},
    '카카오페이': {'ticker': '377300.KS', 'model_file': 'Kakaopay_model.h5'},
    'POSCO': {'ticker': '005490.KS', 'model_file': 'Posco_model.h5'},
    'SK이노베이션': {'ticker': '096770.KS', 'model_file': 'SKinno_model.h5'}
}

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

# --- 모든 기업에 대해 모델 학습 및 저장 ---
for company_name, info in COMPANIES.items():
    print(f"\n--- {company_name} 모델 학습 시작 ---")
    ticker = info['ticker']

    try:
        # yfinance를 사용하여 1년치 데이터 가져오기
        df = yf.download(ticker, period="1y")

        if df.empty:
            print(f"오류: {company_name} ({ticker})에 대한 데이터를 가져올 수 없습니다.")
            continue

        # 보조 지표 계산
        df_indicators = calculate_technical_indicators(df)

        # 예측 대상 (레이블) 생성 - 다음 날 종가가 오늘 종가보다 높으면 1 (상승), 아니면 0 (하락)
        df_indicators['Label'] = (df_indicators['Close'].shift(-1) > df_indicators['Close']).astype(int)

        # NaN 값 제거
        df_indicators.dropna(inplace=True)

        # 훈련 데이터셋 구성
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_10', 'SMA_20', 'RSI', 
                    'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BBL', 'BBM', 'BBU', 'OBV']
        
        X = df_indicators[features]
        y = df_indicators['Label']
        
        if X.empty:
            print(f"경고: 지표 계산 후 유효한 데이터가 충분하지 않습니다.")
            continue

        # 데이터 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 훈련 및 테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # 모델 구축 (Keras Sequential model)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))  # 출력층에 sigmoid 활성화 함수 사용

        # 모델 컴파일
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # 모델 학습
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        
        # 모델 및 스케일러 저장
        save_model(model, info['model_file'])
        # 스케일러는 app.py에서 다시 fit할 수 있으므로 저장하지 않음.
        # 만약 고정된 스케일러를 사용하려면 joblib으로 저장해야 함.
        # 예시: dump(scaler, 'LS_scaler.pkl')

        print(f"--- {company_name} 모델 학습 및 '{info['model_file']}' 파일 저장 완료 ---")

    except Exception as e:
        print(f"오류: {company_name} 모델 학습 중 문제가 발생했습니다: {e}")

print("\n모든 모델 학습이 완료되었습니다.")
