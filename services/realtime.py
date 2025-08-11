import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import cv2
import time
import pickle
import os
from pathlib import Path
from services.modules.eye_analyzer import EyeAnalyzer
from services.modules.head_pose_analyzer import HeadPoseAnalyzer
from services.analysis_service import AnalysisService

# 1. 1D CNN 모델 정의
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, window_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (window_size // 4), 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.clamp(x, -100, 100)

# 2. 실시간 피처 엔지니어링
def preprocess_realtime_data(df, features, scaler):
    base_features = ['ear', 'pitch', 'yaw', 'roll']
    
    # 피처 엔지니어링
    for feature in base_features:
        df[f'{feature}_diff'] = df[feature].diff().fillna(0)
        df[f'{feature}_mean_5'] = df[feature].rolling(window=5, min_periods=1).mean()
        df[f'{feature}_std_5'] = df[feature].rolling(window=5, min_periods=1).std().fillna(0)
    
    df['eye_status_numeric'] = df['eye_status'].map({'OPEN': 1, 'CLOSED': 0})
    df['blink_count'] = df['eye_status_numeric'].diff().eq(-1).rolling(window=5, min_periods=1).sum().fillna(0)
    df['angle_magnitude'] = np.sqrt(df['pitch_diff']**2 + df['yaw_diff']**2 + df['roll_diff']**2)
    
    # NaN 및 inf 처리
    df[features] = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 정규화
    df[features] = scaler.transform(df[features])
    
    return df

# 3. 시퀀스 생성
def create_realtime_sequence(df, features, window_size=10):
    if len(df) < window_size:
        return None
    data = df[features].values[-window_size:]  # 마지막 window_size 프레임
    return np.array([data])

# 4. 실시간 예측 함수
def predict_realtime(model, sequence, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(sequence, dtype=torch.float32).to(device)
        outputs = model(inputs).squeeze()
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        return predicted.cpu().numpy()

# 5. 메인 실시간 예측
def main_realtime():
    # 설정
    project_root = Path(__file__).parent.parent
    model_path = str(project_root / 'models' / 'best_cnn_model.pth')
    scaler_path = str(project_root / 'models' / 'scaler.pkl')
    window_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = ['ear', 'pitch', 'yaw', 'roll',
                'ear_diff', 'pitch_diff', 'yaw_diff', 'roll_diff',
                'ear_mean_5', 'pitch_mean_5', 'yaw_mean_5', 'roll_mean_5',
                'ear_std_5', 'pitch_std_5', 'yaw_std_5', 'roll_std_5',
                'blink_count', 'angle_magnitude']
    
    # 모델 로드
    model = TimeSeriesCNN(input_channels=len(features), window_size=window_size).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # 스케일러 로드
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
    
    # AnalysisService 초기화
    analysis_service = AnalysisService()
    if not (analysis_service.face_landmarker):
        raise ValueError("Failed to initialize MediaPipe Face Landmarker")
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Failed to open webcam")
    
    # 데이터프레임 초기화
    columns = ['timestamp_ms', 'eye_status', 'ear', 'pitch', 'yaw', 'roll']
    df = pd.DataFrame(columns=columns)
    
    print("Starting real-time prediction. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # 프레임에서 피처 추출
            timestamp_ms = int(time.time() * 1000)
            analysis_result = analysis_service.analyze_image(frame, timestamp_ms)
            
            # 피처 추출
            eye_status = analysis_result['eye_status']['status']
            ear = analysis_result['eye_status']['ear_value']
            pitch = analysis_result['head_pose']['pitch']
            yaw = analysis_result['head_pose']['yaw']
            roll = analysis_result['head_pose']['roll']
            
            # 새 데이터 추가
            new_data = pd.DataFrame({
                'timestamp_ms': [timestamp_ms],
                'eye_status': [eye_status],
                'ear': [ear],
                'pitch': [pitch],
                'yaw': [yaw],
                'roll': [roll]
            })
            df = pd.concat([df, new_data], ignore_index=True)
            
            # 피처 엔지니어링 및 정규화
            df_processed = preprocess_realtime_data(df, features, scaler)
            
            # 시퀀스 생성
            sequence = create_realtime_sequence(df_processed, features, window_size)
            
            if sequence is not None:
                # 예측
                prediction = predict_realtime(model, sequence, device)
                label = 'Class 1' if prediction[0] == 1 else 'Class 0'
                
                # 결과 출력
                print(f"Prediction: {label}")
                
                # 프레임에 예측 결과 표시
                cv2.putText(frame, f"Prediction: {label}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 프레임 표시
            cv2.imshow('Webcam', frame)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 메모리 관리: 오래된 데이터 제거
            if len(df) > window_size * 2:
                df = df.tail(window_size * 2)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")

if __name__ == "__main__":
    main_realtime()