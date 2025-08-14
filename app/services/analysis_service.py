import os
import torch
import collections
import numpy as np
from models.lstm_action_classifier import LSTMClassifier

# --- 상수 및 설정 ---
SEQUENCE_LENGTH = 30
INPUT_DIM = 10
HIDDEN_DIM = 64
NUM_LAYERS = 2
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/checkpoints/best_lstm_model.pth')

class FocusAnalysisService:
    """
    실시간 스트림 데이터를 받아 집중도를 분석하고, 세션 기록을 관리합니다.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self._load_model()
        
        # 실시간 프레임 데이터를 저장할 버퍼 (고정 길이 큐)
        self.frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        # 예측 결과 (타임스탬프, 예측값, 신뢰도)를 저장할 리스트
        self.prediction_history = []

    def _load_model(self):
        """ 학습된 PyTorch 모델(.pth)을 로드합니다. """
        try:
            model = LSTMClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(self.device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model.eval()  # 모델을 평가 모드로 설정
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_PATH}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None

    def add_new_frame(self, frame_data):
        """ 새로운 프레임 데이터를 버퍼에 추가하고, 버퍼가 차면 예측을 수행합니다. """
        self.frame_buffer.append(frame_data)

        if len(self.frame_buffer) == SEQUENCE_LENGTH:
            # 버퍼가 가득 차면, 시퀀스를 만들어 예측 실행
            sequence_to_predict = list(self.frame_buffer)
            return self._predict(sequence_to_predict)
        return None

    def _predict(self, sequence):
        """ 시퀀스 데이터를 받아 모델로 예측하고, 결과를 저장 및 반환합니다. """
        if not self.model:
            return {"error": "Model is not loaded."}

        # 1. Feature Engineering (TODO)
        # 입력된 sequence(raw data)를 모델 입력에 맞는 텐서로 변환해야 합니다.
        # 이 부분은 data_service.py의 로직과 동일하게 구현되어야 합니다.
        feature_tensor = self._feature_engineer_sequence(sequence)

        # 2. 모델 추론
        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # 3. 결과 저장 및 반환
        result = {
            "timestamp": sequence[-1]['timestamp'], # 마지막 프레임의 타임스탬프
            "prediction": 'focused' if predicted_class.item() == 1 else 'unfocused',
            "confidence": confidence.item()
        }
        self.prediction_history.append(result)
        return result

    def _feature_engineer_sequence(self, sequence):
        """ 
        하나의 시퀀스(30개 프레임)를 받아 Feature Engineering을 수행하고 
        모델에 입력할 텐서로 변환합니다.
        """
        engineered_features = []

        # 첫 프레임의 특징 벡터 (속도는 0으로 초기화)
        first_frame = sequence[0]
        ear = first_frame.get('eye_status', {}).get('ear_value', 0)
        pitch = first_frame.get('head_pose', {}).get('pitch', 0)
        yaw = first_frame.get('head_pose', {}).get('yaw', 0)
        roll = first_frame.get('head_pose', {}).get('roll', 0)
        is_open = 1 if first_frame.get('eye_status', {}).get('status') == 'OPEN' else 0
        is_closed = 1 if first_frame.get('eye_status', {}).get('status') == 'CLOSED' else 0
        first_feature_vector = [ear, pitch, yaw, roll, 0, 0, 0, 0, is_open, is_closed]
        engineered_features.append(first_feature_vector)

        # 두 번째 프레임부터 속도 계산
        for i in range(1, len(sequence)):
            current_frame = sequence[i]
            prev_frame = sequence[i-1]

            ear = current_frame.get('eye_status', {}).get('ear_value', 0)
            pitch = current_frame.get('head_pose', {}).get('pitch', 0)
            yaw = current_frame.get('head_pose', {}).get('yaw', 0)
            roll = current_frame.get('head_pose', {}).get('roll', 0)

            ear_vel = ear - prev_frame.get('eye_status', {}).get('ear_value', 0)
            pitch_vel = pitch - prev_frame.get('head_pose', {}).get('pitch', 0)
            yaw_vel = yaw - prev_frame.get('head_pose', {}).get('yaw', 0)
            roll_vel = roll - prev_frame.get('head_pose', {}).get('roll', 0)

            is_open = 1 if current_frame.get('eye_status', {}).get('status') == 'OPEN' else 0
            is_closed = 1 if current_frame.get('eye_status', {}).get('status') == 'CLOSED' else 0

            feature_vector = [
                ear, pitch, yaw, roll,
                ear_vel, pitch_vel, yaw_vel, roll_vel,
                is_open, is_closed
            ]
            engineered_features.append(feature_vector)
        
        return torch.tensor([engineered_features], dtype=torch.float32).to(self.device)

    def get_overall_analysis(self):
        """ 세션 전체에 대한 분석 결과를 반환합니다. """
        if not self.prediction_history:
            return {"error": "No predictions available.", "overall_focus_score": 0, "history": []}
        
        # 'focused' 예측의 신뢰도 평균을 내어 전체 집중도 점수 계산
        focus_confidences = [item['confidence'] for item in self.prediction_history if item['prediction'] == 'focused']
        
        # focused 예측이 한 번도 없었을 경우
        if not focus_confidences:
            focus_score = 0.0
        else:
            # 전체 예측 중 focused 예측이 차지하는 비율을 가중치로 적용
            focus_ratio = len(focus_confidences) / len(self.prediction_history)
            avg_focus_confidence = sum(focus_confidences) / len(focus_confidences)
            focus_score = avg_focus_confidence * focus_ratio * 100

        return {
            "total_predictions": len(self.prediction_history),
            "overall_focus_score": round(focus_score, 2),
            "history": self.prediction_history
        }