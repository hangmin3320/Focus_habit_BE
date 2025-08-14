import os
import torch
import collections
import numpy as np
from datetime import datetime
from models.lstm_action_classifier import LSTMClassifier
from services.media_pipe_runner import MediaPipeRunner
from services.modules.eye_analyzer import EyeAnalyzer
from services.modules.head_pose_analyzer import HeadPoseAnalyzer
import cv2 # Added for cv2.cvtColor

SEQUENCE_LENGTH = 30
INPUT_DIM = 10
HIDDEN_DIM = 64
NUM_LAYERS = 2
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/checkpoints/best_lstm_model.pth')

class AnalysisService:
    """
    실시간 스트림 데이터를 받아 집중도를 분석하고, 세션 기록을 관리합니다.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self._load_model()
        
        # Initialize MediaPipe and analysis modules
        self.media_pipe_runner = MediaPipeRunner()
        self.eye_analyzer = EyeAnalyzer()
        self.head_pose_analyzer = HeadPoseAnalyzer()

        # 실시간 프레임 데이터를 저장할 버퍼 (고정 길이 큐)
        self.frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        print(f"AnalysisService __init__ - frame_buffer initialized with length: {len(self.frame_buffer)}") # NEW PRINT
        # 예측 결과 (타임스탬프, 예측값, 신뢰도)를 저장할 리스트
        self.prediction_history = []

    def clear_buffer(self):
        """ 버퍼와 예측 기록을 초기화합니다. """
        self.frame_buffer.clear()
        self.prediction_history.clear()
        print("AnalysisService buffer and prediction history cleared.")

    def close(self):
        """ MediaPipe 리소스를 해제합니다. """
        if self.media_pipe_runner:
            self.media_pipe_runner.close()
            print("AnalysisService MediaPipe resources closed.")

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

    async def analyze_frame(self, frame: np.ndarray):
        """
        단일 프레임을 분석하여 눈 상태, 머리 자세 등을 추출하고,
        시계열 분석을 위해 버퍼에 추가합니다.
        """
        frame_data = {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "eye_status": {}, # Default empty dict
            "head_pose": {}   # Default empty dict
        }
        try:
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error converting frame to RGB: {e}")
            return {"error": f"Frame conversion error: {e}"}
        
        try:
            # Get face landmarks
            face_landmarks = self.media_pipe_runner.get_face_landmarks(frame_rgb)
        except Exception as e:
            print(f"Error getting face landmarks from MediaPipe: {e}")
            return {"error": f"MediaPipe error: {e}"}

        if face_landmarks:
            try:
                # Analyze eye state and head pose
                eye_status = self.eye_analyzer.analyze_frame(face_landmarks)
                head_pose = self.head_pose_analyzer.analyze_frame(face_landmarks, frame.shape[:2]) # Pass only height and width

                # Combine results into frame_data
                frame_data["eye_status"] = eye_status
                frame_data["head_pose"] = head_pose
            except Exception as e:
                print(f"Error during eye/head pose analysis: {e}")
                return {"error": f"Analysis module error: {e}"}
            
            try:
                # Add to buffer and get prediction if buffer is full
                prediction_result = self.add_new_frame(frame_data)
                
                # Return combined result, including prediction if available
                if prediction_result:
                    frame_data["prediction_result"] = prediction_result
                return frame_data
            except Exception as e:
                print(f"Error during adding frame to buffer or prediction: {e}")
                return {"error": f"Prediction error: {e}"}
        else:
            # If no face detected, return frame_data with empty eye_status and head_pose
            print("No face detected in frame.")
            return frame_data

    def add_new_frame(self, frame_data):
        """ 새로운 프레임 데이터를 버퍼에 추가하고, 버퍼가 차면 예측을 수행합니다. """
        print(f"Before append - Buffer length: {len(self.frame_buffer)}")
        self.frame_buffer.append(frame_data)
        print(f"After append - Buffer length: {len(self.frame_buffer)}")

        if len(self.frame_buffer) == SEQUENCE_LENGTH:
            print("Buffer is full. Calling _predict.")
            sequence_to_predict = list(self.frame_buffer)
            prediction_result = self._predict(sequence_to_predict)
            print(f"add_new_frame returning prediction: {prediction_result is not None}")
            return prediction_result
        print("Buffer not full. add_new_frame returning None.")
        return None

    def _feature_engineer_sequence(self, sequence):
        """ 
        하나의 시퀀스(30개 프레임)를 받아 Feature Engineering을 수행하고 
        모델에 입력할 텐서로 변환합니다.
        """
        engineered_features = []

        first_frame = sequence[0]
        ear = first_frame.get('eye_status', {}).get('ear_value', 0.0)
        pitch = first_frame.get('head_pose', {}).get('pitch', 0.0)
        yaw = first_frame.get('head_pose', {}).get('yaw', 0.0)
        roll = first_frame.get('head_pose', {}).get('roll', 0.0)
        is_open = 1.0 if first_frame.get('eye_status', {}).get('status') == 'OPEN' else 0.0
        is_closed = 1.0 if first_frame.get('eye_status', {}).get('status') == 'CLOSED' else 0.0
        first_feature_vector = [ear, pitch, yaw, roll, 0.0, 0.0, 0.0, 0.0, is_open, is_closed]
        engineered_features.append(first_feature_vector)

        for i in range(1, len(sequence)):
            current_frame = sequence[i]
            prev_frame = sequence[i-1]

            ear = current_frame.get('eye_status', {}).get('ear_value', 0.0)
            pitch = current_frame.get('head_pose', {}).get('pitch', 0.0)
            yaw = current_frame.get('head_pose', {}).get('yaw', 0.0)
            roll = current_frame.get('head_pose', {}).get('roll', 0.0)

            ear_vel = ear - prev_frame.get('eye_status', {}).get('ear_value', 0.0)
            pitch_vel = pitch - prev_frame.get('head_pose', {}).get('pitch', 0.0)
            yaw_vel = yaw - prev_frame.get('head_pose', {}).get('yaw', 0.0)
            roll_vel = roll - prev_frame.get('head_pose', {}).get('roll', 0.0)

            is_open = 1.0 if current_frame.get('eye_status', {}).get('status') == 'OPEN' else 0.0
            is_closed = 1.0 if current_frame.get('eye_status', {}).get('status') == 'CLOSED' else 0.0

            feature_vector = [
                ear, pitch, yaw, roll,
                ear_vel, pitch_vel, yaw_vel, roll_vel,
                is_open, is_closed
            ]
            engineered_features.append(feature_vector)
        
        # Check for NaN or Inf values before creating tensor
        final_features = np.array(engineered_features, dtype=np.float32)
        if np.any(np.isnan(final_features)) or np.any(np.isinf(final_features)):
            print("Warning: NaN or Inf values detected in engineered features. Returning error.")
            # You might want to handle this more gracefully, e.g., by returning a default tensor or raising a specific error
            return torch.zeros((1, SEQUENCE_LENGTH, INPUT_DIM), dtype=torch.float32).to(self.device) # Return a dummy tensor

        return torch.tensor([engineered_features], dtype=torch.float32).to(self.device)

    def _predict(self, sequence):
        """ 시퀀스 데이터를 받아 모델로 예측하고, 결과를 저장 및 반환합니다. """
        if not self.model:
            return {"error": "Model is not loaded."}
        
        feature_tensor = self._feature_engineer_sequence(sequence)

        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

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

        first_frame = sequence[0]
        ear = first_frame.get('eye_status', {}).get('ear_value', 0)
        pitch = first_frame.get('head_pose', {}).get('pitch', 0)
        yaw = first_frame.get('head_pose', {}).get('yaw', 0)
        roll = first_frame.get('head_pose', {}).get('roll', 0)
        is_open = 1 if first_frame.get('eye_status', {}).get('status') == 'OPEN' else 0
        is_closed = 1 if first_frame.get('eye_status', {}).get('status') == 'CLOSED' else 0
        first_feature_vector = [ear, pitch, yaw, roll, 0, 0, 0, 0, is_open, is_closed]
        engineered_features.append(first_feature_vector)

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