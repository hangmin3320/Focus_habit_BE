import collections
import numpy as np
from datetime import datetime
# Simplified for debugging - removed unused imports

SEQUENCE_LENGTH = 30 # Keep this as it's used in deque
# Simplified for debugging - removed unused constants

class AnalysisService:
    """
    실시간 스트림 데이터를 받아 집중도를 분석하고, 세션 기록을 관리합니다.
    """
    def __init__(self):
        # Simplified for debugging
        print("AnalysisService __init__ - Simplified for debugging.")
        self.frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        print(f"AnalysisService __init__ - frame_buffer initialized with length: {len(self.frame_buffer)}") # NEW PRINT
        self.prediction_history = []
        # No MediaPipe or PyTorch initialization here for now

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

    

    async def analyze_frame(self, frame: np.ndarray):
        """
        단일 프레임을 분석하여 눈 상태, 머리 자세 등을 추출하고,
        시계열 분석을 위해 버퍼에 추가합니다. (Simplified for debugging)
        """
        frame_data = {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "eye_status": {"status": "SIMPLIFIED_OPEN", "ear_value": 0.3},
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        }
        
        # Add to buffer and get prediction if buffer is full
        prediction_result = self.add_new_frame(frame_data)
        
        # Return combined result, including prediction if available
        if prediction_result:
            frame_data["prediction_result"] = prediction_result
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

    

    

    

    