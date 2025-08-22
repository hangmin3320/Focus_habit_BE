import numpy as np
from datetime import datetime
import cv2
import time # 추가
from collections import deque
from typing import Optional

# 내부 로직이 변경됨에 따라 필요한 클래스들을 새로 import 합니다.
from services.media_pipe_runner import MediaPipeRunner
from services.modules.eye_analyzer import EyeAnalyzer
from services.modules.head_pose_analyzer import HeadPoseAnalyzer
from models.realtimemodel_gru import PersonalizedModelRunner
from services.supabase_service import supabase_service

class AnalysisService:
    """
    웹소켓 세션별로 생성되어, 실시간 스트림 데이터 분석을 지휘(Orchestration)합니다.
    복잡한 AI 추론 로직은 PersonalizedModelRunner에게 위임합니다.
    """
    def __init__(self, user_id: str | None = None, fixed_baseline: Optional[float] = None):
        print(f"AnalysisService for user '{user_id}' initialized.")
        # MediaPipe 및 기본 분석 모듈 초기화
        self.media_pipe_runner = MediaPipeRunner()
        self.eye_analyzer = EyeAnalyzer()
        self.head_pose_analyzer = HeadPoseAnalyzer()
        self.runner = None # AI 추론기는 기본적으로 None

        # user_id가 제공된 경우에만 AI 분석가(PersonalizedModelRunner)를 초기화합니다.
        if user_id:
            self.runner = PersonalizedModelRunner(
                user_id=user_id,
                supabase_service=supabase_service,
                use_personal=True # 개인화 모델 사용을 시도합니다.
            )
        
        # 심박수 관련 변수 초기화
        self._heart_rate_history: list[tuple[float, float]] = [] # (timestamp_sec, heart_rate) 튜플 저장
        self._baseline_heart_rate: Optional[float] = None
        self._session_start_time: float = time.time()
        self._last_frame_data: Optional[dict] = None # 이전 프레임 데이터 저장

        # 고정 기준 심박수가 제공된 경우 설정
        if fixed_baseline is not None:
            self._baseline_heart_rate = fixed_baseline
            print(f"Fixed baseline heart rate set to: {self._baseline_heart_rate}")

    def close(self):
        """ MediaPipe 리소스를 해제합니다. """
        if self.media_pipe_runner:
            self.media_pipe_runner.close()
            print("MediaPipe resources closed.")

    async def analyze_frame(self, frame: np.ndarray, heart_rate: Optional[float] = None):
        """
        단일 프레임을 받아 최종 분석 결과를 반환합니다.
        """
        current_timestamp = int(datetime.now().timestamp() * 1000)
        
        # 1. 기본적인 얼굴 특징점 추출
        face_landmarks = None
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks = self.media_pipe_runner.get_face_landmarks(frame_rgb)
        except Exception as e:
            print(f"Error in MediaPipe processing: {e}")
            # MediaPipe 에러 발생 시에도 심박수 로직은 시도할 수 있도록 face_landmarks는 None 유지

        # 2. 랜드마크 기반의 1차 분석 (눈, 머리 자세) 및 AI 추론
        frame_data = {
            "timestamp": current_timestamp,
            "eye_status": {},
            "head_pose": {}
        }
        
        if face_landmarks: # 얼굴이 탐지된 경우
            try:
                frame_data["eye_status"] = self.eye_analyzer.analyze_frame(face_landmarks)
                frame_data["head_pose"] = self.head_pose_analyzer.analyze_frame(face_landmarks, frame.shape[:2])
            except Exception as e:
                print(f"Error in primary analysis modules: {e}")
                # 1차 분석에 실패해도 AI 추론은 시도할 수 있도록 빈 값으로 넘어갑니다.

            # 3. AI 전문 분석가에게 데이터 전달 및 추론 요청 (self.runner가 있을 경우에만)
            if self.runner:
                try:
                    prediction_results = self.runner.push_and_infer(
                        json_payload=[frame_data],
                        return_json=True
                    )
                    if prediction_results:
                        frame_data["prediction_result"] = prediction_results[-1]["prediction_result"]
                except Exception as e:
                    print(f"Error during AI inference: {e}")
            
            self._last_frame_data = frame_data # 얼굴 탐지 시 현재 프레임 데이터를 저장
            return frame_data

        else: # 얼굴이 탐지되지 않은 경우
            # 심박수 데이터가 제공된 경우에만 심박수 기반 로직 실행
            if heart_rate is not None:
                current_time_sec = time.time()
                
                # (타임스탬프, 심박수) 튜플 저장
                self._heart_rate_history.append((current_time_sec, heart_rate))

                # 세션 시작 후 5분(300초)이 경과했고, 아직 기준 심박수가 계산되지 않았다면
                if current_time_sec - self._session_start_time >= 300 and self._baseline_heart_rate is None:
                    # 5분 동안의 심박수 데이터만 필터링하여 평균 계산
                    relevant_heart_rates = [
                        hr for ts, hr in self._heart_rate_history
                        if self._session_start_time <= ts <= (self._session_start_time + 300)
                    ]

                    if relevant_heart_rates:
                        self._baseline_heart_rate = sum(relevant_heart_rates) / len(relevant_heart_rates)
                        self._heart_rate_history.clear() # 기준값 계산 후 히스토리 비움
                        print(f"Baseline heart rate calculated: {self._baseline_heart_rate}") # 디버깅용
                    else:
                        # 5분 동안 심박수 데이터가 전혀 수집되지 않은 경우 (경고 또는 다른 처리)
                        print("Warning: No heart rate data collected during the 5-minute baseline period.")
                        # 이 경우 self._baseline_heart_rate는 계속 None으로 유지되어 다음 프레임에서 다시 시도
                
                # 기준 심박수가 계산된 경우 졸음 판단
                if self._baseline_heart_rate is not None:
                    if heart_rate < self._baseline_heart_rate:
                        # 졸음 상태로 판단
                        prediction_result = {
                            "timestamp": current_timestamp,
                            "prediction": "drowsiness", # 또는 정의된 상수
                            "confidence": 1.0 # 룰 기반이므로 1.0
                        }
                        # 다른 필드 (eye_status, head_pose)는 이전 프레임 데이터에서 가져오거나 기본값 설정
                        response_data = {
                            "timestamp": current_timestamp,
                            "eye_status": self._last_frame_data.get("eye_status", {"status": "UNKNOWN", "ear_value": 0.0}) if self._last_frame_data else {"status": "UNKNOWN", "ear_value": 0.0},
                            "head_pose": self._last_frame_data.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}) if self._last_frame_data else {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                            "prediction_result": prediction_result
                        }
                        # 이 경우 _last_frame_data는 업데이트하지 않음 (얼굴 미탐지 시에는 이전 데이터 유지)
                        return response_data
                    else:
                        # 비졸음 상태로 판단, 이전 프레임 데이터 반환
                        if self._last_frame_data:
                            return self._last_frame_data
                        else:
                            # 이전 데이터가 없으면 기본 응답 반환
                            return {
                                "timestamp": current_timestamp,
                                "eye_status": {"status": "UNKNOWN", "ear_value": 0.0},
                                "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                                "prediction_result": {"prediction": "focused_by_hr", "confidence": 1.0} # 심박수 기반 비졸음
                            }
                else:
                    # 기준 심박수 측정 중, 이전 프레임 데이터 반환
                    if self._last_frame_data:
                        return self._last_frame_data
                    else:
                        return {
                            "timestamp": current_timestamp,
                            "eye_status": {"status": "UNKNOWN", "ear_value": 0.0},
                            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                            "prediction_result": {"prediction": "measuring_baseline", "confidence": 0.0}
                        }
            else:
                # 심박수 데이터가 없는 경우, 이전 프레임 데이터 반환
                if self._last_frame_data:
                    return self._last_frame_data
                else:
                    return {
                        "timestamp": current_timestamp,
                        "eye_status": {"status": "UNKNOWN", "ear_value": 0.0},
                        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                        "prediction_result": {"prediction": "no_face_no_hr", "confidence": 0.0}
                    }