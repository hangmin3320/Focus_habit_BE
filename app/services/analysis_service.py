import numpy as np
from datetime import datetime
import cv2

# 내부 로직이 변경됨에 따라 필요한 클래스들을 새로 import 합니다.
from services.media_pipe_runner import MediaPipeRunner
from services.modules.eye_analyzer import EyeAnalyzer
from services.modules.head_pose_analyzer import HeadPoseAnalyzer
from models.realtimemodel_gru import PersonalizedModelRunner
from supabase_service import supabase_service

class AnalysisService:
    """
    웹소켓 세션별로 생성되어, 실시간 스트림 데이터 분석을 지휘(Orchestration)합니다.
    복잡한 AI 추론 로직은 PersonalizedModelRunner에게 위임합니다.
    """
    def __init__(self, user_id: str):
        print(f"AnalysisService for user '{user_id}' initialized.")
        # MediaPipe 및 기본 분석 모듈 초기화
        self.media_pipe_runner = MediaPipeRunner()
        self.eye_analyzer = EyeAnalyzer()
        self.head_pose_analyzer = HeadPoseAnalyzer()

        # AI 전문 분석가(PersonalizedModelRunner)를 초기화합니다.
        # Supabase 연동 및 모델 로딩은 Runner가 내부적으로 모두 처리합니다.
        self.runner = PersonalizedModelRunner(
            user_id=user_id, 
            supabase_service=supabase_service,
            use_personal=True # 개인화 모델 사용을 시도합니다.
        )

    def close(self):
        """ MediaPipe 리소스를 해제합니다. """
        if self.media_pipe_runner:
            self.media_pipe_runner.close()
            print("MediaPipe resources closed.")

    async def analyze_frame(self, frame: np.ndarray):
        """
        단일 프레임을 받아 최종 분석 결과를 반환합니다.
        """
        # 1. 기본적인 얼굴 특징점 추출
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks = self.media_pipe_runner.get_face_landmarks(frame_rgb)
        except Exception as e:
            print(f"Error in MediaPipe processing: {e}")
            return {"error": f"MediaPipe error: {e}"}

        # 2. 랜드마크 기반의 1차 분석 (눈, 머리 자세)
        frame_data = {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "eye_status": {},
            "head_pose": {}
        }

        if face_landmarks:
            try:
                frame_data["eye_status"] = self.eye_analyzer.analyze_frame(face_landmarks)
                frame_data["head_pose"] = self.head_pose_analyzer.analyze_frame(face_landmarks, frame.shape[:2])
            except Exception as e:
                print(f"Error in primary analysis modules: {e}")
                # 1차 분석에 실패해도 AI 추론은 시도할 수 있도록 빈 값으로 넘어갑니다.

        # 3. AI 전문 분석가에게 데이터 전달 및 추론 요청
        try:
            # PersonalizedModelRunner는 JSON의 리스트를 입력으로 받습니다.
            prediction_results = self.runner.push_and_infer(
                json_payload=[frame_data], 
                return_json=True
            )
            
            # 추론 결과가 있을 경우, 마지막 결과를 프레임 데이터에 추가합니다.
            if prediction_results:
                frame_data["prediction_result"] = prediction_results[-1]["prediction_result"]

            return frame_data

        except Exception as e:
            print(f"Error during AI inference: {e}")
            return {"error": f"AI inference error: {e}"}