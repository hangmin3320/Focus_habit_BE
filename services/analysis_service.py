from typing import Dict, Any, List
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult, HandLandmarkerResult

from services.modules.eye_analyzer import EyeAnalyzer
from services.modules.head_pose_analyzer import HeadPoseAnalyzer
from services.modules.hand_action_analyzer import HandActionAnalyzer


class AnalysisService:
    """
    여러 분석 모듈을 통합하여 전체 이미지 분석 파이프라인을 관리합니다.
    각 모듈을 호출하고, 결과를 취합하여 최종 분석 결과를 생성합니다.
    """

    def __init__(self, hand_action_model: Any = None):
        """
        AnalysisService를 초기화하고 모든 분석 모듈을 로드합니다.
        Args:
            hand_action_model (Any): 미리 로드된 손 행동 분류 모델.
        """
        # 각 분석 모듈 인스턴스화
        self.eye_analyzer = EyeAnalyzer(ear_threshold=0.2)
        self.head_pose_analyzer = HeadPoseAnalyzer()
        self.hand_action_analyzer = HandActionAnalyzer(model=hand_action_model)

        # MediaPipe Vision Task 초기화
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        HandLandmarker = mp.tasks.vision.HandLandmarker
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # 프로젝트 루트 경로를 기준으로 모델 파일의 절대 경로 생성
        # __file__은 현재 파일의 경로, .parent는 부모 디렉터리를 의미
        # Path(__file__).parent.parent는 services/의 부모, 즉 프로젝트 루트
        project_root = Path(__file__).parent.parent
        face_landmarker_path = str(project_root / 'models/checkpoints/face_landmarker.task')
        hand_landmarker_path = str(project_root / 'models/checkpoints/hand_landmarker.task')

        try:
            # 얼굴 랜드마커 초기화
            face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_landmarker_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1)
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)

            # 손 랜드마커 초기화
            hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_landmarker_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2)
            self.hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)

        except Exception as e:
            print(f"Error initializing MediaPipe Landmarkers: {e}")
            self.face_landmarker = None
            self.hand_landmarker = None

    def analyze_image(self, image: Any, timestamp_ms: int) -> Dict[str, Any]:
        """
        입력된 이미지 프레임에 대해 모든 분석을 수행하고 결과를 반환합니다.
        """
        # 랜드마크 추출
        face_landmarks = self._detect_face_landmarks(image, timestamp_ms)
        hand_landmarks = self._detect_hand_landmarks(image, timestamp_ms)  # 첫 번째 감지된 손 사용
        image_shape = image.shape[:2]

        # 각 분석 모듈 실행
        eye_analysis_result = self.eye_analyzer.analyze_frame(face_landmarks)
        head_pose_result = self.head_pose_analyzer.analyze_frame(face_landmarks, image_shape)
        hand_action_result = self.hand_action_analyzer.analyze_frame(hand_landmarks)

        # 결과 종합
        final_result = {
            "timestamp": timestamp_ms,
            "eye_status": eye_analysis_result,
            "head_pose": head_pose_result,
            "hand_action": hand_action_result,
        }

        return final_result

    def _detect_face_landmarks(self, image: Any, timestamp_ms: int) -> List[Any]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result: FaceLandmarkerResult = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.face_landmarks[0] if result.face_landmarks else []

    def _detect_hand_landmarks(self, image: Any, timestamp_ms: int) -> List[Any]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result: HandLandmarkerResult = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.hand_landmarks[0] if result.hand_landmarks else []


# 사용 예시 (테스트용, 실제 서버 구동시에는 실행되지 않음.)
if __name__ == '__main__':
    import numpy as np
    import time

    # 가짜 데이터 생성
    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # AnalysisService 인스턴스화 (손 행동 분류 모델 없이 테스트)
    # 실제 실행 환경에서는 main.py에서 모델을 로드하여 주입합니다.
    analysis_service = AnalysisService(hand_action_model=None)

    # AnalysisService 전체 파이프라인 테스트
    current_timestamp_ms = int(time.time() * 1000)

    print("Running AnalysisService pipeline test...")
    try:
        # MediaPipe 모델이 초기화되었는지 확인
        if analysis_service.face_landmarker and analysis_service.hand_landmarker:
            full_analysis_result = analysis_service.analyze_image(fake_image, current_timestamp_ms)
            print(f"Full Analysis Result (with black image): {full_analysis_result}")
            print("Test finished successfully.")
        else:
            print("Test skipped: MediaPipe models could not be initialized.")
            print("Please ensure 'face_landmarker.task' and 'hand_landmarker.task' exist.")

    except Exception as e:
        print(f"An error occurred during the test run: {e}")
