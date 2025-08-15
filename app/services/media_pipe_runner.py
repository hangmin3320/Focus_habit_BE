
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 모델 경로 설정
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/checkpoints/face_landmarker.task')

class MediaPipeRunner:
    """
    MediaPipe Face Landmarker를 설정하고 실행하여
    이미지에서 얼굴 랜드마크를 추출하는 클래스.
    """
    def __init__(self):
        # MediaPipe Face Landmarker 초기화
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False, # 표정 데이터는 사용하지 않음
            output_facial_transformation_matrixes=False, # 변환 행렬은 사용하지 않음
            num_faces=1 # 한 명의 얼굴만 감지
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe Face Landmarker initialized.")

    def get_face_landmarks(self, image_rgb: np.ndarray):
        """
        RGB 이미지를 입력받아 얼굴 랜드마크를 추출합니다.

        Args:
            image_rgb (np.ndarray): RGB 형식의 이미지 배열.

        Returns:
            List[Any] or None: 감지된 얼굴의 랜드마크 리스트. 
                               얼굴이 감지되지 않으면 None을 반환합니다.
        """
        # MediaPipe 이미지 형식으로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 랜드마크 감지 실행
        detection_result = self.landmarker.detect(mp_image)

        # 결과에서 랜드마크만 추출
        if detection_result.face_landmarks:
            return detection_result.face_landmarks[0] # 첫 번째 얼굴의 랜드마크 반환
        else:
            return None # 얼굴 미감지 시 None 반환

    def close(self):
        """ MediaPipe FaceLandmarker 리소스를 해제합니다. """
        if self.landmarker:
            self.landmarker.close()
            print("MediaPipe Face Landmarker resources released.")
