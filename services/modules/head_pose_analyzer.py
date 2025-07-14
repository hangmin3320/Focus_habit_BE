import math
from typing import Dict, Any, List
import numpy as np
import cv2


class HeadPoseAnalyzer:
    """
    MediaPipe 얼굴 랜드마크와 OpenCV를 사용하여 머리의 3차원 방향(Pitch, Yaw, Roll)을 추정합니다.
    """

    def __init__(self, smoothing_factor: float = 0.5):
        """
        HeadPoseAnalyzer를 초기화합니다.

        Args:
            smoothing_factor (float): 출력 각도의 노이즈를 줄이기 위한 평활화 계수.
                                      0과 1 사이 값으로, 높을수록 부드러워집니다.
        """
        self.smoothing_factor = smoothing_factor
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_roll = 0.0

    def analyze_frame(self, face_landmarks: List[Any], image_shape: tuple) -> Dict[str, Any]:
        """
        단일 프레임의 얼굴 랜드마크를 분석하여 머리 자세를 추정합니다.
        OpenCV의 solvePnP를 사용하여 2D 이미지 좌표로부터 3D 회전 각도를 계산합니다.

        Args:
            face_landmarks (List[Any]): MediaPipe에서 추출된 전체 얼굴 랜드마크 리스트.
            image_shape (tuple): (height, width) 형태의 이미지 크기 튜플.

        Returns:
            Dict[str, Any]: 머리의 Pitch, Yaw, Roll 각도를 담은 딕셔너리.
                             예: {"pitch": 15.2, "yaw": -5.1, "roll": 2.5}
        """
        if not face_landmarks:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        img_h, img_w = image_shape

        # 3D 모델 포인트 (일반적인 얼굴 모델 기준)
        face_3d_model_points = np.array([
            [0.0, 0.0, 0.0],  # Nose tip
            [0.0, -330.0, -65.0],  # Chin
            [-225.0, 170.0, -135.0],  # Left eye left corner
            [225.0, 170.0, -135.0],  # Right eye right corner
            [-150.0, -150.0, -125.0],  # Left Mouth corner
            [150.0, -150.0, -125.0]  # Right mouth corner
        ], dtype=np.float64)

        # 해당 3D 포인트에 대응하는 2D 랜드마크 인덱스
        face_2d_image_points_indices = [1, 199, 263, 33, 61, 291]

        face_2d_image_points = np.array([
            (face_landmarks[i].x * img_w, face_landmarks[i].y * img_h) for i in face_2d_image_points_indices
        ], dtype=np.float64)

        # 카메라 매개변수 (가정)
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                               [0, focal_length, img_h / 2],
                               [0, 0, 1]], dtype=np.float64)

        # 왜곡 계수 (없다고 가정)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # SolvePnP를 사용하여 회전 및 변환 벡터 계산
        success, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d_model_points,
            face_2d_image_points,
            cam_matrix,
            dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        # 회전 벡터를 회전 행렬로 변환
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # 회전 행렬로부터 오일러 각도(Pitch, Yaw, Roll) 계산
        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # 라디안을 도로 변환
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)

        # 평활화 (Smoothing)
        pitch = self.prev_pitch * self.smoothing_factor + pitch * (1 - self.smoothing_factor)
        yaw = self.prev_yaw * self.smoothing_factor + yaw * (1 - self.smoothing_factor)
        roll = self.prev_roll * self.smoothing_factor + roll * (1 - self.smoothing_factor)

        self.prev_pitch, self.prev_yaw, self.prev_roll = pitch, yaw, roll

        return {
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2)
        }