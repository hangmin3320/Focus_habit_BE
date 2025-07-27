import math
from typing import Dict, Any, List


class EyeAnalyzer:
    RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 390]

    def __init__(self, ear_threshold: float = 0.2):
        self.ear_threshold = ear_threshold

    def _calculate_euclidean_distance(self, p1: Any, p2: Any) -> float:
        """두 랜드마크 포인트 간의 유클리드 거리를 계산"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _calculate_ear(self, eye_landmarks: List[Any]) -> float:
        """
        한쪽 눈의 랜드마크를 사용하여 눈 종횡비(EAR)를 계산합니다.

        EAR 공식: (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        - 세로 길이의 합을 가로 길이로 나누어 눈의 개방 비율을 계산

        Args:
            eye_landmarks (List[Any]): 한쪽 눈을 구성하는 6개의 랜드마크 리스트.

        Returns:
            float: 계산된 EAR 값.
        """
        # 수직 거리 계산
        vertical_dist1 = self._calculate_euclidean_distance(eye_landmarks[1], eye_landmarks[5])
        vertical_dist2 = self._calculate_euclidean_distance(eye_landmarks[2], eye_landmarks[4])

        # 수평 거리 계산
        horizontal_dist = self._calculate_euclidean_distance(eye_landmarks[0], eye_landmarks[3])

        # EAR 계산
        if horizontal_dist == 0:
            return 0.0

        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear

    def analyze_frame(self, face_landmarks: List[Any]) -> Dict[str, Any]:
        """
        단일 프레임의 얼굴 랜드마크를 분석하여 눈 상태를 판별합니다.

        Args:
            face_landmarks (List[Any]): MediaPipe에서 추출된 전체 얼굴 랜드마크 리스트.

        Returns:
            Dict[str, Any]: 눈의 상태("OPEN", "CLOSED", "NO_FACE_DETECTED")와
                             평균 EAR 값을 담은 딕셔너리.
                             예: {"status": "CLOSED", "ear_value": 0.15}
        """
        if not face_landmarks:
            return {"status": "NO_FACE_DETECTED", "ear_value": 0.0}

        # 각 눈의 랜드마크 추출
        right_eye_landmarks = [face_landmarks[i] for i in self.RIGHT_EYE_POINTS]
        left_eye_landmarks = [face_landmarks[i] for i in self.LEFT_EYE_POINTS]

        # 각 눈의 EAR 값 계산
        right_ear = self._calculate_ear(right_eye_landmarks)
        left_ear = self._calculate_ear(left_eye_landmarks)

        # 양쪽 눈의 평균 EAR 값 계산
        avg_ear = (left_ear + right_ear) / 2.0

        # 임계값과 비교하여 눈 상태 판별
        if avg_ear < self.ear_threshold:
            status = "CLOSED"
        else:
            status = "OPEN"

        return {
            "status": status,
            "ear_value": round(avg_ear, 4)
        }