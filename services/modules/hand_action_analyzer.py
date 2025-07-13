from typing import Dict, Any, List, Optional
import numpy as np

# 행동 레이블 정의 (모델 학습 시 사용된 순서와 일치해야 함)
# 예시: 0: HOLDING_PEN, 1: HOLDING_PHONE, 2: NO_ACTION
CLASS_LABELS = ["HOLDING_PEN", "HOLDING_PHONE", "NO_ACTION", "FIST", "OPEN_PALM"]


class HandActionAnalyzer:
    """
    MediaPipe 손 랜드마크를 기반으로 사전 학습된 scikit-learn 모델을 사용하여
    '펜 쥐기', '주먹' 등 정적인 손 행동을 분류합니다.
    """

    def __init__(self, model: Optional[Any] = None, confidence_threshold: float = 0.75):
        """
        HandActionAnalyzer를 초기화합니다.

        Args:
            model (Optional[Any]): 미리 로드된 scikit-learn 분류 모델.
                                   None일 경우, 분석기는 '모델 없음' 상태를 반환합니다.
            confidence_threshold (float): 행동으로 판단하기 위한 최소 신뢰도 점수.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        if self.model:
            print("[INFO] Hand action analysis model loaded successfully.")
        else:
            print("[WARN] Hand action analysis model is not loaded. Analyzer will return 'MODEL_NOT_LOADED'.")

    def _normalize_landmarks(self, hand_landmarks: List[Any]) -> np.ndarray:
        """
        랜드마크 좌표를 정규화하여 모델 입력 형식에 맞게 전처리합니다.
        - 원점 이동: 손목(landmark 0)을 기준으로 모든 좌표를 이동시킵니다.
        - 스케일링: 손목으로부터 가장 먼 랜드마크까지의 거리를 기준으로 모든 좌표를 나눕니다.
        """
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

        base = coords[0]
        coords = coords - base

        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist == 0:
            return coords.flatten()

        coords = coords / max_dist

        # (21, 3) -> (63,)
        return coords.flatten()

    def analyze_frame(self, hand_landmarks: List[Any]) -> Dict[str, Any]:
        """
        단일 프레임의 손 랜드마크를 분석하여 손의 행동을 분류합니다.

        Args:
            hand_landmarks (List[Any]): MediaPipe에서 추출된 한쪽 손의 랜드마크 리스트.

        Returns:
            Dict[str, Any]: 분류된 행동과 신뢰도를 담은 딕셔너리.
                             모델이 없으면 'MODEL_NOT_LOADED'를 반환합니다.
                             예: {"action": "HOLDING_PEN", "confidence": 0.92}
        """
        if not self.model:
            return {"action": "MODEL_NOT_LOADED", "confidence": 0.0}

        if not hand_landmarks or len(hand_landmarks) < 21:
            return {"action": "NO_HAND_DETECTED", "confidence": 0.0}

        # 1. 랜드마크 데이터 전처리
        preprocessed_data = self._normalize_landmarks(hand_landmarks)

        # 2. 모델 추론
        try:
            # scikit-learn 모델은 (n_samples, n_features) 형태의 2D 배열을 기대함
            input_data = preprocessed_data.reshape(1, -1)

            # predict_proba()는 각 클래스에 대한 확률을 (1, n_classes) 형태로 반환
            probabilities = self.model.predict_proba(input_data)[0]

            predicted_class_index = np.argmax(probabilities)
            confidence = probabilities[predicted_class_index]

            action_label = CLASS_LABELS[predicted_class_index]

        except Exception as e:
            print(f"[ERROR] Failed to analyze hand action: {e}")
            return {"action": "ANALYSIS_FAILED", "confidence": 0.0}

        # 3. 신뢰도 임계값 확인
        if confidence < self.confidence_threshold:
            return {"action": "UNKNOWN", "confidence": round(float(confidence), 2)}

        return {
            "action": action_label,
            "confidence": round(float(confidence), 2)
        }
