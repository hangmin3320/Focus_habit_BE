from dataclasses import dataclass, field
import os
from typing import List, Tuple

# 프로젝트의 루트 디렉터리를 기준으로 경로 설정
# 이 파일을 기준으로 app 폴더가 상위 폴더이므로, os.path.dirname을 두 번 사용
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class TrainingConfig:
    """학습 파이프라인에 필요한 모든 설정을 관리하는 데이터 클래스"""

    # --- 경로 설정 ---
    base_dir: str = BASE_DIR
    data_dir: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data"))
    checkpoint_dir: str = field(default_factory=lambda: os.path.join(BASE_DIR, "app", "models", "checkpoints"))
    output_dir: str = field(default_factory=lambda: os.path.join(BASE_DIR, "training_outputs"))

    # --- 데이터 설정 ---
    focus_json_path: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data", "focus.json"))
    nonfocus_json_path: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data", "nonfocus.json"))
    val_ratio: float = 0.2
    test_ratio: float = 0.4 # stratified_time_split 용

    # --- 모델 및 특징 공학 설정 ---
    base_features: Tuple[str, ...] = ("ear", "pitch", "yaw", "roll")
    window_size: int = 25
    overlap: float = 0.5
    label_threshold: float = 0.5 # 시퀀스 레이블링 시 비집중 비율 임계값

    # --- 학습 하이퍼파라미터 ---
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 7
    seed: int = 42

    # --- 하이퍼파라미터 탐색 설정 ---
    sweep_window_sizes: List[int] = field(default_factory=lambda: [50])
    sweep_dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    sweep_kernel_sets: List[Tuple[int, int]] = field(default_factory=lambda: [(3, 32), (5, 32), (3, 64)])

    # --- 개인화 학습 프로파일 설정 ---
    min_train_pr_auc: float = 0.7 # 이 값 미만이면 개인화 모델로 채택하지 않음
    personalization_modes: List[str] = field(default_factory=lambda: ["cal", "head", "cal_head", "transfer_gru", "transfer_cnn_gru"])
    activation_metric: str = "f1" # 최적 모델 선택 시 사용할 평가지표

    def __post_init__(self):
        """인스턴스 생성 후, 필요한 디렉터리들을 생성"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
