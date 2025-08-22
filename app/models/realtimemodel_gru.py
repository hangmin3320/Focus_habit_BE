import json
import numpy as np
import os
import pandas as pd
import pickle
import threading
import torch
import torch.nn as nn
import io

from services.supabase_service import SupabaseService

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL_DIR))

# 로컬 체크포인트 경로는 Fallback용으로 계속 사용합니다.
LOCAL_CKPT_DIR = os.path.join(PROJECT_ROOT, "app", "models", "checkpoints")
TEMPERATURE = 1.0


class TimeSeriesCNNGRU(nn.Module):
    """
    1D-CNN + GRU 기반의 이진 분류 모델.
    (이하 클래스 내용은 변경 없음)
    """

    def __init__(self, input_channels, window_size=25, dropout_rate=0.3, k1=3, k2=32, two_convs=True):
        super().__init__()
        c = 32
        ks = k1
        pad = ks // 2
        self.two_convs = two_convs
        self.conv1 = nn.Conv1d(input_channels, c, kernel_size=ks, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        if self.two_convs:
            self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=pad, bias=False)
            self.bn2 = nn.BatchNorm1d(c)
            self.pool2 = nn.MaxPool1d(2)
        self.gru = nn.GRU(input_size=c, hidden_size=k2, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(k2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        if self.two_convs:
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        return self.fc(h)


def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = 0.0
    return df


def feature_engineer(df, base=('ear', 'pitch', 'yaw', 'roll'), username="USER"):
    if df.empty:
        return df, []
    df = ensure_cols(df, list(base) + ['eye_status', 'prefix'])
    if 'timestamp_ms' not in df.columns: df['timestamp_ms'] = np.arange(len(df))
    if 'prefix' not in df.columns: df['prefix'] = username
    df = df.sort_values(['prefix', 'timestamp_ms'])
    for f in base:
        if f not in df.columns: df[f] = 0.0
    for f in base:
        df[f'{f}_diff'] = df.groupby('prefix')[f].diff().fillna(0)
        m = df.groupby('prefix')[f].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        s = df.groupby('prefix')[f].rolling(window=5, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
        df[f'{f}_mean_5'] = m
        df[f'{f}_std_5'] = s
    df['eye_status_numeric'] = df['eye_status'].apply(lambda x: x.get('status') if isinstance(x, dict) else x).map({'OPEN': 1, 'CLOSED': 0}).fillna(0)
    t = df.groupby('prefix')['eye_status_numeric'].diff().eq(-1)
    df['blink_count'] = t.rolling(window=50, min_periods=1).sum().fillna(0).reset_index(level=0, drop=True)
    df['angle_magnitude'] = np.sqrt(df['pitch_diff'] ** 2 + df['yaw_diff'] ** 2 + df['roll_diff'] ** 2)
    feats = list(base) + [f'{f}_diff' for f in base] + [f'{f}_mean_5' for f in base] + [f'{f}_std_5' for f in base] + [
        'blink_count', 'angle_magnitude']
    df[feats] = df[feats].replace([np.inf, -np.inf], 0).fillna(0)
    return df, feats


def compute_stride(window_size, overlap=0.5):
    return max(1, int(window_size * (1 - overlap)))


def majority_smooth(labels, k=3):
    if k is None or k <= 1: return labels
    from collections import deque
    buf = deque(maxlen=k)
    out = []
    for v in labels:
        buf.append(v)
        out.append(int(round(np.mean(buf))))
    return out


class PersonalizedModelRunner:
    """
    Supabase와 연동하여 개인화 모델을 로드하고, 실시간 추론을 수행하는 런타임.
    """

    def __init__(self, user_id: str, supabase_service: SupabaseService, use_personal: bool = True):
        self.user_id = user_id
        self.supabase_service = supabase_service
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bucket_name = "models"  # Supabase 스토리지 버킷 이름

        self.scaler = None
        self.model = None
        
        # 모델 설정값 (헤더) 및 피처 리스트 초기화
        self.window_size = 25
        self.overlap = 0.5
        self.features = None
        self.threshold = 0.5
        self.smooth_k = None
        self.temperature = TEMPERATURE

        # 개인화 모델 로드를 시도하고, 실패 시 기본 모델을 로드합니다.
        if use_personal:
            self._try_load_personal_model()

        # 개인화 모델 로드에 실패했거나, use_personal=False인 경우 기본 모델을 로드합니다.
        if self.model is None or self.scaler is None:
            print(f"User '{self.user_id}' - Loading baseline model.")
            self._load_baseline_model()

        self.lock = threading.Lock()
        self.buffer = pd.DataFrame({
            'timestamp_ms': pd.Series(dtype='float64'),
            'eye_status': pd.Series(dtype='object'),
            'ear': pd.Series(dtype='float64'),
            'pitch': pd.Series(dtype='float64'),
            'yaw': pd.Series(dtype='float64'),
            'roll': pd.Series(dtype='float64'),
            'prefix': pd.Series(dtype='object'),
        })

    def _try_load_personal_model(self):
        print(f"User '{self.user_id}' - Attempting to load personal model from Supabase.")
        try:
            header_bytes = self.supabase_service.download_file(self.bucket_name, f"{self.user_id}/{self.user_id}_header.json")
            scaler_bytes = self.supabase_service.download_file(self.bucket_name, f"{self.user_id}/{self.user_id}_scaler.pkl")
            model_bytes = self.supabase_service.download_file(self.bucket_name, f"{self.user_id}/{self.user_id}_model.pth")

            if not all([header_bytes, scaler_bytes, model_bytes]):
                print(f"User '{self.user_id}' - Personal model files not found in Supabase.")
                return

            # 헤더 파일 로드 및 설정 적용
            hdr = json.loads(header_bytes.decode('utf-8'))
            self.window_size = int(hdr.get("window_size", 25))
            self.overlap = float(hdr.get("overlap", 0.5))
            self.features = hdr.get("features", None)
            self.threshold = float(hdr.get("threshold", 0.5))
            self.smooth_k = hdr.get("smooth_k", None)
            self.temperature = float(hdr.get("temperature", TEMPERATURE))

            # 스케일러 로드
            self.scaler = pickle.loads(scaler_bytes)

            # 모델 구조 생성 및 가중치 로드
            self.model = TimeSeriesCNNGRU(len(self.features), self.window_size, 0.3, 3, 32, True).to(self.device)
            self.model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=self.device))
            self.model.eval()
            print(f"User '{self.user_id}' - Personal model loaded successfully from Supabase.")

        except Exception as e:
            print(f"An error occurred while loading personal model for user '{self.user_id}': {e}")
            # 실패 시 self.model, self.scaler를 None으로 유지하여 기본 모델을 로드하도록 함
            self.model = None
            self.scaler = None

    def _load_baseline_model(self):
        # Baseline 모델은 고정된 피처 목록을 사용합니다.
        # 동적 피처 엔지니어링 대신, 피처 목록을 직접 정의하여 안정성을 높입니다.
        print("Info: Defining baseline model features statically.")
        base_features = ('ear', 'pitch', 'yaw', 'roll')
        self.features = list(base_features) + \
                        [f'{f}_diff' for f in base_features] + \
                        [f'{f}_mean_5' for f in base_features] + \
                        [f'{f}_std_5' for f in base_features] + \
                        ['blink_count', 'angle_magnitude']

        # 로컬 기본 스케일러 로드
        scaler_path = os.path.join(LOCAL_CKPT_DIR, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # 로컬 기본 모델 로드
        model_path = os.path.join(LOCAL_CKPT_DIR, "baseline_model.pth")
        self.model = TimeSeriesCNNGRU(len(self.features), self.window_size, 0.3, 3, 32, True).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _append_json(self, payload):
        # (이하 메소드 내용은 변경 없음)
        if isinstance(payload, str) and os.path.isfile(payload):
            with open(payload, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif isinstance(payload, (str, bytes)):
            data = json.loads(payload)
        else:
            data = payload
        rows = []
        for it in data:
            ts = it.get('timestamp', it.get('timestamp_ms', None))
            es = it.get('eye_status', {})
            hp = it.get('head_pose', {})
            ear_val = es.get('ear_value', es.get('ear', 0.0))
            rows.append({
                'timestamp_ms': ts if ts is not None else np.nan,
                'eye_status': es.get('status', 'OPEN'),
                'ear': float(ear_val) if ear_val is not None else 0.0,
                'pitch': float(hp.get('pitch', 0.0)),
                'yaw': float(hp.get('yaw', 0.0)),
                'roll': float(hp.get('roll', 0.0)),
                'prefix': self.user_id
            })
        df = pd.DataFrame(rows)
        if 'timestamp_ms' not in df.columns:
            df['timestamp_ms'] = np.arange(len(df))
        if self.buffer.empty:
            self.buffer = df.copy()
        else:
            self.buffer = pd.concat([self.buffer, df], ignore_index=True)

    def push_and_infer(self, json_payload, threshold=None, return_prob=False, smooth_k=None, return_json=False):
        # (이하 메소드 내용은 변경 없음)
        with self.lock:
            self._append_json(json_payload)
            df = self.buffer.copy()
            df, feats = feature_engineer(df, username=self.user_id)
            if self.features is not None:
                for c in self.features:
                    if c not in df.columns: df[c] = 0.0
                feats = self.features
            df[feats] = self.scaler.transform(df[feats])

            stride = compute_stride(self.window_size, self.overlap)
            preds = []
            probs = []
            records = []

            thr = threshold if threshold is not None else self.threshold
            use_smooth_k = smooth_k if smooth_k is not None else self.smooth_k

            for p in df['prefix'].unique():
                g = df[df['prefix'] == p].sort_values('timestamp_ms')
                if len(g) < self.window_size: continue
                arr = g[feats].values
                for i in range(0, len(g) - self.window_size + 1, stride):
                    x = arr[i:i + self.window_size][None, ...]
                    x = torch.tensor(x, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        logit = self.model(x).squeeze(-1)
                        pr = torch.sigmoid(logit / self.temperature).item()
                    probs.append(pr)
                    pred = 1 if pr >= thr else 0
                    preds.append(pred)

                    if return_json:
                        ts = int(g['timestamp_ms'].iloc[i + self.window_size - 1])
                        eye = str(g['eye_status'].iloc[i + self.window_size - 1])
                        ear = float(g['ear'].iloc[i + self.window_size - 1])
                        pitch = float(g['pitch'].iloc[i + self.window_size - 1])
                        yaw = float(g['yaw'].iloc[i + self.window_size - 1])
                        roll = float(g['roll'].iloc[i + self.window_size - 1])
                        conf = float(max(pr, 1.0 - pr))
                        rec = {
                            "timestamp": ts,
                            "eye_status": {"status": eye, "ear_value": ear},
                            "head_pose": {"pitch": pitch, "yaw": yaw, "roll": roll},
                            "prediction_result": {
                                "timestamp": ts,
                                "prediction": float(np.clip(pr * 100.0, 0.0, 100.0)),
                                "confidence": conf
                            }
                        }
                        records.append(rec)

            if use_smooth_k and len(preds) > 0:
                preds = majority_smooth(preds, k=int(use_smooth_k))

            if return_json:
                return records
            return (preds, probs) if return_prob else preds