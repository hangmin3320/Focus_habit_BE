import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import zipfile
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from services.supabase_service import supabase_service
from models.realtimemodel_gru import TimeSeriesCNNGRU, feature_engineer # compute_sequences 제거

# 스크립트의 위치를 기준으로 프로젝트 루트 디렉토리를 찾습니다。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CKPT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")

# (TimeSeriesCNNGRU, feature_engineer 등 모델/전처리 관련 함수들은 realtimemodel_gru.py에서 import하여 사용)
# ... (이전 파일의 유틸리티 함수들은 그대로 유지되거나 realtimemodel_gru.py에서 가져옴)

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = 0.0
    return df


def feature_engineer(df, base=('ear', 'pitch', 'yaw', 'roll'), username="USER"):
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
    df['blink_count'] = t.rolling(window=5, min_periods=1).sum().fillna(0).reset_index(level=0, drop=True)
    df['angle_magnitude'] = np.sqrt(df['pitch_diff'] ** 2 + df['yaw_diff'] ** 2 + df['roll_diff'] ** 2)
    feats = list(base) + [f'{f}_diff' for f in base] + [f'{f}_mean_5' for f in base] + [f'{f}_std_5' for f in base] + [
        'blink_count', 'angle_magnitude']
    df[feats] = df[feats].replace([np.inf, -np.inf], 0).fillna(0)
    return df, feats


def compute_stride(window_size, overlap=0.5):
    return max(1, int(window_size * (1 - overlap)))


def create_sequences(df, features, window_size=25, stride=12, threshold=0.5):
    """
    슬라이딩 윈도우로 (X, y) 시퀀스 생성.

    레이블링:
        - df['label']가 존재하면 윈도우 평균 >= threshold → 1, 아니면 0.
        - df['label']가 없으면 y는 빈 배열.

    Args:
        df (pd.DataFrame): 입력 데이터프레임(정렬 완료).
        features (list[str]): 사용 피처 목록.
        window_size (int): 시퀀스 길이.
        stride (int): 슬라이딩 보폭.
        threshold (float): 다수결/평균 임계값.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            X: (N, window_size, C)
            y: (N,) 이진 레이블(없으면 길이 0)
    """
    X, y = [], []
    for p in df['prefix'].unique():
        g = df[df['prefix'] == p].sort_values('timestamp_ms')
        if len(g) < window_size: continue
        arr = g[features].values
        lab = g['label'].values if 'label' in g.columns else None
        for i in range(0, len(g) - window_size + 1, stride):
            X.append(arr[i:i + window_size])
            if lab is not None:
                seq = lab[i:i + window_size]
                y.append(1 if np.mean(seq) >= threshold else 0)
    X = np.array(X) if len(X) else np.empty((0, len(features), window_size))
    y = np.array(y) if len(y) else np.array([])
    return X, y


def set_trainable(model, mode):
    for p in model.parameters(): p.requires_grad = False
    if mode == 'head':
        for p in model.fc.parameters(): p.requires_grad = True
    elif mode == 'gru':
        for p in model.gru.parameters(): p.requires_grad = True
        for p in model.fc.parameters(): p.requires_grad = True
    elif mode == 'full':
        for p in model.parameters(): p.requires_grad = True

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.count = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience: self.stop = True

def _sweep_threshold(probs, labels, grid=None):
    if grid is None: grid = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_acc = -1.0
    for t in grid:
        acc = ((probs >= t).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t), float(best_acc)

def train_personal(username, df_focus, df_nonfocus, output_dir, base_model_path=None, mode='gru', epochs=5, lr=1e-4, wd=1e-5, window_size=25, overlap=0.5, smooth_k=3, thr_grid=None):
    if base_model_path is None: base_model_path = os.path.join(CKPT_DIR, "baseline_model.pth")
    
    df = pd.concat([df_focus, df_nonfocus], ignore_index=True)
    if 'eye_status' in df.columns:
        df = df[df['eye_status'] != 'NO_FACE_DETECTED']

    df, feats = feature_engineer(df, username=username)
    scaler = StandardScaler()
    df[feats] = scaler.fit_transform(df[feats])

    stride = compute_stride(window_size, overlap)
    X, y = create_sequences(df, feats, window_size, stride, threshold=0.5)
    if X.shape[0] < 20: # 데이터가 너무 적으면 학습 중단
        print(f"User '{username}' - Insufficient data to train. Aborting.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesCNNGRU(len(feats), window_size).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    set_trainable(model, mode)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pos = max(1, int((y_tr == 1).sum()))
    neg = max(1, int((y_tr == 0).sum()))
    pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    early = EarlyStopping(patience=3, min_delta=0.0)
    best_sd = None
    best_val = np.inf

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            lg = model(xb).squeeze(-1)
            loss = criterion(lg, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb).squeeze(-1)
                ls = criterion(lg, yb)
                vloss += ls.item() * xb.size(0)
                n += xb.size(0)
        vloss /= max(1, n)
        if vloss < best_val:
            best_val = vloss
            best_sd = model.state_dict()
        if early.step(vloss) or np.isnan(vloss): break

    if best_sd is not None: model.load_state_dict(best_sd)

    with torch.no_grad():
        lg = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(-1)
        probs = torch.sigmoid(lg).cpu().numpy()
    best_t, best_acc = _sweep_threshold(probs, y, grid=thr_grid)

    # 결과를 지정된 output_dir에 저장
    user_model_path = os.path.join(output_dir, f"{username}_model.pth")
    user_scaler_path = os.path.join(output_dir, f"{username}_scaler.pkl")
    user_header_path = os.path.join(output_dir, f"{username}_header.json")
    torch.save(model.state_dict(), user_model_path)
    with open(user_scaler_path, 'wb') as f: pickle.dump(scaler, f)

    header = {
        "username": username, "window_size": window_size, "overlap": overlap, "stride": stride, "mode": mode, "epochs": epochs, "lr": lr, "wd": wd,
        "features": feats, "best_val_loss": float(best_val), "threshold": best_t
    }
    with open(user_header_path, 'w', encoding='utf-8') as f: json.dump(header, f, ensure_ascii=False)

    return {"model": user_model_path, "scaler": user_scaler_path, "header": user_header_path}

def train_personal_from_storage(user_id: str, storage_path: str):
    """
    Supabase에서 학습 데이터를 다운로드하고, 개인화 학습을 실행한 뒤, 결과물을 다시 업로드합니다.
    FastAPI의 BackgroundTasks에서 실행될 래퍼 함수입니다.
    """
    bucket_name = "models"
    print(f"Starting personal training for user '{user_id}' from '{storage_path}'")

    # 1. Supabase에서 학습 데이터(zip) 다운로드
    zip_bytes = supabase_service.download_file(bucket_name, storage_path)
    if not zip_bytes:
        print(f"Failed to download training data from {storage_path}")
        return

    # 2. 임시 디렉터리에 zip 파일 압축 해제
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(temp_dir)
        
        focus_json_path = os.path.join(temp_dir, "focus.json")
        nonfocus_json_path = os.path.join(temp_dir, "nonfocus.json")

        if not os.path.exists(focus_json_path) or not os.path.exists(nonfocus_json_path):
            print("focus.json or nonfocus.json not found in zip file.")
            return

        df_focus = pd.read_json(focus_json_path)
        df_nonfocus = pd.read_json(nonfocus_json_path)

        # 3. 모델 학습 실행 (결과물은 temp_dir에 저장됨)
        print(f"Starting model training in temp dir: {temp_dir}")
        trained_files = train_personal(user_id, df_focus, df_nonfocus, output_dir=temp_dir)

        if not trained_files:
            print(f"Training failed for user '{user_id}'.")
            return

        # 4. 학습된 결과물(model, scaler, header)을 Supabase에 업로드
        for key, local_path in trained_files.items():
            remote_path = f"{user_id}/{os.path.basename(local_path)}"
            with open(local_path, 'rb') as f:
                file_body = f.read()
            supabase_service.upload_file(bucket_name, remote_path, file_body)
    
    print(f"Personal training for user '{user_id}' completed successfully.")