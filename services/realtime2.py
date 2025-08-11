# ============================================================
# infer_focus_cnn_runtime.py
# JSON -> DataFrame -> 스케일링 -> 모델 추론 (윈도우=20 이전은 출력 없음)
# 체크포인트: app/models/checkpoints/{cnnmodel.pth, cnnscaler.pkl, features.pkl}
# 실행 예:
#   python infer_focus_cnn_runtime.py --in input.json --out preds.csv --threshold 0.5
#   cat input.json | python infer_focus_cnn_runtime.py -          # stdin에서 읽고 stdout(CSV)로 출력
# ============================================================
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ---------- 설정 ----------
WIN_SIZE   = 20
DROPOUT    = 0.1
K1, K2     = 3, 3

CHECKPOINTS_DIR = (Path(__file__).resolve().parent / "app" / "models" / "checkpoints")
MODEL_PTH  = CHECKPOINTS_DIR / "cnnmodel.pth"
SCALER_PKL = CHECKPOINTS_DIR / "cnnscaler.pkl"
FEATS_PKL  = CHECKPOINTS_DIR / "features.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 모델 (학습과 동일) ----------
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, window_size, dropout_rate=0.1, k1=3, k2=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=k1, padding=k1//2)
        self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=k2, padding=k2//2)
        self.relu2 = nn.ReLU(); self.pool2 = nn.MaxPool1d(2)
        self.flat = nn.Flatten()
        self.fc1  = nn.Linear(32 * (window_size // 4), 128)
        self.relu3= nn.ReLU(); self.drop = nn.Dropout(dropout_rate)
        self.fc2  = nn.Linear(128, 1)  # logits
    def forward(self, x):
        x = x.permute(0,2,1)  # [B,T,F] -> [B,F,T]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flat(x)
        x = self.drop(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x  # [B,1]

# ---------- 유틸 ----------
def _load_pickle(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def _check_paths():
    missing = [p for p in [MODEL_PTH, SCALER_PKL, FEATS_PKL] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"필요 파일 없음: {', '.join(str(m) for m in missing)}")

# ---------- JSON -> DataFrame (+파생 피처) ----------
def preprocess_json_to_df(obj):
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and "data" in obj:
        rows = obj["data"]
    elif isinstance(obj, list):
        rows = obj
    else:
        raise ValueError("JSON 형식 오류: 리스트 또는 {'data': 리스트} 여야 함.")

    df = pd.DataFrame(rows)
    req = ["prefix","timestamp_ms","ear","pitch","yaw","roll","eye_status"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"필수 컬럼 누락: {miss}")

    df = df.sort_values(["prefix","timestamp_ms"]).reset_index(drop=True)

    base = ["ear","pitch","yaw","roll"]
    for f in base:
        df[f"{f}_diff"] = df.groupby("prefix")[f].diff().fillna(0)
        df[f"{f}_mean_5"] = (
            df.groupby("prefix")[f].rolling(5, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
        df[f"{f}_std_5"] = (
            df.groupby("prefix")[f].rolling(5, min_periods=1).std().fillna(0)
              .reset_index(level=0, drop=True)
        )
    df["eye_status_numeric"] = df["eye_status"].map({"OPEN":1,"CLOSED":0}).fillna(0)
    df["blink_count"] = (
        df.groupby("prefix")["eye_status_numeric"].diff().eq(-1)
          .rolling(5, min_periods=1).sum().fillna(0).reset_index(level=0, drop=True)
    )
    df["angle_magnitude"] = np.sqrt(df["pitch_diff"]**2 + df["yaw_diff"]**2 + df["roll_diff"]**2)
    return df

# ---------- 윈도우 구성 (20프레임 이전은 출력 없음) ----------
def make_windows(df: pd.DataFrame, features, window_size=WIN_SIZE):
    X_all, meta = [], []
    for p in df["prefix"].unique():
        g = df[df["prefix"]==p].sort_values("timestamp_ms")
        n = len(g)
        if n < window_size:
            continue
        data = g[features].values
        ts   = g["timestamp_ms"].values
        for i in range(n - window_size + 1):
            X_all.append(data[i:i+window_size])
            meta.append((p, int(ts[i+window_size-1])))
    return np.array(X_all), meta

# ---------- 로드 ----------
def load_model_and_scaler():
    _check_paths()
    features = _load_pickle(FEATS_PKL)
    scaler   = _load_pickle(SCALER_PKL)
    model = TimeSeriesCNN(input_channels=len(features), window_size=WIN_SIZE,
                          dropout_rate=DROPOUT, k1=K1, k2=K2)
    state = torch.load(MODEL_PTH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, scaler, features

# ---------- 추론 ----------
def predict_from_json(json_input, threshold: float):
    df = preprocess_json_to_df(json_input)
    model, scaler, features = load_model_and_scaler()
    df[features] = scaler.transform(df[features].replace([np.inf,-np.inf],0).fillna(0))
    X, meta = make_windows(df, features, WIN_SIZE)
    if len(X)==0:
        return pd.DataFrame(columns=["prefix","timestamp_ms","prob","pred"])
    xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(xb).squeeze()
        probs  = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({
        "prefix":[m[0] for m in meta],
        "timestamp_ms":[m[1] for m in meta],
        "prob":probs,
        "pred":preds
    })

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Focus CNN inference")
    ap.add_argument("--in", dest="inp", default="-",
                    help="입력 JSON 파일 경로. '-'이면 stdin")
    ap.add_argument("--out", dest="out", default="",
                    help="출력 파일 경로(.csv 또는 .json). 미지정이면 stdout(CSV)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="이진 임계값 (default: 0.5)")
    args = ap.parse_args()

    # 입력 읽기
    if args.inp == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            print("stdin에서 입력이 비어있음.", file=sys.stderr); sys.exit(1)
        json_obj = json.loads(raw)
    else:
        p = Path(args.inp)
        if not p.exists():
            print(f"입력 파일 없음: {p}", file=sys.stderr); sys.exit(1)
        json_obj = json.loads(p.read_text(encoding="utf-8"))

    # 추론
    try:
        df_out = predict_from_json(json_obj, threshold=args.threshold)
    except Exception as e:
        print(f"추론 중 오류: {e}", file=sys.stderr); sys.exit(2)

    # 출력
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.suffix.lower() == ".json":
            outp.write_text(df_out.to_json(orient="records", force_ascii=False), encoding="utf-8")
        else:
            df_out.to_csv(outp, index=False)
        print(str(outp))
    else:
        # stdout (CSV)
        sys.stdout.write(df_out.to_csv(index=False))

if __name__ == "__main__":
    main()
