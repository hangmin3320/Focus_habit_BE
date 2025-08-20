import os, json, pickle, random, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from torch.utils.data import TensorDataset, DataLoader

# -------------------
# 시드 고정
# -------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------
# 경로/설정 (프로젝트 구조 기준)
# -------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CKPT_DIR     = os.path.join(PROJECT_ROOT, "models", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

USERNAME = "username"

# 베이스 모델/스케일러
BASE_MODEL_PATH  = os.path.join(CKPT_DIR, "baseline_model.pth")
BASE_SCALER_PATH = os.path.join(CKPT_DIR, "scaler.pkl")

# 단일 입력 JSON (집중=0, 비집중=1) — 프로젝트 data 폴더 기준
FOCUS_JSON_PATH    = os.path.join(PROJECT_ROOT, "data", "focus.json")       # label=0
NONFOCUS_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "nonfocus.json")    # label=1

# 슬라이딩 윈도우
WINDOW_SIZE = 25
OVERLAP     = 0.5

# 학습 에폭/얼리스탑
EPOCHS              = 20
EARLY_STOP_PATIENCE = 7
LR                  = 1e-4
WEIGHT_DECAY        = 1e-5

# -------------------
# 데이터 split (0.6 / 0.4)
# -------------------
def split_train_test(df, test_ratio: float = 0.4):
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    k = max(1, int(len(df) * (1 - test_ratio)))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

# -------------------
# 모델
# -------------------
class TimeSeriesCNNGRU(nn.Module):
    def __init__(self, input_channels, window_size=25, dropout_rate=0.3, k1=3, k2=32, two_convs=True):
        super().__init__()
        c=32; ks=k1; pad=ks//2
        self.two_convs=two_convs
        self.conv1=nn.Conv1d(input_channels,c,kernel_size=ks,padding=pad,bias=False)
        self.bn1=nn.BatchNorm1d(c)
        self.relu=nn.ReLU()
        self.pool1=nn.MaxPool1d(2)
        if self.two_convs:
            self.conv2=nn.Conv1d(c,c,kernel_size=ks,padding=pad,bias=False)
            self.bn2=nn.BatchNorm1d(c)
            self.pool2=nn.MaxPool1d(2)
        self.gru=nn.GRU(input_size=c,hidden_size=k2,num_layers=1,batch_first=True,bidirectional=False)
        self.dropout=nn.Dropout(dropout_rate)
        self.fc=nn.Linear(k2,1)
    def forward(self,x):
        # x: (B, T, F)
        x=x.permute(0,2,1)
        x=self.pool1(self.relu(self.bn1(self.conv1(x))))
        if self.two_convs:
            x=self.pool2(self.relu(self.bn2(self.conv2(x))))
        x=x.permute(0,2,1)          # (B, T', C)
        _,h=self.gru(x)             # h: (1, B, H)
        h=self.dropout(h[-1])       # (B, H)
        return self.fc(h)           # (B, 1)

# -------------------
# 유틸/전처리
# -------------------
def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c]=0.0
    return df

def feature_engineer(df, base=('ear','pitch','yaw','roll'), username="USER"):
    df = ensure_cols(df, list(base)+['eye_status','prefix'])
    if 'timestamp_ms' not in df.columns: df['timestamp_ms']=np.arange(len(df))
    if 'prefix' not in df.columns: df['prefix']=username
    df = df.sort_values(['prefix','timestamp_ms'])
    for f in base:
        if f not in df.columns: df[f]=0.0
    for f in base:
        df[f'{f}_diff']    = df.groupby('prefix')[f].diff().fillna(0)
        m = df.groupby('prefix')[f].rolling(window=5,min_periods=1).mean().reset_index(level=0,drop=True)
        s = df.groupby('prefix')[f].rolling(window=5,min_periods=1).std().fillna(0).reset_index(level=0,drop=True)
        df[f'{f}_mean_5']  = m
        df[f'{f}_std_5']   = s
    df['eye_status_numeric'] = df['eye_status'].map({'OPEN':1,'CLOSED':0}).fillna(0)
    t = df.groupby('prefix')['eye_status_numeric'].diff().eq(-1)
    df['blink_count']        = t.rolling(window=100,min_periods=1).sum().fillna(0).reset_index(level=0,drop=True)
    df['angle_magnitude']    = np.sqrt(df['pitch_diff']**2+df['yaw_diff']**2+df['roll_diff']**2)
    feats = list(base) \
          + [f'{f}_diff' for f in base] \
          + [f'{f}_mean_5' for f in base] \
          + [f'{f}_std_5'  for f in base] \
          + ['blink_count','angle_magnitude']
    df[feats] = df[feats].replace([np.inf,-np.inf],0).fillna(0)
    return df, feats

def compute_stride(window_size, overlap=0.5):
    return max(1, int(window_size*(1-overlap)))

def create_sequences(df, features, window_size=25, stride=12, threshold=0.5, return_meta=False):
    X, y, meta = [], [], []
    for p in df['prefix'].unique():
        g  = df[df['prefix']==p].sort_values('timestamp_ms').reset_index(drop=True)
        if len(g) < window_size: continue
        arr = g[features].values
        lab = g['label'].values if 'label' in g.columns else None
        ts  = g['timestamp_ms'].values
        for i in range(0, len(g)-window_size+1, stride):
            X.append(arr[i:i+window_size])
            if lab is not None:
                seq = lab[i:i+window_size]
                y.append(1 if np.mean(seq)>=threshold else 0)
            if return_meta:
                meta.append({
                    "prefix": p,
                    "start_idx": int(i),
                    "end_idx": int(i+window_size-1),
                    "start_ts": float(ts[i]) if len(ts)>0 else np.nan,
                    "end_ts":   float(ts[i+window_size-1]) if len(ts)>0 else np.nan
                })
    X = np.array(X) if len(X) else np.empty((0, window_size, len(features)))
    y = np.array(y) if len(y) else np.array([])
    return (X, y, meta) if return_meta else (X, y)

def json_to_df(path_or_obj, label=None, username="USER"):
    if isinstance(path_or_obj, str) and os.path.isfile(path_or_obj):
        with open(path_or_obj, 'r', encoding='utf-8') as f: data = json.load(f)
    elif isinstance(path_or_obj, (str, bytes)):
        data = json.loads(path_or_obj)
    else:
        data = path_or_obj
    rows=[]
    for it in data:
        ts = it.get('timestamp_ms', it.get('timestamp', None))
        es = it.get('eye_status', {}) or {}
        hp = it.get('head_pose', {}) or {}
        if isinstance(es, str):
            status = es; ear_val = None
        else:
            status  = es.get('status','OPEN')
            ear_val = es.get('ear_value', es.get('ear', None))
        rows.append({
            'timestamp_ms': ts if ts is not None else np.nan,
            'eye_status': status,
            'ear': float(ear_val) if ear_val is not None else float(it.get('ear', 0.0)),
            'pitch': float(hp.get('pitch', it.get('pitch', 0.0))),
            'yaw':   float(hp.get('yaw',   it.get('yaw',   0.0))),
            'roll':  float(hp.get('roll',  it.get('roll',  0.0))),
            'prefix': username,
            'label':  label if label is not None else np.nan    # 집중=0, 비집중=1
        })
    df = pd.DataFrame(rows)
    if 'timestamp_ms' not in df.columns: df['timestamp_ms'] = np.arange(len(df))
    return df

def freeze_batchnorm(m):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()
        for p in m.parameters(): p.requires_grad = False

def set_trainable(model, mode):
    for p in model.parameters(): p.requires_grad=False
    if mode=='head':
        for p in model.fc.parameters():  p.requires_grad=True
    elif mode=='gru':
        for p in model.gru.parameters(): p.requires_grad=True
        for p in model.fc.parameters():  p.requires_grad=True
    elif mode=='cnn_gru':
        for p in model.conv1.parameters(): p.requires_grad=True
        for p in model.gru.parameters():   p.requires_grad=True
        for p in model.fc.parameters():    p.requires_grad=True
    elif mode=='full':
        for p in model.parameters(): p.requires_grad=True

def _sweep_threshold_f1(probs, labels, grid=None):
    if grid is None: grid = np.linspace(0.05, 0.95, 91)
    labels = np.asarray(labels).astype(int).ravel()
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def _best_temperature_from_logits(logits, labels, temps=None):
    if temps is None: temps = np.linspace(0.5, 3.0, 51)
    def nll(temp):
        z = logits/float(max(temp,1e-6))
        p = 1/(1+np.exp(-z))
        eps = 1e-7; p = np.clip(p, eps, 1-eps)
        return float(-(labels*np.log(p)+(1-labels)*np.log(1-p)).mean())
    nlls = [nll(t) for t in temps]
    return float(temps[int(np.argmin(nlls))])

def _fit_scaler_on_df(df, feats):
    scaler = StandardScaler()
    df[feats] = scaler.fit_transform(df[feats])
    return scaler

def _transform_df(df, feats, scaler):
    df[feats] = scaler.transform(df[feats])
    return df

def _normalize_state_dict(sd):
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    sd2 = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith('module.') else k
        sd2[nk] = v
    return sd2

def _make_model_and_load(sd_path, input_dim, window_size, device):
    model = TimeSeriesCNNGRU(input_dim, window_size, 0.3, 3, 32, True).to(device)
    try:
        sd = torch.load(sd_path, map_location=device)
        sd = _normalize_state_dict(sd)
        m  = model.state_dict()
        matched = {k:v for k,v in sd.items() if k in m and hasattr(v,'shape') and v.shape==m[k].shape}
        if matched:
            m.update(matched); model.load_state_dict(m, strict=False)
    except Exception as e:
        print(f"[WARN] checkpoint load failed: {e}. Using random init.")
    return model

def _safe_auc(y_true, probs):
    y_true = np.asarray(y_true).astype(int).ravel()
    probs  = np.asarray(probs).ravel()
    roc_auc = float('nan'); pr_auc = float('nan')
    try: roc_auc = float(roc_auc_score(y_true, probs))
    except Exception: pass
    try: pr_auc  = float(average_precision_score(y_true, probs))
    except Exception: pass
    return roc_auc, pr_auc

# -------------------
# 데이터 준비 (단일 JSON 2개)
# -------------------
def load_from_two_jsons(username, focus_json_path, nonfocus_json_path):
    # 집중=0, 비집중=1
    df_f = json_to_df(focus_json_path,    label=0, username=username)
    df_n = json_to_df(nonfocus_json_path, label=1, username=username)
    df_all = pd.concat([df_f, df_n], ignore_index=True)
    df_all, feats = feature_engineer(df_all, username=username)
    if 'eye_status' in df_all.columns:
        df_all = df_all[df_all['eye_status']!='NO_FACE_DETECTED']
    df_all = df_all.sort_values('timestamp_ms').reset_index(drop=True)
    return df_all, feats

def time_split_df(df, val_ratio=0.2):
    df = df.sort_values('timestamp_ms')
    k  = max(1, int(len(df)*(1-val_ratio)))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

# -------------------
# 헤더 생성
# -------------------
def make_base_header(username, feats, window_size, overlap, threshold=0.5, temperature=1.0):
    return {"username":username,"window_size":window_size,"overlap":overlap,"features":feats,
            "threshold":threshold,"temperature":temperature,"mode":"base"}

# -------------------
# 프로파일 2: cal (base 고정, 검증셋으로 temp+F1-thr 튜닝)
# -------------------
def run_cal(username, df_train, feats, base_model_path, base_scaler_path, window_size, overlap):
    stride = compute_stride(window_size,overlap)
    with open(base_scaler_path,'rb') as f: scaler = pickle.load(f)
    df_tr = _transform_df(df_train.copy(), feats, scaler)
    _, df_val = time_split_df(df_tr, 0.2)

    X, y = create_sequences(df_val, feats, window_size, stride, threshold=0.5)
    if len(X)==0: return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _make_model_and_load(base_model_path, len(feats), window_size, device); model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()

    probs     = 1/(1+np.exp(-logits))
    best_t, _ = _sweep_threshold_f1(probs, y)
    best_temp = _best_temperature_from_logits(logits, y)

    header = {"username":username,"window_size":window_size,"overlap":overlap,"features":feats,
              "threshold":best_t,"temperature":best_temp,"mode":"cal"}
    json.dump(header, open(os.path.join(CKPT_DIR,f"{username}_header_cal.json"),"w"), ensure_ascii=False)
    return {"header":os.path.join(CKPT_DIR,f"{username}_header_cal.json"),
            "threshold":best_t,"temperature":best_temp}

# -------------------
# 학습 공통 (진행 로그 + 얼리스탑)
# -------------------
def train_model(username, df_train, feats, base_model_path, train_mode, window_size, overlap,
                epochs=15, patience=7, lr=1e-4, wd=1e-5, use_base_scaler=True):
    stride = compute_stride(window_size, overlap)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_base_scaler:
        with open(BASE_SCALER_PATH,'rb') as f: scaler = pickle.load(f)
        df_tr = _transform_df(df_train.copy(), feats, scaler)
    else:
        scaler = _fit_scaler_on_df(df_train.copy(), feats)
        df_tr  = _transform_df(df_train.copy(), feats, scaler)

    X, y = create_sequences(df_tr, feats, window_size, stride, threshold=0.5)
    if len(X)==0: return None

    n = len(X); split = int(n*0.8) if n>5 else n
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = (X[split:], y[split:]) if split<n else (X.copy(), y.copy())

    model = _make_model_and_load(base_model_path, len(feats), window_size, device)
    model.apply(freeze_batchnorm)
    set_trainable(model, train_mode)

    pos = max(1,int((y_tr==1).sum())); neg = max(1,int((y_tr==0).sum()))
    pos_weight = torch.tensor([neg/pos], device=device, dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_tr,dtype=torch.float32),
                                        torch.tensor(y_tr,dtype=torch.float32)),
                          batch_size=32, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.tensor(X_val,dtype=torch.float32),
                                        torch.tensor(y_val,dtype=torch.float32)),
                          batch_size=64, shuffle=False)

    best_sd=None; best_val=np.inf; noimp=0
    for ep in range(1, epochs+1):
        # train
        model.train()
        tr_loss_sum=0.0; tr_n=0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            lg  = model(xb).squeeze(-1)
            loss= criterion(lg, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss_sum += loss.item() * xb.size(0); tr_n += xb.size(0)

        # val
        model.eval(); vloss=0.0; v_n=0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb).squeeze(-1)
                ls = criterion(lg, yb)
                vloss += ls.item() * xb.size(0); v_n += xb.size(0)
        tr_loss = tr_loss_sum / max(1,tr_n)
        vloss   = vloss / max(1,v_n)
        print(f"[{train_mode}] epoch {ep:02d}/{epochs} | train_loss={tr_loss:.4f} val_loss={vloss:.4f} noimp={noimp}/{patience}")

        if vloss < best_val:
            best_val = vloss; best_sd = {k:v.detach().clone() for k,v in model.state_dict().items()}; noimp=0
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[{train_mode}] Early stop at epoch {ep}")
                break

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=True)

    # 저장
    model_path  = os.path.join(CKPT_DIR,f"{username}_model_{train_mode}.pth")
    scaler_path = os.path.join(CKPT_DIR,f"{username}_scaler_{train_mode}.pkl")
    torch.save(model.state_dict(), model_path)
    with open(scaler_path,'wb') as f: pickle.dump(scaler, f)

    # 참고용 성능
    with torch.no_grad():
        lg_val   = model(torch.tensor(X_val,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
        probs_va = 1/(1+np.exp(-lg_val))
    acc_val = float(((probs_va>=0.5).astype(int) == y_val.astype(int)).mean())

    with torch.no_grad():
        lg_tr = model(torch.tensor(X_tr,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
        probs_tr = 1/(1+np.exp(-lg_tr))
    roc_auc_tr, pr_auc_tr = _safe_auc(y_tr, probs_tr)

    return {
        "model":model_path,
        "scaler":scaler_path,
        "val_acc@0.5":acc_val,
        "features":feats,
        "train_roc_auc": roc_auc_tr,
        "train_pr_auc":  pr_auc_tr
    }

# -------------------
# 프로파일 3: head
# -------------------
def run_head(username, df_train, feats, window_size, overlap,
             epochs=15, patience=7, lr=1e-4, wd=1e-5, min_train_pr_auc=None):
    out = train_model(username, df_train, feats, BASE_MODEL_PATH, train_mode='head',
                      window_size=window_size, overlap=overlap,
                      epochs=epochs, patience=patience, lr=lr, wd=wd, use_base_scaler=True)
    if out is None: return None

    if min_train_pr_auc is not None:
        if np.isnan(out.get("train_pr_auc", float('nan'))) or out["train_pr_auc"] < float(min_train_pr_auc):
            print(f"[INFO] head 제외: train PR AUC={out.get('train_pr_auc')} < {min_train_pr_auc}")
            return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(out["scaler"],'rb') as f: scaler = pickle.load(f)
    model = _make_model_and_load(out["model"], len(feats), window_size, device); model.eval()

    stride  = compute_stride(window_size, overlap)
    df_trsc = _transform_df(df_train.copy(), feats, scaler)
    _, df_val = time_split_df(df_trsc, 0.2)
    X, y = create_sequences(df_val, feats, window_size, stride, threshold=0.5)

    with torch.no_grad():
        lg = model(torch.tensor(X,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
        probs = 1/(1+np.exp(-lg))
    best_t, _ = _sweep_threshold_f1(probs, y)

    header = {"username":USERNAME,"window_size":window_size,"overlap":overlap,"features":feats,
              "threshold":best_t,"temperature":1.0,"mode":"head"}
    json.dump(header, open(os.path.join(CKPT_DIR,f"{username}_header_head.json"),"w"), ensure_ascii=False)
    return {"header":os.path.join(CKPT_DIR,f"{username}_header_head.json"), **out, "threshold":best_t}

# -------------------
# 프로파일 4: cal_head
# -------------------
def run_cal_head(username, df_train, feats, window_size, overlap,
                 epochs=15, patience=7, lr=1e-4, wd=1e-5, min_train_pr_auc=None):
    out = run_head(username, df_train, feats, window_size, overlap,
                   epochs=epochs, patience=patience, lr=lr, wd=wd, min_train_pr_auc=min_train_pr_auc)
    if out is None: return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(out["scaler"],'rb') as f: scaler = pickle.load(f)
    model = _make_model_and_load(out["model"], len(feats), window_size, device); model.eval()

    stride  = compute_stride(window_size, overlap)
    df_trsc = _transform_df(df_train.copy(), feats, scaler)
    _, df_val = time_split_df(df_trsc, 0.2)
    X, y = create_sequences(df_val, feats, window_size, stride, threshold=0.5)

    with torch.no_grad():
        logits = model(torch.tensor(X,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
    best_temp = _best_temperature_from_logits(logits, y)
    probs     = 1/(1+np.exp(-(logits/max(best_temp,1e-6))))
    best_t, _ = _sweep_threshold_f1(probs, y)

    header = {"username":USERNAME,"window_size":window_size,"overlap":overlap,"features":feats,
              "threshold":best_t,"temperature":best_temp,"mode":"cal_head"}
    json.dump(header, open(os.path.join(CKPT_DIR,f"{username}_header_cal_head.json"),"w"), ensure_ascii=False)
    return {"header":os.path.join(CKPT_DIR,f"{username}_header_cal_head.json"),
            "model":out["model"],"scaler":out["scaler"],"threshold":best_t,"temperature":best_temp}

# -------------------
# 프로파일 5: transfer (gru/cnn_gru/full)
# -------------------
def run_transfer(username, df_train, feats, window_size, overlap, train_mode='gru',
                 epochs=15, patience=7, lr=1e-4, wd=1e-5, min_train_pr_auc=None):
    out = train_model(username, df_train, feats, BASE_MODEL_PATH, train_mode=train_mode,
                      window_size=window_size, overlap=overlap,
                      epochs=epochs, patience=patience, lr=lr, wd=wd, use_base_scaler=False)
    if out is None: return None
    if min_train_pr_auc is not None:
        if np.isnan(out.get("train_pr_auc", float('nan'))) or out["train_pr_auc"] < float(min_train_pr_auc):
            print(f"[INFO] transfer({train_mode}) 제외: train PR AUC={out.get('train_pr_auc')} < {min_train_pr_auc}")
            return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(out["scaler"],'rb') as f: scaler = pickle.load(f)
    model = _make_model_and_load(out["model"], len(feats), window_size, device); model.eval()

    stride  = compute_stride(window_size, overlap)
    df_trsc = _transform_df(df_train.copy(), feats, scaler)
    _, df_val = time_split_df(df_trsc, 0.2)
    X, y = create_sequences(df_val, feats, window_size, stride, threshold=0.5)

    with torch.no_grad():
        lg = model(torch.tensor(X,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
        probs = 1/(1+np.exp(-lg))
    best_t, _ = _sweep_threshold_f1(probs, y)

    header = {"username":USERNAME,"window_size":window_size,"overlap":overlap,"features":feats,
              "threshold":best_t,"temperature":1.0,"mode":f"transfer_{train_mode}"}
    json.dump(header, open(os.path.join(CKPT_DIR,f"{username}_header_transfer_{train_mode}.json")) , ensure_ascii=False)
    return {"header":os.path.join(CKPT_DIR,f"{username}_header_transfer_{train_mode}.json"),
            **out,"threshold":best_t}

# -------------------
# 평가 (DF 입력)
# -------------------
def _evaluate_df(model_path, scaler_path, header, df_test, return_details=False, profile_name=None):
    username    = header["username"]
    window_size = int(header.get("window_size", WINDOW_SIZE))
    overlap     = float(header.get("overlap", OVERLAP))
    feats       = header.get("features")
    if feats is None:
        tmp,_ = feature_engineer(pd.DataFrame([{"ear":0,"pitch":0,"yaw":0,"roll":0,"prefix":username}]), username=username)
        _,feats = feature_engineer(tmp, username=username)

    df = df_test.copy()
    df,_ = feature_engineer(df, username=username)
    df   = df.sort_values("timestamp_ms").reset_index(drop=True)

    with open(scaler_path,'rb') as f: scaler = pickle.load(f)
    df[feats] = scaler.transform(df[feats])

    stride = compute_stride(window_size, overlap)
    X, y, meta = create_sequences(df, feats, window_size, stride, threshold=0.5, return_meta=True)
    if len(X)==0: 
        return (None, pd.DataFrame()) if return_details else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _make_model_and_load(model_path, len(feats), window_size, device); model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X,dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()

    temp  = float(header.get("temperature",1.0))
    probs = 1/(1+np.exp(-(logits/max(temp,1e-6))))
    thr   = float(header.get("threshold",0.5))
    preds = (probs>=thr).astype(int)
    y     = y.astype(int)

    roc_auc, pr_auc = _safe_auc(y, probs)
    metrics = {
        "acc":float(accuracy_score(y,preds)),
        "precision":float(precision_score(y,preds,zero_division=0)),
        "recall":float(recall_score(y,preds,zero_division=0)),
        "f1":float(f1_score(y,preds,zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n":int(len(y))
    }

    if return_details:
        md = pd.DataFrame(meta)
        details = pd.DataFrame({"y_true":y, "y_pred":preds, "prob":probs})
        details = pd.concat([md.reset_index(drop=True), details.reset_index(drop=True)], axis=1)
        details["profile"] = profile_name if profile_name else header.get("mode","unknown")
        return metrics, details

    return metrics

def evaluate_profile_df(profile_name, df_test):
    if profile_name=='base':
        # base는 기본 헤더 즉석 생성
        tmp,_ = feature_engineer(pd.DataFrame([{"ear":0,"pitch":0,"yaw":0,"roll":0,"prefix":USERNAME}]), username=USERNAME)
        _,feats = feature_engineer(tmp, username=USERNAME)
        header  = make_base_header(USERNAME, feats, WINDOW_SIZE, OVERLAP, threshold=0.5, temperature=1.0)
        return _evaluate_df(BASE_MODEL_PATH, BASE_SCALER_PATH, header, df_test, return_details=True, profile_name='base')

    elif profile_name=='cal':
        header = json.load(open(os.path.join(CKPT_DIR,f"{USERNAME}_header_cal.json")))
        return _evaluate_df(BASE_MODEL_PATH, BASE_SCALER_PATH, header, df_test, return_details=True, profile_name='cal')

    elif profile_name=='head':
        header = json.load(open(os.path.join(CKPT_DIR,f"{USERNAME}_header_head.json")))
        model_path  = os.path.join(CKPT_DIR,f"{USERNAME}_model_head.pth")
        scaler_path = os.path.join(CKPT_DIR,f"{USERNAME}_scaler_head.pkl")
        return _evaluate_df(model_path, scaler_path, header, df_test, return_details=True, profile_name='head')

    elif profile_name=='cal_head':
        header = json.load(open(os.path.join(CKPT_DIR,f"{USERNAME}_header_cal_head.json")))
        model_path  = os.path.join(CKPT_DIR,f"{USERNAME}_model_head.pth")
        scaler_path = os.path.join(CKPT_DIR,f"{USERNAME}_scaler_head.pkl")
        return _evaluate_df(model_path, scaler_path, header, df_test, return_details=True, profile_name='cal_head')

    elif profile_name.startswith('transfer'):
        mode = profile_name.split('_',1)[1]
        header = json.load(open(os.path.join(CKPT_DIR,f"{USERNAME}_header_transfer_{mode}.json")))
        model_path  = os.path.join(CKPT_DIR,f"{USERNAME}_model_{mode}.pth")
        scaler_path = os.path.join(CKPT_DIR,f"{USERNAME}_scaler_{mode}.pkl")
        return _evaluate_df(model_path, scaler_path, header, df_test, return_details=True, profile_name=f"transfer_{mode}")

    else:
        raise ValueError("unknown profile_name")

# -------------------
# 비교 & 활성화 (DF 입력) 
# -------------------
def compare_and_activate_df(df_test, metric="f1"):
    results=[]; details_list=[]
    candidates=['base']
    if os.path.isfile(os.path.join(CKPT_DIR,f"{USERNAME}_header_cal.json")):      candidates.append('cal')
    if os.path.isfile(os.path.join(CKPT_DIR,f"{USERNAME}_header_head.json")):     candidates.append('head')
    if os.path.isfile(os.path.join(CKPT_DIR,f"{USERNAME}_header_cal_head.json")): candidates.append('cal_head')
    for m in ['gru','cnn_gru','full']:
        if os.path.isfile(os.path.join(CKPT_DIR,f"{USERNAME}_header_transfer_{m}.json")):
            candidates.append(f"transfer_{m}")

    for name in candidates:
        out = evaluate_profile_df(name, df_test)
        if out is None:
            continue
        metrics, details = out
        if metrics: results.append({"profile":name, **metrics})
        if details is not None and len(details)>0:
            details_list.append(details)

    if not results: return None


    if len(details_list)>0:
        all_details = pd.concat(details_list, ignore_index=True)

    df = pd.DataFrame(results)
    best_row = df.iloc[df[metric].values.argmax()]
    best     = best_row["profile"]

    # 활성화 산출물
    if best=='base':
        tmp,_ = feature_engineer(pd.DataFrame([{"ear":0,"pitch":0,"yaw":0,"roll":0,"prefix":USERNAME}]), username=USERNAME)
        _,feats = feature_engineer(tmp, username=USERNAME)
        header  = make_base_header(USERNAME, feats, WINDOW_SIZE, OVERLAP, 0.5, 1.0)
        json.dump(header, open(os.path.join(CKPT_DIR,"active_header.json"),"w"), ensure_ascii=False)
        shutil.copy2(BASE_MODEL_PATH,  os.path.join(CKPT_DIR,"active_model.pth"))
        shutil.copy2(BASE_SCALER_PATH, os.path.join(CKPT_DIR,"active_scaler.pkl"))

    elif best == 'cal':
        hdr = json.load(open(os.path.join(CKPT_DIR,f"{USERNAME}_header_cal.json")))
        json.dump(hdr, open(os.path.join(CKPT_DIR,"active_header.json"),"w"), ensure_ascii=False)
        shutil.copy2(BASE_MODEL_PATH,  os.path.join(CKPT_DIR,"active_model.pth"))
        shutil.copy2(BASE_SCALER_PATH, os.path.join(CKPT_DIR,"active_scaler.pkl"))

    elif best in ['head','cal_head']:
        header_file = f"{USERNAME}_header_cal_head.json" if best=='cal_head' else f"{USERNAME}_header_head.json"
        hdr = json.load(open(os.path.join(CKPT_DIR,header_file)))
        json.dump(hdr, open(os.path.join(CKPT_DIR,"active_header.json"),"w"), ensure_ascii=False)
        shutil.copy2(os.path.join(CKPT_DIR,f"{USERNAME}_model_head.pth"),  os.path.join(CKPT_DIR,"active_model.pth"))
        shutil.copy2(os.path.join(CKPT_DIR,f"{USERNAME}_scaler_head.pkl"), os.path.join(CKPT_DIR,"active_scaler.pkl"))

    else:  # transfer_*
        mode = best.split('_',1)[1]
        hdr  = json.load(open(os.path.join(CKPT_DIR,f"{USERNAME}_header_transfer_{mode}.json")))
        json.dump(hdr, open(os.path.join(CKPT_DIR,"active_header.json"),"w"), ensure_ascii=False)  # ← 오타 수정
        shutil.copy2(os.path.join(CKPT_DIR,f"{USERNAME}_model_{mode}.pth"),  os.path.join(CKPT_DIR,"active_model.pth"))
        shutil.copy2(os.path.join(CKPT_DIR,f"{USERNAME}_scaler_{mode}.pkl"), os.path.join(CKPT_DIR,"active_scaler.pkl"))

    # 활성화 프로파일 저장
    json.dump({"best_profile":best,"metric":metric,"table":df.to_dict(orient="records")},
              open(os.path.join(CKPT_DIR,"active_profile.json"),"w"))
    return {"best":best, "table":df.to_dict(orient="records")}

# -------------------
# main
# -------------------
if __name__ == "__main__":
    set_seed(42)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1) JSON 2개 불러와서 합치기 (집중=0, 비집중=1)
    df_all, feats = load_from_two_jsons(USERNAME, FOCUS_JSON_PATH, NONFOCUS_JSON_PATH)

    # 2) 0.6/0.4 split (timestamp 순서)
    df_train, df_test = split_train_test(df_all, test_ratio=0.4)

    # 3) 프로파일 생성/학습
    _ = run_cal(USERNAME, df_train, feats, BASE_MODEL_PATH, BASE_SCALER_PATH, WINDOW_SIZE, OVERLAP)

    h = run_head(USERNAME, df_train, feats, WINDOW_SIZE, OVERLAP,
                 epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, lr=LR, wd=WEIGHT_DECAY,
                 min_train_pr_auc=None)
    if h:
        print(f"[head] TRAIN ROC AUC={h['train_roc_auc']}, TRAIN PR AUC={h['train_pr_auc']}")

    ch = run_cal_head(USERNAME, df_train, feats, WINDOW_SIZE, OVERLAP,
                      epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, lr=LR, wd=WEIGHT_DECAY,
                      min_train_pr_auc=None)
    if ch:
        print("[cal_head] temperature + F1 threshold tuned on val")

    t1 = run_transfer(USERNAME, df_train, feats, WINDOW_SIZE, OVERLAP, train_mode='gru',
                      epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, lr=LR, wd=WEIGHT_DECAY,
                      min_train_pr_auc=None)
    t2 = run_transfer(USERNAME, df_train, feats, WINDOW_SIZE, OVERLAP, train_mode='cnn_gru',
                      epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, lr=LR, wd=WEIGHT_DECAY,
                      min_train_pr_auc=None)
    t3 = run_transfer(USERNAME, df_train, feats, WINDOW_SIZE, OVERLAP, train_mode='full',
                      epochs=EPOCHS, patience=EARLY_STOP_PATIENCE, lr=LR, wd=WEIGHT_DECAY,
                      min_train_pr_auc=None)

    summary = compare_and_activate_df(df_test, metric="f1")
    print(summary)
