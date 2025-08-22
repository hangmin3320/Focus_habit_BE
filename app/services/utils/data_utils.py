import os
import json
import numpy as np
import pandas as pd
import torch # Added this import as set_seed uses torch

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    df['blink_count']        = t.rolling(window=5,min_periods=1).sum().fillna(0).reset_index(level=0,drop=True)
    df['angle_magnitude']    = np.sqrt(df['pitch_diff']**2+df['yaw_diff']**2+df['roll_diff']**2)
    feats = list(base) 
          + [f'{f}_diff' for f in base] 
          + [f'{f}_mean_5' for f in base] 
          + [f'{f}_std_5'  for f in base] 
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

def stratified_time_split(df, test_ratio: float = 0.4):
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    # 각 라벨별로 분리
    unique_labels = df['label'].unique()
    train_dfs = []
    test_dfs = []

    for label in unique_labels:
        label_df = df[df['label'] == label].copy()
        if len(label_df) == 0: # 해당 라벨의 데이터가 없으면 건너뛰기
            continue
        
        # 시간 순서대로 정렬
        label_df = label_df.sort_values("timestamp_ms").reset_index(drop=True)
        
        # 분할 지점 계산
        split_idx = max(1, int(len(label_df) * (1 - test_ratio)))
        
        train_dfs.append(label_df.iloc[:split_idx].copy())
        test_dfs.append(label_df.iloc[split_idx:].copy())

    # 분리된 데이터프레임들을 다시 합치기
    train_df_final = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=df.columns)
    test_df_final = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame(columns=df.columns)

    # 최종 데이터프레임들을 시간 순서대로 다시 정렬
    train_df_final = train_df_final.sort_values("timestamp_ms").reset_index(drop=True)
    test_df_final = test_df_final.sort_values("timestamp_ms").reset_index(drop=True)

    return train_df_final, test_df_final

def load_train_csv(file_path):
    df = pd.read_csv(file_path)
    if 'ear' not in df.columns and 'ear_value' in df.columns: df = df.rename(columns={'ear_value':'ear'})
    if 'prefix' not in df.columns: df['prefix'] = 'TRAIN_0'
    if 'timestamp_ms' not in df.columns and 'timestamp' in df.columns: df = df.rename(columns={'timestamp':'timestamp_ms'})
    if 'eye_status' in df.columns: df = df[df['eye_status']!='NO_FACE_DETECTED']
    df['label'] = df['label'].map({1:0,2:0,3:1,4:1,5:1}).fillna(0).astype(int)
    df = df.sort_values(['prefix','timestamp_ms'])
    df, feats = feature_engineer(df)
    return df, feats

def split_train_val_from_csv(csv_path, test_ratio=0.3, seed=42):
    df, feats = load_train_csv(csv_path)
    rng = np.random.RandomState(seed)
    if 'prefix' in df.columns and df['prefix'].nunique() > 1:
        prefixes = df['prefix'].dropna().unique().tolist()
        rng.shuffle(prefixes)
        cut = int(len(prefixes) * (1 - test_ratio))
        train_pref = set(prefixes[:cut]); val_pref = set(prefixes[cut:])
        train_df = df[df['prefix'].isin(train_pref)].copy()
        val_df   = df[df['prefix'].isin(val_pref)].copy()
    else:
        df = df.sort_values('timestamp_ms').reset_index(drop=True)
        cut = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:cut].copy()
        val_df   = df.iloc[cut:].copy()
    base_cols = ['timestamp_ms','eye_status','prefix','label']
    for c in feats:
        if c not in val_df.columns: val_df[c] = 0.0
        if c not in train_df.columns: train_df[c] = 0.0
    train_df = train_df[base_cols + feats]
    val_df   = val_df[base_cols + feats]
    return train_df, val_df, feats