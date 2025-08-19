# focus_train_csv_split7030_cnn_gru.py
import os, warnings, pickle, time, math, json, shutil
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = 0.0
    return df

def feature_engineer(df, base_feats=('ear','pitch','yaw','roll')):
    df = ensure_cols(df, list(base_feats) + ['eye_status','prefix','timestamp_ms'])
    df = df.sort_values(['prefix','timestamp_ms'])
    for f in base_feats:
        df[f] = df[f].fillna(0.0)
        df[f'{f}_diff'] = df.groupby('prefix')[f].diff().fillna(0)
        m = df.groupby('prefix')[f].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        s = df.groupby('prefix')[f].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        df[f'{f}_mean_5'] = m
        df[f'{f}_std_5']  = s
    df['eye_status_numeric'] = df['eye_status'].map({'OPEN':1,'CLOSED':0}).fillna(0).astype(int)
    df['blink_count'] = (
        df.groupby('prefix')['eye_status_numeric']
          .apply(lambda s: s.diff().eq(-1).rolling(window=5, min_periods=1).sum())
          .reset_index(level=0, drop=True)
          .fillna(0)
    )
    df['angle_magnitude'] = np.sqrt(df['pitch_diff']**2 + df['yaw_diff']**2 + df['roll_diff']**2)
    feats = list(base_feats) + [f'{f}_diff' for f in base_feats] + [f'{f}_mean_5' for f in base_feats] + [f'{f}_std_5' for f in base_feats] + ['blink_count','angle_magnitude']
    df[feats] = df[feats].replace([np.inf,-np.inf],0).fillna(0)
    return df, feats

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

def create_sequences(df, features, window_size, stride=None, overlap_ratio=None, threshold=0.5):
    if stride is None:
        stride = 1 if overlap_ratio is None else max(1, int(window_size * (1 - overlap_ratio)))
    X, y = [], []
    for p in df['prefix'].unique():
        g = df[df['prefix']==p].sort_values('timestamp_ms')
        if len(g) < window_size: continue
        data = g[features].values
        target = g['label'].values
        for i in range(0, len(g)-window_size+1, stride):
            seq_y = target[i:i+window_size]
            label = 1 if np.mean(seq_y) >= threshold else 0
            X.append(data[i:i+window_size]); y.append(label)
    if len(X)==0: return np.empty((0, window_size, len(features))), np.array([], dtype=int)
    return np.array(X), np.array(y, dtype=int)

def compute_stride(window_size, mode, train_overlap=0.75, val_overlap=0.5, test_overlap=0.5):
    if mode=='train': return max(1, int(window_size * (1 - train_overlap)))
    if mode=='val':   return max(1, int(window_size * (1 - val_overlap)))
    if mode=='test':  return max(1, int(window_size * (1 - test_overlap)))
    return 1

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        if len(sequences)==0 or len(labels)==0: raise ValueError('empty dataset')
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i], self.labels[i]

class TimeSeriesCNNGRU(nn.Module):
    def __init__(self, input_channels, window_size, dropout_rate=0.3, k1=3, k2=64, two_convs=True):
        super().__init__()
        c = 32; ks = k1; pad = ks // 2
        self.two_convs = two_convs
        self.conv1 = nn.Conv1d(input_channels, c, kernel_size=ks, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(c)
        self.relu  = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        if self.two_convs:
            self.conv2 = nn.Conv1d(c, c, kernel_size=ks, padding=pad, bias=False)
            self.bn2   = nn.BatchNorm1d(c)
            self.pool2 = nn.MaxPool1d(2)
        self.gru = nn.GRU(input_size=c, hidden_size=k2, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(k2, 1)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        if self.two_convs:
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.permute(0,2,1)
        out, h_n = self.gru(x)
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logit = self.fc(h_last)
        return logit

class EarlyStopping:
    def __init__(self, patience=7, delta=0, model_path='best.pth'):
        self.patience = patience; self.delta = delta
        self.best_loss = np.inf; self.counter = 0
        self.best_model_path = model_path; self.early_stop = False
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    def __call__(self, val_loss, model):
        if np.isnan(val_loss):
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
            return
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss; self.counter = 0
            torch.save(model.state_dict(), self.best_model_path)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage_mb():
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024**2)
    gpu_mb = 0.0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return rss_mb, gpu_mb

def save_training_curves(metrics_df, title_suffix, save_path_prefix):
    if metrics_df.empty: return
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train_loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"loss {title_suffix}"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(metrics_df["epoch"], metrics_df["train_acc"], label="train_acc")
    plt.plot(metrics_df["epoch"], metrics_df["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.title(f"acc {title_suffix}"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_curves.png", dpi=150)
    plt.close()

def save_roc_curve(fpr, tpr, title, save_path):
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1],'--', label="random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def bar_plot(labels, values, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(max(10, len(labels)*0.6),5))
    plt.bar(labels, values)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train_with_external_val(train_df, val_df, features, window_size, dropout_rate, k1, k2, base_out_dir, device,
                            batch_size=32, epochs=20, patience=7, lr=1e-3, wd=1e-5, two_convs=True,
                            train_overlap=0.75, val_overlap=0.5, label_threshold=0.6,
                            select_policy="auc_then_loss"):
    tag = f'win{window_size}_do{str(dropout_rate).replace(".","")}_k{k1}_h{k2}_cnn-gru'
    run_dir = os.path.join(base_out_dir, 'pipelines', tag)
    os.makedirs(run_dir, exist_ok=True)

    scaler = StandardScaler()
    tdf = train_df.copy(); vdf = val_df.copy()
    tdf[features] = scaler.fit_transform(tdf[features])
    vdf[features] = scaler.transform(vdf[features])
    with open(os.path.join(run_dir, f'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    train_stride = compute_stride(window_size, 'train', train_overlap=train_overlap, val_overlap=val_overlap)
    val_stride   = compute_stride(window_size, 'val',   train_overlap=train_overlap, val_overlap=val_overlap)

    X_train, y_train = create_sequences(tdf, features, window_size, stride=train_stride, threshold=label_threshold)
    X_val,   y_val   = create_sequences(vdf, features, window_size, stride=val_stride,   threshold=label_threshold)

    if X_train.shape[0]==0 or X_val.shape[0]==0:
        err = f'No sequences created. train={X_train.shape[0]} val={X_val.shape[0]} tag={tag}'
        with open(os.path.join(run_dir, 'error.txt'), 'w', encoding='utf-8') as f: f.write(err)
        raise RuntimeError(err)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    pos_count = max(1, int((y_train==1).sum()))
    neg_count = max(1, int((y_train==0).sum()))
    ratio = min(neg_count / max(1, pos_count), 50.0)
    pos_weight = torch.tensor([ratio], device=device, dtype=torch.float32)

    model = TimeSeriesCNNGRU(len(features), window_size, dropout_rate, k1, k2, two_convs=two_convs).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_loss_path = os.path.join(run_dir, f'best_loss.pth')
    best_acc_path  = os.path.join(run_dir, f'best_acc.pth')
    best_auc_path  = os.path.join(run_dir, f'best_auc.pth')
    last_path      = os.path.join(run_dir, f'last.pth')

    early = EarlyStopping(patience=patience, model_path=best_loss_path)
    train_losses, val_losses, train_accs, val_accs, val_aucs = [], [], [], [], []
    best_acc, best_auc = -1.0, -1.0
    epoch_times = []
    if device.type=='cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        t0 = time.time()
        model.train(); run_loss=0.0; correct=0; total=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            run_loss += loss.item()*x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total += y.size(0); correct += (preds==y).sum().item()
        tr_loss = run_loss/len(train_loader.dataset); tr_acc = correct/total

        model.eval(); v_loss=0.0; v_correct=0; v_total=0; all_probs=[]; all_labels=[]
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                v_loss += loss.item()*x.size(0)
                probs = torch.sigmoid(logits); preds = (probs>=0.5).float()
                v_total += y.size(0); v_correct += (preds==y).sum().item()
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_labels.extend(y.detach().cpu().numpy().tolist())

        va_loss = v_loss/len(val_loader.dataset) if len(val_loader.dataset) else float('inf')
        va_acc = v_correct/v_total if v_total else 0.0
        try:
            va_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
        except Exception:
            va_auc = np.nan

        train_losses.append(tr_loss); val_losses.append(va_loss); train_accs.append(tr_acc); val_accs.append(va_acc); val_aucs.append(va_auc)

        if va_acc>best_acc:
            best_acc=va_acc; torch.save(model.state_dict(), best_acc_path)
        if not np.isnan(va_auc) and va_auc>best_auc:
            best_auc=va_auc; torch.save(model.state_dict(), best_auc_path)

        early(va_loss, model)
        epoch_times.append(time.time()-t0)
        if early.early_stop: break

    torch.save(model.state_dict(), last_path)

    if select_policy == "auc_then_loss":
        chosen = best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else best_loss_path
    elif select_policy == "loss_then_auc":
        chosen = best_loss_path if os.path.exists(best_loss_path) else (best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else last_path)
    elif select_policy == "auc_only":
        chosen = best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else last_path
    elif select_policy == "loss_only":
        chosen = best_loss_path if os.path.exists(best_loss_path) else last_path
    else:
        chosen = best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else best_loss_path

    final_best_path = os.path.join(run_dir, f'best_model.pth')
    if chosen != final_best_path and os.path.exists(chosen):
        shutil.copyfile(chosen, final_best_path)
    elif not os.path.exists(chosen):
        shutil.copyfile(last_path, final_best_path)

    model.load_state_dict(torch.load(final_best_path, map_location=device))

    model.eval(); v_correct=0; v_total=0; v_probs=[]; v_preds=[]; v_labels=[]
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits); preds=(probs>=0.5).float()
            v_total += y.size(0); v_correct += (preds==y).sum().item()
            v_probs.extend(probs.cpu().numpy().tolist()); v_preds.extend(preds.cpu().numpy().tolist()); v_labels.extend(y.cpu().numpy().tolist())

    val_acc_final = v_correct/v_total if v_total else 0.0
    roc_png = None
    try:
        val_auc_final = roc_auc_score(np.array(v_labels), np.array(v_probs))
        fpr,tpr,_ = roc_curve(np.array(v_labels), np.array(v_probs))
        roc_png = os.path.join(run_dir, f"roc.png")
        save_roc_curve(fpr,tpr, title=f"ROC {tag}", save_path=roc_png)
    except Exception:
        val_auc_final = float('nan')

    metrics_df = pd.DataFrame({
        "epoch": list(range(1,len(train_losses)+1)),
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc": train_accs, "val_acc": val_accs, "val_auc": val_aucs
    })
    metrics_csv = os.path.join(run_dir, f"training_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    save_training_curves(metrics_df, title_suffix=tag, save_path_prefix=os.path.join(run_dir, f"train"))

    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float('nan')
    rss_mb, gpu_mb = get_memory_usage_mb()

    rep_txt = os.path.join(run_dir, f"classification_report.txt")
    if len(v_labels) and len(v_preds):
        rep = classification_report(v_labels, v_preds, target_names=['class0','class1'], zero_division=0, output_dict=False)
        with open(rep_txt, 'w', encoding='utf-8') as f: f.write(rep)
    else:
        with open(rep_txt, 'w', encoding='utf-8') as f: f.write("Empty validation predictions.")

    print(f"[DONE] {tag} acc_best={np.nanmax(val_accs) if len(val_accs) else float('nan'):.4f} auc_best={(np.nanmax(val_aucs) if len(val_aucs) else float('nan')):.4f} acc_final={val_acc_final:.4f} auc_final={(0.0 if math.isnan(val_auc_final) else val_auc_final):.4f}")
    print(f"       saved_dir={run_dir}")

    return {
        "윈도우": window_size, "드롭아웃": dropout_rate, "커널1(ks)": k1, "GRU히든": k2,
        "학습 시퀀스": int(len(X_train)), "검증 시퀀스": int(len(X_val)),
        "최고 검증 손실": float(np.nanmin(val_losses)) if len(val_losses) else float('inf'),
        "최고 검증 정확도": float(np.nanmax(val_accs)) if len(val_accs) else 0.0,
        "최고 검증 AUC": float(np.nanmax(val_aucs)) if len(val_aucs) else float('nan'),
        "최종 검증 정확도": float(val_acc_final),
        "최종 검증 AUC": float(val_auc_final) if not math.isnan(val_auc_final) else np.nan,
        "평균 에포크 시간(초)": avg_epoch_time, "피크 GPU 메모리(MB)": gpu_mb, "프로세스 RSS 메모리(MB)": rss_mb,
        "학습 가능 파라미터 수": int(count_parameters(model)),
        "run_dir": run_dir, "best_model_path": final_best_path, "metrics_csv": metrics_csv, "roc_png": roc_png
    }

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_csv = '경로지정'
    out_dir = '경로지정'
    os.makedirs(os.path.join(out_dir, 'pipelines'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'cnn_final_result'), exist_ok=True)

    train_df, val_df, feats_train = split_train_val_from_csv(train_csv, test_ratio=0.3, seed=42)

    window_list = [50]
    dropout_list = [0.1, 0.3, 0.5]
    kernel_sets = [(3,32),(5,32),(3,64)]

    results = []
    for ws in window_list:
        for dr in dropout_list:
            for k1,k2 in kernel_sets:
                print("\n" + "="*90)
                print(f"학습 시작: win={ws}, do={dr}, CNN-GRU(k={k1}, h={k2})")
                print("="*90)
                try:
                    summary = train_with_external_val(
                        train_df=train_df, val_df=val_df, features=feats_train, window_size=ws,
                        dropout_rate=dr, k1=k1, k2=k2, base_out_dir=out_dir, device=device,
                        batch_size=32, epochs=20, patience=7, lr=1e-3, wd=1e-5,
                        two_convs=True, train_overlap=0.75, val_overlap=0.5, label_threshold=0.6,
                        select_policy="auc_then_loss"
                    )
                    results.append(summary)
                except Exception as e:
                    tag = f'win{ws}_do{str(dr).replace(".","")}_k{k1}_h{k2}_cnn-gru'
                    err_dir = os.path.join(out_dir, 'pipelines', tag)
                    os.makedirs(err_dir, exist_ok=True)
                    with open(os.path.join(err_dir, 'error.txt'), 'w', encoding='utf-8') as f:
                        f.write(str(e))
                    print(f"[ERROR] {tag} -> {e}")

    summary_df = pd.DataFrame(results, columns=[
        "윈도우","드롭아웃","커널1(ks)","GRU히든",
        "학습 시퀀스","검증 시퀀스",
        "최고 검증 손실","최고 검증 정확도","최고 검증 AUC",
        "최종 검증 정확도","최종 검증 AUC",
        "평균 에포크 시간(초)","피크 GPU 메모리(MB)","프로세스 RSS 메모리(MB)","학습 가능 파라미터 수",
        "run_dir","best_model_path","metrics_csv","roc_png"
    ])

    final_dir = os.path.join(out_dir, "cnn_final_result")
    summary_csv = os.path.join(final_dir, "summary_all.csv")
    summary_df.to_csv(summary_csv, index=False)

    def rank_key(row):
        auc = row.get("최고 검증 AUC", np.nan)
        acc = row.get("최고 검증 정확도", 0.0)
        loss = row.get("최고 검증 손실", np.inf)
        auc_sort = -auc if not np.isnan(auc) else np.inf
        return (auc_sort, -acc, loss)

    ranked = summary_df.copy()
    ranked["_rank_key"] = ranked.apply(rank_key, axis=1)
    ranked = ranked.sort_values(by="_rank_key").drop(columns=["_rank_key"])
    ranking_csv = os.path.join(final_dir, "ranking.csv")
    ranked.to_csv(ranking_csv, index=False)

    lbls = [f"w{int(r['윈도우'])}/do={r['드롭아웃']}/k={int(r['커널1(ks)'])}/h={int(r['GRU히든'])}" for _, r in ranked.iterrows()]
    vals_acc_best  = ranked["최고 검증 정확도"].fillna(0).tolist()
    vals_auc_best  = ranked["최고 검증 AUC"].fillna(0).tolist()
    vals_acc_final = ranked["최종 검증 정확도"].fillna(0).tolist()
    vals_auc_final = ranked["최종 검증 AUC"].fillna(0).tolist()

    bar_plot(lbls, vals_acc_best,  "config", "acc", "최고 검증 정확도 (CNN-GRU)", os.path.join(final_dir, "best_val_acc.png"))
    bar_plot(lbls, vals_auc_best,  "config", "auc", "최고 검증 AUC (CNN-GRU)", os.path.join(final_dir, "best_val_auc.png"))
    bar_plot(lbls, vals_acc_final, "config", "acc", "최종 검증 정확도 (CNN-GRU)", os.path.join(final_dir, "final_val_acc.png"))
    bar_plot(lbls, vals_auc_final, "config", "auc", "최종 검증 AUC (CNN-GRU)", os.path.join(final_dir, "final_val_auc.png"))

    print(f"\n요약: {summary_csv}")
    print(f"순위표: {ranking_csv}")
    print(f"파이프라인 루트: {os.path.join(out_dir, 'pipelines')}")

if __name__ == "__main__":
    main()
