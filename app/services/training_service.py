# app/services/training_service.py

import os
import json
import pickle
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from torch.utils.data import TensorDataset, DataLoader

from services.utils import training_helpers

from config import TrainingConfig
from models.realtimemodel_gru import TimeSeriesCNNGRU
from services.supabase_service import SupabaseService
from services.utils import data_utils


class TrainingService:
    """
    모델 학습, 평가, 개인화 및 배포 파이프라인을 총괄하는 서비스 클래스.
    """

    def __init__(self, config: TrainingConfig, supabase_service: SupabaseService = None):
        self.config = config
        self.supabase_service = supabase_service
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_utils.set_seed(self.config.seed)

    def _freeze_batchnorm(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
            for p in m.parameters(): p.requires_grad = False

    def _set_trainable(self, model, mode):
        for p in model.parameters(): p.requires_grad = False
        if mode == 'head':
            for p in model.fc.parameters():  p.requires_grad = True
        elif mode == 'gru':
            for p in model.gru.parameters(): p.requires_grad = True
            for p in model.fc.parameters():  p.requires_grad = True
        elif mode == 'cnn_gru':
            for p in model.conv1.parameters(): p.requires_grad = True
            for p in model.gru.parameters():   p.requires_grad = True
            for p in model.fc.parameters():    p.requires_grad = True
        elif mode == 'full':
            for p in model.parameters(): p.requires_grad = True

    def _sweep_threshold_f1(self, probs, labels, grid=None):
        if grid is None: grid = np.linspace(0.05, 0.95, 91)
        labels = np.asarray(labels).astype(int).ravel()
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t), float(best_f1)

    def _best_temperature_from_logits(self, logits, labels, temps=None):
        if temps is None: temps = np.linspace(0.5, 3.0, 51)

        def nll(temp):
            z = logits / float(max(temp, 1e-6))
            p = 1 / (1 + np.exp(-z))
            eps = 1e-7
            p = np.clip(p, eps, 1 - eps)
            return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

        nlls = [nll(t) for t in temps]
        return float(temps[int(np.argmin(nlls))])

    def _fit_scaler_on_df(self, df, feats):
        scaler = StandardScaler()
        df[feats] = scaler.fit_transform(df[feats])
        return scaler

    def _transform_df(self, df, feats, scaler):
        df[feats] = scaler.transform(df[feats])
        return df

    def _normalize_state_dict(self, sd):
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        sd2 = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith('module.') else k
            sd2[nk] = v
        return sd2

    def _make_model_and_load(self, sd_path, input_dim, window_size):
        model = TimeSeriesCNNGRU(input_dim, window_size, self.config.dropout_rate, self.config.kernel_size,
                                 self.config.hidden_size, self.config.two_convs).to(self.device)
        try:
            sd = torch.load(sd_path, map_location=self.device)
            sd = self._normalize_state_dict(sd)
            m = model.state_dict()
            matched = {k: v for k, v in sd.items() if k in m and hasattr(v, 'shape') and v.shape == m[k].shape}
            if matched:
                m.update(matched);
                model.load_state_dict(m, strict=False)
        except Exception as e:
            print(f"[WARN] checkpoint load failed: {e}. Using random init.")
        return model

    def _safe_auc(self, y_true, probs):
        y_true = np.asarray(y_true).astype(int).ravel()
        probs = np.asarray(probs).ravel()
        roc_auc = float('nan');
        pr_auc = float('nan')
        try:
            roc_auc = float(roc_auc_score(y_true, probs))
        except Exception:
            pass
        try:
            pr_auc = float(average_precision_score(y_true, probs))
        except Exception:
            pass
        return roc_auc, pr_auc

    def run_personalization_pipeline(self, user_id: str, focus_data: list, non_focus_data: list):
        """특정 사용자를 위한 전체 개인화 파이프라인을 실행합니다."""
        print(f"--- {user_id} 사용자를 위한 개인화 파이프라인 시작 ---")

        df_focus = data_utils.json_to_df(focus_data, label=0, username=user_id)
        df_non_focus = data_utils.json_to_df(non_focus_data, label=1, username=user_id)
        df_all = pd.concat([df_focus, df_non_focus], ignore_index=True)
        df_all, features = data_utils.feature_engineer(df_all, base_features=self.config.base_features,
                                                       username=user_id)
        df_all = df_all[df_all['eye_status'] != 'NO_FACE_DETECTED'].sort_values('timestamp_ms').reset_index(drop=True)

        df_train, df_test = data_utils.stratified_time_split(df_all, test_ratio=self.config.test_ratio)

        profiles = {}
        for mode in self.config.personalization_modes:
            print(f"\n--- 프로파일 실행: {mode} ---")
            if mode == 'cal':
                result = self._run_cal_profile(user_id, df_train, features)
            elif mode == 'head':
                result = self._run_head_profile(user_id, df_train, features, fine_tune_mode='head')
            elif mode == 'cal_head':
                result = self._run_cal_head_profile(user_id, df_train, features)
            elif mode.startswith('transfer'):
                tune_mode = mode.split('_', 1)[1]
                result = self._run_transfer_profile(user_id, df_train, features, fine_tune_mode=tune_mode)
            else:
                print(f"알 수 없는 프로파일 모드: {mode}")
                continue

            if result:
                profiles[mode] = result
                print(f"프로파일 {mode} 실행 완료.")
            else:
                print(f"프로파일 {mode} 실행 실패 또는 조건 미달.")

        if not profiles:
            print("실행 가능한 프로파일이 없어 파이프라인을 종료합니다.")
            return None

        best_profile_summary = self._compare_and_activate(user_id, df_test, profiles)

        print(f"--- {user_id} 사용자를 위한 개인화 파이프라인 종료 ---")
        return best_profile_summary

    def _train_with_external_val(self, train_df, val_df, features, window_size, dropout_rate, k1, k2, base_out_dir,
                                 device,
                                 batch_size=32, epochs=20, patience=7, lr=1e-3, wd=1e-5, two_convs=True,
                                 train_overlap=0.75, val_overlap=0.5, label_threshold=0.6,
                                 select_policy="auc_then_loss"):
        tag = f'win{window_size}_do{str(dropout_rate).replace(".", "")}_k{k1}_h{k2}_cnn-gru'
        run_dir = os.path.join(base_out_dir, 'pipelines', tag)
        os.makedirs(run_dir, exist_ok=True)

        scaler = StandardScaler()
        tdf = train_df.copy();
        vdf = val_df.copy()
        tdf[features] = scaler.fit_transform(tdf[features])
        vdf[features] = scaler.transform(vdf[features])
        with open(os.path.join(run_dir, f'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        train_stride = data_utils.compute_stride(window_size, 'train', train_overlap=train_overlap,
                                                 val_overlap=val_overlap)
        val_stride = data_utils.compute_stride(window_size, 'val', train_overlap=train_overlap, val_overlap=val_overlap)

        X_train, y_train = data_utils.create_sequences(tdf, features, window_size, stride=train_stride,
                                                       threshold=label_threshold)
        X_val, y_val = data_utils.create_sequences(vdf, features, window_size, stride=val_stride,
                                                   threshold=label_threshold)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            err = f'No sequences created. train={X_train.shape[0]} val={X_val.shape[0]} tag={tag}'
            with open(os.path.join(run_dir, 'error.txt'), 'w', encoding='utf-8') as f: f.write(err)
            raise RuntimeError(err)

        train_loader = DataLoader(training_helpers.TimeSeriesDataset(X_train, y_train), batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(training_helpers.TimeSeriesDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        pos_count = max(1, int((y_train == 1).sum()))
        neg_count = max(1, int((y_train == 0).sum()))
        ratio = min(neg_count / max(1, pos_count), 50.0)
        pos_weight = torch.tensor([ratio], device=device, dtype=torch.float32)

        model = TimeSeriesCNNGRU(len(features), window_size, dropout_rate, k1, k2, two_convs=two_convs).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_loss_path = os.path.join(run_dir, f'best_loss.pth')
        best_acc_path = os.path.join(run_dir, f'best_acc.pth')
        best_auc_path = os.path.join(run_dir, f'best_auc.pth')
        last_path = os.path.join(run_dir, f'last.pth')

        early = training_helpers.EarlyStopping(patience=patience, model_path=best_loss_path)
        train_losses, val_losses, train_accs, val_accs, val_aucs = [], [], [], [], []
        best_acc, best_auc = -1.0, -1.0
        epoch_times = []
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            t0 = time.time()
            model.train();
            run_loss = 0.0;
            correct = 0;
            total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward();
                optimizer.step()
                run_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total += y.size(0);
                correct += (preds == y).sum().item()
            tr_loss = run_loss / len(train_loader.dataset);
            tr_acc = correct / total

            model.eval();
            v_loss = 0.0;
            v_correct = 0
            v_total = 0;
            all_probs = [];
            all_labels = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x).squeeze(-1)
                    loss = criterion(logits, y)
                    v_loss += loss.item() * x.size(0)
                    probs = torch.sigmoid(logits);
                    preds = (probs >= 0.5).float()
                    v_total += y.size(0);
                    v_correct += (preds == y).sum().item()
                    all_probs.extend(probs.detach().cpu().numpy().tolist());
                    all_labels.extend(y.detach().cpu().numpy().tolist())

            va_loss = v_loss / len(val_loader.dataset) if len(val_loader.dataset) else float('inf')
            va_acc = v_correct / v_total if v_total else 0.0
            try:
                va_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
            except Exception:
                va_auc = np.nan

            train_losses.append(tr_loss);
            val_losses.append(va_loss);
            train_accs.append(tr_acc);
            val_accs.append(va_acc);
            val_aucs.append(va_auc)

            if va_acc > best_acc:
                best_acc = va_acc;
                torch.save(model.state_dict(), best_acc_path)
            if not np.isnan(va_auc) and va_auc > best_auc:
                best_auc = va_auc;
                torch.save(model.state_dict(), best_auc_path)

            early(va_loss, model)
            epoch_times.append(time.time() - t0)
            if early.early_stop: break

        torch.save(model.state_dict(), last_path)

        if select_policy == "auc_then_loss":
            chosen = best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else best_loss_path
        elif select_policy == "loss_then_auc":
            chosen = best_loss_path if os.path.exists(best_loss_path) else (
                best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else last_path)
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

        model.eval();
        v_correct = 0;
        v_total = 0;
        v_probs = [];
        v_preds = [];
        v_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                probs = torch.sigmoid(logits);
                preds = (probs >= 0.5).float()
                v_total += y.size(0);
                v_correct += (preds == y).sum().item()
                v_probs.extend(probs.cpu().numpy().tolist());
                v_preds.extend(preds.cpu().numpy().tolist());
                v_labels.extend(y.cpu().numpy().tolist())

        val_acc_final = v_correct / v_total if v_total else 0.0
        roc_png = None
        try:
            val_auc_final = roc_auc_score(np.array(v_labels), np.array(v_probs))
            fpr, tpr, _ = roc_curve(np.array(v_labels), np.array(v_probs))
            roc_png = os.path.join(run_dir, f"roc.png")
            training_helpers.save_roc_curve(fpr, tpr, title=f"ROC {tag}", save_path=roc_png)
        except Exception:
            val_auc_final = np.nan

        metrics_df = pd.DataFrame({
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses, "val_loss": val_losses,
            "train_acc": train_accs, "val_acc": val_accs, "val_auc": val_aucs
        })
        metrics_csv = os.path.join(run_dir, f"training_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        training_helpers.save_training_curves(metrics_df, title_suffix=tag,
                                              save_path_prefix=os.path.join(run_dir, f"train"))

        avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float('nan')
        rss_mb, gpu_mb = training_helpers.get_memory_usage_mb()

        rep_txt = os.path.join(run_dir, f"classification_report.txt")
        if len(v_labels) and len(v_preds):
            rep = classification_report(v_labels, v_preds, target_names=['class0', 'class1'], zero_division=0,
                                        output_dict=False)
            with open(rep_txt, 'w', encoding='utf-8') as f:
                f.write(rep)
        else:
            with open(rep_txt, 'w', encoding='utf-8') as f:
                f.write("Empty validation predictions.")

        print(
            f"[DONE] {tag} acc_best={np.nanmax(val_accs) if len(val_accs) else float('nan'):.4f} auc_best={(np.nanmax(val_aucs) if len(val_aucs) else float('nan')):.4f} acc_final={val_acc_final:.4f} auc_final={(0.0 if math.isnan(val_auc_final) else val_auc_final):.4f}")
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
            "학습 가능 파라미터 수": int(training_helpers.count_parameters(model)),
            "run_dir": run_dir, "best_model_path": final_best_path, "metrics_csv": metrics_csv, "roc_png": roc_png
        }

    def run_hyperparameter_sweep(self, train_csv_path: str, val_csv_path: str = None):
        """주어진 학습 데이터로 하이퍼파라미터 탐색을 수행합니다."""
        print("--- 하이퍼파라미터 탐색 시작 ---")
        device = self.device

        # Load and split data
        train_df, val_df, feats_train = data_utils.split_train_val_from_csv(train_csv_path,
                                                                            test_ratio=self.config.validation_ratio,
                                                                            seed=self.config.seed)

        window_list = self.config.window_sizes
        dropout_list = self.config.dropout_rates
        kernel_sets = self.config.kernel_sets

        results = []
        for ws in window_list:
            for dr in dropout_list:
                for k1, k2 in kernel_sets:
                    print("\n" + "=" * 90)
                    print(f"학습 시작: win={ws}, do={dr}, CNN-GRU(k={k1}, h={k2})")
                    print("=" * 90)
                    try:
                        summary = self._train_with_external_val(
                            train_df=train_df, val_df=val_df, features=feats_train, window_size=ws,
                            dropout_rate=dr, k1=k1, k2=k2, base_out_dir=self.config.output_dir, device=device,
                            batch_size=self.config.batch_size, epochs=self.config.epochs,
                            patience=self.config.early_stop_patience,
                            lr=self.config.learning_rate, wd=self.config.weight_decay,
                            two_convs=self.config.two_convs, train_overlap=self.config.train_overlap,
                            val_overlap=self.config.val_overlap, label_threshold=self.config.label_threshold,
                            select_policy=self.config.select_policy
                        )
                        results.append(summary)
                    except Exception as e:
                        tag = f'win{ws}_do{str(dr).replace(".", "")}_k{k1}_h{k2}_cnn-gru'
                        err_dir = os.path.join(self.config.output_dir, 'pipelines', tag)
                        os.makedirs(err_dir, exist_ok=True)
                        with open(os.path.join(err_dir, 'error.txt'), 'w', encoding='utf-8') as f:
                            f.write(str(e))
                        print(f"[ERROR] {tag} -> {e}")

        summary_df = pd.DataFrame(results, columns=[
            "윈도우", "드롭아웃", "커널1(ks)", "GRU히든",
            "학습 시퀀스", "검증 시퀀스",
            "최고 검증 손실", "최고 검증 정확도", "최고 검증 AUC",
            "최종 검증 정확도", "최종 검증 AUC",
            "평균 에포크 시간(초)", "피크 GPU 메모리(MB)", "프로세스 RSS 메모리(MB)", "학습 가능 파라미터 수",
            "run_dir", "best_model_path", "metrics_csv", "roc_png"
        ])

        final_dir = os.path.join(self.config.output_dir, "cnn_final_result")
        os.makedirs(final_dir, exist_ok=True)
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

        lbls = [f"w{int(r['윈도우'])}/do={r['드롭아웃']}/k={int(r['커널1(ks)'])}/h={int(r['GRU히든'])}" for _, r in
                ranked.iterrows()]
        vals_acc_best = ranked["최고 검증 정확도"].fillna(0).tolist()
        vals_auc_best = ranked["최고 검증 AUC"].fillna(0).tolist()
        vals_acc_final = ranked["최종 검증 정확도"].fillna(0).tolist()
        vals_auc_final = ranked["최종 검증 AUC"].fillna(0).tolist()

        training_helpers.bar_plot(lbls, vals_acc_best, "config", "acc", "최고 검증 정확도 (CNN-GRU)",
                                  os.path.join(final_dir, "best_val_acc.png"))
        training_helpers.bar_plot(lbls, vals_auc_best, "config", "auc", "최고 검증 AUC (CNN-GRU)",
                                  os.path.join(final_dir, "best_val_auc.png"))
        training_helpers.bar_plot(lbls, vals_acc_final, "config", "acc", "최종 검증 정확도 (CNN-GRU)",
                                  os.path.join(final_dir, "final_val_acc.png"))
        training_helpers.bar_plot(lbls, vals_auc_final, "config", "auc", "최종 검증 AUC (CNN-GRU)",
                                  os.path.join(final_dir, "final_val_auc.png"))

        print(f"\n요약: {summary_csv}")
        print(f"순위표: {ranking_csv}")
        print(f"파이프라인 루트: {os.path.join(self.config.output_dir, 'pipelines')}")

    def _run_cal_profile(self, username, df_train, feats):
        stride = data_utils.compute_stride(self.config.window_size, self.config.overlap)
        with open(self.config.base_scaler_path, 'rb') as f: scaler = pickle.load(f)
        df_tr = self._transform_df(df_train.copy(), feats, scaler)
        _, df_val = data_utils.stratified_time_split(df_tr, test_ratio=self.config.validation_ratio)

        X, y = data_utils.create_sequences(df_val, feats, self.config.window_size, stride,
                                           threshold=self.config.label_threshold)
        if len(X) == 0: return None

        device = self.device
        model = self._make_model_and_load(self.config.base_model_path, len(feats), self.config.window_size);
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()

        probs = 1 / (1 + np.exp(-logits))
        best_t, _ = self._sweep_threshold_f1(probs, y)
        best_temp = self._best_temperature_from_logits(logits, y)

        header = {"username": username, "window_size": self.config.window_size, "overlap": self.config.overlap,
                  "features": feats,
                  "threshold": best_t, "temperature": best_temp, "mode": "cal"}
        json.dump(header, open(os.path.join(self.config.checkpoint_dir, f"{username}_header_cal.json"), "w"),
                  ensure_ascii=False)
        return {"header": os.path.join(self.config.checkpoint_dir, f"{username}_header_cal.json"),
                "threshold": best_t, "temperature": best_temp}

    def _run_head_profile(self, username, df_train, feats, fine_tune_mode):
        out = self._train_model(username, df_train, feats, self.config.base_model_path, train_mode='head',
                                use_base_scaler=True)
        if out is None: return None

        if self.config.min_train_pr_auc is not None:
            if np.isnan(out.get("train_pr_auc", float('nan'))) or out["train_pr_auc"] < float(
                    self.config.min_train_pr_auc):
                print(f"[INFO] head 제외: train PR AUC={out.get('train_pr_auc')} < {self.config.min_train_pr_auc}")
                return None

        device = self.device
        with open(out["scaler"], 'rb') as f:
            scaler = pickle.load(f)
        model = self._make_model_and_load(out["model"], len(feats), self.config.window_size);
        model.eval()

        stride = data_utils.compute_stride(self.config.window_size, self.config.overlap)
        df_trsc = self._transform_df(df_train.copy(), feats, scaler)
        _, df_val = data_utils.stratified_time_split(df_trsc, test_ratio=self.config.validation_ratio)
        X, y = data_utils.create_sequences(df_val, feats, self.config.window_size, stride,
                                           threshold=self.config.label_threshold)

        with torch.no_grad():
            lg = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
            probs = 1 / (1 + np.exp(-lg))
        best_t, _ = self._sweep_threshold_f1(probs, y)

        header = {"username": username, "window_size": self.config.window_size, "overlap": self.config.overlap,
                  "features": feats,
                  "threshold": best_t, "temperature": 1.0, "mode": "head"}
        json.dump(header, open(os.path.join(self.config.checkpoint_dir, f"{username}_header_head.json"), "w"),
                  ensure_ascii=False)
        return {"header": os.path.join(self.config.checkpoint_dir, f"{username}_header_head.json"), **out,
                "threshold": best_t}

    def _run_cal_head_profile(self, username, df_train, feats):
        out = self._run_head_profile(username, df_train, feats,
                                     fine_tune_mode=None)  # fine_tune_mode is not used in run_cal_head
        if out is None: return None

        device = self.device
        with open(out["scaler"], 'rb') as f: scaler = pickle.load(f)
        model = self._make_model_and_load(out["model"], len(feats), self.config.window_size);
        model.eval()

        stride = data_utils.compute_stride(self.config.window_size, self.config.overlap)
        df_trsc = self._transform_df(df_train.copy(), feats, scaler)
        _, df_val = data_utils.stratified_time_split(df_trsc, test_ratio=self.config.validation_ratio)
        X, y = data_utils.create_sequences(df_val, feats, self.config.window_size, stride,
                                           threshold=self.config.label_threshold)

        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
        best_temp = self._best_temperature_from_logits(logits, y)
        probs = 1 / (1 + np.exp(-(logits / max(best_temp, 1e-6))))
        best_t, _ = self._sweep_threshold_f1(probs, y)

        header = {"username": username, "window_size": self.config.window_size, "overlap": self.config.overlap,
                  "features": feats,
                  "threshold": best_t, "temperature": best_temp, "mode": "cal_head"}
        json.dump(header, open(os.path.join(self.config.checkpoint_dir, f"{username}_header_cal_head.json"), "w"),
                  ensure_ascii=False)
        return {"header": os.path.join(self.config.checkpoint_dir, f"{username}_header_cal_head.json"),
                "model": out["model"], "scaler": out["scaler"], "threshold": best_t, "temperature": best_temp}

    def _run_transfer_profile(self, username, df_train, feats, fine_tune_mode):
        out = self._train_model(username, df_train, feats, self.config.base_model_path, train_mode=fine_tune_mode,
                                use_base_scaler=False)
        if out is None: return None
        if self.config.min_train_pr_auc is not None:
            if np.isnan(out.get("train_pr_auc", float('nan'))) or out["train_pr_auc"] < float(
                    self.config.min_train_pr_auc):
                print(
                    f"[INFO] transfer({fine_tune_mode}) 제외: train PR AUC={out.get('train_pr_auc')} < {self.config.min_train_pr_auc}")
                return None

        device = self.device
        with open(out["scaler"], 'rb') as f:
            scaler = pickle.load(f)
        model = self._make_model_and_load(out["model"], len(feats), self.config.window_size);
        model.eval()

        stride = data_utils.compute_stride(self.config.window_size, self.config.overlap)
        df_trsc = self._transform_df(df_train.copy(), feats, scaler)
        _, df_val = data_utils.stratified_time_split(df_trsc, test_ratio=self.config.validation_ratio)
        X, y = data_utils.create_sequences(df_val, feats, self.config.window_size, stride,
                                           threshold=self.config.label_threshold)

        with torch.no_grad():
            lg = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
            probs = 1 / (1 + np.exp(-lg))
        best_t, _ = self._sweep_threshold_f1(probs, y)

        header = {"username": username, "window_size": self.config.window_size, "overlap": self.config.overlap,
                  "features": feats,
                  "threshold": best_t, "temperature": 1.0, "mode": f"transfer_{fine_tune_mode}"}
        json.dump(header,
                  open(os.path.join(self.config.checkpoint_dir, f"{username}_header_transfer_{fine_tune_mode}.json")),
                  ensure_ascii=False)
        return {"header": os.path.join(self.config.checkpoint_dir, f"{username}_header_transfer_{fine_tune_mode}.json"),
                **out, "threshold": best_t}

    def _train_model(self, username, df_train, feats, base_model_path, train_mode, use_base_scaler):
        stride = data_utils.compute_stride(self.config.window_size, self.config.overlap)
        device = self.device

        if use_base_scaler:
            with open(self.config.base_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df_tr = self._transform_df(df_train.copy(), feats, scaler)
        else:
            scaler = self._fit_scaler_on_df(df_train.copy(), feats)
            df_tr = self._transform_df(df_train.copy(), feats, scaler)

        X, y = data_utils.create_sequences(df_tr, feats, self.config.window_size, stride,
                                           threshold=self.config.label_threshold)
        if len(X) == 0: return None

        n = len(X);
        split = int(n * 0.8) if n > 5 else n
        X_tr, y_tr = X[:split], y[:split]
        X_val, y_val = (X[split:], y[split:]) if split < n else (X.copy(), y.copy())

        model = self._make_model_and_load(base_model_path, len(feats), self.config.window_size)
        model.apply(self._freeze_batchnorm)
        self._set_trainable(model, train_mode)

        pos = max(1, int((y_tr == 1).sum()));
        neg = max(1, int((y_tr == 0).sum()))
        pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=self.config.learning_rate,
                               weight_decay=self.config.weight_decay)

        train_dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=self.config.batch_size, shuffle=True)
        val_dl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=self.config.batch_size, shuffle=False)

        best_sd = None;
        best_val = np.inf;
        noimp = 0
        for ep in range(1, self.config.epochs + 1):
            # train
            model.train()
            tr_loss_sum = 0.0;
            tr_n = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                lg = model(xb).squeeze(-1)
                loss = criterion(lg, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss_sum += loss.item() * xb.size(0);
                tr_n += xb.size(0)

            # val
            model.eval();
            vloss = 0.0;
            v_n = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    lg = model(xb).squeeze(-1)
                    ls = criterion(lg, yb)
                    vloss += ls.item() * xb.size(0);
                    v_n += xb.size(0)
            tr_loss = tr_loss_sum / max(1, tr_n)
            vloss = vloss / max(1, v_n)
            print(
                f"[{train_mode}] epoch {ep:02d}/{self.config.epochs} | train_loss={tr_loss:.4f} val_loss={vloss:.4f} noimp={noimp}/{self.config.early_stop_patience}")

            if vloss < best_val:
                best_val = vloss;
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()};
                noimp = 0
            else:
                noimp += 1
                if noimp >= self.config.early_stop_patience:
                    print(f"[{train_mode}] Early stop at epoch {ep}")
                    break

        if best_sd is not None:
            model.load_state_dict(best_sd, strict=True)

        # 저장
        model_path = os.path.join(self.config.checkpoint_dir, f"{username}_model_{train_mode}.pth")
        scaler_path = os.path.join(self.config.checkpoint_dir, f"{username}_scaler_{train_mode}.pkl")
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # 참고용 성능
        with torch.no_grad():
            lg_val = model(torch.tensor(X_val, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
            probs_va = 1 / (1 + np.exp(-lg_val))
        acc_val = float(((probs_va >= 0.5).astype(int) == y_val.astype(int)).mean())

        with torch.no_grad():
            lg_tr = model(torch.tensor(X_tr, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()
            probs_tr = 1 / (1 + np.exp(-lg_tr))
        roc_auc_tr, pr_auc_tr = self._safe_auc(y_tr, probs_tr)

        return {
            "model": model_path,
            "scaler": scaler_path,
            "val_acc@0.5": acc_val,
            "features": feats,
            "train_roc_auc": roc_auc_tr,
            "train_pr_auc": pr_auc_tr
        }

    def _evaluate_profile(self, profile_name, profile_artifacts, df_test):
        def _evaluate_df_internal(model_path, scaler_path, header, df_test, return_details=False, profile_name=None):
            username = header["username"]
            window_size = int(header.get("window_size", self.config.window_size))
            overlap = float(header.get("overlap", self.config.overlap))
            feats = header.get("features")
            if feats is None:
                tmp, _ = data_utils.feature_engineer(
                    pd.DataFrame([{"ear": 0, "pitch": 0, "yaw": 0, "roll": 0, "prefix": username}]), username=username)
                _, feats = data_utils.feature_engineer(tmp, username=username)

            df = df_test.copy()
            df, _ = data_utils.feature_engineer(df, username=username)
            df = df.sort_values("timestamp_ms").reset_index(drop=True)

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df[feats] = scaler.transform(df[feats])

            stride = data_utils.compute_stride(window_size, overlap)
            X, y, meta = data_utils.create_sequences(df, feats, window_size, stride,
                                                     threshold=self.config.label_threshold, return_meta=True)
            if len(X) == 0:
                return (None, pd.DataFrame()) if return_details else None

            device = self.device
            model = self._make_model_and_load(model_path, len(feats), window_size);
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()

            temp = float(header.get("temperature", 1.0))
            probs = 1 / (1 + np.exp(-(logits / max(temp, 1e-6))))
            thr = float(header.get("threshold", 0.5))
            preds = (probs >= thr).astype(int)
            y = y.astype(int)

            roc_auc, pr_auc = self._safe_auc(y, probs)
            metrics = {
                "acc": float(accuracy_score(y, preds)),
                "precision": float(precision_score(y, preds, zero_division=0)),
                "recall": float(recall_score(y, preds, zero_division=0)),
                "f1": float(f1_score(y, preds, zero_division=0)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "n": int(len(y))
            }

            if return_details:
                md = pd.DataFrame(meta)
                details = pd.DataFrame({"y_true": y, "y_pred": preds, "prob": probs})
                details = pd.concat([md.reset_index(drop=True), details.reset_index(drop=True)], axis=1)
                details["profile"] = profile_name if profile_name else header.get("mode", "unknown")
                return metrics, details

            return metrics

        # Logic from evaluate_profile_df
        if profile_name == 'base':
            tmp, _ = data_utils.feature_engineer(
                pd.DataFrame([{"ear": 0, "pitch": 0, "yaw": 0, "roll": 0, "prefix": self.config.username}]),
                username=self.config.username)
            _, feats = data_utils.feature_engineer(tmp, username=self.config.username)
            header = {"username": self.config.username, "window_size": self.config.window_size,
                      "overlap": self.config.overlap, "features": feats,
                      "threshold": 0.5, "temperature": 1.0, "mode": "base"}
            return _evaluate_df_internal(self.config.base_model_path, self.config.base_scaler_path, header, df_test,
                                         return_details=True, profile_name='base')

        elif profile_name == 'cal':
            header = json.load(
                open(os.path.join(self.config.checkpoint_dir, f"{self.config.username}_header_cal.json")))
            return _evaluate_df_internal(self.config.base_model_path, self.config.base_scaler_path, header, df_test,
                                         return_details=True, profile_name='cal')

        elif profile_name == 'head':
            header = json.load(
                open(os.path.join(self.config.checkpoint_dir, f"{self.config.username}_header_head.json")))
            model_path = os.path.join(self.config.checkpoint_dir, f"{self.config.username}_model_head.pth")
            scaler_path = os.path.join(self.config.checkpoint_dir, f"{self.config.username}_scaler_head.pkl")
            return _evaluate_df_internal(model_path, scaler_path, header, df_test, return_details=True,
                                         profile_name='head')

        elif profile_name == 'cal_head':
            header = json.load(
                open(os.path.join(self.config.checkpoint_dir, f"{self.config.username}_header_cal_head.json")))
            model_path = os.path.join(self.config.checkpoint_dir, f"{self.config.username}_model_head.pth")
            scaler_path = os.path.join(self.config.checkpoint_dir, f"{self.config.username}_scaler_head.pkl")
            return _evaluate_df_internal(model_path, scaler_path, header, df_test, return_details=True,
                                         profile_name='cal_head')

        elif profile_name.startswith('transfer'):
            mode = profile_name.split('_', 1)[1]
            header = json.load(
                open(os.path.join(self.config.checkpoint_dir, f"{self.config.username}_header_transfer_{mode}.json")))
            model_path = os.path.join(self.config.checkpoint_dir, f"{self.config.username}_model_{mode}.pth")
            scaler_path = os.path.join(self.config.checkpoint_dir, f"{self.config.username}_scaler_{mode}.pkl")
            return _evaluate_df_internal(model_path, scaler_path, header, df_test, return_details=True,
                                         profile_name=f"transfer_{mode}")

        else:
            raise ValueError("unknown profile_name")

    def _compare_and_activate(self, username, df_test, profiles):
        results = [];
        details_list = []
        candidates = ['base']
        if os.path.isfile(
            os.path.join(self.config.checkpoint_dir, f"{username}_header_cal.json")):      candidates.append('cal')
        if os.path.isfile(
            os.path.join(self.config.checkpoint_dir, f"{username}_header_head.json")):     candidates.append('head')
        if os.path.isfile(
            os.path.join(self.config.checkpoint_dir, f"{username}_header_cal_head.json")): candidates.append('cal_head')
        for m in ['gru', 'cnn_gru', 'full']:
            if os.path.isfile(os.path.join(self.config.checkpoint_dir, f"{username}_header_transfer_{m}.json")):
                candidates.append(f"transfer_{m}")

        for name in candidates:
            out = self._evaluate_profile(name, None, df_test)  # profile_artifacts is not used here
            if out is None:
                continue
            metrics, details = out
            if metrics: results.append({"profile": name, **metrics})
            if details is not None and len(details) > 0:
                details_list.append(details)

        if not results: return None

        if len(details_list) > 0:
            all_details = pd.concat(details_list, ignore_index=True)

        df = pd.DataFrame(results)
        best_row = df.iloc[df[self.config.metric].values.argmax()]
        best = best_row["profile"]

        # 활성화 산출물
        if best == 'base':
            tmp, _ = data_utils.feature_engineer(
                pd.DataFrame([{"ear": 0, "pitch": 0, "yaw": 0, "roll": 0, "prefix": username}]), username=username)
            _, feats = data_utils.feature_engineer(tmp, username=username)
            header = {"username": username, "window_size": self.config.window_size, "overlap": self.config.overlap,
                      "features": feats,
                      "threshold": 0.5, "temperature": 1.0, "mode": "base"}
            json.dump(header, open(os.path.join(self.config.checkpoint_dir, "active_header.json"), "w"),
                      ensure_ascii=False)
            shutil.copy2(self.config.base_model_path, os.path.join(self.config.checkpoint_dir, "active_model.pth"))
            shutil.copy2(self.config.base_scaler_path, os.path.join(self.config.checkpoint_dir, "active_scaler.pkl"))

        elif best == 'cal':
            hdr = json.load(open(os.path.join(self.config.checkpoint_dir, f"{username}_header_cal.json")))
            json.dump(hdr, open(os.path.join(self.config.checkpoint_dir, "active_header.json"), "w"),
                      ensure_ascii=False)
            shutil.copy2(self.config.base_model_path, os.path.join(self.config.checkpoint_dir, "active_model.pth"))
            shutil.copy2(self.config.base_scaler_path, os.path.join(self.config.checkpoint_dir, "active_scaler.pkl"))

        elif best in ['head', 'cal_head']:
            header_file = f"{username}_header_cal_head.json" if best == 'cal_head' else f"{username}_header_head.json"
            hdr = json.load(open(os.path.join(self.config.checkpoint_dir, header_file)))
            json.dump(hdr, open(os.path.join(self.config.checkpoint_dir, "active_header.json"), "w"),
                      ensure_ascii=False)
            shutil.copy2(os.path.join(self.config.checkpoint_dir, f"{username}_model_head.pth"),
                         os.path.join(self.config.checkpoint_dir, "active_model.pth"))
            shutil.copy2(os.path.join(self.config.checkpoint_dir, f"{username}_scaler_head.pkl"),
                         os.path.join(self.config.checkpoint_dir, "active_scaler.pkl"))

        else:  # transfer_*
            mode = best.split('_', 1)[1]
            hdr = json.load(open(os.path.join(self.config.checkpoint_dir, f"{username}_header_transfer_{mode}.json")))
            json.dump(hdr, open(os.path.join(self.config.checkpoint_dir, "active_header.json"), "w"),
                      ensure_ascii=False)
            shutil.copy2(os.path.join(self.config.checkpoint_dir, f"{username}_model_{mode}.pth"),
                         os.path.join(self.config.checkpoint_dir, "active_model.pth"))
            shutil.copy2(os.path.join(self.config.checkpoint_dir, f"{username}_scaler_{mode}.pkl"),
                         os.path.join(self.config.checkpoint_dir, "active_scaler.pkl"))

        # 활성화 프로파일 저장
        json.dump({"best_profile": best, "metric": self.config.metric, "table": df.to_dict(orient="records")},
                  open(os.path.join(self.config.checkpoint_dir, "active_profile.json"), "w"))
        return {"best": best, "table": df.to_dict(orient="records")}

    def _train_with_external_val(self, train_df, val_df, features, window_size, dropout_rate, k1, k2, base_out_dir,
                                 device,
                                 batch_size=32, epochs=20, patience=7, lr=1e-3, wd=1e-5, two_convs=True,
                                 train_overlap=0.75, val_overlap=0.5, label_threshold=0.6,
                                 select_policy="auc_then_loss"):
        tag = f'win{window_size}_do{str(dropout_rate).replace(".", "")}_k{k1}_h{k2}_cnn-gru'
        run_dir = os.path.join(base_out_dir, 'pipelines', tag)
        os.makedirs(run_dir, exist_ok=True)

        scaler = StandardScaler()
        tdf = train_df.copy();
        vdf = val_df.copy()
        tdf[features] = scaler.fit_transform(tdf[features])
        vdf[features] = scaler.transform(vdf[features])
        with open(os.path.join(run_dir, f'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        train_stride = data_utils.compute_stride(window_size, 'train', train_overlap=train_overlap,
                                                 val_overlap=val_overlap)
        val_stride = data_utils.compute_stride(window_size, 'val', train_overlap=train_overlap, val_overlap=val_overlap)

        X_train, y_train = data_utils.create_sequences(tdf, features, window_size, stride=train_stride,
                                                       threshold=label_threshold)
        X_val, y_val = data_utils.create_sequences(vdf, features, window_size, stride=val_stride,
                                                   threshold=label_threshold)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            err = f'No sequences created. train={X_train.shape[0]} val={X_val.shape[0]} tag={tag}'
            with open(os.path.join(run_dir, 'error.txt'), 'w', encoding='utf-8') as f: f.write(err)
            raise RuntimeError(err)

        train_loader = DataLoader(training_helpers.TimeSeriesDataset(X_train, y_train), batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(training_helpers.TimeSeriesDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        pos_count = max(1, int((y_train == 1).sum()))
        neg_count = max(1, int((y_train == 0).sum()))
        ratio = min(neg_count / max(1, pos_count), 50.0)
        pos_weight = torch.tensor([ratio], device=device, dtype=torch.float32)

        model = TimeSeriesCNNGRU(len(features), window_size, dropout_rate, k1, k2, two_convs=two_convs).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_loss_path = os.path.join(run_dir, f'best_loss.pth')
        best_acc_path = os.path.join(run_dir, f'best_acc.pth')
        best_auc_path = os.path.join(run_dir, f'best_auc.pth')
        last_path = os.path.join(run_dir, f'last.pth')

        early = training_helpers.EarlyStopping(patience=patience, model_path=best_loss_path)
        train_losses, val_losses, train_accs, val_accs, val_aucs = [], [], [], [], []
        best_acc, best_auc = -1.0, -1.0
        epoch_times = []
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            t0 = time.time()
            model.train();
            run_loss = 0.0;
            correct = 0;
            total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward();
                optimizer.step()
                run_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total += y.size(0);
                correct += (preds == y).sum().item()
            tr_loss = run_loss / len(train_loader.dataset);
            tr_acc = correct / total

            model.eval();
            v_loss = 0.0;
            v_correct = 0;
            v_total = 0;
            all_probs = [];
            all_labels = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x).squeeze(-1)
                    loss = criterion(logits, y)
                    v_loss += loss.item() * x.size(0)
                    probs = torch.sigmoid(logits);
                    preds = (probs >= 0.5).float()
                    v_total += y.size(0);
                    v_correct += (preds == y).sum().item()
                    all_probs.extend(probs.detach().cpu().numpy().tolist());
                    all_labels.extend(y.detach().cpu().numpy().tolist())

            va_loss = v_loss / len(val_loader.dataset) if len(val_loader.dataset) else float('inf')
            va_acc = v_correct / v_total if v_total else 0.0
            try:
                va_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
            except Exception:
                va_auc = np.nan

            train_losses.append(tr_loss);
            val_losses.append(va_loss);
            train_accs.append(tr_acc);
            val_accs.append(va_acc);
            val_aucs.append(va_auc)

            if va_acc > best_acc:
                best_acc = va_acc;
                torch.save(model.state_dict(), best_acc_path)
            if not np.isnan(va_auc) and va_auc > best_auc:
                best_auc = va_auc;
                torch.save(model.state_dict(), best_auc_path)

            early(va_loss, model)
            epoch_times.append(time.time() - t0)
            if early.early_stop: break

        torch.save(model.state_dict(), last_path)

        if select_policy == "auc_then_loss":
            chosen = best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else best_loss_path
        elif select_policy == "loss_then_auc":
            chosen = best_loss_path if os.path.exists(best_loss_path) else (
                best_auc_path if (not np.isnan(best_auc) and os.path.exists(best_auc_path)) else last_path)
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

        model.eval();
        v_correct = 0;
        v_total = 0;
        v_probs = [];
        v_preds = [];
        v_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                probs = torch.sigmoid(logits);
                preds = (probs >= 0.5).float()
                v_total += y.size(0);
                v_correct += (preds == y).sum().item()
                v_probs.extend(probs.cpu().numpy().tolist());
                v_preds.extend(preds.cpu().numpy().tolist());
                v_labels.extend(y.cpu().numpy().tolist())

        val_acc_final = v_correct / v_total if v_total else 0.0
        roc_png = None
        try:
            val_auc_final = roc_auc_score(np.array(v_labels), np.array(v_probs))
            fpr, tpr, _ = roc_curve(np.array(v_labels), np.array(v_probs))
            roc_png = os.path.join(run_dir, f"roc.png")
            training_helpers.save_roc_curve(fpr, tpr, title=f"ROC {tag}", save_path=roc_png)
        except Exception:
            val_auc_final = np.nan

        metrics_df = pd.DataFrame({
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses, "val_loss": val_losses,
            "train_acc": train_accs, "val_acc": val_accs, "val_auc": val_aucs
        })
        metrics_csv = os.path.join(run_dir, f"training_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        training_helpers.save_training_curves(metrics_df, title_suffix=tag,
                                              save_path_prefix=os.path.join(run_dir, f"train"))

        avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float('nan')
        rss_mb, gpu_mb = training_helpers.get_memory_usage_mb()

        rep_txt = os.path.join(run_dir, f"classification_report.txt")
        if len(v_labels) and len(v_preds):
            rep = classification_report(v_labels, v_preds, target_names=['class0', 'class1'], zero_division=0,
                                        output_dict=False)
            with open(rep_txt, 'w', encoding='utf-8') as f:
                f.write(rep)
        else:
            with open(rep_txt, 'w', encoding='utf-8') as f:
                f.write("Empty validation predictions.")

        print(
            f"[DONE] {tag} acc_best={np.nanmax(val_accs) if len(val_accs) else float('nan'):.4f} auc_best={(np.nanmax(val_aucs) if len(val_aucs) else float('nan')):.4f} acc_final={val_acc_final:.4f} auc_final={(0.0 if math.isnan(val_auc_final) else val_auc_final):.4f}")
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
            "학습 가능 파라미터 수": int(training_helpers.count_parameters(model)),
            "run_dir": run_dir, "best_model_path": final_best_path, "metrics_csv": metrics_csv, "roc_png": roc_png
        }

    def run_hyperparameter_sweep(self, train_csv_path: str, val_csv_path: str = None):
        """주어진 학습 데이터로 하이퍼파라미터 탐색을 수행합니다."""
        print("--- 하이퍼파라미터 탐색 시작 ---")
        device = self.device

        # Load and split data
        train_df, val_df, feats_train = data_utils.split_train_val_from_csv(train_csv_path,
                                                                            test_ratio=self.config.validation_ratio,
                                                                            seed=self.config.seed)

        window_list = self.config.window_sizes
        dropout_list = self.config.dropout_rates
        kernel_sets = self.config.kernel_sets

        results = []
        for ws in window_list:
            for dr in dropout_list:
                for k1, k2 in kernel_sets:
                    print("\n" + "=" * 90)
                    print(f"학습 시작: win={ws}, do={dr}, CNN-GRU(k={k1}, h={k2})")
                    print("=" * 90)
                    try:
                        summary = self._train_with_external_val(
                            train_df=train_df, val_df=val_df, features=feats_train, window_size=ws,
                            dropout_rate=dr, k1=k1, k2=k2, base_out_dir=self.config.output_dir, device=device,
                            batch_size=self.config.batch_size, epochs=self.config.epochs,
                            patience=self.config.early_stop_patience,
                            lr=self.config.learning_rate, wd=self.config.weight_decay,
                            two_convs=self.config.two_convs, train_overlap=self.config.train_overlap,
                            val_overlap=self.config.val_overlap, label_threshold=self.config.label_threshold,
                            select_policy=self.config.select_policy
                        )
                        results.append(summary)
                    except Exception as e:
                        tag = f'win{ws}_do{str(dr).replace(".", "")}_k{k1}_h{k2}_cnn-gru'
                        err_dir = os.path.join(self.config.output_dir, 'pipelines', tag)
                        os.makedirs(err_dir, exist_ok=True)
                        with open(os.path.join(err_dir, 'error.txt'), 'w', encoding='utf-8') as f:
                            f.write(str(e))
                        print(f"[ERROR] {tag} -> {e}")

        summary_df = pd.DataFrame(results, columns=[
            "윈도우", "드롭아웃", "커널1(ks)", "GRU히든",
            "학습 시퀀스", "검증 시퀀스",
            "최고 검증 손실", "최고 검증 정확도", "최고 검증 AUC",
            "최종 검증 정확도", "최종 검증 AUC",
            "평균 에포크 시간(초)", "피크 GPU 메모리(MB)", "프로세스 RSS 메모리(MB)", "학습 가능 파라미터 수",
            "run_dir", "best_model_path", "metrics_csv", "roc_png"
        ])

        final_dir = os.path.join(self.config.output_dir, "cnn_final_result")
        os.makedirs(final_dir, exist_ok=True)
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

        lbls = [f"w{int(r['윈도우'])}/do={r['드롭아웃']}/k={int(r['커널1(ks)'])}/h={int(r['GRU히든'])}" for _, r in
                ranked.iterrows()]
        vals_acc_best = ranked["최고 검증 정확도"].fillna(0).tolist()
        vals_auc_best = ranked["최고 검증 AUC"].fillna(0).tolist()
        vals_acc_final = ranked["최종 검증 정확도"].fillna(0).tolist()
        vals_auc_final = ranked["최종 검증 AUC"].fillna(0).tolist()

        training_helpers.bar_plot(lbls, vals_acc_best, "config", "acc", "최고 검증 정확도 (CNN-GRU)",
                                  os.path.join(final_dir, "best_val_acc.png"))
        training_helpers.bar_plot(lbls, vals_auc_best, "config", "auc", "최고 검증 AUC (CNN-GRU)",
                                  os.path.join(final_dir, "best_val_auc.png"))
        training_helpers.bar_plot(lbls, vals_acc_final, "config", "acc", "최종 검증 정확도 (CNN-GRU)",
                                  os.path.join(final_dir, "final_val_acc.png"))
        training_helpers.bar_plot(lbls, vals_auc_final, "config", "auc", "최종 검증 AUC (CNN-GRU)",
                                  os.path.join(final_dir, "final_val_auc.png"))

        print(f"\n요약: {summary_csv}")
        print(f"순위표: {ranking_csv}")
        print(f"파이프라인 루트: {os.path.join(self.config.output_dir, 'pipelines')}")
