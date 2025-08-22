import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        if len(sequences)==0 or len(labels)==0: raise ValueError('empty dataset')
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i], self.labels[i]

class EarlyStopping:
    def __init__(self, patience=7, delta=0, model_path='best.pth'):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_model_path = model_path
        self.early_stop = False
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def __call__(self, val_loss, model):
        if np.isnan(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.best_model_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

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
