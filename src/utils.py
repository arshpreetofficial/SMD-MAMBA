from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np
import torch
import yaml

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0
    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        ensure_dir(self.path.parent)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    def log(self, row):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow({k: row.get(k, "") for k in self.fieldnames})

def save_checkpoint(path, model, optimizer, epoch, best_metric, config=None):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": best_metric,
        "config": config,
    }, path)

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
