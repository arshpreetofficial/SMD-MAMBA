from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score

def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3):
    labels_np = labels.detach().cpu().numpy()
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels_np, preds, labels=list(range(num_classes)))

    sensitivity, specificity = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity.append(tp / (tp + fn + 1e-8))
        specificity.append(tn / (tn + fp + 1e-8))

    try:
        auc = roc_auc_score(labels_np, probs, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(labels_np, preds),
        "precision_macro": precision_score(labels_np, preds, average="macro", zero_division=0),
        "sensitivity_macro": float(np.mean(sensitivity)),
        "specificity_macro": float(np.mean(specificity)),
        "f1_macro": f1_score(labels_np, preds, average="macro", zero_division=0),
        "auc_ovr": auc,
    }
