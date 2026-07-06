from __future__ import annotations
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from smd_mamba.data.dataset import MRIDataset
from smd_mamba.evaluation.metrics import compute_classification_metrics
from smd_mamba.training.train import build_model
from smd_mamba.training.utils import ensure_dir, get_device, load_checkpoint, load_yaml, save_json

@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Evaluate SMD-Mamba on external data")
    p.add_argument("--config", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="outputs/oasis_external")
    args = p.parse_args()

    config = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    ds = MRIDataset(args.test_csv, input_size=config["data"]["input_size"], normalization=config["data"]["normalization"])
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    all_logits, all_labels, records = [], [], []
    for batch in tqdm(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        probs = outputs["probabilities"].detach().cpu()
        preds = probs.argmax(dim=1)

        all_logits.append(outputs["logits"].detach())
        all_labels.append(labels.detach())

        for i, sid in enumerate(batch["subject_id"]):
            records.append({
                "subject_id": sid,
                "label": int(labels[i].cpu()),
                "prediction": int(preds[i]),
                "prob_CN": float(probs[i, 0]),
                "prob_MCI": float(probs[i, 1]) if probs.shape[1] > 1 else None,
                "prob_AD": float(probs[i, 2]) if probs.shape[1] > 2 else None,
            })

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(logits, labels, num_classes=config["data"]["num_classes"])

    save_json(output_dir / "external_metrics.json", metrics)
    pd.DataFrame(records).to_csv(output_dir / "external_predictions.csv", index=False)
    print(metrics)

if __name__ == "__main__":
    main()
