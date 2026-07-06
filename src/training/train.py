from __future__ import annotations
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from smd_mamba.data.dataset import MRIDataset
from smd_mamba.evaluation.metrics import compute_classification_metrics
from smd_mamba.models.smd_mamba import SMDMamba
from smd_mamba.training.losses import SMDMambaLoss
from smd_mamba.training.utils import AverageMeter, CSVLogger, ensure_dir, get_device, load_yaml, save_checkpoint, seed_everything

def build_parser():
    p = argparse.ArgumentParser("Train SMD-Mamba")
    p.add_argument("--config", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--output_dir", default="outputs/adni_training")
    return p

def build_model(config):
    m, d = config["model"], config["data"]
    return SMDMamba(
        in_channels=d["in_channels"],
        num_classes=d["num_classes"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        state_dim=m["state_dim"],
        expansion_factor=m["expansion_factor"],
        num_blocks=m["num_blocks"],
        dropout=m["dropout"],
        low_pass_cutoff=m["low_pass_cutoff"],
        high_pass_cutoff=m["high_pass_cutoff"],
    )

def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train(train)
    meter = AverageMeter()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)["total_loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)["total_loss"]
                if train:
                    loss.backward()
                    optimizer.step()

        meter.update(loss.item(), images.size(0))
        all_logits.append(outputs["logits"].detach())
        all_labels.append(labels.detach())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(logits, labels, num_classes=model.num_classes)
    metrics["loss"] = meter.avg
    return metrics

def main():
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    seed_everything(config.get("seed", 42))

    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    device = get_device()

    train_ds = MRIDataset(args.train_csv, input_size=config["data"]["input_size"], normalization=config["data"]["normalization"])
    val_ds = MRIDataset(args.val_csv, input_size=config["data"]["input_size"], normalization=config["data"]["normalization"])

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = SMDMambaLoss(lambda_ortho=config["training"]["lambda_ortho"])
    scaler = GradScaler() if config["training"]["mixed_precision"] and device.type == "cuda" else None

    logger = CSVLogger(log_dir / "training_log.csv", ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_f1", "val_auc"])
    best_acc = -1.0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, None, train=False)
        scheduler.step()

        print(f"Epoch {epoch:03d} | Train Acc {train_metrics['accuracy']:.4f} | Val Acc {val_metrics['accuracy']:.4f} | Val AUC {val_metrics['auc_ovr']:.4f}")
        logger.log({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_macro"],
            "val_auc": val_metrics["auc_ovr"],
        })

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, best_acc, config=config)

if __name__ == "__main__":
    main()
