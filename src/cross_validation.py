from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from smd_mamba.training.utils import ensure_dir

def main():
    p = argparse.ArgumentParser("Create subject-level folds")
    p.add_argument("--manifest_csv", required=True)
    p.add_argument("--output_dir", default="manifests/folds")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.manifest_csv)
    subject_df = df.groupby("subject_id").first().reset_index()
    subjects = subject_df["subject_id"].values
    labels = subject_df["label"].values

    fold_dir = ensure_dir(args.output_dir)
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])
        df[df["subject_id"].isin(train_subjects)].to_csv(fold_dir / f"fold_{fold_idx}_train.csv", index=False)
        df[df["subject_id"].isin(val_subjects)].to_csv(fold_dir / f"fold_{fold_idx}_val.csv", index=False)
        print(f"Saved fold {fold_idx}")

if __name__ == "__main__":
    main()
