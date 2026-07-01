#!/bin/bash
set -e
python -m smd_mamba.training.train --config configs/smd_mamba_config.yaml --train_csv manifests/adni_train.csv --val_csv manifests/adni_val.csv --output_dir outputs/adni_training
