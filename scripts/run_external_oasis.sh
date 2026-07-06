#!/bin/bash
set -e
python -m smd_mamba.evaluation.evaluate_external --config configs/smd_mamba_config.yaml --test_csv manifests/oasis_external.csv --checkpoint outputs/adni_training/checkpoints/best_model.pt --output_dir outputs/oasis_external
