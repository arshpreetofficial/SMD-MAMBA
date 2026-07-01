#!/bin/bash
set -e
python -m smd_mamba.training.cross_validation --manifest_csv manifests/adni_all.csv --output_dir manifests/folds --num_folds 5 --seed 42
