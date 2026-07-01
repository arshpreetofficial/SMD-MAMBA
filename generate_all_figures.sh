#!/bin/bash
set -e
python -m smd_mamba.visualization.latent_disentanglement
python -m smd_mamba.visualization.frequency_filtering_demo
python -m smd_mamba.visualization.robustness_chart
python -m smd_mamba.visualization.pareto_frontier
