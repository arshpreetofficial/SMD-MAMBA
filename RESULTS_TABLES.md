# Reported Result Tables

## Cross-Domain Zero-Shot Generalization: ADNI to OASIS

| Model | Target ACC (%) | Drop (%) | Target SEN (%) | Target SPE (%) | Target AUC |
|---|---:|---:|---:|---:|---:|
| 3D-ResNet50 | 71.08 ± 1.15 | -17.46 | 68.42 ± 1.31 | 73.15 ± 1.02 | 0.772 |
| 3D-DenseNet121 | 74.52 ± 0.98 | -15.69 | 72.11 ± 1.12 | 76.24 ± 0.85 | 0.801 |
| AlzVNet | 78.34 ± 0.84 | -13.76 | 76.05 ± 0.95 | 79.91 ± 0.71 | 0.844 |
| 3D-cGAN | 79.24 ± 0.92 | -12.32 | 77.85 ± 1.04 | 80.36 ± 0.87 | 0.841 |
| 3D-ViT | 79.15 ± 0.91 | -14.97 | 77.30 ± 1.05 | 80.48 ± 0.82 | 0.839 |
| SpectroCVT-Net | 88.52 ± 0.45 | -6.62 | 87.33 ± 0.52 | 89.47 ± 0.41 | 0.921 |
| SMD-Mamba | 93.76 ± 0.24 | -2.69 | 92.81 ± 0.31 | 94.48 ± 0.20 | 0.956 |

## Computational Efficiency

| Network | Parameters (M) | FLOPs (G) | Latency (ms) |
|---|---:|---:|---:|
| 3D-ResNet50 | 46.2 | 64.2 | 58 |
| 3D-DenseNet121 | 32.8 | 48.5 | 72 |
| 3D-ViT | 92.6 | 124.5 | 184 |
| Swin-UNETR | 64.1 | 89.4 | 115 |
| Standard 3D Vision Mamba | 26.5 | 34.1 | 38 |
| SMD-Mamba | 28.4 | 36.8 | 42 |
