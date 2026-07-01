# SMD-Mamba

## Mitigating Scanner-Induced Domain Shift in Alzheimer's Diagnosis via Frequency-Domain Morphological Disentanglement and State Space Modeling

This repository provides a PyTorch implementation template for **SMD-Mamba**, a scanner-robust 3D structural MRI framework for Alzheimer's disease diagnosis.

## Core Idea

SMD-Mamba addresses scanner-induced domain shift by combining:

1. **Spectral-Morphological Disentanglement (SMD)**
   - 3D Fourier transform of volumetric MRI features.
   - Low-frequency stream for scanner-invariant biological morphology.
   - High-frequency stream for scanner-specific style/noise.
   - Orthogonal disentanglement loss.

2. **3D Omni-Directional State Space Modeling**
   - Linear-complexity volumetric sequence processing.
   - Multi-directional whole-brain scanning.
   - Morphology-guided state transitions.

3. **Zero-shot Cross-domain Diagnosis**
   - Train on ADNI.
   - Test directly on unseen OASIS.
   - No target-domain fine-tuning.

## Repository Structure

```text
SMD-Mamba/
├── configs/smd_mamba_config.yaml
├── data/sample_manifest.csv
├── docs/
│   ├── METHODS_SUMMARY.md
│   ├── RESULTS_TABLES.md
│   └── figures/
├── scripts/
│   ├── run_train_adni.sh
│   ├── run_external_oasis.sh
│   ├── run_cross_validation.sh
│   └── generate_all_figures.sh
├── src/smd_mamba/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── fddm.py
│   │   ├── mamba_blocks.py
│   │   └── smd_mamba.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── train.py
│   │   ├── cross_validation.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate_external.py
│   └── visualization/
│       ├── latent_disentanglement.py
│       ├── frequency_filtering_demo.py
│       ├── robustness_chart.py
│       └── pareto_frontier.py
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── setup.py
```

## Dataset Manifest

```csv
subject_id,image_path,label,site,scanner
ADNI_001,/path/to/ADNI_001_T1w.nii.gz,0,ADNI,Siemens
ADNI_002,/path/to/ADNI_002_T1w.nii.gz,1,ADNI,GE
ADNI_003,/path/to/ADNI_003_T1w.nii.gz,2,ADNI,Philips
```

Labels:

```text
0 = CN
1 = MCI
2 = AD
```

## Installation

```bash
git clone https://github.com/<your-username>/SMD-Mamba.git
cd SMD-Mamba
conda create -n smd-mamba python=3.10 -y
conda activate smd-mamba
pip install -r requirements.txt
pip install -e .
```

## Training on ADNI

```bash
bash scripts/run_train_adni.sh
```

## Zero-shot External Validation on OASIS

```bash
bash scripts/run_external_oasis.sh
```

## Subject-level Cross-validation

```bash
bash scripts/run_cross_validation.sh
```

## Generate Paper Figures

```bash
bash scripts/generate_all_figures.sh
```

## Objective Function

```text
L_total = L_CE + lambda_ortho * L_ortho
L_ortho = ||Z_morph^T Z_style||_F^2
```

## Reported Result Summary

| Model | ADNI ACC (%) | OASIS ACC (%) | Drop (%) |
|---|---:|---:|---:|
| 3D-ResNet50 | 88.54 | 71.08 | 17.46 |
| 3D-ViT | 94.12 | 79.15 | 14.97 |
| 3D-cGAN | 91.56 | 79.24 | 12.32 |
| SpectroCVT-Net | 95.14 | 88.52 | 6.62 |
| SMD-Mamba | 96.45 | 93.76 | 2.69 |

## Citation

```bibtex
@article{kaur2026smdmamba,
  title={Mitigating Scanner-Induced Domain Shift in Alzheimer's Diagnosis via Frequency-Domain Morphological Disentanglement and State Space Modeling},
  author={Kaur, Arshpreet and Kaur, Jagdeep},
  year={2026}
}
```

## Disclaimer

This repository is for research use only and is not a certified medical device.
