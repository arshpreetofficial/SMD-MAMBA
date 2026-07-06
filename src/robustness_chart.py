from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    models = ["3D-ResNet50", "3D-ViT", "3D-cGAN", "SpectroCVT-Net", "SMD-Mamba"]
    adni = np.array([88.54, 94.12, 91.56, 95.14, 96.45])
    oasis = np.array([71.08, 79.15, 79.24, 88.52, 93.76])
    drop = adni - oasis

    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.bar(x - width/2, adni, width, label="ADNI Source")
    ax.bar(x + width/2, oasis, width, label="OASIS Target")
    for i in range(len(models)):
        ax.annotate(f"Δ {drop[i]:.2f}%", xy=(x[i], oasis[i]), xytext=(x[i], adni[i] + 1), ha="center", arrowprops=dict(arrowstyle="->"))
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Cross-Domain Robustness: ADNI Source to OASIS Target")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(65, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/figures/cross_domain_robustness_300dpi.png", dpi=300)
    fig.savefig("outputs/figures/cross_domain_robustness.pdf")

if __name__ == "__main__":
    main()
