from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    models = ["3D-ResNet50", "3D-DenseNet121", "3D-ViT", "Swin-UNETR", "Standard 3D Vision Mamba", "SMD-Mamba"]
    latency = np.array([58, 72, 184, 115, 38, 42])
    params = np.array([46.2, 32.8, 92.6, 64.1, 26.5, 28.4])
    acc = np.array([71.08, 74.52, 79.15, 80.22, 82.14, 93.76])

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.scatter(latency, acc, s=params * 16, alpha=0.75, edgecolor="black")
    for i, model in enumerate(models):
        ax.annotate(model, (latency[i] + 2, acc[i] + 0.25), fontsize=9)
    ax.set_xlabel("Inference Latency per 3D Volume (ms)")
    ax.set_ylabel("Cross-Domain Accuracy on OASIS (%)")
    ax.set_title("Efficiency vs. Accuracy Pareto Frontier")
    ax.set_xlim(0, 210)
    ax.set_ylim(65, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/figures/pareto_frontier_300dpi.png", dpi=300)
    fig.savefig("outputs/figures/pareto_frontier.pdf")

if __name__ == "__main__":
    main()
