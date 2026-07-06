from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def synthetic_latents(n=240, d=32):
    rng = np.random.default_rng(42)
    labels = np.repeat([0, 1, 2], n // 3)
    domains = np.tile(np.repeat([0, 1], n // 6), 3)
    z_morph = rng.normal(size=(n, d))
    z_style = rng.normal(size=(n, d))
    for c in range(3):
        z_morph[labels == c, :2] += np.array([c * 4.0, (c % 2) * 3.0])
    z_style[domains == 0, :2] += np.array([-3.0, 0.0])
    z_style[domains == 1, :2] += np.array([3.0, 0.0])
    return z_morph, z_style, labels, domains

def embed(x):
    return TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30).fit_transform(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None)
    parser.add_argument("--output", default="outputs/figures/latent_disentanglement_300dpi.png")
    args = parser.parse_args()

    if args.npz is None:
        z_morph, z_style, labels, domains = synthetic_latents()
    else:
        data = np.load(args.npz)
        z_morph, z_style, labels, domains = data["z_morph"], data["z_style"], data["labels"], data["domains"]

    emb_morph = embed(z_morph)
    emb_style = embed(z_style)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    axes[0].scatter(emb_morph[:, 0], emb_morph[:, 1], c=labels, s=18, cmap="viridis")
    axes[0].set_title(r"$Z_{morph}$ by disease class")
    axes[1].scatter(emb_morph[:, 0], emb_morph[:, 1], c=domains, s=18, cmap="coolwarm")
    axes[1].set_title(r"$Z_{morph}$ by dataset/scanner")
    axes[2].scatter(emb_style[:, 0], emb_style[:, 1], c=domains, s=18, cmap="coolwarm")
    axes[2].set_title(r"$Z_{style}$ by dataset/scanner")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    fig.savefig(str(Path(args.output).with_suffix(".pdf")))

if __name__ == "__main__":
    main()
