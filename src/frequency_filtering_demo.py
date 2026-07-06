from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def synthetic_mri_like_image(size=256):
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    brain = np.exp(-((x / 0.75) ** 2 + (y / 0.95) ** 2) * 2.5)
    ventricles = np.exp(-((x / 0.13) ** 2 + (y / 0.22) ** 2) * 8)
    image = brain - 0.9 * ventricles
    image += 0.08 * np.random.default_rng(42).normal(size=(size, size))
    return np.clip(image, 0, 1)

def radial_mask(shape, cutoff, high_pass=False):
    h, w = shape
    y = np.fft.fftfreq(h).reshape(h, 1)
    x = np.fft.fftfreq(w).reshape(1, w)
    r = np.sqrt(x*x + y*y)
    mask = (r <= cutoff).astype(np.float32)
    return 1.0 - mask if high_pass else mask

def main():
    image = synthetic_mri_like_image()
    f = np.fft.fft2(image)
    spectrum = np.log1p(np.abs(np.fft.fftshift(f)))
    low = np.fft.ifft2(f * radial_mask(image.shape, 0.08)).real
    high = np.fft.ifft2(f * radial_mask(image.shape, 0.08, high_pass=True)).real

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=300)
    for ax, im, title in zip(
        axes,
        [image, spectrum, gaussian_filter(low, 1), high],
        ["Input MRI Slice", "DFT Magnitude Spectrum", "Low-Pass Morphology", "High-Pass Style/Noise"],
    ):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Frequency Spectrum Filtering Visualization")
    fig.tight_layout()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/figures/frequency_filtering_demo_300dpi.png", dpi=300)
    fig.savefig("outputs/figures/frequency_filtering_demo.pdf")

if __name__ == "__main__":
    main()
