from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if mask.sum() == 0:
        mask = np.ones_like(volume, dtype=bool)
    return ((volume - volume[mask].mean()) / (volume[mask].std() + 1e-6)).astype(np.float32)

def minmax_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if mask.sum() == 0:
        mask = np.ones_like(volume, dtype=bool)
    return ((volume - volume[mask].min()) / (volume[mask].max() - volume[mask].min() + 1e-6)).astype(np.float32)

def normalize_by_mode(volume: np.ndarray, mode: str = "zscore") -> np.ndarray:
    if mode == "zscore":
        return zscore_normalize(volume)
    if mode == "minmax":
        return minmax_normalize(volume)
    if mode == "none":
        return volume.astype(np.float32)
    raise ValueError(f"Unknown normalization mode: {mode}")

def resize_volume_tensor(volume: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    volume = volume.unsqueeze(0)
    volume = F.interpolate(volume, size=target_size, mode="trilinear", align_corners=False)
    return volume.squeeze(0)
