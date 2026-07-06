from __future__ import annotations
import torch
import torch.nn as nn
from .fddm import FrequencyDomainDisentanglement
from .mamba_blocks import MorphologyGuidedMambaBlock

class SMDMamba(nn.Module):
    """
    Scanner-Mitigated Domain Mamba for 3D sMRI classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        patch_size: int = 8,
        embed_dim: int = 96,
        state_dim: int = 16,
        expansion_factor: int = 2,
        num_blocks: int = 4,
        dropout: float = 0.1,
        low_pass_cutoff: float = 0.18,
        high_pass_cutoff: float = 0.18,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.fddm = FrequencyDomainDisentanglement(embed_dim, low_pass_cutoff, high_pass_cutoff)
        self.blocks = nn.ModuleList([
            MorphologyGuidedMambaBlock(embed_dim, expansion_factor, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.patch_embed(x)
        sequence = feature_map.flatten(2).transpose(1, 2)

        fddm_outputs = self.fddm(feature_map)
        z_morph = fddm_outputs["z_morph"]
        z_style = fddm_outputs["z_style"]

        for block in self.blocks:
            sequence = block(sequence, z_morph)

        sequence = self.norm(sequence)
        pooled = sequence.mean(dim=1)
        logits = self.classifier(pooled)
        probabilities = torch.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "features": pooled,
            "tokens": sequence,
            "z_morph": z_morph,
            "z_style": z_style,
            "z_morph_map": fddm_outputs["z_morph_map"],
            "z_style_map": fddm_outputs["z_style_map"],
            "spectrum": fddm_outputs["spectrum"],
        }
