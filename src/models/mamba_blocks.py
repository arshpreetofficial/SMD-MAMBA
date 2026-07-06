from __future__ import annotations
import torch
import torch.nn as nn

class MorphologyGuidedMambaBlock(nn.Module):
    """
    Lightweight morphology-guided state-space-style sequence block.
    It uses Z_morph to gate token mixing, approximating the paper's
    morphology-guided discretization behavior in a pure PyTorch template.
    """
    def __init__(self, embed_dim: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_morph = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        self.morph_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.depthwise_mixer = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z_morph: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm_x(x)
        morph_norm = self.norm_morph(z_morph)

        h = self.input_projection(x_norm)
        gate = self.morph_gate(torch.cat([x_norm, morph_norm], dim=-1))
        h = h * gate

        h = self.depthwise_mixer(h.transpose(1, 2)).transpose(1, 2)
        h = self.output_projection(h)
        return residual + self.dropout(h)
