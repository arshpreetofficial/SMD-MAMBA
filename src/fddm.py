from __future__ import annotations
import torch
import torch.nn as nn

def radial_frequency_mask(shape, cutoff, device, high_pass=False):
    d, h, w = shape
    z = torch.fft.fftfreq(d, device=device).view(d, 1, 1)
    y = torch.fft.fftfreq(h, device=device).view(1, h, 1)
    x = torch.fft.fftfreq(w, device=device).view(1, 1, w)
    radius = torch.sqrt(x * x + y * y + z * z)
    mask = (radius <= cutoff).float()
    if high_pass:
        mask = 1.0 - mask
    return mask.view(1, 1, d, h, w)

class FrequencyDomainDisentanglement(nn.Module):
    """
    Branch A: Fourier-domain morphology/style disentanglement.
    Low frequencies -> Z_morph.
    High frequencies -> Z_style.
    """
    def __init__(self, embed_dim: int, low_pass_cutoff: float = 0.18, high_pass_cutoff: float = 0.18):
        super().__init__()
        self.low_pass_cutoff = low_pass_cutoff
        self.high_pass_cutoff = high_pass_cutoff
        self.morph_projector = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.style_projector = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, d, h, w = feature_map.shape
        freq = torch.fft.fftn(feature_map, dim=(-3, -2, -1), norm="ortho")
        low_mask = radial_frequency_mask((d, h, w), self.low_pass_cutoff, feature_map.device, high_pass=False)
        high_mask = radial_frequency_mask((d, h, w), self.high_pass_cutoff, feature_map.device, high_pass=True)

        morph_freq = freq * low_mask
        style_freq = freq * high_mask

        z_morph_map = torch.fft.ifftn(morph_freq, dim=(-3, -2, -1), norm="ortho").real
        z_style_map = torch.fft.ifftn(style_freq, dim=(-3, -2, -1), norm="ortho").real

        z_morph = z_morph_map.flatten(2).transpose(1, 2)
        z_style = z_style_map.flatten(2).transpose(1, 2)

        z_morph = self.morph_projector(z_morph)
        z_style = self.style_projector(z_style)

        spectrum = torch.log1p(torch.abs(torch.fft.fftshift(freq, dim=(-3, -2, -1))))
        return {
            "z_morph_map": z_morph_map,
            "z_style_map": z_style_map,
            "z_morph": z_morph,
            "z_style": z_style,
            "spectrum": spectrum,
        }
