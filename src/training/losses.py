from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def orthogonal_disentanglement_loss(z_morph: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
    z_morph = F.normalize(z_morph.flatten(1), dim=1)
    z_style = F.normalize(z_style.flatten(1), dim=1)
    cross = torch.sum(z_morph * z_style, dim=1)
    return torch.mean(cross.pow(2))

class SMDMambaLoss(nn.Module):
    """
    L_total = L_CE + lambda_ortho * L_ortho
    """
    def __init__(self, lambda_ortho: float = 0.1):
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        ce_loss = self.ce(outputs["logits"], labels)
        ortho_loss = orthogonal_disentanglement_loss(outputs["z_morph"], outputs["z_style"])
        total_loss = ce_loss + self.lambda_ortho * ortho_loss
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "ortho_loss": ortho_loss.detach(),
        }
