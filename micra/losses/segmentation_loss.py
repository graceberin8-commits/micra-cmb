"""
Section 3.7 & 4.5: Segmentation Loss (Eq. 22) and Joint Training Objective.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from micra.config import TrainingConfig


class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice loss for extreme foreground-background imbalance."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = probs.reshape(probs.shape[0], -1)
        t = targets.reshape(targets.shape[0], -1).float()
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class MICRAJointLoss(nn.Module):
    """
    Implements Eq. (22) and the joint MICRA optimization objective (Section 3.6, 3.7, 4.5):
        L_{seg} = \\lambda_{BCE} L_{BCE} + \\lambda_{Dice} L_{Dice}       [Eq. 22]
        L_{total} = L_{seg} + \\lambda_{proto} L_{proto}
    with Section 4.5 default weights:
        \\lambda_{BCE} = 0.05, \\lambda_{Dice} = 256, \\lambda_{proto} = 8.
    """

    def __init__(self, config: TrainingConfig, normalize_weights: bool = True):
        super().__init__()
        self.cfg = config
        self.dice_loss = SoftDiceLoss()
        self.normalize_weights = normalize_weights

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
        l_proto: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        targets_f = targets.float()
        l_bce = F.binary_cross_entropy_with_logits(logits, targets_f)
        l_dice = self.dice_loss(probs, targets_f)

        # Eq. (22): L_seg = \lambda_BCE * L_BCE + \lambda_Dice * L_Dice
        raw_l_seg = self.cfg.lambda_bce * l_bce + self.cfg.lambda_dice * l_dice
        raw_l_total = raw_l_seg + self.cfg.lambda_proto * l_proto

        if self.normalize_weights:
            denom = max(1.0, self.cfg.lambda_bce + self.cfg.lambda_dice + self.cfg.lambda_proto)
            l_total = raw_l_total / denom
            l_seg = raw_l_seg / denom
        else:
            l_total = raw_l_total
            l_seg = raw_l_seg

        return {
            "loss": l_total,
            "l_seg": l_seg.detach(),
            "l_bce": l_bce.detach(),
            "l_dice": l_dice.detach(),
            "l_proto": l_proto.detach(),
            "raw_loss": raw_l_total.detach(),
        }
