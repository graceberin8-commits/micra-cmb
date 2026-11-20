"""
Section 3.7: Segmentation Decoder and Pixel-Wise Prediction Head (Eqs. 20–21).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRefinementBlock(nn.Module):
    """Upsampling + skip-connection fusion + residual convolutional refinement."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([x, skip], dim=1)
        out = self.conv1(cat)
        out = self.conv2(out) + self.residual(cat)
        return out


class MICRASegmentationDecoder(nn.Module):
    """
    Implements Section 3.7 (Eqs. 20–21):
      - Eq. (20): D_z = Decoder(F_z^{CTX}; \\theta_d)
      - Eq. (21): \\hat{M}_z = \\sigma(W_m D_z + b_m)
    Optionally incorporates the explicit SCP sphericity channel S_z(u) (Section 1, Table 6, Table 7)
    and multi-scale spacing-conditioned skip connections.
    """

    def __init__(
        self,
        ctx_channels: int = 384,
        skip_channels: int = 384,
        expose_sphericity_channel: bool = True,
        decoder_channels: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.expose_sphericity_channel = expose_sphericity_channel
        in_top = ctx_channels + (1 if expose_sphericity_channel else 0)

        d1, d2, d3 = decoder_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_top, d1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.GELU(),
        )
        # Stage 1: fuse with Scale 2 (H/4)
        self.stage1 = DecoderRefinementBlock(d1, skip_channels, d2)
        # Stage 2: upsample to Scale 1 (H/2) and fuse
        self.stage2 = DecoderRefinementBlock(d2, skip_channels, d3)
        # Final full-resolution refinement (H x W)
        self.final_refine = nn.Sequential(
            nn.Conv2d(d3, d3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d3),
            nn.GELU(),
        )
        # Eq. (21): W_m D_z + b_m followed by sigmoid \sigma(.)
        self.pred_head = nn.Conv2d(d3, 1, kernel_size=1, bias=True)

    def forward(
        self,
        f_z_ctx: torch.Tensor,
        conditioned_scales: List[torch.Tensor],
        sphericity_map: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_z_ctx: (B, C, H/4, W/4) persistence-aware contextual features F_z^{CTX}
            conditioned_scales: list of 4 spacing-conditioned scale maps [F_1^{SC}, F_2^{SC}, F_3^{SC}, F_4^{SC}]
            sphericity_map: (B, 1, H/4, W/4) SCP sphericity map S_z
            output_size: (H, W) full spatial dimensions
        Returns:
            prob_map: (B, 1, H, W) pixel-wise segmentation probability map \hat{M}_z (Eq. 21)
            logits: (B, 1, H, W) raw pre-sigmoid logits W_m D_z + b_m
            d_z: (B, d3, H, W) decoded feature representation D_z (Eq. 20)
        """
        if self.expose_sphericity_channel:
            if sphericity_map.shape[-2:] != f_z_ctx.shape[-2:]:
                sphericity_map = F.interpolate(
                    sphericity_map, size=f_z_ctx.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([f_z_ctx, sphericity_map], dim=1)
        else:
            x = f_z_ctx

        x = self.bottleneck(x)
        x = self.stage1(x, conditioned_scales[1])  # H/4
        x = self.stage2(x, conditioned_scales[0])  # H/2

        x_full = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        d_z = self.final_refine(x_full)  # Eq. (20)
        logits = self.pred_head(d_z)
        prob_map = torch.sigmoid(logits)  # Eq. (21)
        return prob_map, logits, d_z
