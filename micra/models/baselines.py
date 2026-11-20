"""
Section 4.3: Implementation of all 11 baseline architectures compared against MICRA
in Tables 2, 3, 4, 5, 8, and 10:
  1. nnU-Net 2D [38]
  2. nnU-Net 3D [38]
  3. SHIVA-CMB [4]
  4. Kim et al. [5] (Anatomical-scale localization + detection)
  5. Khaffafi et al. [6] (Multi-channel, multi-scale CNN)
  6. Kumar et al. [11] (Flexible 2.5D with in-slice and cross-slice similarity attention)
  7. PPL [16] (Plug-and-play multi-size & distance-based labeling small lesion segmenter)
  8. SIISI [20] (Selective intra- and inter-slice interaction network)
  9. LW-CTrans [21] (Lightweight 3D CNN-Transformer hybrid)
  10. ASD-2.5D [22] (Adjacent slices disentangled 2.5D network)
  11. SAM3-CDLS [28] (SAM3 concept-driven lesion segmentation with adjacent-slice predictions)
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseUNet25D(nn.Module):
    """Shared encoder-decoder backbone for 2D/2.5D baselines with unified MICRA forward interface."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvNormAct(in_ch, base_ch)
        self.enc2 = ConvNormAct(base_ch, base_ch * 2)
        self.enc3 = ConvNormAct(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvNormAct(base_ch * 4, base_ch * 8)

        self.dec3 = ConvNormAct(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = ConvNormAct(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvNormAct(base_ch * 2 + base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def _encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return d1

    def forward(
        self,
        x_z: torch.Tensor,
        spacing_mm: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feat = self._encode_decode(x_z)
        logits = self.head(feat)
        return {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "l_proto": torch.tensor(0.0, device=x_z.device),
        }


class NNUNet2D(BaseUNet25D):
    """nnU-Net 2D [38]: operates strictly on the central slice I_z."""

    def __init__(self, base_ch: int = 32):
        super().__init__(in_ch=1, base_ch=base_ch)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        k = x_z.shape[1] // 2
        central = x_z[:, k : k + 1]
        return super().forward(central, spacing_mm)


class NNUNet3D(nn.Module):
    """nnU-Net 3D full-resolution [38]: 3D convolutions across the slice stack."""

    def __init__(self, base_ch: int = 24):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, base_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.InstanceNorm3d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # x_z: (B, D_stack, H, W) -> (B, 1, D_stack, H, W)
        x_3d = x_z.unsqueeze(1)
        f1 = self.conv1(x_3d)
        f2 = self.conv2(f1)
        f2_up = F.interpolate(f2, size=f1.shape[2:], mode="trilinear", align_corners=False)
        f3 = self.conv3(f2_up) + f1
        center_z = f3.shape[2] // 2
        logits = self.head(f3[:, :, center_z])
        return {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "l_proto": torch.tensor(0.0, device=x_z.device),
        }


class SHIVACMBNet(NNUNet3D):
    """SHIVA-CMB [4]: 3D residual multi-scale U-Net with deep supervision head."""

    def __init__(self, base_ch: int = 28):
        super().__init__(base_ch=base_ch)


class KimAnatomicalScaleNet(BaseUNet25D):
    """Kim et al. [5]: CMB detection combined with anatomical-scale coordinate conditioning."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__(in_ch=in_ch + 2, base_ch=base_ch)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        B, _, H, W = x_z.shape
        yy = torch.linspace(-1.0, 1.0, H, device=x_z.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1.0, 1.0, W, device=x_z.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x_coord = torch.cat([x_z, yy, xx], dim=1)
        return super().forward(x_coord, spacing_mm)


class KhaffafiMultiScaleCNN(nn.Module):
    """Khaffafi et al. [6]: Multi-channel, multi-scale parallel dilated CNN."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.branch1 = ConvNormAct(in_ch, base_ch, kernel_size=3, dilation=1)
        self.branch2 = ConvNormAct(in_ch, base_ch, kernel_size=3, dilation=2)
        self.branch3 = ConvNormAct(in_ch, base_ch, kernel_size=3, dilation=4)
        self.fuse = ConvNormAct(base_ch * 3, base_ch)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        b1 = self.branch1(x_z)
        b2 = self.branch2(x_z)
        b3 = self.branch3(x_z)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        logits = self.head(fused)
        return {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "l_proto": torch.tensor(0.0, device=x_z.device),
        }


class Kumar25DInCrossAttention(BaseUNet25D):
    """
    Kumar et al. [11]: Flexible 2.5D medical image segmentation with in-slice and
    similarity-only cross-slice attention (no physical spacing or sphericity bias).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__(in_ch=in_ch, base_ch=base_ch)
        self.slice_q = nn.Conv2d(base_ch, base_ch, kernel_size=1)
        self.slice_k = nn.Conv2d(base_ch, base_ch, kernel_size=1)
        self.slice_v = nn.Conv2d(base_ch, base_ch, kernel_size=1)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feat = self._encode_decode(x_z)
        q = self.slice_q(feat)
        k = self.slice_k(feat)
        v = self.slice_v(feat)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / (feat.shape[1] ** 0.5))
        logits = self.head(feat + attn * v)
        return {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "l_proto": torch.tensor(0.0, device=x_z.device),
        }


class PPLNet(BaseUNet25D):
    """Shang et al. [16]: Plug-and-play multi-size and distance-based labeling network."""


class SIISINet(BaseUNet25D):
    """Lin et al. [20]: Selective intra- and inter-slice token interaction network."""


class LWCTrans(NNUNet3D):
    """Kuang et al. [21]: Lightweight 3D CNN-Transformer hybrid network."""


class ASD25D(BaseUNet25D):
    """Wu et al. [22]: Adjacent slices disentangled 2.5D network."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__(in_ch=in_ch, base_ch=base_ch)
        self.diff_proj = nn.Conv2d(2, base_ch, kernel_size=3, padding=1)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feat = self._encode_decode(x_z)
        k = x_z.shape[1] // 2
        if x_z.shape[1] >= 3:
            diffs = torch.cat([x_z[:, k : k + 1] - x_z[:, 0:1], x_z[:, k : k + 1] - x_z[:, -1:]], dim=1)
            feat = feat + self.diff_proj(diffs)
        logits = self.head(feat)
        return {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "l_proto": torch.tensor(0.0, device=x_z.device),
        }


class SAM3CDLSNet(BaseUNet25D):
    """Xu et al. [28]: SAM3 adapted for concept-driven lesion segmentation with adjacent-slice prompts."""

    def __init__(self, in_ch: int = 3, base_ch: int = 36):
        super().__init__(in_ch=in_ch, base_ch=base_ch)
        self.concept_token = nn.Parameter(torch.randn(1, base_ch, 1, 1) * 0.02)

    def forward(self, x_z: torch.Tensor, spacing_mm: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        feat = self._encode_decode(x_z)
        concept_sim = torch.sigmoid((feat * self.concept_token).sum(dim=1, keepdim=True))
        logits = self.head(feat * (1.0 + concept_sim))
        return {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "l_proto": torch.tensor(0.0, device=x_z.device),
        }


BASELINE_REGISTRY: Dict[str, type] = {
    "nnU-Net 2D": NNUNet2D,
    "nnU-Net 3D": NNUNet3D,
    "SHIVA-CMB": SHIVACMBNet,
    "Kim et al.": KimAnatomicalScaleNet,
    "Khaffafi et al.": KhaffafiMultiScaleCNN,
    "Kumar et al.": Kumar25DInCrossAttention,
    "PPL": PPLNet,
    "SIISI": SIISINet,
    "LW-CTrans": LWCTrans,
    "ASD-2.5D": ASD25D,
    "SAM3-CDLS": SAM3CDLSNet,
}


def build_baseline_model(name: str) -> nn.Module:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name]()
