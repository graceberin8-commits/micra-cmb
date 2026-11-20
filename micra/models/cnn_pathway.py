"""
Section 3.3 & 4.5: High-Resolution CNN Feature Extraction Pathway (ResNet-18) and
Dual-Pathway Multi-Scale Feature Integration (Eqs. 7–8).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from micra.config import CNNPathwayConfig


class BasicResBlock(nn.Module):
    """Standard ResNet-18 basic residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class HighResCNNPathway(nn.Module):
    """
    Implements Eq. (7):
        F_z^{CNN} = \\Phi_{CNN}(X_z; \\theta_{CNN})
    A ResNet-18 architecture without aggressive initial max-pooling so fine-grained small lesions
    (< 16-pixel ViT token size) are preserved across four spatial scales:
        Scale 1: H/2 x W/2 (64 channels)
        Scale 2: H/4 x W/4 (128 channels)
        Scale 3: H/8 x W/8 (256 channels)
        Scale 4: H/16 x W/16 (512 channels)
    """

    def __init__(self, in_slices: int, config: CNNPathwayConfig):
        super().__init__()
        self.cfg = config
        c1, c2, c3, c4 = config.scale_channels

        # High-resolution stem (stride 2 -> H/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_slices, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        # 4 ResNet-18 stages (2 residual blocks per stage)
        self.layer1 = nn.Sequential(BasicResBlock(c1, c1, stride=1), BasicResBlock(c1, c1, stride=1))
        self.layer2 = nn.Sequential(BasicResBlock(c1, c2, stride=2), BasicResBlock(c2, c2, stride=1))
        self.layer3 = nn.Sequential(BasicResBlock(c2, c3, stride=2), BasicResBlock(c3, c3, stride=1))
        self.layer4 = nn.Sequential(BasicResBlock(c3, c4, stride=2), BasicResBlock(c4, c4, stride=1))

    def forward(self, x_z: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x_z)
        f1 = self.layer1(x)   # (B, 64, H/2, W/2)
        f2 = self.layer2(f1)  # (B, 128, H/4, W/4)
        f3 = self.layer3(f2)  # (B, 256, H/8, W/8)
        f4 = self.layer4(f3)  # (B, 512, H/16, W/16)
        return [f1, f2, f3, f4]


class DualPathwayMultiScaleFusion(nn.Module):
    """
    Implements Eq. (8) at each of the L = 4 spatial scales:
        F_z^{COMB} = Fuse(F_z^{DINO}, F_z^{CNN})
    Supports ablating the High-Resolution CNN pathway (Table 6 row '- High-resolution CNN pathway').
    """

    def __init__(self, scale_channels: List[int], use_cnn: bool = True):
        super().__init__()
        self.use_cnn = use_cnn
        self.fusers = nn.ModuleList()
        for ch in scale_channels:
            in_ch = ch * 2 if use_cnn else ch
            self.fusers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                )
            )

    def forward(
        self,
        dino_scales: List[torch.Tensor],
        cnn_scales: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        fused_scales: List[torch.Tensor] = []
        for idx, fuser in enumerate(self.fusers):
            fd = dino_scales[idx]
            if self.use_cnn and len(cnn_scales) > idx:
                fc = cnn_scales[idx]
                if fd.shape[-2:] != fc.shape[-2:]:
                    fd = F.interpolate(fd, size=fc.shape[-2:], mode="bilinear", align_corners=False)
                cat = torch.cat([fd, fc], dim=1)
                fused_scales.append(fuser(cat))
            else:
                fused_scales.append(fuser(fd))
        return fused_scales
