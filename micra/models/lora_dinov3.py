"""
Section 3.3 & 4.5: Parameter-Efficient DINOv3 Vision Transformer Pathway with Low-Rank Adaptation (Eqs. 5–6).
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from micra.config import DINOv3Config


class LoRALinear(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) in Eq. (6):
        W' = W_0 + \\Delta W = W_0 + B A
    where W_0 in R^{d_o x d_i} is frozen, A in R^{r x d_i}, and B in R^{d_o x r}
    are low-rank trainable matrices with rank r.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        enable_lora: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.enable_lora = enable_lora and (rank > 0)
        self.scaling = alpha / float(max(1, rank))

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.trunc_normal_(self.weight, std=0.02)

        if self.enable_lora:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.enable_lora and self.lora_A is not None and self.lora_B is not None:
            delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + delta
        return out


class DINOv3AttentionBlock(nn.Module):
    """
    DINOv3 ViT-S/16 Transformer Block (384-d, 6 heads) with LayerScale and optional LoRA
    on the query (W_Q) and value (W_V) projections (Section 4.5).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = LoRALinear(
            embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, enable_lora=use_lora
        )
        self.k_proj = LoRALinear(
            embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, enable_lora=False
        )
        self.v_proj = LoRALinear(
            embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, enable_lora=use_lora
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.gamma1 = nn.Parameter(1e-4 * torch.ones(embed_dim))

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gamma2 = nn.Parameter(1e-4 * torch.ones(embed_dim))

    def freeze_pretrained(self) -> None:
        self.q_proj.freeze_base_weights()
        self.k_proj.freeze_base_weights()
        self.v_proj.freeze_base_weights()
        for module in [self.norm1, self.out_proj, self.norm2, self.mlp]:
            for p in module.parameters():
                p.requires_grad_(False)
        self.gamma1.requires_grad_(False)
        self.gamma2.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        x = x + self.gamma1 * out
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class DINOv3BackboneWithLoRA(nn.Module):
    """
    Implements Section 3.3 & 4.5:
      1. Learnable 3-channel input projection Proj_3(X_z) in Eq. (5):
             X_z^D = Proj_3(X_z)
      2. Pretrained DINOv3 ViT-S/16 backbone (384-d, 12 blocks, 6 heads, patch size 16),
         frozen base weights, with LoRA (r=8) on Query and Value projections of the last 6 blocks (Eq. 6).
      3. Multi-scale intermediate feature map extraction projected to 4 spatial pyramid levels
         (1/2, 1/4, 1/8, 1/16 of input resolution).
    """

    def __init__(self, in_slices: int, config: DINOv3Config):
        super().__init__()
        self.cfg = config
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim

        # Eq. (5): X_z^D = Proj_3(X_z)
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_slices, 3, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(3, affine=True),
            nn.GELU(),
        )

        # Patch embedding (16x16 patch tokens)
        self.patch_embed = nn.Conv2d(
            3, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 256, config.embed_dim) * 0.02)

        # 12 Transformer blocks; LoRA applied to Q and V of the last 6 blocks
        first_lora_block = max(0, config.depth - config.lora_last_n_blocks)
        self.blocks = nn.ModuleList(
            [
                DINOv3AttentionBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    use_lora=(idx >= first_lora_block),
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                )
                for idx in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        # Multi-scale spatial re-projection heads for the 4 extracted transformer depths
        # Target scales: Scale 1 (1/2), Scale 2 (1/4), Scale 3 (1/8), Scale 4 (1/16)
        self.scale_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(config.embed_dim, 64, kernel_size=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(config.embed_dim, 128, kernel_size=1),
                    nn.GroupNorm(16, 128),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(config.embed_dim, 256, kernel_size=1),
                    nn.GroupNorm(16, 256),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(config.embed_dim, 512, kernel_size=1),
                    nn.GroupNorm(32, 512),
                    nn.GELU(),
                ),
            ]
        )

        if config.freeze_backbone:
            self._freeze_base_backbone()

    def _freeze_base_backbone(self) -> None:
        for p in self.patch_embed.parameters():
            p.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        for blk in self.blocks:
            blk.freeze_pretrained()
        for p in self.norm.parameters():
            p.requires_grad_(False)

    def _interpolate_pos_embed(self, num_patches: int, h_p: int, w_p: int) -> torch.Tensor:
        if num_patches == self.pos_embed.shape[1]:
            return self.pos_embed
        side = int(math.sqrt(self.pos_embed.shape[1]))
        pe = self.pos_embed.reshape(1, side, side, self.embed_dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h_p, w_p), mode="bilinear", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, num_patches, self.embed_dim)

    def forward(self, x_z: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x_z: (B, 2k+1, H, W) neighbouring-slice input
        Returns:
            dino_scales: list of 4 spatial feature maps at resolutions (H/2, H/4, H/8, H/16)
                         with channels [64, 128, 256, 512]
            x_z_d: (B, 3, H, W) projected 3-channel representation (Eq. 5)
        """
        B, _, H, W = x_z.shape
        x_z_d = self.proj3(x_z)  # Eq. (5)

        # Ensure dimensions are padded to multiples of patch_size (16) if needed
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x_z_d, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_padded = x_z_d

        tokens_2d = self.patch_embed(x_padded)  # (B, D, H_p, W_p)
        _, _, h_p, w_p = tokens_2d.shape
        tokens = tokens_2d.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self._interpolate_pos_embed(tokens.shape[1], h_p, w_p)

        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, tokens], dim=1)

        extracted: List[torch.Tensor] = []
        extract_set = set(self.cfg.extract_layers)
        for idx, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if idx in extract_set:
                patch_tok = self.norm(x[:, 1:, :])
                fmap = patch_tok.transpose(1, 2).reshape(B, self.embed_dim, h_p, w_p)
                extracted.append(fmap)

        target_sizes = [
            (max(1, H // 2), max(1, W // 2)),
            (max(1, H // 4), max(1, W // 4)),
            (max(1, H // 8), max(1, W // 8)),
            (max(1, H // 16), max(1, W // 16)),
        ]
        dino_scales: List[torch.Tensor] = []
        for fmap, adapter, t_size in zip(extracted, self.scale_adapters, target_sizes):
            proj = adapter(fmap)
            up = F.interpolate(proj, size=t_size, mode="bilinear", align_corners=False)
            dino_scales.append(up)

        return dino_scales, x_z_d
