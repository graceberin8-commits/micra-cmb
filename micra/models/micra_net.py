"""
Full MICRA Architecture (Multi-scale Inter-slice Contextual Representation Architecture)
integrating Stages (i)–(v) and Mimic-Aware Hard-Negative Prototype Learning (Section 3.1–3.7).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from micra.config import MICRAConfig
from micra.models.lora_dinov3 import DINOv3BackboneWithLoRA
from micra.models.cnn_pathway import HighResCNNPathway, DualPathwayMultiScaleFusion
from micra.models.spacing_conditioning import SpacingConditionedMultiScaleModule
from micra.models.scp_attention import SphericityConsistentPersistenceAttention
from micra.models.mimic_learning import MimicAwareHardNegativeModule
from micra.models.decoder import MICRASegmentationDecoder


class MICRANet(nn.Module):
    """
    Complete MICRA Network (Fig. 1, Section 3.1–3.7):
      1. LoRA-adapted DINOv3 ViT-S/16 pathway (Eqs. 5–6)
      2. High-Resolution ResNet-18 CNN pathway & 4-scale feature fusion (Eqs. 7–8)
      3. Spacing-Conditioned Multi-Scale Lesion Representation (Eqs. 9–12)
      4. Sphericity-Consistent Persistence (SCP) Cross-Slice Attention (Eqs. 1, 13–16)
      5. Mimic-Aware Hard-Negative Prototype Learning (Eqs. 17–19, active during training)
      6. Multi-Scale Segmentation Decoder & Sigmoid Prediction Head (Eqs. 20–21)
    """

    def __init__(self, config: Optional[MICRAConfig] = None):
        super().__init__()
        self.cfg = config or MICRAConfig()
        in_slices = self.cfg.num_input_slices
        scale_chs = list(self.cfg.cnn.scale_channels)

        # Pathway 1: Pretrained DINOv3 ViT-S/16 + LoRA (r=8 on Q, V of last 6 blocks)
        self.dinov3_pathway = DINOv3BackboneWithLoRA(in_slices=in_slices, config=self.cfg.dinov3)

        # Pathway 2: Parallel High-Resolution ResNet-18 CNN pathway
        if self.cfg.cnn.enabled:
            self.cnn_pathway: Optional[HighResCNNPathway] = HighResCNNPathway(
                in_slices=in_slices, config=self.cfg.cnn
            )
        else:
            self.cnn_pathway = None

        # Multi-scale Dual-Pathway Fusion (Eq. 8)
        self.fusion = DualPathwayMultiScaleFusion(
            scale_channels=scale_chs, use_cnn=self.cfg.cnn.enabled
        )

        # Spacing-Conditioned Multi-Scale Lesion Module (Eqs. 9–12)
        self.spacing_module = SpacingConditionedMultiScaleModule(
            scale_channels=tuple(scale_chs), config=self.cfg.spacing
        )

        # Sphericity-Consistent Persistence (SCP) Cross-Slice Attention (Eqs. 1, 13–16)
        unified_ch = self.cfg.spacing.unified_channels
        self.scp_attention = SphericityConsistentPersistenceAttention(
            channels=unified_ch, config=self.cfg.scp
        )

        # Mimic-Aware Hard-Negative Prototype Learning Module (Eqs. 17–19)
        self.mimic_module = MimicAwareHardNegativeModule(
            in_channels=unified_ch, config=self.cfg.mimic
        )

        # Segmentation Decoder (Eqs. 20–21)
        expose_sz = (
            self.cfg.scp.expose_sphericity_channel
            and self.cfg.scp.enabled
            and self.cfg.scp.mode == "scp_full"
        )
        self.decoder = MICRASegmentationDecoder(
            ctx_channels=unified_ch,
            skip_channels=unified_ch,
            expose_sphericity_channel=expose_sz,
            decoder_channels=(256, 128, 64),
        )

    def get_optimizer_param_groups(self) -> List[Dict]:
        """
        Section 4.5: AdamW with learning rates 1e-5 (LoRA parameters) and 1e-4 (new layers),
        weight decay 1e-4.
        """
        lora_params = []
        new_layer_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            else:
                new_layer_params.append(param)

        return [
            {"params": lora_params, "lr": self.cfg.training.lr_lora, "name": "lora"},
            {"params": new_layer_params, "lr": self.cfg.training.lr_new_layers, "name": "new_layers"},
        ]

    def count_parameters(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "total_M": total / 1e6,
            "trainable_M": trainable / 1e6,
        }

    def forward(
        self,
        x_z: torch.Tensor,
        spacing_mm: torch.Tensor,
        cohort_ids: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_z: (B, 2k+1, H, W) 2.5D slice stack input
            spacing_mm: (B, 3) physical voxel spacing [v_x, v_y, v_z] in mm
            cohort_ids: Optional (B,) cohort indices {0, 1, 2}
            gt_mask: Optional (B, 1, H, W) ground-truth mask of central slice (for training L_proto)
            current_epoch: current epoch index
            return_intermediates: whether to include feature response maps for Fig. 6 visualization
        Returns:
            Dictionary with keys:
              - "prob": (B, 1, H, W) predicted segmentation probability map \hat{M}_z (Eq. 21)
              - "logits": (B, 1, H, W) pre-sigmoid logits
              - "l_proto": scalar prototype contrastive loss L_{proto} (Eq. 19)
              - "sphericity_map": (B, 1, H/4, W/4) SCP sphericity score map S_z
              - "scale_weights": (4,) learned multi-scale weights \alpha_l (Eq. 12)
              - (optional intermediates for Fig. 6)
        """
        _, _, H, W = x_z.shape

        # Stage (ii): Dual-pathway feature extraction (Eqs. 5–8)
        dino_scales, x_z_d = self.dinov3_pathway(x_z)
        cnn_scales = self.cnn_pathway(x_z) if self.cnn_pathway is not None else []
        fused_scales = self.fusion(dino_scales, cnn_scales)

        # Stage (iii): Spacing-conditioned multi-scale lesion representation (Eqs. 9–12)
        f_z_sp, conditioned_scales, alpha_weights = self.spacing_module(
            fused_scales, spacing_mm=spacing_mm, cohort_ids=cohort_ids
        )

        # Stage (iv): Sphericity-Consistent Persistence cross-slice attention (Eqs. 1, 13–16)
        f_z_ctx, sphericity_map, attn_weights = self.scp_attention(
            f_z_sp=f_z_sp, x_z=x_z, spacing_mm=spacing_mm
        )

        # Stage (v): Segmentation decoding (Eqs. 20–21)
        prob_map, logits, d_z = self.decoder(
            f_z_ctx=f_z_ctx,
            conditioned_scales=conditioned_scales,
            sphericity_map=sphericity_map,
            output_size=(H, W),
        )

        # Training-only Mimic-Aware Hard-Negative Prototype Learning (Eqs. 17–19)
        if gt_mask is not None and self.cfg.mimic.enabled and self.cfg.training.lambda_proto > 0:
            l_proto, mimic_aux = self.mimic_module(
                f_z_ctx=f_z_ctx,
                x_z=x_z,
                gt_mask=gt_mask,
                sphericity_map=sphericity_map,
                spacing_mm=spacing_mm,
                current_epoch=current_epoch,
            )
        else:
            l_proto = torch.tensor(0.0, device=x_z.device, dtype=x_z.dtype)
            mimic_aux = {}

        outputs: Dict[str, torch.Tensor] = {
            "prob": prob_map,
            "logits": logits,
            "l_proto": l_proto,
            "sphericity_map": sphericity_map,
            "scale_weights": alpha_weights,
            "attn_weights": attn_weights,
        }

        if return_intermediates:
            # Multi-scale response proxy (Fig. 6e) and Spatial context response proxy (Fig. 6f)
            ms_resp = torch.norm(f_z_sp, p=2, dim=1, keepdim=True)
            ctx_resp = torch.norm(f_z_ctx, p=2, dim=1, keepdim=True) * sphericity_map
            outputs["multiscale_response"] = ms_resp
            outputs["spatial_context_response"] = ctx_resp
            outputs["decoded_features"] = d_z
            outputs["projected_input"] = x_z_d
            outputs.update(mimic_aux)

        return outputs
