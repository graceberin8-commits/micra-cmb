"""
Section 3.4: Spacing-Conditioned Multi-Scale Lesion Representation (Eqs. 9–12).
Also implements the Table 6 / Table 7 mechanism variants:
  - "spacing": FiLM on continuous log-normalized physical voxel spacing (Eqs. 9–11)
  - "cohort_id": FiLM on one-hot cohort identifier (Table 7 H2 control)
  - "none": No spacing conditioning (Table 6 / Table 7 baseline)
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from micra.config import SpacingConditioningConfig


class SpacingConditionedMultiScaleModule(nn.Module):
    """
    Implements Section 3.4 (Eqs. 9–12):
      - Eq. (9):  v^* = log(v / v_{ref})
      - Eq. (10): e_v = MLP(v^*; \\theta_v)
      - Eq. (11): F_l^{SC} = \\gamma_l(e_v) \\odot P_l(F_l) + \\beta_l(e_v)
      - Eq. (12): F_z^{SP} = \\sum_{l=1}^L \\alpha_l F_l^{SC},  \\sum_{l=1}^L \\alpha_l = 1
    """

    def __init__(
        self,
        scale_channels: Tuple[int, ...],
        config: SpacingConditioningConfig,
    ):
        super().__init__()
        self.cfg = config
        self.num_scales = len(scale_channels)
        self.unified_ch = config.unified_channels

        self.register_buffer(
            "v_ref",
            torch.tensor(config.reference_spacing_mm, dtype=torch.float32).view(1, 3),
        )

        # Eq. (10): e_v = MLP(v^*; \theta_v) (or cohort one-hot embedding for Table 7 H2 test)
        in_cond_dim = config.num_cohorts if config.mode == "cohort_id" else 3
        self.spacing_mlp = nn.Sequential(
            nn.Linear(in_cond_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
        )

        # Feature projections P_l(.) and FiLM modulation heads \gamma_l(e_v), \beta_l(e_v)
        self.projections = nn.ModuleList()
        self.film_heads = nn.ModuleList()
        for ch in scale_channels:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(ch, self.unified_ch, kernel_size=1, bias=False),
                    nn.GroupNorm(16, self.unified_ch),
                    nn.GELU(),
                )
            )
            self.film_heads.append(nn.Linear(config.embed_dim, 2 * self.unified_ch))

        # Eq. (12): Learnable scale contribution logits -> normalized via softmax so sum_{l=1}^L alpha_l = 1
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))

    def compute_normalized_spacing(self, spacing_mm: torch.Tensor) -> torch.Tensor:
        """
        Implements Eq. (9):
            v^* = log(v / v_{ref})
        """
        v_safe = torch.clamp(spacing_mm, min=1e-3)
        return torch.log(v_safe / self.v_ref)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        spacing_mm: torch.Tensor,
        cohort_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            multi_scale_features: List of L=4 feature tensors [F_1, F_2, F_3, F_4]
            spacing_mm: (B, 3) tensor of physical voxel spacings [v_x, v_y, v_z] in mm
            cohort_ids: Optional (B,) tensor of integer cohort IDs {0, 1, 2}
        Returns:
            f_z_sp: (B, C_u, H_2, W_2) aggregated spacing-conditioned representation F_z^{SP} (Eq. 12)
            conditioned_scales: List of L modulated feature maps F_l^{SC} (Eq. 11)
            alpha_weights: (L,) normalized scale weights \alpha_l
        """
        B = spacing_mm.shape[0]
        target_hw = multi_scale_features[1].shape[-2:]  # Align to H/4 x W/4 for cross-slice attention

        if self.cfg.mode == "spacing":
            v_star = self.compute_normalized_spacing(spacing_mm)  # Eq. (9)
            e_v = self.spacing_mlp(v_star)  # Eq. (10)
        elif self.cfg.mode == "cohort_id":
            if cohort_ids is None:
                cohort_ids = torch.zeros(B, dtype=torch.long, device=spacing_mm.device)
            # Note: under LOCO, unseen cohort ID can be out-of-distribution
            one_hot = F.one_hot(
                torch.clamp(cohort_ids, 0, self.cfg.num_cohorts - 1),
                num_classes=self.cfg.num_cohorts,
            ).float()
            e_v = self.spacing_mlp(one_hot)
        else:
            e_v = None

        conditioned_scales: List[torch.Tensor] = []
        aligned_scales: List[torch.Tensor] = []

        for l_idx, f_l in enumerate(multi_scale_features):
            p_l = self.projections[l_idx](f_l)
            if e_v is not None and self.cfg.mode != "none":
                film_params = self.film_heads[l_idx](e_v)  # (B, 2 * C_u)
                gamma_raw, beta = torch.chunk(film_params, 2, dim=-1)
                # Center gamma around 1.0 for stable residual FiLM conditioning
                gamma = (1.0 + 0.5 * torch.tanh(gamma_raw)).unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                f_l_sc = gamma * p_l + beta  # Eq. (11)
            else:
                f_l_sc = p_l

            conditioned_scales.append(f_l_sc)
            if f_l_sc.shape[-2:] != target_hw:
                f_aligned = F.interpolate(f_l_sc, size=target_hw, mode="bilinear", align_corners=False)
            else:
                f_aligned = f_l_sc
            aligned_scales.append(f_aligned)

        # Eq. (12): F_z^{SP} = \sum_{l=1}^L \alpha_l F_l^{SC} with \sum \alpha_l = 1
        alpha_weights = F.softmax(self.scale_logits, dim=0)
        f_z_sp = torch.zeros_like(aligned_scales[0])
        for l_idx, f_aligned in enumerate(aligned_scales):
            f_z_sp = f_z_sp + alpha_weights[l_idx] * f_aligned

        return f_z_sp, conditioned_scales, alpha_weights
