"""
Section 3.1 & 3.5: Sphericity-Consistent Persistence (SCP) / Persistence-Aware Cross-Slice
Contextual Learning (Eqs. 1, 13–16).
"""

from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from micra.config import SCPAttentionConfig


class ScaleSpaceBlobRadiusEstimator(nn.Module):
    """
    Estimates local candidate blob radius R_z(u) in millimetres using multi-scale
    scale-normalized Laplacian-of-Gaussian (LoG) filtering [34] over physical scales
    \\Sigma = [0.5, 2.5] mm (Section 1, 3.1, 4.5), combined with a learned morphological
    descriptor \\Psi(F_{z+j}^{SP}) (Section 3.5, Eq. 14).
    """

    def __init__(self, feat_channels: int, config: SCPAttentionConfig):
        super().__init__()
        self.cfg = config
        self.register_buffer(
            "physical_scales_mm",
            torch.linspace(config.sigma_min_mm, config.sigma_max_mm, config.num_scales_log),
        )
        # Morphological descriptor extractor \Psi(.) in Eq. (14)
        self.psi = nn.Sequential(
            nn.Conv2d(feat_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
        )
        # Radius refinement head mapping morphological descriptor + scale-space LoG response -> radius in [0, R_max] mm
        self.radius_head = nn.Sequential(
            nn.Conv2d(16 + config.num_scales_log, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _multiscale_log_response(
        self,
        slice_img: torch.Tensor,
        inplane_spacing_mm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scale-normalized LoG response for dark (hypointense) blobs on T2* MRI
        across physical scales sigma in [0.5, 2.5] mm.
        Args:
            slice_img: (B, 1, h, w) normalized T2* slice
            inplane_spacing_mm: (B,) average in-plane voxel size (v_x + v_y)/2 in mm
        Returns:
            responses: (B, num_scales, h, w) positive blob responses
            est_radius_log_mm: (B, 1, h, w) scale-space radius estimate R = sqrt(2) * sigma^*
        """
        B, _, h, w = slice_img.shape
        device = slice_img.device
        responses = []

        # Invert dark T2* foci inside brain mask so hypointense blobs produce positive LoG peaks
        fg = (slice_img > 0.05).float()
        inv_img = (1.0 - slice_img) * fg

        mean_vxy = float(torch.clamp(inplane_spacing_mm.mean(), min=0.3, max=2.0).item())
        for sigma_mm in self.physical_scales_mm:
            sigma_px = max(0.6, float(sigma_mm.item()) / mean_vxy)
            ksize = int(2 * round(2.5 * sigma_px) + 1)
            ksize = max(3, min(ksize, 11))
            half = ksize // 2
            coords = torch.arange(-half, half + 1, device=device, dtype=slice_img.dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            r2 = xx ** 2 + yy ** 2
            s2 = sigma_px ** 2
            # Scale-normalized 2D LoG kernel: -sigma^2 * \nabla^2 G_sigma
            kernel = ((2.0 * s2 - r2) / (s2 ** 2)) * torch.exp(-r2 / (2.0 * s2))
            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.abs().sum() + 1e-6)
            kernel = kernel.view(1, 1, ksize, ksize)

            resp = F.conv2d(inv_img, kernel, padding=half)
            responses.append(F.relu(resp))

        resp_stack = torch.cat(responses, dim=1)  # (B, S, h, w)
        # Soft-argmax over physical scales -> R(u) = sqrt(2) * sigma^*(u)
        weights = F.softmax(resp_stack * 8.0, dim=1)
        scales_view = self.physical_scales_mm.view(1, -1, 1, 1)
        sigma_star = (weights * scales_view).sum(dim=1, keepdim=True)
        max_resp = resp_stack.max(dim=1, keepdim=True).values
        blob_gate = torch.sigmoid((max_resp - 0.03) * 25.0)
        est_radius_mm = math.sqrt(2.0) * sigma_star * blob_gate
        return resp_stack, est_radius_mm

    def forward(
        self,
        feat_sp: torch.Tensor,
        slice_img: torch.Tensor,
        inplane_spacing_mm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            psi_feat: (B, 16, h, w) morphological descriptor \Psi(F^{SP})
            radius_mm: (B, 1, h, w) estimated candidate radius in mm in [0, R_max]
        """
        _, _, h, w = feat_sp.shape
        if slice_img.shape[-2:] != (h, w):
            slice_low = F.interpolate(slice_img, size=(h, w), mode="bilinear", align_corners=False)
        else:
            slice_low = slice_img

        psi_feat = self.psi(feat_sp)
        log_stack, log_radius_mm = self._multiscale_log_response(slice_low, inplane_spacing_mm)
        refined_frac = self.radius_head(torch.cat([psi_feat, log_stack], dim=1))
        radius_mm = 0.5 * log_radius_mm + 0.5 * (refined_frac * self.cfg.r_max_mm)
        radius_mm = torch.clamp(radius_mm, min=0.0, max=self.cfg.r_max_mm)
        return psi_feat, radius_mm


class SphericityConsistentPersistenceAttention(nn.Module):
    """
    Implements Section 3.1 & 3.5 (Eqs. 1, 13–16):
      - Closed-form spherical cross-section radius prediction at slice offset d = |j| * v_z:
            r_{pred}(R, d) = \\sqrt{\\max(0, R^2 - d^2)}                   [Eq. 1]
      - Query, Key, Value projections across neighbouring slices j in {-k, ..., k}:
            [Q_z, K_{z+j}, V_{z+j}] = [F_z^{SP} W_Q, F_{z+j}^{SP} W_K, F_{z+j}^{SP} W_V]  [Eq. 13]
      - Persistence score combining physical spherical cross-section consistency and morphological descriptor:
            P_{z,j}(u) = \\exp\\left(-\\frac{\\Delta_{z,j}(u)}{\\tau_p}\\right)            [Eq. 14]
      - Persistence-biased cross-slice attention weights:
            \\alpha_{z,j}(u) = \\frac{\\exp(e_{z,j}(u) + \\lambda_p \\log(P_{z,j}(u) + \\epsilon))}
                                    {\\sum_{r=-k}^k \\exp(e_{z,r}(u) + \\lambda_p \\log(P_{z,r}(u) + \\epsilon))} [Eq. 15]
      - Contextual aggregation with residual connection and normalization:
            F_z^{CTX} = Norm\\left(F_z^{SP} + W_O \\sum_{j=-k}^k \\alpha_{z,j} \\odot V_{z+j}\\right) [Eq. 16]
      - Also exposes the explicit sphericity map S_z(u) to the decoder (Section 1, Table 6, Table 7).
    """

    def __init__(self, channels: int, config: SCPAttentionConfig):
        super().__init__()
        self.cfg = config
        self.channels = channels
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Eq. (13): W_Q, W_K, W_V projections
        self.w_q = nn.Conv2d(channels, self.inner_dim, kernel_size=1, bias=False)
        self.w_k = nn.Conv2d(channels, self.inner_dim, kernel_size=1, bias=False)
        self.w_v = nn.Conv2d(channels, self.inner_dim, kernel_size=1, bias=False)
        self.w_o = nn.Conv2d(self.inner_dim, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(16, channels)

        # Neighbor-slice alignment adapters when generating F_{z+j}^{SP} from adjacent slice context
        self.neighbor_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(16, channels),
                    nn.GELU(),
                )
                for _ in range(2)  # j = -1 and j = +1
            ]
        )

        self.blob_estimator = ScaleSpaceBlobRadiusEstimator(channels, config)

    @staticmethod
    def predicted_spherical_cross_section_radius(
        radius_center_mm: torch.Tensor,
        slice_distance_mm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements Eq. (1):
            r = \\sqrt{\\max(0, R^2 - d^2)}
        For a sphere of physical radius R (mm), a slice plane at distance d = |j| * v_z (mm)
        from its centre cuts a disc of radius r. If d >= R, the sphere does not intersect
        the neighboring slice (r = 0).
        """
        d_view = slice_distance_mm.view(-1, 1, 1, 1)
        r_sq = torch.clamp(radius_center_mm ** 2 - d_view ** 2, min=0.0)
        return torch.sqrt(r_sq + 1e-12) * (radius_center_mm > d_view).float()

    def compute_persistence_score(
        self,
        psi_center: torch.Tensor,
        psi_neighbor: torch.Tensor,
        r_center_mm: torch.Tensor,
        r_neighbor_mm: torch.Tensor,
        slice_offset_j: int,
        vz_mm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements Eq. (14):
            P_{z,j}(u) = \\exp(-\\Delta_{z,j}(u) / \\tau_p)
        Depending on self.cfg.mode (Table 7 mechanism analysis):
          - "scp_full" or "scp_bias_only":
              Physical mismatch between observed neighbor radius r_{z+j}(u) and predicted
              spherical cross-section r_{pred}(R_z(u), |j|*v_z), plus normalized morphological L1 difference.
              Crucially:
                * If a vessel persists across thick slices (v_z = 3.0 mm) with constant radius r_{z+j} > 0
                  while r_{pred} = 0 (since d = 3.0 mm > R), the mismatch is large -> P_{z,j} is low!
                * If a small CMB at v_z = 3.0 mm disappears in the next slice (r_{z+j} = 0) and
                  r_{pred} = 0, the mismatch is 0 -> P_{z,j} = 1 (fully consistent with a sphere!).
          - "learned_persistence":
              Eq. (14) using only ||\\Psi(F_z^{SP})(u) - \\Psi(F_{z+j}^{SP})(u)||_1 / \\tau_p.
        """
        if slice_offset_j == 0:
            return torch.ones_like(r_center_mm)

        morph_l1 = torch.mean(torch.abs(psi_center - psi_neighbor), dim=1, keepdim=True)
        if self.cfg.mode == "learned_persistence":
            return torch.exp(-morph_l1 / max(self.cfg.tau_p_mm, 1e-4))

        d_mm = abs(float(slice_offset_j)) * vz_mm
        r_pred_mm = self.predicted_spherical_cross_section_radius(r_center_mm, d_mm)
        spherical_mismatch_mm = torch.abs(r_neighbor_mm - r_pred_mm)

        # Combine physical spherical cross-section mismatch (in mm) with feature-level morphological distance
        total_mismatch = spherical_mismatch_mm + 0.25 * morph_l1
        p_zj = torch.exp(-total_mismatch / max(self.cfg.tau_p_mm, 1e-4))
        return torch.clamp(p_zj, min=0.0, max=1.0)

    def forward(
        self,
        f_z_sp: torch.Tensor,
        x_z: torch.Tensor,
        spacing_mm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_z_sp: (B, C, h, w) spacing-conditioned representation of central slice
            x_z: (B, 2k+1, H, W) input slice stack [I_{z-1}, I_z, I_{z+1}]
            spacing_mm: (B, 3) physical voxel spacing [v_x, v_y, v_z] in mm
        Returns:
            f_z_ctx: (B, C, h, w) persistence-aware contextual representation F_z^{CTX} (Eq. 16)
            sphericity_map: (B, 1, h, w) explicit sphericity score map S_z(u) in [0, 1]
            attn_weights: (B, 2k+1, h, w) cross-slice attention map averaged over heads
        """
        B, C, h, w = f_z_sp.shape
        num_slices = x_z.shape[1]
        k = num_slices // 2

        # 2D ablation mode (Table 6 row '- Cross-slice attention (2D)', Table 7 row '2D (no cross-slice)')
        if not self.cfg.enabled or self.cfg.mode == "2d" or k == 0:
            ones_sph = torch.ones((B, 1, h, w), device=f_z_sp.device, dtype=f_z_sp.dtype)
            uniform_attn = torch.ones((B, 1, h, w), device=f_z_sp.device, dtype=f_z_sp.dtype)
            return f_z_sp, ones_sph, uniform_attn

        x_low = F.interpolate(x_z, size=(h, w), mode="bilinear", align_corners=False)
        inplane_mm = 0.5 * (spacing_mm[:, 0] + spacing_mm[:, 1])
        vz_mm = spacing_mm[:, 2]

        # Build slice representations F_{z+j}^{SP} for j in {-1, 0, +1}
        f_prev = self.neighbor_generators[0](torch.cat([f_z_sp, x_low[:, 0:1]], dim=1))
        f_curr = f_z_sp
        f_next = self.neighbor_generators[1](torch.cat([f_z_sp, x_low[:, -1:]], dim=1))
        slice_features = [f_prev, f_curr, f_next]
        slice_Offsets = [-1, 0, 1]
        slice_imgs = [x_low[:, 0:1], x_low[:, k : k + 1], x_low[:, -1:]]

        # Estimate morphological descriptors \Psi and candidate radii R_{z+j}(u) in mm
        psi_list: List[torch.Tensor] = []
        rad_list: List[torch.Tensor] = []
        for f_s, img_s in zip(slice_features, slice_imgs):
            psi_s, rad_s = self.blob_estimator(f_s, img_s, inplane_mm)
            psi_list.append(psi_s)
            rad_list.append(rad_s)

        # Eq. (13): Q_z from central slice, K_{z+j} and V_{z+j} from slice z+j
        q_z = self.w_q(f_curr).view(B, self.num_heads, self.head_dim, h, w)

        logits_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []
        persistence_scores: List[torch.Tensor] = []

        eff_lambda_p = 0.0 if self.cfg.mode == "similarity_only" else self.cfg.lambda_p

        for idx, j_off in enumerate(slice_Offsets):
            f_j = slice_features[idx]
            k_j = self.w_k(f_j).view(B, self.num_heads, self.head_dim, h, w)
            v_j = self.w_v(f_j).view(B, self.num_heads, self.head_dim, h, w)
            values_list.append(v_j)

            # Scaled query-key similarity e_{z,j}(u)
            e_zj = (q_z * k_j).sum(dim=2) * self.scale  # (B, heads, h, w)

            # Eq. (14): Persistence score P_{z,j}(u)
            p_zj = self.compute_persistence_score(
                psi_center=psi_list[1],
                psi_neighbor=psi_list[idx],
                r_center_mm=rad_list[1],
                r_neighbor_mm=rad_list[idx],
                slice_offset_j=j_off,
                vz_mm=vz_mm,
            )  # (B, 1, h, w)
            persistence_scores.append(p_zj)

            # Eq. (15): e_{z,j}(u) + \lambda_p \log(P_{z,j}(u) + \epsilon)
            bias_zj = eff_lambda_p * torch.log(p_zj + self.cfg.eps)
            logits_list.append(e_zj + bias_zj)

        # Normalize attention weights across j in {-k, ..., k} (Eq. 15)
        logits_stack = torch.stack(logits_list, dim=2)  # (B, heads, 3, h, w)
        alpha_zj = F.softmax(logits_stack, dim=2)  # (B, heads, 3, h, w)

        values_stack = torch.stack(values_list, dim=2)  # (B, heads, 3, head_dim, h, w)
        agg = (alpha_zj.unsqueeze(3) * values_stack).sum(dim=2)  # (B, heads, head_dim, h, w)
        agg = agg.reshape(B, self.inner_dim, h, w)

        # Eq. (16): F_z^{CTX} = Norm(F_z^{SP} + W_O \sum_{j=-k}^k \alpha_{z,j} \odot V_{z+j})
        f_z_ctx = self.norm(f_curr + self.w_o(agg))

        # Explicit sphericity map S_z(u): average physical persistence across adjacent slices j in {-1, +1}
        sphericity_map = 0.5 * (persistence_scores[0] + persistence_scores[2])
        mean_attn = alpha_zj.mean(dim=1)  # (B, 3, h, w)

        return f_z_ctx, sphericity_map, mean_attn
