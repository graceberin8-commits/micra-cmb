"""
Section 3.6 & 4.5: Mimic-Aware Hard-Negative Prototype Learning (Eqs. 17–19).
Extracts Frangi vesselness V_i [35], Lindeberg scale-normalized LoG blobness B_i [34],
and SCP sphericity S_i to form the mimic descriptor d_i^{mimic} (Eq. 17), computes
hard-negative selection score h_i (Eq. 18), maintains the positive CMB prototype c^+
via EMA and negative mimic prototype memory bank {c_j^-}, and computes the prototype
contrastive objective L_{proto} (Eq. 19).
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from micra.config import MimicLearningConfig


classMimicDescriptorOut = Tuple[torch.Tensor, torch.Tensor]


class MimicAwareHardNegativeModule(nn.Module):
    """
    Implements Section 3.6 (Eqs. 17–19) with Section 4.5 hyperparameters:
      - beta_b = 0.4, N_c = 21, E_w = 10, eta_c = 0.1, delta_g = 2.0 mm,
        lambda_m = 1.0, mu = 0.5, K_h = 128, N_q = 8, K_n = 0.5, N_r = 0.5, tau_c = 0.5.
    """

    def __init__(self, in_channels: int, config: MimicLearningConfig):
        super().__init__()
        self.cfg = config
        self.feat_dim = config.feature_dim

        # Projection head mapping contextual features F_z^{CTX} -> normalized embedding f_i
        self.embedding_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.feat_dim),
            nn.GELU(),
            nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, bias=True),
        )

        # Positive CMB prototype c^+ (updated via EMA with momentum \eta_c = 0.1)
        self.register_buffer("pos_prototype", F.normalize(torch.randn(1, self.feat_dim), dim=-1))
        self.register_buffer("pos_initialized", torch.tensor(False, dtype=torch.bool))

        # Hard-negative mimic prototype memory bank {c_j^-}_{j=1}^{K_h} (K_h = 128)
        self.register_buffer(
            "neg_memory_bank",
            F.normalize(torch.randn(config.memory_bank_size, self.feat_dim), dim=-1),
        )
        self.register_buffer("bank_ptr", torch.tensor(0, dtype=torch.long))

        # Reference morphological descriptor d^{ref} for vessel/elongated mimic structures:
        # high vesselness V_i = 1.0, moderate focal blobness B_i = 0.5, low sphericity (1 - S_i = 1.0)
        self.register_buffer("d_ref", F.normalize(torch.tensor([[1.0, 0.5, 1.0]]), dim=-1))

    @torch.no_grad()
    def extract_vesselness_and_blobness(
        self,
        central_slice: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes 2D multi-scale Frangi Vesselness V(u) [35] and scale-normalized LoG Blobness B(u) [34]
        from the central T2* slice using Hessian eigenvalues (\lambda_1, \lambda_2) with |\lambda_1| <= |\lambda_2|.
        """
        # Invert dark T2* structures inside brain mask so dark vessels/blobs have negative curvature peaks
        fg = (central_slice > 0.05).float()
        inv = (1.0 - central_slice) * fg

        # Smooth with Gaussian kernel before second derivatives
        k_smooth = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=central_slice.device,
            dtype=central_slice.dtype,
        ).view(1, 1, 3, 3) / 16.0
        sm = F.conv2d(inv, k_smooth, padding=1)

        # Second-order finite difference Hessian kernels
        k_xx = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
            device=central_slice.device,
            dtype=central_slice.dtype,
        ).view(1, 1, 3, 3)
        k_yy = k_xx.transpose(-1, -2)
        k_xy = torch.tensor(
            [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]],
            device=central_slice.device,
            dtype=central_slice.dtype,
        ).view(1, 1, 3, 3)

        hxx = F.conv2d(sm, k_xx, padding=1)
        hyy = F.conv2d(sm, k_yy, padding=1)
        hxy = F.conv2d(sm, k_xy, padding=1)

        # Eigenvalues of 2x2 Hessian matrix
        trace = hxx + hyy
        diff = hxx - hyy
        disc = torch.sqrt(torch.clamp(diff ** 2 + 4.0 * (hxy ** 2), min=1e-12))
        eig1 = 0.5 * (trace + disc)
        eig2 = 0.5 * (trace - disc)

        # Sort so |lambda_1| <= |lambda_2|
        swap = torch.abs(eig1) > torch.abs(eig2)
        lam1 = torch.where(swap, eig2, eig1)
        lam2 = torch.where(swap, eig1, eig2)

        # Frangi Vesselness [35]: R_B = |lam1| / (|lam2| + eps), S = sqrt(lam1^2 + lam2^2)
        rb = torch.abs(lam1) / (torch.abs(lam2) + 1e-5)
        s_norm = torch.sqrt(lam1 ** 2 + lam2 ** 2 + 1e-12)
        beta_sq = 2.0 * (self.cfg.beta_b ** 2)
        c_sq = 2.0 * (0.15 ** 2)

        vesselness = torch.exp(-(rb ** 2) / max(beta_sq, 1e-4)) * (1.0 - torch.exp(-(s_norm ** 2) / c_sq))
        vesselness = vesselness * (lam2 < 0).float() * fg

        # Scale-normalized LoG Blobness: response is strong when both lam1 < 0 and lam2 < 0 and |lam1| ~ |lam2|
        lap = -(lam1 + lam2)
        isotropy = torch.exp(-((1.0 - rb) ** 2) / max(beta_sq, 1e-4))
        blobness = torch.sigmoid(lap * 10.0 - 1.5) * isotropy * ((lam1 < 0) & (lam2 < 0)).float() * fg

        return torch.clamp(vesselness, 0.0, 1.0), torch.clamp(blobness, 0.0, 1.0)

    @torch.no_grad()
    def _enqueue_hard_negatives(self, neg_feats: torch.Tensor) -> None:
        if neg_feats.numel() == 0:
            return
        n = neg_feats.shape[0]
        bank_size = self.cfg.memory_bank_size
        ptr = int(self.bank_ptr.item())
        if n >= bank_size:
            self.neg_memory_bank.copy_(neg_feats[:bank_size])
            self.bank_ptr.fill_(0)
        else:
            end = ptr + n
            if end <= bank_size:
                self.neg_memory_bank[ptr:end].copy_(neg_feats)
            else:
                first = bank_size - ptr
                self.neg_memory_bank[ptr:].copy_(neg_feats[:first])
                self.neg_memory_bank[: n - first].copy_(neg_feats[first:])
            self.bank_ptr.fill_(end % bank_size)

    def forward(
        self,
        f_z_ctx: torch.Tensor,
        x_z: torch.Tensor,
        gt_mask: torch.Tensor,
        sphericity_map: torch.Tensor,
        spacing_mm: torch.Tensor,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the Mimic-Aware Prototype Contrastive Loss L_{proto} (Eq. 19).
        Args:
            f_z_ctx: (B, C, h, w) contextual feature representation
            x_z: (B, 2k+1, H, W) input slice stack
            gt_mask: (B, 1, H, W) binary ground-truth mask M_z
            sphericity_map: (B, 1, h, w) SCP sphericity map S_z
            spacing_mm: (B, 3) voxel spacing [v_x, v_y, v_z] in mm
            current_epoch: current training epoch (for warmup E_w)
        Returns:
            l_proto: scalar tensor L_{proto} (Eq. 19)
            aux_dict: dictionary containing vesselness, blobness, and hard-negative scores
        """
        B, _, h, w = f_z_ctx.shape
        k = x_z.shape[1] // 2
        central_slice = F.interpolate(x_z[:, k : k + 1], size=(h, w), mode="bilinear", align_corners=False)
        mask_low = (F.interpolate(gt_mask.float(), size=(h, w), mode="nearest") > 0.5).float()

        # Project contextual features to normalized embeddings f_i in R^D
        emb_map = F.normalize(self.embedding_proj(f_z_ctx), dim=1)  # (B, D, h, w)

        # Eq. (17): Extract morphological mimic descriptor d_i^{mimic} = [V_i, B_i, 1 - S_i]
        vesselness, blobness = self.extract_vesselness_and_blobness(central_slice)
        non_sphericity = 1.0 - torch.clamp(sphericity_map, 0.0, 1.0)
        d_mimic_map = torch.cat([vesselness, blobness, non_sphericity], dim=1)  # (B, 3, h, w)
        d_mimic_norm = F.normalize(d_mimic_map + 1e-6, dim=1)

        # Flatten spatial locations
        emb_flat = emb_map.permute(0, 2, 3, 1).reshape(-1, self.feat_dim)  # (N, D)
        d_flat = d_mimic_norm.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)
        pos_mask_flat = (mask_low.reshape(-1) > 0.5)

        # Ground-truth exclusion zone of radius \delta_g = 2.0 mm around true CMBs
        mean_vxy = float(torch.clamp(spacing_mm[:, :2].mean(), min=0.4, max=2.0).item())
        excl_px = max(1, int(round(self.cfg.gt_exclusion_mm / mean_vxy)))
        dilated_gt = F.max_pool2d(mask_low, kernel_size=2 * excl_px + 1, stride=1, padding=excl_px)
        fg_brain = (central_slice.reshape(-1) > 0.05)
        neg_candidate_mask = (dilated_gt.reshape(-1) < 0.5) & fg_brain

        # Update positive prototype c^+ via EMA (\eta_c = 0.1) when positive CMB pixels are present
        pos_indices = torch.nonzero(pos_mask_flat, as_tuple=False).squeeze(-1)
        if pos_indices.numel() > 0:
            pos_feats = emb_flat[pos_indices]
            mean_pos = F.normalize(pos_feats.mean(dim=0, keepdim=True).detach(), dim=-1)
            if not bool(self.pos_initialized.item()):
                self.pos_prototype.copy_(mean_pos)
                self.pos_initialized.fill_(True)
            else:
                eta = self.cfg.prototype_momentum
                updated = F.normalize((1.0 - eta) * self.pos_prototype + eta * mean_pos, dim=-1)
                self.pos_prototype.copy_(updated)
        else:
            # Fallback to top blobness candidates if batch slice has no foreground mask
            top_b_idx = torch.topk(blobness.reshape(-1), k=min(4, emb_flat.shape[0])).indices
            pos_feats = emb_flat[top_b_idx]

        # Sample up to N_q = 8 query positive embeddings
        if pos_feats.shape[0] > self.cfg.num_positives_per_step:
            perm = torch.randperm(pos_feats.shape[0], device=pos_feats.device)[: self.cfg.num_positives_per_step]
            pos_queries = pos_feats[perm]
        else:
            pos_queries = pos_feats

        # Eq. (18): Hard-negative selection score h_i = sim(f_i, c^+) + \lambda_m sim(d_i^{mimic}, d^{ref})
        neg_indices = torch.nonzero(neg_candidate_mask, as_tuple=False).squeeze(-1)
        if neg_indices.numel() == 0:
            neg_indices = torch.nonzero(~pos_mask_flat, as_tuple=False).squeeze(-1)

        neg_feats_all = emb_flat[neg_indices]
        neg_d_all = d_flat[neg_indices]

        sim_feat_pos = torch.matmul(neg_feats_all, self.pos_prototype.t()).squeeze(-1)
        sim_morph_ref = torch.matmul(neg_d_all, self.d_ref.t()).squeeze(-1)
        h_scores = sim_feat_pos + self.cfg.lambda_m * sim_morph_ref  # Eq. (18)

        # Select top N_c = 21 hard-negative candidates
        k_mine = min(self.cfg.num_candidates, max(1, int(round(self.cfg.num_candidates * self.cfg.hard_negative_ratio * 2.0))), h_scores.shape[0])
        top_neg_idx = torch.topk(h_scores, k=k_mine).indices
        mined_neg_feats = neg_feats_all[top_neg_idx]

        if self.training:
            self._enqueue_hard_negatives(mined_neg_feats.detach())

        # Combine mined batch hard negatives with memory bank negative prototypes {c_j^-}
        all_neg_prototypes = torch.cat([mined_neg_feats, self.neg_memory_bank.detach()], dim=0)

        # Eq. (19): L_{proto} = -log( exp(sim(f_i, c^+)/tau_c) / (exp(sim(f_i, c^+)/tau_c) + \sum_{j \in H_i} exp(sim(f_i, c_j^-)/tau_c)) )
        tau_c = max(self.cfg.tau_c, 1e-4)
        pos_sim = torch.matmul(pos_queries, self.pos_prototype.t()) / tau_c  # (N_q, 1)
        neg_sim = torch.matmul(pos_queries, all_neg_prototypes.t()) / tau_c  # (N_q, K_neg)

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        targets = torch.zeros(pos_queries.shape[0], dtype=torch.long, device=pos_queries.device)
        l_proto = F.cross_entropy(logits, targets)

        # Apply smooth warmup over E_w = 10 epochs if configured
        if self.training and self.cfg.warmup_epochs > 0 and current_epoch < self.cfg.warmup_epochs:
            warmup_scale = float(current_epoch + 1) / float(self.cfg.warmup_epochs)
            l_proto = l_proto * warmup_scale

        aux = {
            "vesselness": vesselness,
            "blobness": blobness,
            "hard_negative_max_score": h_scores.max().detach() if h_scores.numel() > 0 else torch.tensor(0.0),
        }
        return l_proto, aux
