"""
Section 3.2 & 3.4: Acquisition-Aware Augmentation and Continuous Resolution Perturbation (Eq. 4).
"""

from typing import Tuple
import numpy as np
from scipy import ndimage
from micra.config import AugmentationConfig


class AcquisitionAwareAugmentation:
    """
    Implements Eq. (4):
        \\tilde{X}_z = T_a(X_z; a)
    Includes:
    1. Spatial transformations applied consistently to X_z and M_z.
    2. Intensity transformations (contrast/gamma variation, Gaussian/Rician noise) applied only to X_z.
    3. Continuous Resolution Perturbation (Section 1, 3.4, 4.5, Table 6 & 7):
       Perturbs effective in-plane and through-plane resolution with scale factor s in [0.5, 2.0],
       slice thickness threshold v_z^thr = 2.0 mm, and maximum perturbation mix m_max = 0.7,
       updating the physical voxel spacing vector v accordingly so spacing cannot act as a proxy for site.
    """

    def __init__(self, config: AugmentationConfig):
        self.cfg = config

    def __call__(
        self,
        slice_stack: np.ndarray,
        mask_2d: np.ndarray,
        spacing: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            slice_stack: (C, H, W) where C = 2k + 1
            mask_2d: (H, W) binary mask of central slice
            spacing: (3,) array [v_x, v_y, v_z] in mm
            rng: numpy random generator
        Returns:
            aug_stack: (C, H, W)
            aug_mask: (H, W)
            aug_spacing: (3,) perturbed voxel spacing [v_x', v_y', v_z'] in mm
        """
        if not self.cfg.enabled:
            return slice_stack, mask_2d, spacing

        x = slice_stack.copy()
        m = mask_2d.copy()
        v = spacing.copy().astype(np.float32)

        # 1. Spatial transformations (applied consistently to X_z and M_z)
        if rng.random() < self.cfg.flip_prob:
            x = np.flip(x, axis=2).copy()
            m = np.flip(m, axis=1).copy()
        if rng.random() < self.cfg.flip_prob:
            x = np.flip(x, axis=1).copy()
            m = np.flip(m, axis=0).copy()

        if rng.random() < 0.5 and self.cfg.rotate_deg > 0:
            angle = float(rng.uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg))
            for c in range(x.shape[0]):
                x[c] = ndimage.rotate(x[c], angle, reshape=False, order=1, mode="nearest")
            m = (ndimage.rotate(m, angle, reshape=False, order=0, mode="nearest") > 0.5).astype(np.float32)

        # 2. Continuous Resolution Perturbation (Section 3.4 & 4.5)
        if self.cfg.resolution_perturbation and rng.random() < self.cfg.max_perturbation_prob:
            s_min, s_max = self.cfg.scale_range
            s_inplane = float(rng.uniform(s_min, s_max))

            # Update reported physical in-plane spacing continuously
            v[0] = np.clip(v[0] * s_inplane, 0.35, 1.5)
            v[1] = np.clip(v[1] * s_inplane, 0.35, 1.5)

            # When s_inplane > 1.0, simulate coarser in-plane acquisition via Gaussian MTF smoothing
            if s_inplane > 1.05:
                sigma_px = 0.45 * (s_inplane - 1.0)
                for c in range(x.shape[0]):
                    x[c] = ndimage.gaussian_filter(x[c], sigma=sigma_px)

            # Through-plane resolution perturbation controlled by v_z^thr = 2.0 mm and m_max = 0.7
            s_z = float(rng.uniform(s_min, s_max))
            if v[2] < self.cfg.vz_threshold_mm and x.shape[0] >= 3:
                # Simulate thicker slice partial-volume mixing with adjacent slices
                mix_alpha = float(rng.uniform(0.0, self.cfg.max_perturbation_prob * 0.4))
                center_idx = x.shape[0] // 2
                x[center_idx] = (1.0 - mix_alpha) * x[center_idx] + 0.5 * mix_alpha * (
                    x[center_idx - 1] + x[center_idx + 1]
                )
                v[2] = np.clip(v[2] * max(1.0, s_z), 0.6, 3.5)
            else:
                v[2] = np.clip(v[2] * float(rng.uniform(0.85, 1.25)), 0.6, 3.5)

        # 3. Acquisition-aware intensity transformations (restricted to MRI inputs X_z)
        if rng.random() < 0.5:
            gamma = float(rng.uniform(*self.cfg.contrast_gamma_range))
            fg = x > 1e-4
            x_fg = np.clip(x[fg], 1e-5, 1.0)
            x[fg] = np.power(x_fg, gamma)

        if rng.random() < 0.5 and self.cfg.noise_std > 0:
            sigma_n = float(rng.uniform(0.0, self.cfg.noise_std))
            noise = rng.normal(0.0, sigma_n, size=x.shape).astype(np.float32)
            fg_mask = (x > 1e-4).astype(np.float32)
            x = np.clip(x + noise * fg_mask, 0.0, 1.0)

        return x.astype(np.float32), m.astype(np.float32), v.astype(np.float32)
