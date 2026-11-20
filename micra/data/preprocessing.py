"""
Section 3.2: Input Formulation and Preprocessing (Eqs. 2–3)
and Section 4.2 Protocol P3 (Spacing Shift resampling).
"""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage


def validate_volume_and_spacing(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Validates MRI volume I in R^{H x W x D}, voxel spacing v = [v_x, v_y, v_z] in mm (Eq. 3),
    and aligns ground-truth segmentation mask M in {0, 1}^{H x W x D}.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume (H, W, D), got shape {volume.shape}")
    v = np.asarray(spacing, dtype=np.float32)
    if v.shape != (3,) or np.any(v <= 0):
        raise ValueError(f"Voxel spacing v = [v_x, v_y, v_z] must be 3 positive values in mm, got {spacing}")

    vol_f32 = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask_bin = None
    if mask is not None:
        if mask.shape != volume.shape:
            raise ValueError(f"Volume shape {volume.shape} and mask shape {mask.shape} do not match.")
        mask_bin = (mask > 0.5).astype(np.float32)

    return vol_f32, v, mask_bin


def normalize_t2star_intensity(
    volume: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Consistent T2*-weighted MRI intensity normalization:
    foreground brain mask estimation, percentile clipping [p_low, p_high],
    and z-score standardization followed by rescaling to [0, 1] inside foreground.
    """
    foreground = volume > (np.percentile(volume, 5.0) + eps)
    if np.sum(foreground) < 10:
        foreground = np.ones_like(volume, dtype=bool)

    fg_vals = volume[foreground]
    low_val = np.percentile(fg_vals, p_low)
    high_val = np.percentile(fg_vals, p_high)
    clipped = np.clip(volume, low_val, high_val)

    mean_val = float(np.mean(clipped[foreground]))
    std_val = float(np.std(clipped[foreground])) + eps
    z_norm = (clipped - mean_val) / std_val

    # Map robust z-score [-3, 3] into [0, 1] while keeping zero background at 0
    norm_vol = np.clip((z_norm + 3.0) / 6.0, 0.0, 1.0)
    norm_vol[~foreground] = 0.0
    return norm_vol.astype(np.float32)


def extract_25d_slice_stack(
    volume: np.ndarray,
    z: int,
    k: int = 1,
) -> np.ndarray:
    """
    Implements Eq. (2):
        X_z = Stack(I_{z-k}, ..., I_z, ..., I_{z+k})
    with replicate boundary handling for slices near volume boundaries (z < k or z >= D - k).
    Returns array of shape (2k + 1, H, W).
    """
    _, _, depth = volume.shape
    if not (0 <= z < depth):
        raise IndexError(f"Central slice index z={z} out of bounds for depth D={depth}")

    slices = []
    for offset in range(-k, k + 1):
        z_idx = int(np.clip(z + offset, 0, depth - 1))
        slices.append(volume[:, :, z_idx])
    return np.stack(slices, axis=0).astype(np.float32)


def resample_volume_spacing_shift(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    slice_step_m: int = 1,
    target_inplane_mm: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Implements Protocol P3 (Section 4.2):
    - Subsamples or pools through-plane slices by step m in {2, 3} (e.g. RSS 0.8 mm -> 1.6 mm or 2.4 mm).
    - Or resamples in-plane spacing to ALFA in-plane spacing (1.0 x 1.0 mm).
    Returns (resampled_volume, resampled_mask, new_spacing).
    """
    vx, vy, vz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    new_vol = volume.copy()
    new_mask = mask.copy()

    if slice_step_m > 1:
        # Simulate thicker slices with partial-volume averaging over m slices and step m
        d = new_vol.shape[2]
        n_steps = max(1, d // slice_step_m)
        vol_slices = []
        mask_slices = []
        for idx in range(n_steps):
            start = idx * slice_step_m
            end = min(d, start + slice_step_m)
            vol_slices.append(np.mean(new_vol[:, :, start:end], axis=2))
            mask_slices.append(np.max(new_mask[:, :, start:end], axis=2))
        new_vol = np.stack(vol_slices, axis=2)
        new_mask = np.stack(mask_slices, axis=2)
        vz = vz * float(slice_step_m)

    if target_inplane_mm is not None:
        tx, ty = float(target_inplane_mm[0]), float(target_inplane_mm[1])
        zoom_h = vx / tx
        zoom_w = vy / ty
        new_vol = ndimage.zoom(new_vol, (zoom_h, zoom_w, 1.0), order=1)
        new_mask = (ndimage.zoom(new_mask, (zoom_h, zoom_w, 1.0), order=0) > 0.5).astype(np.float32)
        vx, vy = tx, ty

    return new_vol.astype(np.float32), new_mask.astype(np.float32), (vx, vy, vz)
