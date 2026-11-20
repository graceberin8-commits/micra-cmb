"""
Section 3.7, 4.1 & 4.5: Three-Dimensional Probability Volume Reassembly,
Thresholding (Eq. 23), 3D Connected Component Extraction, and Equivalent Spherical
Diameter Calculation D_{eq} = (6V / \\pi)^{1/3}.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
from scipy import ndimage


@dataclass
class DetectedLesion3D:
    lesion_id: int
    centroid_voxel: Tuple[float, float, float]  # (y, x, z) in voxel indices
    centroid_mm: Tuple[float, float, float]  # (y_mm, x_mm, z_mm) in physical coordinates
    num_voxels: int
    volume_mm3: float
    diameter_eq_mm: float
    size_group: str  # "small" (<3 mm), "medium" (3-5 mm), "large" (>5 mm)
    score: float  # Mean/peak probability score inside component
    num_slices_spanned: int
    single_slice: bool


def threshold_probability_volume(
    prob_volume: np.ndarray,
    tau: float = 0.5,
) -> np.ndarray:
    """
    Implements Eq. (23):
        M^{3D}(x, y, z) = 1 if \\hat{M}^{3D}(x, y, z) >= \\tau else 0
    """
    return (prob_volume >= tau).astype(np.float32)


def compute_equivalent_spherical_diameter_mm(volume_mm3: float) -> float:
    """
    Section 4.1:
        D_{eq} = (6 V / \\pi)^{1/3}
    """
    if volume_mm3 <= 0.0:
        return 0.0
    return float(((6.0 * volume_mm3) / math.pi) ** (1.0 / 3.0))


def classify_lesion_size_group(
    diameter_eq_mm: float,
    small_thresh_mm: float = 3.0,
    large_thresh_mm: float = 5.0,
) -> str:
    """
    Section 4.1 & Table 1 / Table 5:
      - "small":  D_{eq} < 3 mm
      - "medium": 3 mm <= D_{eq} <= 5 mm
      - "large":  D_{eq} > 5 mm
    """
    if diameter_eq_mm < small_thresh_mm:
        return "small"
    elif diameter_eq_mm <= large_thresh_mm:
        return "medium"
    else:
        return "large"


def extract_3d_lesion_candidates(
    prob_volume: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    tau: float = 0.5,
    min_voxels: int = 1,
) -> Tuple[np.ndarray, List[DetectedLesion3D]]:
    """
    Extracts 3D connected lesion components from the reconstructed probability volume \\hat{M}^{3D}
    after thresholding at \\tau (Eq. 23).
    """
    bin_vol = threshold_probability_volume(prob_volume, tau=tau)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)  # 18-connectivity
    labeled, num_features = ndimage.label(bin_vol, structure=structure)

    vx, vy, vz = float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2])
    voxel_vol_mm3 = vx * vy * vz

    lesions: List[DetectedLesion3D] = []
    filtered_bin = np.zeros_like(bin_vol)

    for comp_id in range(1, num_features + 1):
        coords = np.argwhere(labeled == comp_id)
        n_vox = coords.shape[0]
        if n_vox < min_voxels:
            continue

        filtered_bin[labeled == comp_id] = 1.0
        cy, cx, cz = coords.mean(axis=0)
        cy_mm, cx_mm, cz_mm = float(cy * vy), float(cx * vx), float(cz * vz)

        vol_mm3 = float(n_vox) * voxel_vol_mm3
        d_eq = compute_equivalent_spherical_diameter_mm(vol_mm3)
        sz_group = classify_lesion_size_group(d_eq)

        comp_probs = prob_volume[labeled == comp_id]
        score = float(0.7 * np.max(comp_probs) + 0.3 * np.mean(comp_probs))
        unique_z = np.unique(coords[:, 2])
        n_slices = int(unique_z.size)

        lesions.append(
            DetectedLesion3D(
                lesion_id=len(lesions) + 1,
                centroid_voxel=(float(cy), float(cx), float(cz)),
                centroid_mm=(cy_mm, cx_mm, cz_mm),
                num_voxels=int(n_vox),
                volume_mm3=vol_mm3,
                diameter_eq_mm=d_eq,
                size_group=sz_group,
                score=score,
                num_slices_spanned=n_slices,
                single_slice=(n_slices == 1),
            )
        )

    # Sort candidates by descending confidence score
    lesions.sort(key=lambda item: item.score, reverse=True)
    return filtered_bin, lesions
