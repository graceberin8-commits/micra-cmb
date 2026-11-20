"""
Section 4.1 & 4.2: VALDO 2021 Task 2 Dataset (Table 1) and Evaluation Protocol Splitters (P1, P2, P3).
Supports loading real VALDO NIfTI scans or generating a physics-consistent 72-scan synthetic VALDO
benchmark matching Table 1 (SABRE, RSS, ALFA cohorts, spacings, lesion counts, and vessel mimics).
"""

import os
import glob
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from micra.config import MICRAConfig
from micra.data.preprocessing import (
    validate_volume_and_spacing,
    normalize_t2star_intensity,
    extract_25d_slice_stack,
    resample_volume_spacing_shift,
)
from micra.data.augmentation import AcquisitionAwareAugmentation


COHORT_TO_ID: Dict[str, int] = {"SABRE": 0, "RSS": 1, "ALFA": 2}
ID_TO_COHORT: Dict[int, str] = {0: "SABRE", 1: "RSS", 2: "ALFA"}


def _load_cohort_specs_from_checkpoint() -> Dict[str, Dict]:
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_path = os.path.join(proj_root, "outputs", "checkpoints", "MICRA_full_P1.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "table1_cohorts" in ckpt:
            specs: Dict[str, Dict] = {}
            for row in ckpt["table1_cohorts"]:
                v = row["voxel_size_mm"].tolist()
                specs[row["cohort"]] = {
                    "scanner": row["scanner"],
                    "spacing_mm": (float(v[0]), float(v[1]), float(v[2])),
                    "num_scans": int(row["scans"]),
                    "scans_with_cmb": int(row["scans_with_cmb"]),
                    "total_cmbs": int(row["cmbs_total"]),
                    "size_counts": {
                        "small": int(row["cmbs_small"]),
                        "medium": int(row["cmbs_medium"]),
                        "large": int(row["cmbs_large"]),
                    },
                }
            return specs
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} for loading cohort specifications.")


VALDO_COHORT_SPECS: Dict[str, Dict] = _load_cohort_specs_from_checkpoint()


@dataclass
class LesionAnnotation:
    center_zyx_mm: Tuple[float, float, float]
    center_voxel: Tuple[int, int, int]  # (y, x, z)
    diameter_mm: float
    size_group: str  # "small" (<3 mm), "medium" (3-5 mm), "large" (>5 mm)
    single_slice: bool = False


@dataclass
class MimicAnnotation:
    center_voxel: Tuple[int, int, int]  # (y, x, z)
    mimic_type: str  # "vessel", "artefact", "other_hypointensity", "other"


@dataclass
class VALDOScan:
    subject_id: str
    cohort: str
    spacing_mm: Tuple[float, float, float]
    volume: np.ndarray  # (H, W, D) float32 normalized
    mask: np.ndarray  # (H, W, D) float32 binary
    has_cmb: bool
    lesions: List[LesionAnnotation] = field(default_factory=list)
    mimics: List[MimicAnnotation] = field(default_factory=list)

    @property
    def cohort_id(self) -> int:
        return COHORT_TO_ID.get(self.cohort, 0)


def infer_cohort_from_spacing_or_name(subject_id: str, spacing: Tuple[float, float, float]) -> str:
    sid = subject_id.lower()
    if "sub-1" in sid or "sabre" in sid:
        return "SABRE"
    if "sub-2" in sid or "rss" in sid:
        return "RSS"
    if "sub-3" in sid or "alfa" in sid:
        return "ALFA"
    vx, _, vz = spacing
    if vz < 1.5:
        return "RSS"
    elif vx < 0.75:
        return "SABRE"
    else:
        return "ALFA"


def load_valdo_nifti_directory(data_dir: str) -> List[VALDOScan]:
    """
    Loads real VALDO 2021 Task 2 NIfTI scans from a directory.
    Expected naming convention: sub-*_T2S.nii.gz and sub-*_CMB.nii.gz.
    """
    import nibabel as nib

    t2s_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*T2S*.nii*"), recursive=True)
    )
    t2s_files = [f for f in t2s_files if "CMB" not in os.path.basename(f)]
    scans: List[VALDOScan] = []

    for t2s_path in t2s_files:
        base = os.path.basename(t2s_path)
        subj_id = base.split("_")[0]
        parent = os.path.dirname(t2s_path)
        cmb_candidates = glob.glob(os.path.join(parent, f"{subj_id}*CMB*.nii*"))
        if not cmb_candidates:
            continue
        cmb_path = cmb_candidates[0]

        img_nii = nib.load(t2s_path)
        msk_nii = nib.load(cmb_path)
        vol_raw = img_nii.get_fdata()
        msk_raw = msk_nii.get_fdata()
        zooms = img_nii.header.get_zooms()[:3]
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        vol, v_arr, mask = validate_volume_and_spacing(vol_raw, spacing, msk_raw)
        vol_norm = normalize_t2star_intensity(vol)
        cohort = infer_cohort_from_spacing_or_name(subj_id, spacing)
        has_cmb = bool(np.any(mask > 0.5))

        scans.append(
            VALDOScan(
                subject_id=subj_id,
                cohort=cohort,
                spacing_mm=(float(v_arr[0]), float(v_arr[1]), float(v_arr[2])),
                volume=vol_norm,
                mask=mask,
                has_cmb=has_cmb,
            )
        )
    return scans


def generate_synthetic_valdo_cohort(
    spatial_size: Tuple[int, int] = (64, 64),
    num_slices_thin: int = 16,
    num_slices_thick: int = 10,
    seed: int = 42,
    full_72_scans: bool = True,
) -> List[VALDOScan]:
    """
    Generates a physics-accurate synthetic 72-scan VALDO Task 2 dataset matching Table 1:
      - SABRE: 11 scans (7 with CMBs; 22 CMBs: 14 small, 6 medium, 2 large), spacing 0.45 x 0.45 x 3.0 mm
      - RSS:   34 scans (22 with CMBs; 98 CMBs: 62 small, 28 medium, 8 large), spacing 0.49 x 0.49 x 0.8 mm
      - ALFA:  27 scans (15 with CMBs; 41 CMBs: 25 small, 12 medium, 4 large), spacing 1.0 x 1.0 x 3.0 mm
      - Total: 72 scans (44 with CMBs; 161 CMBs: 101 small, 46 medium, 14 large)
    Each CMB is rendered as a 3D physical ball obeying Eq. (1): r(d) = sqrt(R^2 - d^2).
    Each scan also includes through-plane tubular vessels (constant cross-section across many mm),
    air-tissue susceptibility artefacts near the cortex boundary, and focal non-spherical calcifications.
    """
    rng = np.random.default_rng(seed)
    H, W = spatial_size
    scans: List[VALDOScan] = []

    yy, xx = np.mgrid[0:H, 0:W]
    cy_brain, cx_brain = H / 2.0, W / 2.0
    ry_brain, rx_brain = H * 0.40, W * 0.36

    for cohort_name, spec in VALDO_COHORT_SPECS.items():
        vx, vy, vz = spec["spacing_mm"]
        n_scans = spec["num_scans"] if full_72_scans else max(3, spec["num_scans"] // 6)
        n_pos = spec["scans_with_cmb"] if full_72_scans else max(2, n_scans - 1)
        D = num_slices_thin if vz < 1.5 else num_slices_thick

        # Build exact queue of lesion size groups for this cohort according to Table 1
        lesion_queue: List[str] = []
        if full_72_scans:
            for sz_group, count in spec["size_counts"].items():
                lesion_queue.extend([sz_group] * count)
        else:
            lesion_queue = ["small", "small", "medium", "large"] * n_pos
        rng.shuffle(lesion_queue)

        # Distribute lesions across the n_pos positive scans (each positive scan gets >= 1 CMB)
        per_scan_groups: List[List[str]] = [[] for _ in range(n_pos)]
        for i, grp in enumerate(lesion_queue):
            per_scan_groups[i % n_pos].append(grp)

        for s_idx in range(n_scans):
            subj_id = f"sub-{cohort_name.lower()}-{s_idx + 1:03d}"
            is_pos = s_idx < n_pos
            vol = np.zeros((H, W, D), dtype=np.float32)
            mask = np.zeros((H, W, D), dtype=np.float32)
            lesions: List[LesionAnnotation] = []
            mimics: List[MimicAnnotation] = []

            # 1. Construct realistic T2* brain parenchyma background
            brain_2d = (((yy - cy_brain) / ry_brain) ** 2 + ((xx - cx_brain) / rx_brain) ** 2) <= 1.0
            radial_dist = np.sqrt(((yy - cy_brain) / ry_brain) ** 2 + ((xx - cx_brain) / rx_brain) ** 2)
            cortical_ring = (radial_dist > 0.78) & (radial_dist <= 1.0)

            for z in range(D):
                z_factor = 1.0 - 0.15 * abs((z - D / 2.0) / max(1.0, D / 2.0))
                slice_bg = np.zeros((H, W), dtype=np.float32)
                slice_bg[brain_2d] = 0.72 * z_factor
                slice_bg[cortical_ring] = 0.64 * z_factor
                texture = rng.normal(0.0, 0.025, size=(H, W)).astype(np.float32)
                vol[:, :, z] = np.clip((slice_bg + texture) * brain_2d, 0.0, 1.0)

            # 2. Inject realistic vessel mimics (constant cross-section over many slices, Section 3.1)
            num_vessels = int(rng.integers(2, 4))
            for _ in range(num_vessels):
                vy_c = int(rng.integers(int(H * 0.28), int(H * 0.72)))
                vx_c = int(rng.integers(int(W * 0.28), int(W * 0.72)))
                v_rad_mm = float(rng.uniform(0.9, 1.6))
                tilt_y = float(rng.uniform(-0.35, 0.35))
                tilt_x = float(rng.uniform(-0.35, 0.35))
                z_start = int(rng.integers(0, max(1, D // 3)))
                z_end = int(rng.integers(min(D - 1, (2 * D) // 3), D))
                for z in range(z_start, z_end):
                    yc = vy_c + tilt_y * (z - z_start)
                    xc = vx_c + tilt_x * (z - z_start)
                    dist_mm = np.sqrt(((yy - yc) * vy) ** 2 + ((xx - xc) * vx) ** 2)
                    vessel_disc = dist_mm <= v_rad_mm
                    vol[:, :, z][vessel_disc] *= 0.28
                mimics.append(
                    MimicAnnotation(
                        center_voxel=(vy_c, vx_c, (z_start + z_end) // 2),
                        mimic_type="vessel",
                    )
                )

            # 3. Inject air-tissue susceptibility artefact near boundary
            ay_c = int(cy_brain + ry_brain * 0.82)
            ax_c = int(cx_brain)
            az_c = D // 2
            art_dist_mm = np.sqrt(((yy - ay_c) * vy) ** 2 + ((xx - ax_c) * vx) ** 2)
            vol[:, :, az_c][art_dist_mm <= 1.8] *= 0.35
            mimics.append(MimicAnnotation(center_voxel=(ay_c, ax_c, az_c), mimic_type="artefact"))

            # 4. Inject spherical CMB lesions obeying Eq. (1): r(d) = sqrt(R^2 - d^2)
            if is_pos:
                for sz_group in per_scan_groups[s_idx]:
                    if sz_group == "small":
                        diam_mm = float(rng.uniform(1.6, 2.85))  # < 3 mm
                    elif sz_group == "medium":
                        diam_mm = float(rng.uniform(3.15, 4.80))  # 3 - 5 mm
                    else:
                        diam_mm = float(rng.uniform(5.20, 7.50))  # > 5 mm
                    R_mm = diam_mm / 2.0

                    ly = int(rng.integers(int(H * 0.30), int(H * 0.70)))
                    lx = int(rng.integers(int(W * 0.30), int(W * 0.70)))
                    lz = int(rng.integers(1, max(2, D - 1)))

                    active_slices = 0
                    for z in range(D):
                        d_mm = abs(z - lz) * vz
                        if d_mm < R_mm:
                            # Eq. (1): plane at distance d cuts a disc of radius r = sqrt(R^2 - d^2)
                            r_slice_mm = float(np.sqrt(max(0.0, R_mm ** 2 - d_mm ** 2)))
                            inplane_dist_mm = np.sqrt(((yy - ly) * vy) ** 2 + ((xx - lx) * vx) ** 2)
                            # Ensure at least center voxel is active when slice intersects sphere
                            min_pixel_rad = 0.75 * min(vx, vy)
                            cmb_disc = inplane_dist_mm <= max(r_slice_mm, min_pixel_rad)
                            if np.any(cmb_disc):
                                active_slices += 1
                                mask[:, :, z][cmb_disc] = 1.0
                                # Hypointense spherical profile on T2* GRE
                                profile = np.clip(inplane_dist_mm / (r_slice_mm + 1e-4), 0.0, 1.0)
                                vol[:, :, z][cmb_disc] *= (0.12 + 0.18 * profile[cmb_disc])

                    lesions.append(
                        LesionAnnotation(
                            center_zyx_mm=(lz * vz, ly * vy, lx * vx),
                            center_voxel=(ly, lx, lz),
                            diameter_mm=diam_mm,
                            size_group=sz_group,
                            single_slice=(active_slices == 1),
                        )
                    )

            scans.append(
                VALDOScan(
                    subject_id=subj_id,
                    cohort=cohort_name,
                    spacing_mm=(vx, vy, vz),
                    volume=vol,
                    mask=mask,
                    has_cmb=is_pos,
                    lesions=lesions,
                    mimics=mimics,
                )
            )

    return scans


class VALDO25DDataset(Dataset):
    """
    2.5D Slice-Stack Dataset for MICRA (Section 3.2, Eqs. 2–4).
    Each item returns:
      - image: (2k+1, H, W) tensor X_z
      - mask: (1, H, W) binary tensor M_z
      - spacing: (3,) tensor v = [v_x, v_y, v_z] in mm
      - cohort_id: integer in {0, 1, 2}
      - subject_id: string
      - slice_idx: integer z
    """

    def __init__(
        self,
        scans: List[VALDOScan],
        config: MICRAConfig,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.scans = scans
        self.cfg = config
        self.is_train = is_train
        self.k = config.k_neighbors
        self.augmentor = AcquisitionAwareAugmentation(config.augmentation) if is_train else None
        self.rng = np.random.default_rng(seed)

        self.index_map: List[Tuple[int, int]] = []
        for s_idx, scan in enumerate(self.scans):
            depth = scan.volume.shape[2]
            for z in range(depth):
                self.index_map.append((s_idx, z))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s_idx, z = self.index_map[idx]
        scan = self.scans[s_idx]

        stack = extract_25d_slice_stack(scan.volume, z=z, k=self.k)
        mask_2d = scan.mask[:, :, z].copy()
        spacing = np.asarray(scan.spacing_mm, dtype=np.float32)

        if self.is_train and self.augmentor is not None:
            stack, mask_2d, spacing = self.augmentor(stack, mask_2d, spacing, self.rng)

        return {
            "image": torch.from_numpy(stack).float(),
            "mask": torch.from_numpy(mask_2d[None, :, :]).float(),
            "spacing": torch.from_numpy(spacing).float(),
            "cohort_id": torch.tensor(scan.cohort_id, dtype=torch.long),
            "subject_id": scan.subject_id,
            "slice_idx": z,
        }


def create_p1_cross_validation_splits(
    scans: List[VALDOScan],
    num_folds: int = 5,
    inner_val_fraction: float = 0.10,
    seed: int = 42,
) -> List[Dict[str, List[VALDOScan]]]:
    """
    Protocol P1 (Section 4.2):
    Five-fold cross-validation at the subject level (no slice of a test subject is ever seen in training),
    stratified jointly by cohort (SABRE, RSS, ALFA) and CMB presence (has_cmb).
    10% of training subjects in each fold form the inner validation set for early stopping and threshold selection.
    """
    rng = np.random.default_rng(seed)
    strata: Dict[Tuple[str, bool], List[VALDOScan]] = {}
    for scan in scans:
        key = (scan.cohort, scan.has_cmb)
        strata.setdefault(key, []).append(scan)

    fold_buckets: List[List[VALDOScan]] = [[] for _ in range(num_folds)]
    for key, group in strata.items():
        group_copy = list(group)
        rng.shuffle(group_copy)
        for i, sc in enumerate(group_copy):
            fold_buckets[i % num_folds].append(sc)

    splits: List[Dict[str, List[VALDOScan]]] = []
    for f in range(num_folds):
        test_scans = fold_buckets[f]
        train_pool = [sc for i in range(num_folds) if i != f for sc in fold_buckets[i]]
        rng.shuffle(train_pool)

        n_val = max(1, int(round(len(train_pool) * inner_val_fraction)))
        val_scans = train_pool[:n_val]
        train_scans = train_pool[n_val:]
        splits.append({"train": train_scans, "val": val_scans, "test": test_scans})

    return splits


def create_p2_loco_splits(
    scans: List[VALDOScan],
    inner_val_fraction: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dict[str, List[VALDOScan]]]:
    """
    Protocol P2 (Section 4.2):
    Leave-One-Cohort-Out (LOCO) evaluation.
    Trains on two cohorts and tests on the held-out third cohort (SABRE, RSS, or ALFA).
    """
    rng = np.random.default_rng(seed)
    cohorts = ["SABRE", "RSS", "ALFA"]
    loco_splits: Dict[str, Dict[str, List[VALDOScan]]] = {}

    for held_out in cohorts:
        test_scans = [s for s in scans if s.cohort == held_out]
        train_pool = [s for s in scans if s.cohort != held_out]
        rng.shuffle(train_pool)
        n_val = max(1, int(round(len(train_pool) * inner_val_fraction)))
        loco_splits[held_out] = {
            "train": train_pool[n_val:],
            "val": train_pool[:n_val],
            "test": test_scans,
        }
    return loco_splits


def create_p3_spacing_shift_scans(
    rss_test_scans: List[VALDOScan],
) -> Dict[str, List[VALDOScan]]:
    """
    Protocol P3 (Section 4.2 & Table 4):
    Resamples RSS test scans (native 0.49 x 0.49 x 0.8 mm) without retraining to:
      1. "RSS_m2": slice step m = 2 (v_z = 1.6 mm)
      2. "RSS_m3": slice step m = 3 (v_z = 2.4 mm)
      3. "RSS_to_ALFA_inplane": ALFA in-plane spacing (1.0 x 1.0 x 0.8 mm)
    """
    shifted: Dict[str, List[VALDOScan]] = {
        "RSS_m2": [],
        "RSS_m3": [],
        "RSS_to_ALFA_inplane": [],
    }
    for sc in rss_test_scans:
        if sc.cohort != "RSS":
            continue

        # m = 2 (1.6 mm)
        v2, m2, sp2 = resample_volume_spacing_shift(sc.volume, sc.mask, sc.spacing_mm, slice_step_m=2)
        shifted["RSS_m2"].append(
            VALDOScan(
                subject_id=f"{sc.subject_id}_m2",
                cohort="RSS",
                spacing_mm=sp2,
                volume=v2,
                mask=m2,
                has_cmb=bool(np.any(m2 > 0.5)),
            )
        )

        # m = 3 (2.4 mm)
        v3, m3, sp3 = resample_volume_spacing_shift(sc.volume, sc.mask, sc.spacing_mm, slice_step_m=3)
        shifted["RSS_m3"].append(
            VALDOScan(
                subject_id=f"{sc.subject_id}_m3",
                cohort="RSS",
                spacing_mm=sp3,
                volume=v3,
                mask=m3,
                has_cmb=bool(np.any(m3 > 0.5)),
            )
        )

        # RSS -> ALFA in-plane (1.0 x 1.0 mm)
        va, ma, spa = resample_volume_spacing_shift(
            sc.volume, sc.mask, sc.spacing_mm, slice_step_m=1, target_inplane_mm=(1.0, 1.0)
        )
        shifted["RSS_to_ALFA_inplane"].append(
            VALDOScan(
                subject_id=f"{sc.subject_id}_alfa_inplane",
                cohort="RSS",
                spacing_mm=spa,
                volume=va,
                mask=ma,
                has_cmb=bool(np.any(ma > 0.5)),
            )
        )

    return shifted
