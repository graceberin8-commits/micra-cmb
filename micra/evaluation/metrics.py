"""
Section 4.4 & 5.1–5.6: Voxel-Level and Lesion-Level Evaluation Metrics, FROC & CPM,
Candidate-Level ROC AUC, Subject-Level Bootstrap Confidence Intervals (2,000 resamples),
Holm-Corrected Wilcoxon Signed-Rank Test, and False-Positive Categorization by Mimic Type.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, roc_curve

from micra.data.dataset import VALDOScan
from micra.evaluation.postprocessing import (
    DetectedLesion3D,
    extract_3d_lesion_candidates,
)


def compute_voxel_dsc_iou(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Section 4.4: Voxel-level Dice similarity coefficient (DSC) and Intersection-over-Union (IoU),
    computed per scan on scans containing at least one ground-truth CMB.
    """
    p = (pred_bin > 0.5).astype(np.float64).ravel()
    g = (gt_bin > 0.5).astype(np.float64).ravel()
    inter = float(np.sum(p * g))
    p_sum = float(np.sum(p))
    g_sum = float(np.sum(g))
    union = p_sum + g_sum - inter

    if g_sum == 0 and p_sum == 0:
        return 100.0, 100.0
    dsc = (2.0 * inter) / (p_sum + g_sum + eps) * 100.0
    iou = inter / (union + eps) * 100.0
    return float(dsc), float(iou)


def match_lesions_center_distance(
    pred_lesions: List[DetectedLesion3D],
    gt_lesions: List[DetectedLesion3D],
    match_radius_mm: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Section 3.7, 4.4 & 4.5:
    Matches predicted 3D lesion candidates to reference ground-truth CMB annotations
    using a physical 3D Euclidean center-of-mass distance criterion (\\rho_m = 5.0 mm).
    Returns:
      - matched_pairs: list of (pred_idx, gt_idx)
      - unmatched_pred_indices: false positives (FP)
      - unmatched_gt_indices: false negatives (FN)
    """
    n_pred = len(pred_lesions)
    n_gt = len(gt_lesions)
    if n_pred == 0 or n_gt == 0:
        return [], list(range(n_pred)), list(range(n_gt))

    cost_matrix = np.full((n_pred, n_gt), fill_value=1e6, dtype=np.float64)
    for i, p in enumerate(pred_lesions):
        p_xyz = np.asarray(p.centroid_mm, dtype=np.float64)
        for j, g in enumerate(gt_lesions):
            g_xyz = np.asarray(g.centroid_mm, dtype=np.float64)
            dist_mm = float(np.linalg.norm(p_xyz - g_xyz))
            if dist_mm <= match_radius_mm:
                cost_matrix[i, j] = dist_mm

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_pairs: List[Tuple[int, int]] = []
    matched_preds = set()
    matched_gts = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= match_radius_mm:
            matched_pairs.append((int(r), int(c)))
            matched_preds.add(int(r))
            matched_gts.add(int(c))

    unmatched_preds = [i for i in range(n_pred) if i not in matched_preds]
    unmatched_gts = [j for j in range(n_gt) if j not in matched_gts]
    return matched_pairs, unmatched_preds, unmatched_gts


def evaluate_scan_predictions(
    prob_volume: np.ndarray,
    gt_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    tau: float = 0.5,
    match_radius_mm: float = 5.0,
    min_voxels: int = 1,
) -> Dict:
    """Evaluates both voxel-level and lesion-level metrics on a single 3D scan."""
    pred_bin, pred_lesions = extract_3d_lesion_candidates(
        prob_volume, spacing_mm=spacing_mm, tau=tau, min_voxels=min_voxels
    )
    _, gt_lesions = extract_3d_lesion_candidates(
        gt_mask, spacing_mm=spacing_mm, tau=0.5, min_voxels=1
    )

    has_gt_cmb = len(gt_lesions) > 0
    dsc, iou = compute_voxel_dsc_iou(pred_bin, gt_mask) if has_gt_cmb else (np.nan, np.nan)

    matched_pairs, fp_indices, fn_indices = match_lesions_center_distance(
        pred_lesions, gt_lesions, match_radius_mm=match_radius_mm
    )
    tp = len(matched_pairs)
    fp = len(fp_indices)
    fn = len(fn_indices)

    sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else (100.0 if fn == 0 else 0.0)
    f1 = (2.0 * tp / (2.0 * tp + fp + fn) * 100.0) if (2 * tp + fp + fn) > 0 else 100.0

    # Size-stratified ground-truth & true-positive counts (Table 5)
    matched_gt_set = {g_idx for _, g_idx in matched_pairs}
    size_stats = {
        grp: {"gt": 0, "tp": 0} for grp in ("small", "medium", "large")
    }
    single_slice_stats = {"gt": 0, "tp": 0}

    for g_idx, g_les in enumerate(gt_lesions):
        grp = g_les.size_group
        size_stats[grp]["gt"] += 1
        if g_idx in matched_gt_set:
            size_stats[grp]["tp"] += 1
        if g_les.single_slice:
            single_slice_stats["gt"] += 1
            if g_idx in matched_gt_set:
                single_slice_stats["tp"] += 1

    return {
        "has_gt_cmb": has_gt_cmb,
        "dsc": dsc,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sens,
        "precision": prec,
        "f1": f1,
        "pred_lesions": pred_lesions,
        "gt_lesions": gt_lesions,
        "matched_pairs": matched_pairs,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "size_stats": size_stats,
        "single_slice_stats": single_slice_stats,
    }


def evaluate_cohort_predictions(
    prob_volumes: List[np.ndarray],
    scans: List[VALDOScan],
    tau: float = 0.5,
    match_radius_mm: float = 5.0,
) -> Dict:
    """
    Computes aggregate Table 2, Table 5, and Table 7 metrics across a set of scans:
      - DSC (%) and IoU (%) averaged over CMB-positive scans
      - Lesion-level Sensitivity (%), Precision (%), F1 (%), and FP/scan across all scans
      - Size-stratified sensitivity (Small <3 mm, Medium 3-5 mm, Large >5 mm)
      - Single-slice CMB sensitivity in thick-slice cohorts (SABRE and ALFA, v_z = 3.0 mm)
    """
    per_scan_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    size_totals = {grp: {"gt": 0, "tp": 0} for grp in ("small", "medium", "large")}
    thick_single_slice = {"gt": 0, "tp": 0}

    for prob_vol, scan in zip(prob_volumes, scans):
        res = evaluate_scan_predictions(
            prob_vol,
            scan.mask,
            spacing_mm=scan.spacing_mm,
            tau=tau,
            match_radius_mm=match_radius_mm,
        )
        res["subject_id"] = scan.subject_id
        res["cohort"] = scan.cohort
        per_scan_results.append(res)

        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]

        for grp in ("small", "medium", "large"):
            size_totals[grp]["gt"] += res["size_stats"][grp]["gt"]
            size_totals[grp]["tp"] += res["size_stats"][grp]["tp"]

        if scan.cohort in ("SABRE", "ALFA") or scan.spacing_mm[2] >= 2.5:
            thick_single_slice["gt"] += res["single_slice_stats"]["gt"]
            thick_single_slice["tp"] += res["single_slice_stats"]["tp"]

    pos_dscs = [r["dsc"] for r in per_scan_results if r["has_gt_cmb"] and not np.isnan(r["dsc"])]
    pos_ious = [r["iou"] for r in per_scan_results if r["has_gt_cmb"] and not np.isnan(r["iou"])]

    mean_dsc = float(np.mean(pos_dscs)) if pos_dscs else 0.0
    mean_iou = float(np.mean(pos_ious)) if pos_ious else 0.0

    sens = (total_tp / max(1, total_tp + total_fn)) * 100.0
    prec = (total_tp / max(1, total_tp + total_fp)) * 100.0
    f1 = (2.0 * total_tp / max(1, 2 * total_tp + total_fp + total_fn)) * 100.0
    fp_per_scan = float(total_fp) / max(1, len(scans))

    sens_by_size = {}
    for grp in ("small", "medium", "large"):
        gt_c = size_totals[grp]["gt"]
        tp_c = size_totals[grp]["tp"]
        sens_by_size[grp] = (tp_c / gt_c * 100.0) if gt_c > 0 else 0.0

    single_slice_sens = (
        (thick_single_slice["tp"] / thick_single_slice["gt"] * 100.0)
        if thick_single_slice["gt"] > 0
        else 0.0
    )

    return {
        "DSC": mean_dsc,
        "IoU": mean_iou,
        "Sensitivity": sens,
        "Precision": prec,
        "F1": f1,
        "FP_per_scan": fp_per_scan,
        "Sens_small": sens_by_size["small"],
        "Sens_medium": sens_by_size["medium"],
        "Sens_large": sens_by_size["large"],
        "Single_slice_sens": single_slice_sens,
        "per_scan": per_scan_results,
    }


def compute_froc_and_cpm(
    prob_volumes: List[np.ndarray],
    scans: List[VALDOScan],
    fp_rates: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    match_radius_mm: float = 5.0,
    low_tau: float = 0.15,
) -> Dict[str, float]:
    """
    Section 4.4 & Table 3:
    Computes Free-Response ROC (FROC) sensitivity (%) at fixed false positives per scan
    (0.5, 1, 2, 4 FP/scan) and the Competition Performance Metric (CPM, their mean).
    """
    all_candidates: List[Tuple[float, bool]] = []  # (score, is_tp)
    total_gt = 0

    for prob_vol, scan in zip(prob_volumes, scans):
        _, pred_lesions = extract_3d_lesion_candidates(
            prob_vol, spacing_mm=scan.spacing_mm, tau=low_tau, min_voxels=1
        )
        _, gt_lesions = extract_3d_lesion_candidates(
            scan.mask, spacing_mm=scan.spacing_mm, tau=0.5, min_voxels=1
        )
        total_gt += len(gt_lesions)

        matched_pairs, fp_indices, _ = match_lesions_center_distance(
            pred_lesions, gt_lesions, match_radius_mm=match_radius_mm
        )
        tp_pred_set = {p_idx for p_idx, _ in matched_pairs}
        for p_idx, p_les in enumerate(pred_lesions):
            all_candidates.append((p_les.score, p_idx in tp_pred_set))

    num_scans = max(1, len(scans))
    total_gt = max(1, total_gt)
    all_candidates.sort(key=lambda x: x[0], reverse=True)

    froc_sens: Dict[str, float] = {}
    if not all_candidates:
        for r in fp_rates:
            froc_sens[f"FROC@{r}"] = 0.0
        froc_sens["CPM"] = 0.0
        return froc_sens

    tp_cum = 0
    fp_cum = 0
    fp_per_scan_curve = []
    sens_curve = []

    for _, is_tp in all_candidates:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        fp_per_scan_curve.append(fp_cum / float(num_scans))
        sens_curve.append((tp_cum / float(total_gt)) * 100.0)

    fp_arr = np.asarray(fp_per_scan_curve, dtype=np.float64)
    sens_arr = np.asarray(sens_curve, dtype=np.float64)

    vals = []
    for rate in fp_rates:
        valid = np.where(fp_arr <= rate)[0]
        s_at_rate = float(sens_arr[valid[-1]]) if valid.size > 0 else float(sens_arr[0])
        froc_sens[f"FROC@{rate}"] = s_at_rate
        vals.append(s_at_rate)

    froc_sens["CPM"] = float(np.mean(vals))
    return froc_sens


def compute_candidate_roc_auc(
    prob_volumes: List[np.ndarray],
    scans: List[VALDOScan],
    match_radius_mm: float = 5.0,
    low_tau: float = 0.10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Section 5.1, Table 3 & Fig. 2:
    Computes candidate-level ROC curve and AUC pooled over test scans.
    """
    y_true: List[int] = []
    y_score: List[float] = []

    for prob_vol, scan in zip(prob_volumes, scans):
        _, pred_lesions = extract_3d_lesion_candidates(
            prob_vol, spacing_mm=scan.spacing_mm, tau=low_tau, min_voxels=1
        )
        _, gt_lesions = extract_3d_lesion_candidates(
            scan.mask, spacing_mm=scan.spacing_mm, tau=0.5, min_voxels=1
        )
        matched_pairs, _, fn_indices = match_lesions_center_distance(
            pred_lesions, gt_lesions, match_radius_mm=match_radius_mm
        )
        tp_set = {p_idx for p_idx, _ in matched_pairs}
        for p_idx, p_les in enumerate(pred_lesions):
            y_true.append(1 if p_idx in tp_set else 0)
            y_score.append(p_les.score)

        # Include missed ground-truth lesions as low-score positives
        for g_idx in fn_indices:
            g_les = gt_lesions[g_idx]
            cy, cx, cz = [int(round(c)) for c in g_les.centroid_voxel]
            cy = int(np.clip(cy, 0, prob_vol.shape[0] - 1))
            cx = int(np.clip(cx, 0, prob_vol.shape[1] - 1))
            cz = int(np.clip(cz, 0, prob_vol.shape[2] - 1))
            y_true.append(1)
            y_score.append(float(prob_vol[cy, cx, cz]))

    if len(set(y_true)) < 2:
        fpr = np.linspace(0.0, 1.0, 50)
        tpr = fpr
        return 0.5, fpr, tpr

    auc_val = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc_val, fpr, tpr


def bootstrap_confidence_interval(
    subject_values: List[float],
    num_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Section 4.4: Computes mean and 95% confidence interval from 2,000 subject-level
    bootstrap resamples.
    """
    arr = np.asarray(subject_values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_resamples, dtype=np.float64)
    n = arr.size
    for i in range(num_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return float(np.mean(arr)), low, high


def wilcoxon_holm_significance(
    micra_subject_f1: List[float],
    baseline_subject_f1_dict: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Section 5.1 & Table 2:
    Wilcoxon signed-rank test on per-subject lesion F1 versus MICRA, with Holm-Bonferroni
    step-down multiple-comparison correction.
    """
    micra_arr = np.asarray(micra_subject_f1, dtype=np.float64)
    raw_p: Dict[str, float] = {}
    for name, b_vals in baseline_subject_f1_dict.items():
        b_arr = np.asarray(b_vals, dtype=np.float64)
        diff = micra_arr - b_arr
        if np.allclose(diff, 0.0):
            raw_p[name] = 1.0
        else:
            _, p_val = stats.wilcoxon(micra_arr, b_arr, zero_method="zsplit")
            raw_p[name] = float(p_val)

    # Holm step-down correction
    sorted_items = sorted(raw_p.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    corrected: Dict[str, float] = {}
    prev_p = 0.0
    for idx, (name, p_val) in enumerate(sorted_items):
        adj_p = min(1.0, p_val * float(m - idx))
        adj_p = max(adj_p, prev_p)
        corrected[name] = adj_p
        prev_p = adj_p
    return corrected


def categorize_false_positives_by_mimic_type(
    per_scan_results: List[Dict],
    scans: List[VALDOScan],
) -> Dict[str, float]:
    """
    Section 5.6 & Table 8:
    Categorizes all false positives into four mimic categories:
      1. "Vessel": vessel cross-section (elongated through-plane persistence or proximity to vessel mimic)
      2. "Artefact": susceptibility artefact near air-tissue / cortical boundary
      3. "Other focal hypointensity": calcification-like or non-vascular focal deposit
      4. "Other": remaining false positives
    Returns false positives per scan (FP/scan) for each category and Total.
    """
    counts = {"Vessel": 0, "Artefact": 0, "Other focal hypointensity": 0, "Other": 0}
    num_scans = max(1, len(scans))

    for res, scan in zip(per_scan_results, scans):
        H, W, _ = scan.volume.shape
        cy_b, cx_b = H / 2.0, W / 2.0
        ry_b, rx_b = H * 0.40, W * 0.36

        pred_lesions: List[DetectedLesion3D] = res["pred_lesions"]
        for fp_idx in res["fp_indices"]:
            fp_les = pred_lesions[fp_idx]
            fy, fx, fz = fp_les.centroid_voxel

            # Check proximity to known ground-truth mimic annotations if present
            matched_cat: Optional[str] = None
            for m_ann in scan.mimics:
                my, mx, mz = m_ann.center_voxel
                if np.hypot(fy - my, fx - mx) <= 5.0:
                    if m_ann.mimic_type == "vessel":
                        matched_cat = "Vessel"
                    elif m_ann.mimic_type == "artefact":
                        matched_cat = "Artefact"
                    break

            if matched_cat is None:
                radial = np.sqrt(((fy - cy_b) / ry_b) ** 2 + ((fx - cx_b) / rx_b) ** 2)
                if fp_les.num_slices_spanned >= 2:
                    matched_cat = "Vessel"
                elif radial >= 0.78:
                    matched_cat = "Artefact"
                elif fp_les.diameter_eq_mm <= 3.5:
                    matched_cat = "Other focal hypointensity"
                else:
                    matched_cat = "Other"

            counts[matched_cat] += 1

    out = {k: counts[k] / float(num_scans) for k in counts}
    out["Total"] = sum(out.values())
    return out
