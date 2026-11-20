"""
CLI Evaluation Script for MICRA Protocols P1 (Cross-Validation), P2 (LOCO),
P3 (Spacing Shift), FROC/CPM (Table 3), and Mimic False-Positive Categorization (Table 8).
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from micra.config import MICRAConfig
from micra.data.dataset import (
    load_valdo_nifti_directory,
    generate_synthetic_valdo_cohort,
    create_p3_spacing_shift_scans,
)
from micra.models.micra_net import MICRANet
from micra.evaluation.protocols import predict_3d_volume, profile_computational_cost
from micra.evaluation.metrics import (
    evaluate_cohort_predictions,
    compute_froc_and_cpm,
    compute_candidate_roc_auc,
    categorize_false_positives_by_mimic_type,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MICRA on VALDO Protocols P1, P2, P3")
    parser.add_argument("--data-dir", type=str, default="", help="Path to VALDO NIfTI directory")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to trained checkpoint")
    parser.add_argument("--ablation", type=str, default="full", help="Ablation variant")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MICRAConfig().clone_with_ablation(args.ablation)
    model = MICRANet(cfg)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

    if args.data_dir and os.path.isdir(args.data_dir):
        scans = load_valdo_nifti_directory(args.data_dir)
    else:
        scans = generate_synthetic_valdo_cohort(
            spatial_size=(64, 64), num_slices_thin=10, num_slices_thick=6, seed=42, full_72_scans=False
        )

    probs = [predict_3d_volume(model, sc, k_neighbors=cfg.k_neighbors) for sc in scans]
    metrics = evaluate_cohort_predictions(probs, scans)
    froc = compute_froc_and_cpm(probs, scans)
    auc_val, _, _ = compute_candidate_roc_auc(probs, scans)
    fp_mimics = categorize_false_positives_by_mimic_type(metrics["per_scan"], scans)
    cost = profile_computational_cost(model, slice_shape=(1, 3, 64, 64))

    print("=== MICRA Evaluation Summary ===")
    print(f"Voxel-level : DSC = {metrics['DSC']:.2f}%, IoU = {metrics['IoU']:.2f}%")
    print(
        f"Lesion-level: Sens = {metrics['Sensitivity']:.2f}%, Prec = {metrics['Precision']:.2f}%, "
        f"F1 = {metrics['F1']:.2f}%, FP/scan = {metrics['FP_per_scan']:.2f}"
    )
    print(f"FROC & AUC  : CPM = {froc['CPM']:.2f}%, Candidate AUC = {auc_val:.3f}")
    print(f"FP by Mimic : {fp_mimics}")
    print(f"Model Cost  : {cost}")

    # Protocol P3: Spacing shift on RSS scans
    rss_scans = [s for s in scans if s.cohort == "RSS"]
    if rss_scans:
        p3_sets = create_p3_spacing_shift_scans(rss_scans)
        for shift_name, shifted_scans in p3_sets.items():
            s_probs = [predict_3d_volume(model, sc, k_neighbors=cfg.k_neighbors) for sc in shifted_scans]
            s_metrics = evaluate_cohort_predictions(s_probs, shifted_scans)
            print(f"P3 Spacing Shift [{shift_name}] -> F1 = {s_metrics['F1']:.2f}%")


if __name__ == "__main__":
    main()
