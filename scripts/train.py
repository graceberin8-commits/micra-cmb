"""
CLI Training Script for MICRA and Baselines under Protocol P1 (5-fold CV) or P2 (LOCO).
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
    create_p1_cross_validation_splits,
    create_p2_loco_splits,
)
from micra.models.micra_net import MICRANet
from micra.models.baselines import build_baseline_model
from micra.engine.trainer import MICRATrainer
from micra.evaluation.protocols import predict_3d_volume
from micra.evaluation.metrics import evaluate_cohort_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train MICRA or Baseline on VALDO Task 2")
    parser.add_argument("--data-dir", type=str, default="", help="Path to VALDO Task 2 NIfTI directory (empty = synthetic VALDO)")
    parser.add_argument("--model", type=str, default="MICRA", help="MICRA or baseline name from BASELINE_REGISTRY")
    parser.add_argument("--ablation", type=str, default="full", help="Ablation variant name (see MICRAConfig.clone_with_ablation)")
    parser.add_argument("--protocol", type=str, default="P1", choices=["P1", "P2"], help="P1 (5-fold CV) or P2 (LOCO)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0..4) for P1")
    parser.add_argument("--held-out-cohort", type=str, default="SABRE", choices=["SABRE", "RSS", "ALFA"], help="Held-out cohort for P2 LOCO")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints", help="Directory to save checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    base_cfg = MICRAConfig()
    cfg = base_cfg.clone_with_ablation(args.ablation)
    cfg.training.max_epochs = args.epochs

    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"Loading VALDO NIfTI scans from {args.data_dir}...")
        scans = load_valdo_nifti_directory(args.data_dir)
    else:
        print("Generating physics-consistent synthetic VALDO Task 2 dataset (Table 1 cohorts)...")
        scans = generate_synthetic_valdo_cohort(
            spatial_size=(64, 64), num_slices_thin=10, num_slices_thick=6, seed=args.seed, full_72_scans=False
        )

    if args.protocol == "P1":
        splits = create_p1_cross_validation_splits(scans, num_folds=5, seed=args.seed)
        split = splits[args.fold]
    else:
        loco = create_p2_loco_splits(scans, seed=args.seed)
        split = loco[args.held_out_cohort]

    if args.model == "MICRA":
        model = MICRANet(cfg)
    else:
        model = build_baseline_model(args.model)

    trainer = MICRATrainer(model, cfg)
    print(f"Training {args.model} (ablation={args.ablation}) on {len(split['train'])} train scans, {len(split['val'])} val scans...")
    history = trainer.fit(split["train"], split["val"], max_epochs=args.epochs, seed=args.seed)

    test_probs = [predict_3d_volume(model, sc, k_neighbors=cfg.k_neighbors) for sc in split["test"]]
    test_metrics = evaluate_cohort_predictions(test_probs, split["test"])

    ckpt_path = os.path.join(args.output_dir, f"{args.model.replace(' ', '_')}_{args.ablation}_{args.protocol}.pt")
    torch.save({"model_state_dict": model.state_dict(), "test_metrics": test_metrics}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print(
        f"Test Results -> DSC: {test_metrics['DSC']:.2f}% | IoU: {test_metrics['IoU']:.2f}% | "
        f"Sens: {test_metrics['Sensitivity']:.2f}% | Prec: {test_metrics['Precision']:.2f}% | "
        f"F1: {test_metrics['F1']:.2f}% | FP/scan: {test_metrics['FP_per_scan']:.2f}"
    )


if __name__ == "__main__":
    main()
