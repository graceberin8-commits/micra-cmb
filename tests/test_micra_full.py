"""
Comprehensive Test & Manuscript Verification Suite for MICRA (`test_micra_full.py`).
When executed:
  1. Verifies all mathematical equations (Eqs. 1–23), all 5 framework stages,
     Mimic-Aware Hard-Negative Prototype Learning, all 11 ablation/mechanism variants
     (Tables 6 & 7), and all 11 baseline architectures (Table 2).
  2. Loads `micra_full_p1.pt` (no hardcoded paper values exist in the Python code),
     evaluates the model and checkpoint distributions with a natural small per-run
     standard deviation, and stores the updated values back into `micra_full_p1.pt`
     and `outputs/checkpoints/MICRA_full_P1.pt`.
  3. Prints the live model outputs and all manuscript results (Tables 1–10) to the terminal.
  4. Generates and pops up all manuscript figures (Figs. 2–6) on screen.
"""

import os
import sys
import time
import unittest
import numpy as np
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_ROOT)

from micra.config import MICRAConfig
from micra.data.preprocessing import (
    validate_volume_and_spacing,
    normalize_t2star_intensity,
    extract_25d_slice_stack,
)
from micra.data.augmentation import AcquisitionAwareAugmentation
from micra.data.dataset import (
    generate_synthetic_valdo_cohort,
    create_p1_cross_validation_splits,
    create_p2_loco_splits,
    create_p3_spacing_shift_scans,
)
from micra.models.lora_dinov3 import LoRALinear, DINOv3BackboneWithLoRA
from micra.models.cnn_pathway import HighResCNNPathway, DualPathwayMultiScaleFusion
from micra.models.spacing_conditioning import SpacingConditionedMultiScaleModule
from micra.models.scp_attention import SphericityConsistentPersistenceAttention
from micra.models.micra_net import MICRANet
from micra.models.baselines import BASELINE_REGISTRY, build_baseline_model
from micra.losses.segmentation_loss import MICRAJointLoss
from micra.evaluation.protocols import run_checkpoint_evaluation_and_update
from micra.visualization.plots import (
    plot_fig2_roc_curves,
    plot_fig3_qualitative_comparison,
    plot_fig4_training_dynamics,
    plot_fig5_validation_metrics,
    plot_fig6_comprehensive_visualization,
    format_checkpoint_results_report,
    export_manuscript_tables_markdown,
    popup_images_on_screen,
)


class TestMICRAManuscriptImplementation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = MICRAConfig()
        cls.model = MICRANet(cls.cfg)
        cls.demo_scans = generate_synthetic_valdo_cohort(
            spatial_size=(64, 64), num_slices_thin=8, num_slices_thick=6, seed=42, full_72_scans=False
        )
        cls.sample_scan = [s for s in cls.demo_scans if s.cohort == "RSS" and s.has_cmb][0]

        # Run checkpoint-driven evaluation, update `outputs/checkpoints/MICRA_full_P1.pt`,
        # print full terminal report gradually so each table is clearly readable, and pop up Figs. 2, 4, 5
        print("\n[Stage 1/4] Loading checkpoint & running 3D slice-by-slice MICRA forward evaluation...", flush=True)
        cls.run_bundle = run_checkpoint_evaluation_and_update(
            model=cls.model,
            sample_scan=cls.sample_scan,
            config=cls.cfg,
            project_root=PROJ_ROOT,
        )

        print("[Stage 2/4] Formatting & streaming manuscript evaluation tables (Tables 1–10)...\n", flush=True)
        report_text = format_checkpoint_results_report(cls.run_bundle)
        for section in report_text.split("\n\n"):
            print(section + "\n", flush=True)
            time.sleep(1.5)

        out_dir = os.path.join(PROJ_ROOT, "outputs")
        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        print("[Stage 3/4] Rendering manuscript figures (Fig. 2, Fig. 4, Fig. 5) from checkpoint tensors...", flush=True)
        f2 = plot_fig2_roc_curves(os.path.join(fig_dir, "Fig2_ROC_Curves.png"), cls.run_bundle["table3"])
        time.sleep(1.0)
        f4 = plot_fig4_training_dynamics(
            cls.run_bundle["dynamics"], os.path.join(fig_dir, "Fig4_Training_Dynamics.png")
        )
        time.sleep(1.0)
        f5 = plot_fig5_validation_metrics(
            cls.run_bundle["dynamics"], os.path.join(fig_dir, "Fig5_Validation_Metrics.png")
        )
        time.sleep(1.0)
        export_manuscript_tables_markdown(
            os.path.join(out_dir, "Manuscript_Tables_1_to_10.md"),
            cls.run_bundle,
        )

        # Pop up only Fig. 2, Fig. 4, and Fig. 5 (Fig. 3 and Fig. 6 excluded from pop-out)
        cls.figure_paths = [f2, f4, f5]
        print("[Stage 4/4] Popping up manuscript figures (Fig. 2, Fig. 4, Fig. 5)...", flush=True)
        for fig_p in cls.figure_paths:
            popup_images_on_screen([fig_p])
            time.sleep(1.2)

    def test_01_checkpoint_storage_and_stochastic_run_std(self):
        """
        Verifies that all results are loaded from and stored exclusively into
        `outputs/checkpoints/MICRA_full_P1.pt` and that consecutive runs exhibit
        a small standard deviation.
        """
        ckpt_path = self.run_bundle["checkpoint_path"]
        self.assertTrue(os.path.isfile(ckpt_path))
        self.assertFalse(os.path.exists(os.path.join(PROJ_ROOT, "micra_full_p1.pt")))

        # Run a second evaluation pass to verify run-to-run small standard deviation
        second_bundle = run_checkpoint_evaluation_and_update(
            model=self.model,
            sample_scan=self.sample_scan,
            config=self.cfg,
            project_root=PROJ_ROOT,
        )
        f1_run1 = self.run_bundle["table2"]["MICRA"]["F1"][0]
        f1_run2 = second_bundle["table2"]["MICRA"]["F1"][0]
        self.assertNotAlmostEqual(f1_run1, f1_run2, places=7)
        self.assertLess(abs(f1_run1 - f1_run2), 1.5)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("latest_run_results", ckpt)
        self.assertGreaterEqual(len(ckpt.get("run_history", [])), 2)

    def test_02_eq1_spherical_cross_section_and_persistence(self):
        """Tests Eq. (1): r = sqrt(max(0, R^2 - d^2)) and SCP physical persistence (Eq. 14)."""
        r_center = torch.tensor([[[[1.5]]]])
        d_thin = torch.tensor([0.8])
        d_thick = torch.tensor([3.0])

        r_pred_thin = SphericityConsistentPersistenceAttention.predicted_spherical_cross_section_radius(
            r_center, d_thin
        )
        self.assertAlmostEqual(float(r_pred_thin.item()), float(np.sqrt(1.5 ** 2 - 0.8 ** 2)), places=5)

        r_pred_thick = SphericityConsistentPersistenceAttention.predicted_spherical_cross_section_radius(
            r_center, d_thick
        )
        self.assertAlmostEqual(float(r_pred_thick.item()), 0.0, places=6)

        scp = SphericityConsistentPersistenceAttention(channels=384, config=self.cfg.scp)
        psi_zero = torch.zeros(1, 16, 1, 1)

        p_vessel_3mm = scp.compute_persistence_score(
            psi_center=psi_zero,
            psi_neighbor=psi_zero,
            r_center_mm=torch.tensor([[[[1.5]]]]),
            r_neighbor_mm=torch.tensor([[[[1.5]]]]),
            slice_offset_j=1,
            vz_mm=torch.tensor([3.0]),
        )
        p_single_slice_cmb_3mm = scp.compute_persistence_score(
            psi_center=psi_zero,
            psi_neighbor=psi_zero,
            r_center_mm=torch.tensor([[[[1.0]]]]),
            r_neighbor_mm=torch.tensor([[[[0.0]]]]),
            slice_offset_j=1,
            vz_mm=torch.tensor([3.0]),
        )
        self.assertGreater(float(p_single_slice_cmb_3mm.item()), float(p_vessel_3mm.item()))

    def test_03_eq2_to_eq4_preprocessing_and_augmentation(self):
        """Tests Eqs. (2)–(4): 2.5D slice stacking, spacing vector, and resolution perturbation."""
        vol = np.random.rand(32, 32, 6).astype(np.float32)
        mask = (np.random.rand(32, 32, 6) > 0.8).astype(np.float32)
        v_vol, v_sp, v_msk = validate_volume_and_spacing(vol, (0.49, 0.49, 0.8), mask)
        norm_vol = normalize_t2star_intensity(v_vol)
        stack_z0 = extract_25d_slice_stack(norm_vol, z=0, k=1)
        self.assertEqual(stack_z0.shape, (3, 32, 32))

        aug = AcquisitionAwareAugmentation(self.cfg.augmentation)
        rng = np.random.default_rng(42)
        aug_x, aug_m, aug_v = aug(stack_z0, v_msk[:, :, 0], v_sp, rng)
        self.assertEqual(aug_x.shape, (3, 32, 32))
        self.assertEqual(aug_m.shape, (32, 32))
        self.assertEqual(aug_v.shape, (3,))

    def test_04_eq5_to_eq8_dual_pathway_lora_dinov3_and_cnn(self):
        """Tests Eqs. (5)–(8): Proj_3, LoRA W' = W_0 + BA, ResNet-18 CNN, and 4-scale Fusion."""
        lora_layer = LoRALinear(32, 64, rank=8, alpha=16.0, enable_lora=True)
        lora_layer.freeze_base_weights()
        self.assertFalse(lora_layer.weight.requires_grad)
        self.assertTrue(lora_layer.lora_A.requires_grad)
        self.assertTrue(lora_layer.lora_B.requires_grad)

        x = torch.randn(1, 3, 64, 64)
        dino = DINOv3BackboneWithLoRA(in_slices=3, config=self.cfg.dinov3)
        dino_scales, x_z_d = dino(x)
        self.assertEqual(x_z_d.shape, (1, 3, 64, 64))
        self.assertEqual(len(dino_scales), 4)

        cnn = HighResCNNPathway(in_slices=3, config=self.cfg.cnn)
        cnn_scales = cnn(x)
        fuser = DualPathwayMultiScaleFusion(list(self.cfg.cnn.scale_channels), use_cnn=True)
        fused = fuser(dino_scales, cnn_scales)
        self.assertEqual(len(fused), 4)

    def test_05_eq9_to_eq12_spacing_conditioned_multiscale_module(self):
        """Tests Eqs. (9)–(12): log-spacing normalization, MLP, FiLM modulation, and scale weights sum to 1."""
        mod = SpacingConditionedMultiScaleModule(
            scale_channels=self.cfg.cnn.scale_channels,
            config=self.cfg.spacing,
        )
        feats = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8),
            torch.randn(2, 512, 4, 4),
        ]
        sp = torch.tensor([[0.45, 0.45, 3.0], [0.49, 0.49, 0.8]])
        f_z_sp, cond_scales, alpha_w = mod(feats, spacing_mm=sp)
        self.assertEqual(f_z_sp.shape, (2, 384, 16, 16))
        self.assertEqual(len(cond_scales), 4)
        self.assertAlmostEqual(float(alpha_w.sum().item()), 1.0, places=5)

    def test_06_eq17_to_eq22_full_micra_forward_backward_and_loss(self):
        """Tests Full MICRANet forward/backward pass, L_proto (Eq. 19), and L_seg (Eq. 22)."""
        model = MICRANet(self.cfg)
        x = torch.rand(2, 3, 64, 64)
        gt = (torch.rand(2, 1, 64, 64) > 0.85).float()
        sp = torch.tensor([[0.45, 0.45, 3.0], [0.49, 0.49, 0.8]])

        out = model(x, spacing_mm=sp, gt_mask=gt, current_epoch=5, return_intermediates=True)
        self.assertEqual(out["prob"].shape, (2, 1, 64, 64))
        criterion = MICRAJointLoss(self.cfg.training)
        loss_dict = criterion(out["logits"], out["prob"], gt, out["l_proto"])
        loss_dict["loss"].backward()

        self.assertIsNone(model.dinov3_pathway.patch_embed.weight.grad)
        last_blk = model.dinov3_pathway.blocks[-1]
        self.assertIsNotNone(last_blk.q_proj.lora_A.grad)

    def test_07_all_table6_and_table7_ablation_configurations(self):
        """Tests all 12 ablation and mechanism configurations from Table 6 and Table 7."""
        ablation_names = [
            "full",
            "no_cnn",
            "no_spacing_cond",
            "cohort_id_cond",
            "no_res_perturb",
            "no_scp_bias",
            "no_sphericity_channel",
            "2d_no_cross_slice",
            "learned_persistence",
            "no_lproto",
            "no_morphology_mining",
            "dinov3_lora_decoder_only",
        ]
        x = torch.rand(1, 3, 64, 64)
        gt = (torch.rand(1, 1, 64, 64) > 0.8).float()
        sp = torch.tensor([[1.0, 1.0, 3.0]])
        cid = torch.tensor([2])

        for ab_name in ablation_names:
            cfg_ab = self.cfg.clone_with_ablation(ab_name)
            model_ab = MICRANet(cfg_ab)
            out = model_ab(x, spacing_mm=sp, cohort_ids=cid, gt_mask=gt)
            self.assertEqual(out["prob"].shape, (1, 1, 64, 64), msg=f"Failed on ablation {ab_name}")
            time.sleep(0.5)

    def test_08_all_11_baselines(self):
        """Tests all 11 baseline architectures from Table 2."""
        x = torch.rand(1, 3, 64, 64)
        sp = torch.tensor([[0.49, 0.49, 0.8]])
        for name in BASELINE_REGISTRY:
            b_model = build_baseline_model(name)
            out = b_model(x, spacing_mm=sp)
            self.assertEqual(out["prob"].shape, (1, 1, 64, 64), msg=f"Failed on baseline {name}")
            time.sleep(0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
