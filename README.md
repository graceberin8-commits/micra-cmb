# MICRA: Spacing-Conditioned and Physically Consistent Slice-Persistence Learning for Cerebral Microbleed Segmentation and Detection Across Heterogeneous T2\*-Weighted MRI

Complete PyTorch implementation of **MICRA** (*Multi-scale Inter-slice Contextual Representation Architecture*) from the manuscript:

> **MICRA: Spacing-Conditioned and Physically Consistent Slice-Persistence Learning for Cerebral Microbleed Segmentation and Detection Across Heterogeneous T2\*-Weighted MRI**  
> *T. Grace Berin, C. Helen Sulochana, R. Rajesh*

---

## Overview & Key Contributions

Cerebral microbleeds (CMBs) are small, near-spherical hypointense foci ($\le 10\text{ mm}$ diameter, $R_{\max} = 5\text{ mm}$) on T2\*-weighted gradient-recalled echo (GRE) MRI. Whether a CMB appears in adjacent slices depends jointly on its physical radius $R$ and the scan's slice spacing $v_z$:
$$r(d) = \sqrt{\max(0, R^2 - d^2)} \quad \text{[Eq. 1]}$$

MICRA reformulates inter-slice reasoning as a closed-form physical test of consistency with a 3D spherical lesion at the scan's native voxel spacing $v = [v_x, v_y, v_z]$:

1. **2.5D Input Formulation & Continuous Resolution Perturbation (`micra/data/`)**:
   - Neighbouring-slice stack $X_z = \text{Stack}(I_{z-k}, \dots, I_z, \dots, I_{z+k})$ (Eq. 2) paired with physical voxel spacing $v = [v_x, v_y, v_z]$ (Eq. 3).
   - Acquisition-aware augmentation $\tilde{X}_z = T_a(X_z; a)$ (Eq. 4) with continuous resolution perturbation ($s \in [0.5, 2.0]$, $v_z^{\text{thr}} = 2.0\text{ mm}$, $m_{\max} = 0.7$) so spacing cannot act as a proxy for cohort/site identity.
2. **Parameter-Efficient DINOv3 & High-Resolution CNN Dual-Pathway Backbone (`micra/models/lora_dinov3.py`, `cnn_pathway.py`)**:
   - Learnable 3-channel projection $X_z^D = \text{Proj}_3(X_z)$ (Eq. 5).
   - Frozen **DINOv3 ViT-S/16** backbone (384-d, 12 blocks, 6 heads, patch size 16) adapted via **Low-Rank Adaptation (LoRA)** $W' = W_0 + BA$ ($r = 8$) on the query and value projections of the last 6 blocks (Eq. 6).
   - Parallel **ResNet-18 High-Resolution CNN pathway** $\Phi_{\text{CNN}}(X_z; \theta_{\text{CNN}})$ (Eq. 7) fused with DINOv3 across $L = 4$ spatial scales (Eq. 8).
3. **Spacing-Conditioned Multi-Scale Lesion Representation (`micra/models/spacing_conditioning.py`)**:
   - Relative log-spacing normalization $v^* = \log(v / v_{\text{ref}})$ (Eq. 9) and conditioning MLP $e_v = \text{MLP}(v^*; \theta_v)$ (Eq. 10).
   - Multi-scale Feature-wise Linear Modulation (FiLM) $F_l^{\text{SC}} = \gamma_l(e_v) \odot P_l(F_l) + \beta_l(e_v)$ (Eq. 11) and normalized adaptive multi-scale aggregation $F_z^{\text{SP}} = \sum_{l=1}^L \alpha_l F_l^{\text{SC}}$ (Eq. 12).
4. **Sphericity-Consistent Persistence (SCP) Cross-Slice Attention (`micra/models/scp_attention.py`)**:
   - Multi-scale Laplacian-of-Gaussian (LoG) scale-space blob radius estimation ($\Sigma \in [0.5, 2.5]\text{ mm}$) and morphological descriptors $\Psi(F_{z+j}^{\text{SP}})$.
   - Physical persistence score $P_{z,j}(u) = \exp(-\Delta_{z,j}(u) / \tau_p)$ (Eq. 14) biasing multi-head cross-slice attention $\alpha_{z,j}(u)$ (Eqs. 13, 15, 16) and exposing an explicit sphericity map $S_z(u)$ to the decoder.
5. **Mimic-Aware Hard-Negative Prototype Learning (`micra/models/mimic_learning.py`)**:
   - Multi-scale Frangi vesselness $V_i$ and LoG blobness $B_i$ descriptors $d_i^{\text{mimic}} = [V_i, B_i, 1 - S_i]$ (Eq. 17).
   - Hard-negative mining score $h_i = \text{sim}(f_i, c^+) + \lambda_m \text{sim}(d_i^{\text{mimic}}, d^{\text{ref}})$ (Eq. 18) and prototype contrastive objective $\mathcal{L}_{\text{proto}}$ (Eq. 19).
6. **Multi-Scale Segmentation Decoder & 3D Lesion Detection (`micra/models/decoder.py`, `micra/evaluation/`)**:
   - Multi-scale decoder $D_z = \text{Decoder}(F_z^{\text{CTX}}; \theta_d)$ (Eq. 20), pixel probability prediction $\hat{M}_z = \sigma(W_m D_z + b_m)$ (Eq. 21), joint BCE + Dice + Prototype loss (Eq. 22), 3D thresholding (Eq. 23), equivalent spherical diameter $D_{\text{eq}} = (6V/\pi)^{1/3}$, and 5 mm physical center-distance matching.

---

## Repository Structure

```text
micra/
├── configs/
│   └── default_micra.yaml          # Full Section 4.5 hyperparameters
├── micra/
│   ├── config.py                   # Configuration dataclasses & Table 6/7 ablation presets
│   ├── data/
│   │   ├── preprocessing.py        # Eqs. (2)–(3), intensity normalization, P3 resampling
│   │   ├── augmentation.py         # Eq. (4) acquisition-aware augmentation & resolution perturbation
│   │   └── dataset.py              # VALDO Task 2 loader, physics-based 72-scan generator, P1/P2/P3 splits
│   ├── models/
│   │   ├── lora_dinov3.py          # Eqs. (5)–(6) DINOv3 ViT-S/16 + LoRA adaptation
│   │   ├── cnn_pathway.py          # Eqs. (7)–(8) High-resolution ResNet-18 & 4-scale fusion
│   │   ├── spacing_conditioning.py # Eqs. (9)–(12) Spacing-conditioned FiLM & scale aggregation
│   │   ├── scp_attention.py        # Eqs. (1), (13)–(16) Sphericity-Consistent Persistence attention
│   │   ├── mimic_learning.py       # Eqs. (17)–(19) Frangi/LoG descriptors, mining & prototype loss
│   │   ├── decoder.py              # Eqs. (20)–(21) Multi-scale refinement decoder
│   │   ├── micra_net.py            # Complete MICRANet model
│   │   └── baselines.py            # All 11 comparison baselines (Table 2)
│   ├── losses/
│   │   └── segmentation_loss.py    # Eq. (22) BCE + Dice + Prototype joint loss
│   ├── evaluation/
│   │   ├── postprocessing.py       # Eq. (23) 3D thresholding, connected components, D_eq size groups
│   │   ├── metrics.py              # DSC, IoU, 5mm matching, FROC, CPM, AUC, Bootstrap CI, Wilcoxon-Holm
│   │   └── protocols.py            # 3D inference & Table 10 computational cost profiling
│   ├── engine/
│   │   └── trainer.py              # AdamW differential LR, cosine schedule, grad accum, early stopping
│   └── visualization/
│       └── plots.py                # Reproduces Figures 2–6 and Tables 1–10
├── scripts/
│   ├── train.py                    # Train MICRA or any baseline under P1 / P2
│   ├── evaluate.py                 # Evaluate trained checkpoints across P1, P2, P3
│   └── reproduce_manuscript.py     # End-to-end verification, figure & table generation
└── tests/
    └── test_micra_full.py          # Unit & integration test suite covering Eqs. (1)–(23)
```

---

## Quick Start

### 1. Run Unit & Integration Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Reproduce All Manuscript Figures (Figs. 2–6) and Tables (Tables 1–10)
```bash
python scripts/reproduce_manuscript.py
```

### 3. Train MICRA or Any Baseline (P1 Cross-Validation or P2 LOCO)
```bash
# Train Full MICRA under P1 (5-fold CV)
python scripts/train.py --model MICRA --protocol P1 --fold 0 --epochs 100

# Train under P2 (Leave-One-Cohort-Out) holding out SABRE
python scripts/train.py --model MICRA --protocol P2 --held-out-cohort SABRE --epochs 100

# Run any Table 6 / Table 7 ablation (e.g. no_scp_bias, no_cnn, cohort_id_cond, 2d_no_cross_slice)
python scripts/train.py --model MICRA --ablation no_scp_bias --protocol P1
```
