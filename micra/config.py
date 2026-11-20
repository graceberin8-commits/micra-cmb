"""
Configuration dataclasses containing all exact hyperparameters from Section 4.5
of the MICRA manuscript.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DINOv3Config:
    arch: str = "vit_small_patch16"
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    patch_size: int = 16
    mlp_ratio: float = 4.0
    freeze_backbone: bool = True
    lora_rank: int = 8  # r = 8 (grid searched over {4, 8, 16})
    lora_alpha: float = 16.0
    lora_last_n_blocks: int = 6  # LoRA on Q and V of last 6 blocks
    extract_layers: Tuple[int, ...] = (3, 6, 9, 12)


@dataclass
class CNNPathwayConfig:
    backbone: str = "resnet18"
    base_channels: int = 64
    scale_channels: Tuple[int, ...] = (64, 128, 256, 512)
    enabled: bool = True


@dataclass
class SpacingConditioningConfig:
    mode: str = "spacing"  # "spacing", "cohort_id", or "none"
    reference_spacing_mm: Tuple[float, float, float] = (0.5, 0.5, 1.0)
    embed_dim: int = 128
    num_scales: int = 4
    unified_channels: int = 384
    num_cohorts: int = 3  # SABRE, RSS, ALFA


@dataclass
class SCPAttentionConfig:
    enabled: bool = True
    # Modes for Table 6 & Table 7 mechanism analysis:
    # "scp_full": closed-form spherical persistence bias + sphericity channel S_z
    # "scp_bias_only": closed-form spherical persistence bias without S_z channel
    # "learned_persistence": exp(-||Psi(F_z) - Psi(F_{z+j})||_1 / tau_p)
    # "similarity_only": standard cross-slice attention (lambda_p = 0)
    # "2d": no cross-slice attention
    mode: str = "scp_full"
    num_heads: int = 6
    head_dim: int = 64  # 6 heads * 64-d = 384-d
    r_max_mm: float = 5.0  # R_max = 5 mm (10 mm diameter clinical bound)
    tau_p_mm: float = 1.5  # tau_p = 1.5 mm
    lambda_p: float = 0.25  # lambda_p = 0.25
    sigma_min_mm: float = 0.5  # Sigma = 0.5-2.5 mm
    sigma_max_mm: float = 2.5
    num_scales_log: int = 5
    delta_mm: float = 0.8  # delta = 0.8 mm
    expose_sphericity_channel: bool = True
    eps: float = 1e-6


@dataclass
class MimicLearningConfig:
    enabled: bool = True
    beta_b: float = 0.4  # beta_b = 0.4 (Frangi blobness/tubularity trade-off)
    num_candidates: int = 21  # N_c = 21
    warmup_epochs: int = 10  # E_w = 10
    prototype_momentum: float = 0.1  # eta_c = 0.1
    gt_exclusion_mm: float = 2.0  # delta_g = 2.0 mm
    lambda_m: float = 1.0  # lambda_m = 1.0
    mu: float = 0.5  # mu = 0.5
    memory_bank_size: int = 128  # K_h = 128
    num_positives_per_step: int = 8  # N_q = 8
    hard_negative_ratio: float = 0.5  # K_n = 0.5
    negative_sampling_ratio: float = 0.5  # N_r = 0.5
    tau_c: float = 0.5  # tau_c = 0.5
    feature_dim: int = 128


@dataclass
class AugmentationConfig:
    enabled: bool = True
    resolution_perturbation: bool = True
    scale_range: Tuple[float, float] = (0.5, 2.0)  # s in [0.5, 2.0]
    vz_threshold_mm: float = 2.0  # v_z^thr = 2.0 mm
    max_perturbation_prob: float = 0.7  # m_max = 0.7
    flip_prob: float = 0.5
    rotate_deg: float = 15.0
    noise_std: float = 0.03
    contrast_gamma_range: Tuple[float, float] = (0.85, 1.15)


@dataclass
class TrainingConfig:
    lr_lora: float = 1e-5
    lr_new_layers: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    max_epochs: int = 100
    early_stopping_patience: int = 15
    batch_size: int = 1
    grad_accum_steps: int = 4
    mixed_precision: bool = True
    lambda_bce: float = 0.05  # lambda_BCE = 0.05
    lambda_dice: float = 256.0  # lambda_Dice = 256
    lambda_proto: float = 8.0  # lambda_proto = 8
    inner_val_fraction: float = 0.10


@dataclass
class EvaluationConfig:
    threshold: float = 0.5  # tau
    v_min: int = 50  # V_min = 50 (minimum lesion physical/sub-voxel volume score)
    min_voxels: int = 2
    match_radius_mm: float = 5.0  # rho_m = 5.0 mm
    small_diameter_mm: float = 3.0  # D_eq < 3 mm
    large_diameter_mm: float = 5.0  # D_eq > 5 mm
    froc_fp_rates: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    bootstrap_resamples: int = 2000
    num_folds: int = 5
    seeds: Tuple[int, ...] = (42, 123, 456)


@dataclass
class MICRAConfig:
    k_neighbors: int = 1  # k = 1 -> 3 adjacent slices (z-1, z, z+1)
    r_max_mm: float = 5.0
    decoder_channels: Tuple[int, ...] = (384, 256, 128, 64)
    full_parameter_scale: bool = False  # Set True for 89.7M parameter full capacity mode
    dinov3: DINOv3Config = field(default_factory=DINOv3Config)
    cnn: CNNPathwayConfig = field(default_factory=CNNPathwayConfig)
    spacing: SpacingConditioningConfig = field(default_factory=SpacingConditioningConfig)
    scp: SCPAttentionConfig = field(default_factory=SCPAttentionConfig)
    mimic: MimicLearningConfig = field(default_factory=MimicLearningConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def num_input_slices(self) -> int:
        return 2 * self.k_neighbors + 1

    def clone_with_ablation(self, ablation_name: str) -> "MICRAConfig":
        """
        Creates a modified configuration corresponding to any row of Table 6
        (Ablation study) or Table 7 (Mechanism analysis).
        """
        import copy
        cfg = copy.deepcopy(self)
        if ablation_name == "full":
            return cfg
        elif ablation_name == "no_cnn":
            cfg.cnn.enabled = False
        elif ablation_name == "no_spacing_cond":
            cfg.spacing.mode = "none"
        elif ablation_name == "cohort_id_cond":
            cfg.spacing.mode = "cohort_id"
        elif ablation_name == "no_res_perturb":
            cfg.augmentation.resolution_perturbation = False
        elif ablation_name == "no_scp_bias":
            cfg.scp.lambda_p = 0.0
            cfg.scp.mode = "similarity_only"
        elif ablation_name == "no_sphericity_channel":
            cfg.scp.expose_sphericity_channel = False
            cfg.scp.mode = "scp_bias_only"
        elif ablation_name == "2d_no_cross_slice":
            cfg.scp.enabled = False
            cfg.scp.mode = "2d"
        elif ablation_name == "learned_persistence":
            cfg.scp.mode = "learned_persistence"
        elif ablation_name == "no_lproto":
            cfg.training.lambda_proto = 0.0
            cfg.mimic.enabled = False
        elif ablation_name == "no_morphology_mining":
            cfg.mimic.lambda_m = 0.0
        elif ablation_name == "dinov3_lora_decoder_only":
            cfg.cnn.enabled = False
            cfg.spacing.mode = "none"
            cfg.scp.enabled = False
            cfg.scp.mode = "2d"
            cfg.training.lambda_proto = 0.0
            cfg.mimic.enabled = False
        else:
            raise ValueError(f"Unknown ablation configuration: {ablation_name}")
        return cfg
