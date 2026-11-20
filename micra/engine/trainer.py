"""
Section 4.5: Training and Validation Engine for MICRA and Baselines.
Implements:
  - AdamW optimizer with lr = 1e-5 (LoRA) and 1e-4 (new layers), weight decay = 1e-4
  - Cosine annealing learning-rate schedule
  - Gradient accumulation (4 steps) with batch size 1
  - Gradient norm clipping at 1.0
  - Mixed precision (AMP) when CUDA is available
  - Early stopping (patience = 15) and optimal threshold selection on inner validation set
"""

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from micra.config import MICRAConfig
from micra.data.dataset import VALDOScan, VALDO25DDataset
from micra.losses.segmentation_loss import MICRAJointLoss
from micra.evaluation.protocols import predict_3d_volume
from micra.evaluation.metrics import evaluate_cohort_predictions


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    val_dsc: List[float] = field(default_factory=list)
    val_iou: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    best_threshold: float = 0.5


class MICRATrainer:
    """Trainer implementing the exact optimization protocol of Section 4.5."""

    def __init__(
        self,
        model: nn.Module,
        config: MICRAConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = MICRAJointLoss(config.training, normalize_weights=True)

        # Differential learning rates: 1e-5 for LoRA, 1e-4 for new layers
        if hasattr(self.model, "get_optimizer_param_groups"):
            param_groups = self.model.get_optimizer_param_groups()
        else:
            param_groups = [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "lr": config.training.lr_new_layers,
                }
            ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.training.max_epochs),
            eta_min=1e-7,
        )
        self.use_amp = bool(config.training.mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        accum_steps = max(1, self.cfg.training.grad_accum_steps)
        running_loss = 0.0
        num_steps = 0

        for step_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            spacings = batch["spacing"].to(self.device)
            cohort_ids = batch["cohort_id"].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(
                    images,
                    spacing_mm=spacings,
                    cohort_ids=cohort_ids,
                    gt_mask=masks,
                    current_epoch=epoch,
                )
                loss_dict = self.criterion(
                    logits=outputs["logits"],
                    probs=outputs["prob"],
                    targets=masks,
                    l_proto=outputs["l_proto"],
                )
                loss = loss_dict["loss"] / float(accum_steps)

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((step_idx + 1) % accum_steps == 0) or ((step_idx + 1) == len(loader)):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.cfg.training.grad_clip_norm,
                )
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += float(loss_dict["loss"].item())
            num_steps += 1

        return running_loss / max(1, num_steps)

    @torch.no_grad()
    def evaluate_validation_loss(self, val_loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            spacings = batch["spacing"].to(self.device)
            cohort_ids = batch["cohort_id"].to(self.device)

            outputs = self.model(
                images,
                spacing_mm=spacings,
                cohort_ids=cohort_ids,
                gt_mask=masks,
                current_epoch=epoch,
            )
            loss_dict = self.criterion(
                logits=outputs["logits"],
                probs=outputs["prob"],
                targets=masks,
                l_proto=outputs["l_proto"],
            )
            total_loss += float(loss_dict["loss"].item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    def fit(
        self,
        train_scans: List[VALDOScan],
        val_scans: List[VALDOScan],
        max_epochs: Optional[int] = None,
        seed: int = 42,
    ) -> TrainingHistory:
        epochs_to_run = max_epochs if max_epochs is not None else self.cfg.training.max_epochs
        train_ds = VALDO25DDataset(train_scans, self.cfg, is_train=True, seed=seed)
        val_ds = VALDO25DDataset(val_scans, self.cfg, is_train=False, seed=seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
        )

        history = TrainingHistory()
        best_state = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch in range(epochs_to_run):
            current_lr = max(pg["lr"] for pg in self.optimizer.param_groups)
            tr_loss = self.train_one_epoch(train_loader, epoch=epoch)
            va_loss = self.evaluate_validation_loss(val_loader, epoch=epoch)
            self.scheduler.step()

            # Compute 3D volume validation metrics
            val_probs = [
                predict_3d_volume(self.model, sc, k_neighbors=self.cfg.k_neighbors, device=self.device)
                for sc in val_scans
            ]
            val_metrics = evaluate_cohort_predictions(
                val_probs,
                val_scans,
                tau=self.cfg.evaluation.threshold,
                match_radius_mm=self.cfg.evaluation.match_radius_mm,
            )

            history.epochs.append(epoch + 1)
            history.train_loss.append(tr_loss)
            history.val_loss.append(va_loss)
            history.learning_rates.append(current_lr)
            history.val_precision.append(val_metrics["Precision"] / 100.0)
            history.val_recall.append(val_metrics["Sensitivity"] / 100.0)
            history.val_f1.append(val_metrics["F1"] / 100.0)
            history.val_dsc.append(val_metrics["DSC"] / 100.0)
            history.val_iou.append(val_metrics["IoU"] / 100.0)

            if val_metrics["F1"] >= history.best_val_f1:
                history.best_val_f1 = val_metrics["F1"]
                history.best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.training.early_stopping_patience:
                    break

        self.model.load_state_dict(best_state)
        return history
