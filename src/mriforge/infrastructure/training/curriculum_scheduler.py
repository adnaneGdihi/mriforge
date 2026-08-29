"""Curriculum Learning Scheduler for Motion Correction Models.

Implements 3-phase alternating optimization to prevent joint divergence:
1. Phase 1: Freeze motion, train encoder/decoder
2. Phase 2: Freeze network, optimize motion parameters
3. Phase 3: Joint fine-tuning

Reference: Kuklisova-Murgasova et al., MIA 2012 (adapted for deep learning)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phases for curriculum learning."""

    VOLUME_ONLY = "volume_only"  # Phase 1: Train volume, freeze motion
    MOTION_ONLY = "motion_only"  # Phase 2: Train motion, freeze volume
    JOINT = "joint"  # Phase 3: Train both


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""

    phase: TrainingPhase
    epochs: int
    learning_rate: float
    description: str


class CurriculumScheduler:
    """3-Phase Curriculum Learning Scheduler.

    For NeSVoR and similar slice-to-volume motion correction models.
    Decouples the ill-posed inverse problem into solvable sub-problems.

    Default schedule:
    - Phase 1 (40%): Volume reconstruction (identity motion)
    - Phase 2 (30%): Motion estimation (frozen volume)
    - Phase 3 (30%): Joint fine-tuning
    """

    def __init__(
        self,
        total_epochs: int,
        phase_fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
        base_lr: float = 1e-4,
        motion_lr_scale: float = 0.1,
        joint_lr_scale: float = 0.5,
    ):
        """
        Args:
            total_epochs: Total training epochs
            phase_fractions: Fraction of epochs for (volume, motion, joint)
            base_lr: Base learning rate for volume training
            motion_lr_scale: LR multiplier for motion-only phase
            joint_lr_scale: LR multiplier for joint phase
        """
        assert sum(phase_fractions) == 1.0, "Phase fractions must sum to 1.0"

        self.total_epochs = total_epochs
        self.base_lr = base_lr

        # Compute epoch boundaries
        vol_epochs = int(total_epochs * phase_fractions[0])
        mot_epochs = int(total_epochs * phase_fractions[1])
        joint_epochs = total_epochs - vol_epochs - mot_epochs

        self.phases: list[PhaseConfig] = [
            PhaseConfig(
                phase=TrainingPhase.VOLUME_ONLY,
                epochs=vol_epochs,
                learning_rate=base_lr,
                description="Phase 1: Training volume representation (motion frozen)",
            ),
            PhaseConfig(
                phase=TrainingPhase.MOTION_ONLY,
                epochs=mot_epochs,
                learning_rate=base_lr * motion_lr_scale,
                description="Phase 2: Optimizing motion parameters (volume frozen)",
            ),
            PhaseConfig(
                phase=TrainingPhase.JOINT,
                epochs=joint_epochs,
                learning_rate=base_lr * joint_lr_scale,
                description="Phase 3: Joint fine-tuning",
            ),
        ]

        self._current_phase_idx = 0
        self._epochs_in_phase = 0

    def get_phase(self, epoch: int) -> PhaseConfig:
        """Get training phase for given epoch."""
        cumulative = 0
        for phase_cfg in self.phases:
            cumulative += phase_cfg.epochs
            if epoch < cumulative:
                return phase_cfg
        return self.phases[-1]  # Default to last phase

    def configure_model(
        self,
        model: nn.Module,
        epoch: int,
        volume_params: list[str] | None = None,
        motion_params: list[str] | None = None,
    ) -> tuple[Iterator[nn.Parameter], float]:
        """Configure model parameters based on current phase.

        Args:
            model: The model to configure
            epoch: Current epoch
            volume_params: Parameter name patterns for volume component
            motion_params: Parameter name patterns for motion component

        Returns:
            (parameters_to_train, learning_rate)
        """
        phase_cfg = self.get_phase(epoch)

        # Default parameter patterns
        if volume_params is None:
            volume_params = ["inr", "hash_grid", "volume", "decoder"]
        if motion_params is None:
            motion_params = ["deform", "motion", "translation", "rotation"]

        def matches_patterns(name: str, patterns: list[str]) -> bool:
            """matches_patterns.

            Args:
                name (str): Description.
                patterns (list[str]): Description.
            Returns:
                bool: Description.
            """
            return any(p in name.lower() for p in patterns)

        # Configure requires_grad based on phase
        trainable_params = []

        for name, param in model.named_parameters():
            is_volume = matches_patterns(name, volume_params)
            is_motion = matches_patterns(name, motion_params)

            if phase_cfg.phase == TrainingPhase.VOLUME_ONLY:
                param.requires_grad = is_volume or (not is_motion)
            elif phase_cfg.phase == TrainingPhase.MOTION_ONLY:
                param.requires_grad = is_motion
            else:  # JOINT
                param.requires_grad = True

            if param.requires_grad:
                trainable_params.append(param)

        logger.info(
            f"[CURRICULUM] Epoch {epoch}: {phase_cfg.phase.value} | "
            f"LR={phase_cfg.learning_rate:.2e} | "
            f"Trainable params: {sum(p.numel() for p in trainable_params):,}"
        )

        return iter(trainable_params), phase_cfg.learning_rate

    def on_phase_change(self, old_phase: TrainingPhase, new_phase: TrainingPhase) -> None:
        """Callback for phase transitions."""
        logger.info(f"[CURRICULUM] Phase transition: {old_phase.value} → {new_phase.value}")

    def step(self, epoch: int, model: nn.Module) -> None:
        """Update curriculum state for new epoch."""
        old_phase = self.get_phase(max(0, epoch - 1))
        new_phase = self.get_phase(epoch)

        if old_phase.phase != new_phase.phase:
            self.on_phase_change(old_phase.phase, new_phase.phase)


class MotionFreezeCallback:
    """Callback to freeze/unfreeze motion parameters during training."""

    def __init__(self, motion_patterns: list[str] = None):
        """__init__.

        Args:
            motion_patterns (list[str]): Description.
        """
        self.motion_patterns = motion_patterns or ["deform", "motion", "translation"]

    def freeze_motion(self, model: nn.Module) -> None:
        """Freeze motion-related parameters."""
        for name, param in model.named_parameters():
            if any(p in name.lower() for p in self.motion_patterns):
                param.requires_grad = False

    def unfreeze_motion(self, model: nn.Module) -> None:
        """Unfreeze motion-related parameters."""
        for name, param in model.named_parameters():
            if any(p in name.lower() for p in self.motion_patterns):
                param.requires_grad = True

    def reset_motion_to_identity(self, model: nn.Module) -> None:
        """Reset motion parameters to identity transformation."""
        for name, param in model.named_parameters():
            if any(p in name.lower() for p in self.motion_patterns):
                if "bias" in name or "translation" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    # Identity initialization for weight matrices
                    if param.dim() == 2 and param.size(0) == param.size(1):
                        nn.init.eye_(param)


__all__ = [
    "CurriculumScheduler",
    "MotionFreezeCallback",
    "PhaseConfig",
    "TrainingPhase",
]
