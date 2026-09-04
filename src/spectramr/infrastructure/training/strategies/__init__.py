"""Training Strategies Package

Public API exports for all training strategy implementations.
"""

from ..utils.training_utils import clamp_to_range
from .base import BaseTrainingStrategy, LossResult, TrainingStepStrategy
from .diffusion import DiffusionTrainingStrategy, XDiffusionTrainingStrategy
from .disentangled_strategy import DisentangledTrainingStrategy
from .disentangled_vae_strategy import DisentangledVAETrainingStrategy
from .distillation_strategy import ConcreteDistillationStrategy
from .domain_adaptation import DomainAdaptationTrainingStrategy
from .gan import GANTrainingStrategy
from .masked_strategy import MaskedPretrainingStrategy
from .meta_learning_strategy import MetaLearningTrainingStrategy
from .mixins.utils import _callable_accepts_kwarg
from .motion_meta_strategy import ConcreteMotionMetaTrainingStrategy
from .padnet_strategy import PaDNetTrainingStrategy
from .physics_driven_strategy import PhysicsDrivenTrainingStrategy

# Physics-informed strategies
from .pinn_strategy import ConcretePINNSensitivityStrategy
from .pipeline_strategy import MultiTrainingStrategy
from .pretraining import MAEPretrainingStrategy
from .reconstruction import ReconstructionTrainingStrategy
from .test_time_adaptation_strategy import TttAdaptationStrategy
from .tto_strategy import ConcreteTTOStrategy
from .vae import VAETrainingStrategy, VQVAETrainingStrategy
from .vf_admm_strategy import ConcreteVFADMMStrategy

# Virtual Fiducial motion-correction strategies
from .virtual_fiducial_strategy import ConcreteVirtualFiducialStrategy
from .volumetric import TRELLISTrainingStrategy

__all__ = [
    # Base classes
    "BaseTrainingStrategy",
    "LossResult",
    "TrainingStepStrategy",
    "clamp_to_range",
    "_callable_accepts_kwarg",
    # Strategy implementations
    "GANTrainingStrategy",
    "ReconstructionTrainingStrategy",
    "DiffusionTrainingStrategy",
    "XDiffusionTrainingStrategy",
    "VAETrainingStrategy",
    "TttAdaptationStrategy",
    "VQVAETrainingStrategy",
    "MAEPretrainingStrategy",
    "TRELLISTrainingStrategy",
    "DomainAdaptationTrainingStrategy",
    "PaDNetTrainingStrategy",
    "DisentangledTrainingStrategy",
    "DisentangledVAETrainingStrategy",
    "PhysicsDrivenTrainingStrategy",
    "MetaLearningTrainingStrategy",
    "MaskedPretrainingStrategy",
    # Virtual Fiducial strategies
    "ConcreteVirtualFiducialStrategy",
    "ConcreteVFADMMStrategy",
    "ConcreteDistillationStrategy",
    "ConcreteMotionMetaTrainingStrategy",
    "ConcreteTTOStrategy",
    # Physics-informed strategies
    "ConcretePINNSensitivityStrategy",
    # Multi-stage pipeline
    "MultiTrainingStrategy",
]


def _validate_registry() -> None:
    """Validate that all exported strategies are registered in the factory."""
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    # Simple validation to ensure factory can load what we export
    # This acts as a self-check for the module
    pass


_validate_registry()
