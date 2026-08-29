"""Meta-Learning for MRI Reconstruction (Meta-VarNet).

This package implements Model-Agnostic Meta-Learning (MAML) for rapidly
adapting reconstruction networks to new accelerations or anatomies.

Key Components:
- MetaVarNet: Variation Network compatible with functional parameter updates.
- MAML: The meta-learning training loop (Approximate Second-Order).
"""

from .maml import MAML
from .meta_varnet import MetaVarNet

__all__ = ["MAML", "MetaVarNet"]
