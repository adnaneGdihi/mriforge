"""DeepSpeed ZeRO backend.

Named ``deepspeed_backend`` rather than ``deepspeed`` so that
``from mriforge.infrastructure.distributed import deepspeed`` can never be
mistaken for the upstream package at a call site.
"""

from .config_builder import DERIVED_KEYS, build_deepspeed_config
from .runtime import (
    DeepSpeedStepPolicy,
    deepspeed_available,
    initialize_deepspeed_engine,
    require_deepspeed,
)

__all__ = [
    "DERIVED_KEYS",
    "DeepSpeedStepPolicy",
    "build_deepspeed_config",
    "deepspeed_available",
    "initialize_deepspeed_engine",
    "require_deepspeed",
]
