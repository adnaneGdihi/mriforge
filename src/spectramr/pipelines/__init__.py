"""Pipelines Package - Unified Entry Points.

This package provides function-based pipelines that replace
the previous class-based orchestration hierarchy.

Usage:
    from spectramr.pipelines import (
        run_training_pipeline,
        run_inference_pipeline,
        make_model, make_optimizer, make_dataset, make_dataloader,
        run_hpo_grid, run_hpo_random,
        run_ablation_study, run_loss_ablation,
    )

Exports resolve LAZILY (PEP 562), mirroring ``spectramr/__init__.py``.

Why this package cannot import its submodules eagerly
-----------------------------------------------------
``from .infer import run_inference_pipeline`` at module scope made *every*
import of anything under ``spectramr.pipelines`` -- including
``spectramr.pipelines.fit``, which ``spectramr.api`` needs -- drag in
``infer -> inference_factory -> cold_diffusion_inference_strategy ->
models.diffusion.kspace_process``.

That is fatal specifically for OUT-OF-TREE PLUGINS. ``discover_plugins`` runs at
module-import time (``core/metrics/__init__``, ``models/losses/__init__``), so a
plugin doing the documented ``from spectramr.api import register_model`` was
importing this package while ``models.diffusion.kspace_process`` was still
half-built, and got::

    ImportError: cannot import name 'inject_reverse_step_noise' from partially
    initialized module 'spectramr.models.diffusion.kspace_process'

The plugin body was abandoned mid-way, nothing registered, and nothing was
raised. Deferring these imports to first attribute access breaks the cycle:
``spectramr.pipelines.fit`` no longer pulls ``.infer``. (``pipelines/train.py``
does not import ``.infer`` at module level, so ``fit`` -> ``train`` is clean.)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static-only imports so type-checkers see the lazily-exported names
    # (resolved at runtime via ``__getattr__`` below -- zero import cost here).
    from spectramr.pipelines.ablation import (
        run_ablation_study as run_ablation_study,
    )
    from spectramr.pipelines.ablation import (
        run_loss_ablation as run_loss_ablation,
    )
    from spectramr.pipelines.ablation import (
        train_and_score as train_and_score,
    )
    from spectramr.pipelines.hpo import (
        run_hpo_grid as run_hpo_grid,
    )
    from spectramr.pipelines.hpo import (
        run_hpo_random as run_hpo_random,
    )
    from spectramr.pipelines.infer import (
        run_inference_pipeline as run_inference_pipeline,
    )
    from spectramr.pipelines.make import (
        make_dataloader as make_dataloader,
    )
    from spectramr.pipelines.make import (
        make_dataset as make_dataset,
    )
    from spectramr.pipelines.make import (
        make_model as make_model,
    )
    from spectramr.pipelines.make import (
        make_optimizer as make_optimizer,
    )
    from spectramr.pipelines.train import (
        run_training_pipeline as run_training_pipeline,
    )

# NOTE: ``KoopmanAdvectionPipeline`` / ``QuantumImplicitNeRFPipeline`` were
# removed (2026-06-11). They were redundant model+loss composites — their model
# cores are already registered (``neural_advection`` / ``dynamic_mr_nerf``, both
# ``training_mode="virtual_fiducial"``) and their losses too
# (``koopman_linearity`` / ``hyperelastic_jacobian``), so they ran via the
# standard ``run_training_pipeline`` + VF strategy with no special command.
# Express the methods through ``model_type`` + ``objectives`` instead.

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Training and inference
    "run_training_pipeline": ("spectramr.pipelines.train", "run_training_pipeline"),
    "run_inference_pipeline": ("spectramr.pipelines.infer", "run_inference_pipeline"),
    # Model/optimizer/data creation
    "make_model": ("spectramr.pipelines.make", "make_model"),
    "make_optimizer": ("spectramr.pipelines.make", "make_optimizer"),
    "make_dataset": ("spectramr.pipelines.make", "make_dataset"),
    "make_dataloader": ("spectramr.pipelines.make", "make_dataloader"),
    # Hyperparameter optimization
    "run_hpo_grid": ("spectramr.pipelines.hpo", "run_hpo_grid"),
    "run_hpo_random": ("spectramr.pipelines.hpo", "run_hpo_random"),
    # Ablation studies
    "run_ablation_study": ("spectramr.pipelines.ablation", "run_ablation_study"),
    "run_loss_ablation": ("spectramr.pipelines.ablation", "run_loss_ablation"),
    "train_and_score": ("spectramr.pipelines.ablation", "train_and_score"),
}

__all__ = [
    "make_dataloader",
    "make_dataset",
    "make_model",
    "make_optimizer",
    "run_ablation_study",
    "run_hpo_grid",
    "run_hpo_random",
    "run_inference_pipeline",
    "run_loss_ablation",
    "run_training_pipeline",
    "train_and_score",
]


def __getattr__(name: str) -> Any:
    """Resolve a pipeline export lazily (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'spectramr.pipelines' has no attribute {name!r}")
    module_path, attr = target
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
