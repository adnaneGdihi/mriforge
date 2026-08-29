"""Pipelines Package - Unified Entry Points.

This package provides function-based pipelines that replace
the previous class-based orchestration hierarchy.

Usage:
    from mriforge.pipelines import (
        run_training_pipeline,
        run_inference_pipeline,
        make_model, make_optimizer, make_dataset, make_dataloader,
        run_hpo_grid, run_hpo_random,
        run_ablation_study, run_loss_ablation,
    )
"""

from .ablation import run_ablation_study, run_loss_ablation, train_and_score
from .hpo import run_hpo_grid, run_hpo_random
from .infer import run_inference_pipeline
from .make import make_dataloader, make_dataset, make_model, make_optimizer
from .train import run_training_pipeline

# NOTE: ``KoopmanAdvectionPipeline`` / ``QuantumImplicitNeRFPipeline`` were
# removed (2026-06-11). They were redundant model+loss composites — their model
# cores are already registered (``neural_advection`` / ``dynamic_mr_nerf``, both
# ``training_mode="virtual_fiducial"``) and their losses too
# (``koopman_linearity`` / ``hyperelastic_jacobian``), so they ran via the
# standard ``run_training_pipeline`` + VF strategy with no special command.
# Express the methods through ``model_type`` + ``objectives`` instead.

__all__ = [
    # Training and inference
    "run_training_pipeline",
    "run_inference_pipeline",
    # Model/optimizer/data creation
    "make_model",
    "make_optimizer",
    "make_dataset",
    "make_dataloader",
    # Hyperparameter optimization
    "run_hpo_grid",
    "run_hpo_random",
    # Ablation studies
    "run_ablation_study",
    "run_loss_ablation",
    "train_and_score",
]
