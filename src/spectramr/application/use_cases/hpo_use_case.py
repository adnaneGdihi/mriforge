"""HPO Use Case
============

Use case for running hyperparameter optimization operations.

This module implements the HPOUseCase class, which serves as the main
entry point for HPO operations in the Clean Architecture. It orchestrates
the HPO process by delegating to the HPOCoordinator while providing
a clean interface for the application layer.

The use case follows the Command pattern, accepting a request dataclass
and returning a response dataclass with HPO results.

Previous behavior (now fixed)
-----------------------------
Earlier versions of this use case hardcoded ``n_trials=10``, ``device="cpu"``,
and ``objective_metric="fid"`` regardless of caller input — and the
``HPOCoordinator`` it delegated to was a stub returning placeholder
results. The coordinator now actually runs Optuna via
``EnhancedHPOptimizer``, and this use case forwards every HPO knob from
the ``HPORequest`` straight through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.coordination.hpo_coordinator import HPOCoordinator
from spectramr.infrastructure.logging import LoggingService

logger = logging.getLogger(__name__)


@dataclass
class HPORequest:
    """Request for an HPO run.

    Attributes:
        config_path: Path to the base training YAML each trial mutates.
        model_types: List of registered ``model_type`` names to optimize
            (one Optuna study per type).
        n_trials: Number of trials per model.
        objective_metric: CSV column to optimize. Auto-detects whether
            to maximize (PSNR, SSIM, ...) or minimize (loss, error, ...).
        device: Requested device for the trainer subprocess. **Currently
            accepted and not forwarded** -- ``HPOCoordinator.execute_hpo``
            binds it and never reads it, so the trials inherit whatever
            the base YAML declares (issue filed; pitfall #15). Kept as a
            declared knob rather than dropped, because wiring it is the
            smaller fix than removing the flag (#16, prefer wiring).
        max_iter_per_trial: Truncated training budget per trial.
        sampler_type: Optuna sampler — ``tpe`` (default), ``cmaes``, ``nsga2``.
        pruner_type: Optuna pruner — ``hyperband`` (default), ``median``,
            ``successive_halving``, ``threshold``.
        storage_url: Optuna storage URL (e.g. ``sqlite:///path.db``) for
            resumable / parallelizable studies. None = in-memory.
        output_dir: Where per-trial subdirectories are written. If empty,
            defaults to ``<config.training.output_dir>/hpo``.
        cost_weight: If > 0, switch to multi-objective optimization
            (objective_metric, training_time).
        max_training_time: Soft per-trial budget in hours (advisory).
        enable_cost_optimization: If True, treat as multi-objective.
    """

    config_path: str
    model_types: list[str]
    n_trials: int = 50
    objective_metric: str = "val_loss"
    device: str | None = None
    max_iter_per_trial: int = 30000
    sampler_type: str = "tpe"
    pruner_type: str = "hyperband"
    storage_url: str | None = None
    output_dir: str = ""
    cost_weight: float = 0.0
    max_training_time: float | None = None
    enable_cost_optimization: bool = False
    # Search space — what HPO is actually allowed to vary per trial. Without
    # this the HPO is a no-op (every trial runs the base config unchanged).
    # Mutually-exclusive: pass at most one of (search_preset, search_space_path).
    # If both are None and search_space_dict is None too, HPOCoordinator logs
    # a warning and runs identical trials.
    search_preset: str | None = None  # name in spectramr.pipelines.hpo_search_spaces.PRESETS
    search_space_path: str | None = None  # path to a YAML search-space spec
    search_space_dict: dict | None = None  # programmatic spec (overrides preset/path)


@dataclass
class HPOResponse:
    """Response from an HPO run.

    Attributes:
        results: Per-model results dict (best_params / best_value /
            n_trials / pareto_front when multi-objective).
        success: True if HPO completed for all model types.
    """

    results: dict[str, Any] = field(default_factory=dict)
    success: bool = False


class HPOUseCase:
    """Use case for hyperparameter optimization.

    Acts as a thin facade over ``HPOCoordinator``. The use case is
    responsible for:

    1. Validating the request.
    2. Resolving config-derived defaults (e.g. output directory).
    3. Forwarding all knobs to the coordinator.
    4. Wrapping success / failure into an ``HPOResponse``.

    Dependencies:
        - LoggingService (DI-resolved by the bootstrap)
    """

    def __init__(self, logging_service: LoggingService):
        self.logging_service = logging_service

    def execute(self, request: HPORequest) -> HPOResponse:
        """Execute the HPO run end-to-end.

        Returns ``HPOResponse(success=False)`` on any uncaught exception so
        the CLI / pipeline can react without crashing.
        """
        self._validate_request(request)
        logger.info("Starting HPO use case")

        try:
            self.logging_service.log_info(
                f"Starting HPO for model types: {request.model_types} "
                f"(n_trials={request.n_trials}, sampler={request.sampler_type}, "
                f"pruner={request.pruner_type})"
            )

            result = self._execute_hpo(request)

            self.logging_service.log_info(
                f"HPO completed for {len(request.model_types)} model types"
            )
            return HPOResponse(results=result, success=True)

        except Exception as e:
            logger.exception(f"HPO use case failed: {e}")
            self.logging_service.log_error(f"HPO failed: {e}")
            return HPOResponse(results={}, success=False)

    def _validate_request(self, request: HPORequest) -> None:
        if not request.config_path:
            raise ValueError("config_path must be provided")
        if not request.model_types:
            raise ValueError("model_types must be a non-empty list")
        if request.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {request.n_trials}")
        if request.max_iter_per_trial < 1:
            raise ValueError(f"max_iter_per_trial must be >= 1, got {request.max_iter_per_trial}")

    def _execute_hpo(self, request: HPORequest) -> dict[str, Any]:
        """Forward every HPO knob from the request to the coordinator.

        Resolves ``output_dir`` from the request, falling back to the base
        config's ``training.output_dir`` + ``/hpo`` so users don't need
        to specify it explicitly when the YAML already defines a sensible
        location. Resolves the search space from the request's preset name
        or YAML path (programmatic dict wins if all three are set).
        """
        config = TrainingSettings.from_yaml(request.config_path)

        # Output dir resolution: explicit request > config-derived default.
        output_dir = request.output_dir
        if not output_dir:
            base_out = config.training.output_dir or "experiments/results/hpo"
            output_dir = f"{base_out}/hpo"

        # Search-space resolution. Precedence: dict > YAML path > preset.
        from spectramr.pipelines.hpo_search_spaces import (
            SearchSpace,
            load_preset,
        )

        search_space: SearchSpace | None = None
        if request.search_space_dict:
            search_space = SearchSpace.from_dict(request.search_space_dict)
        elif request.search_space_path:
            search_space = SearchSpace.from_yaml(request.search_space_path)
        elif request.search_preset:
            search_space = load_preset(request.search_preset)

        # data_root retained for backward compat; coordinator currently
        # only uses output_dir but the signature accepts both.
        data_root = config.data.source.root or ""

        # ``HPOCoordinator`` lives in infrastructure and can't import
        # pipeline code (layer-direction rule + audit 17 F5). The
        # application layer is allowed to depend on both, so we wire
        # the concrete subprocess-based objective factory in here.
        from spectramr.pipelines.hpo import subprocess_training_objective

        coordinator = HPOCoordinator()
        return coordinator.execute_hpo(
            model_types=request.model_types,
            input_lr_dir=data_root,
            input_hr_dir=data_root,
            output_dir=output_dir,
            n_trials=request.n_trials,
            config_path=request.config_path,
            device=request.device,
            objective_metric=request.objective_metric,
            cost_weight=request.cost_weight,
            max_training_time=request.max_training_time,
            enable_cost_optimization=request.enable_cost_optimization,
            storage_url=request.storage_url,
            sampler_type=request.sampler_type,
            pruner_type=request.pruner_type,
            max_iter_per_trial=request.max_iter_per_trial,
            search_space=search_space,
            objective_factory=subprocess_training_objective,
        )
