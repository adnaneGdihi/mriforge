"""HPO Coordinator — wires the use-case layer to the actual optimizer.

The previous implementation was a stub returning placeholder results
regardless of input, which made the entire HPO advertised surface (use
case, CLI, coordinator) non-functional end-to-end. This module now
actually delegates to ``EnhancedHPOptimizer`` (the existing Optuna-backed
implementation in ``src/models/tuning/enhanced_hyperparameter_tuning.py``)
and to the subprocess-based trial runner in ``src/pipelines/hpo.py``.

For each requested model type, the coordinator:

1. Builds an ``EnhancedHPOConfig`` for the optimizer, reading sampler /
   pruner / search-space settings from the supplied YAML config.
2. Creates an Optuna study (SQLite-backed when a storage URL is given so
   trials can resume across machines).
3. Runs ``optimizer.optimize(objective_fn, model_type, n_trials)`` where
   the objective is the subprocess trainer from
   ``spectramr.pipelines.hpo.subprocess_training_objective``.
4. **Writes ``best_config.yaml`` + ``best_params.json``** under the model's
   HPO output directory: a YAML that's the base config with the winning
   hyperparameters applied (ready to feed back into ``train --config``)
   and the raw Optuna parameter dict for provenance.
5. Returns the structured results from ``optimizer._process_results()``,
   augmented with paths to the written best-config artifacts.

The subprocess trainer pattern is the same one used by
``scripts/hpo/optuna_kan_dual_domain.py`` — each trial spawns
``python -m spectramr.cli train --config <generated.yaml>`` so a trial crash
(NaN, OOM, lib mismatch) can't poison the parent Optuna study.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spectramr.config.io import read_yaml_dict, write_yaml_dict
from spectramr.infrastructure.hpo.dotted_override import apply_dotted_override

# Type alias for the per-trial objective-factory injection point.
# Concrete implementation lives in ``src/pipelines/hpo.py`` (the
# subprocess-based factory). Defining the seam here keeps
# ``HPOCoordinator`` from importing upward — see
# ``TODO/audit/17_remaining_infrastructure.md`` F5 / CLAUDE.md
# layer-direction rule.
ObjectiveFactory = Callable[..., Callable[..., float]]


class HPOCoordinator:
    """Coordinator for Hyperparameter Optimization across one or more models.

    See module docstring for the wiring story.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def execute_hpo(
        self,
        model_types: list[str],
        input_lr_dir: str,
        input_hr_dir: str,
        output_dir: str,
        n_trials: int = 10,
        config_path: str | None = None,
        device: str | None = None,
        objective_metric: str = "val_loss",
        cost_weight: float = 0.0,
        max_training_time: float | None = None,
        enable_cost_optimization: bool = False,
        storage_url: str | None = None,
        sampler_type: str = "tpe",
        pruner_type: str = "hyperband",
        max_iter_per_trial: int = 30000,
        search_space: SearchSpace | None = None,  # noqa: F821
        objective_factory: ObjectiveFactory | None = None,
    ) -> dict[str, Any]:
        """Execute HPO for the given model types.

        Args:
            model_types: List of model_type names registered via
                ``@register_model``. One Optuna study is created per type.
            input_lr_dir / input_hr_dir / output_dir: Inherited from the
                old signature for backward compatibility; only ``output_dir``
                is currently used (each model gets its own subdirectory for
                trial configs / checkpoints / metrics).
            n_trials: Number of trials per model.
            config_path: Required — path to the base training YAML each trial
                will mutate.
            device: Requested device. **Not currently forwarded** -- this
                parameter is bound and never read, so each trial inherits
                the base YAML's device (pitfall #15; issue filed). The
                docstring used to claim it reached the subprocess YAML,
                which made it a facade (#16).
            objective_metric: Column name in ``training_metrics.csv`` to
                maximize. Use ``val_loss`` for minimization (sign-flipped here).
            cost_weight: If > 0, switches to multi-objective optimization
                using (objective_metric, training_time) as the two axes.
            max_training_time: Max wall-clock per trial in hours (advisory).
            enable_cost_optimization: If True, use multi-objective (Pareto).
            storage_url: Optuna storage URL. Pass ``sqlite:///path.db`` for
                resumable / parallelizable studies.
            sampler_type: Optuna sampler. ``tpe`` (default), ``cmaes``, ``nsga2``.
            pruner_type: Optuna pruner. ``hyperband`` (default), ``median``,
                ``successive_halving``, ``threshold``.
            max_iter_per_trial: Truncated training budget per trial.
        """
        if not config_path:
            raise ValueError("config_path is required — HPO needs a base YAML to mutate per trial.")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Base config not found: {config_path}")

        # Lazy imports so the module stays importable on systems without
        # optuna / scipy / torch installed for the lighter use cases.
        try:
            from spectramr.models.tuning.enhanced_hyperparameter_tuning import (
                EnhancedHPOConfig,
                EnhancedHPOptimizer,
                MultiObjectiveConfig,
            )
        except ImportError as e:
            raise ImportError(
                "EnhancedHPOptimizer is unavailable. Install optuna with "
                "`uv pip install optuna` and re-run."
            ) from e

        # The objective factory is injected by the caller (a layer
        # allowed to import from ``spectramr.pipelines``). The infrastructure
        # layer must not pull pipeline code in itself — CLAUDE.md
        # layer-direction rule + audit 17 F5.
        if objective_factory is None:
            raise ValueError(
                "HPOCoordinator.execute_hpo requires `objective_factory=` "
                "(typically `spectramr.pipelines.hpo.subprocess_training_objective`). "
                "Callers in `pipelines/` or `application/` can import that "
                "function and pass it in; `infrastructure/` cannot."
            )

        # Build the per-trial overrides_fn from the search-space spec.
        # If no search space is provided, log a loud warning — the run will
        # be a no-op (every trial uses the base config unchanged).
        if search_space is None or len(search_space) == 0:
            self.logger.warning(
                "No search_space provided to HPO; every trial will run the "
                "base config unchanged. Pass `search_space=...` to actually "
                "vary hyperparameters between trials."
            )
            overrides_fn = None
        else:
            overrides_fn = search_space.make_overrides_fn()
            self.logger.info(
                f"HPO will sample {len(search_space)} hyperparameters per trial: "
                f"{list(search_space.params.keys())}"
            )

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {}

        for model_type in model_types:
            self.logger.info(f"=== HPO for model: {model_type} ===")

            model_out = out_root / f"hpo_{model_type}"
            model_out.mkdir(parents=True, exist_ok=True)

            # Multi-objective only when cost optimization is asked for.
            multi_obj = None
            if enable_cost_optimization and cost_weight > 0:
                multi_obj = MultiObjectiveConfig(
                    objectives=[objective_metric, "training_time_seconds"],
                    directions=["maximize", "minimize"],
                )

            cfg = EnhancedHPOConfig(
                multi_objective=multi_obj,
                sampler_type=sampler_type,
                pruner_type=pruner_type,
            )

            optimizer = EnhancedHPOptimizer(cfg)
            study_name = f"hpo_{model_type}"
            optimizer.create_study(
                study_name=study_name,
                storage_url=storage_url,
                load_if_exists=True,
            )

            objective = objective_factory(
                base_config_path=str(config_path),
                output_root=model_out,
                metric_name=objective_metric,
                max_iterations=max_iter_per_trial,
                overrides_fn=overrides_fn,
            )

            try:
                model_results = optimizer.optimize(
                    objective_fn=objective,
                    model_type=model_type,
                    n_trials=n_trials,
                )
                # Materialize the winning config + params on disk so the
                # user can re-run training directly (`train --config best_config.yaml`).
                best_params = model_results.get("best_params") or {}
                if not best_params:
                    self.logger.warning(
                        f"[HPO/{model_type}] No completed trials — skipping "
                        "best_config.yaml write (all trials failed or were pruned)."
                    )
                    artifacts = {}
                else:
                    artifacts = self._write_best_config(
                        base_config_path=str(config_path),
                        best_params=best_params,
                        best_value=model_results.get("best_value"),
                        out_dir=model_out,
                        model_type=model_type,
                        objective_metric=objective_metric,
                    )
                results[model_type] = {
                    "status": "completed",
                    **model_results,
                    **artifacts,
                }
            except Exception as e:
                self.logger.exception(f"HPO for {model_type} failed: {e}")
                results[model_type] = {
                    "status": "failed",
                    "error": str(e),
                }

        return results

    @staticmethod
    def _apply_param_override(cfg: dict, dotted: str, value: Any) -> None:
        """Apply a dotted-path override to a nested config dict in place.

        Delegates to ``apply_dotted_override`` from ``hpo_search_spaces`` so
        that list-selector syntax (e.g. ``losses.kspace[name=log_spectral].weight``)
        is handled correctly in the written best_config.yaml.
        """
        apply_dotted_override(cfg, dotted, value)

    def _write_best_config(
        self,
        base_config_path: str,
        best_params: dict[str, Any],
        best_value: Any,
        out_dir: Path,
        model_type: str,
        objective_metric: str,
    ) -> dict[str, str]:
        """Serialize ``best_config.yaml`` + ``best_params.json`` and return paths.

        The YAML is the base config with every Optuna-sampled parameter
        applied as a dotted-path override, plus three audit-friendly
        annotations injected under ``metadata``:

        * ``hpo_source`` — pointer back to the base YAML
        * ``hpo_objective_metric`` — what we optimized for
        * ``hpo_best_value`` — the achieved value (so the YAML is
          self-documenting about *why* these particular params were picked)

        The JSON is the raw Optuna parameter dict + objective value, used
        for downstream analysis tooling (TPE prior visualization, etc.)
        and as a provenance record independent of any YAML mutation.
        """
        try:
            # Route through the config-layer helper so the SSOT
            # pre-commit guard (tests/unit/test_no_yaml_reparse.py)
            # stays clean. See TODO/backlog_ssot_and_layering_cleanup.md.
            cfg = read_yaml_dict(base_config_path)
        except Exception as e:
            self.logger.warning(f"Could not load base config to materialize best_config: {e}")
            return {}

        cfg = copy.deepcopy(cfg)
        for dotted, value in best_params.items():
            self._apply_param_override(cfg, dotted, value)

        cfg.setdefault("metadata", {})
        cfg["metadata"]["hpo_source"] = base_config_path
        cfg["metadata"]["hpo_objective_metric"] = objective_metric
        if best_value is not None:
            try:
                cfg["metadata"]["hpo_best_value"] = float(best_value)
            except (TypeError, ValueError):
                cfg["metadata"]["hpo_best_value"] = str(best_value)
        cfg["metadata"]["name"] = f"{cfg.get('metadata', {}).get('name', model_type)}_hpo_best"

        # Repoint the output dir so a follow-up training run doesn't
        # overwrite the HPO trial directories.
        cfg.setdefault("training", {})["output_dir"] = str(
            out_dir.parent / f"{model_type}_hpo_winner"
        )

        yaml_path = out_dir / "best_config.yaml"
        json_path = out_dir / "best_params.json"
        try:
            write_yaml_dict(cfg, yaml_path)
            with open(json_path, "w") as fh:
                json.dump(
                    {
                        "model_type": model_type,
                        "objective_metric": objective_metric,
                        "best_value": best_value,
                        "best_params": best_params,
                        "base_config": base_config_path,
                    },
                    fh,
                    indent=2,
                    default=str,
                )
            self.logger.info(f"[HPO/{model_type}] wrote best_config -> {yaml_path}")
            self.logger.info(f"[HPO/{model_type}] wrote best_params -> {json_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write best-config artifacts: {e}")
            return {}

        return {
            "best_config_path": str(yaml_path),
            "best_params_path": str(json_path),
        }
