"""Enhanced Hyperparameter Optimization Module
==========================================

Advanced HPO features for MRIForge research project including:
- Multi-objective optimization with Pareto fronts
- Distributed studies for multi-machine optimization
- Advanced pruners (ASHA, Hyperband, Median, Percentile)
- Auto batch size finder with memory-aware optimization
- Conditional search spaces based on architecture
- Performance profiling integration

Author: MRIForge Research Project
Date: August 2025
"""

import json  # noqa: F401
import logging

logger = logging.getLogger(__name__)
import multiprocessing as mp
import os  # noqa: F401
import sys  # noqa: F401
import time
import traceback  # noqa: F401
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401
from typing import Any

import numpy as np
import optuna
import torch
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.pruners import (
    HyperbandPruner,
    MedianPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    ThresholdPruner,
)
from optuna.samplers import CmaEsSampler, NSGAIISampler, TPESampler
from optuna.trial import TrialState
from torch import nn

# Import model factory
from ..factories.model_factory import get_model_factory

# Import existing modules
from .hyperparameter_tuning import HyperparameterConfig

# Initialize issue tracker
# init_issue_tracker(log_dir="logs", csv_filename="enhanced_hpo_issues.csv")

# Import model factory
# from ...factories.model_factory import get_model_factory

# Import training functions
TRAINING_AVAILABLE = True


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""

    objectives: list[str] = field(default_factory=lambda: ["fid", "psnr"])
    directions: list[str] = field(default_factory=lambda: ["minimize", "maximize"])
    weights: list[float] | None = None
    reference_points: list[float] | None = None
    population_size: int = 50
    n_generations: int = 100


@dataclass
class DistributedConfig:
    """Configuration for distributed optimization."""

    n_workers: int = mp.cpu_count()
    backend: str = "threading"  # threading, multiprocessing, ray
    redis_url: str | None = None
    ray_address: str | None = None
    heartbeat_interval: int = 60
    max_retries: int = 3


@dataclass
class AutoBatchSizeConfig:
    """Configuration for automatic batch size optimization."""

    min_batch_size: int = 1
    max_batch_size: int = 64
    memory_threshold: float = 0.8  # Use 80% of available memory
    growth_factor: float = 2.0
    max_trials_per_size: int = 3
    patience: int = 5


@dataclass
class ConditionalSearchSpace:
    """Conditional search space configuration."""

    base_params: dict[str, Any] = field(default_factory=dict)
    conditional_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class EnhancedHPOConfig:
    """Enhanced HPO configuration combining all advanced features."""

    base_config: HyperparameterConfig = field(default_factory=HyperparameterConfig)

    # Multi-objective settings
    multi_objective: MultiObjectiveConfig | None = None

    # Distributed settings
    distributed: DistributedConfig | None = None

    # Auto batch size settings
    auto_batch_size: AutoBatchSizeConfig | None = None

    # Conditional search spaces
    conditional_spaces: dict[str, ConditionalSearchSpace] = field(default_factory=dict)

    # Advanced pruners
    pruner_type: str = "median"  # median, percentile,
    # successive_halving, hyperband, threshold
    pruner_config: dict[str, Any] = field(default_factory=dict)

    # Performance profiling
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_flops: bool = True

    # Sampler settings
    sampler_type: str = "tpe"  # tpe, cmaes, nsgaii
    sampler_config: dict[str, Any] = field(default_factory=dict)


class MemoryProfiler:
    """Memory profiling utilities for batch size optimization."""

    @staticmethod
    def get_gpu_memory_info() -> dict[str, float]:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0, "utilization": 0}

        try:
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total - reserved

            return {
                "total": total / 1024**3,  # GB
                "used": allocated / 1024**3,
                "free": free / 1024**3,
                "utilization": allocated / total,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"total": 0, "used": 0, "free": 0, "utilization": 0}

    @staticmethod
    def estimate_model_memory(
        model: nn.Module,
        batch_size: int,
        input_shape: tuple[int, ...],
    ) -> float:
        """Estimate memory usage for a model with given batch size."""
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape)

            # Move to GPU if available - use device-agnostic pattern
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dummy_input = dummy_input.to(device)
            model = model.to(device)

            # Forward pass to measure memory
            with torch.no_grad():
                _ = model(dummy_input)

            memory_info = MemoryProfiler.get_gpu_memory_info()
            return memory_info["used"]

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return float("inf")  # Indicate OOM
            raise e


class AutoBatchSizeFinder:
    """Automatic batch size finder with memory-aware optimization."""

    def __init__(self, config: AutoBatchSizeConfig):
        """__init__.

        Args:
            config (AutoBatchSizeConfig): Description.
        """
        self.config = config
        self.best_batch_size = config.min_batch_size

    def find_optimal_batch_size(
        self,
        model_fn: Callable,
        input_shape: tuple[int, ...],
        device: str = "cuda",
    ) -> int:
        """Find optimal batch size that maximizes memory utilization without OOM.

        Args:
            model_fn: Function that returns a model instance
            input_shape: Input tensor shape (excluding batch dimension)
            device: Target device

        Returns:
            Optimal batch size

        """
        logger.info("Starting automatic batch size optimization...")

        current_size = self.config.min_batch_size
        best_size = current_size
        best_memory_utilization = 0

        while current_size <= self.config.max_batch_size:
            try:
                # Create model
                model = model_fn()

                # Estimate memory usage
                memory_used = MemoryProfiler.estimate_model_memory(
                    model,
                    current_size,
                    input_shape,
                )

                if memory_used == float("inf"):
                    # OOM occurred
                    logger.warning(f"Batch size {current_size} caused OOM, reducing...")
                    current_size = max(1, current_size // 2)
                    break

                # Calculate memory utilization
                memory_info = MemoryProfiler.get_gpu_memory_info()
                utilization = memory_used / memory_info["total"]

                logger.info(
                    f"Batch size {current_size}: {utilization:.2%} memory utilization",
                )

                if utilization <= self.config.memory_threshold:
                    best_size = current_size
                    best_memory_utilization = utilization

                    # Try larger batch size
                    current_size = int(current_size * self.config.growth_factor)
                else:
                    # Memory threshold exceeded
                    break

            except Exception as e:
                logger.error(f"Error testing batch size {current_size}: {e}")
                current_size = max(1, current_size // 2)
                continue  # Continue with reduced batch size

        self.best_batch_size = best_size
        logger.info(
            f"Optimal batch size found: {best_size} ({best_memory_utilization:.2%} utilization)",
        )

        return best_size


class AdvancedPrunerFactory:
    """Factory for creating advanced Optuna pruners."""

    @staticmethod
    def create_pruner(
        pruner_type: str,
        config: dict[str, Any],
    ) -> optuna.pruners.BasePruner:
        """Create pruner instance based on type and configuration."""
        if pruner_type == "median":
            n_startup_trials = config.get("n_startup_trials", 5)
            n_warmup_steps = config.get("n_warmup_steps", 10)
            interval_steps = config.get("interval_steps", 1)
            return MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
                interval_steps=interval_steps,
            )

        if pruner_type == "percentile":
            percentile = config.get("percentile", 25.0)
            n_startup_trials = config.get("n_startup_trials", 5)
            n_warmup_steps = config.get("n_warmup_steps", 10)
            interval_steps = config.get("interval_steps", 1)
            return PercentilePruner(
                percentile=percentile,
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
                interval_steps=interval_steps,
            )

        if pruner_type == "successive_halving":
            min_resource = config.get("min_resource", 1)
            reduction_factor = config.get("reduction_factor", 4)
            min_early_stopping_rate = config.get("min_early_stopping_rate", 0)
            return SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
            )

        if pruner_type == "hyperband":
            min_resource = config.get("min_resource", 1)
            max_resource = config.get("max_resource", 100)
            reduction_factor = config.get("reduction_factor", 3)
            return HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor,
            )

        if pruner_type == "threshold":
            lower = config.get("lower")
            upper = config.get("upper")
            n_warmup_steps = config.get("n_warmup_steps", 10)
            interval_steps = config.get("interval_steps", 1)
            return ThresholdPruner(
                lower=lower,
                upper=upper,
                n_warmup_steps=n_warmup_steps,
                interval_steps=interval_steps,
            )

        logger.warning(f"Unknown pruner type '{pruner_type}', using MedianPruner")
        return MedianPruner()


class SamplerFactory:
    """Factory for creating advanced Optuna samplers."""

    @staticmethod
    def create_sampler(
        sampler_type: str,
        config: dict[str, Any],
        n_objectives: int = 1,
    ) -> optuna.samplers.BaseSampler:
        """Create sampler instance based on type and configuration."""
        if sampler_type == "tpe":
            n_startup_trials = config.get("n_startup_trials", 10)
            n_ei_candidates = config.get("n_ei_candidates", 24)
            return TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=n_ei_candidates,
            )

        if sampler_type == "cmaes":
            sigma0 = config.get("sigma0")
            seed = config.get("seed")
            return CmaEsSampler(sigma0=sigma0, seed=seed)

        if sampler_type == "nsgaii" or n_objectives > 1:
            population_size = config.get("population_size", 50)
            mutation_prob = config.get("mutation_prob")
            crossover_prob = config.get("crossover_prob", 0.9)
            swapping_prob = config.get("swapping_prob", 0.5)
            seed = config.get("seed")
            return NSGAIISampler(
                population_size=population_size,
                mutation_prob=mutation_prob,
                crossover_prob=crossover_prob,
                swapping_prob=swapping_prob,
                seed=seed,
            )

        logger.warning(f"Unknown sampler type '{sampler_type}', using TPESampler")
        return TPESampler()


class ConditionalSearchSpaceManager:
    """Manager for conditional search spaces."""

    def __init__(self, conditional_spaces: dict[str, ConditionalSearchSpace]):
        """__init__.

        Args:
            conditional_spaces (dict[str, ConditionalSearchSpace]): Description.
        """
        self.conditional_spaces = conditional_spaces

    def get_search_space(self, model_type: str) -> dict[str, Any]:
        """Get search space for a specific model type with conditional parameters."""
        if model_type not in self.conditional_spaces:
            # Return default search space
            return self._get_default_search_space()

        space_config = self.conditional_spaces[model_type]

        # Start with base parameters
        search_space = space_config.base_params.copy()

        # Add conditional parameters based on dependencies
        for _param, conditions in space_config.conditional_params.items():
            # For now, include all conditional parameters
            # In a more advanced implementation, this could be dynamic
            search_space.update(conditions)

        return search_space

    def _get_default_search_space(self) -> dict[str, Any]:
        """Get default search space when no conditional space is defined."""
        return {
            "lr": FloatDistribution(1e-6, 1e-3, log=True),
            "weight_decay": FloatDistribution(1e-7, 1e-3, log=True),
            "gradient_clip": FloatDistribution(0.1, 5.0),
            "batch_size": IntDistribution(1, 32),
        }


class MultiObjectiveOptimizer:
    """Multi-objective optimization with Pareto front tracking."""

    def __init__(self, config: MultiObjectiveConfig):
        """__init__.

        Args:
            config (MultiObjectiveConfig): Description.
        """
        self.config = config
        self.pareto_front: list[dict[str, Any]] = []
        self.objective_history: list[list[float]] = []

    def update_pareto_front(
        self,
        trial_values: dict[str, Any],
        objectives: list[float],
    ) -> bool:
        """Update Pareto front with new trial results.

        Args:
            trial_values: Trial parameter values
            objectives: list of objective values

        Returns:
            True if Pareto front was updated

        """
        if not self.pareto_front:
            self.pareto_front.append(
                {"params": trial_values, "objectives": objectives.copy()},
            )
            self.objective_history.append(objectives.copy())
            return True

        # Check if this point dominates any existing points
        dominated_indices = []
        is_dominated = False

        for i, existing in enumerate(self.pareto_front):
            existing_objectives = existing["objectives"]

            # Check dominance
            dominates_existing = all(
                (
                    obj <= existing_obj
                    if self.config.directions[j] == "minimize"
                    else obj >= existing_obj
                )
                for j, (obj, existing_obj) in enumerate(
                    zip(objectives, existing_objectives, strict=False),
                )
            )

            existing_dominates = all(
                (
                    existing_obj <= obj
                    if self.config.directions[j] == "minimize"
                    else existing_obj >= obj
                )
                for j, (obj, existing_obj) in enumerate(
                    zip(objectives, existing_objectives, strict=False),
                )
            )

            if dominates_existing:
                dominated_indices.append(i)
            elif existing_dominates:
                is_dominated = True
                break

        if is_dominated:
            return False

        # Remove dominated points
        for i in reversed(dominated_indices):
            del self.pareto_front[i]

        # Add new point
        self.pareto_front.append(
            {"params": trial_values, "objectives": objectives.copy()},
        )
        self.objective_history.append(objectives.copy())

        return True

    def get_pareto_optimal_points(self) -> list[dict[str, Any]]:
        """Get current Pareto optimal points."""
        return self.pareto_front.copy()

    def get_hypervolume(self, reference_point: list[float] | None = None) -> float:
        """Calculate hypervolume of Pareto front."""
        if not self.pareto_front:
            return 0.0

        if reference_point is None:
            reference_point = self.config.reference_points
            if reference_point is None:
                # Use worst values as reference
                reference_point = []
                for i, direction in enumerate(self.config.directions):
                    if direction == "minimize":
                        ref_val = max(obj[i] for obj in self.objective_history)
                    else:
                        ref_val = min(obj[i] for obj in self.objective_history)
                    reference_point.append(ref_val * 1.1)  # 10% buffer

        # Simple hypervolume calculation for 2D case
        if len(self.config.objectives) == 2:
            objectives = np.array([p["objectives"] for p in self.pareto_front])
            return self._calculate_2d_hypervolume(objectives, reference_point)

        # Monte Carlo hypervolume approximation for >2 objectives
        objectives = np.array([p["objectives"] for p in self.pareto_front])
        ref = np.array(reference_point)

        # Number of random samples for Monte Carlo
        n_samples = 10000

        # Generate random points in the hyperrectangle bounded by reference
        samples = np.random.uniform(
            low=objectives.min(axis=0), high=ref, size=(n_samples, len(ref))
        )

        # Count how many samples are dominated by any Pareto point
        dominated_count = 0
        for sample in samples:
            for obj in objectives:
                # Check if sample is dominated (all dims worse than Pareto point)
                if np.all(sample >= obj):
                    dominated_count += 1
                    break

        # Hypervolume = volume of hyperrectangle * fraction dominated
        hyperrect_volume = np.prod(ref - objectives.min(axis=0))
        return hyperrect_volume * (dominated_count / n_samples)

    def _calculate_2d_hypervolume(
        self,
        objectives: np.ndarray,
        reference_point: list[float],
    ) -> float:
        """Calculate 2D hypervolume."""
        if len(objectives) == 0:
            return 0.0

        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]

        hypervolume = 0.0
        prev_x = reference_point[0]

        for i in range(len(sorted_objectives)):
            if i < len(sorted_objectives) - 1:
                width = prev_x - sorted_objectives[i, 0]
            else:
                width = prev_x - sorted_objectives[i, 0]

            height = reference_point[1] - sorted_objectives[i, 1]
            hypervolume += width * height
            prev_x = sorted_objectives[i, 0]

        return max(0, hypervolume)


class DistributedStudyManager:
    """Manager for distributed hyperparameter optimization."""

    def __init__(self, config: DistributedConfig):
        """__init__.

        Args:
            config (DistributedConfig): Description.
        """
        self.config = config
        self.workers = []
        self.executor = None

    def initialize_workers(self):
        """Initialize worker processes or threads."""
        if self.config.backend == "threading":
            self.executor = ThreadPoolExecutor(max_workers=self.config.n_workers)
        elif self.config.backend == "multiprocessing":
            # Use ProcessPoolExecutor for true parallelism
            from concurrent.futures import ProcessPoolExecutor

            self.executor = ProcessPoolExecutor(max_workers=self.config.n_workers)
        elif self.config.backend == "ray":
            import ray

            if not ray.is_initialized():
                ray.init(address=self.config.ray_address)

    def optimize_distributed(
        self,
        objective_fn: Callable,
        study: optuna.Study,
        n_trials: int,
    ) -> list[optuna.Trial]:
        """Run distributed optimization."""
        if self.executor is None:
            self.initialize_workers()

        futures = []
        completed_trials = []

        # Submit initial batch of trials
        for _i in range(min(n_trials, self.config.n_workers)):
            future = self.executor.submit(self._run_trial, objective_fn, study)
            futures.append(future)

        # Process completed trials and submit new ones
        while futures:
            for future in as_completed(futures, timeout=1.0):
                try:
                    trial = future.result()
                    completed_trials.append(trial)

                    # Submit new trial if more work to do
                    if len(completed_trials) < n_trials:
                        new_future = self.executor.submit(
                            self._run_trial,
                            objective_fn,
                            study,
                        )
                        futures.append(new_future)

                    futures.remove(future)
                    break

                except Exception as e:
                    logger.error(f"Trial failed: {e}")
                    futures.remove(future)
                    break

        return completed_trials

    def _run_trial(self, objective_fn: Callable, study: optuna.Study) -> optuna.Trial:
        """Run a single trial."""
        trial = study.ask()
        try:
            value = objective_fn(trial)
            study.tell(trial, value)
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            study.tell(trial, float("inf"))
        return trial

    def shutdown(self):
        """Shutdown distributed workers."""
        if self.executor:
            self.executor.shutdown(wait=True)


class EnhancedHPOptimizer:
    """Enhanced hyperparameter optimizer with all advanced features."""

    def __init__(self, config: EnhancedHPOConfig):
        """__init__.

        Args:
            config (EnhancedHPOConfig): Description.
        """
        self.config = config
        self.study = None
        self.multi_objective_optimizer = None
        self.distributed_manager = None
        self.batch_size_finder = None
        self.search_space_manager = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all optimization components."""
        # Multi-objective optimizer
        if self.config.multi_objective:
            self.multi_objective_optimizer = MultiObjectiveOptimizer(
                self.config.multi_objective,
            )

        # Distributed manager
        if self.config.distributed:
            self.distributed_manager = DistributedStudyManager(self.config.distributed)

        # Auto batch size finder
        if self.config.auto_batch_size:
            self.batch_size_finder = AutoBatchSizeFinder(self.config.auto_batch_size)

        # Conditional search space manager
        self.search_space_manager = ConditionalSearchSpaceManager(
            self.config.conditional_spaces,
        )

    def create_study(
        self,
        study_name: str,
        storage_url: str | None = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """Create Optuna study with advanced configuration."""
        # Determine directions for multi-objective
        if self.config.multi_objective:
            directions = self.config.multi_objective.directions
        else:
            directions = None

        # Create sampler
        n_objectives = (
            len(self.config.multi_objective.objectives) if self.config.multi_objective else 1
        )
        sampler = SamplerFactory.create_sampler(
            self.config.sampler_type,
            self.config.sampler_config,
            n_objectives,
        )

        # Create pruner
        pruner = AdvancedPrunerFactory.create_pruner(
            self.config.pruner_type,
            self.config.pruner_config,
        )

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=load_if_exists,
            directions=directions,
            sampler=sampler,
            pruner=pruner,
        )

        return self.study

    def optimize(
        self,
        objective_fn: Callable,
        model_type: str,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Run enhanced hyperparameter optimization.

        Args:
            objective_fn: Objective function to optimize
            model_type: Model type for conditional search spaces
            n_trials: Number of trials to run

        Returns:
            Optimization results

        """
        if self.study is None:
            raise ValueError("Study not created. Call create_study() first.")

        logger.info(f"Starting enhanced HPO with {n_trials} trials for {model_type}")

        # Get search space for model type
        # Get search space for model type (for future use)
        self.search_space_manager.get_search_space(model_type)

        # Auto batch size optimization
        if self.batch_size_finder:
            optimal_batch_size = self.batch_size_finder.find_optimal_batch_size(
                lambda: self._create_model_for_batch_size_test(model_type),
                (1, 64, 64),  # Example input shape, should be parameterized
            )
            logger.info(f"Using optimal batch size: {optimal_batch_size}")

        # Run optimization
        if self.distributed_manager and self.config.distributed:
            # Distributed optimization
            logger.info("Running distributed optimization...")
            # Run distributed optimization
            self.distributed_manager.optimize_distributed(
                objective_fn,
                self.study,
                n_trials,
            )
        else:
            # Standard optimization
            logger.info("Running standard optimization...")
            self.study.optimize(objective_fn, n_trials=n_trials)

        # Process results
        results = self._process_results()

        # Shutdown distributed manager
        if self.distributed_manager:
            self.distributed_manager.shutdown()

        return results

    def _create_model_for_batch_size_test(self, model_type: str) -> nn.Module:
        """Create model instance for batch size testing."""
        factory = get_model_factory()
        generator, _ = factory.create_model_pair(
            model_type,
            in_channels=1,
            out_channels=1,
        )
        return generator

    def _process_results(self) -> dict[str, Any]:
        """Process optimization results."""
        results = {
            "best_params": {},
            "best_value": None,
            "n_trials": len(self.study.trials),
            "success_rate": 0,
            "pareto_front": [],
        }

        if self.config.multi_objective:
            # Multi-objective results
            pareto_trials = self.study.best_trials
            results["pareto_front"] = [
                {"params": trial.params, "values": trial.values, "number": trial.number}
                for trial in pareto_trials
            ]

            if self.multi_objective_optimizer:
                results["pareto_front"] = self.multi_objective_optimizer.get_pareto_optimal_points()
                results["hypervolume"] = self.multi_objective_optimizer.get_hypervolume()

        else:
            # Single-objective results
            best_trial = self.study.best_trial
            results["best_params"] = best_trial.params
            results["best_value"] = best_trial.value

        # Calculate success rate
        completed_trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        results["success_rate"] = (
            len(completed_trials) / len(self.study.trials) if self.study.trials else 0
        )

        return results

    def get_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        if self.study is None:
            return "No study available"

        report = []
        report.append("# Enhanced HPO Optimization Report")
        report.append(f"Study: {self.study.study_name}")
        report.append(f"Total trials: {len(self.study.trials)}")

        if self.config.multi_objective:
            report.append(
                f"Pareto front size: "
                f"{len(self.multi_objective_optimizer.pareto_front) if self.multi_objective_optimizer else 0}",
            )
            if self.multi_objective_optimizer:
                report.append(
                    f"Hypervolume: {self.multi_objective_optimizer.get_hypervolume():.4f}",
                )
        else:
            best_trial = self.study.best_trial
            report.append(f"Best value: {best_trial.value:.4f}")
            report.append(f"Best trial: {best_trial.number}")

        return "\n".join(report)


# Convenience functions for easy usage
def create_enhanced_hpo_config(
    model_type: str = "unet",
    n_trials: int = 50,
    enable_multi_objective: bool = False,
    enable_distributed: bool = False,
    enable_auto_batch_size: bool = True,
    pruner_type: str = "median",
) -> EnhancedHPOConfig:
    """Create enhanced HPO configuration with sensible defaults."""
    # Base configuration
    base_config = HyperparameterConfig(
        n_trials=n_trials,
        study_name=f"enhanced_hpo_{model_type}_{int(time.time())}",
    )

    # Multi-objective configuration
    multi_objective = None
    if enable_multi_objective:
        multi_objective = MultiObjectiveConfig(
            objectives=["fid", "psnr"],
            directions=["minimize", "maximize"],
        )

    # Distributed configuration
    distributed = None
    if enable_distributed:
        distributed = DistributedConfig(
            n_workers=min(mp.cpu_count(), 4),
            backend="threading",
        )

    # Auto batch size configuration
    auto_batch_size = None
    if enable_auto_batch_size:
        auto_batch_size = AutoBatchSizeConfig()

    # Pruner configuration
    pruner_config = {}
    if pruner_type == "hyperband":
        pruner_config = {"min_resource": 1, "max_resource": 100, "reduction_factor": 3}
    elif pruner_type == "successive_halving":
        pruner_config = {"min_resource": 1, "reduction_factor": 4}

    return EnhancedHPOConfig(
        base_config=base_config,
        multi_objective=multi_objective,
        distributed=distributed,
        auto_batch_size=auto_batch_size,
        pruner_type=pruner_type,
        pruner_config=pruner_config,
    )


def run_enhanced_hyperparameter_optimization(
    model_type: str,
    config: EnhancedHPOConfig | None = None,
    objective_fn: Callable | None = None,
) -> dict[str, Any]:
    """Run enhanced hyperparameter optimization with all features.

    Args:
        model_type: Type of model to optimize
        config: Enhanced HPO configuration
        objective_fn: Custom objective function

    Returns:
        Optimization results

    """
    if config is None:
        config = create_enhanced_hpo_config(model_type)

    optimizer = EnhancedHPOptimizer(config)

    # Create study
    storage_url = f"sqlite:///training_experiments/enhanced_hpo_{model_type}.db"
    optimizer.create_study(
        study_name=config.base_config.study_name,
        storage_url=storage_url,
    )

    # Define default objective function if not provided
    if objective_fn is None:

        def default_objective(trial: optuna.Trial) -> float:
            # Simple dummy objective for demonstration
            """default_objective.

            Args:
                trial (optuna.Trial): Description.
            Returns:
                float: Description.
            """
            trial.suggest_float("lr", 1e-6, 1e-3, log=True)
            trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

            # Simulate training with random performance
            # In real usage, this would call actual training
            return np.random.uniform(10, 50)  # FID-like score

        objective_fn = default_objective

    # Run optimization
    results = optimizer.optimize(objective_fn, model_type, config.base_config.n_trials)

    # Log results
    logger.info(f"Enhanced HPO completed for {model_type}")
    logger.info(f"Results: {results}")

    return results


if __name__ == "__main__":
    # Example usage
    config = create_enhanced_hpo_config(
        model_type="unet",
        enable_multi_objective=True,
        enable_distributed=True,
        pruner_type="hyperband",
    )

    results = run_enhanced_hyperparameter_optimization("unet", config)
    print("Optimization Results:", results)
