import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
import yaml


@dataclass
class HyperparameterConfig:
    """Base configuration for hyperparameters."""

    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 4
    lambda_l1: float = 100.0
    lambda_perceptual: float = 10.0
    weight_decay: float = 0.0001

    def to_dict(self) -> dict[str, Any]:
        """to_dict.

        Returns:
            dict[str, Any]: Description.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class HyperparameterTuner:
    """
    Bayesian Hyperparameter Optimization using Optuna.

    Automates the search for optimal training hyperparameters (LR, Regularization, Batch Size)
    for MRI reconstruction models.
    """

    def __init__(
        self,
        experiment_config_path: str,
        study_name: str = "mri_recon_tuning",
        storage: str = "sqlite:///db.sqlite3",
    ):
        """__init__.

        Args:
            experiment_config_path (str): Description.
            study_name (str): Description.
            storage (str): Description.
        """
        self.config_path = experiment_config_path
        with open(experiment_config_path) as f:
            self.base_config = yaml.safe_load(f)

        self.study_name = study_name
        self.storage = storage
        self.logger = logging.getLogger(__name__)

    def objective(
        self, trial: optuna.Trial, training_fn: Callable[[dict, optuna.Trial], float]
    ) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object
            training_fn: Function that takes (config, trial) and returns validation metric (e.g. PSNR, Val Loss).
                         Should return a scalar to MAXIMIZE or MINIMIZE depending on study direction.

        Returns:
            Review metric.
        """
        # 1. Sample Hyperparameters
        # We define a search space based on common MRI needs
        # Ideally, search space should be configurable/injected.
        # Here we hardcode a standard search for demonstration/blueprint.

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        # Loss weights
        lambda_l1 = trial.suggest_float("lambda_l1", 0.1, 10.0)
        # lambda_perceptual = trial.suggest_float("lambda_perceptual", 0.0, 1.0)

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # 2. Update Config
        config = self.base_config.copy()

        # Helper to update nested dict
        if "training" not in config:
            config["training"] = {}
        config["training"]["lr"] = lr
        config["training"]["batch_size"] = batch_size

        if "loss" not in config["training"]:
            config["training"]["loss"] = {}
        if "weights" not in config["training"]["loss"]:
            config["training"]["loss"]["weights"] = {}

        config["training"]["loss"]["weights"]["l1"] = lambda_l1
        # config['training']['loss']['weights']['perceptual'] = lambda_perceptual

        # 3. Run Short Training Job (Pruning enabled)
        # Note: You must inject the trial object into your trainer to report intermediate metrics
        # and support trial.report() and trial.should_prune()

        try:
            val_metric = training_fn(config, trial)
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()  # Or return worst score

        return val_metric

    def run_optimization(
        self, training_fn: Callable, n_trials=50, direction="minimize"
    ) -> dict[str, Any]:
        """
        Execute the optimization loop.

        Args:
            training_fn: Callable executing the training.
            n_trials: Number of trials
            direction: "minimize" (loss) or "maximize" (PSNR)

        Returns:
            Best hyperparameters
        """
        study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
        )

        # Wrap objective to inject training_fn
        func = lambda trial: self.objective(trial, training_fn)

        study.optimize(func, n_trials=n_trials)

        self.logger.info(f"Best params: {study.best_params}")
        self.logger.info(f"Best value: {study.best_value}")

        return study.best_params


# Alias for compatibility
HyperparameterOptimizer = HyperparameterTuner
