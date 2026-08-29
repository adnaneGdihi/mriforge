#!/usr/bin/env python3
"""Hyperparameter Tuning Tool for MRIForge.

This script runs a hyperparameter search using Optuna.
Usage:
    python -m mriforge.tools.tune --config experiments/configs/base_config.yaml --trials 10
"""

import argparse

import optuna
import yaml

from mriforge.models.tuning import HyperparameterTuner
from mriforge.pipelines.train import run_training_pipeline


def main() -> None:
    """main."""
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to base config yaml")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--output", type=str, default="tuning_results.yaml", help="Output file")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"[Tuning] CLI: Loaded base config from {args.config}")

    # Initialize Tuner
    tuner = HyperparameterTuner(
        base_config=config,
        n_trials=args.trials,
        study_name="mriforge_optimization",
        storage_url="sqlite:///tuning.db",  # Persistent storage
    )

    # Define wrapper to adapt signatures
    def training_wrapper(cfg, trial):
        # Enforce short training for tuning speed if not specified
        """training_wrapper.

        Args:
            cfg (Any): Description.
            trial (Any): Description.
        Returns:
            Any: Description.
        """
        if "training" not in cfg:
            cfg["training"] = {}

        # Override max_iterations for faster trials (e.g. 1 epoch equivalent or fixed steps)
        # Unless config explicitly sets 'tuning_iterations'
        # defaulting to 500 steps for quick feedback
        if "tuning_iterations" in cfg.get("training", {}):
            cfg["training"]["max_iterations"] = cfg["training"]["tuning_iterations"]
        else:
            # Default to 500 steps if not present, to avoid running full training
            # Check if max_iterations is huge from base config
            current_max = cfg["training"].get("max_iterations", 100000)
            if current_max > 1000:
                print(
                    f"[Tuning] Limiting max_iterations to 1000 for tuning trial (was {current_max})"
                )
                cfg["training"]["max_iterations"] = 1000

        # Convert dict to namespace if needed by pipeline?
        # run_training_pipeline expects an object (Namespace) or dict.
        # It handles getattr(config, ...) so Namespace is preferred, but dict works if handled.
        # train.py L175: if hasattr(config, "training")...
        # A raw dict does NOT have attributes.
        # We need to wrap it in a SimpleNamespace or ensure run_training_pipeline handles dicts.
        # train.py seems to mix hasattr and key access or expects an object.
        # Let's fix this by converting to a simple class/namespace.

        class ConfigObj:
            """ConfigObj class."""

            def __init__(self, dictionary):
                """__init__.

                Args:
                    dictionary (Any): Description.
                """
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        value = ConfigObj(value)
                    setattr(self, key, value)

            def __getitem__(self, item):
                """__getitem__.

                Args:
                    item (Any): Description.
                Returns:
                    Any: Description.
                """
                return getattr(self, item)

            def __contains__(self, item):
                """__contains__.

                Args:
                    item (Any): Description.
                Returns:
                    Any: Description.
                """
                return hasattr(self, item)

        cfg_obj = ConfigObj(cfg)

        result = run_training_pipeline(cfg_obj)

        # Check for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        return result.get("best_val_loss", float("inf"))

    # Run Optimization
    print(f"[Tuning] Starting {args.trials} trials...")
    best_params = tuner.optimize(training_wrapper)

    print("\n[Tuning] Optimization Complete!")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Save results
    with open(args.output, "w") as f:
        yaml.dump(best_params, f)
    print(f"[Tuning] Saved best params to {args.output}")


if __name__ == "__main__":
    main()
