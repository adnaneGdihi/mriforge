"""Integration tests for Hyperparameter Optimization (HPO).

Tests:
- Grid search over learning rates
- Random search over model architectures
- Bayesian optimization with validation metrics
- Early stopping based on validation performance
- Multi-objective optimization (PSNR + SSIM)

HPO is critical for finding optimal configurations across:
- Learning rates, batch sizes, schedulers
- Model architectures (depth, width, attention)
- Loss weights (lambda_l1, lambda_kspace, etc.)
- Physics parameters (noise levels, DC weights)
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from spectramr.config.settings import TrainingSettings


class TestGridSearchHPO:
    """Test grid search hyperparameter optimization."""

    @pytest.fixture
    def temp_hpo_dir(self):
        """Create temporary directory for HPO experiments."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_config_dict(self, temp_hpo_dir):
        """Base configuration for HPO experiments."""
        return {
            "config_version": "1.0",
            "metadata": {
                "name": "hpo_grid_search",
                "output_dir": str(temp_hpo_dir / "checkpoints"),
                "seed": 42,
            },
            "logging": {
                "experiment_name": "hpo_grid_search",
            },
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
                "base_channels": 16,
            },
            "data": {
                "dataset_type": "synthetic",
                "batch_size": 2,
                "num_workers": 0,
                "image_size": [64, 64],
            },
            "optimization": {
                "learning_rate": 1e-4,  # Will be varied
                "use_amp": False,
                "optimizer_type": "adam",
            },
            "training": {
                "training_mode": "reconstruction",
                # ``epochs`` lives under ``training``. This fixture declared
                # ``optimization.num_epochs``, which was never a field there --
                # the schema is ``extra="forbid"``, so every test using this
                # fixture failed to construct at all (4 records on cluster job
                # 8004252, all reported as "1 validation error for
                # TrainingSettings" with no hint that the key was invented).
                "epochs": 2,
                "log_every_n_steps": 1,
            },
            "losses": {
                "reconstruction": {
                    "lambda_l1": 1.0,  # Will be varied
                },
            },
            "physics": {
                "data_consistency": {
                    "enabled": True,
                    "train_noise_level": 0.01,
                },
            },
        }

    def test_learning_rate_grid_search(self, base_config_dict, temp_hpo_dir):
        """Test grid search over learning rates."""
        learning_rates = [1e-5, 1e-4, 1e-3]

        configs = []
        for lr in learning_rates:
            config_dict = base_config_dict.copy()
            config_dict["optimization"]["learning_rate"] = lr
            config_dict["metadata"]["name"] = f"hpo_lr_{lr:.0e}"

            # Save config
            config_path = temp_hpo_dir / f"config_lr_{lr:.0e}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Load and validate
            config = TrainingSettings.from_yaml(str(config_path))
            assert config.optimization.optimizer.learning_rate == lr

            configs.append(config)

        assert len(configs) == 3

        # In real HPO: run each config, track val_loss, select best

    def test_loss_weight_grid_search(self, base_config_dict, temp_hpo_dir):
        """Test grid search over loss weights."""
        lambda_l1_values = [0.5, 1.0, 2.0]
        lambda_kspace_values = [0.0, 0.1, 0.5]

        configs = []
        for l1 in lambda_l1_values:
            for kspace in lambda_kspace_values:
                config_dict = base_config_dict.copy()
                config_dict["losses"]["reconstruction"]["lambda_l1"] = l1

                # Add k-space loss
                if "kspace" not in config_dict["losses"]["reconstruction"]:
                    config_dict["losses"]["reconstruction"][
                        "lambda_kspace"
                    ] = kspace

                config_dict["metadata"]["name"] = f"hpo_l1_{l1}_kspace_{kspace}"

                config_path = temp_hpo_dir / f"config_l1_{l1}_kspace_{kspace}.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f)

                config = TrainingSettings.from_yaml(str(config_path))
                configs.append(config)

        # 3x3 grid = 9 configs
        assert len(configs) == 9

    def test_batch_size_scheduler_grid(self, base_config_dict, temp_hpo_dir):
        """Test grid search over batch sizes and schedulers."""
        batch_sizes = [2, 4, 8]
        schedulers = ["constant", "cosine", "step"]

        configs = []
        for bs in batch_sizes:
            for scheduler in schedulers:
                config_dict = base_config_dict.copy()
                config_dict["data"]["batch_size"] = bs
                config_dict["optimization"]["scheduler_type"] = scheduler

                config_dict["metadata"]["name"] = f"hpo_bs_{bs}_sched_{scheduler}"

                config_path = temp_hpo_dir / f"config_bs_{bs}_sched_{scheduler}.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f)

                config = TrainingSettings.from_yaml(str(config_path))
                assert config.data.loader.batch_size == bs

                configs.append(config)

        assert len(configs) == 9


class TestRandomSearchHPO:
    """Test random search hyperparameter optimization."""

    @pytest.fixture
    def hpo_search_space(self):
        """Define HPO search space for random sampling."""
        return {
            "learning_rate": {
                "type": "log_uniform",
                "min": 1e-5,
                "max": 1e-2,
            },
            "batch_size": {
                "type": "choice",
                "values": [2, 4, 8, 16],
            },
            "base_channels": {
                "type": "choice",
                "values": [16, 32, 64],
            },
            "lambda_l1": {
                "type": "uniform",
                "min": 0.1,
                "max": 2.0,
            },
            "train_noise_level": {
                "type": "uniform",
                "min": 0.005,
                "max": 0.05,
            },
        }

    def test_random_sample_from_search_space(self, hpo_search_space):
        """Test random sampling from HPO search space."""
        import random

        num_samples = 10
        samples = []

        for _ in range(num_samples):
            sample = {}

            for param_name, param_config in hpo_search_space.items():
                if param_config["type"] == "log_uniform":
                    # Sample from log-uniform distribution
                    log_min = torch.log10(torch.tensor(param_config["min"]))
                    log_max = torch.log10(torch.tensor(param_config["max"]))
                    log_value = random.uniform(log_min.item(), log_max.item())
                    sample[param_name] = 10**log_value

                elif param_config["type"] == "uniform":
                    sample[param_name] = random.uniform(
                        param_config["min"], param_config["max"]
                    )

                elif param_config["type"] == "choice":
                    sample[param_name] = random.choice(param_config["values"])

            samples.append(sample)

        assert len(samples) == num_samples

        # Verify samples are within bounds
        for sample in samples:
            assert 1e-5 <= sample["learning_rate"] <= 1e-2
            assert sample["batch_size"] in [2, 4, 8, 16]
            assert sample["base_channels"] in [16, 32, 64]
            assert 0.1 <= sample["lambda_l1"] <= 2.0
            assert 0.005 <= sample["train_noise_level"] <= 0.05

    def test_random_search_creates_diverse_configs(self, hpo_search_space, tmp_path):
        """Test that random search creates diverse configurations."""
        import random

        random.seed(42)
        num_trials = 5

        base_config = {
            "config_version": "1.0",
            "metadata": {
                "name": "hpo_random_search",
                "output_dir": str(tmp_path / "checkpoints"),
            },
            "logging": {
                "experiment_name": "hpo_random_search",
            },
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {
                "dataset_type": "synthetic",
                "num_workers": 0,
            },
            "optimization": {
                "optimizer_type": "adam",
            },
            "training": {
                "training_mode": "reconstruction",
                # See the note above: epochs belongs under ``training``.
                "epochs": 1,
            },
            "losses": {
                "reconstruction": {},
            },
            "physics": {
                "data_consistency": {
                    "enabled": True,
                },
            },
        }

        configs = []
        for trial in range(num_trials):
            # Sample hyperparameters
            sample = {}
            for param_name, param_config in hpo_search_space.items():
                if param_config["type"] == "choice":
                    sample[param_name] = random.choice(param_config["values"])
                elif param_config["type"] == "uniform":
                    sample[param_name] = random.uniform(
                        param_config["min"], param_config["max"]
                    )
                elif param_config["type"] == "log_uniform":
                    import math

                    log_min = math.log10(param_config["min"])
                    log_max = math.log10(param_config["max"])
                    log_value = random.uniform(log_min, log_max)
                    sample[param_name] = 10**log_value

            # Create config
            config_dict = base_config.copy()
            config_dict["optimization"]["learning_rate"] = sample.get(
                "learning_rate", 1e-4
            )
            config_dict["data"]["batch_size"] = sample.get("batch_size", 2)
            config_dict["model"]["base_channels"] = sample.get("base_channels", 16)
            config_dict["losses"]["reconstruction"]["lambda_l1"] = sample.get(
                "lambda_l1", 1.0
            )
            config_dict["physics"]["data_consistency"]["train_noise_level"] = (
                sample.get("train_noise_level", 0.01)
            )

            config_dict["metadata"]["name"] = f"hpo_trial_{trial}"

            config_path = tmp_path / f"config_trial_{trial}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            config = TrainingSettings.from_yaml(str(config_path))
            configs.append(config)

        # Verify diversity (at least 2 different learning rates)
        learning_rates = [c.optimization.optimizer.learning_rate for c in configs]
        assert len(set(learning_rates)) >= 2, "Configs not diverse enough"


class TestBayesianOptimizationHPO:
    """Test Bayesian optimization for HPO (conceptual tests)."""

    def test_gaussian_process_surrogate_model(self):
        """Test Gaussian Process as surrogate for black-box optimization."""
        # In real Bayesian optimization:
        # 1. Sample initial random configs
        # 2. Train models and get validation metrics
        # 3. Fit GP on (config, metric) pairs
        # 4. Use acquisition function (EI, UCB) to select next config
        # 5. Repeat until convergence

        # Simulate: 5 trials with known (hyperparameter, metric) pairs
        trials = [
            {"lr": 1e-4, "val_loss": 0.5},
            {"lr": 1e-3, "val_loss": 0.3},
            {"lr": 1e-5, "val_loss": 0.6},
            {"lr": 5e-4, "val_loss": 0.25},
            {"lr": 2e-4, "val_loss": 0.4},
        ]

        # Find best trial
        best_trial = min(trials, key=lambda x: x["val_loss"])

        assert best_trial["lr"] == 5e-4
        assert best_trial["val_loss"] == 0.25

        # In production: use `optuna`, `ray.tune`, or `hyperopt` for BO

    def test_acquisition_function_expected_improvement(self):
        """Test Expected Improvement acquisition function."""
        # EI(x) = E[max(f_best - f(x), 0)]
        # where f(x) ~ GP(mu(x), sigma(x))

        # Simulated GP predictions
        candidates = [
            {"lr": 1e-4, "mu": 0.35, "sigma": 0.05},  # Low uncertainty
            {"lr": 1e-3, "mu": 0.24, "sigma": 0.10},  # EI = (0.25 - 0.24) * 0.1 = 0.001
            {"lr": 5e-5, "mu": 0.40, "sigma": 0.02},  # Worse mean
        ]

        f_best = 0.25  # Current best validation loss

        # Simplified EI: prioritize high uncertainty when close to f_best
        ei_scores = []
        for cand in candidates:
            improvement = max(f_best - cand["mu"], 0)
            ei = improvement * cand["sigma"]  # Simplified (ignores probability)
            ei_scores.append(ei)

        # Candidate 2 (high sigma) should have highest EI
        best_idx = ei_scores.index(max(ei_scores))
        assert candidates[best_idx]["lr"] == 1e-3
        assert abs(ei_scores[best_idx] - 0.001) < 1e-6

    def test_multi_objective_optimization_pareto_front(self):
        """Test multi-objective optimization (PSNR, SSIM, training time)."""
        # Trials with multiple objectives
        trials = [
            {"lr": 1e-4, "psnr": 30.0, "ssim": 0.85, "time": 100},
            {"lr": 1e-3, "psnr": 32.0, "ssim": 0.90, "time": 120},  # Pareto optimal
            {"lr": 5e-4, "psnr": 31.0, "ssim": 0.88, "time": 110},
            {"lr": 2e-4, "psnr": 29.0, "ssim": 0.82, "time": 95},
        ]

        # Find Pareto front: trials where no other trial is better in all objectives
        def dominates(t1, t2):
            """Check if t1 Pareto-dominates t2 (higher PSNR/SSIM, lower time)."""
            better_psnr = t1["psnr"] >= t2["psnr"]
            better_ssim = t1["ssim"] >= t2["ssim"]
            better_time = t1["time"] <= t2["time"]

            # At least one strict improvement
            strict = (
                t1["psnr"] > t2["psnr"]
                or t1["ssim"] > t2["ssim"]
                or t1["time"] < t2["time"]
            )

            return better_psnr and better_ssim and better_time and strict

        pareto_front = []
        for t1 in trials:
            dominated = False
            for t2 in trials:
                if t1 != t2 and dominates(t2, t1):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(t1)

        # Trial with lr=1e-3 should be on Pareto front
        assert any(t["lr"] == 1e-3 for t in pareto_front)


class TestEarlyStoppingHPO:
    """Test early stopping strategies for HPO."""

    def test_early_stopping_patience(self):
        """Test early stopping based on validation loss patience."""
        # Simulated validation losses over epochs
        val_losses = [
            0.5,
            0.4,
            0.35,
            0.36,
            0.37,
            0.38,
            0.39,
        ]  # Starts improving, then plateaus

        patience = 3
        best_loss = float("inf")
        patience_counter = 0

        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = len(val_losses) - 1

        # Should stop at epoch 5 (3 epochs of no improvement after epoch 2)
        assert stopped_epoch == 5

    def test_early_stopping_min_delta(self):
        """Test early stopping with minimum delta threshold."""
        val_losses = [0.5, 0.49, 0.488, 0.487, 0.486]  # Very slow improvement

        min_delta = 0.01  # Require at least 1% improvement
        patience = 2
        best_loss = float("inf")
        patience_counter = 0

        for epoch, loss in enumerate(val_losses):
            improvement = best_loss - loss
            if improvement > min_delta:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = len(val_losses) - 1

        # Should stop early due to insufficient improvement
        assert stopped_epoch < len(val_losses) - 1

    def test_hyperband_successive_halving(self):
        """Test Hyperband successive halving for efficient HPO."""
        # Hyperband: Allocate more resources to promising configs
        # Start with 16 configs, train 1 epoch each
        # Keep top 50%, train 2 more epochs (total 3)
        # Keep top 50%, train 4 more epochs (total 7)

        # Simulated configs with validation losses after 1 epoch
        configs = [{"id": i, "val_loss_epoch1": 0.5 + i * 0.05} for i in range(16)]

        # Round 1: Train all for 1 epoch, keep top 8
        round1 = sorted(configs, key=lambda x: x["val_loss_epoch1"])[:8]
        assert len(round1) == 8
        assert round1[0]["id"] == 0  # Best config

        # Round 2: Train top 8 for 3 more epochs, keep top 4
        for c in round1:
            c["val_loss_epoch3"] = c["val_loss_epoch1"] - 0.1  # Improvement

        round2 = sorted(round1, key=lambda x: x["val_loss_epoch3"])[:4]
        assert len(round2) == 4

        # Round 3: Train top 4 for 7 total epochs, keep best
        for c in round2:
            c["val_loss_epoch7"] = c["val_loss_epoch3"] - 0.05

        final_best = min(round2, key=lambda x: x["val_loss_epoch7"])

        assert final_best["id"] == 0  # Best config wins


class TestHPOMetricsTracking:
    """Test metrics tracking for HPO experiments."""

    def test_hpo_metrics_aggregation(self):
        """Test aggregating metrics across HPO trials."""
        trials = [
            {
                "trial_id": 0,
                "lr": 1e-4,
                "val_psnr": 30.5,
                "val_ssim": 0.85,
                "train_time": 120,
            },
            {
                "trial_id": 1,
                "lr": 1e-3,
                "val_psnr": 32.0,
                "val_ssim": 0.90,
                "train_time": 100,
            },
            {
                "trial_id": 2,
                "lr": 5e-4,
                "val_psnr": 31.0,
                "val_ssim": 0.88,
                "train_time": 110,
            },
        ]

        # Best PSNR
        best_psnr_trial = max(trials, key=lambda x: x["val_psnr"])
        assert best_psnr_trial["trial_id"] == 1

        # Best SSIM
        best_ssim_trial = max(trials, key=lambda x: x["val_ssim"])
        assert best_ssim_trial["trial_id"] == 1

        # Fastest training
        fastest_trial = min(trials, key=lambda x: x["train_time"])
        assert fastest_trial["trial_id"] == 1

    def test_hpo_leaderboard_ranking(self):
        """Test ranking HPO trials by composite score."""
        trials = [
            {"trial_id": 0, "val_psnr": 30.0, "val_ssim": 0.85},
            {"trial_id": 1, "val_psnr": 32.0, "val_ssim": 0.90},
            {"trial_id": 2, "val_psnr": 31.0, "val_ssim": 0.88},
        ]

        # Composite score: weighted sum
        weight_psnr = 0.6
        weight_ssim = 0.4

        for trial in trials:
            # Normalize PSNR to [0, 1] (assume max 40 dB)
            psnr_norm = trial["val_psnr"] / 40.0
            ssim_norm = trial["val_ssim"]

            trial["composite_score"] = weight_psnr * psnr_norm + weight_ssim * ssim_norm

        # Sort by composite score
        leaderboard = sorted(trials, key=lambda x: x["composite_score"], reverse=True)

        # Trial 1 should be best
        assert leaderboard[0]["trial_id"] == 1

    def test_hpo_reproducibility_with_seed(self, tmp_path):
        """Test HPO reproducibility with fixed random seed."""
        import random

        def run_hpo_trial(seed, num_samples=3):
            random.seed(seed)
            samples = [random.uniform(1e-5, 1e-3) for _ in range(num_samples)]
            return samples

        # Run twice with same seed
        samples1 = run_hpo_trial(seed=42)
        samples2 = run_hpo_trial(seed=42)

        assert samples1 == samples2

        # Different seed should give different samples
        samples3 = run_hpo_trial(seed=123)
        assert samples1 != samples3


@pytest.mark.slow
class TestLargeScaleHPO:
    """Large-scale HPO tests (marked slow - for manual validation)."""

    def test_parallel_hpo_trials(self):
        """Test parallel execution of HPO trials (conceptual)."""
        # In production: Use Ray Tune or Optuna for parallel trials
        # Example: 100 trials x 4 GPUs = 25 trials per GPU

        num_trials = 100
        num_gpus = 4
        trials_per_gpu = num_trials // num_gpus

        assert trials_per_gpu == 25

        # Each GPU would run trials_per_gpu sequentially
        # Ray Tune handles scheduling automatically

    def test_hpo_resource_allocation(self):
        """Test resource allocation for HPO trials."""
        # Resources per trial
        resources_per_trial = {
            "cpu": 4,
            "gpu": 0.25,  # 4 trials per GPU
            "memory_gb": 16,
        }

        total_trials = 20
        total_gpus = total_trials * resources_per_trial["gpu"]

        assert total_gpus == 5.0  # Need 5 GPUs for 20 trials

    def test_hpo_checkpoint_all_trials(self, tmp_path):
        """Test checkpoint saving for all HPO trials."""
        num_trials = 5

        checkpoint_paths = []
        for trial_id in range(num_trials):
            checkpoint_dir = tmp_path / f"trial_{trial_id}"
            checkpoint_dir.mkdir(exist_ok=True)

            # Simulate checkpoint save
            checkpoint = {
                "trial_id": trial_id,
                "val_loss": 0.5 - trial_id * 0.05,
            }

            checkpoint_path = checkpoint_dir / "best.pt"
            torch.save(checkpoint, checkpoint_path)

            checkpoint_paths.append(checkpoint_path)

        # Verify all checkpoints saved
        assert len(checkpoint_paths) == num_trials
        assert all(p.exists() for p in checkpoint_paths)

        # Load best checkpoint
        best_val_loss = float("inf")
        best_checkpoint_path = None

        for checkpoint_path in checkpoint_paths:
            ckpt = torch.load(checkpoint_path, weights_only=False)
            if ckpt["val_loss"] < best_val_loss:
                best_val_loss = ckpt["val_loss"]
                best_checkpoint_path = checkpoint_path

        assert best_checkpoint_path is not None
        best_ckpt = torch.load(best_checkpoint_path, weights_only=False)
        assert best_ckpt["trial_id"] == 4  # Last trial has lowest loss
