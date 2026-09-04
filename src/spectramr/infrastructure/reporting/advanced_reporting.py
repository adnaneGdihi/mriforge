"""Advanced reporting system for spectraMR training.

This module provides comprehensive reporting capabilities including:
- Training progress reports
- Performance analysis reports
- Comparative analysis reports
- Experiment tracking and visualization
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spectramr.config.config import TrainingConfig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _worker_plot(metrics_history, output_dir, experiment_name):
    """Worker process for generating plots."""
    try:
        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Progress: {experiment_name}", fontsize=16)

        # Loss curves
        axes[0, 0].plot(
            metrics_history["epoch"],
            metrics_history["g_loss"],
            label="Generator Loss",
            linewidth=2,
        )
        axes[0, 0].plot(
            metrics_history["epoch"],
            metrics_history["d_loss"],
            label="Discriminator Loss",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Image quality metrics
        if any(p > 0 for p in metrics_history["psnr"]):
            axes[0, 1].plot(
                metrics_history["epoch"],
                metrics_history["psnr"],
                label="PSNR",
                linewidth=2,
                color="green",
            )
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("PSNR (dB)")
            axes[0, 1].set_title("Image Quality - PSNR")
            axes[0, 1].grid(True, alpha=0.3)

        if any(s > 0 for s in metrics_history["ssim"]):
            ax2 = axes[0, 1].twinx()
            ax2.plot(
                metrics_history["epoch"],
                metrics_history["ssim"],
                label="SSIM",
                linewidth=2,
                color="orange",
            )
            ax2.set_ylabel("SSIM")
            ax2.legend(loc="upper right")

        # Learning rate schedule
        if any(lr > 0 for lr in metrics_history["learning_rate"]):
            axes[1, 0].plot(
                metrics_history["epoch"],
                metrics_history["learning_rate"],
                linewidth=2,
                color="red",
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_title("Learning Rate Schedule")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True, alpha=0.3)

        # Gradient norms
        if any(g > 0 for g in metrics_history["gradient_norm_g"]):
            axes[1, 1].plot(
                metrics_history["epoch"],
                metrics_history["gradient_norm_g"],
                label="Generator",
                linewidth=2,
            )
            axes[1, 1].plot(
                metrics_history["epoch"],
                metrics_history["gradient_norm_d"],
                label="Discriminator",
                linewidth=2,
            )
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Gradient Norm")
            axes[1, 1].set_title("Gradient Norms")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "training_progress.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        logger.warning("Plotting failed in worker process: %s", e)


class TrainingReport:
    """Comprehensive training report generator."""

    def __init__(self, experiment_name: str, output_dir: str | Path = "./reports"):
        """__init__.

        Args:
            experiment_name (str): Description.
            output_dir (str | Path): Description.
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: dict[str, list[float]] = {}
        self.config: object | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def start_training(self, config: object):
        """Initialize training report."""
        self.config = config
        self.start_time = datetime.now()

        # Initialize metrics tracking
        self.metrics_history = {
            "epoch": [],
            "g_loss": [],
            "d_loss": [],
            "psnr": [],
            "ssim": [],
            "learning_rate": [],
            "gradient_norm_g": [],
            "gradient_norm_d": [],
        }

    def log_epoch_metrics(
        self,
        epoch: int,
        g_loss: float,
        d_loss: float,
        psnr: float | None = None,
        ssim: float | None = None,
        lr: float | None = None,
        grad_norm_g: float | None = None,
        grad_norm_d: float | None = None,
    ):
        """Log metrics for an epoch."""
        self.metrics_history["epoch"].append(epoch)
        self.metrics_history["g_loss"].append(g_loss)
        self.metrics_history["d_loss"].append(d_loss)
        self.metrics_history["psnr"].append(psnr or 0.0)
        self.metrics_history["ssim"].append(ssim or 0.0)
        self.metrics_history["learning_rate"].append(lr or 0.0)
        self.metrics_history["gradient_norm_g"].append(grad_norm_g or 0.0)
        self.metrics_history["gradient_norm_d"].append(grad_norm_d or 0.0)

    def end_training(self):
        """Finalize training report."""
        self.end_time = datetime.now()

    def generate_training_progress_report(self) -> str:
        """Generate training progress report."""
        if not self.config or not self.start_time:
            return "Training not started"

        duration = (self.end_time or datetime.now()) - self.start_time

        report = []
        report.append("=" * 60)
        report.append(f"TRAINING REPORT: {self.experiment_name}")
        report.append("=" * 60)
        report.append("")

        # Training summary
        report.append("TRAINING SUMMARY")
        report.append("-" * 30)
        cfg = self.config
        model_type = getattr(cfg, "model_type", "unknown")
        epochs = getattr(cfg, "epochs", "unknown")
        batch_size = getattr(cfg, "batch_size", "unknown")
        learning_rate = getattr(
            cfg,
            "learning_rate",
            getattr(cfg, "base_learning_rate", "unknown"),
        )
        report.append(f"Model Type: {model_type}")
        report.append(f"Epochs: {epochs}")
        report.append(f"Batch Size: {batch_size}")
        report.append(f"Learning Rate: {learning_rate}")
        report.append(f"Duration: {duration}")
        report.append("")

        # Final metrics
        if self.metrics_history["epoch"]:
            final_g_loss = self.metrics_history["g_loss"][-1]
            final_d_loss = self.metrics_history["d_loss"][-1]
            final_psnr = self.metrics_history["psnr"][-1]
            final_ssim = self.metrics_history["ssim"][-1]

            report.append("FINAL METRICS")
            report.append("-" * 30)
            report.append(f"Generator Loss: {final_g_loss:.4f}")
            report.append(f"Discriminator Loss: {final_d_loss:.4f}")
            if final_psnr > 0:
                report.append(f"PSNR: {final_psnr:.2f} dB")
            if final_ssim > 0:
                report.append(f"SSIM: {final_ssim:.4f}")
            report.append("")

        # Best metrics
        if self.metrics_history["psnr"]:
            best_psnr_idx = np.argmax(self.metrics_history["psnr"])
            best_psnr = self.metrics_history["psnr"][best_psnr_idx]
            best_psnr_epoch = self.metrics_history["epoch"][best_psnr_idx]

            report.append("BEST METRICS")
            report.append("-" * 30)
            report.append(
                f"Best PSNR: {best_psnr:.2f} dB (Epoch {best_psnr_epoch})",
            )

            best_ssim_idx = np.argmax(self.metrics_history["ssim"])
            best_ssim = self.metrics_history["ssim"][best_ssim_idx]
            best_ssim_epoch = self.metrics_history["epoch"][best_ssim_idx]
            report.append(f"Best SSIM: {best_ssim:.4f} (Epoch {best_ssim_epoch})")
            report.append("")

        return "\n".join(report)

    def generate_plots(self):
        """Generate training visualization plots (Offloaded)."""
        if not self.metrics_history["epoch"]:
            return

        # Detach data to CPU lists/dicts first
        # metrics_history is already a dict of lists of floats, so it's picklable
        history_copy = {k: list(v) for k, v in self.metrics_history.items()}

        # Start worker process
        p = multiprocessing.Process(
            target=_worker_plot,
            args=(history_copy, self.output_dir, self.experiment_name),
        )
        p.start()
        # Do not join(). Let it run in background.

    def save_metrics_csv(self):
        """Save metrics history to CSV."""
        df = pd.DataFrame(self.metrics_history)
        csv_path = self.output_dir / "metrics_history.csv"
        df.to_csv(csv_path, index=False)

    def save_report_json(self):
        """Save complete report as JSON."""
        report_data = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__ if self.config else None,
            "metrics_history": self.metrics_history,
            "training_duration": str(
                (self.end_time or datetime.now()) - (self.start_time or datetime.now()),
            ),
            "final_metrics": {
                "g_loss": (
                    self.metrics_history["g_loss"][-1] if self.metrics_history["g_loss"] else None
                ),
                "d_loss": (
                    self.metrics_history["d_loss"][-1] if self.metrics_history["d_loss"] else None
                ),
                "psnr": (
                    self.metrics_history["psnr"][-1] if self.metrics_history["psnr"] else None
                ),
                "ssim": (
                    self.metrics_history["ssim"][-1] if self.metrics_history["ssim"] else None
                ),
            },
        }

        json_path = self.output_dir / "training_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def generate_complete_report(self):
        """Generate and save complete training report."""
        # Generate text report
        text_report = self.generate_training_progress_report()
        with open(self.output_dir / "training_report.txt", "w") as f:
            f.write(text_report)

        # Generate plots
        self.generate_plots()

        # Save metrics CSV
        self.save_metrics_csv()

        # Save JSON report
        self.save_report_json()

        logger.info("Complete training report saved to: %s", self.output_dir)
        return self.output_dir


class ComparativeReport:
    """Comparative analysis across multiple experiments."""

    def __init__(
        self,
        experiment_names: list[str],
        output_dir: str | Path = "./reports",
    ):
        """__init__.

        Args:
            experiment_names (list[str]): Description.
            output_dir (str | Path): Description.
        """
        self.experiment_names = experiment_names
        self.output_dir = Path(output_dir) / "comparative_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_data: dict[str, dict[str, Any]] = {}

    def load_experiment_data(self, experiment_name: str, report_dir: str | Path):
        """Load data from experiment report."""
        report_dir = Path(report_dir)
        json_path = report_dir / "training_report.json"

        if json_path.exists():
            with open(json_path) as f:
                self.experiment_data[experiment_name] = json.load(f)

    def generate_comparison_report(self) -> str:
        """Generate comparative analysis report."""
        if not self.experiment_data:
            return "No experiment data loaded"

        report = []
        report.append("=" * 60)
        report.append("COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary table
        report.append("EXPERIMENT SUMMARY")
        report.append("-" * 30)
        report.append("<15")
        report.append("-" * 60)

        for exp_name, data in self.experiment_data.items():
            # config = data.get("config", {})  # Unused
            final_metrics = data.get("final_metrics", {})

            # model_type = config.get("model_type", "Unknown")  # Unused
            # psnr = final_metrics.get("psnr", "N/A")  # Unused
            # ssim = final_metrics.get("ssim", "N/A")  # Unused
            # duration = data.get("training_duration", "N/A")  # Unused

            report.append("<15")

        report.append("")

        # Detailed comparison
        report.append("DETAILED COMPARISON")
        report.append("-" * 30)

        metrics_to_compare = ["psnr", "ssim", "g_loss", "d_loss"]
        for metric in metrics_to_compare:
            report.append(f"\n{metric.upper()} Comparison:")
            for exp_name, data in self.experiment_data.items():
                final_metrics = data.get("final_metrics", {})
                value = final_metrics.get(metric)
                if value is not None:
                    report.append(f"  {exp_name}: {value:.4f}")
                else:
                    report.append(f"  {exp_name}: N/A")

        return "\n".join(report)

    def generate_comparison_plots(self):
        """Generate comparative visualization plots."""
        if not self.experiment_data:
            return

        # Extract final metrics for comparison
        experiments = []
        psnr_values = []
        ssim_values = []
        g_loss_values = []
        d_loss_values = []

        for exp_name, data in self.experiment_data.items():
            experiments.append(exp_name)
            final_metrics = data.get("final_metrics", {})

            psnr_values.append(final_metrics.get("psnr", 0))
            ssim_values.append(final_metrics.get("ssim", 0))
            g_loss_values.append(final_metrics.get("g_loss", 0))
            d_loss_values.append(final_metrics.get("d_loss", 0))

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Comparative Analysis Across Experiments", fontsize=16)

        # PSNR comparison
        axes[0, 0].bar(experiments, psnr_values, color="skyblue")
        axes[0, 0].set_title("Final PSNR Comparison")
        axes[0, 0].set_ylabel("PSNR (dB)")
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45)

        # SSIM comparison
        axes[0, 1].bar(experiments, ssim_values, color="lightgreen")
        axes[0, 1].set_title("Final SSIM Comparison")
        axes[0, 1].set_ylabel("SSIM")
        plt.setp(axes[0, 1].get_xticklabels(), rotation=45)

        # Loss comparison
        x = np.arange(len(experiments))
        width = 0.35
        axes[1, 0].bar(
            x - width / 2,
            g_loss_values,
            width,
            label="Generator",
            color="orange",
        )
        axes[1, 0].bar(
            x + width / 2,
            d_loss_values,
            width,
            label="Discriminator",
            color="red",
        )
        axes[1, 0].set_title("Final Loss Comparison")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(experiments, rotation=45)
        axes[1, 0].legend()

        # Training time comparison (placeholder) # IMPL
        # Use actual training duration data for comparison
        axes[1, 1].bar(experiments, range(len(experiments)), color="lightcoral")
        axes[1, 1].set_title("Training Time Comparison")
        axes[1, 1].set_ylabel("Relative Time")
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comparative_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )

    def generate_parameter_performance_scatter_plots(self):
        """Generate scatter plots correlating parameters with performance metrics."""
        if not self.experiment_data:
            return

        # Extract data for scatter plots
        experiments_data = []
        for exp_name, data in self.experiment_data.items():
            config = data.get("config", {})
            final_metrics = data.get("final_metrics", {})

            exp_data = {
                "name": exp_name,
                "model_type": config.get("model_type", "unknown"),
                "learning_rate": config.get("learning_rate", 0.0002),
                "batch_size": config.get("batch_size", 8),
                "epochs": config.get("epochs", 100),
                "l1_weight": config.get("l1_weight", 100.0),
                "psnr": final_metrics.get("psnr", 0),
                "ssim": final_metrics.get("ssim", 0),
                "fid": final_metrics.get("fid", 0),
                "inference_time": final_metrics.get("inference_time", 0),
                "model_params": final_metrics.get("model_params", 0),
            }
            experiments_data.append(exp_data)

        if not experiments_data:
            return

        df = pd.DataFrame(experiments_data)

        # Create scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Parameter vs Performance Correlation Analysis", fontsize=16)

        # Learning rate vs PSNR
        if "learning_rate" in df.columns and "psnr" in df.columns:
            sns.scatterplot(
                data=df,
                x="learning_rate",
                y="psnr",
                hue="model_type",
                ax=axes[0, 0],
            )
            axes[0, 0].set_title("Learning Rate vs PSNR")
            axes[0, 0].set_xlabel("Learning Rate")
            axes[0, 0].set_ylabel("PSNR (dB)")

        # Batch size vs PSNR
        if "batch_size" in df.columns and "psnr" in df.columns:
            sns.scatterplot(
                data=df,
                x="batch_size",
                y="psnr",
                hue="model_type",
                ax=axes[0, 1],
            )
            axes[0, 1].set_title("Batch Size vs PSNR")
            axes[0, 1].set_xlabel("Batch Size")
            axes[0, 1].set_ylabel("PSNR (dB)")

        # Model params vs PSNR
        if "model_params" in df.columns and "psnr" in df.columns:
            sns.scatterplot(
                data=df,
                x="model_params",
                y="psnr",
                hue="model_type",
                ax=axes[0, 2],
            )
            axes[0, 2].set_title("Model Parameters vs PSNR")
            axes[0, 2].set_xlabel("Model Parameters (M)")
            axes[0, 2].set_ylabel("PSNR (dB)")

        # Model params vs Inference time
        if "model_params" in df.columns and "inference_time" in df.columns:
            sns.scatterplot(
                data=df,
                x="model_params",
                y="inference_time",
                hue="model_type",
                ax=axes[1, 0],
            )
            axes[1, 0].set_title("Model Parameters vs Inference Time")
            axes[1, 0].set_xlabel("Model Parameters (M)")
            axes[1, 0].set_ylabel("Inference Time (ms)")

        # PSNR vs Inference time
        if "psnr" in df.columns and "inference_time" in df.columns:
            sns.scatterplot(
                data=df,
                x="psnr",
                y="inference_time",
                hue="model_type",
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("PSNR vs Inference Time")
            axes[1, 1].set_xlabel("PSNR (dB)")
            axes[1, 1].set_ylabel("Inference Time (ms)")

        # SSIM vs FID
        if "ssim" in df.columns and "fid" in df.columns:
            sns.scatterplot(data=df, x="ssim", y="fid", hue="model_type", ax=axes[1, 2])
            axes[1, 2].set_title("SSIM vs FID")
            axes[1, 2].set_xlabel("SSIM")
            axes[1, 2].set_ylabel("FID")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "parameter_performance_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save correlation data
        correlation_data = {}
        if len(df) > 1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            correlation_data = correlation_matrix.to_dict()

        with open(self.output_dir / "parameter_correlations.json", "w") as f:
            json.dump(correlation_data, f, indent=2)

    def save_comparison_report(self):
        """Save comparative analysis report."""
        # Generate text report
        text_report = self.generate_comparison_report()
        with open(self.output_dir / "comparative_report.txt", "w") as f:
            f.write(text_report)

        # Generate plots
        self.generate_comparison_plots()
        self.generate_parameter_performance_scatter_plots()

        logger.info("Comparative analysis saved to: %s", self.output_dir)


def create_training_report(
    experiment_name: str,
    config: TrainingConfig,
    output_dir: str | Path = "./reports",
) -> TrainingReport:
    """Create a training report instance."""
    report = TrainingReport(experiment_name, output_dir)
    report.start_training(config)
    return report


def create_comparative_report(
    experiment_names: list[str],
    experiment_dirs: list[str | Path],
    output_dir: str | Path = "./reports",
) -> ComparativeReport:
    """Create a comparative report across experiments."""
    report = ComparativeReport(experiment_names, output_dir)

    for name, exp_dir in zip(experiment_names, experiment_dirs, strict=False):
        report.load_experiment_data(name, exp_dir)

    return report
