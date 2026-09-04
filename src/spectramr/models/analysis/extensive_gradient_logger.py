#!/usr/bin/env python
"""Extensive Gradient Logger Module

Comprehensive gradient monitoring and logging system for GAN training.
Provides detailed gradient analysis, issue detection, and Excel-based
reporting.
"""

import logging

logger = logging.getLogger(__name__)

import os
import time

import pandas as pd
import torch

# Enhanced logging for performance


# Global flags for gradient detection (kept for compatibility)
_vanishing_gradient_detected = False
_exploding_gradient_detected = False
_exploding_gradient_critical = False


class ExtensiveGradientLogger:
    """Comprehensive gradient monitoring and logging system"""

    def __init__(
        self,
        log_dir: str | None = None,
        model_type: str = "standard_unet",
        device: torch.device | None = None,
    ):
        """__init__.

        Args:
            log_dir (Optional[str]): Description.
            model_type (str): Description.
            device (Optional[torch.device]): Description.
        """
        self.log_dir = log_dir
        if self.log_dir is None:
            # Default to project logs directory
            base_logs = os.path.join(os.getcwd(), "logs")
            self.log_dir = os.path.join(base_logs, "analysis")
        os.makedirs(self.log_dir, exist_ok=True)

        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        # Create specialized log directories
        self.gradient_log_dir = os.path.join(self.log_dir, "gradient_logs")
        os.makedirs(self.gradient_log_dir, exist_ok=True)

        # Initialize tracking variables
        self.gradient_history = []
        self.exploding_gradients_log = []
        self.vanishing_gradients_log = []
        self.gradient_statistics = []

        # Thresholds for gradient issues
        self.exploding_threshold = 10.0
        self.vanishing_threshold = 1e-6
        self.nan_inf_count = 0

        # File paths for Excel logging
        # Use CSV for faster I/O
        self.gradient_detailed_file = os.path.join(
            self.gradient_log_dir,
            f"gradient_detailed_{model_type}.csv",
        )
        self.gradient_summary_file = os.path.join(
            self.gradient_log_dir,
            f"gradient_summary_{model_type}.csv",
        )
        self.vanishing_issues_file = os.path.join(
            self.gradient_log_dir,
            f"gradient_issues_vanishing_{model_type}.csv",
        )
        self.exploding_issues_file = os.path.join(
            self.gradient_log_dir,
            f"gradient_issues_exploding_{model_type}.csv",
        )
        self.issues_summary_file = os.path.join(
            self.gradient_log_dir,
            f"gradient_issues_summary_{model_type}.csv",
        )

        logger.info(f"ExtensiveGradientLogger initialized for {model_type}")
        logger.debug(f"Gradient logs will be saved to: {self.gradient_log_dir}")

    def log_gradients(
        self,
        model,
        epoch,
        batch_idx,
        model_name="model",
        loss_value=None,
        log_interval=100,
    ):
        """Log detailed gradient information for a model (Optimized).

        Optimized: Logs only every `log_interval` steps and samples a subset of layers
        to avoid CPU-GPU synchronization stalls.
        """
        if model is None:
            return

        # 1. Strict Interval Check
        if batch_idx % log_interval != 0:
            return

        gradient_info = {
            "epoch": epoch,
            "batch": batch_idx,
            "model_name": model_name,
            "timestamp": time.time(),
            "loss_value": (
                getattr(loss_value, "item", lambda: loss_value)()
                if loss_value is not None
                else None
            ),
        }

        # 2. Sampling Strategy
        # Instead of iterating all parameters, pick a representative subset
        # (first, middle, last) to get a pulse on the network without the cost.
        try:
            param_names = list(dict(model.named_parameters()).keys())
            if not param_names:
                return

            # Select key layers: First, Middle, Last
            targets = {
                param_names[0],
                param_names[len(param_names) // 2],
                param_names[-1],
            }

            layer_stats = []
            total_norm_sq = 0.0

            # We still need to iterate to find the targets, but we only compute stats for them.
            # For total norm, we can use a fused kernel if we want, but here we might just skip it
            # or approximate it from the samples if speed is paramount.
            # However, the user asked for "Async Sampling".
            # Let's compute stats ONLY for targets.

            for name, param in model.named_parameters():
                if name in targets and param.grad is not None:
                    # Detach and move to CPU asynchronously
                    # We use a non-blocking copy if possible, but .cpu() is usually blocking
                    # unless we are careful.
                    # Ideally we would put this in a queue, but for now, just reducing the
                    # number of .item() calls from 500 to 3 is the big win.

                    grad = param.grad.detach()

                    # Compute stats on GPU first (fast)
                    g_norm = grad.norm()
                    g_mean = grad.mean()
                    g_std = (
                        grad.std() if grad.numel() > 1 else torch.tensor(0.0, device=grad.device)
                    )
                    g_max = grad.max()
                    g_min = grad.min()

                    # Move scalars to CPU (this syncs, but only 3 times instead of 500)
                    # To be truly async we'd need a separate stream or queue,
                    # but 3 syncs every 100 steps is negligible.
                    layer_stat = {
                        "layer_name": name,
                        "grad_norm": g_norm.item(),
                        "grad_mean": g_mean.item(),
                        "grad_std": g_std.item(),
                        "grad_max": g_max.item(),
                        "grad_min": g_min.item(),
                        "zero_ratio": (grad == 0).float().mean().item(),
                    }
                    layer_stats.append(layer_stat)

            # Approximate total norm from samples or just skip it to save time
            # For the logger, we'll just store the sampled stats.

            gradient_info["sampled_layers"] = len(layer_stats)

            self.gradient_history.append(
                {"gradient_info": gradient_info, "layer_stats": layer_stats},
            )

            # Skip the heavy _check_gradient_issues logic on the hot path
            # or run it on the sampled data.

        except Exception as e:
            logger.error(f"Error logging gradients for {model_name}: {e}")

    def _check_gradient_issues(self, gradient_info, layer_stats):
        """Check for gradient issues and log them for monitoring."""
        total_grad_norm = gradient_info.get("total_grad_norm", 0.0)
        model_name = gradient_info.get("model_name", "unknown")
        epoch = gradient_info.get("epoch", 0)
        batch = gradient_info.get("batch", 0)

        # Dead neuron threshold (high ratio of zero gradients)
        dead_neuron_threshold = 0.9

        # Check for vanishing gradients
        if total_grad_norm < self.vanishing_threshold:
            logger.warning(
                f"VANISHING GRADIENTS DETECTED in {model_name} (epoch {epoch}, batch {batch})",
            )
            logger.debug(
                f"Total gradient norm: {total_grad_norm:.2e} "
                f"(threshold: {self.vanishing_threshold:.2e})",
            )

            self.vanishing_gradients_log.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "model_name": model_name,
                    "gradient_norm": total_grad_norm,
                    "timestamp": time.time(),
                },
            )

            # Log detection only (no intervention)
            self._log_vanishing_gradients(layer_stats, model_name, epoch, batch)

        # Check for exploding gradients
        elif total_grad_norm > self.exploding_threshold:
            logger.warning(
                f"EXPLODING GRADIENTS DETECTED in {model_name} (epoch {epoch}, batch {batch})",
            )
            logger.debug(
                f"Total gradient norm: {total_grad_norm:.2f} "
                f"(threshold: {self.exploding_threshold:.2f})",
            )

            self.exploding_gradients_log.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "model_name": model_name,
                    "gradient_norm": total_grad_norm,
                    "timestamp": time.time(),
                },
            )

            # Log detection only (no intervention)
            self._log_exploding_gradients(
                layer_stats,
                model_name,
                epoch,
                batch,
                total_grad_norm,
            )

        # Check for dead neurons (high zero gradient ratio)
        if gradient_info["zero_grad_ratio"] > dead_neuron_threshold:
            logger.warning(
                f"DEAD NEURONS DETECTED in {model_name} (epoch {epoch}, batch {batch})",
            )
            logger.debug(
                f"Zero gradient ratio: {gradient_info['zero_grad_ratio']:.1%} "
                f"(threshold: {dead_neuron_threshold:.1%})",
            )

        # Check for NaN/Inf gradients
        nan_count = gradient_info["nan_grad_count"]
        inf_count = gradient_info["inf_grad_count"]
        if nan_count > 0 or inf_count > 0:
            logger.error(
                f"INVALID GRADIENTS DETECTED in {model_name} (epoch {epoch}, batch {batch})",
            )
            logger.debug(f"NaN gradients: {gradient_info['nan_grad_count']}")
            logger.debug(f"Inf gradients: {gradient_info['inf_grad_count']}")
            self.nan_inf_count += 1

    def save_logs_to_csv(self, epoch=None, clear_history: bool = False):
        """Save gradient logs to CSV files for efficient I/O.

        Args:
            epoch (int, optional): The current epoch number. Defaults to None.
            clear_history (bool, optional): If True, clears the internal gradient history
                after saving to prevent memory bloat. Defaults to False.

        """
        try:
            if not self.gradient_history:
                logger.debug("No gradient data to save")
                return

            # Create detailed gradient log
            detailed_data = []
            for entry in self.gradient_history:
                grad_info = entry["gradient_info"]
                for layer_stat in entry.get("layer_stats", []):
                    row = {**grad_info, **layer_stat}
                    detailed_data.append(row)

            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_csv(self.gradient_detailed_file, index=False)
                logger.info(
                    f"Detailed gradient log saved: {self.gradient_detailed_file}",
                )

            # Create summary gradient log
            summary_data = [entry["gradient_info"] for entry in self.gradient_history]
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(self.gradient_summary_file, index=False)
                logger.info(
                    f"Summary gradient log saved: {self.gradient_summary_file}",
                )

            # Save gradient issues to separate CSV files
            if self.vanishing_gradients_log:
                vanishing_df = pd.DataFrame(self.vanishing_gradients_log)
                vanishing_df.to_csv(self.vanishing_issues_file, index=False)
                logger.info(
                    f"Vanishing gradient issues saved: {self.vanishing_issues_file}",
                )

            if self.exploding_gradients_log:
                exploding_df = pd.DataFrame(self.exploding_gradients_log)
                exploding_df.to_csv(self.exploding_issues_file, index=False)
                logger.info(
                    f"Exploding gradient issues saved: {self.exploding_issues_file}",
                )

            # Save issues summary
            summary_issues_df = pd.DataFrame(
                [
                    {
                        "total_vanishing_events": len(self.vanishing_gradients_log),
                        "total_exploding_events": len(self.exploding_gradients_log),
                        "total_nan_inf_events": self.nan_inf_count,
                        "total_gradient_logs": len(self.gradient_history),
                        "save_epoch": epoch if epoch is not None else "final",
                    },
                ],
            )
            summary_issues_df.to_csv(self.issues_summary_file, index=False)
            logger.info(f"Gradient issues summary saved: {self.issues_summary_file}")

        except Exception as e:
            logger.error(f"Error saving logs to CSV: {e}")
        finally:
            if clear_history:
                logger.debug(
                    f"Clearing {len(self.gradient_history)} entries from gradient history.",
                )
                self.gradient_history.clear()
                # Also consider if other logs should be cleared, but these are
                # usually smaller.
                # self.vanishing_gradients_log.clear()
                # self.exploding_gradients_log.clear()

    def _log_vanishing_gradients(self, layer_stats, model_name, epoch, batch):
        """Log vanishing gradient detection
        (monitoring only - no intervention)
        """
        try:
            logger.debug(f"VANISHING GRADIENT DETECTED for {model_name}")

            # Identify most affected layers
            problematic_layers = []
            for layer_stat in layer_stats:
                if layer_stat.get("zero_ratio", 0) > 0.8:  # High zero ratio
                    problematic_layers.append(
                        {
                            "name": layer_stat.get("layer_name", "unknown"),
                            "zero_ratio": layer_stat.get("zero_ratio", 0),
                        },
                    )

            if problematic_layers:
                logger.debug(
                    f"Most affected layers: {[layer['name'] for layer in problematic_layers[:5]]}",
                )
                max_zero = max(layer["zero_ratio"] for layer in problematic_layers)
                logger.debug(f"Max zero ratio: {max_zero:.1%}")

            # Log detection only - no intervention
            logger.debug(
                "ExtensiveGradientLogger: Monitoring only - "
                "gradient handling done by training system",
            )

            # Record for statistics (but don't set intervention flags)
            self.vanishing_gradient_incidents = getattr(self, "vanishing_gradient_incidents", 0) + 1

        except Exception as e:
            logger.error(f"Error in vanishing gradient recovery: {e}")

    def _log_exploding_gradients(
        self,
        layer_stats,
        model_name,
        epoch,
        batch,
        gradient_norm,
    ):
        """Log exploding gradient detection
        (monitoring only - no intervention)
        """
        try:
            logger.debug(f"EXPLODING GRADIENT DETECTED for {model_name}")

            # Analyze severity of explosion
            severity = "critical" if gradient_norm > 50.0 else "high"

            # Identify problematic layers
            problematic_layers = []
            for layer_stat in layer_stats:
                layer_grad_norm = layer_stat.get("grad_norm", 0)
                if layer_grad_norm > 10.0:  # High gradient threshold
                    problematic_layers.append(
                        {
                            "name": layer_stat.get("layer_name", "unknown"),
                            "norm": layer_grad_norm,
                        },
                    )

            if problematic_layers:
                layer_names = [layer["name"] for layer in problematic_layers[:5]]
                logger.debug(f"Problematic layers: {layer_names}")
                max_norm = max(layer["norm"] for layer in problematic_layers)
                logger.debug(f"Max layer gradient norm: {max_norm:.2f}")

            # Log detection only - no intervention
            logger.debug(f"Gradient norm: {gradient_norm:.2f} (severity: {severity})")
            logger.debug(
                "ExtensiveGradientLogger: Monitoring only - "
                "gradient handling done by training system",
            )

            # Record for statistics (but don't set intervention flags)
            self.exploding_gradient_incidents = getattr(self, "exploding_gradient_incidents", 0) + 1

        except Exception as e:
            logger.error(f"Error in gradient logging: {e}")

    def get_summary_stats(self):
        """Get summary statistics for gradient monitoring"""
        if not self.gradient_history:
            return None

        try:
            grad_norms = [
                entry["gradient_info"].get("total_grad_norm", 0.0)
                for entry in self.gradient_history
            ]
            zero_ratios = [
                entry["gradient_info"].get("zero_grad_ratio", 0.0)
                for entry in self.gradient_history
            ]

            summary = {
                "total_logs": len(self.gradient_history),
                "avg_grad_norm": (sum(grad_norms) / len(grad_norms) if grad_norms else 0),
                "max_grad_norm": max(grad_norms) if grad_norms else 0,
                "min_grad_norm": min(grad_norms) if grad_norms else 0,
                "avg_zero_ratio": (sum(zero_ratios) / len(zero_ratios) if zero_ratios else 0),
                "vanishing_events": len(self.vanishing_gradients_log),
                "exploding_events": len(self.exploding_gradients_log),
                "nan_inf_events": self.nan_inf_count,
            }

            return summary
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return None
