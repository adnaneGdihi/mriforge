"""Stability Linter Module
=======================

Provides comprehensive stability analysis for deep learning models,
particularly GANs and diffusion models. This module analyzes model
architecture, parameter distributions, gradient flow, and training
stability characteristics.

Features:
- Architectural stability analysis
- Parameter distribution analysis
- Gradient flow analysis
- Training stability assessment
- Comprehensive reporting and scoring

Author: MRIForge Research Project
Date: January 2025
"""

import logging

logger = logging.getLogger(__name__)

import json
import os
from typing import Any

import numpy as np
import torch
from torch import nn


class StabilityAnalyzer:
    """Comprehensive stability analyzer for deep learning models."""

    def __init__(self, model: nn.Module, model_type: str):
        """Initialize the stability analyzer.

        Args:
            model: PyTorch model to analyze
            model_type: Type/name of the model

        """
        self.model = model
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the correct device
        self.model = self.model.to(self.device)

        # Analysis results
        self.analysis_results = {
            "architecture_score": 0,
            "parameter_score": 0,
            "gradient_score": 0,
            "training_score": 0,
            "overall_score": 0,
        }

        self.issues = []
        self.recommendations = []

    def analyze_architecture(self) -> dict[str, Any]:
        """Analyze model architecture for stability issues.

        Returns:
            Dictionary with architecture analysis results

        """
        results = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "layer_count": 0,
            "activation_functions": set(),
            "normalization_layers": [],
            "skip_connections": False,
            "residual_connections": False,
        }

        def count_parameters(module):
            """count_parameters.

            Args:
                module (Any): Description.
            Returns:
                Any: Description.
            """
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        def analyze_module(module):
            """analyze_module.

            Args:
                module (Any): Description.
            Returns:
                Any: Description.
            """
            nonlocal results
            results["layer_count"] += 1

            # Count parameters
            total, trainable = count_parameters(module)
            results["total_parameters"] += total
            results["trainable_parameters"] += trainable

            # Check for normalization layers
            norm_types = (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.InstanceNorm2d,
            )
            if isinstance(module, norm_types):
                results["normalization_layers"].append(type(module).__name__)

            # Check for activation functions
            act_types = (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)
            if isinstance(module, act_types):
                results["activation_functions"].add(type(module).__name__)

            # Check for residual/skip connections (heuristic)
            module_name = type(module).__name__.lower()
            if hasattr(module, "residual") or "residual" in module_name:
                results["residual_connections"] = True
            if hasattr(module, "skip") or "skip" in module_name:
                results["skip_connections"] = True

        self.model.apply(lambda m: analyze_module(m))

        # Calculate architecture score
        score = 100

        # Penalize very large models
        if results["total_parameters"] > 100_000_000:  # 100M+
            score -= 20
            self.issues.append(
                "Very large model (>100M parameters) - may be unstable",
            )
        elif results["total_parameters"] > 30_000_000:  # 30M+
            score -= 10
            self.issues.append(
                "Large model (>30M parameters) - monitor memory usage",
            )

        # Reward normalization layers
        if len(results["normalization_layers"]) == 0:
            score -= 15
            self.issues.append(
                "No normalization layers found - may cause training instability",
            )
        elif len(results["normalization_layers"]) > results["layer_count"] * 0.1:
            score += 5  # Good normalization coverage

        # Reward residual/skip connections
        if results["residual_connections"] or results["skip_connections"]:
            score += 10
            self.recommendations.append(
                "Good use of residual/skip connections for gradient flow",
            )

        results["score"] = max(0, min(100, score))
        return results

    def analyze_parameters(self) -> dict[str, Any]:
        """Analyze parameter distributions and initialization.

        Returns:
            Dictionary with parameter analysis results

        """
        results = {
            "parameter_stats": {},
            "initialization_quality": 0,
            "gradient_flow_potential": 0,
        }

        if not list(self.model.parameters()):
            results["score"] = 50  # Neutral score for parameterless models
            return results

        all_params = []
        all_grads = []

        # Sample parameters for large models to avoid memory issues
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > 10_000_000:  # 10M+ params, sample 10%
            sample_ratio = 0.1
        elif total_params > 1_000_000:  # 1M+ params, sample 25%
            sample_ratio = 0.25
        else:
            sample_ratio = 1.0

        param_count = 0
        for _name, param in self.model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy().flatten()

                # Sample parameters for large models
                if sample_ratio < 1.0:
                    sample_size = max(1, int(len(param_data) * sample_ratio))
                    indices = np.random.choice(
                        len(param_data),
                        sample_size,
                        replace=False,
                    )
                    param_data = param_data[indices]

                all_params.extend(param_data)

                if param.grad is not None:
                    grad_data = param.grad.data.cpu().numpy().flatten()

                    # Sample gradients too
                    if sample_ratio < 1.0:
                        sample_size = max(1, int(len(grad_data) * sample_ratio))
                        indices = np.random.choice(
                            len(grad_data),
                            sample_size,
                            replace=False,
                        )
                        grad_data = grad_data[indices]

                    all_grads.extend(grad_data)

                param_count += 1
                # Limit to first 100 parameters for very large models
                if total_params > 100_000_000 and param_count > 100:
                    break

        if all_params:
            params_array = np.array(all_params)

            results["parameter_stats"] = {
                "mean": float(np.mean(params_array)),
                "std": float(np.std(params_array)),
                "min": float(np.min(params_array)),
                "max": float(np.max(params_array)),
                "zeros_ratio": float(
                    np.count_nonzero(params_array == 0) / len(params_array),
                ),
            }

            # Analyze initialization quality
            std = results["parameter_stats"]["std"]
            if 0.01 < std < 1.0:
                results["initialization_quality"] = 80
            elif std < 0.01:
                results["initialization_quality"] = 40
                self.issues.append(
                    "Parameters have very low variance - poor initialization",
                )
            else:
                results["initialization_quality"] = 60
                self.issues.append(
                    "Parameters have high variance - may cause instability",
                )

        # Analyze gradient flow if gradients exist
        if all_grads:
            grads_array = np.array(all_grads)
            grad_std = float(np.std(grads_array))
            results["gradient_flow_potential"] = min(100, grad_std * 1000)
            # Scale for scoring

        # Calculate parameter score
        score = 70  # Base score

        if results["parameter_stats"]:
            zeros_ratio = results["parameter_stats"]["zeros_ratio"]
            if zeros_ratio > 0.5:
                score -= 20
                self.issues.append(
                    "High ratio of zero parameters - dead neurons possible",
                )
            elif zeros_ratio < 0.1:
                score += 10

        results["score"] = max(0, min(100, score))
        return results

    def analyze_gradient_flow(self, lightweight: bool = False) -> dict[str, Any]:
        """Analyze gradient flow characteristics with optional lightweight mode.

        Args:
            lightweight: If True, skip expensive gradient computation for
                large models

        Returns:
            Dictionary with gradient flow analysis results

        """
        results = {
            "gradient_stats": {},
            "vanishing_gradient_risk": 0,
            "exploding_gradient_risk": 0,
        }

        # Skip expensive gradient analysis for very large models in
        # lightweight mode
        total_params = sum(p.numel() for p in self.model.parameters())
        if lightweight and total_params > 50_000_000:  # 50M+ params
            results["score"] = 60  # Neutral score
            results["lightweight_mode"] = True
            self.recommendations.append(
                "Large model - gradient analysis skipped for speed",
            )
            return results

        # Create dummy input for gradient analysis
        try:
            # Try to infer input shape from first layer
            first_layer = None
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                    first_layer = module
                    break

            if first_layer is None:
                results["score"] = 50
                return results

            # Create smaller dummy input for speed
            if isinstance(first_layer, nn.Linear):
                dummy_input = torch.randn(2, first_layer.in_features)
                # Smaller batch
            elif isinstance(first_layer, nn.Conv2d):
                # Smaller spatial dimensions
                dummy_input = torch.randn(2, first_layer.in_channels, 32, 32)
            elif isinstance(first_layer, nn.Conv3d):
                # Smaller 3D dimensions
                dummy_input = torch.randn(1, first_layer.in_channels, 16, 16, 16)
            else:
                dummy_input = torch.randn(2, 1, 32, 32)  # Default smaller

            dummy_input = dummy_input.to(self.device)
            self.model.zero_grad()

            # Forward pass
            output = self.model(dummy_input)
            if isinstance(output, tuple):
                output = output[0]

            # Backward pass
            loss = output.mean()
            loss.backward()

            # Analyze gradients more efficiently
            grad_norms = []
            zero_count = 0
            total_params_with_grad = 0

            # Process gradients in batches to avoid memory issues
            for _name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Compute norm on GPU if possible
                    grad_norm = param.grad.data.norm().item()
                    grad_norms.append(grad_norm)
                    total_params_with_grad += 1

                    # Early detection of zero gradients
                    if grad_norm < 1e-8:
                        zero_count += 1

            if grad_norms:
                grad_norms = np.array(grad_norms)
                results["gradient_stats"] = {
                    "mean_norm": float(np.mean(grad_norms)),
                    "std_norm": float(np.std(grad_norms)),
                    "min_norm": float(np.min(grad_norms)),
                    "max_norm": float(np.max(grad_norms)),
                    "zero_gradients": zero_count,
                    "total_params_with_grad": total_params_with_grad,
                }

                # Analyze gradient health
                mean_norm = results["gradient_stats"].get("mean_norm", 1e-3)
                if mean_norm < 1e-6:
                    results["vanishing_gradient_risk"] = 90
                    self.issues.append("High risk of vanishing gradients")
                elif mean_norm < 1e-4:
                    results["vanishing_gradient_risk"] = 60
                    self.issues.append("Moderate risk of vanishing gradients")
                else:
                    results["vanishing_gradient_risk"] = 20

                if mean_norm > 10:
                    results["exploding_gradient_risk"] = 90
                    self.issues.append("High risk of exploding gradients")
                elif mean_norm > 1:
                    results["exploding_gradient_risk"] = 60
                    self.issues.append("Moderate risk of exploding gradients")
                else:
                    results["exploding_gradient_risk"] = 20

        except Exception as e:
            logger.warning(f"Gradient flow analysis failed: {e}")
            results["error"] = str(e)

        # Calculate gradient score
        score = 80  # Base score

        if results.get("gradient_stats"):
            zero_grads = results["gradient_stats"].get("zero_gradients", 0)
            total_params = results["gradient_stats"].get("total_params_with_grad", 1)
            zero_ratio = zero_grads / total_params if total_params > 0 else 0

            if zero_ratio > 0.5:
                score -= 30
                self.issues.append("High ratio of zero gradients - dead layers")
            elif zero_ratio > 0.2:
                score -= 15
                self.issues.append("Significant zero gradients detected")

        results["score"] = max(0, min(100, score))
        return results

    def analyze_training_stability(self) -> dict[str, Any]:
        """Analyze training stability characteristics.

        Returns:
            Dictionary with training stability analysis results

        """
        results = {
            "learning_rate_sensitivity": 50,  # Default neutral
            "batch_size_recommendation": "medium",
            "optimizer_recommendations": [],
            "regularization_needs": [],
        }

        # Analyze based on model characteristics
        total_params = sum(p.numel() for p in self.model.parameters())

        if total_params > 100_000_000:  # 100M+
            results["learning_rate_sensitivity"] = 80  # Very sensitive
            results["batch_size_recommendation"] = "small"
            results["optimizer_recommendations"] = ["AdamW", "Lookahead"]
            results["regularization_needs"] = ["weight_decay", "gradient_clipping"]
            self.recommendations.append(
                "Use small batch sizes (4-8) for large models",
            )
            self.recommendations.append(
                "Use conservative learning rates (1e-5 to 1e-4)",
            )

        elif total_params > 30_000_000:  # 30M+
            results["learning_rate_sensitivity"] = 70
            results["batch_size_recommendation"] = "medium"
            results["optimizer_recommendations"] = ["Adam", "AdamW"]
            results["regularization_needs"] = ["gradient_clipping"]
            self.recommendations.append("Use medium batch sizes (8-16)")

        else:  # Smaller models
            results["learning_rate_sensitivity"] = 40
            results["batch_size_recommendation"] = "large"
            results["optimizer_recommendations"] = ["Adam", "SGD"]
            self.recommendations.append("Can use larger batch sizes (16-32)")

        # Check for specific model types
        if "diffusion" in self.model_type.lower():
            results["learning_rate_sensitivity"] = 90
            self.recommendations.append(
                "Diffusion models are very sensitive to learning rate",
            )
            self.recommendations.append(
                "Use learning rate warmup for diffusion models",
            )

        if "gan" in self.model_type.lower() or "stylegan" in self.model_type.lower():
            results["regularization_needs"].append("spectral_normalization")
            self.recommendations.append("GANs benefit from spectral normalization")

        if "transformer" in self.model_type.lower() or "vit" in self.model_type.lower():
            results["regularization_needs"].append("dropout")
            self.recommendations.append(
                "Transformers benefit from dropout regularization",
            )

        # Calculate training score
        score = 100 - results["learning_rate_sensitivity"]
        # Inverse relationship
        results["score"] = max(0, min(100, score))

        return results

    def run_full_analysis(self) -> dict[str, Any]:
        """Run complete stability analysis.

        Returns:
            Dictionary with complete analysis results

        """
        logger.info(f"Running stability analysis for {self.model_type}...")

        # Run all analyses
        arch_results = self.analyze_architecture()
        param_results = self.analyze_parameters()

        # Use lightweight gradient analysis for large models
        total_params = sum(p.numel() for p in self.model.parameters())
        use_lightweight = total_params > 50_000_000  # 50M+ params
        grad_results = self.analyze_gradient_flow(lightweight=use_lightweight)

        train_results = self.analyze_training_stability()

        # Store individual scores
        self.analysis_results["architecture_score"] = arch_results["score"]
        self.analysis_results["parameter_score"] = param_results["score"]
        self.analysis_results["gradient_score"] = grad_results["score"]
        self.analysis_results["training_score"] = train_results["score"]

        # Calculate overall score (weighted average)
        weights = {
            "architecture_score": 0.3,
            "parameter_score": 0.2,
            "gradient_score": 0.3,
            "training_score": 0.2,
        }

        overall_score = sum(self.analysis_results[key] * weight for key, weight in weights.items())

        self.analysis_results["overall_score"] = int(round(overall_score, 1))

        # Compile final report
        report = {
            "model_type": self.model_type,
            "stability_score": self.analysis_results["overall_score"],
            "architecture_analysis": arch_results,
            "parameter_analysis": param_results,
            "gradient_analysis": grad_results,
            "training_analysis": train_results,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "timestamp": str(torch.randint(0, 1000000, (1,)).item()),
            # Simple timestamp
        }

        logger.info(
            f"Stability analysis completed for {self.model_type}. "
            f"Score: {report['stability_score']}/100",
        )

        return report


def run_stability_analysis(
    model: nn.Module,
    model_type: str,
    save_report: bool = True,
    report_path: str | None = None,
) -> dict[str, Any]:
    """Run comprehensive stability analysis on a model.

    Args:
        model: PyTorch model to analyze
        model_type: Type/name of the model
        save_report: Whether to save the report to file
        report_path: Path to save the report (optional)

    Returns:
        Dictionary containing stability analysis results

    """
    try:
        analyzer = StabilityAnalyzer(model, model_type)
        report = analyzer.run_full_analysis()

        if save_report and report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            # Save JSON report
            json_path = f"{report_path}.json"
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            # Save text summary
            txt_path = f"{report_path}.txt"
            with open(txt_path, "w") as f:
                f.write(f"Stability Analysis Report for {model_type}\n")
                f.write("=" * 50 + "\n\n")
                stability_score = report.get("stability_score", "N/A")
                if isinstance(stability_score, (int, float)):
                    f.write(f"Overall Stability Score: {stability_score}/100\n\n")
                else:
                    f.write(f"Overall Stability Score: {stability_score}\n\n")

                f.write("Architecture Analysis:\n")
                total_params = report["architecture_analysis"].get(
                    "total_parameters",
                    "N/A",
                )
                f.write(f"  - Total Parameters: {total_params:,}\n")
                trainable_params = report["architecture_analysis"].get(
                    "trainable_parameters",
                    "N/A",
                )
                f.write(f"  - Trainable Parameters: {trainable_params:,}\n")
                arch_score = report["architecture_analysis"].get("score", "N/A")
                f.write(f"  - Architecture Score: {arch_score}/100\n\n")

                f.write("Parameter Analysis:\n")
                if "parameter_stats" in report["parameter_analysis"]:
                    stats = report["parameter_analysis"]["parameter_stats"]
                    mean_val = stats.get("mean", "N/A")
                    std_val = stats.get("std", "N/A")
                    if isinstance(mean_val, (int, float)):
                        f.write(f"  - Mean: {mean_val:.6f}\n")
                    else:
                        f.write(f"  - Mean: {mean_val}\n")
                    if isinstance(std_val, (int, float)):
                        f.write(f"  - Std: {std_val:.6f}\n")
                    else:
                        f.write(f"  - Std: {std_val}\n")
                    param_score = report["parameter_analysis"].get("score", "N/A")
                    f.write(f"  - Parameter Score: {param_score}/100\n\n")

                f.write("Gradient Analysis:\n")
                if "gradient_stats" in report["gradient_analysis"]:
                    stats = report["gradient_analysis"]["gradient_stats"]
                    mean_norm = stats.get("mean_norm", "N/A")
                    zero_grads = stats.get("zero_gradients", "N/A")
                    if isinstance(mean_norm, (int, float)):
                        f.write(f"  - Mean Gradient Norm: {mean_norm:.6f}\n")
                    else:
                        f.write(f"  - Mean Gradient Norm: {mean_norm}\n")
                    f.write(f"  - Zero Gradients: {zero_grads}\n")
                    grad_score = report["gradient_analysis"].get("score", "N/A")
                    f.write(f"  - Gradient Score: {grad_score}/100\n\n")

                f.write("Training Analysis:\n")
                lr_sens = report["training_analysis"].get(
                    "learning_rate_sensitivity",
                    "N/A",
                )
                f.write(f"  - Learning Rate Sensitivity: {lr_sens}/100\n")
                batch_rec = report["training_analysis"].get(
                    "batch_size_recommendation",
                    "N/A",
                )
                f.write(f"  - Recommended Batch Size: {batch_rec}\n")
                train_score = report["training_analysis"].get("score", "N/A")
                f.write(f"  - Training Score: {train_score}/100\n\n")

                if report.get("issues"):
                    f.write("Issues Found:\n")
                    f.writelines(f"  - {issue}\n" for issue in report["issues"])
                    f.write("\n")

                if report.get("recommendations"):
                    f.write("Recommendations:\n")
                    f.writelines(f"  - {rec}\n" for rec in report["recommendations"])
                    f.write("\n")

            logger.info(f"Stability report saved to {json_path} and {txt_path}")

        return report

    except Exception as e:
        logger.error(f"Stability analysis failed for {model_type}: {e}")

        # Return minimal error report
        error_report = {
            "model_type": model_type,
            "stability_score": 0,
            "error": str(e),
            "issues": [f"Analysis failed: {e}"],
            "recommendations": ["Manual review recommended"],
            "timestamp": str(torch.randint(0, 1000000, (1,)).item()),
        }

        return error_report


# Convenience function for quick stability check
def quick_stability_check(model: nn.Module, model_type: str) -> float:
    """Quick stability check returning only the score.

    Args:
        model: PyTorch model to check
        model_type: Type/name of the model

    Returns:
        Stability score (0-100)

    """
    report = run_stability_analysis(model, model_type, save_report=False)
    return report.get("stability_score", 0)


if __name__ == "__main__":
    # Example usage
    from torch import nn

    # Create a simple test model
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

    # Run analysis
    report = run_stability_analysis(
        model,
        "test_model",
        save_report=True,
        report_path="./test_stability",
    )
    logger.debug(f"Stability Score: {report['stability_score']}/100")
