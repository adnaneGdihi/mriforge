"""Model stability analysis and gradient monitoring utilities."""

import numpy as np
import torch
from torch import nn


class GradientStabilityAnalyzer:
    """Analyzes gradient stability and characteristics during training."""

    def __init__(self):
        """Initialize the gradient stability analyzer."""

    def analyze_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module | None = None,
    ) -> dict[str, float]:
        """Analyze gradient characteristics of a model.

        Args:
            model: The neural network model to analyze
            x: Input tensor
            y: Target tensor
            criterion: Loss function (default: MSELoss)

        Returns:
            Dictionary containing gradient metrics

        """
        if criterion is None:
            criterion = nn.MSELoss()

        # Ensure model is in training mode
        was_training = model.training
        model.train()

        # Forward pass
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        loss.backward()

        # Collect gradient statistics
        grad_norms = []
        grad_means = []
        grad_stds = []
        grad_maxes = []

        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_count += 1
                grad = param.grad.data
                grad_norms.append(torch.norm(grad).item())
                grad_means.append(torch.mean(torch.abs(grad)).item())
                # Only compute std if tensor has more than 1 element
                if grad.numel() > 1:
                    grad_stds.append(torch.std(grad).item())
                grad_maxes.append(torch.max(torch.abs(grad)).item())

        # Restore original training mode
        model.train(was_training)

        # Clear gradients
        model.zero_grad()

        return {
            "gradient_norm": np.mean(grad_norms) if grad_norms else 0.0,
            "gradient_std": np.mean(grad_stds) if grad_stds else 0.0,
            "gradient_max": np.max(grad_maxes) if grad_maxes else 0.0,
            "gradient_mean": np.mean(grad_means) if grad_means else 0.0,
            "param_count": param_count,
        }

    def analyze_layer_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """Analyze gradients at the layer level.

        Args:
            model: The neural network model
            x: Input tensor
            y: Target tensor

        Returns:
            Dictionary mapping layer names to gradient statistics

        """
        criterion = nn.MSELoss()

        # Forward pass
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        layer_gradients = {}

        for name, module in model.named_modules():
            if not list(module.parameters()):
                continue

            layer_norms = []
            layer_means = []

            for param in module.parameters():
                if param.grad is not None:
                    layer_norms.append(torch.norm(param.grad.data).item())
                    layer_means.append(torch.mean(torch.abs(param.grad.data)).item())

            if layer_norms:
                layer_gradients[name] = {
                    "norm": np.mean(layer_norms),
                    "mean": np.mean(layer_means),
                    "max_norm": np.max(layer_norms),
                }

        model.zero_grad()
        return layer_gradients


class TrainingStabilityMonitor:
    """Monitors training stability over time."""

    def __init__(self, window_size: int = 10, convergence_threshold: float = 1e-4):
        """Initialize the training stability monitor.

        Args:
            window_size: Number of recent values to consider for
                         stability analysis
            convergence_threshold: Threshold for convergence detection

        """
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.loss_history: list[float] = []
        self.gradient_history: list[float] = []

    def record_loss(self, loss: float):
        """Record a loss value."""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

    def record_gradient_norm(self, grad_norm: float):
        """Record a gradient norm value."""
        self.gradient_history.append(grad_norm)
        if len(self.gradient_history) > self.window_size:
            self.gradient_history.pop(0)

    def detect_convergence(self) -> bool:
        """Detect if training has converged based on loss history."""
        if len(self.loss_history) < self.window_size:
            return False

        # Check if loss has been decreasing and is below threshold
        recent_losses = self.loss_history[-self.window_size :]
        min_loss = min(recent_losses)
        max_loss = max(recent_losses)

        # Convergence if the range is small relative to the minimum loss
        # OR if losses are absolutely small (for very converged models)
        if min_loss > 0:
            relative_range = (max_loss - min_loss) / min_loss
            if relative_range < self.convergence_threshold:
                return True

        # Absolute convergence for very small losses
        return max_loss < self.convergence_threshold

    def detect_instability(self, variance_threshold: float = 0.1) -> bool:
        """Detect training instability based on loss variance."""
        if len(self.loss_history) < self.window_size:
            return False

        recent_losses = self.loss_history[-self.window_size :]

        # Calculate coefficient of variation
        mean_loss = np.mean(recent_losses)
        if mean_loss == 0:
            return False

        std_loss = np.std(recent_losses)
        cv = std_loss / mean_loss

        return cv > variance_threshold

    def get_loss_statistics(self) -> dict[str, float]:
        """Get statistics of recent loss values."""
        if not self.loss_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        losses = np.array(self.loss_history)
        return {
            "mean": np.mean(losses),
            "std": np.std(losses),
            "min": np.min(losses),
            "max": np.max(losses),
        }

    def get_gradient_statistics(self) -> dict[str, float]:
        """Get statistics of recent gradient norms."""
        if not self.gradient_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        grads = np.array(self.gradient_history)
        return {
            "mean": np.mean(grads),
            "std": np.std(grads),
            "min": np.min(grads),
            "max": np.max(grads),
        }


class ModelConvergenceChecker:
    """Checks for model convergence during training."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """Initialize the convergence checker.

        Args:
            patience: Number of epochs to wait before declaring convergence
            min_delta: Minimum change in loss to be considered significant

        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait_count = 0

    def check_convergence(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        patience: int | None = None,
        min_delta: float | None = None,
    ) -> bool:
        """Check if model has converged.

        Args:
            model: The model to check
            x: Input data
            y: Target data
            patience: Override default patience
            min_delta: Override default min_delta

        Returns:
            True if converged, False otherwise

        """
        if patience is not None:
            self.patience = patience
        if min_delta is not None:
            self.min_delta = min_delta

        # Compute current loss
        criterion = nn.MSELoss()
        with torch.no_grad():
            output = model(x)
            current_loss = criterion(output, y).item()

        # Check for improvement
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
            return False
        self.wait_count += 1
        return self.wait_count >= self.patience

    def check_convergence_with_history(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_history: list[float],
    ) -> bool:
        """Check convergence using provided loss history.

        Args:
            model: The model (unused, for API compatibility)
            x: Input data (unused, for API compatibility)
            y: Target data (unused, for API compatibility)
            loss_history: list of recent loss values

        Returns:
            True if converged based on history

        """
        if len(loss_history) < 2:
            return False

        # Check if losses are stabilizing
        recent_losses = loss_history[-min(len(loss_history), self.patience + 1) :]
        if len(recent_losses) < 2:
            return False

        # Check if the change in recent losses is below threshold
        max_recent = max(recent_losses)
        min_recent = min(recent_losses)

        if min_recent == 0:
            return max_recent < self.min_delta

        relative_change = (max_recent - min_recent) / min_recent
        return relative_change < self.min_delta

    def should_early_stop(self, loss_history: list[float]) -> bool:
        """Determine if training should early stop based on loss history.

        Args:
            loss_history: list of loss values over training

        Returns:
            True if should early stop

        """
        return self.check_convergence_with_history(None, None, None, loss_history)

    def reset(self):
        """Reset the convergence checker state."""
        self.best_loss = float("inf")
        self.wait_count = 0
