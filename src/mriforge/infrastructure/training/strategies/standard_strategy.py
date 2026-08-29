from typing import Any

import torch
import torch.nn as nn

from mriforge.infrastructure.training.builders.environment import TrainingEnvironment
from mriforge.infrastructure.training.step_io import accepts_step_io
from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class StandardTrainingStrategy(BaseTrainingStrategy):
    """Standard supervised training strategy with custom loss function.

    Implements basic supervised learning loop for arbitrary criterion functions.
    Flexible approach supporting any PyTorch loss module without paradigm-specific
    logic. Useful for simple tasks or custom loss experiments.

    ## Concept

    **Objective**: Simple supervised learning via forward-backward-update cycle:
    1. Forward: predictions = generator(inputs)
    2. Backward: loss = criterion(predictions, targets)
    3. Update: optimizer.step() with loss gradients

    Contrasts with specialized strategies (GAN, Diffusion, etc.) which enforce
    specific training paradigms. This strategy defers all loss computation to
    the provided criterion.

    ## Usage Pattern

    1. **Define Criterion**: Any differentiable PyTorch loss
       - Built-in: nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, etc.
       - Custom: Custom loss module with forward(pred, target) → scalar

    2. **Initialize Strategy**:
       ```python
       criterion = nn.L1Loss()
       strategy = StandardTrainingStrategy(
           criterion=criterion,
           device='cuda'
       )
       ```

    3. **Train Loop** (inherited from BaseTrainingStrategy):
       - Calls _compute_losses_impl → criterion(predictions, targets)
       - Returns dict with 'loss' key
       - AMP, gradient clipping, etc. applied automatically

    ## Configuration

    - Loss function: Provided directly (``criterion`` arg), not configuration-based
      (the strategy reads no ``training.*`` knob; the prior docstring advertised
      an unread ``training_mode`` knob — removed, audit 2026-06)
    - No paradigm-specific params (no GAN discriminator, no diffusion steps)

    ## Loss Components

    - **Primary Loss**: criterion(predictions, targets) (100%)
    - No secondary losses, regularization, or weighting
    - Flexibility: Criterion can internally combine multiple losses

    ## Key Features

    ✅ **Flexible**: Works with any differentiable loss function
    ✅ **Simple**: Minimal overhead, no paradigm-specific logic
    ✅ **Custom**: Easy to experiment with novel loss designs
    ✅ **Transparent**: All loss logic in criterion, fully controllable

    ## Limitations

    - ❌ **No Auto-Balancing**: Loss weights managed by criterion, not strategy
    - ❌ **No Physics**: No automatic data consistency, physics constraints
    - ❌ **No Multi-Task**: Single loss only, no secondary objectives
    - ❌ **No Scheduling**: Learning rate scheduling not built-in (see config)

    ## Typical Use Cases

    1. **Prototyping**: Quick experiments with new loss functions
    2. **Baselines**: Simple supervised reconstruction baseline
    3. **Custom Research**: Proprietary loss functions not in standard strategies
    4. **Testing**: Validate loss implementation before integration

    ## Inference

    1. Load pre-trained generator
    2. Forward pass: output = generator(input)
    3. Output: Generated/reconstructed output

    ## Comparison with Other Strategies

    | Feature | Standard | Reconstruction | GAN | Diffusion |
    |---------|----------|----------------|-----|-----------|
    | Loss Function | Custom criterion | Config-based (L1/L2/SSIM) | Adversarial | Score-based |
    | Complexity | Minimal | Low | High (discriminator) | Medium |
    | Flexibility | Maximum | Medium | Low | Low |
    | Physics Support | No | Optional (config) | Optional | Optional |
    | Training Stability | Depends on criterion | Stable | Unstable | Stable |

    Attributes:
        state: TrainingState with generator
        criterion: Loss function (nn.Module with forward(pred, target) → Tensor)
        device: Computation device (CUDA/CPU)

    Examples:
        >>> # Example 1: L1 reconstruction
        >>> criterion = nn.L1Loss()
        >>> strategy = StandardTrainingStrategy(criterion=criterion, device='cuda')
        >>>
        >>> # Example 2: Custom loss
        >>> class CustomLoss(nn.Module):
        ...     def forward(self, pred, target):
        ...         l1 = nn.functional.l1_loss(pred, target)
        ...         l2 = nn.functional.mse_loss(pred, target)
        ...         return l1 + 0.1 * l2
        >>>
        >>> criterion = CustomLoss()
        >>> strategy = StandardTrainingStrategy(criterion=criterion, device='cuda')
    """

    def __init__(
        self,
        criterion: nn.Module,
        env: TrainingEnvironment,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            criterion (nn.Module): Loss criterion for supervised training.
            env (TrainingEnvironment): Training environment (REQUIRED — the base
                strategy raises ``TypeError`` on a ``None`` env, so the previous
                ``env=None`` default could only ever crash at construction).
        """
        super().__init__(env=env, **kwargs)
        self.criterion = criterion

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Forward pass
        """_compute_losses_impl.

        Args:
            input_batch (torch.Tensor): Description.
            target_batch (torch.Tensor): Description.
            epoch (int): Description.
        Returns:
            dict[str, torch.Tensor]: Description.
        """
        predictions = self.env.generator(input_batch)

        # Compute loss
        loss = self.criterion(predictions, target_batch)

        return {"loss": loss, "g_total_loss": loss}

    @accepts_step_io
    @torch.no_grad()
    def validation_step(
        self,
        batch: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """validation_step.

        Args:
            batch (Any): Description.
        Returns:
            dict[str, float]: Description.
        """
        input_batch, target_batch = self._unpack_batch(batch)

        self.env.generator.eval()
        predictions = self.env.generator(input_batch)
        loss = self.criterion(predictions, target_batch)

        return {"val_loss": loss.item()}
