from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class MAMLWrapper(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) Wrapper.

    Enables meta-learning by wrapping a model and providing functionality
    to perform inner-loop updates (adaptation) that are differentiable
    in the outer loop.

    Reference:
        Finn, C., et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation
        of Deep Networks. ICML.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01, first_order: bool = False):
        """__init__.

        Args:
            model (nn.Module): Description.
            inner_lr (float): Description.
            first_order (bool): Description.
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.first_order = first_order  # If True, ignore second-order derivatives (FOMAML)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        num_inner_steps: int = 1,
        functional_forward: Callable = None,
    ) -> torch.Tensor:
        """
        Executes MAML loop:
        1. Clone initial weights theta.
        2. Perform inner updates on Support Set (theta -> theta_prime).
        3. Evaluate theta_prime on Query Set.

        Args:
            support_x: Support set inputs
            support_y: Support set targets
            query_x: Query set inputs
            num_inner_steps: Number of gradient steps in inner loop
            functional_forward: Optional custom forward function that accepts (x, params).
                                If None, tries to use model's standard forward logic via monkey-patching or stateless API.

        Returns:
            Predictions on query_x using adapted weights.

        forward method for MAMLWrapper.

        Executes PyTorch tensor operations.

        Args:
            support_x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            support_y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            query_x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            num_inner_steps (int): Expected input tensor.
            functional_forward (Callable): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Capture initial weights (fast weights)
        # We need a way to run the model with functional parameters.
        # PyTorch >= 2.0 has torch.func (functional) which is best.
        # Fallback: manually handle parameters.

        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # 2. Inner Loop (Adaptation)
        for _ in range(num_inner_steps):
            # Forward on support set using current fast_weights
            support_pred = self._exec_forward(support_x, fast_weights, functional_forward)

            # Loss
            loss = F.mse_loss(support_pred, support_y)

            # Gradients with respect to fast_weights
            # create_graph=True ensures we can backprop through this update later (unless first_order)
            grads = autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Update fast_weights (SGD)
            # theta' = theta - alpha * grad
            fast_weights = {
                name: param - self.inner_lr * grad if grad is not None else param
                for (name, param), grad in zip(fast_weights.items(), grads, strict=False)
            }

        # 3. Outer Loop (Query)
        # Forward on query set using adapted weights
        query_pred = self._exec_forward(query_x, fast_weights, functional_forward)

        return query_pred

    def _exec_forward(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor],
        custom_forward: Callable = None,
    ) -> torch.Tensor:
        """
        Executes model forward pass using functional parameters.
        """
        if custom_forward:
            return custom_forward(x, params)

        # Use torch.func if available (PyTorch 2.0+)
        if hasattr(torch, "func") and hasattr(torch.func, "functional_call"):
            return torch.func.functional_call(self.model, params, (x,))

        # Fallback for older environments or if torch.func is missing
        raise RuntimeError(
            "MAMLWrapper requires torch.func (PyTorch 2.0+) or a 'functional_forward' callable."
        )
