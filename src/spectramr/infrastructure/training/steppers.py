import torch
import torch.nn as nn

from spectramr.infrastructure.training.interfaces import IOptimizerStepper


class StandardOptimizerStepper(IOptimizerStepper):
    """Standard implementation of optimizer stepper handling AMP and clipping."""

    def __init__(
        self,
        scaler: torch.amp.GradScaler | None = None,
        gradient_clipping: float | None = None,
        use_amp: bool = False,
    ):
        """__init__.

        Args:
            scaler (Optional[torch.amp.GradScaler]): Description.
            gradient_clipping (Optional[float]): Description.
            use_amp (bool): Description.
        """
        self.scaler = scaler
        self.gradient_clipping = gradient_clipping
        self.use_amp = use_amp

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """scale_loss.

        Args:
            loss (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def perform_backward(self, loss: torch.Tensor) -> None:
        """perform_backward.

        Args:
            loss (torch.Tensor): Description.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor | None = None) -> bool:
        """step.

        Args:
            optimizer (torch.optim.Optimizer): Description.
            loss (Optional[torch.Tensor]): Description.
        Returns:
            bool: Description.
        """
        if self.scaler is not None:
            if self.gradient_clipping is not None:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self._get_params(optimizer), self.gradient_clipping)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if self.gradient_clipping is not None:
                nn.utils.clip_grad_norm_(self._get_params(optimizer), self.gradient_clipping)
            optimizer.step()
        return True

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """zero_grad.

        Args:
            optimizer (torch.optim.Optimizer): Description.
        """
        optimizer.zero_grad(set_to_none=True)

    def _get_params(self, optimizer: torch.optim.Optimizer):
        """_get_params.

        Args:
            optimizer (torch.optim.Optimizer): Description.
        Returns:
            Any: Description.
        """
        params = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                params.append(p)
        return params
