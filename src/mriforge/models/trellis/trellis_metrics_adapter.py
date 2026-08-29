"""Adapters for TRELLIS volume quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Import the unified metric registry
from mriforge.core.metrics.registry import MetricsRegistry


@dataclass
class TrellisMetricsResult:
    """Structured container for TRELLIS quality metrics."""

    psnr: float
    ssim: float
    mse: float
    nmse: float
    mae: float
    rmse: float

    def as_dict(self, prefix: str | None = None) -> dict[str, float]:
        """as_dict.

        Args:
            prefix (str | None): Description.
        Returns:
            dict[str, float]: Description.
        """
        if prefix:
            return {f"{prefix}_{k}": float(v) for k, v in self.__dict__.items()}
        return {k: float(v) for k, v in self.__dict__.items()}


class TrellisMetricsAdapter:
    """Bridges TRELLIS volume evaluations with the core metrics library."""

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        data_range: float = 2.0,
    ) -> None:
        """__init__.

        Args:
            device (str | torch.device): Description.
            data_range (float): Description.
        """
        self.device = str(device)
        self.data_range = data_range

    @torch.no_grad()
    def compute(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> TrellisMetricsResult:
        """Compute primary TRELLIS metrics for a prediction/target pair."""

        pred_tensor = self._prepare(prediction)
        target_tensor = self._prepare(target)

        def _get_val(name, p, t):
            """_get_val.

            Args:
                name (Any): Description.
                p (Any): Description.
                t (Any): Description.
            Returns:
                Any: Description.
            """
            m = MetricsRegistry.get(name, device=self.device, data_range=self.data_range)
            v = m(p, t)
            return v.item() if hasattr(v, "item") else float(v)

        psnr = _get_val("psnr", pred_tensor, target_tensor)
        ssim = _get_val("ssim", pred_tensor, target_tensor)
        mse = _get_val("mse", pred_tensor, target_tensor)
        # NMSE = MSE / target_variance
        target_var = torch.var(target_tensor).item()
        nmse = mse / target_var if target_var > 0 else mse
        mae = _get_val("mae", pred_tensor, target_tensor)
        rmse = _get_val("rmse", pred_tensor, target_tensor)

        return TrellisMetricsResult(
            psnr=psnr,
            ssim=ssim,
            mse=mse,
            nmse=nmse,
            mae=mae,
            rmse=rmse,
        )

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> TrellisMetricsResult:
        """__call__.

        Args:
            prediction (Tensor): Description.
            target (Tensor): Description.
        Returns:
            TrellisMetricsResult: Description.
        """
        return self.compute(prediction, target)

    def _prepare(self, tensor: Tensor) -> Tensor:
        """_prepare.

        Args:
            tensor (Tensor): Description.
        Returns:
            Tensor: Description.
        """
        tensor = tensor.to(torch.float32)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 5:
            raise ValueError(
                "TRELLIS metrics expect tensors with 3D volume semantics",
            )

        if tensor.ndim == 4:  # shape (1, C, H, W) -> treat as depth 1
            tensor = tensor.unsqueeze(2)

        batch, channels, depth, height, width = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3, 4).reshape(
            batch * depth,
            channels,
            height,
            width,
        )
        return tensor.to(self.device)

    def summarize(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        prefix: str | None = None,
    ) -> dict[str, float]:
        """Return metrics as a dictionary optionally namespaced with prefix."""

        return self.compute(prediction, target).as_dict(prefix)


def compute_trellis_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    device: str | torch.device = "cpu",
    data_range: float = 2.0,
    prefix: str | None = None,
) -> dict[str, float]:
    """Convenience helper mirroring the adapter interface."""

    adapter = TrellisMetricsAdapter(device=device, data_range=data_range)
    return adapter.summarize(prediction, target, prefix=prefix)
