"""Evaluation Metrics for Inference

Paradigm-specific evaluation metrics including FID, IS, PSNR, SSIM,
and other quantitative measures for assessing generation quality.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.core.metrics.evaluation_metrics import FID, IS, PSNR
from spectramr.core.metrics.evaluation_metrics import SSIMMetric as SSIM
from spectramr.core.metrics.registry import list_available

logger = logging.getLogger(__name__)


class BaseEvaluationMetric:
    """Base class for evaluation metrics."""

    def __init__(self, device: torch.device):
        """Initialize evaluation metric.

        Args:
            device: Device for computations
        """
        self.device = device

    def compute(
        self, generated: torch.Tensor, real: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Compute the evaluation metric.

        Args:
            generated: Generated samples
            real: Real samples for comparison (if needed)

        Returns:
            Dictionary with metric values
        """
        raise NotImplementedError("Subclasses must implement compute method")

    def reset(self):
        """Reset metric state."""
        pass


class PSNRMetric(BaseEvaluationMetric):
    """Peak Signal-to-Noise Ratio metric."""

    def __init__(self, device: torch.device):
        """__init__.

        Args:
            device (torch.device): Description.
        """
        super().__init__(device)
        self.metric = PSNR(device=device)

    def compute(
        self, generated: torch.Tensor, real: torch.Tensor | None = None
    ) -> dict[str, float]:
        """compute.

        Args:
            generated (torch.Tensor): Description.
            real (Optional[torch.Tensor]): Description.
        Returns:
            dict[str, float]: Description.
        """
        if real is None:
            raise ValueError("PSNR requires real samples for comparison")

        # Ensure correct device
        if generated.device != self.device:
            generated = generated.to(self.device)
        if real.device != self.device:
            real = real.to(self.device)

        try:
            val = self.metric(generated, real)
            return {"psnr": val.item()}
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "PSNR computation failed: %s",
                e,
                extra={"shapes": (generated.shape, real.shape)},
            )
            return {"psnr": float("-inf")}
        except AttributeError:
            # val is not a tensor
            return {"psnr": float("-inf")}


class SSIMMetric(BaseEvaluationMetric):
    """Structural Similarity Index Measure."""

    def __init__(self, device: torch.device, window_size: int = 11):
        """__init__.

        Args:
            device (torch.device): Description.
            window_size (int): Description.
        """
        super().__init__(device)
        self.metric = SSIM(window_size=window_size, device=device)

    def compute(
        self, generated: torch.Tensor, real: torch.Tensor | None = None
    ) -> dict[str, float]:
        """compute.

        Args:
            generated (torch.Tensor): Description.
            real (Optional[torch.Tensor]): Description.
        Returns:
            dict[str, float]: Description.
        """
        if real is None:
            raise ValueError("SSIM requires real samples for comparison")

        if generated.device != self.device:
            generated = generated.to(self.device)
        if real.device != self.device:
            real = real.to(self.device)

        try:
            val = self.metric(generated, real)
            return {"ssim": val.item()}
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "SSIM computation failed: %s",
                e,
                extra={"shapes": (generated.shape, real.shape)},
            )
            return {"ssim": 0.0}
        except AttributeError:
            # val is not a tensor
            return {"ssim": 0.0}


class FIDMetric(BaseEvaluationMetric):
    """Fréchet Inception Distance for GAN evaluation."""

    def __init__(self, device: torch.device, inception_model: nn.Module | None = None):
        """__init__.

        Args:
            device (torch.device): Description.
            inception_model (Optional[nn.Module]): Description.
        """
        super().__init__(device)
        self.metric = FID(device=device)
        # Inception model is handled internally by SSOT FID, but we keep the signature for compatibility
        # If inception_model is passed, we might ignore it or pass it if SSOT supports it.
        # SSOT FID currently handles its own model loading.

    def update(self, generated: torch.Tensor, real: torch.Tensor | None = None):
        """update.

        Args:
            generated (torch.Tensor): Description.
            real (Optional[torch.Tensor]): Description.
        Returns:
            Any: Description.
        """
        if hasattr(self.metric, "fid") and self.metric.fid is not None:
            if generated.device != self.device:
                generated = generated.to(self.device)
            if real is not None and real.device != self.device:
                real = real.to(self.device)

            # Ensure uint8 and 3 channels for torchmetrics
            if generated.dtype != torch.uint8:
                generated = (generated * 255).clamp(0, 255).byte()
            if real is not None and real.dtype != torch.uint8:
                real = (real * 255).clamp(0, 255).byte()

            if generated.shape[1] == 1:
                generated = generated.repeat(1, 3, 1, 1)
            if real is not None and real.shape[1] == 1:
                real = real.repeat(1, 3, 1, 1)

            self.metric.fid.update(generated, real=False)
            if real is not None:
                self.metric.fid.update(real, real=True)

    def compute(
        self, generated: torch.Tensor = None, real: torch.Tensor | None = None
    ) -> dict[str, float]:
        """compute.

        Args:
            generated (torch.Tensor): Description.
            real (Optional[torch.Tensor]): Description.
        Returns:
            dict[str, float]: Description.
        """
        if hasattr(self.metric, "fid") and self.metric.fid is not None:
            try:
                # If arguments provided, update first
                if generated is not None:
                    self.update(generated, real)
                val = self.metric.fid.compute()
                return {"fid": val.item()}
            except Exception as e:
                return {"fid_error": str(e)}

        if generated is None or real is None:
            return {"fid": 0.0}

        try:
            val = self.metric(real, generated)
            return {"fid": val.item()}
        except Exception as e:
            return {"fid_error": str(e)}

    def reset(self):
        """reset.

        Returns:
            Any: Description.
        """
        if hasattr(self.metric, "fid") and self.metric.fid is not None:
            self.metric.fid.reset()


class ISMetric(BaseEvaluationMetric):
    """Inception Score for GAN evaluation."""

    def __init__(self, device: torch.device, inception_model: nn.Module | None = None):
        """__init__.

        Args:
            device (torch.device): Description.
            inception_model (Optional[nn.Module]): Description.
        """
        super().__init__(device)
        self.metric = IS(device=device)

    def update(self, generated: torch.Tensor):
        """update.

        Args:
            generated (torch.Tensor): Description.
        Returns:
            Any: Description.
        """
        self.metric.update(generated)

    def compute(self, generated: torch.Tensor = None) -> dict[str, float]:
        """compute.

        Args:
            generated (torch.Tensor): Description.
        Returns:
            dict[str, float]: Description.
        """
        if generated is not None:
            self.metric.update(generated)

        # IS compute returns float
        val = self.metric.compute()
        return {"inception_score": val}

    def reset(self):
        """reset.

        Returns:
            Any: Description.
        """
        self.metric.reset()


class ParadigmSpecificEvaluator:
    """Evaluator with paradigm-specific metrics."""

    def __init__(self, device: torch.device, training_mode: TrainingModeTypes):
        """Initialize paradigm-specific evaluator.

        Args:
            device: Device for computations
            training_mode: Training paradigm (GAN, diffusion, VAE, etc.)
        """
        self.device = device
        self.training_mode = training_mode

        # Initialize metrics based on paradigm
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> dict[str, BaseEvaluationMetric]:
        """Initialize appropriate metrics for the training paradigm."""
        metrics = {}

        # Common metrics for all paradigms
        metrics["psnr"] = PSNRMetric(self.device)
        metrics["ssim"] = SSIMMetric(self.device)

        # Paradigm-specific metrics
        if self.training_mode == TrainingModeTypes.GAN:
            try:
                metrics["fid"] = FIDMetric(self.device)
                metrics["is"] = ISMetric(self.device)
            except ImportError:
                pass  # Skip if torchvision not available

        elif self.training_mode == TrainingModeTypes.DIFFUSION:
            # Diffusion models benefit from distribution-based metrics
            try:
                metrics["fid"] = FIDMetric(self.device)
            except ImportError:
                pass

        elif self.training_mode == TrainingModeTypes.VAE:
            # VAE-specific metrics (reconstruction quality)
            metrics["reconstruction_error"] = ReconstructionErrorMetric(self.device)

        return metrics

    def evaluate(
        self,
        generated: torch.Tensor,
        real: torch.Tensor | None = None,
        update_stats: bool = True,
    ) -> dict[str, float]:
        """Evaluate generated samples.

        Args:
            generated: Generated samples
            real: Real samples for comparison
            update_stats: Whether to update running statistics

        Returns:
            Dictionary with all metric values
        """
        results = {}

        # 1. Base/Legacy Metrics
        for name, metric in self.metrics.items():
            try:
                if update_stats and hasattr(metric, "update"):
                    if real is not None:
                        metric.update(generated, real)
                    else:
                        metric.update(generated)

                # Compute metric
                # Some metrics (like FID) need sufficient samples and might fail for small batches
                if hasattr(metric, "compute"):
                    try:
                        metric_result = metric.compute(generated, real)
                        results.update(metric_result)
                    except ValueError as _exc:
                        logger.debug(
                            "Suppressed exception: %s", _exc
                        )  # Ignore "not enough samples" errors for per-batch evaluation
                    except Exception as e:
                        results[f"{name}_error"] = str(e)

            except Exception as e:
                results[f"{name}_error"] = str(e)

        # 2. Comprehensive Medical Metrics
        if real is not None:
            try:
                from spectramr.core.metrics import compute_metric

                # Calculate basic regression metrics
                results["nmse"] = compute_metric("nmse", generated, real, device=self.device)
                results["rmse"] = compute_metric("rmse", generated, real, device=self.device)
                results["mae"] = compute_metric("mae", generated, real, device=self.device)

                # Calculate perceptual/structural metrics
                results["psnr_medical"] = compute_metric(
                    "psnr", generated, real, device=self.device
                )
                results["ssim_medical"] = compute_metric(
                    "ssim", generated, real, device=self.device
                )

                # Advanced MRI Metrics
                try:
                    results["hfen"] = compute_metric("hfen", generated, real, device=self.device)
                    if generated.shape[1] == 2:  # Complex/2-channel
                        results["complex_hfen"] = compute_metric(
                            "complex_hfen", generated, real, device=self.device
                        )
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

                try:
                    results["vif"] = compute_metric("vif", generated, real, device=self.device)
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

                try:
                    results["gmsd"] = compute_metric("gmsd", generated, real, device=self.device)
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

                try:
                    results["fsim"] = compute_metric("fsim", generated, real, device=self.device)
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

            except Exception as e:
                results["medical_metrics_error"] = str(e)

        # 3. Extensive Dynamic Metrics (User Request)
        # Iterate over ALL registered metrics to ensure comprehensive coverage,
        # but FILTER by INPUT_SIGNATURE so that landmark / displacement-field /
        # 3-D-volume / distributional metrics don't propagate NaNs into the
        # report tables (regression of the 2026-05-05 NaN-tables bug).
        if real is not None:
            from spectramr.core.metrics.registry import MetricsRegistry

            available_metrics = list_available()
            existing_keys = set(results.keys())
            # Skip lists
            _ALREADY_HANDLED = {
                "psnr",
                "ssim",
                "mse",
                "mae",
                "rmse",
                "nmse",
                "nrmse",
                "fid",
                "is",
                "kid",
                "medfid",
                "inception_score",
                "med_fid",
                "frd",
                "rfs",  # require accumulation, run via
                "sliced_wasserstein",  # distributional pathway
                "wasserstein_1d",
                "mmd_metric",
            }
            for metric_name in available_metrics:
                canonical_name = metric_name.lower()
                if canonical_name in existing_keys or canonical_name in _ALREADY_HANDLED:
                    continue

                # INPUT_SIGNATURE gate — only image-pair metrics participate
                # in the (pred, target)-image dynamic loop.
                cls = MetricsRegistry._metrics.get(canonical_name)
                input_signature = (
                    getattr(cls, "INPUT_SIGNATURE", "image_pair") if cls else "image_pair"
                )
                if input_signature != "image_pair":
                    # Record the skip so the report knows the metric exists
                    # but is intentionally not evaluated on image pairs.
                    results[f"{canonical_name}__skipped"] = (
                        f"requires non-image input ({input_signature})"
                    )
                    continue

                try:
                    val = compute_metric(metric_name, generated, real, device=self.device)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    # Final defensive guard — reject NaN/Inf at the boundary so
                    # the report aggregator doesn't have to.
                    import math as _math

                    if isinstance(val, float) and (_math.isnan(val) or _math.isinf(val)):
                        results[f"{canonical_name}__nan"] = True
                        continue
                    results[canonical_name] = val
                except Exception as exc:
                    # Don't swallow silently — at least record the failure.
                    # See findings booklet 2026-05-05 C-1.
                    import logging

                    logging.getLogger(__name__).warning(
                        "Metric '%s' failed during compute_all_metrics: %s",
                        canonical_name,
                        exc,
                    )
                    results[f"{canonical_name}__error"] = str(exc)

        return results

    def reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()


class ReconstructionErrorMetric(BaseEvaluationMetric):
    """Reconstruction error metric for VAE and autoencoder models."""

    def compute(
        self, generated: torch.Tensor, real: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Compute reconstruction error.

        Args:
            generated: Reconstructed/generated samples
            real: Original samples

        Returns:
            Reconstruction error metrics
        """
        if real is None:
            raise ValueError("Reconstruction error requires original samples")

        if generated.device != self.device:
            generated = generated.to(self.device)
        if real.device != self.device:
            real = real.to(self.device)

        # L1 and L2 reconstruction errors
        l1_error = F.l1_loss(generated, real, reduction="mean")
        l2_error = F.mse_loss(generated, real, reduction="mean")

        return {
            "reconstruction_l1": l1_error.item(),
            "reconstruction_l2": l2_error.item(),
            "reconstruction_rmse": torch.sqrt(l2_error).item(),
        }
