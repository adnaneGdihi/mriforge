"""Model Validation Mixin Module.

This module provides a reusable mixin for model validation, handling
batch unpacking, prediction generation, and metric computation.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any

import torch

from spectramr.infrastructure.training.utils.training_utils import clamp_to_range

from .utils import pick_present

logger = logging.getLogger(__name__)


class _FallbackRangeUnresolvedError(Exception):
    """The fallback PSNR could not resolve a data_range, so it reports nothing.

    Internal control flow only. Distinct from a generic exception so the
    fallback's own ``except`` cannot mistake "no contract holds for this data"
    for "the fallback itself is broken" and log the wrong diagnosis.
    """


class ModelValidationMixin(ABC):
    #: Metrics the input-as-prediction baseline is graded on. Intersected with
    #: the arm's configured specs by ``ValidationMetricsComputer.compute(only=)``,
    #: never added to them (one owner of "which metrics does this arm grade on").
    _ZF_BASELINE_METRICS: tuple[str, ...] = ("psnr", "hfen")
    #: Reported once per strategy instance when the baseline cannot be formed.
    _zf_baseline_not_applicable_reported: bool = False

    def _zero_filled_baseline_metrics(
        self, input_batch, target_batch, computer, transform_fn, val_config
    ) -> dict[str, float]:
        """``val_zf_<metric>``: the input graded against the target like a prediction.

        For a loader-masked k-space arm the input IS the zero-filled measurement;
        for an image-to-image arm it is the identity baseline. Both answer "did
        the model beat what it was handed?". Not applicable (reported once, then
        silent) when the input cannot be brought to the target's shape by the
        arm's own metric transform -- an SR arm whose input is a smaller image.
        """
        if input_batch is None or target_batch is None:
            return {}
        try:
            zf_in = input_batch
            zf_tgt = target_batch
            if zf_in.ndim == 5:
                b_dim, c_dim, h_dim, w_dim, d_dim = zf_in.shape
                zf_in = zf_in.permute(0, 4, 1, 2, 3).reshape(b_dim * d_dim, c_dim, h_dim, w_dim)
            if zf_tgt.ndim == 5:
                b_dim, c_dim, h_dim, w_dim, d_dim = zf_tgt.shape
                zf_tgt = zf_tgt.permute(0, 4, 1, 2, 3).reshape(b_dim * d_dim, c_dim, h_dim, w_dim)
            if transform_fn:
                zf_in, zf_tgt = transform_fn(zf_in, zf_tgt, val_config)
            if zf_in.shape != zf_tgt.shape:
                raise ValueError(f"input {tuple(zf_in.shape)} vs target {tuple(zf_tgt.shape)}")
            baseline = computer.compute(zf_in, zf_tgt, only=self._ZF_BASELINE_METRICS)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            if not self._zf_baseline_not_applicable_reported:
                self._zf_baseline_not_applicable_reported = True
                logger.info(
                    "[ModelValidationMixin] zero-filled/identity baseline not applicable "
                    "for this arm (%s); no val_zf_* columns will be reported.",
                    exc,
                )
            return {}
        return {f"val_zf_{k}": v for k, v in baseline.items()}

    """Mixin for standardized model validation logic.

    Provides a template method `validation_step` that orchestrates:
    1. Batch unpacking
    2. Prediction generation (via hook)
    3. Metric computation
    4. Error handling
    """

    @abstractmethod
    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass for validation.

        Args:
            input_batch: Input tensor
            batch_context: Batch context dictionary
            **kwargs: Additional arguments

        Returns:
            Generated output tensor
        """
        pass

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Performs a single validation step with evaluation metrics.

        Args:
            batch: Raw batch data
            input_batch: Optional pre-unpacked input
            target_batch: Optional pre-unpacked target
            **kwargs: Additional context (e.g. batch_context)

        Returns:
            Dictionary of computed metrics
        """
        batch = (input_batch, target_batch)
        # POP (not get): line 95 re-forwards **kwargs to _validation_forward and also
        # passes batch_context positionally. With .get, a caller that supplies
        # batch_context= (the cross-field / field-flow / field-bridge validation_step
        # overrides do) would hit "multiple values for argument 'batch_context'".
        batch_context = kwargs.pop("batch_context", None)

        # Unpack if needed
        if input_batch is None or target_batch is None:
            # Assume self._unpack_batch exists (from BatchPreparationMixin)
            unpack_fn = self._unpack_batch
            if unpack_fn:
                lr, hr = unpack_fn(batch)
                input_batch = pick_present(input_batch, lr)
                target_batch = pick_present(target_batch, hr)

        # Fail gracefully if still missing
        if input_batch is None or target_batch is None:
            logger.warning(
                "[ModelValidationMixin] _unpack_batch returned None for input/target. "
                "Batch type=%s, keys=%s. Returning empty metrics.",
                type(batch).__name__,
                list(batch.keys()) if isinstance(batch, dict) else "N/A",
            )
            return {}

        self.generator_model.eval()
        # Prepare context if missing
        if batch_context is None:
            batch_context = {"use_dc": False, "measured_kspace": None}

        # Generate predictions (Hook)
        hr_fakes = self._validation_forward(input_batch, batch_context, **kwargs)

        # Handle models that return tuples (e.g., DisentangledMRI, VAE)
        if isinstance(hr_fakes, (tuple, list)):
            hr_fakes = hr_fakes[0]

        # Handle models that return dicts
        if isinstance(hr_fakes, dict):
            hr_fakes = hr_fakes.get(
                "reconstruction",
                hr_fakes.get("output", next(iter(hr_fakes.values()), None)),
            )

        if hr_fakes is None:
            logger.warning(
                "[ModelValidationMixin] _validation_forward returned None. Cannot compute metrics."
            )
            return {}

        # Ensure tensors are on the same device
        if target_batch is not None and hr_fakes.device != target_batch.device:
            hr_fakes = hr_fakes.to(target_batch.device)

        # Optional output range enforcement
        config = self.state and self.state.config
        training_config = getattr(config, "training", None)
        if training_config and getattr(training_config, "enforce_output_range", False):
            hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

        # Compute Metrics
        metrics = {}
        val_config = config.validation if config else None
        compute_img_metrics = val_config.scoring.enable_image_metrics if val_config else True

        if compute_img_metrics and target_batch is not None:
            # Assume self.validation_metrics_computer exists
            computer = self.validation_metrics_computer
            if computer:
                try:
                    # Globally flatten 5D tensors to 4D to prevent "too many values to unpack (expected 4)" in metrics
                    if target_batch.ndim == 5:
                        b_dim, c_dim, h_dim, w_dim, d_dim = target_batch.shape
                        target_batch = target_batch.permute(0, 4, 1, 2, 3).reshape(
                            b_dim * d_dim, c_dim, h_dim, w_dim
                        )
                    if hr_fakes.ndim == 5:
                        b_dim, c_dim, h_dim, w_dim, d_dim = hr_fakes.shape
                        hr_fakes = hr_fakes.permute(0, 4, 1, 2, 3).reshape(
                            b_dim * d_dim, c_dim, h_dim, w_dim
                        )

                    # Apply transforms if available
                    transform_fn = self._apply_metric_transforms
                    if transform_fn:
                        output_trans, target_trans = transform_fn(
                            hr_fakes, target_batch, val_config
                        )
                    else:
                        output_trans, target_trans = hr_fakes, target_batch

                    # Evidential UNets output 4 parameters (mean, var, alpha, beta)
                    # We only want to compute structural metrics on the predicted mean
                    if (
                        getattr(config.model, "model_type", "") == "evidential_unet"
                        and output_trans.shape[1] > target_trans.shape[1]
                    ):
                        output_trans = output_trans[:, : target_trans.shape[1]]

                    computed = computer.compute(output_trans, target_trans)

                    # Apply prefix 'val_'
                    for k, v in computed.items():
                        metrics[f"val_{k}"] = v

                    # Zero-filled / identity baseline (cohort review 2026-09-02,
                    # T0.7): the INPUT, graded exactly like the prediction on the
                    # cheap full-reference metrics the arm already computes. A
                    # model scoring below its own input is the failure every
                    # dashboard was green on (experiment_11_attention_none,
                    # 2026-08-18); the cold-diffusion path emits its own from the
                    # masked measurement and says so with ``_owns_zero_filled_baseline``.
                    if not getattr(self, "_owns_zero_filled_baseline", False):
                        metrics.update(
                            self._zero_filled_baseline_metrics(
                                input_batch, target_batch, computer, transform_fn, val_config
                            )
                        )
                except torch.cuda.OutOfMemoryError:
                    # NN#3 (WS-7 SM-002): never swallow CUDA OOM — it leaves the
                    # allocator in a corrupt state and the fallback metric path
                    # below would OOM again. Propagate so the run fails loudly
                    # instead of silently reporting degraded validation metrics.
                    raise
                except Exception as e:
                    logger_service = self.logging_service
                    tb_str = traceback.format_exc()
                    if logger_service:
                        logger_service.log_warning(
                            f"[ModelValidationMixin] Validation metrics computation failed: {e}\n"
                            f"  pred shape={hr_fakes.shape}, dtype={hr_fakes.dtype}\n"
                            f"  target shape={target_batch.shape}, dtype={target_batch.dtype}\n"
                            f"  Traceback:\n{tb_str}"
                        )
                    else:
                        logger.warning(
                            "[ModelValidationMixin] Validation metrics failed: %s\n%s",
                            e,
                            tb_str,
                        )

                    # Fallback: compute basic PSNR and SSIM so we never return empty metrics
                    try:
                        pred_for_fallback = hr_fakes.float()
                        target_for_fallback = target_batch.float()

                        # A 2-channel prediction is real/imag for MOST strategies, but a
                        # distribution head emits [mean, logvar]. RSS-ing THOSE computes
                        # sqrt(mean^2 + logvar^2) -- mixing an intensity with a log-variance
                        # -- and since logvar is near-constant across the image it drags the
                        # graded prediction toward an input-independent field (the b29
                        # wrong-channel metric). The strategy already declares which it is.
                        _dist_head = bool(getattr(self, "predicts_distribution_params", False))

                        # Ensure same number of channels via magnitude conversion
                        if pred_for_fallback.shape[1] != target_for_fallback.shape[1]:
                            if pred_for_fallback.shape[1] == 2:
                                pred_for_fallback = (
                                    pred_for_fallback[:, 0:1]
                                    if _dist_head
                                    else torch.sqrt(
                                        pred_for_fallback[:, 0:1] ** 2
                                        + pred_for_fallback[:, 1:2] ** 2
                                    )
                                )
                            if target_for_fallback.shape[1] == 2:
                                target_for_fallback = torch.sqrt(
                                    target_for_fallback[:, 0:1] ** 2
                                    + target_for_fallback[:, 1:2] ** 2
                                )
                            # If still mismatched, take first channel
                            if pred_for_fallback.shape[1] != target_for_fallback.shape[1]:
                                pred_for_fallback = pred_for_fallback[:, :1]
                                target_for_fallback = target_for_fallback[:, :1]

                        mse = torch.nn.functional.mse_loss(
                            pred_for_fallback, target_for_fallback
                        ).item()

                        # The SAME data_range policy the registered `psnr` uses, so
                        # `val_psnr` means one thing whichever branch produced it.
                        # This used to derive a PER-IMAGE `max - min` under the same
                        # key. See docs/metric_outcome_contract.rst, "Range-sensitive
                        # metrics".
                        from spectramr.core.metrics.evaluation_metrics import (
                            resolve_image_data_range,
                        )
                        from spectramr.core.metrics.outcome import (
                            MetricNotApplicableError,
                        )

                        try:
                            data_range = resolve_image_data_range(
                                target_for_fallback, None, metric_name="val_psnr"
                            )
                        except MetricNotApplicableError as dr_exc:
                            # No contract holds, so there is no PSNR to report.
                            logger.warning(
                                "[ModelValidationMixin] Fallback val_psnr skipped: %s",
                                dr_exc,
                            )
                            metrics["val_mse"] = mse
                            raise _FallbackRangeUnresolvedError from None

                        if mse > 0:
                            psnr = 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()
                        else:
                            psnr = float("nan")

                        metrics["val_psnr"] = psnr
                        metrics["val_mse"] = mse
                        logger.info(
                            "[ModelValidationMixin] Fallback metrics (same data_range "
                            "policy as the registered psnr): val_psnr=%.2f, val_mse=%.6f",
                            psnr,
                            mse,
                        )
                    except _FallbackRangeUnresolvedError:
                        pass
                    except Exception as fallback_e:
                        logger.warning(
                            "[ModelValidationMixin] Fallback metrics also failed: %s",
                            fallback_e,
                        )

        # ---- Validation loss for the learning-curve figure (fig_1_2) --------
        # The image-metric block above emits val_ssim/psnr/lpips but NO scalar
        # loss, so validation_metrics.csv carried no loss column and
        # fig_1_2_learning_curves had a train-only series. Emit a paradigm-agnostic
        # validation reconstruction loss (magnitude-matched L1 on the prediction /
        # target already in hand) as ``val_loss``. It is always computable here
        # without the strategy's batch dict; strategies with their own validation
        # loss (standard / diffusion) override validation_step and set ``val_loss``
        # there, so this never double-writes. Monitoring scalar only — never crash
        # validation for it (the .item() sync is off the hot training loop).
        if compute_img_metrics and target_batch is not None and "val_loss" not in metrics:
            try:
                pred_l = hr_fakes.float()
                tgt_l = target_batch.float()
                if pred_l.shape[1] != tgt_l.shape[1]:
                    if pred_l.shape[1] == 2:
                        # See the fallback path above: [mean, logvar] is not real/imag,
                        # so RSS here corrupts val_loss the same way it corrupted PSNR.
                        pred_l = (
                            pred_l[:, 0:1]
                            if getattr(self, "predicts_distribution_params", False)
                            else torch.sqrt(pred_l[:, 0:1] ** 2 + pred_l[:, 1:2] ** 2)
                        )
                    if tgt_l.shape[1] == 2:
                        tgt_l = torch.sqrt(tgt_l[:, 0:1] ** 2 + tgt_l[:, 1:2] ** 2)
                    if pred_l.shape[1] != tgt_l.shape[1]:
                        pred_l, tgt_l = pred_l[:, :1], tgt_l[:, :1]
                metrics["val_loss"] = torch.nn.functional.l1_loss(pred_l, tgt_l).item()
            except Exception as exc:  # monitoring scalar — never fatal to validation
                logger.warning("[ModelValidationMixin] val_loss computation skipped: %s", exc)

        return metrics
