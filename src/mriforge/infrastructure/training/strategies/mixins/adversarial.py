"""
Adversarial Mixin Module

This module contains the AdversarialMixin for GAN-based training logic.
"""

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from mriforge.infrastructure.training.loop_state import resolve_loop_iteration
from mriforge.infrastructure.training.optimization_utils import (
    AsyncMetricsReporter,
    OptimizationMetrics,
)
from mriforge.infrastructure.training.strategy_helpers import (
    StrategyInitializationHelper,
)
from mriforge.infrastructure.training.utils.training_utils import clamp_to_range
from mriforge.models.losses.computers import UnifiedGANLossComputer

if TYPE_CHECKING:
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


def _resolve_disc_updates(config: Any) -> int:
    """Return the discriminator-updates-per-G-step count.

    v6.0 SSOT path: ``config.losses.gan.disc_updates``. Reading a
    top-level ``config.disc_updates`` is the legacy flat-config pattern
    (CLAUDE.md pitfall #1) and silently falls back to 1 on a v6.0
    config because ``extra="forbid"`` rejects the unknown top-level
    field — see ``TODO/audit/05_strategies_core_mixins_builders.md`` F10.

    Raises:
        ValueError: if ``config.losses.gan`` is not configured. Any
            adversarial-mixin caller has a GAN strategy in MRO and must
            ship a populated ``losses.gan`` block.
    """
    _losses = getattr(config, "losses", None)
    gan_losses = getattr(_losses, "gan", None)
    if gan_losses is None:
        raise ValueError(
            "AdversarialMixin requires `config.losses.gan` to be set. "
            "Add a `losses.gan:` block to the YAML (the canonical home "
            "for `disc_updates`, `gan_loss_type`, R1 schedule, etc.)."
        )
    return int(gan_losses.disc_updates)


class AdversarialMixin:
    """Mixin for adversarial training (GAN) logic."""

    def setup_adversarial(
        self: "BaseTrainingStrategy", expected_modes: tuple[str, ...] = ("gan",)
    ) -> None:
        """Initialize adversarial components."""
        if hasattr(self, "_verify_strategy_config"):
            self._verify_strategy_config(expected_modes=expected_modes)

        if hasattr(self, "_log_config_features") and hasattr(self, "logging_service"):
            self._log_config_features(self.logging_service)

        config = self.env.config if hasattr(self, "env") and self.env else self.state.config
        device = self.device

        self.loss_computer = UnifiedGANLossComputer(
            config=config,
            device=device,
        )

        StrategyInitializationHelper.initialize_profiling_service(self, fallback_enabled=False)

        self.async_metrics_reporter = AsyncMetricsReporter(batch_size=10)
        self.optimization_metrics = OptimizationMetrics(name="gan_training")

        def metrics_callback(metrics: dict) -> None:
            """Async callback for reporting aggregated metrics."""
            # Use getattr to safely access logging_service
            logging_service = self.logging_service
            if logging_service is not None and logger.isEnabledFor(logging.DEBUG):
                for key, value in metrics.items():
                    logging_service.log_debug(
                        "Aggregated metric: %s=%.6f",
                        key,
                        value,
                        model_type=self.state.model_type,
                    )

        self.async_metrics_reporter.set_callback(metrics_callback)

        self._step_counter = 0

    def _check_discriminator_features(self: "BaseTrainingStrategy") -> bool:
        """Check if discriminator supports feature extraction."""
        try:
            disc_forward = self.discriminator_model.forward
            if disc_forward and hasattr(disc_forward, "__code__"):
                return "return_features" in disc_forward.__code__.co_varnames
        except (AttributeError, TypeError) as _exc:
            logger.debug("Suppressed exception: %s", _exc)
        return False

    def _should_apply_r1_regularization(
        self: "BaseTrainingStrategy", epoch: int | None, iteration: int = 0
    ) -> bool:
        """Determine if R1 regularization should be applied this step."""
        try:
            if self.config.losses and self.config.losses.gan:
                interval = self.config.losses.gan.r1_interval
            else:
                interval = 16
        except AttributeError:
            interval = 16

        if interval > 0:
            if iteration > 0:
                return (iteration % interval) == 0
            if epoch is not None:
                return (epoch % interval) == 0
            return False

        try:
            if (
                self.config.losses
                and self.config.losses.gan
                and hasattr(self.config.losses.gan, "r1_probability")
            ):
                probability = self.config.losses.gan.r1_probability
            else:
                probability = 1.0
        except AttributeError:
            probability = 1.0

        should_apply = torch.rand(()) < probability
        return should_apply

    def train_step_adversarial(
        self: "BaseTrainingStrategy",
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Adversarial training step returning closures for the Trainer.
        """
        self._step_counter += 1

        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        if input_batch is not None:
            input_batch = self._to_device(input_batch)
        if target_batch is not None:
            target_batch = self._to_device(target_batch)

        iteration = kwargs.get("iteration", kwargs.get("step", 0))
        config = self.env.config if hasattr(self, "env") and self.env else self.state.config
        num_d_updates = _resolve_disc_updates(config)

        step_configs = []

        # State block used for metric reporting
        self._last_step_metrics = {}

        def make_d_closure():
            """make_d_closure.

            Returns:
                Any: Description.
            """

            def d_closure() -> torch.Tensor:
                """d_closure.

                Returns:
                    torch.Tensor: Description.
                """
                with torch.no_grad():
                    hr_fakes = self.generator_model(input_batch)

                if hr_fakes.device != target_batch.device:
                    hr_fakes = hr_fakes.to(target_batch.device)

                d_loss_output = (
                    self.loss_computer.compute_discriminator_loss(
                        real=target_batch,
                        fake=hr_fakes,
                        discriminator=self.discriminator_model,
                        epoch=epoch,
                        iteration=iteration,
                    )
                    if hasattr(self.loss_computer, "compute_discriminator_loss")
                    else None
                )

                if d_loss_output is None:
                    disc_outputs_d = {
                        "real_pred": self.discriminator_model(target_batch),
                        "fake_pred": self.discriminator_model(hr_fakes),
                    }
                    d_loss_output = self.loss_computer.compute(
                        pred=hr_fakes,
                        target=target_batch,
                        epoch=epoch,
                        iteration=iteration,
                        discriminator=self.discriminator_model,
                        discriminator_outputs=disc_outputs_d,
                    )

                d_total = d_loss_output.total if hasattr(d_loss_output, "total") else d_loss_output

                # Store detached metrics. Keep them on-device — `get_last_metrics`
                # resolves to Python floats at a coarser cadence, so no per-step
                # D2H sync happens inside the closure (NN#9).
                with torch.no_grad():
                    self._last_step_metrics["d_total_loss"] = d_total.detach()
                    if hasattr(d_loss_output, "components"):
                        for k, v in d_loss_output.components.items():
                            self._last_step_metrics[f"d_{k}"] = v.detach()

                return d_total

            return d_closure

        for _ in range(num_d_updates):
            step_configs.append(
                {
                    "optimizer": self.state.opt_d,
                    "closure": make_d_closure(),
                    "model": self.discriminator_model,
                    "name": "discriminator",
                }
            )

        def g_closure() -> torch.Tensor:
            """g_closure.

            Returns:
                torch.Tensor: Description.
            """
            hr_fakes = self.generator_model(input_batch)

            if hr_fakes.device != target_batch.device:
                hr_fakes = hr_fakes.to(target_batch.device)

            if hasattr(self.config, "training") and getattr(
                self.config.training, "enforce_output_range", False
            ):
                hr_fakes = clamp_to_range(hr_fakes, enable=True, telemetry=False)

            g_loss_output = (
                self.loss_computer.compute_generator_loss(
                    pred=hr_fakes,
                    target=target_batch,
                    discriminator=self.discriminator_model,
                    epoch=epoch,
                    iteration=iteration,
                )
                if hasattr(self.loss_computer, "compute_generator_loss")
                else None
            )

            if g_loss_output is None:
                if self.discriminator_model:
                    disc_outputs = {
                        "fake_pred": self.discriminator_model(hr_fakes),
                    }
                    g_loss_output = self.loss_computer.compute(
                        pred=hr_fakes,
                        target=target_batch,
                        epoch=epoch,
                        iteration=iteration,
                        input_batch=input_batch,
                        discriminator=self.discriminator_model,
                        discriminator_outputs=disc_outputs,
                    )
                    g_total = (
                        g_loss_output.total if hasattr(g_loss_output, "total") else g_loss_output
                    )
                else:
                    criterion = self.env.criterion_l1 or nn.L1Loss().to(target_batch.device)
                    g_total = criterion(hr_fakes, target_batch)
            else:
                g_total = g_loss_output.total if hasattr(g_loss_output, "total") else g_loss_output

            # Store detached metrics. Keep them on-device — `get_last_metrics`
            # resolves to Python floats at a coarser cadence, so no per-step
            # D2H sync happens inside the closure (NN#9).
            with torch.no_grad():
                if g_total is not None:
                    self._last_step_metrics["g_total_loss"] = g_total.detach()
                if hasattr(g_loss_output, "components"):
                    for k, v in g_loss_output.components.items():
                        key = f"g_{k}" if not str(k).startswith("loss_") else k
                        self._last_step_metrics[key] = v.detach()

                if hasattr(self, "_compute_training_metrics"):
                    # Live iteration (loop_state seam), not the frozen
                    # ``self.env.step`` (=0) — restores the train-metric
                    # interval throttle (pitfall #16).
                    current_step = resolve_loop_iteration(self)
                    train_metrics = self._compute_training_metrics(
                        pred=hr_fakes,
                        target=target_batch,
                        config=self.config,
                        current_step=current_step,
                    )
                    self._last_step_metrics.update(
                        {
                            k: v.detach() if isinstance(v, torch.Tensor) else v
                            for k, v in train_metrics.items()
                        }
                    )

            return g_total

        step_configs.append(
            {
                "optimizer": self.state.opt_g,
                "closure": g_closure,
                "model": self.generator_model,
                "name": "generator",
            }
        )

        return step_configs

    def get_last_metrics(self) -> dict[str, Any]:
        """Return the detached component metrics, ON-DEVICE (#707).

        The explicit ``.detach().item()`` here was the clearest case: the closures
        at ``:227`` and ``:309`` store these on-device with a comment saying
        ``get_last_metrics`` converts them, and `training_loop` then called it on
        every iteration. The sync count per step never dropped -- it only moved.
        Values are already detached at the store sites, so this is a plain copy.
        """
        return dict(getattr(self, "_last_step_metrics", {}))
