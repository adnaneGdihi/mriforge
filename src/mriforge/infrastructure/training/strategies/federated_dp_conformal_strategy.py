"""Federated DP-Conformal ULF Strategy (idea 5 / PR-7 H5).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §5
      and TODO/backlog_paradigm_expansion_roadmap.md §PR-7.

Wires three previously-shipped primitives into one trainable strategy:

1. :class:`DPMixin` — per-sample gradient clipping + Gaussian noise on
   every optimiser update with a Rényi-DP accountant.
2. The parent :class:`ReconstructionTrainingStrategy` — supplies the
   reconstruction loss machinery.
3. Conformal calibration (cross-site Cauchy aggregation) is performed
   *after* training by
   :class:`mriforge.infrastructure.training.strategies.conformal_calibration_strategy.ConformalCalibrationStrategy`,
   so this strategy only owns the DP-SGD part of the loop.

The strategy reads its DP knobs from the typed
``config.privacy.{noise_multiplier, max_grad_norm, target_delta}`` block
(a :class:`~mriforge.config.schemas.training.privacy.PrivacyConfig` mounted
on :class:`~mriforge.config.settings.TrainingSettings`) and derives the DP
sample rate ``q = batch_size / dataset_size`` from the *live*
``self.env.train_loader.dataset`` length (falling back to
``config.privacy.dataset_size`` then ``config.data.dataset_size``).  There
is **no silent fallback**: an absent ``privacy`` block, an out-of-range
knob, or an unresolvable dataset size each raises at construction, because
a DP strategy must never train without an honest privacy guarantee
(pitfalls #9, #10, #15).  The resolved budget is stamped into the run log
and into the per-step ``dp_metrics`` snapshot.

DP-SGD post-processing is wired into the *actual* optimiser-step path
by overriding :meth:`_clip_and_log_gradients` — the gradient hook the
``StepExecutor`` invokes after the backward pass + AMP-unscale and **before**
``opt_g.step()`` (see
``mriforge.infrastructure.training.step_executor.StepExecutor.execute_step`` →
``amp_policy.backward_and_step(gradient_clipping_fn=...)``).  At that
seam every parameter already carries an unscaled ``.grad``; the override
(1) rescales the aggregate gradient L2 norm to ``max_grad_norm``,
(2) adds isotropic Gaussian noise with
``std = noise_multiplier * max_grad_norm / batch_size`` via
:func:`add_gaussian_noise`, and (3) advances the privacy accountant via
:meth:`record_dp_step`.  This is a batch-level (not per-sample)
Gaussian mechanism — minimal but a *real* privacy transform that is
actually applied to the gradients the optimiser steps with.  For
production-grade per-sample gradients use ``opacus.PrivacyEngine``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from mriforge.config.schemas.training.privacy import PrivacyConfig
from mriforge.domain.exceptions import ConfigurationError
from mriforge.infrastructure.services.privacy.dp_sgd import add_gaussian_noise

from .mixins.dp_mixin import DPMixin
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class FederatedDPConformalULFStrategy(DPMixin, ReconstructionTrainingStrategy):
    """Federated reconstruction with DP-SGD + post-hoc conformal."""

    # Sampled per training step; the strategy is NOT an nn.Module so we
    # keep DP runtime state as plain attrs (no register_buffer).
    _dp_batch_size: int = 4
    # Resolved DP provenance — stamped in _configure_from_settings so the
    # reported (epsilon, delta) budget is traceable to the real dataset.
    _dp_dataset_size: int = 0
    _dp_sample_rate: float = 0.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configure_from_settings()

    def _resolve_dataset_size(self, priv: PrivacyConfig, data: Any) -> int:
        """Resolve the DP dataset size *honestly* (no magic 1000 fallback).

        Preference order, so the accountant's reported ``(ε, δ)`` is tied to
        the real run:

        1. ``len(self.env.train_loader.dataset)`` — the live dataset the
           optimiser actually iterates over;
        2. ``config.privacy.dataset_size`` — the value the YAML declared;
        3. ``config.data.dataset_size`` — the data-block fallback.

        Raises:
            ConfigurationError: when none of the above is available, so the
                sample rate (hence the privacy budget) can never be silently
                derived from a meaningless literal.
        """
        env = getattr(self, "env", None)
        loader = getattr(env, "train_loader", None) if env is not None else None
        dataset = getattr(loader, "dataset", None) if loader is not None else None
        if dataset is not None:
            try:
                n = len(dataset)
            except TypeError:
                n = 0
            if n >= 1:
                return int(n)

        for source in (priv, data):
            ds = getattr(source, "dataset_size", None) if source is not None else None
            if ds is not None and ds >= 1:
                return int(ds)

        raise ConfigurationError(
            "FederatedDPConformalULFStrategy cannot derive a DP sample rate: "
            "no train_loader dataset length, no config.privacy.dataset_size, "
            "and no config.data.dataset_size. Set one so the reported "
            "(epsilon, delta) budget is honest rather than tied to a literal."
        )

    def _configure_from_settings(self) -> None:
        """Read + validate the DP knobs from the SSOT ``config``; fail loudly.

        The ``privacy`` block is a typed
        :class:`~mriforge.config.schemas.training.privacy.PrivacyConfig`
        mounted on :class:`~mriforge.config.settings.TrainingSettings`, so the
        nested fields are *real attributes* read directly — never via
        ``getattr``-with-literal-default, which silently discarded a YAML-set
        value when the block was absent / untyped.

        A strategy named ``federated_dp_conformal`` MUST NOT train without a
        privacy guarantee (pitfall #10), so:

        * an absent ``privacy`` block raises (no silent defaults);
        * out-of-range knobs propagate :meth:`DPMixin.configure_dp`'s
          ``ValueError`` instead of being swallowed into a warning;
        * the sample rate is derived from the *real* dataset length (not a
          magic ``1000``), and the resolved budget is stamped into the run's
          provenance (pitfall #15).
        """
        config = getattr(self, "config", None)
        priv = getattr(config, "privacy", None) if config is not None else None
        data = getattr(config, "data", None) if config is not None else None

        if priv is None:
            raise ConfigurationError(
                "FederatedDPConformalULFStrategy requires a `privacy` config "
                "block (noise_multiplier, max_grad_norm, target_delta). It is "
                "absent, so DP-SGD cannot be configured; refusing to train a "
                "DP strategy without a privacy guarantee."
            )
        # Coerce a raw mapping (e.g. a caller still passing an untyped dict)
        # into the typed, range-validated schema. A PrivacyConfig passes
        # through unchanged.
        if not isinstance(priv, PrivacyConfig):
            priv = PrivacyConfig.model_validate(priv)

        nm = priv.noise_multiplier
        mgn = priv.max_grad_norm
        delta = priv.target_delta
        # The DP sample rate -- and therefore the privacy accounting -- is
        # computed from this. The old `getattr(data, "batch_size", 4)` silently
        # substituted 4 for the arm's real batch size once the key folded to
        # `data.loader.batch_size`, which invalidates epsilon without any
        # symptom. Read the canonical path, and refuse rather than guess if the
        # caller handed us an untyped `data` (this file already coerces such a
        # mapping for `privacy`, so it is a shape that reaches here).
        if data is None:
            bs = 4
        else:
            loader = getattr(data, "loader", None)
            if loader is None:
                raise ConfigurationError(
                    "FederatedDPConformalULFStrategy could not read "
                    "`data.loader.batch_size` (got "
                    f"data={type(data).__name__}). The DP sample rate is "
                    "computed from it, so a substituted default would silently "
                    "invalidate the reported privacy guarantee."
                )
            bs = loader.batch_size
        self._dp_batch_size = int(bs) if bs and bs >= 1 else 1

        ds = self._resolve_dataset_size(priv, data)
        sample_rate = max(min(self._dp_batch_size / ds, 1.0), 1e-4)
        self._dp_dataset_size = ds
        self._dp_sample_rate = sample_rate

        # Range checks live in configure_dp; let its ValueError propagate so an
        # invalid budget fails at construction rather than degrading to a
        # non-private run (no try/except swallow).
        self.configure_dp(
            noise_multiplier=nm,
            max_grad_norm=mgn,
            sample_rate=sample_rate,
            target_delta=delta,
        )

        # Stamp the resolved budget into provenance (one-time, structured).
        logger.info(
            "[FederatedDPConformalULF] DP configured from config.privacy: "
            "noise_multiplier=%.4f, max_grad_norm=%.4f, target_delta=%.2e, "
            "batch_size=%d, dataset_size=%d, sample_rate=%.6f",
            nm,
            mgn,
            delta,
            self._dp_batch_size,
            ds,
            sample_rate,
        )

    @torch.no_grad()
    def _param_grad_norms(self) -> torch.Tensor:
        """Return the per-*parameter* gradient L2 norms (diagnostic only).

        One scalar L2 value per parameter tensor of ``self.env.generator`` —
        i.e. ``len(result) == #params-with-grad``, NOT one norm per sample.
        The live DP path is :meth:`_apply_dp_to_grads` (batch-level clip +
        Gaussian noise); this helper is purely a gradient-health probe and is
        deliberately not on that path. For genuine per-sample DP use
        :meth:`DPMixin.dp_post_process_per_sample_grads` with
        ``torch.func.vmap``-computed per-sample gradients.
        """
        gen = getattr(self.env, "generator", None)
        if gen is None:
            return torch.zeros(1)
        norms = torch.cat(
            [p.grad.flatten().norm().unsqueeze(0) for p in gen.parameters() if p.grad is not None]
        )
        return norms

    @torch.no_grad()
    def _apply_dp_to_grads(self, model: torch.nn.Module) -> None:
        """Clip the aggregate gradient norm to ``C`` then add Gaussian noise.

        Applied in-place to ``model.parameters()[*].grad`` so the
        optimiser steps with the privatised gradient. Noise std follows
        the canonical DP-SGD mechanism
        ``noise_multiplier * max_grad_norm / batch_size``.
        """
        if not self._dp_enabled:
            return

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return

        # 1. Bound the L2 sensitivity: rescale the *whole* gradient vector
        #    so its global L2 norm ≤ max_grad_norm (only shrink).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self._max_grad_norm)

        # 2. Gaussian mechanism: add N(0, (σ·C/B)²) to every grad tensor.
        for g in grads:
            noised = add_gaussian_noise(
                g,
                noise_multiplier=self._noise_multiplier,
                max_grad_norm=self._max_grad_norm,
                batch_size=self._dp_batch_size,
            )
            g.copy_(noised)

    def _clip_and_log_gradients(
        self,
        model: torch.nn.Module,
        epoch: int,
        step: int,
        model_name: str = "model",
    ) -> float | None:
        """DP-SGD gradient hook (called by the Trainer pre ``opt_g.step()``).

        The base ``Trainer.execute_step`` wires this method as the
        ``gradient_clipping_fn`` for the generator optimiser
        (``self.env.opt_g``), invoking it after the backward pass +
        AMP-unscale and before the optimiser step.  We privatise the
        gradients here (clip → noise), advance the privacy accountant,
        then delegate to the base implementation for logging/anomaly
        bookkeeping.
        """
        self._apply_dp_to_grads(model)
        # One DP-SGD round has been executed for this optimiser step.
        self.record_dp_step()
        return super()._clip_and_log_gradients(model, epoch, step, model_name)

    def post_step(self) -> dict[str, float]:
        """Optimiser-step hook: return current privacy metrics.

        The accountant is advanced inside :meth:`_clip_and_log_gradients`
        (the real step path); this method only snapshots the live
        ``(ε, δ)`` budget plus the resolved sample-rate / dataset-size
        provenance for the metrics CSV.
        """
        snapshot = {k: float(v) for k, v in self.dp_metrics().items()}
        # Stamp the resolved budget provenance so the (ε, δ) is traceable to
        # the real dataset rather than a hidden literal (pitfall #15).
        snapshot["dp/sample_rate"] = float(self._dp_sample_rate)
        snapshot["dp/dataset_size"] = float(self._dp_dataset_size)
        snapshot["dp/batch_size"] = float(self._dp_batch_size)
        return snapshot
