"""DiffusionMixin beta-range forwarding (2026-07 ldm cohort review).

``training.diffusion.beta_start`` / ``beta_end`` are schema fields with a
``beta_end > beta_start`` validator, but ``initialize_diffusion_parameters``
never forwarded them to :class:`DiffusionScheduler`, which therefore always
built its linear schedule from its own ``1e-4 .. 0.02`` defaults. The knob was
declared by 16 arms and read by none (pitfall #15).

These pin the forwarding, and pin that a widened range actually moves the
``betas`` tensor — the pre-fix code passed this file's shape assertions while
producing the default schedule.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.training.utils.diffusion_mixin import DiffusionStrategyMixin


class _Host(DiffusionStrategyMixin):
    """Minimal concrete host — the mixin is only ever used via composition."""


def test_beta_range_reaches_the_scheduler() -> None:
    host = _Host()
    host.initialize_diffusion_parameters(
        num_timesteps=10,
        beta_schedule="linear",
        beta_start=0.05,
        beta_end=0.5,
    )
    assert host.scheduler.beta_start == pytest.approx(0.05)
    assert host.scheduler.beta_end == pytest.approx(0.5)
    # The endpoints of the linear schedule ARE the declared range.
    assert host.scheduler.betas[0].item() == pytest.approx(0.05)
    assert host.scheduler.betas[-1].item() == pytest.approx(0.5)


def test_declared_range_differs_from_the_default_schedule() -> None:
    """Regression: pre-fix both hosts produced the identical default betas."""
    declared = _Host()
    declared.initialize_diffusion_parameters(
        num_timesteps=16, beta_schedule="linear", beta_start=0.02, beta_end=0.3
    )
    default = _Host()
    default.initialize_diffusion_parameters(num_timesteps=16, beta_schedule="linear")

    assert not torch.allclose(declared.scheduler.betas, default.scheduler.betas)
    assert default.scheduler.betas[0].item() == pytest.approx(1e-4)
    assert default.scheduler.betas[-1].item() == pytest.approx(0.02)


def test_defaults_are_unchanged_when_caller_omits_the_range() -> None:
    """Callers that never passed a range must keep the historical schedule."""
    host = _Host()
    host.initialize_diffusion_parameters(num_timesteps=1000, beta_schedule="linear")
    assert host.scheduler.beta_start == pytest.approx(1e-4)
    assert host.scheduler.beta_end == pytest.approx(0.02)


def test_cosine_schedule_ignores_the_beta_range() -> None:
    """The cosine schedule is parameterised by T alone — the range is inert."""
    wide = _Host()
    wide.initialize_diffusion_parameters(
        num_timesteps=32, beta_schedule="cosine", beta_start=0.2, beta_end=0.9
    )
    narrow = _Host()
    narrow.initialize_diffusion_parameters(
        num_timesteps=32, beta_schedule="cosine", beta_start=1e-5, beta_end=1e-3
    )
    assert torch.allclose(wide.scheduler.betas, narrow.scheduler.betas)
