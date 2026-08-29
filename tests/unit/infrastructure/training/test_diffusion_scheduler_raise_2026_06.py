"""Regression: DiffusionScheduler silently coerced unknown beta schedules.

2026-06-11 infrastructure audit. An unsupported ``beta_schedule`` (typo, or a
``sqrt`` / ``quadratic`` value advertised by ``training.noise_schedule``) was
silently rewritten to ``linear`` and the ``else: raise`` was dead. It now raises
(pitfall #9).
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.training.schedulers.diffusion_scheduler import (
    DiffusionScheduler,
)


def test_unknown_beta_schedule_raises():
    with pytest.raises(ValueError, match="Unknown beta schedule"):
        DiffusionScheduler(num_timesteps=10, beta_schedule="quadratic")


def test_supported_schedules_still_build():
    for sched in ("linear", "cosine"):
        s = DiffusionScheduler(num_timesteps=10, beta_schedule=sched)
        assert s.betas.numel() == 10
        assert s.beta_schedule == sched  # not silently rewritten
