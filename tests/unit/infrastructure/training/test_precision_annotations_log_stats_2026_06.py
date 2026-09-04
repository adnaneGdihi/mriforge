"""Regression: VAE KL precision cast did 12 GPU syncs per training step.

2026-06-11 infrastructure audit. ``prepare_latent_for_kl_computation`` passed
``log_stats=True`` and ``cast_to_fp32`` evaluated the stat f-strings eagerly
(``tensor.min()/.max()/.mean()`` each ``.item()``-sync the GPU) regardless of
log level. The stats are now gated on the effective DEBUG level and the KL path
defaults ``log_stats=False``.
"""

from __future__ import annotations

import logging

import torch

from spectramr.infrastructure.training.precision_annotations import (
    PrecisionAwareCaster,
    VAEPrecisionManager,
)


def test_cast_to_fp32_does_not_sync_when_debug_disabled(monkeypatch):
    caster = PrecisionAwareCaster("vae")
    logger = logging.getLogger("spectramr.infrastructure.training.precision_annotations")
    logger.setLevel(logging.WARNING)  # DEBUG disabled

    calls = {"min": 0}
    real_min = torch.Tensor.min

    def counting_min(self, *a, **k):
        calls["min"] += 1
        return real_min(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "min", counting_min, raising=True)
    t = torch.randn(4, 4)
    caster.cast_to_fp32(t, reason="x", log_stats=True)
    assert calls["min"] == 0  # eager stat f-strings must be skipped


def test_kl_latent_prep_defaults_to_no_stats():
    mgr = VAEPrecisionManager(device=torch.device("cpu"))
    mu = torch.randn(2, 8)
    logvar = torch.randn(2, 8)
    mu32, logvar32 = mgr.prepare_latent_for_kl_computation(mu, logvar)
    assert mu32.dtype == torch.float32
    assert logvar32.dtype == torch.float32
