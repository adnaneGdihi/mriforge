"""Regression test for the 2026-06 audit fix to VAE KL annealing.

The start->end KL anneal knobs (anneal_kl_beta / kl_beta_start / kl_anneal_steps)
were unbacked: the live path read only kl_beta_end and ignored ``iteration``, so
the dead ``VAETrainingStrategy._get_kl_weight`` was the sole reader. Annealing is
now performed by ``UnifiedVAELossComputer._get_loss_weight('kl', ..., iteration)``.
"""
from __future__ import annotations

import types

from mriforge.models.losses.computers.unified_vae import UnifiedVAELossComputer


def _computer(vae_ns) -> UnifiedVAELossComputer:
    c = object.__new__(UnifiedVAELossComputer)  # bypass heavy __init__
    c.config = types.SimpleNamespace(
        training=types.SimpleNamespace(vae=vae_ns, latent=None)
    )
    return c


def test_kl_weight_linearly_anneals_with_iteration() -> None:
    vae_ns = types.SimpleNamespace(
        anneal_kl_beta=True,
        kl_beta_start=0.0,
        kl_beta_end=1.0,
        kl_anneal_steps=100,
    )
    c = _computer(vae_ns)
    assert c._get_loss_weight("kl", 0, 0) == 0.0  # start
    assert abs(c._get_loss_weight("kl", 0, 50) - 0.5) < 1e-6  # halfway
    assert c._get_loss_weight("kl", 0, 100) == 1.0  # end
    assert c._get_loss_weight("kl", 0, 500) == 1.0  # clamped past warmup


def test_kl_weight_constant_when_anneal_disabled() -> None:
    vae_ns = types.SimpleNamespace(
        anneal_kl_beta=False,
        kl_beta_start=0.0,
        kl_beta_end=0.25,
        kl_anneal_steps=100,
    )
    c = _computer(vae_ns)
    # Without annealing the weight is kl_beta_end regardless of iteration.
    assert c._get_loss_weight("kl", 0, 0) == 0.25
    assert c._get_loss_weight("kl", 0, 9999) == 0.25
