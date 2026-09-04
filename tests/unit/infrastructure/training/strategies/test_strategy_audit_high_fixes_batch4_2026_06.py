"""Regression tests for the 2026-06 strategy-audit subtle fixes (batch 4).

Covers vae (β=0 honored), motion_meta (block-layout destack), and
fmri_surface (dead use_cayley / tau_reg terms removed).
"""
from __future__ import annotations

import inspect


def test_vae_does_not_force_kl_weight_zero_to_one() -> None:
    from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy

    src = inspect.getsource(VAETrainingStrategy)
    # The β=0 → β=1 override is gone (a deliberate lambda_kl_divergence: 0 must
    # stay 0). Grep the exact removed construct (not a bare token).
    assert "kl_weight = 1.0" not in src
    assert "if kl_weight == 0:" not in src


def test_motion_meta_uses_block_layout_destack() -> None:
    from spectramr.infrastructure.training.strategies.motion_meta_strategy import (
        ConcreteMotionMetaTrainingStrategy,
    )

    src = inspect.getsource(ConcreteMotionMetaTrainingStrategy)
    # Interleaved indexing on a cat([real, imag]) layout is gone; block slicing
    # ([:C] / [C:]) is used instead.
    assert "pred[:, 0::2]" not in src
    assert "pred[:, :cp]" in src and "pred[:, cp:]" in src


def test_fmri_surface_dead_terms_removed() -> None:
    from spectramr.infrastructure.training.strategies import fmri_surface_strategies as mod

    # The dead-helper imports are gone from the module namespace.
    assert not hasattr(mod, "cayley_transform")
    assert not hasattr(mod, "enforce_1d_beltrami_constraint")

    spd_src = inspect.getsource(mod.RiemannianDFCDiffusionStrategy)
    assert "self.use_cayley" not in spd_src

    hrf_src = inspect.getsource(mod.HRFManifoldDiffusionStrategy)
    # The identically-zero tau term and its no-op knob are gone.
    assert "self.tau_eps" not in hrf_src
    assert "hrf_diffusion_tau_anchor" not in hrf_src
