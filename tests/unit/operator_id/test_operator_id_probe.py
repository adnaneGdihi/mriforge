"""Tests for the operator-identification Tier-2 forward probe (Proposal 1).

Exercises the dedicated runtime probe that the audit ladder dispatches to for
``operator_id`` arms: the happy path (all four properties hold) and the
failure paths (no operator_id block, no deterministic generator, a broken
adjoint).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from spectramr.infrastructure.validation.operator_id_probe import operator_id_forward_probe


def _config(operator_id: dict | None, patch=(16, 16, 1)):
    return SimpleNamespace(
        training=SimpleNamespace(operator_id=operator_id),
        data=SimpleNamespace(patch_size=list(patch)),
        model=SimpleNamespace(model_type="bch_operator_conditioner"),
    )


def test_probe_passes_for_valid_operator_id_config():
    cfg = _config(
        {
            "mode_dictionary": ["motion_rigid", "bias_field_multiplicative", "b0_phase"],
            "bch_order": 2,
            "krylov_dim": 16,
            "covariance_rank": 4,
            "num_contrasts": 3,
            "rho_bins": 8,
        }
    )
    result = operator_id_forward_probe(cfg, device="cpu")
    assert result.passed, result.message
    assert result.category == "operator_id_probe"
    assert "grad-norm" in result.message


def test_probe_skipped_when_no_operator_id_block():
    result = operator_id_forward_probe(_config(None), device="cpu")
    assert result.passed
    assert result.category == "no_probe"


def test_probe_fails_with_only_noise_modes():
    cfg = _config(
        {
            "mode_dictionary": ["rician_noise", "gaussian_noise"],
            "bch_order": 1,
            "krylov_dim": 8,
            "covariance_rank": 2,
        }
    )
    result = operator_id_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "operator_basis"


def test_probe_detects_broken_adjoint(monkeypatch):
    # Inject a generator whose adjoint is wrong; the probe must catch it.
    # The probe does `from ...degradation_generators import build_generators`
    # at call time, so patching the source module's attribute is what takes.
    import spectramr.infrastructure.physics.degradation_generators as gen_mod
    from spectramr.infrastructure.physics.magnus_exponential import LinearOperator

    real_build = gen_mod.build_generators

    def _broken_build(modes, h, w, *, device="cpu", dtype=torch.complex64):
        gens, det, noise = real_build(modes, h, w, device=device, dtype=dtype)
        # Wrap the first generator with a deliberately wrong adjoint (identity).
        g0 = gens[0]
        broken = LinearOperator(g0.apply, lambda v: v, name="broken")
        return [broken, *gens[1:]], det, noise

    monkeypatch.setattr(gen_mod, "build_generators", _broken_build)

    cfg = _config({"mode_dictionary": ["motion_rigid", "b0_phase"], "krylov_dim": 8})
    result = operator_id_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "adjoint_identity"
