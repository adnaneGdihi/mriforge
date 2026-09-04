"""Regression test for the 2026-06 audit fix to ``HamiltonianAcquisitionStrategy``.

The model-architecture fields (potential arch/dims/layers, integrator, dt,
T_total, shots, samples) are duplicated across ``training.hamiltonian_acquisition``
and ``model.model_kwargs`` (the model's SSOT). A blind schema trim would break the
shipped YAML (extra="forbid"); instead the strategy now RAISES if the two copies
diverge, so the duplication can never silently desync the built model from the
configured acquisition.
"""
from __future__ import annotations

import types

import pytest

from spectramr.config.schemas.training.hamiltonian_acquisition import (
    HamiltonianAcquisitionConfig,
)
from spectramr.infrastructure.training.strategies.hamiltonian_acquisition_strategy import (  # noqa: E501
    HamiltonianAcquisitionStrategy,
)


def _strategy(model_kwargs: dict) -> HamiltonianAcquisitionStrategy:
    s = object.__new__(HamiltonianAcquisitionStrategy)
    s.ham_cfg = HamiltonianAcquisitionConfig()  # all defaults
    s.config = types.SimpleNamespace(
        model=types.SimpleNamespace(model_kwargs=model_kwargs)
    )
    return s


def test_matching_model_kwargs_passes() -> None:
    cfg = HamiltonianAcquisitionConfig()
    s = _strategy(
        {
            "num_shots": cfg.num_shots,
            "potential_hidden_dim": cfg.potential_hidden_dim,
            "integrator": cfg.integrator,
            "spatial_dims": 2,  # extra (non-shared) field is ignored
        }
    )
    s._assert_model_kwargs_consistent()  # no raise


def test_divergent_model_kwargs_raises() -> None:
    cfg = HamiltonianAcquisitionConfig()
    s = _strategy({"num_shots": cfg.num_shots + 7})
    with pytest.raises(ValueError, match="num_shots"):
        s._assert_model_kwargs_consistent()


def test_absent_model_kwargs_is_noop() -> None:
    # Nothing to cross-check -> no raise.
    _strategy({})._assert_model_kwargs_consistent()
