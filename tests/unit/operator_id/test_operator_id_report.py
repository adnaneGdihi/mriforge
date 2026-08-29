"""Tests for the operator-ID reporting utilities (Proposal 1).

Covers the two headline analyses: the commutator-interaction matrix (pure
physics, no trained model) and the recovered severity field read off the
conditioner.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from mriforge.models.analysis.operator_id_report import (
    commutator_interaction_matrix,
    severity_field,
)
from mriforge.models.operator_id.bch_conditioner import BCHOperatorConditioner

_MODES = ("motion_rigid", "bias_field_multiplicative", "b0_phase", "gibbs_truncation")


def test_commutator_matrix_shape_and_symmetry():
    matrix, modes = commutator_interaction_matrix(_MODES, height=8, width=8)
    k = len(modes)
    assert matrix.shape == (k, k)
    # Deterministic modes only (no noise modes in _MODES).
    assert modes == list(_MODES)
    # Symmetric with a zero diagonal.
    assert torch.allclose(matrix, matrix.t(), atol=1e-6)
    assert torch.allclose(torch.diagonal(matrix), torch.zeros(k, dtype=matrix.dtype))


def test_commutator_matrix_excludes_noise_modes():
    modes_with_noise = (*_MODES, "rician_noise", "gaussian_noise")
    matrix, modes = commutator_interaction_matrix(modes_with_noise, height=8, width=8)
    assert "rician_noise" not in modes
    assert "gaussian_noise" not in modes
    assert matrix.shape == (len(_MODES), len(_MODES))


def test_commutator_matrix_normalized_to_unit_peak():
    matrix, _ = commutator_interaction_matrix(_MODES, height=8, width=8, normalize=True)
    assert matrix.max() <= 1.0 + 1e-6
    # At least one pair must genuinely fail to commute (advection vs diagonal).
    assert matrix.max() > 0.0


def test_commuting_basis_is_all_zero():
    # Two diagonal (Hermitian) generators commute exactly ⇒ zero interaction.
    matrix, modes = commutator_interaction_matrix(
        ("bias_field_multiplicative", "b0_phase"), height=8, width=8
    )
    # b0_phase is i*diag(phi); bias is diag(b) — diagonal operators commute.
    assert len(modes) == 2
    assert torch.allclose(matrix, torch.zeros_like(matrix), atol=1e-6)


def test_severity_field_shape_and_nonnegative():
    cond = BCHOperatorConditioner(num_modes=4, num_contrasts=3, height=16, width=16)
    field = severity_field(
        cond,
        contrast_indices=(0, 1, 2),
        field_coords=(-0.52, 0.0),
    )
    assert field.shape == (3, 2, 4)
    assert torch.all(field >= 0.0)


def test_severity_field_defaults_to_full_vocabulary():
    cond = BCHOperatorConditioner(num_modes=3, num_contrasts=4, height=16, width=16)
    field = severity_field(cond)  # default contrasts + single field coord 0.0
    assert field.shape == (4, 1, 3)
