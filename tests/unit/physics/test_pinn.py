"""Unit tests for src/spectramr/infrastructure/physics/pinn.py.

Focus: the ``get_pde`` factory dispatch contract.

The finding notes ``get_pde`` is an if/elif string dispatch whose advertised set
(``wave_equation``, ``bloch_equation``, ``maxwell_regularizer``) is not validated
at config-load time. The schema-level fix (tightening ``PINNConfig.pde_type`` to a
``Literal``) lives in ``config/schemas/physics.py`` and is tracked as a cross-file
change; here we regression-lock the in-module runtime backstop: ``get_pde`` must
RAISE on an unknown ``pde_type`` (never silently fall back to a default) and must
accept exactly the advertised set.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.physics.pinn import (
    BlochEquation,
    MaxwellRegularizer,
    WaveEquation,
    get_pde,
)


@pytest.mark.physics
def test_get_pde_accepts_advertised_set() -> None:
    """All three advertised pde_type names resolve to their PDE classes."""
    assert isinstance(get_pde("wave_equation"), WaveEquation)
    assert isinstance(get_pde("bloch_equation"), BlochEquation)
    assert isinstance(get_pde("maxwell_regularizer"), MaxwellRegularizer)


@pytest.mark.physics
def test_get_pde_raises_on_unknown() -> None:
    """An unknown pde_type must RAISE (no silent fallback, pitfall #9)."""
    with pytest.raises(ValueError, match="Unknown PDE type"):
        get_pde("not_a_pde")


@pytest.mark.physics
def test_get_pde_error_lists_advertised_set() -> None:
    """The raise message advertises the full accepted set so the failure is
    actionable at startup."""
    with pytest.raises(ValueError) as excinfo:
        get_pde("maxwell")  # close-but-wrong name must still raise
    msg = str(excinfo.value)
    assert "wave_equation" in msg
    assert "bloch_equation" in msg
    assert "maxwell_regularizer" in msg
