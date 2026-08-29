"""Unit tests for ``infrastructure.physics.integration`` contracts.

Regression coverage for the WS-4 physics review of the forward-operator slice:

* ``CoilSensitivityEstimator``: no silent ESPIRiT->JSENSE substitution. An
  explicit ``method="espirit"`` (or ``estimate_espirit``) must RAISE when sigpy
  is unavailable; ``method="auto"`` is the only path that may downgrade, and a
  genuine (non-ImportError) ESPIRiT failure must propagate rather than be
  swallowed (pitfall #9).
* ``BlochEquationSolver.evolve_phase_multi_isochromat(None, ...)`` must return a
  device-agnostic 0-dim unity phasor (the removed ``self.b0_device`` attribute
  was never assigned).
* ``create_physics_operator`` resolves through the operator registry with no
  hardcoded if/elif fallback; an unknown operator type raises ``ValueError``
  (pitfall #9 / #12).
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.integration import (
    BlochEquationSolver,
    CoilSensitivityEstimator,
    create_physics_operator,
)
from mriforge.infrastructure.physics.implementations import FFT2DOperator


def _kspace_real_last(b: int, c: int, h: int, w: int) -> torch.Tensor:
    """Build a deterministic multi-coil k-space in the [B, C, H, W, 2] layout."""
    torch.manual_seed(0)
    return torch.randn(b, c, h, w, 2)


class TestCoilSensitivityEstimatorNoSilentFallback:
    def test_estimate_espirit_raises_without_sigpy(self):
        """estimate_espirit must surface the missing optional dependency.

        sigpy is not installed in this environment, so this exercises the real
        ImportError branch; the fix turns the former silent JSENSE substitution
        into a raise.
        """
        try:
            import sigpy.mri  # noqa: F401
        except ImportError:
            cse = CoilSensitivityEstimator()
            ks = _kspace_real_last(1, 4, 16, 16)
            with pytest.raises(ImportError, match="sigpy is required"):
                cse.estimate_espirit(ks)
        else:
            pytest.skip("sigpy installed; ImportError branch unreachable")

    def test_forward_espirit_raises_without_sigpy(self):
        try:
            import sigpy.mri  # noqa: F401
        except ImportError:
            cse = CoilSensitivityEstimator()
            ks = _kspace_real_last(1, 4, 16, 16)
            with pytest.raises(ImportError, match="sigpy is required"):
                cse.forward(ks, method="espirit")
        else:
            pytest.skip("sigpy installed; ImportError branch unreachable")

    def test_forward_auto_still_falls_back_to_jsense_on_missing_sigpy(self):
        """auto MAY downgrade to JSENSE on a missing-dependency ImportError."""
        cse = CoilSensitivityEstimator()
        ks = _kspace_real_last(1, 4, 16, 16)
        out = cse.forward(ks, method="auto")
        assert out.shape == (1, 4, 16, 16, 2)
        assert torch.isfinite(out).all()

    def test_forward_auto_propagates_genuine_espirit_error(self, monkeypatch):
        """A real (non-ImportError) ESPIRiT failure must NOT be swallowed.

        Pre-fix the catch-all ``except (ImportError, Exception)`` returned JSENSE
        on every error; now only ImportError is recoverable in the auto path.
        """
        cse = CoilSensitivityEstimator()

        def _boom(*a, **k):
            raise RuntimeError("simulated ESPIRiT failure")

        monkeypatch.setattr(cse, "estimate_espirit", _boom)
        ks = _kspace_real_last(1, 4, 16, 16)
        with pytest.raises(RuntimeError, match="simulated ESPIRiT failure"):
            cse.forward(ks, method="auto")


class TestBlochNoB0Map:
    def test_no_b0map_returns_device_agnostic_unity_phasor(self):
        solver = BlochEquationSolver(t1=1000.0, t2=100.0, te=50.0)
        phasor = solver.evolve_phase_multi_isochromat(None, te=50.0)
        assert phasor.dim() == 0  # 0-dim scalar broadcasts on any device
        assert torch.allclose(phasor, torch.ones(()))

    def test_forward_no_b0map_is_pure_t2_decay(self):
        solver = BlochEquationSolver(t1=1000.0, t2=100.0, te=50.0)
        m0 = torch.zeros((1, 4, 4, 3))
        m0[..., 2] = 1.0
        sig = solver.forward(m0, b0_map=None)
        mag = torch.linalg.vector_norm(sig, dim=-1)
        expected = math.exp(-50.0 / 100.0)
        assert torch.allclose(mag, torch.full_like(mag, expected), atol=1e-5)


class TestCreatePhysicsOperatorRegistry:
    def test_fft2d_resolves_through_registry(self):
        op = create_physics_operator("fft2d")
        assert isinstance(op, FFT2DOperator)

    def test_unknown_operator_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown operator type"):
            create_physics_operator("totally_unknown_operator")

    def test_registration_fires_on_import(self):
        from mriforge.infrastructure.physics.registry import list_operators

        assert "fft2d" in list_operators()
