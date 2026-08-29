"""Canonical paired test for the relocated MRFDictlessMatcher.

The matcher moved from ``application/use_cases`` to ``domain/services`` so the
MRF training strategies (infrastructure) import it rightward instead of leftward
into the application layer (pitfall #13). These tests pin (a) the new canonical
home works, (b) the old application path still re-exports the SAME classes via
the shim (the ``test_dead_orchestration_deleted_2026_06`` live-module pin
depends on it), and (c) a functional NN round-trip.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from mriforge.domain.services.mrf_dictless_matching import (  # noqa: E402
    MRFDictionary,
    MRFDictlessMatcher,
)


def test_shim_reexports_same_objects() -> None:
    from mriforge.application.use_cases import mrf_dictless_matching_use_case as shim

    assert shim.MRFDictlessMatcher is MRFDictlessMatcher
    assert shim.MRFDictionary is MRFDictionary


def test_build_and_query_round_trip() -> None:
    """A trivial identity embedding must map each query to its own dictionary
    row (the nearest neighbour of an entry is itself)."""

    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    torch.manual_seed(0)
    D, N = 16, 8
    fingerprints = torch.randn(D, N)
    params = torch.randn(D, 5)
    matcher = MRFDictlessMatcher(_Identity(), device="cpu", normalise=False)
    matcher.build_index(fingerprints, params)
    out = matcher.query(fingerprints, k=1)
    assert out["indices"].shape == (D, 1)
    assert np.array_equal(out["indices"][:, 0], np.arange(D))


def test_query_before_build_raises() -> None:
    matcher = MRFDictlessMatcher(torch.nn.Identity(), device="cpu")
    with pytest.raises(RuntimeError, match="build_index"):
        matcher.query(torch.randn(4, 8))
