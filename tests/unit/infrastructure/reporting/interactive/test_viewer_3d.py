"""Tests for the interactive 3-D volumetric viewer.

Targets ``mriforge.infrastructure.reporting.interactive.viewer_3d``: builds a
Z-slice scrubber from a case carrying a 3-D ``*_volume`` array, and soft-skips
honestly on 2-D-only cases (single-slice data) or empty input.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from mriforge.infrastructure.reporting.interactive import viewer_3d as v3d

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def _vol_case():
    rng = np.random.default_rng(0)
    tgt = rng.random((10, 32, 32)).astype(np.float32)
    pred = (tgt + 0.05 * rng.random((10, 32, 32))).astype(np.float32)
    return {"prediction_volume": pred, "target_volume": tgt, "rank": "best",
            "case_id": "vol0", "prediction": pred[5], "target": tgt[5]}


def _flat_case():
    return {"prediction": np.zeros((16, 16), np.float32),
            "target": np.zeros((16, 16), np.float32), "rank": "best",
            "case_id": "flat"}


def test_returns_none_on_empty():
    assert v3d.volume_viewer_3d_div([]) is None
    assert v3d.volume_viewer_3d_div(None) is None


def test_soft_skips_on_2d_only_cases():
    # single-slice data (no *_volume) -> None, the honest outcome on M4Raw
    assert v3d.volume_viewer_3d_div([_flat_case(), _flat_case()]) is None


def test_builds_slice_scrubber_from_volume():
    div = v3d.volume_viewer_3d_div([_flat_case(), _vol_case()])
    if _HAS_PLOTLY:
        assert isinstance(div, str)
        assert "volume-3d" in div
        assert "slice" in div.lower()
        assert 'src="https://cdn' not in div
    else:
        assert div is None


def test_none_when_plotly_absent(monkeypatch):
    monkeypatch.setattr(v3d, "has_plotly", lambda: False)
    assert v3d.volume_viewer_3d_div([_vol_case()]) is None
