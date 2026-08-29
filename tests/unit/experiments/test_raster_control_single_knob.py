"""The Hilbert-Mamba family's raster CONTROL must differ by exactly one knob.

The family's entire thesis is "Hilbert ordering beats raster", yet before
2026-06 no arm ran raster, so the headline comparison was never measured (#17).
``baseline_raster_mamba_unet`` is the missing single-knob control: it must be
byte-identical to ``baseline_hilbert_mamba_unet`` except for the Mamba scan
order (``model_kwargs.linearization_mode``). This test pins that contract so the
control cannot silently drift into a confounded comparison.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

_BASE = pathlib.Path("experiments/inprogress/geomamba_ulf/baselines")
_HILBERT = _BASE / "baseline_hilbert_mamba_unet.yaml"
_RASTER = _BASE / "baseline_raster_mamba_unet.yaml"


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_raster_control_exists() -> None:
    assert _RASTER.exists(), "raster control arm is missing"


def test_linearization_mode_is_the_only_model_difference() -> None:
    hib = _load(_HILBERT)["model"]
    ras = _load(_RASTER)["model"]
    assert ras["model_type"] == hib["model_type"]
    assert ras["in_channels"] == hib["in_channels"]
    assert ras["out_channels"] == hib["out_channels"]
    # raster control sets raster_2d; hilbert baseline sets hilbert_2d.
    assert ras["model_kwargs"]["linearization_mode"] == "raster_2d"
    assert hib["model_kwargs"]["linearization_mode"] == "hilbert_2d"
    # Every OTHER model_kwarg is identical (matched capacity).
    hib_other = {k: v for k, v in hib["model_kwargs"].items() if k != "linearization_mode"}
    ras_other = {k: v for k, v in ras["model_kwargs"].items() if k != "linearization_mode"}
    assert hib_other == ras_other


def test_data_optimization_losses_validation_blocks_are_identical() -> None:
    """The controlled blocks (everything but provenance/paths) must match, so the
    only thing that can drive a val_psnr delta is the scan order."""
    hib = _load(_HILBERT)
    ras = _load(_RASTER)
    for block in ("data", "optimization", "losses", "validation", "metrics", "physics"):
        assert ras.get(block) == hib.get(block), f"block '{block}' diverges from the control"


def test_raster_control_declares_the_hilbert_baseline() -> None:
    ras_meta = _load(_RASTER)["metadata"]
    assert ras_meta["baseline"] == "baseline_hilbert_mamba_unet"


def test_raster_control_is_registered_in_the_campaign() -> None:
    """The control must be in the campaign manifest so it is actually RUN
    alongside its Hilbert sibling — otherwise the Hilbert-vs-raster comparison
    is defined but never executed."""
    camp = yaml.safe_load(
        (
            pathlib.Path("experiments/campaigns")
            / "geomamba_ulf_super_resolution.yaml"
        ).read_text(encoding="utf-8")
    )
    names = {e["name"] for e in camp["experiments"]}
    assert "baseline_raster_mamba_unet" in names
    assert "baseline_hilbert_mamba_unet" in names
