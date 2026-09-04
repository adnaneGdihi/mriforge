"""Guard: four duplicate transform modules stay deleted, live ones stay live.

Deleted 2026-08-04. Each was a caller-free second implementation of a job the
data layer already does, and three of the four **disagreed numerically** with
the live one -- which makes them worse than clutter: wiring one would silently
change results while looking like a fix.

This file is deliberately a real test, not a docstring claim. ``data/datasets/
__init__.py`` asserted that a legacy factory bridge "was removed 2026-05-15"
and named ``test_d18_legacy_dataset_factory_deleted.py`` as the regression
guard -- that file never existed, and the bridge was still on disk and still
monkeypatching at import a quarter later. A deletion note nobody executes is
not a guard.
"""

from __future__ import annotations

import importlib

import pytest

DELETED = [
    # (module, why, the live implementation that supersedes it)
    (
        "spectramr.data.transforms.coil_compression_transform",
        "self-declared DeprecationWarning; eigendecomposed the IMAGE-space coil "
        "covariance where the live one uses k-space, applied a 99%-energy rank "
        "cut, and compressed each image on its own basis -- which destroys the "
        "input/target SNR difference the live shared-basis path preserves",
        "spectramr.data.transforms.coil_compression",
    ),
    (
        "spectramr.data.transforms.coil_sensitivity_transform",
        "self-declared DeprecationWarning; its method vocabulary "
        "(auto/low_rank/fallback) has no counterpart in the estimate_smaps SSOT "
        "vocabulary, so a YAML written against its docstring mapped onto nothing",
        "spectramr.infrastructure.physics.coil_sensitivity",
    ),
    (
        "spectramr.data.transforms.concomitant_phase_compensation",
        "took data.shape[-2:] on a TorchIO (C,W,H,D) tensor -- that is (H,D), "
        "not the (W,H) its own comment claimed -- so the concomitant phase ramp "
        "was applied across the wrong plane; the live operator is also "
        "differentiable and in-graph, where this one was neither",
        "spectramr.infrastructure.physics.concomitant_phase_operator",
    ),
    (
        "spectramr.data.transforms.non_cartesian_simulation",
        "weighted k-space by dcf.sqrt() where the live transform uses dcf, "
        "giving a different adjoint scaling (not a constant factor), and emitted "
        "a renormalised 2-channel tensor where the live one writes a magnitude "
        "image",
        "spectramr.data.transforms.non_cartesian",
    ),
]


@pytest.mark.parametrize(
    "module, why, _live", DELETED, ids=[m.rsplit(".", 1)[-1] for m, _, _ in DELETED]
)
def test_duplicate_module_stays_deleted(module, why, _live):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "_module, _why, live",
    DELETED,
    ids=[live.rsplit(".", 1)[-1] for _, _, live in DELETED],
)
def test_the_live_implementation_is_still_importable(_module, _why, live):
    """Deleting a duplicate must not take the survivor with it."""
    assert importlib.import_module(live) is not None


def test_svd_coil_compression_is_the_only_coil_compressor_in_the_data_layer():
    from spectramr.data.transforms.coil_compression import SVDCoilCompressionTransform

    assert SVDCoilCompressionTransform is not None


def test_non_cartesian_simulation_transform_is_the_only_nufft_simulator():
    from spectramr.data.transforms.non_cartesian import NonCartesianSimulationTransform

    assert NonCartesianSimulationTransform is not None
