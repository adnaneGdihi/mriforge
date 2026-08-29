"""Tier-1 audit tests for the three SPECTRA backend checks.

Mirrors test_health_checker_json_and_new_checks.py: builds SimpleNamespace
config stubs and asserts each check fires correctly on positive AND negative
configs (passed / severity / category / yaml_keys / fix_hint). No real config
loading — these are pure Tier-1 unit tests.

Covers:
- check_spectra_backend_schema     (selected-without-schema guard)
- check_spectra_precision_envelope (FP8 output precision for diagnostic recon)
- check_spectra_gridding_sized     (J**spatial_dims vs accumulator depth)
- registration in run_all_checks
"""

from __future__ import annotations

import types

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker
from tests.utils.config_block_stub import block_stub
from tests.utils.data_config_stub import DataConfigStub


def _ns(**kwargs: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kwargs)


# --------------------------------------------------------------------------
# check_spectra_backend_schema
# --------------------------------------------------------------------------


def test_backend_schema_skips_when_no_backend() -> None:
    """No SPECTRA backend configured -> info pass (not applicable)."""
    cfg = _ns(backend_acceleration=None, undersampling=None)
    result = ConfigHealthChecker().check_spectra_backend_schema(cfg)
    assert result.passed is True
    assert result.severity == "info"


def test_backend_schema_errors_on_stray_selection_without_block() -> None:
    """Selecting spectra elsewhere without the typed block is an error."""
    cfg = _ns(backend_acceleration=None, undersampling=_ns(backend="spectra"))
    result = ConfigHealthChecker().check_spectra_backend_schema(cfg)
    assert result.passed is False
    assert result.severity == "error"
    assert result.category == "backend_acceleration"
    assert "backend_acceleration" in result.yaml_keys
    assert result.fix_hint is not None


def test_backend_schema_errors_on_wrong_backend_value() -> None:
    """A backend_acceleration block whose backend != spectra is an error."""
    cfg = _ns(backend_acceleration=_ns(backend="nvdla"), acceleration=None)
    result = ConfigHealthChecker().check_spectra_backend_schema(cfg)
    assert result.passed is False
    assert result.severity == "error"
    assert "backend_acceleration.backend" in result.yaml_keys


def test_backend_schema_passes_on_well_formed_block() -> None:
    """A well-formed spectra block passes."""
    cfg = _ns(backend_acceleration=_ns(backend="spectra"), acceleration=None)
    result = ConfigHealthChecker().check_spectra_backend_schema(cfg)
    assert result.passed is True
    assert result.severity == "info"


# --------------------------------------------------------------------------
# check_spectra_precision_envelope
# --------------------------------------------------------------------------


def test_precision_envelope_skips_when_no_backend() -> None:
    cfg = _ns(backend_acceleration=None)
    result = ConfigHealthChecker().check_spectra_precision_envelope(cfg)
    assert result.passed is True
    assert result.severity == "info"


def test_precision_envelope_errors_on_fp8_for_diagnostic_recon() -> None:
    """FP8 output precision for a reconstruction task is rejected."""
    cfg = _ns(
        backend_acceleration=_ns(pe_mode_policy="fp8"),
        training=_ns(training_mode="reconstruction"),
        model=_ns(model_type="modl"),
    )
    result = ConfigHealthChecker().check_spectra_precision_envelope(cfg)
    assert result.passed is False
    assert result.severity == "error"
    assert result.category == "backend_acceleration"
    assert "backend_acceleration.pe_mode_policy" in result.yaml_keys
    assert result.fix_hint is not None
    assert "BF16" in result.fix_hint or "bf16" in result.fix_hint


def test_precision_envelope_passes_on_bf16_for_recon() -> None:
    """BF16 meets the diagnostic envelope for a recon task."""
    cfg = _ns(
        backend_acceleration=_ns(pe_mode_policy="bf16"),
        training=_ns(training_mode="reconstruction"),
        model=_ns(model_type="modl"),
    )
    result = ConfigHealthChecker().check_spectra_precision_envelope(cfg)
    assert result.passed is True
    assert result.severity == "info"


def test_precision_envelope_allows_fp8_for_non_diagnostic_task() -> None:
    """FP8 is acceptable for a non-reconstruction task (e.g. translation)."""
    cfg = _ns(
        backend_acceleration=_ns(pe_mode_policy="fp8"),
        training=_ns(training_mode="gan"),
        model=_ns(model_type="cyclegan"),
    )
    result = ConfigHealthChecker().check_spectra_precision_envelope(cfg)
    assert result.passed is True


# --------------------------------------------------------------------------
# check_spectra_gridding_sized
# --------------------------------------------------------------------------


def test_gridding_sized_skips_when_no_backend() -> None:
    cfg = _ns(backend_acceleration=None)
    result = ConfigHealthChecker().check_spectra_gridding_sized(cfg)
    assert result.passed is True
    assert result.severity == "info"


def test_gridding_sized_passes_for_small_2d_kernel() -> None:
    """J=4 over 2-D (16 accumulators) is within the bound."""
    cfg = _ns(
        backend_acceleration=_ns(gridding_kernel_width=4),
        data=DataConfigStub(patch_size=[64, 64]),
    )
    result = ConfigHealthChecker().check_spectra_gridding_sized(cfg)
    assert result.passed is True
    assert result.severity == "info"


def test_gridding_sized_errors_for_oversized_2d_kernel() -> None:
    """J=8 over 2-D (64 accumulators) exceeds the synthesized bound."""
    cfg = _ns(
        backend_acceleration=_ns(gridding_kernel_width=8),
        data=DataConfigStub(patch_size=[64, 64]),
    )
    result = ConfigHealthChecker().check_spectra_gridding_sized(cfg)
    assert result.passed is False
    assert result.severity == "error"
    assert result.category == "backend_acceleration"
    assert "backend_acceleration.gridding_kernel_width" in result.yaml_keys
    assert result.fix_hint is not None


def test_gridding_sized_errors_for_oversized_3d_kernel() -> None:
    """J=7 over 3-D (343 accumulators) exceeds the bound of 6**3=216."""
    cfg = _ns(
        backend_acceleration=_ns(gridding_kernel_width=7),
        data=DataConfigStub(patch_size=[32, 32, 32]),
    )
    result = ConfigHealthChecker().check_spectra_gridding_sized(cfg)
    assert result.passed is False
    assert result.severity == "error"


def test_gridding_sized_defaults_to_2d_without_patch_size() -> None:
    """Missing patch_size defaults spatial_dims to 2 (J=4 -> 16 <= 36 pass)."""
    cfg = _ns(backend_acceleration=_ns(gridding_kernel_width=4), data=None)
    result = ConfigHealthChecker().check_spectra_gridding_sized(cfg)
    assert result.passed is True


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_spectra_checks_registered_in_run_all_checks() -> None:
    """All three SPECTRA checks run as part of the full ladder."""
    # The full ladder runs ~80 checks; check_model_registry (and friends) read
    # model.model_type / model_kwargs without getattr guards, so the stub must
    # carry those fields. "modl" is a registered model so the registry check
    # resolves cleanly and we reach the three SPECTRA checks under test.
    cfg = _ns(
        model=_ns(
            model_type="modl",
            model_kwargs={},
            in_channels=1,
            out_channels=1,
        ),
        training=None,
        data=None,
        losses=None,
        physics=None,
        backend_acceleration=None,
        undersampling=None,
        # `run_all_checks` runs EVERY check, and several read
        # `optimization.optimizer.*`; a config with no optimization block at all
        # is a shape no real run produces.
        optimization=block_stub("optimization"),
    )
    report = ConfigHealthChecker().run_all_checks(cfg)
    names = {r.check_name for r in report.results}
    assert "spectra_backend_schema" in names
    assert "spectra_precision_envelope" in names
    assert "spectra_gridding_sized" in names
