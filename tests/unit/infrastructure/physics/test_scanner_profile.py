"""Scanner-conditioned degradation: profile physics + axis derivation.

Pure-formula, fully local: 0.3 T and 3 T must produce measurably different
chemical-shift and B0 severities, the derivation must stay theta-monotone (the
theta-invariance trap), and the axis set must respect the protocol.
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.physics.scanner_profile import (
    FASTMRI_3T,
    M4RAW_03T,
    ScannerProfile,
    active_axes,
    axis_kwargs,
    resolve_profile,
)


def test_chemical_shift_scales_with_field_strength() -> None:
    # gamma * B0 * ppm / bw -- linear in B0. 3 T >> 0.3 T.
    assert FASTMRI_3T.chemical_shift_max_px() > M4RAW_03T.chemical_shift_max_px()
    # Same bandwidth would give exactly the 10x B0 ratio; different bandwidths keep
    # 3 T clearly larger.
    same_bw_03 = ScannerProfile(name="a", b0_T=0.3, readout_bw_hz_per_px=250.0)
    same_bw_30 = ScannerProfile(name="b", b0_T=3.0, readout_bw_hz_per_px=250.0)
    assert same_bw_30.chemical_shift_max_px() == pytest.approx(
        10.0 * same_bw_03.chemical_shift_max_px(), rel=1e-6
    )


def test_axis_kwargs_chemical_shift_is_theta_scaled_and_scanner_conditioned() -> None:
    # theta-monotone: shift grows with theta (avoids the theta-invariance trap).
    lo = axis_kwargs(FASTMRI_3T, "chemical_shift", 0.2)["shift_pixels"]
    hi = axis_kwargs(FASTMRI_3T, "chemical_shift", 1.0)["shift_pixels"]
    assert 0.0 < lo < hi
    # scanner-conditioned: 3 T shift > 0.3 T shift at the same theta.
    assert (
        axis_kwargs(FASTMRI_3T, "chemical_shift", 1.0)["shift_pixels"]
        > axis_kwargs(M4RAW_03T, "chemical_shift", 1.0)["shift_pixels"]
    )
    # never forwards the raw b0_T/bw (which would flatten the axis).
    kw = axis_kwargs(FASTMRI_3T, "chemical_shift", 0.5)
    assert "b0_T" not in kw and "bw_per_pixel_hz" not in kw
    assert set(kw) == {"shift_pixels", "fat_fraction"}


def test_axis_kwargs_b0_scales_with_field_and_3t_is_the_reference() -> None:
    assert axis_kwargs(FASTMRI_3T, "b0", 1.0)["strength"] == pytest.approx(1.0)
    assert (
        axis_kwargs(M4RAW_03T, "b0", 1.0)["strength"]
        < axis_kwargs(FASTMRI_3T, "b0", 1.0)["strength"]
    )
    # theta-monotone
    assert (
        axis_kwargs(M4RAW_03T, "b0", 0.2)["strength"]
        < axis_kwargs(M4RAW_03T, "b0", 0.8)["strength"]
    )


def test_axis_kwargs_b0_geometric_distortion_emits_the_delegate_kwarg_name() -> None:
    # Regression (#16 facade): the b0_geometric_distortion delegate reads
    # ``b0_strength`` (not ``strength``). Emitting ``strength`` was swallowed by **kw
    # and fell back to theta, so 0.3 T and 3 T gave byte-identical output.
    kw3 = axis_kwargs(FASTMRI_3T, "b0_geometric_distortion", 1.0)
    kw03 = axis_kwargs(M4RAW_03T, "b0_geometric_distortion", 1.0)
    assert set(kw3) == {"b0_strength"}  # the delegate's actual param name
    assert "strength" not in kw3  # the previously-swallowed wrong name is gone
    assert kw3["b0_strength"] == pytest.approx(1.0)  # 3 T is the reference
    assert kw03["b0_strength"] < kw3["b0_strength"]  # scanner-conditioned
    # theta-monotone (avoids the theta-invariance trap)
    assert (
        axis_kwargs(M4RAW_03T, "b0_geometric_distortion", 0.2)["b0_strength"]
        < axis_kwargs(M4RAW_03T, "b0_geometric_distortion", 0.8)["b0_strength"]
    )


def test_axis_kwargs_returns_empty_for_unconditioned_axes() -> None:
    assert axis_kwargs(FASTMRI_3T, "rician", 0.5) == {}
    assert axis_kwargs(FASTMRI_3T, "gibbs", 1.0) == {}


def test_active_axes_gates_by_protocol() -> None:
    cands = [
        "complex_gaussian",
        "chemical_shift",
        "cartesian_undersamp",
        "nyquist_ghost",
        "b0",
    ]
    # M4Raw: R=1 -> no undersampling; spin-echo -> no N/2 ghost; fat present.
    m4 = active_axes(M4RAW_03T, cands)
    assert "cartesian_undersamp" not in m4
    assert "nyquist_ghost" not in m4
    assert "chemical_shift" in m4 and "complex_gaussian" in m4 and "b0" in m4
    # fastMRI: R=4 -> undersampling present.
    assert "cartesian_undersamp" in active_axes(FASTMRI_3T, cands)


def test_active_axes_drops_chemical_shift_without_fat() -> None:
    dry = ScannerProfile(name="dry", b0_T=3.0, fat_fraction=0.0)
    assert "chemical_shift" not in active_axes(dry, ["chemical_shift", "b0"])


def test_active_axes_explicit_allow_list_overrides() -> None:
    p = ScannerProfile(name="p", b0_T=3.0, active_axes=("b0",))
    assert active_axes(p, ["b0", "rician", "chemical_shift"]) == ["b0"]


def test_resolve_profile_raises_on_unknown() -> None:
    assert resolve_profile("m4raw_0.3T") is M4RAW_03T
    with pytest.raises(KeyError):
        resolve_profile("siemens_7T")


def test_to_meta_is_dependency_free_primitives() -> None:
    meta = FASTMRI_3T.to_meta()
    assert meta["name"] == "fastmri_3T" and meta["b0_T"] == 3.0
    assert all(isinstance(v, (str, int, float, type(None))) for v in meta.values())
