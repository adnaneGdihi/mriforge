"""Tier-1 gradcheck tests for DifferentiableBlochLayer / DifferentiableBlochSimulator.

``torch.autograd.gradcheck`` operates in fp64.  Every test here allocates
fp64 inputs with ``requires_grad=True`` and uses the default (fp32-unfriendly)
tolerances adjusted for fp64: eps=1e-6, atol=1e-5.

Physiological values used throughout:
  T1  = 1000 ms  (≈ grey matter at 3 T)
  T2  = 100 ms   (≈ grey matter at 3 T)
  T2* = 50 ms
  PD  = 1.0      (normalised proton density)
  TR  = 10.0 ms  (fast SPGR)
  TE  = 3.0 ms   (fast SPGR)
  flip = π/6 rad (30°)

Constraint enforced implicitly: the layer uses ``softplus(t1_map)`` internally
so T2 ≤ T1 is not required on the raw input tensor — the layer itself maps any
real input to positive T1/T2 values.  We nonetheless initialise with
physiologically ordered values to keep gradcheck near a meaningful operating
point.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spectramr.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer
from spectramr.infrastructure.physics.bloch_simulation import DifferentiableBlochSimulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-6
_ATOL = 1e-5

# Spatial shape for all tests: 1 batch, 1 channel, 2x2 (small → fast finite diff)
_SHAPE = (1, 1, 2, 2)

# Representative physiological values (ms units, as the layer expects)
_T1_VAL = 1000.0
_T2_VAL = 100.0
_PD_VAL = 1.0
_TR = 10.0  # ms
_TE = 3.0  # ms
_FA = torch.pi / 6.0  # 30°


def _make_fp64(value: float, shape: tuple[int, ...] = _SHAPE) -> torch.Tensor:
    """Return a small fp64 tensor filled with *value* and requires_grad=True."""
    return torch.full(shape, value, dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------------------
# DifferentiableBlochLayer (SPGR) — real-valued output (no B0/B1 maps)
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_spgr_bloch_wrt_t1():
    """Gradcheck of SPGR forward w.r.t. T1 map (real output, no B0)."""
    layer = DifferentiableBlochLayer(sequence_type="SPGR").double()

    t1 = _make_fp64(_T1_VAL)
    t2 = _make_fp64(_T2_VAL)
    pd = _make_fp64(_PD_VAL)
    fa = torch.tensor(_FA, dtype=torch.float64)

    def fn(t1_in):
        return layer(t1_in, t2, pd, tr=_TR, te=_TE, flip_angle_rad=fa)

    assert gradcheck(fn, (t1,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_spgr_bloch_wrt_t2():
    """Gradcheck of SPGR forward w.r.t. T2 map (real output, no B0)."""
    layer = DifferentiableBlochLayer(sequence_type="SPGR").double()

    t1 = _make_fp64(_T1_VAL)
    t2 = _make_fp64(_T2_VAL)
    pd = _make_fp64(_PD_VAL)
    fa = torch.tensor(_FA, dtype=torch.float64)

    def fn(t2_in):
        return layer(t1, t2_in, pd, tr=_TR, te=_TE, flip_angle_rad=fa)

    assert gradcheck(fn, (t2,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_spgr_bloch_wrt_pd():
    """Gradcheck of SPGR forward w.r.t. proton density map (real output)."""
    layer = DifferentiableBlochLayer(sequence_type="SPGR").double()

    t1 = _make_fp64(_T1_VAL)
    t2 = _make_fp64(_T2_VAL)
    pd = _make_fp64(_PD_VAL)
    fa = torch.tensor(_FA, dtype=torch.float64)

    def fn(pd_in):
        return layer(t1, t2, pd_in, tr=_TR, te=_TE, flip_angle_rad=fa)

    assert gradcheck(fn, (pd,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_spgr_bloch_wrt_all_tissue_jointly():
    """Joint gradcheck w.r.t. T1, T2, and PD simultaneously (real output)."""
    layer = DifferentiableBlochLayer(sequence_type="SPGR").double()

    t1 = _make_fp64(_T1_VAL)
    t2 = _make_fp64(_T2_VAL)
    pd = _make_fp64(_PD_VAL)
    fa = torch.tensor(_FA, dtype=torch.float64)

    def fn(t1_in, t2_in, pd_in):
        return layer(t1_in, t2_in, pd_in, tr=_TR, te=_TE, flip_angle_rad=fa)

    assert gradcheck(fn, (t1, t2, pd), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_spgr_bloch_with_b1_map():
    """Gradcheck w.r.t. B1 transmit field map (real output, no B0)."""
    layer = DifferentiableBlochLayer(sequence_type="SPGR").double()

    t1 = _make_fp64(_T1_VAL)
    t2 = _make_fp64(_T2_VAL)
    pd = _make_fp64(_PD_VAL)
    b1 = _make_fp64(1.0)  # nominal B1, slight variation ok for gradcheck
    fa = torch.tensor(_FA, dtype=torch.float64)

    def fn(b1_in):
        return layer(t1, t2, pd, tr=_TR, te=_TE, flip_angle_rad=fa, b1_map=b1_in)

    assert gradcheck(fn, (b1,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_spgr_bloch_complex_output_real_part():
    """Gradcheck w.r.t. T1 when B0 is supplied (complex output); checks real part."""
    layer = DifferentiableBlochLayer(sequence_type="SPGR").double()

    t1 = _make_fp64(_T1_VAL)
    t2 = _make_fp64(_T2_VAL)
    pd = _make_fp64(_PD_VAL)
    b0 = _make_fp64(10.0)  # 10 Hz off-resonance
    fa = torch.tensor(_FA, dtype=torch.float64)

    # gradcheck requires real-valued output (or a scalar functional).
    # We extract the real part to keep the function real-valued.
    def fn(t1_in):
        sig = layer(t1_in, t2, pd, tr=_TR, te=_TE, flip_angle_rad=fa, b0_map=b0)
        return sig.real

    assert gradcheck(fn, (t1,), eps=_EPS, atol=_ATOL, raise_exception=True)


# ---------------------------------------------------------------------------
# DifferentiableBlochSimulator (SE / GRE)
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_bloch_simulator_se_wrt_tissue_maps():
    """Gradcheck of DifferentiableBlochSimulator (SE) w.r.t. tissue_maps.

    tissue_maps layout: [B, 3, H, W] = (PD, T1, T2) concatenated on dim-1.
    The simulator applies torch.abs() internally, so all-positive input is safe.
    """
    sim = DifferentiableBlochSimulator(sequence_type="SE").double()
    # tissue_maps: (PD=1.0, T1=1000 ms, T2=100 ms)
    tissue_maps = (
        torch.tensor(
            [[[[_PD_VAL]], [[_T1_VAL]], [[_T2_VAL]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        .expand(1, 3, 2, 2)
        .clone()
        .requires_grad_(True)
    )

    def fn(tissue):
        return sim(tissue, TR=_TR, TE=_TE)

    assert gradcheck(fn, (tissue_maps,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_bloch_simulator_gre_wrt_tissue_maps():
    """Gradcheck of DifferentiableBlochSimulator (GRE) w.r.t. tissue_maps."""
    sim = DifferentiableBlochSimulator(sequence_type="GRE").double()
    tissue_maps = (
        torch.tensor(
            [[[[_PD_VAL]], [[_T1_VAL]], [[_T2_VAL]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        .expand(1, 3, 2, 2)
        .clone()
        .requires_grad_(True)
    )

    def fn(tissue):
        return sim(tissue, TR=_TR, TE=_TE, alpha=30.0)

    assert gradcheck(fn, (tissue_maps,), eps=_EPS, atol=_ATOL, raise_exception=True)


# ---------------------------------------------------------------------------
# Doc-contract regression: forward() docstring must not carry the duplicated
# auto-generated boilerplate that contradicted the real (concomitant-aware)
# signature.  The boilerplate (a) re-documented args generically, (b) omitted
# the keyword-only concomitant params, and (c) claimed irrelevant
# AMP/CUDA-stream/DataStagingService behaviour for this pure-physics layer.
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_forward_docstring_drops_autogenerated_boilerplate():
    """forward() docstring must no longer claim DataStagingService / generic args."""
    doc = DifferentiableBlochLayer.forward.__doc__ or ""
    # The misleading auto-generated boilerplate is gone...
    assert "DataStagingService" not in doc
    assert "Executes PyTorch tensor operations." not in doc
    assert "Expected input tensor." not in doc
    # ...while the real, hand-written contract is preserved.
    assert "SPGR Magnitude" in doc
    assert "B1 Correction" in doc


@pytest.mark.physics
def test_forward_signature_documents_concomitant_knobs():
    """The keyword-only concomitant knobs remain part of the real forward() seam."""
    import inspect

    params = inspect.signature(DifferentiableBlochLayer.forward).parameters
    for knob in (
        "include_concomitant",
        "field_strength_t",
        "gradient_amplitudes_mT_per_m",
        "voxel_size_mm",
        "concomitant_auto_threshold_t",
    ):
        assert knob in params, f"forward() lost concomitant knob {knob!r}"
        assert params[knob].kind is inspect.Parameter.KEYWORD_ONLY

    # And the layer still runs end-to-end with the documented real-output path.
    layer = DifferentiableBlochLayer(sequence_type="SPGR")
    t1 = torch.full(_SHAPE, _T1_VAL)
    t2 = torch.full(_SHAPE, _T2_VAL)
    pd = torch.full(_SHAPE, _PD_VAL)
    out = layer(t1, t2, pd, tr=_TR, te=_TE, flip_angle_rad=_FA)
    assert out.shape == _SHAPE
    assert not torch.is_complex(out)  # no b0_map -> real magnitude
