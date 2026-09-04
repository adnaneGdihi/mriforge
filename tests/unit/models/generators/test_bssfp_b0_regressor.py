"""BSSFPB0Regressor — phase-cycled bSSFP → absolute B0 (Hz).

The Path-B model head for exp_vf_29. A frozen elliptical-signal-model
phase-cycle DFT prior + a zero-initialised CNN residual: at init the prior
dominates (Rec 3 — an untrained model cannot underperform the closed form),
and ``last_field_estimate`` is a B0 FIELD in Hz, never a corrected image (the
``bssfp_peak_finder`` facade, pitfall #16).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.multi_acquisition import MultiAcquisitionSimulator
from spectramr.models.generators.vf_field_generators import BSSFPB0Regressor
from spectramr.models.registry import get_model_capabilities


def test_registered_with_hz_field_units():
    caps = get_model_capabilities("bssfp_b0_regressor")
    assert caps is not None
    assert caps.output_field_units == "Hz"


def test_forward_returns_real_hz_field_not_complex_image():
    m = BSSFPB0Regressor(in_channels=4, out_channels=1)
    stack = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    out = m(stack)
    # a B0 field: [B, 1, H, W], real-valued (NOT a complex/2C image)
    assert out.shape == (2, 1, 16, 16)
    assert not out.is_complex()
    est = m.last_field_estimate
    assert est is not None and est.shape == (2, 1, 16, 16)
    assert not est.is_complex()


def test_accepts_real_interleaved_stack():
    # 2N real channels (first N real, next N imag) → N complex phase-cycles.
    m = BSSFPB0Regressor(in_channels=4, out_channels=1)
    stack_ri = torch.randn(2, 8, 16, 16)
    out = m(stack_ri)
    assert out.shape == (2, 1, 16, 16)
    assert not out.is_complex()


def test_dft_prior_recovers_b0_structure_at_init():
    # Untrained (zero-init residual) → the frozen DFT prior recovers the field's
    # spatial structure on a real synthesised phase-cycled stack.
    sim = MultiAcquisitionSimulator("bssfp_banding", n_phase_cycles=8)
    img = 0.5 + torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(3))
    ramp = torch.linspace(-40.0, 40.0, 32).view(1, 1, 1, 32).expand(2, 1, 32, 32)
    res = sim(img, external_field=ramp)
    m = BSSFPB0Regressor(
        in_channels=8,
        out_channels=1,
        tr_ms=sim.tr_bssfp_ms,
        t1_ms=sim.t1_ms,
        t2_ms=sim.t2_ms,
        flip_deg=60.0,
    )
    with torch.no_grad():
        est = m(res.stack)
    a = (est - est.mean()).flatten()
    b = (res.field - res.field.mean()).flatten()
    corr = (a * b).sum() / (a.norm() * b.norm() + 1e-8)
    assert float(corr) > 0.9


def test_use_dft_prior_false_is_the_no_mechanism_baseline():
    # The Δf=0 / no-mechanism baseline: with the phase-cycle DFT prior disabled,
    # the model sees only magnitude (no banding phase), so it CANNOT recover the
    # off-resonance structure — the control the real arm must beat (pitfall #17).
    sim = MultiAcquisitionSimulator("bssfp_banding", n_phase_cycles=8)
    img = 0.5 + torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(7))
    ramp = torch.linspace(-40.0, 40.0, 32).view(1, 1, 1, 32).expand(2, 1, 32, 32)
    res = sim(img, external_field=ramp)

    def _struct_corr(model):
        with torch.no_grad():
            est = model(res.stack)
        a = (est - est.mean()).flatten()
        b = (res.field - res.field.mean()).flatten()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

    common = dict(in_channels=8, out_channels=1, tr_ms=sim.tr_bssfp_ms,
                  t1_ms=sim.t1_ms, t2_ms=sim.t2_ms, flip_deg=60.0)
    on = BSSFPB0Regressor(use_dft_prior=True, **common)
    off = BSSFPB0Regressor(use_dft_prior=False, **common)
    # prior ON recovers structure at init; prior OFF (zero-init residual) does not
    assert _struct_corr(on) > 0.9
    assert abs(_struct_corr(off)) < 0.5


def test_residual_head_trainable_and_grad_flows():
    m = BSSFPB0Regressor(in_channels=4, out_channels=1)
    assert m.get_parameter_count() > 0  # the residual head is trainable
    stack = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    out = m(stack)
    out.pow(2).mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
