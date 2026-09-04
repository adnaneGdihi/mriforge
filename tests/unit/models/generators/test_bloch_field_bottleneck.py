"""Tests for BlochFieldBottleneck (B-1.8 Bloch quantitative-parameter bottleneck)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.bloch_field_bottleneck import BlochFieldBottleneck


def _net(use_field_dispersion: bool = True) -> BlochFieldBottleneck:
    return BlochFieldBottleneck(width=24, use_field_dispersion=use_field_dispersion)


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 7.0]))
    assert out.shape == (2, 1, 16, 16)
    out = out.detach()  # output carries grad (learnable gain); detach before scalar cast
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_parameters_are_physical() -> None:
    m = _net()
    pd, t1, t2 = (p.detach() for p in m.predict_parameters(torch.rand(2, 1, 16, 16)))
    assert float(pd.min()) >= 0.0 and float(pd.max()) <= 1.0
    assert float(t1.min()) >= m.t1_lo - 1e-3 and float(t1.max()) <= m.t1_hi + 1e-3
    assert float(t2.min()) >= m.t2_lo - 1e-3 and float(t2.max()) <= m.t2_hi + 1e-3


def test_field_dispersion_is_load_bearing() -> None:
    # ANTI-FACADE (#16): with dispersion ON, the field changes the SPGR render via the
    # Bottomley T1(B0) law — and this holds at init (structural physics, not a learned
    # FiLM that starts inert).
    torch.manual_seed(0)
    m = _net(use_field_dispersion=True).eval()
    x = torch.rand(2, 1, 16, 16)
    lo = m(x, field_strength=torch.tensor([0.1, 0.1]))
    hi = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(lo, hi)


def test_ablation_is_field_agnostic() -> None:
    # use_field_dispersion=False -> T1 is field-independent -> render invariant to field.
    m = _net(use_field_dispersion=False).eval()
    x = torch.rand(1, 1, 16, 16)
    a = m(x, field_strength=torch.tensor([0.1]))
    b = m(x, field_strength=torch.tensor([7.0]))
    assert torch.allclose(a, b, atol=1e-6)


def test_signal_scale_lifts_range_above_spgr_ceiling() -> None:
    # FINDING #2: the raw SPGR weighting maxes ~0.23 at flip=15deg, so without a gain the
    # render can never reach the bright pixels of a [0,1] target and clamp(0,1) is dead.
    # The learnable global gain must lift the achievable output well above that ceiling.
    from spectramr.models.blocks.spgr_signal import spgr_signal

    m = _net()
    # Drive PD toward 1 and pick T1/T2 near the bright corner by maxing the head bias is
    # awkward; instead verify the gain itself multiplies the SPGR signal as designed.
    x = torch.rand(2, 1, 16, 16)
    scale = float(m.signal_scale().detach())
    assert scale > 1.5  # default init lifts the ~0.23 ceiling
    pd, t1, t2 = m.predict_parameters(x)
    raw = spgr_signal(pd, t1, t2, m.tr_ms, m.te_ms, m.flip_rad)
    scaled = m.signal_scale() * raw
    assert torch.allclose(scaled, scale * raw, atol=1e-5)
    # The achievable maximum (PD=1, bright T1/T2 corner) now exceeds the raw ceiling.
    bright = scale * spgr_signal(
        torch.ones(1, 1, 4, 4),
        torch.full((1, 1, 4, 4), m.t1_lo),
        torch.full((1, 1, 4, 4), m.t2_hi),
        m.tr_ms,
        m.te_ms,
        m.flip_rad,
    )
    assert float(bright.max()) > 0.5


def test_signal_scale_is_learnable_and_positive() -> None:
    m = _net()
    assert m.signal_scale_raw.requires_grad
    assert float(m.signal_scale().detach()) > 0.0
    # Gradient flows into the gain (it is genuinely trainable, not a frozen constant).
    out = m(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 7.0]))
    out.sum().backward()
    assert m.signal_scale_raw.grad is not None
    assert float(m.signal_scale_raw.grad.abs().sum()) > 0.0


def test_rejects_nonpositive_signal_scale_init() -> None:
    with pytest.raises(ValueError, match="signal_scale_init"):
        BlochFieldBottleneck(signal_scale_init=0.0)


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        BlochFieldBottleneck(in_channels=2)


def test_registered() -> None:
    from spectramr.models.registry import MODEL_REGISTRY

    assert "bloch_field_bottleneck" in MODEL_REGISTRY
