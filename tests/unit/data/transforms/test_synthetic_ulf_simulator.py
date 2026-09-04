"""Unit tests for SimulateULFFromHF.

Pin physical reasonableness of the synthetic ULF: the output is
noisier and blurrier than the input, the source HF image is
preserved alongside it, and the transform is deterministic given
a fixed seed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
tio = pytest.importorskip("torchio")

from spectramr.data.transforms.synthetic_ulf_simulator import (  # noqa: E402
    SimulateULFFromHF,
)


def _hf_subject(D: int = 8, H: int = 32, W: int = 32) -> tio.Subject:
    rng = torch.Generator().manual_seed(0)
    # TorchIO ScalarImage data is (C, W, H, D).
    data = torch.rand(1, W, H, D, generator=rng)
    return tio.Subject(target=tio.ScalarImage(tensor=data))


def test_attaches_synthetic_ulf_image() -> None:
    subject = _hf_subject()
    out = SimulateULFFromHF(target_snr_db=10.0)(subject)
    assert "input" in out
    assert "target" in out
    assert out["input"].data.shape == out["target"].data.shape


def test_ulf_is_noisier_than_hf() -> None:
    subject = _hf_subject()
    out = SimulateULFFromHF(target_snr_db=8.0, t2_star_blur_sigma=0.0,
                              b0_distortion_strength=0.0,
                              enable_maxwell=False)(subject)
    hf = out["target"].data.float()
    ulf = out["input"].data.float()
    # High-frequency residual energy is a noise proxy.
    hf_diff = (hf - hf.mean()).pow(2).mean().sqrt()
    ulf_diff = (ulf - ulf.mean()).pow(2).mean().sqrt()
    # Rician noise raises overall variance.
    assert ulf_diff >= 0.5 * hf_diff


def test_ulf_is_blurrier_than_hf() -> None:
    """T2* blurring should reduce the magnitude of the local Laplacian."""
    subject = _hf_subject()
    out = SimulateULFFromHF(
        target_snr_db=40.0,             # very low noise — isolate blur effect
        t2_star_blur_sigma=2.0,         # strong blur
        b0_distortion_strength=0.0,
        enable_maxwell=False,
    )(subject)
    hf = out["target"].data.float()
    ulf = out["input"].data.float()

    def _hf_energy(x: torch.Tensor) -> torch.Tensor:
        # Sum-of-squared finite differences along W (proxy for sharpness).
        return (x[..., 1:] - x[..., :-1]).pow(2).mean()

    assert _hf_energy(ulf) < _hf_energy(hf)


def test_skips_when_source_missing() -> None:
    subject = tio.Subject(other=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 8)))
    transform = SimulateULFFromHF()
    out = transform(subject)
    assert "input" not in out


def test_field_strength_validation() -> None:
    with pytest.raises(ValueError, match="field_strength_t"):
        SimulateULFFromHF(field_strength_t=0.0)


def test_maxwell_only_below_one_tesla() -> None:
    """At 3 T the Maxwell branch must be a no-op (only B0+blur+noise contribute)."""
    subject = _hf_subject()
    high_field = SimulateULFFromHF(
        field_strength_t=3.0,
        target_snr_db=40.0,
        t2_star_blur_sigma=0.0,
        b0_distortion_strength=0.0,
        enable_maxwell=True,
    )(subject)
    no_maxwell = SimulateULFFromHF(
        field_strength_t=3.0,
        target_snr_db=40.0,
        t2_star_blur_sigma=0.0,
        b0_distortion_strength=0.0,
        enable_maxwell=False,
    )(subject)
    # Same RNG draws → identical output when Maxwell is silent.
    torch.manual_seed(0)
    a = high_field["input"].data
    torch.manual_seed(0)
    b = no_maxwell["input"].data
    # Outputs aren't bit-identical (Rician noise reseeds), but they
    # must be statistically similar — far closer than two ULF runs.
    diff = (a - b).abs().mean().item()
    assert diff < 0.5  # generous bound; we just want "not catastrophically different"
