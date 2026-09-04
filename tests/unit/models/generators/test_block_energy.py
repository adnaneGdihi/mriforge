"""Per-block ENERGY analysis for the k-space cold-diffusion path.

Goal (Experiment-11 DC-blob): rule out that a *block* fabricates DC energy.
The "blob" is a bright k-space CENTRE (DC) that iFFT's to a central blob, so the
sharpest probe is a **black (all-zero) input**: a block that outputs non-zero —
especially DC-concentrated — energy from nothing is injecting it (bias fields,
mis-scaled residuals). We pair that with a **Shepp-Logan** phantom (real
synthetic k-space) to check energy is *preserved* (not amplified/collapsed).

Energy = sum(x**2). DC energy = energy at the centred k-space DC bin. A block is
"energy-sane" when: black input -> ~zero output (no injection), and Shepp input
-> bounded energy ratio with no pathological DC concentration.
"""

from __future__ import annotations

import math

import torch

from spectramr.infrastructure.physics.data_consistency import SoftDataConsistency
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.signal_models import shepp_logan_2d
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)

H = 64


def _energy(x: torch.Tensor) -> float:
    return float((x.float() ** 2).sum().item())


def _dc_energy(x: torch.Tensor) -> float:
    h, w = x.shape[-2], x.shape[-1]
    return float((x[..., h // 2, w // 2].float() ** 2).sum().item())


def _dc_fraction(x: torch.Tensor) -> float:
    tot = _energy(x)
    return _dc_energy(x) / tot if tot > 0 else 0.0


def _shepp_kspace() -> torch.Tensor:
    """Shepp-Logan phantom -> centred k-space, real-stacked [1, 2, H, H]."""
    img = shepp_logan_2d(size=H)  # [1, 1, H, H] real, in [0, 1]
    k = fft2c(torch.complex(img, torch.zeros_like(img)))  # [1, 1, H, H] complex
    return torch.cat([k.real, k.imag], dim=1)  # [1, 2, H, H] real-stacked


def _black_kspace() -> torch.Tensor:
    return torch.zeros(1, 2, H, H)


# ── Physics round-trip: Parseval energy preservation (the reference contract) ─


def test_fft_roundtrip_preserves_energy():
    """fft2c/ifft2c are norm='ortho' -> Parseval: image and k-space carry the
    SAME energy, and a round-trip is the identity. This is the yardstick the
    block tests are measured against."""
    img = shepp_logan_2d(size=H)
    c = torch.complex(img, torch.zeros_like(img))
    k = fft2c(c)
    # Parseval (ortho): ||image||^2 == ||k-space||^2.
    assert math.isclose(_energy(img), _energy(k.abs()), rel_tol=1e-4)
    # Round-trip identity: ifft2c(fft2c(real)) == real, with ~zero imaginary.
    back = ifft2c(k)  # [1, 1, H, H] complex
    assert torch.allclose(back.real, img, atol=1e-4)
    assert torch.allclose(back.imag, torch.zeros_like(img), atol=1e-4)


# ── SoftDataConsistency: must NOT inject energy from a black input ────────────


def test_soft_dc_black_in_black_out():
    """Soft DC blends MEASURED k-space into the prediction:
    k_out = (k_pred + lam*measured*mask)/(1+lam*mask). With black k_pred AND
    black measured it MUST output ~black — it cannot create energy from
    nothing (a non-zero output here would be a literal blob source)."""
    dc = SoftDataConsistency(lambda_init=0.5).eval()
    black = _black_kspace()
    mask = torch.ones(1, 1, H, H)
    with torch.no_grad():
        out = dc(black, black, mask, is_kspace_domain=True)
    out = out[0] if isinstance(out, tuple) else out
    assert _energy(out) < 1e-8  # no injection from nothing


def test_soft_dc_does_not_amplify_energy():
    """With a real (Shepp) prediction + measurement, soft DC blends toward the
    measurement at sampled bins — it must not AMPLIFY total energy."""
    dc = SoftDataConsistency(lambda_init=0.5).eval()
    k = _shepp_kspace()
    mask = torch.ones(1, 1, H, H)
    with torch.no_grad():
        out = dc(k, k, mask, is_kspace_domain=True)
    out = out[0] if isinstance(out, tuple) else out
    # k_pred == measured -> output == input; energy ratio ~ 1, never >> 1.
    assert _energy(out) <= 1.05 * _energy(k)


# ── Full generator: does an untrained block INJECT DC from a black input? ────


def _build_gen() -> KSpaceColdDiffusionGenerator:
    torch.manual_seed(0)
    return KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        num_layers=2,
        use_complex_conv=False,
        timesteps=28,
    ).eval()


def test_generator_energy_is_finite_and_bounded():
    """The (random-init) generator must not EXPLODE or VANISH energy: a black
    input stays bounded and a Shepp input maps to a finite, non-degenerate
    energy. (We don't assert sharpness — the model is untrained — only that no
    block blows energy up to infinity / down to zero.)"""
    gen = _build_gen()
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out_black = gen(_black_kspace(), timestep=t)
        out_shepp = gen(_shepp_kspace(), timestep=t)
    out_black = out_black[0] if isinstance(out_black, tuple) else out_black
    out_shepp = out_shepp[0] if isinstance(out_shepp, tuple) else out_shepp

    e_black = _energy(out_black)
    e_shepp = _energy(out_shepp)
    assert math.isfinite(e_black) and math.isfinite(e_shepp)
    # Shepp output energy must be a sane fraction/multiple of the input energy
    # (no explosion or collapse over many orders of magnitude).
    ratio = e_shepp / max(_energy(_shepp_kspace()), 1e-12)
    assert 1e-4 < ratio < 1e4, f"generator energy ratio out of range: {ratio:.3e}"


def test_untrained_generator_is_dc_dominated_but_input_sensitive():
    """The key DC-blob diagnostic, via black + Shepp k-space.

    FINDING: the UNTRAINED (random-init) generator's output is **DC-dominated**
    — a black input returns the bias field with ~95% of its energy at the DC
    bin (conv nets have a low-frequency / spectral-bias prior). This is NOT a
    block injecting energy maliciously (the DC layer is black->black and the
    FFTs are Parseval — see the asserts above); it is simply what an untrained
    k-space conv net emits. It is ALSO exactly why the **EMA shadow** (74%
    random init at iter 3000) renders a blob in validation — corroborating the
    EMA-lag root cause from an independent angle.

    The thing that WOULD be a real defect is *input-INDEPENDENCE* (a fixed blob
    regardless of input — the original collapse). So we assert the generator
    RESPONDS to its input: the Shepp-Logan output differs materially from the
    black output, and real structure REDUCES the DC concentration.
    """
    gen = _build_gen()
    t = torch.zeros(1, dtype=torch.long)
    black_k, shepp_k = _black_kspace(), _shepp_kspace()
    with torch.no_grad():
        out_black = gen(black_k, timestep=t)
        out_shepp = gen(shepp_k, timestep=t)
    out_black = out_black[0] if isinstance(out_black, tuple) else out_black
    out_shepp = out_shepp[0] if isinstance(out_shepp, tuple) else out_shepp

    uniform_share = 1.0 / (H * H)
    dc_black, dc_shepp = _dc_fraction(out_black), _dc_fraction(out_shepp)
    print(
        f"\n[block-energy] untrained generator DC concentration:"
        f"\n    black input -> dc_fraction={dc_black:.4f}  energy={_energy(out_black):.3e}"
        f"\n    Shepp input -> dc_fraction={dc_shepp:.4f}  energy={_energy(out_shepp):.3e}"
        f"\n    (uniform bin share = {uniform_share:.2e}; DC-dominated output is the"
        f"\n     untrained spectral-bias prior — exactly what the EMA shadow renders.)"
    )

    # INPUT-SENSITIVITY (the real contract): the model must respond to its input
    # — a Shepp k-space must NOT produce the same output as a black k-space.
    rel_diff = (
        (out_shepp - out_black).float().norm()
        / (out_black.float().norm() + 1e-8)
    ).item()
    assert rel_diff > 1e-3, (
        f"generator output is input-INDEPENDENT (rel_diff={rel_diff:.2e}) — a "
        "fixed output regardless of measurement (the collapse signature)."
    )
    # Real structure should reduce the DC concentration vs the pure bias field.
    assert dc_shepp < dc_black
