"""Ultra-low-field physics-noise MAP via half-quadratic splitting (B-2.1).

ULF→HF restoration framed as the MAP estimate

    x_hat = argmin_x  w/2 ‖y − A x‖²  +  R_θ(x),   w = 1/σ²,

solved by half-quadratic splitting / ADMM. Two physical ingredients make this a
genuine inverse problem rather than a denoiser:

* **Forward operator ``A`` = a field-dependent low-pass** (the ULF effective-
  resolution loss): ``A x = ifft2c(H · fft2c(x))`` with a Gaussian OTF ``H`` whose
  cutoff shrinks with field strength. The data term ``‖y − A x‖²`` therefore
  constrains the *recoverable low-frequency band*, and the learned prior fills the
  high-frequency band the low field cannot measure. (FFT round-trips go through the
  physics SSOT :func:`fft2c`/:func:`ifft2c`.)
* **Hoult-calibrated data weight** ``w = 1/σ²``, ``σ = sigma_floor/(B0/3)^{7/4}``
  (reusing :func:`_hoult_snr_scale`, the physics SSOT): a 0.1 T measurement is
  ~hundreds of times noisier than 3 T, so the data term is correctly weak at ULF
  and the prior dominates — the method does not over-trust a noise-dominated
  measurement. This SNR-calibrated balance is the contribution over an
  unconstrained generator.

The prior ``R_θ`` is a **learned restoration prior**: the spectral-norm-bounded
generator inherited from :class:`PnPStrategy` (``training.pnp``) as the prox. (We
do not claim strict Romano-Elad-Milanfar RED convergence — the generator is a
learned restoration network, not a Gaussian denoiser.)

The pure helpers (:func:`ulf_sigma`, :func:`ulf_degrade`, :func:`ulf_map_solve`)
hold the testable math; the strategy wires them to the trained generator and runs
the solve at validation so the physics mechanism is exercised, not a facade.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.models.generators.pmps_corruption_operators import _hoult_snr_scale

from .pnp_strategy import PnPStrategy

#: Cache OTF tensors by (h, w, cutoff, device, dtype). The build does an extra
#: fft2c to locate the DC bin; field_degrade/ulf_map_solve call this per step, so
#: memoising keeps the hot path allocation-free (the OTF is a read-only constant).
_OTF_CACHE: dict[tuple, torch.Tensor] = {}


def _fft2c_dc_index(h: int, w: int, device: torch.device) -> tuple[int, int]:
    """Locate the fft2c DC bin EMPIRICALLY (do not hand-derive fftshift parity).

    fft2c's centering is NOT the standard fftshift on ODD axes (it applies an
    odd-axis roll — see the physics audit), so ``h // 2`` is wrong for odd sizes
    (e.g. fft2c DC for H=7 is index 4, not 3). The DC of a constant image is its
    only nonzero centered coefficient, so its argmax is the true DC bin for both
    parities.
    """
    spec = fft2c(torch.ones(1, 1, h, w, device=device)).abs()[0, 0]
    flat = int(spec.argmax())
    return flat // w, flat % w


def _lowpass_otf(
    h: int, w: int, cutoff: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Centered real Gaussian low-pass OTF on the fft2c (centered) grid.

    ``H`` is 1 at the fft2c DC bin and decays radially; ``cutoff`` is the
    half-width in normalised frequency (smaller cutoff ⇒ more blur ⇒ lower
    effective resolution). The Gaussian is centred on the EMPIRICAL DC bin so the
    OTF is conjugate-symmetric about it. For EVEN sizes (the data case — MNI 0.5mm
    is even, e.g. 256) ``ifft2c(H·fft2c(real)).imag`` is exactly 0; for odd sizes a
    tiny residual remains (fft2c's odd-axis roll is not a clean symmetric layout),
    still ~70x smaller than the previous bug. The previous ``linspace(-1,1,h)`` grid
    placed the peak at index ``h//2-1`` for even ``h`` and missed the odd-axis roll
    entirely, discarding up to ~90% imaginary energy via ``.real``. Defining ``A``
    directly through ``H`` keeps the forward operator and closed-form inverse
    self-consistent.
    """
    key = (h, w, round(float(cutoff), 6), str(device), str(dtype))
    cached = _OTF_CACHE.get(key)
    if cached is not None:
        return cached
    dc_y, dc_x = _fft2c_dc_index(h, w, device)
    yy = (torch.arange(h, device=device, dtype=dtype) - dc_y) / max(h / 2.0, 1.0)
    xx = (torch.arange(w, device=device, dtype=dtype) - dc_x) / max(w / 2.0, 1.0)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    d2 = grid_x * grid_x + grid_y * grid_y
    cut = max(float(cutoff), 1e-3)
    h_otf = torch.exp(-d2 / (2.0 * cut * cut)).view(1, 1, h, w)
    _OTF_CACHE[key] = h_otf
    return h_otf


def ulf_degrade(x: torch.Tensor, blur_cutoff: float) -> torch.Tensor:
    """Apply the field-dependent low-pass forward operator ``A x`` (real output)."""
    h_otf = _lowpass_otf(x.shape[-2], x.shape[-1], blur_cutoff, x.device, torch.float32)
    return ifft2c(h_otf * fft2c(x)).real


def ulf_sigma(source_field: float, sigma_floor: float) -> float:
    """Hoult ULF noise std: ``sigma_floor / (source_field/3)^{7/4}``.

    Lower field ⇒ smaller Hoult scale ⇒ larger σ (noisier measurement).
    """
    return float(sigma_floor) / max(_hoult_snr_scale(float(source_field)), 1e-6)


def ulf_map_solve(
    y: torch.Tensor,
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    *,
    sigma: float,
    rho: float,
    iterations: int,
    blur_cutoff: float = 0.5,
    data_weight_max: float = 1.0e3,
) -> torch.Tensor:
    """Half-quadratic-splitting MAP solve with a field-dependent low-pass ``A``.

    Alternates a closed-form Fourier data-consistency prox (for
    ``A = ifft2c ∘ H ∘ fft2c``) and the learned-restoration prox. The closed form
    of ``argmin_x w/2‖y−Ax‖² + ρ/2‖x−(z−u)‖²`` in the (unitary) fft2c domain is

        X = (w·H·Y + ρ·F(z−u)) / (w·H² + ρ),   x = ifft2c(X).real

    so the data term constrains the passband (``H≈1``) and the prior fills the
    stopband (``H≈0``). The data weight ``w = min(1/σ², data_weight_max)`` is the
    Hoult-calibrated precision: tiny at ULF (prior dominates), large at high field.

    Args:
        y: the ULF measurement image ``[B, 1, H, W]`` (real magnitude).
        denoiser: prox-on-prior callable ``x -> D(x)`` (same shape).
        sigma: ULF noise std (see :func:`ulf_sigma`).
        rho: ADMM penalty (``training.pnp.rho``).
        iterations: ADMM iterations (``training.pnp.iterations``).
        blur_cutoff: low-pass cutoff of ``A`` (smaller ⇒ more ULF resolution loss).
        data_weight_max: clamp on ``1/σ²``.
    """
    w = min(1.0 / max(sigma**2, 1e-12), float(data_weight_max))
    h_otf = _lowpass_otf(y.shape[-2], y.shape[-1], blur_cutoff, y.device, torch.float32)
    y_k = fft2c(y)
    denom = w * h_otf * h_otf + rho
    x = y.clone()
    z = x.clone()
    u = torch.zeros_like(x)
    for _ in range(int(iterations)):
        # x-update: closed-form Fourier data prox for the low-pass A (H real).
        x_k = (w * h_otf * y_k + rho * fft2c(z - u)) / denom
        x = ifft2c(x_k).real
        # z-update: learned-restoration prox on the prior.
        out = denoiser(x + u)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("reconstruction", next(iter(out.values())))
        z = out
        # dual update.
        u = u + (x - z)
    return z


class UlfMapStrategy(PnPStrategy):
    """ULF→HF physics-noise MAP (B-2.1): PnP denoiser + Hoult-weighted data term."""

    def _setup_strategy_specific_components(self) -> None:
        # ulf_map reuses the reconstruction/PnP plumbing; accept its mode
        # explicitly (the parent only accepts reconstruction-family modes).
        self._verify_strategy_config(expected_modes=("ulf_map", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        self._ulf_cfg = getattr(self.config.training, "ulf_map", None)

    def _ulf_sigma(self) -> float:
        cfg = getattr(self, "_ulf_cfg", None)
        source = float(getattr(cfg, "source_field", 0.1)) if cfg is not None else 0.1
        floor = float(getattr(cfg, "sigma_floor", 0.05)) if cfg is not None else 0.05
        return ulf_sigma(source, floor)

    @torch.no_grad()
    def run_ulf_map(self, y: torch.Tensor) -> torch.Tensor:
        """Solve the ULF→HF physics-noise MAP for measurement ``y``.

        Uses the spectral-norm-bounded denoiser (PnP prox-on-prior) and the
        Hoult-calibrated data weight. Returns ``y`` unchanged only when PnP is
        disabled (no denoiser available) — the arm enables ``training.pnp``.
        """
        denoiser = getattr(self, "_denoiser", None) or self._build_denoiser()
        pnp = getattr(self, "_pnp_cfg", None)
        if denoiser is None or pnp is None or not getattr(pnp, "enabled", False):
            return y
        ulf = getattr(self, "_ulf_cfg", None)
        data_weight_max = (
            float(getattr(ulf, "data_weight_max", 1.0e3)) if ulf is not None else 1.0e3
        )
        blur_cutoff = float(getattr(ulf, "blur_cutoff", 0.5)) if ulf is not None else 0.5
        return ulf_map_solve(
            y,
            denoiser,
            sigma=self._ulf_sigma(),
            rho=float(pnp.rho),
            iterations=int(pnp.iterations),
            blur_cutoff=blur_cutoff,
            data_weight_max=data_weight_max,
        )

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **_: Any
    ) -> torch.Tensor:
        """Validation prediction = the physics-noise MAP solve on the ULF input.

        Wires the headline mechanism (Hoult-weighted data term + denoiser prox)
        into the validated path so the image metrics score the actually-restored
        image (anti-facade: the σ(B0) likelihood is exercised, not inert).
        """
        x0 = batch_context.get("input", input_batch)
        return self.run_ulf_map(x0)
