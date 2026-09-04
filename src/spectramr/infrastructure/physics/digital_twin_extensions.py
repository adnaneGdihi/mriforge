"""Digital Twin extensions: deterministic per-theta degradations (D1-D31).

This module *extends* :mod:`digital_twin_simulator` with the missing
degradation families (D4, D5, D7, D8, D12, D13, D17, D18, D20) called
out in the meta-evaluation spec, and provides a single
:func:`apply_degradation` entry point that maps a degradation name and
a scalar severity ``theta in [0, 1]`` (plus a seed) to a deterministic
output tensor.

The deterministic-seed contract is the key requirement for the
ranking pipeline downstream (Sobol indices, mutual information,
Bradley-Terry pairs): the same ``(theta, seed)`` MUST produce the same
``hat{x}`` so the same forward model can be evaluated by many metrics.

References
----------
* Lustig et al., "Sparse MRI: The application of compressed sensing for
  rapid MR imaging," MRM 58:1182-1195, 2007. (variable-density / Poisson)
* Knoll et al., "Second order total generalized variation (TGV) for MRI,"
  MRM 65:480-491, 2011. (radial / spiral undersampling)
* Andersson et al., "How to correct susceptibility distortions in
  spin-echo echo-planar images," NeuroImage 20:870-888, 2003. (D17 model)
* Bernstein, King & Zhou, "Handbook of MRI Pulse Sequences,"
  Elsevier, 2004. (Maxwell concomitant + N/2 ghost)
* Pruessmann et al., "SENSE: sensitivity encoding for fast MRI,"
  MRM 42:952-962, 1999. (D18 sensitivity miscalibration)
* Saltelli et al., "Variance based sensitivity analysis of model
  output. Design and estimator for the total sensitivity index,"
  Comput. Phys. Commun. 181:259-270, 2010. (deterministic-seed contract
  is required for the Sobol pick-freeze estimator)
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from spectramr.config.schemas.enums import ArtifactOrigin

# Re-export existing simulators so callers do not need two imports.
from spectramr.infrastructure.physics.digital_twin_simulator import (  # noqa: F401
    simulate_b0_geometric_distortion,
    simulate_b0_inhomogeneity,
    simulate_b1_bias_field,
    simulate_chemical_shift,
    simulate_concomitant_phase,
    simulate_eddy_current,
    simulate_elastic_motion,
    simulate_gibbs_ringing,
    simulate_gradient_nonlinearity,
    simulate_magic_angle,
    simulate_periodic_motion,
    simulate_random_shot_motion,
    simulate_rf_crosstalk,
    simulate_rf_zipper,
    simulate_rigid_motion_kspace,
    simulate_spike_noise,
    simulate_undersampling,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _generator(device: torch.device, seed: int | None) -> torch.Generator:
    """Build a torch generator pinned to ``device`` and seeded.

    The Sobol / MI / BT estimators require deterministic outputs for
    a given ``(theta, seed)``. This helper isolates the seeded RNG
    from any global state.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))
    else:
        g.manual_seed(0)
    return g


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x
    return torch.complex(x, torch.zeros_like(x))


#: Elements above which ``torch.quantile`` raises; sample down instead.
_QUANTILE_MAX_ELEMS = 2**24


def _signal_level(mag: torch.Tensor, q: float = 0.99) -> float:
    """Robust signal scale of a magnitude image: its ``q``-quantile.

    Every noise axis must derive its noise power from the signal so the operator is
    **scale-invariant** — degrading then rescaling has to equal rescaling then
    degrading, because sim2rank sweeps the RAW coil-combined magnitude and normalises
    by p99 only afterwards. p99 rather than ``max`` so one hot voxel cannot set the
    noise floor.
    """
    flat = mag.detach().reshape(-1).float()
    if flat.numel() > _QUANTILE_MAX_ELEMS:
        flat = flat[:: flat.numel() // _QUANTILE_MAX_ELEMS + 1]
    return float(torch.quantile(flat, q).clamp(min=1e-12))


def _outer_line_count(H: int, n_center: int, R: float) -> int:
    """Outer PE lines to acquire so the TOTAL kept count realises acceleration ``R``.

    The declared acceleration is a statement about the whole acquisition:
    ``R = H / (lines actually acquired)`` — the fastMRI convention. The lines
    already spent on the fully-sampled ACS centre therefore come *out of* the
    budget, they are not free (issue #246: the old code budgeted
    ``(H - n_center) / R`` outer lines *on top of* the ACS, so a declared R = 8
    realised R = 5.3).
    """
    n_total = max(n_center, round(H / max(R, 1e-6)))
    return max(0, n_total - n_center)


def _calibrate_bernoulli_scale(
    p: torch.Tensor,
    target: float,
    iters: int = 40,
) -> torch.Tensor:
    """Scale ``c`` such that ``mean(clamp(c * p, max=1)) == target``.

    Bisection (``mean(clamp(c*p, 1))`` is monotone non-decreasing in ``c``), run
    branchlessly on-device so it introduces no host syncs.

    Issue #246: a Bernoulli mask that keeps each sample with probability ``p / R``
    realises an acceleration of ``R / E[p]``, **not** ``R`` — and with a
    centre-peaked ``p`` whose mean is ~0.2, a declared R = 8 came out at R = 36.
    Calibrating the scale makes the realised rate equal the declared one.
    """
    tgt = torch.as_tensor(float(target), device=p.device, dtype=p.dtype)
    lo = torch.zeros((), device=p.device, dtype=p.dtype)
    # At c = 1 / min(p) every entry clamps to 1, so the mean is 1 >= any target.
    hi = (1.0 / p.clamp(min=1e-8).min()).reshape(())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = (mid * p).clamp(max=1.0).mean() < tgt
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


# ──────────────────────────────────────────────────────────────────────
# D1 / D2 — Deterministic noise variants
# ──────────────────────────────────────────────────────────────────────


def degrade_complex_gaussian_noise(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    snr_max_db: float = 40.0,
    snr_min_db: float = 2.0,
) -> torch.Tensor:
    r"""D1 — complex Gaussian k-space noise → Rician magnitude noise.

    ``theta=0`` → ``snr_max_db`` (clean), ``theta=1`` → ``snr_min_db``.

    .. math::
        y = M F x + \eta,\quad \eta \sim \mathcal{CN}(0, \sigma^2 I)
    """
    x = _ensure_complex(x)
    device = x.device
    g = _generator(device, seed)
    snr_db = snr_max_db + (snr_min_db - snr_max_db) * float(theta)

    k = fft2c(x)
    rms = k.abs().pow(2).mean().sqrt().clamp(min=1e-8)
    # Complex noise eta = n_re + i*n_im with each channel ~ N(0, sigma^2) has
    # power E|eta|^2 = 2*sigma^2. To realise the *declared* SNR
    # (signal_power / noise_power = 10^(snr_db/10)) the per-channel std must carry
    # half the noise power, hence the 1/sqrt(2). Without it the realised SNR is
    # 3.01 dB hotter than declared (the severity axis the rankers calibrate to).
    sigma = rms / (10 ** (snr_db / 20.0)) / (2.0**0.5)

    noise = torch.complex(
        torch.empty_like(k.real).normal_(0.0, 1.0, generator=g) * sigma,
        torch.empty_like(k.imag).normal_(0.0, 1.0, generator=g) * sigma,
    )
    return ifft2c(k + noise)


def degrade_rician_noise(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    sigma_max: float = 0.3,
) -> torch.Tensor:
    """D2 — image-space Rician noise on the magnitude image.

    Applied directly to ``|x|`` so the bias term ``sigma * sqrt(pi/2)``
    at zero signal is preserved.

    ``sigma_max`` is a **fraction of the image's own p99 signal**, matching the unit
    this axis declares in ``_SEVERITY``. It used to be applied as a literal intensity,
    which made D2 the only noise axis that was not scale-invariant (D1 derives sigma
    from the k-space RMS at a declared SNR). ``sim2rank.py`` sweeps the RAW
    coil-combined magnitude (p99 ~ 1e-4 on fastMRI brain) and divides by p99 only
    *after* the sweep, so a literal ``sigma = theta * 0.3`` landed ~4 orders of
    magnitude above the signal: psnr pinned to its -30 dB floor, ssim to 0.0 and
    nema_snr to a constant at EVERY severity level, and the snapshot triptychs washed
    out to a uniform noise field.
    """
    mag = x.abs() if x.is_complex() else x
    g = _generator(mag.device, seed)
    sigma = float(theta) * sigma_max * _signal_level(mag)
    eta1 = torch.empty_like(mag).normal_(0.0, sigma, generator=g)
    eta2 = torch.empty_like(mag).normal_(0.0, sigma, generator=g)
    return torch.sqrt((mag + eta1) ** 2 + eta2**2)


# ──────────────────────────────────────────────────────────────────────
# D3 — Cartesian undersampling (deterministic)
# ──────────────────────────────────────────────────────────────────────


def degrade_cartesian_undersampling(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,
    center_fraction: float = 0.08,
) -> torch.Tensor:
    """D3 — Cartesian PE-line undersampling.

    Acceleration ramps from 1.0 (theta=0) to ``R_max`` (theta=1), and the
    **realised** rate matches the declared one: exactly ``round(H / R)`` PE lines
    are acquired, the ACS centre included in that budget, with the outer lines
    drawn from the non-ACS pool (issue #246 — the old form both over-budgeted the
    outer lines and drew them from a permutation of *all* H lines, so draws that
    landed back on the ACS were wasted; a declared R = 8 realised R = 5.3).
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x
    H = x.shape[-2]
    device = x.device
    g = _generator(device, seed)

    n_center = max(1, int(H * center_fraction))
    cs = H // 2 - n_center // 2
    ce = cs + n_center
    n_outer = _outer_line_count(H, n_center, R)

    outer_pool = torch.cat(
        [
            torch.arange(0, cs, device=device),
            torch.arange(ce, H, device=device),
        ]
    )
    perm = torch.randperm(outer_pool.numel(), generator=g, device=device)
    outer_idx = outer_pool[perm[:n_outer]]

    mask = torch.zeros(1, 1, H, 1, device=device)
    mask[..., cs:ce, :] = 1.0
    mask[..., outer_idx, :] = 1.0
    return ifft2c(fft2c(x) * mask)


# ──────────────────────────────────────────────────────────────────────
# D4 — Variable-density / Poisson-disk undersampling
# ──────────────────────────────────────────────────────────────────────


def degrade_variable_density_2d(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,
    alpha: float = 3.0,
) -> torch.Tensor:
    r"""D4 (2-D) — Variable-density 2-D Bernoulli sampling.

    Sample probability decays as :math:`p(k) \propto (1 + \|k\|)^{-\alpha}`,
    **rescaled so that the expected kept fraction is exactly** :math:`1/R`. This
    is Bernoulli, not Poisson disk (no minimum-distance constraint). Appropriate
    for non-Cartesian acquisitions; for Cartesian see
    :func:`degrade_variable_density_cartesian`.

    M5 (digital-twin review): renamed from
    ``degrade_variable_density_undersampling`` to reflect the actual sampling
    pattern. The old name remains as an alias.

    Issue #246: the mask used to keep each sample with probability ``p / R``.
    Since :math:`\mathbb{E}[p] \approx 0.2` for the centre-peaked density, the
    *realised* acceleration was ``R / E[p]`` — a declared 1.0 -> 8.0x sweep in
    fact ran 6.0 -> 36.3x. The axis was consequently **saturated at its first
    step**: theta = 0.05 already delivered 72% of the theta = 1 distortion, and
    the usable dynamic range of the whole axis was 1.4x. The scale is now
    calibrated (:func:`_calibrate_bernoulli_scale`) so declared R == realised R.
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    device = x.device
    g = _generator(device, seed)

    ky = torch.linspace(-1, 1, H, device=device).view(-1, 1)
    kx = torch.linspace(-1, 1, W, device=device).view(1, -1)
    radius = torch.sqrt(kx**2 + ky**2)
    p = (1.0 + radius) ** (-alpha)
    p = p / p.max()

    # The 5x5 DC block is always acquired; it is part of the sampling budget,
    # so the Bernoulli target is corrected for it rather than added on top.
    n_dc = min(5, H) * min(5, W)
    target = max(0.0, (float(H * W) / R - n_dc) / float(H * W))
    p_keep = (_calibrate_bernoulli_scale(p, target) * p).clamp(max=1.0)

    rand = torch.rand(H, W, generator=g, device=device)
    keep = (rand < p_keep).float()

    cy, cx = H // 2, W // 2
    keep[cy - 2 : cy + 3, cx - 2 : cx + 3] = 1.0

    mask = keep.view(1, 1, H, W)
    return ifft2c(fft2c(x) * mask)


# Back-compatibility alias: the old name still resolves.
degrade_variable_density_undersampling = degrade_variable_density_2d


def degrade_variable_density_cartesian(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,
    alpha: float = 3.0,
    center_fraction: float = 0.08,
) -> torch.Tensor:
    r"""D4b --- Variable-density *Cartesian* PE-line sampling.

    Samples whole phase-encode *lines* (each readout is acquired or
    skipped together, the physical reality of Cartesian acquisition).
    Line probability decays radially as
    :math:`p(k_y) \propto (1 + |k_y|)^{-\alpha}` and the threshold is
    calibrated against the target acceleration by quantile so the
    realised count matches the declared ``R`` (this is what the prior
    2-D Bernoulli version did *not* honour).

    M5 sibling (digital-twin review).
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x
    H = x.shape[-2]
    device = x.device
    g = _generator(device, seed)

    n_center = max(1, int(H * center_fraction))
    cs = H // 2 - n_center // 2
    ce = cs + n_center

    ky = torch.linspace(-1, 1, H, device=device)
    p_lines = (1.0 + ky.abs()) ** (-alpha)
    # Mask out center lines so they don't compete with outer-line draws.
    p_lines_outer = p_lines.clone()
    p_lines_outer[cs:ce] = 0.0

    # Outer-line budget: the ACS lines count TOWARDS the declared acceleration,
    # they are not free (issue #246 — the old `(H - n_center) / R` budget realised
    # R = 5.3 at a declared R = 8).
    n_target = max(1, _outer_line_count(H, n_center, R))
    # Incoherent variable-density draw: keep n_target DISTINCT outer lines sampled
    # WITHOUT replacement with probability proportional to p(k_y). ``topk`` would
    # deterministically keep the highest-p (central) lines -- a contiguous low-pass
    # block (resolution loss + Gibbs ringing), which is NOT the incoherent aliasing
    # that variable-density sampling is defined by (Lustig 2007). ``multinomial`` keeps
    # the EXACT declared count (honours #246) while distributing the kept lines
    # incoherently across the spectrum.
    top = torch.multinomial(p_lines_outer, n_target, replacement=False, generator=g)

    mask = torch.zeros(1, 1, H, 1, device=device)
    mask[..., cs:ce, :] = 1.0
    mask[..., top, :] = 1.0
    return ifft2c(fft2c(x) * mask)


# ──────────────────────────────────────────────────────────────────────
# D5 — Radial undersampling (Cartesian-grid approximation)
# ──────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _radial_spoke_cover(
    H: int,
    W: int,
    seed: int,
    device_str: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden-angle spoke coverage of a Cartesian grid, ordered by spoke index.

    Returns ``(first, coverage)`` where ``first[i, j]`` is the index of the FIRST
    golden-angle spoke that covers cell ``(i, j)`` and ``coverage[n]`` is the
    fraction of the grid covered by spokes ``0..n``. Because golden-angle spoke
    *prefixes* are near-uniform, ``mask(n) = (first < n)`` is the mask of an
    ``n``-spoke acquisition, and ``coverage`` is monotone — which is what lets
    :func:`degrade_radial_undersampling` pick the spoke count that realises a
    requested acceleration.

    The spoke budget is ``ceil(2*pi*r_corner)``, twice the ``pi*r`` needed for
    gapless coverage out to the grid **corner** — so the full spoke set covers
    every cell and ``theta = 0`` is a true identity. Cached per grid+seed: the
    cover is a property of the trajectory, not of the image.
    """
    device = torch.device(device_str)
    g = _generator(device, seed)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase = torch.rand(1, generator=g, device=device).item() * 2 * math.pi

    r_corner = math.sqrt(H**2 + W**2) / 2.0
    n_cap = max(8, math.ceil(2.0 * math.pi * r_corner))

    ky = torch.arange(H, device=device, dtype=torch.float32) - H // 2
    kx = torch.arange(W, device=device, dtype=torch.float32) - W // 2
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")

    first = torch.full((H, W), n_cap, device=device, dtype=torch.long)
    for j in range(n_cap):
        ang = phase + j * golden
        # Distance to the line through the origin at this angle: a 1-px-wide spoke.
        d = (KX * math.sin(ang) - KY * math.cos(ang)).abs()
        first = torch.where(
            (d < 0.5) & (first == n_cap),
            torch.full_like(first, j),
            first,
        )

    counts = torch.bincount(first.flatten(), minlength=n_cap + 1).float()
    coverage = counts.cumsum(0) / float(H * W)  # coverage[n] = frac covered by 0..n
    return first, coverage


def degrade_radial_undersampling(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,
) -> torch.Tensor:
    """D5 — Radial spoke undersampling, Cartesian-grid approximation.

    Golden-angle spokes are laid on the Cartesian grid; cells under any spoke are
    kept. Simulates the streaking artefact of a true radial acquisition without
    requiring a NUFFT.

    The severity is the **acceleration**: ``R`` ramps 1.0 (theta=0) to ``R_max``
    (theta=1), and the spoke count is chosen so the realised kept fraction of
    k-space is ``1/R``. ``R = 1`` is a fully-sampled radial acquisition, i.e. the
    identity.

    Issue #246: the axis used to ramp a raw spoke *count* from 200 down to 16, and
    200 golden-angle spokes do **not** cover a 128^2 Cartesian grid — they leave
    ~32% of the cells (the corners the ``r < half`` disk cut discarded, plus the
    peripheral gaps between spokes) unsampled. ``D(x, theta=0)`` therefore came
    back with a rel-L2 of 0.069 against its own input, a pedestal larger than the
    *entire* theta=1 distortion of ``partial_fourier`` (0.088), silently
    contaminating every statistic anchored on the clean baseline.
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x

    H, W = x.shape[-2], x.shape[-1]
    first, coverage = _radial_spoke_cover(H, W, int(seed), str(x.device))

    # Smallest spoke count whose coverage reaches the target sampling fraction.
    # NOTE: there is deliberately no `n_spokes_min` floor. The declared severity is
    # the ACCELERATION, and a spoke floor silently overrides it on small grids: at
    # 64^2 the old floor of 16 spokes forced 22.4% coverage where R=8 asks for
    # 12.5%, so the axis realised R=7.6 at theta=1 and stopped being monotone
    # between theta=0.5 and theta=1 (both clamped to the same 16-spoke mask).
    target = torch.as_tensor(1.0 / R, device=coverage.device, dtype=coverage.dtype)
    n_spokes = max(1, int(torch.searchsorted(coverage, target).item()) + 1)

    mask = (first < n_spokes).float().view(1, 1, H, W)
    return ifft2c(fft2c(x) * mask)


# ──────────────────────────────────────────────────────────────────────
# D7 — Non-rigid pulsatile motion (deterministic phase)
# ──────────────────────────────────────────────────────────────────────


def degrade_pulsatile_motion(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_amplitude_px: float = 30.0,
    base_freq_hz: float = 1.2,
    tr_s: float = 0.5,
) -> torch.Tensor:
    r"""D7 --- Cardiac-like pulsatile non-rigid deformation.

    The pulsation amplitude is evaluated at the per-PE-line acquisition
    time :math:`t(k_y) = k_y \cdot \mathrm{TR} / N_{\mathrm{PE}}`, so
    different PE lines see different points of the cardiac cycle. This
    is what makes the artefact ghost in the PE direction; a single
    static-amplitude pulse (the prior implementation) would just be a
    static deformation and a reconstructor could invert it. The
    pulsation modulates the k-space along PE rather than the image as
    a whole (Bernstein, King & Zhou 2004, Ch. 11).

    Args:
        x: complex image ``[B, C, H, W]``.
        theta: severity in ``[0, 1]``.
        seed: deterministic per-call seed.
        max_amplitude_px: peak displacement at theta=1, in pixels.
        base_freq_hz: pulsation frequency (cardiac ~1.2 Hz default).
        tr_s: TR in seconds; together with ``H`` determines the per-line
            acquisition time.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )

    if float(theta) <= 0.0:
        return x
    radial = torch.exp(-((X**2 + Y**2) / 0.5))  # breathing envelope
    amp = float(theta) * max_amplitude_px * 2.0 / max(H, W)  # px -> normalised-coord scale
    phase = torch.rand(1, generator=g, device=device).item() * 2 * math.pi

    # FIX (advisor): the pulsation must modulate the K-SPACE PE index k_y, not the image
    # row. The prior code applied one image-space grid_sample whose displacement varied
    # with the image row Y; image y and k_y are Fourier conjugates, so a row-varying warp
    # produces NO PE-direction ghost. Here the deformation is evaluated at the per-PE-line
    # acquisition time t(k_y)=k_y*TR, so the cardiac cycle completes
    # (base_freq_hz * H * TR) times across k-space -- capped so the ghosts stay discrete
    # rather than aliasing to noise.
    n_cycles = min(6.0, max(1.0, base_freq_hz * tr_s * H))
    n_poses = min(8, H)
    pose_pulse = torch.sin(
        2 * math.pi * n_cycles * torch.arange(n_poses, device=device) / n_poses + phase
    )  # cardiac phase at each acquisition-time pose

    # Deform the object once per pose (non-rigid radial breathing at that amplitude),
    # fft2c each, then gather each PE line from the pose acquired at its time -- the
    # correct per-k_y forward model (cf. simulate_rigid_motion_kspace's rotation gather).
    poses_k = []
    for a in pose_pulse:
        disp = amp * a * radial
        grid = torch.stack([X + disp * X, Y + disp * Y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        real = F.grid_sample(
            x.real, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        imag = F.grid_sample(
            x.imag, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        poses_k.append(fft2c(torch.complex(real, imag)))
    poses_k = torch.stack(poses_k, dim=0)  # [P, B, C, H, W]

    line_to_pose = (torch.arange(H, device=device) * n_poses // H).clamp(max=n_poses - 1)
    idx = line_to_pose.view(1, 1, 1, H, 1).expand(1, B, x.shape[1], H, W)
    return ifft2c(torch.gather(poses_k, 0, idx).squeeze(0))


# ──────────────────────────────────────────────────────────────────────
# D8 — N/2 (Nyquist) ghost from EPI even/odd phase mismatch
# ──────────────────────────────────────────────────────────────────────


def degrade_nyquist_ghost(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_phase_rad: float = math.pi,
) -> torch.Tensor:
    """D8 — N/2 ghost from EPI even/odd readout phase mismatch.

    Even readout lines receive zero phase, odd lines receive
    ``Δφ = theta * max_phase_rad``. This produces a ghost at FOV/2.

    Formula (per Bernstein et al.):
        K[k_y odd] *= exp(j Δφ)
    """
    x = _ensure_complex(x)
    H = x.shape[-2]
    device = x.device

    dphi = float(theta) * max_phase_rad
    line_phase = torch.zeros(H, device=device)
    line_phase[1::2] = dphi
    phase_ramp = torch.exp(1j * line_phase).view(1, 1, H, 1)

    return ifft2c(fft2c(x) * phase_ramp)


# ──────────────────────────────────────────────────────────────────────
# D12 — T2* / acquisition blur via Gaussian PSF
# ──────────────────────────────────────────────────────────────────────


def degrade_t2star_blur(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_sigma_px: float = 4.0,
) -> torch.Tensor:
    """D12 — Spatially uniform T2*-induced PSF blur (Gaussian).

    True T2* is spatially varying; we approximate as a uniform Gaussian
    PSF whose σ ramps with ``theta``. For exact spatially-varying T2*,
    use the simulate_magic_angle path which models per-voxel decay.

    Implementation note: ``F.conv2d`` is run as a *depthwise* convolution
    (``groups = C``) so multi-channel inputs (multi-coil / multi-contrast)
    receive the same blur per channel without channel-mixing. The prior
    ``groups=1`` form silently misbroadcast for ``C > 1`` (C4 in the
    digital-twin review).
    """
    sigma = max(0.0, float(theta) * max_sigma_px)
    if sigma < 1e-3:
        return x
    is_complex = x.is_complex()
    work = x if not is_complex else x.real
    work_imag = None if not is_complex else x.imag

    C = x.shape[1]
    k = max(3, int(2 * round(2.5 * sigma) + 1))
    coords = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g1d = g1d / g1d.sum()
    kernel = (g1d.unsqueeze(0) * g1d.unsqueeze(1)).view(1, 1, k, k).expand(C, 1, k, k)

    pad = k // 2
    out_real = F.conv2d(work.float(), kernel, padding=pad, groups=C)
    if is_complex:
        out_imag = F.conv2d(work_imag.float(), kernel, padding=pad, groups=C)
        return torch.complex(out_real, out_imag)
    return out_real


# ──────────────────────────────────────────────────────────────────────
# D13 — Slice-thickness / through-plane PVE (in-plane proxy)
# ──────────────────────────────────────────────────────────────────────


def degrade_slice_thickness_pve(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_kernel_px: int = 20,
) -> torch.Tensor:
    """D13 — In-plane proxy for through-plane partial-volume effect.

    For 3-D data the PVE is a through-plane convolution. For 2-D
    sections we use a one-axis Gaussian (along the readout) as a proxy
    that preserves the *anisotropic* nature of slice-thickness blur.
    """
    sigma_px = float(theta) * (max_kernel_px / 4.0)
    if sigma_px < 1e-3:
        return x
    C = x.shape[1]
    k = max(3, int(2 * round(2.5 * sigma_px) + 1))
    coords = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g1d = torch.exp(-(coords**2) / (2.0 * sigma_px**2))
    g1d = g1d / g1d.sum()
    # Anisotropic kernel: blur only along the last axis (readout).
    # Depthwise (groups=C) so multi-channel inputs blur per channel
    # without channel-mixing (C4 in the digital-twin review).
    kernel = g1d.view(1, 1, 1, k).expand(C, 1, 1, k)
    pad_w = k // 2

    is_complex = x.is_complex()
    real = F.conv2d(x.real if is_complex else x.float(), kernel, padding=(0, pad_w), groups=C)
    if is_complex:
        imag = F.conv2d(x.imag, kernel, padding=(0, pad_w), groups=C)
        return torch.complex(real, imag)
    return real


# ──────────────────────────────────────────────────────────────────────
# D28 — Coupled effective-resolution loss + noise (no resampling)
# ──────────────────────────────────────────────────────────────────────


def degrade_resolution_snr(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    fwhm_max_px: float = 2.5,
    snr_floor_db: float = 20.0,
) -> torch.Tensor:
    r"""D28 — coupled effective-resolution loss + noise, no resampling.

    A single ``theta`` ramps BOTH the effective spatial resolution and the
    noise -- the two quantities the SNR/resolution literature holds inseparable
    (``SNR ∝ voxel_volume · sqrt(T_acq)``; Edelstein 1986; Larson). The
    acquisition matrix is unchanged: resolution is lost by a **smooth k-space
    apodization** that broadens the point-spread function (effective resolution
    = PSF FWHM), never by resampling.

    - *Resolution.* Multiply centred k-space by a Gaussian window whose
      image-domain PSF has ``FWHM = theta · fwhm_max_px``. A Gaussian PSF of
      std ``sigma_x`` px is the Fourier pair of ``exp(-2 pi^2 sigma_x^2 (f)^2)``
      in normalised frequency ``f = k / N``, so the window is
      ``W(f) = exp(-2 pi^2 sigma_x^2 f^2)`` with ``sigma_x = FWHM / 2.3548``.
    - *Coupled noise.* Complex Gaussian k-space noise, calibrated per-image to
      the clean k-space RMS, with sigma linear in ``theta`` from 0 (clean) to
      the ``snr_floor_db`` level at ``theta=1``.

    This is distinct from ``t2star_blur`` (a fixed-noise Gaussian blur) and from
    ``complex_gaussian`` (fixed-resolution noise): here the two co-vary on one
    axis, so a metric that trades sharpness for noise cannot hide behind an axis
    that only moves one of them. ``theta=0`` is an exact identity (no
    apodization, no noise).
    """
    x = _ensure_complex(x)
    device = x.device
    t = float(theta)
    k = fft2c(x)
    rms = k.abs().pow(2).mean().sqrt().clamp(min=1e-8)

    # Effective-resolution loss: smooth Gaussian k-space apodization (no resample).
    fwhm = t * fwhm_max_px
    if fwhm > 1e-6:
        sigma_x = fwhm / 2.35482  # FWHM -> Gaussian std (px)
        h, w = k.shape[-2], k.shape[-1]
        fy = (torch.arange(h, device=device, dtype=torch.float32) - h // 2) / h
        fx = (torch.arange(w, device=device, dtype=torch.float32) - w // 2) / w
        f2 = fy[:, None] ** 2 + fx[None, :] ** 2  # normalised radial frequency^2
        window = torch.exp(-2.0 * (math.pi**2) * (sigma_x**2) * f2)
        k = k * window

    # Coupled noise: sigma linear in theta, calibrated to the clean signal RMS.
    if t > 0.0:
        g = _generator(device, seed)
        sigma = t * rms / (10 ** (snr_floor_db / 20.0)) / (2.0**0.5)
        noise = torch.complex(
            torch.empty_like(k.real).normal_(0.0, 1.0, generator=g) * sigma,
            torch.empty_like(k.imag).normal_(0.0, 1.0, generator=g) * sigma,
        )
        k = k + noise
    return ifft2c(k)


# ──────────────────────────────────────────────────────────────────────
# Motion types: coherent drift / abrupt jump / random jitter (D29-D31)
#
# Distinct motion TYPES beyond rigid_motion (D6, random per-PE-line SE(2)) and
# pulsatile_motion (D7, periodic non-rigid). Each displaces the object by d(k_y)
# px along READOUT while phase-encode line k_y is acquired -- a per-PE-line
# Fourier shift exp(-i 2pi k_x d(k_y)) on centred k-space. A per-line-VARYING d
# is what produces ghosting (a constant shift is globally recoverable). The
# schedule d(k_y) is what differs:
#   translation_drift : linear ramp   -> coherent directional smear
#   sudden_motion     : single step   -> discrete double-image ghost
#   random_motion     : i.i.d. jitter -> broadband ghosting (the pure-translation
#                       sibling of rigid_motion, no rotation)
# All are deterministic in (theta, seed): drift/step are closed-form; random_motion
# draws from a seeded generator.
# ──────────────────────────────────────────────────────────────────────


def _apply_pe_line_shift(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Displace the object by ``dx[k_y]`` px along readout while PE line ``k_y`` is
    acquired: multiply centred k-space by ``exp(-i 2pi k_x dx(k_y))`` (Fourier shift
    theorem, per line)."""
    k = fft2c(x)
    h, w = k.shape[-2], k.shape[-1]
    kx = torch.linspace(-0.5, 0.5, w, device=k.device).view(1, 1, 1, -1)
    dxv = dx.to(device=k.device, dtype=torch.float32).view(1, 1, h, 1)
    return ifft2c(k * torch.exp(-1j * 2.0 * math.pi * kx * dxv))


def degrade_translation_drift(
    x: torch.Tensor, theta: float, seed: int = 0, max_drift_px: float = 10.0
) -> torch.Tensor:
    """D29 -- coherent linear translational drift across the acquisition."""
    x = _ensure_complex(x)
    t = float(theta)
    if t <= 0.0:
        return x
    h = x.shape[-2]
    ky = torch.arange(h, device=x.device, dtype=torch.float32)
    dx = t * max_drift_px * (2.0 * ky / max(h - 1, 1) - 1.0)
    return _apply_pe_line_shift(x, dx)


def degrade_sudden_motion(
    x: torch.Tensor, theta: float, seed: int = 0, max_shift_px: float = 12.0
) -> torch.Tensor:
    """D30 -- single abrupt bulk shift partway through the acquisition."""
    x = _ensure_complex(x)
    t = float(theta)
    if t <= 0.0:
        return x
    h = x.shape[-2]
    dx = torch.zeros(h, device=x.device, dtype=torch.float32)
    dx[h // 2 :] = t * max_shift_px  # step at the k-space centre
    return _apply_pe_line_shift(x, dx)


def degrade_random_motion(
    x: torch.Tensor, theta: float, seed: int = 0, max_shift_px: float = 3.0
) -> torch.Tensor:
    """D31 -- i.i.d. per-PE-line random translational jitter (broadband ghosting)."""
    x = _ensure_complex(x)
    t = float(theta)
    if t <= 0.0:
        return x
    h = x.shape[-2]
    g = _generator(x.device, seed)
    dx = (torch.rand(h, generator=g, device=x.device) * 2.0 - 1.0) * (t * max_shift_px)
    return _apply_pe_line_shift(x, dx)


# ──────────────────────────────────────────────────────────────────────
# D17 — Susceptibility (dipole-convolution B0 with EPI dropout)
# ──────────────────────────────────────────────────────────────────────


def degrade_susceptibility(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_dropout_pct: float = 0.5,
    b0_axis: str = "y",
    slice_plane: str = "axial",
) -> torch.Tensor:
    """D17 — Susceptibility-induced phase shift + magnitude dropout.

    Random low-frequency susceptibility map (chi) is convolved with
    a 2-D dipole kernel (B0(r) = chi * d_z(r)) to produce a ΔB0 map.
    The resulting phase modulates the image, and a magnitude dropout
    is applied where ΔB0 is large (proxy for through-slice dephasing).

    The 3-D Fourier-domain dipole is :math:`D(\\mathbf{k}) = k_z^2 /
    \\|\\mathbf{k}\\|^2 - 1/3` (Schenck 1996). For a 2-D slice we pick
    one in-plane projection; the choice depends on slice plane and
    scanner-frame :math:`B_0` axis:

    * ``slice_plane="axial"``, ``b0_axis="z"`` --- the slice plane is
      perpendicular to :math:`B_0`; the projected 2-D kernel is
      :math:`-1/3` (a scaling, no spatial structure). We use the
      ``b0_axis``-along-PE approximation as a stand-in for the
      cylindrical-object approximation; document this caveat.
    * ``slice_plane="sagittal"`` or ``"coronal"`` --- the slice plane
      contains :math:`B_0`. We use ``k_{b0_axis}^2 / k^2 - 1/3``
      directly.

    M2 (digital-twin review): exposing these as kwargs lets callers
    pick the correct projection; default reproduces legacy
    ``ky^2 / k^2 - 1/3`` (b0_axis="y", slice_plane="axial").
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    # Low-frequency chi map (random Gaussian then upsampled)
    chi_low = torch.empty(B, 1, H // 16 + 1, W // 16 + 1, device=device).normal_(
        0.0,
        1.0,
        generator=g,
    )
    chi = F.interpolate(chi_low, size=(H, W), mode="bicubic", align_corners=False)

    # 2-D dipole projection — see docstring for the choice rationale.
    ky = torch.linspace(-1, 1, H, device=device).view(-1, 1)
    kx = torch.linspace(-1, 1, W, device=device).view(1, -1)
    k2 = (kx**2 + ky**2).clamp(min=1e-6)
    if b0_axis not in ("x", "y"):
        raise ValueError(f"b0_axis must be 'x' or 'y' for 2-D, got {b0_axis!r}.")
    if slice_plane not in ("axial", "sagittal", "coronal"):
        raise ValueError(f"slice_plane must be 'axial'/'sagittal'/'coronal', got {slice_plane!r}.")
    k_along_b0_sq = (ky if b0_axis == "y" else kx) ** 2
    dipole = (k_along_b0_sq / k2 - 1.0 / 3.0).view(1, 1, H, W)

    # Route through fft_ops so we get norm="ortho" + AMP-safe casting
    # (CLAUDE.md pitfall #2 + M1 in the digital-twin review). The chi
    # field is real, so we cast to complex once before the round-trip.
    chi_c = torch.complex(chi, torch.zeros_like(chi))
    chi_k = fft2c(chi_c)
    db0 = ifft2c(chi_k * dipole).real

    db0 = db0 / db0.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    phase = torch.exp(1j * 2.0 * math.pi * float(theta) * db0)

    # Magnitude dropout proportional to |db0|^2 (intra-voxel dephasing
    # near susceptibility hot-spots). Smooth quadratic shading produces
    # visible dimming over larger areas than the prior |db0|>0.6 binary
    # mask. Scale chosen so theta=1 yields up to ~80 % dropout at peaks.
    dropout = torch.exp(-8.0 * float(theta) * max_dropout_pct * db0.pow(2))
    keep = dropout.clamp(min=0.0, max=1.0)

    return x * phase * keep


# ──────────────────────────────────────────────────────────────────────
# D18 — Coil-sensitivity miscalibration (residual aliasing)
# ──────────────────────────────────────────────────────────────────────


def degrade_coil_sensitivity_miscal(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_relative_error: float = 1.0,
) -> torch.Tensor:
    """D18 — Coil-sensitivity miscalibration (single-coil proxy).

    Multiplies the image by a smooth ratio map ``Ŝ/S`` whose deviation
    from 1.0 grows with ``theta``. For multi-coil this would be the
    residual aliasing left after SENSE; here we emulate the effect on
    the combined image as a smooth low-frequency intensity error.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    # Smooth low-frequency error (4 random modes).
    # Mod1: batch the 12 random scalars in a single on-device draw and
    # slice; the prior implementation forced 12 host syncs via .item().
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    params = torch.rand(3, 4, generator=g, device=device)  # [fx/fy/ph, 4 modes]
    fxs = params[0] * 3.0  # [4]
    fys = params[1] * 3.0  # [4]
    phs = params[2] * 2 * math.pi  # [4]
    err = torch.zeros(B, 1, H, W, device=device)
    for j in range(4):
        err += torch.cos(fxs[j] * math.pi * X + fys[j] * math.pi * Y + phs[j])
    err = err / 4.0
    ratio = 1.0 + float(theta) * max_relative_error * err
    return x * ratio


# ──────────────────────────────────────────────────────────────────────
# D20 — Geometric warp (low-frequency displacement field)
# ──────────────────────────────────────────────────────────────────────


def degrade_geometric_warp(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_amplitude_px: float = 8.0,
    n_modes: int = 5,
) -> torch.Tensor:
    """D20 — Low-frequency geometric warp via random Fourier modes.

    Mimics registration error / motion residue via a smooth bilinear
    deformation. Amplitude ramps with ``theta``.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    amp = float(theta) * max_amplitude_px / max(H, W)
    # Mod1: batch the 3*n_modes random scalars in a single draw.
    params = torch.rand(3, n_modes, generator=g, device=device)
    fxs = params[0] * 3.0
    fys = params[1] * 3.0
    phs = params[2] * 2 * math.pi
    dx = torch.zeros(B, H, W, device=device)
    dy = torch.zeros(B, H, W, device=device)
    for j in range(n_modes):
        dx += amp * torch.sin(fxs[j] * math.pi * X + fys[j] * math.pi * Y + phs[j])
        dy += amp * torch.cos(fxs[j] * math.pi * X - fys[j] * math.pi * Y + phs[j])

    grid = torch.stack([X + dx, Y + dy], dim=-1)
    real = F.grid_sample(x.real, grid, mode="bilinear", padding_mode="border", align_corners=True)
    imag = F.grid_sample(x.imag, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return torch.complex(real, imag)


# ──────────────────────────────────────────────────────────────────────
# Bias field (D11 — image-space) with deterministic seed
# ──────────────────────────────────────────────────────────────────────


def degrade_bias_field(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_log_b: float = 1.2,
) -> torch.Tensor:
    """D11 — Multiplicative bias field, low-frequency log-Gaussian.

    Deterministic-seed version of :func:`simulate_b1_bias_field` that
    produces the same field for the same ``(theta, seed)``.

    Mod1 (digital-twin review): four scalar parameters are drawn in a
    single ``torch.rand`` call and sliced on device, eliminating four
    GPU→CPU host syncs per call.
    """
    x = _ensure_complex(x)
    _, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    rand4 = torch.rand(4, generator=g, device=device)
    fx, fy = rand4[0] * 2.0, rand4[1] * 2.0
    ph_x, ph_y = rand4[2] * 2 * math.pi, rand4[3] * 2 * math.pi

    log_b = (
        float(theta)
        * max_log_b
        * torch.cos(fx * math.pi * X + ph_x)
        * torch.cos(fy * math.pi * Y + ph_y)
    )
    bias = torch.exp(log_b).view(1, 1, H, W)
    return x * bias


# ──────────────────────────────────────────────────────────────────────
# Wave 3 (Min2) — acquisition-realism degradations
# ──────────────────────────────────────────────────────────────────────


def degrade_partial_fourier(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    min_fraction: float = 0.625,
    axis: str = "ky",
) -> torch.Tensor:
    r"""Partial-Fourier acquisition along the phase-encode axis.

    Acquires only ``frac`` of k-space lines (the high-frequency half is
    zeroed); a homodyne-style POCS reconstruction would invert this
    later but the digital twin emits the raw partially-acquired
    k-space. Used in TSE / FSE to shorten echo trains.

    ``theta = 0`` --- full acquisition (no truncation).
    ``theta = 1`` --- minimum fraction (default 5/8 = 0.625).
    """
    x = _ensure_complex(x)
    frac = 1.0 - (1.0 - min_fraction) * float(theta)
    H, W = x.shape[-2], x.shape[-1]
    device = x.device
    k = fft2c(x)
    mask = torch.ones(1, 1, H, W, device=device)
    # `mask[..., -cut:, :]` is a trap at cut == 0: -0 == 0, so the slice is [0:] -- the
    # WHOLE axis -- and the "truncate nothing" case zeroed the entire mask, returning a
    # BLACK image. theta=0 gives frac=1.0 gives cut=0, so the declared clean anchor of
    # this axis was an all-zero image (as was every theta < 1/(0.375*H): theta <= 0.02 at
    # H=128). Index from the front and skip the write when there is nothing to cut.
    if axis == "ky":
        cut = round(H * (1.0 - frac))
        if cut > 0:
            mask[..., H - cut :, :] = 0.0
    elif axis == "kx":
        cut = round(W * (1.0 - frac))
        if cut > 0:
            mask[..., W - cut :] = 0.0
    else:
        raise ValueError(f"axis must be 'ky' or 'kx'; got {axis!r}.")
    return ifft2c(k * mask)


def degrade_iq_ghost(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_dc_offset: float = 0.20,
    max_iq_imbalance: float = 0.35,
) -> torch.Tensor:
    r"""Quadrature-receiver ghost: DC offset + I/Q amplitude/phase mismatch.

    Distinct from the N/2 EPI ghost (D8): the artefact is an additive
    central-FOV spike (DC) and a conjugate-symmetric mirror image
    (I/Q imbalance). Modelled as a constant added to k-space DC plus
    a small fraction of the conjugate-flipped image.
    """
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    H, W = x.shape[-2], x.shape[-1]
    cy, cx = H // 2, W // 2

    rand2 = torch.rand(2, generator=g, device=x.device)
    dc_amp = max_dc_offset * float(theta) * (1.0 + 0.1 * (rand2[0] - 0.5))
    iq_amp = max_iq_imbalance * float(theta) * (1.0 + 0.1 * (rand2[1] - 0.5))

    # Conjugate mirror (flips around FOV centre).
    mirror = x.conj().flip(dims=(-2, -1))
    out = x + iq_amp * mirror

    # Receiver DC offset -> the classic central-point artifact: a bright spike at the
    # FOV CENTRE. Physically a real constant added to every k-space ADC sample IFFTs to a
    # delta at the image centre; adding it to the SINGLE k-space DC sample instead (as
    # before) IFFTs to a flat image-wide pedestal, which is not the named central-FOV
    # spike (advisor). Realise the central point directly in image space.
    out[..., cy, cx] = out[..., cy, cx] + dc_amp * out.abs().max()
    return out


def degrade_stimulated_echo(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_amp: float = 0.40,
    max_shift_px: float = 6.0,
) -> torch.Tensor:
    r"""Stimulated-echo ghost from imperfect refocusing in TSE / FSE.

    A small fraction of the signal is delayed by an integer number of
    echo spacings and re-imaged with a per-line phase ramp, producing
    a low-amplitude ghost shifted along the PE axis. Relevant for
    the TSE protocol used in M4Raw (Lyu et al. 2023).
    """
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    H, W = x.shape[-2], x.shape[-1]

    rand = torch.rand(1, generator=g, device=x.device)
    shift_px = float(theta) * max_shift_px * (0.7 + 0.6 * rand.item())
    amp = float(theta) * max_amp

    ky = torch.linspace(-0.5, 0.5, H, device=x.device).view(1, 1, -1, 1)
    phase_ramp = torch.exp(-1j * 2 * math.pi * ky * shift_px)

    k = fft2c(x)
    return ifft2c(k + amp * k * phase_ramp)


# ──────────────────────────────────────────────────────────────────────
# D27 (slice_profile) — REMOVED. It was a duplicate of D12 (t2star_blur).
#
# Issue #245. ``degrade_slice_profile`` was an isotropic depthwise Gaussian blur
# with ``max_sigma_px = 3.5``; ``degrade_t2star_blur`` (D12) is an isotropic
# depthwise Gaussian blur with ``max_sigma_px = 4.0``. Same operator, same
# family, differing only in a scalar: the cosine similarity of their residuals at
# theta = 1 was **0.998**. They were counted as two independent "orthogonal"
# physics axes, which double-counted the low-pass family in every per-family
# aggregation, inflated D in the Borda sum, and inflated the family count in the
# CDRS variance penalty.
#
# Its docstring claimed to be a *through-plane* effect ("the slice-select axis
# getting a non-rectangular profile"). A through-plane effect cannot be expressed
# on the single 2-D magnitude slices this sweep runs on — there is no z-neighbour
# to partial-volume with (pitfall #19: the hypothesis is untestable on this
# data). The honest move is to delete the axis, not to dress the same blur up
# twice. The anisotropic through-plane *proxy* survives as D13 (``slice_pve``);
# the isotropic in-plane PSF survives as D12 (``t2star_blur``).
#
# ``apply_degradation("slice_profile", ...)`` now raises with the list of valid
# names (pitfall #9: no silent fallback). Consumers must drop the axis — see the
# migration note in docs/degradation_simulator_correctness.rst.
# ──────────────────────────────────────────────────────────────────────


def degrade_concomitant_phase(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    b0_T: float = 0.3,
    max_gradient_mT_per_m: float = 30.0,
    voxel_size_mm: float = 1.0,
    duration_ms: float = 5.0,
) -> torch.Tensor:
    """Min1 entry point for Maxwell concomitant phase as a D-axis."""

    x = _ensure_complex(x)
    grad = float(theta) * max_gradient_mT_per_m
    return simulate_concomitant_phase(
        x,
        im_size=x.shape[-2:],
        b0_T=b0_T,
        gradient_amplitudes_mT_per_m=grad,
        voxel_size_mm=voxel_size_mm,
        duration_ms=duration_ms,
    )


# ──────────────────────────────────────────────────────────────────────
# Registry: name → (callable, default kwargs)
# ──────────────────────────────────────────────────────────────────────


MAGNITUDE: str = "magnitude"
PHASE: str = "phase"


@dataclass(frozen=True)
class DegradationSpec:
    """Specification for one degradation in the meta-evaluation taxonomy."""

    code: str  # D-number from the spec, e.g. "D4"
    name: str  # short name used as dict key, e.g. "vd_undersampling"
    family: str  # "noise", "sampling", "motion", "field", "geometric"
    fn: Callable[..., torch.Tensor]
    description: str
    severity: SeveritySpec  # REQUIRED: what theta means physically. See severity.py.
    # REQUIRED: which component of the complex image this degradation actually
    # perturbs — a subset of {"magnitude", "phase"}. See `affects` in _AFFECTS.
    affects: frozenset[str]
    # REQUIRED: where in the imaging chain the artifact arises. See _ORIGIN.
    origin: ArtifactOrigin

    @property
    def phase_only(self) -> bool:
        """True when the degradation leaves ``|x|`` untouched and only moves phase.

        Issue #223. ``concomitant_phase`` (D23) is the sole pure-phase axis in the
        taxonomy: at theta=1 its magnitude rel-L2 is 4.6e-08 while its complex
        rel-L2 is 0.71. The sweep takes ``.abs()`` before scoring, so the axis was
        handed to the metrics as a **constant** — every "metric sensitivity to
        concomitant phase" number was the sensitivity of a metric to nothing.

        A phase-only axis must be routed to the phase metrics against the
        phase-bearing complex tensor, never magnitude-scored.
        """
        return self.affects == frozenset({PHASE})

    @property
    def affects_magnitude(self) -> bool:
        return MAGNITUDE in self.affects

    @property
    def affects_phase(self) -> bool:
        return PHASE in self.affects


def _delegate_b0_geometric(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        # dwell_time controls the geometric pixel-shift magnitude
        # (the displacement is proportional to dwell_time * Delta_B0).
        # Bumped 1e-5 -> 5e-5 so theta=1 produces a clearly visible
        # voxel shift at the FOV periphery (Halbach low-field regime).
        b0_strength = float(kw.get("b0_strength", float(theta)))
        dwell_time = float(kw.get("dwell_time", 5e-5))
        out = simulate_b0_geometric_distortion(
            x, x.shape[-2:], dwell_time=dwell_time, b0_strength=b0_strength
        )
    return out


def _delegate_gradient_nl(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        # severity is multiplied by max(H, W) inside the underlying function to scale
        # polynomial coefficients in normalised [-1, 1] coords. The 1e-3 ceiling was
        # ~10-15x too strong (theta=1 gave a peripheral warp far beyond the ~5 % FOV it
        # claimed, verging on a gross geometric_warp, advisor); 7e-5 lands the corner
        # displacement near ~5 % FOV (~13 px at 256), matching clinical Halbach systems
        # (Koolstra et al. 2021).
        severity = float(kw.get("severity", float(theta) * 7e-5))
        out = simulate_gradient_nonlinearity(x, x.shape[-2:], severity=severity)
    return out


def _delegate_magic_angle(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        te = float(kw.get("te", 20.0))
        t2_base = float(kw.get("t2_base", 10.0))
        t2_magic = t2_base + (float(kw.get("t2_magic", 30.0)) - t2_base) * float(theta)
        # #223: 0.05 -> 0.30 of the ANATOMY SUPPORT (not of the whole FOV, most of
        # which is air). An MSK slice through a tendon/ligament-rich region — the
        # only anatomy where the magic-angle effect exists at all — is easily 30%
        # ordered collagen by area. At 0.05-of-FOV the axis peaked at a rel-L2 of
        # 0.14 even on its luckiest seed, ~5x weaker than the median axis (0.40);
        # it now lands at 0.33-0.48, in family with the rest of the taxonomy.
        out = simulate_magic_angle(
            x,
            x.shape[-2:],
            te=te,
            t2_base=t2_base,
            t2_magic=t2_magic,
            collagen_fraction=float(kw.get("collagen_fraction", 0.30)),
            tissue_class=kw.get("tissue_class", "msk"),
            foreground_mask=kw.get("foreground_mask"),
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Declared severity ranges (the SSOT for "what does theta=1 actually mean?")
#
# Each entry is transcribed from the degrader's own theta -> physical mapping and
# its default bounds. Keep them in sync: if you change a degrader's ramp, change
# the spec in the same edit. `_reg` below makes a missing entry an import error,
# so a new degradation cannot be added without declaring what its severity means.
#
# Values are at the DEFAULT kwargs. A caller overriding e.g. `R_max` overrides the
# physical range too; the sweep records the resolved value in its provenance.
# ──────────────────────────────────────────────────────────────────────

_P = PhysicalParam

_SEVERITY: Mapping[str, SeveritySpec] = {
    # --- noise -------------------------------------------------------------
    # NOTE: SNR *decreases* with severity. theta=1 is the 2 dB (worst) end.
    "complex_gaussian": SeveritySpec(
        (_P("snr", "dB", 40.0, 2.0),),
        notes="k-space AWGN; theta ramps SNR down, so the physical minimum is the "
        "most-degraded end.",
    ),
    "rician": SeveritySpec(
        (_P("sigma", "fraction of p99 signal", 0.0, 0.3),),
        notes="image-space Rician magnitude noise; sigma is scaled by the input's own "
        "p99 so the axis is scale-invariant (p99 not max, so one hot voxel cannot set "
        "the noise floor).",
    ),
    # --- sampling ----------------------------------------------------------
    "cartesian_undersamp": SeveritySpec(
        (_P("acceleration", "x", 1.0, 8.0),),
        notes="PE-line undersampling with an 8% fully-sampled centre.",
    ),
    "vd_undersamp": SeveritySpec(
        (_P("acceleration", "x", 1.0, 8.0),),
        notes="2-D Bernoulli variable density; the Bernoulli scale is calibrated so "
        "the REALISED rate equals this declared one (#246).",
    ),
    "vd_cartesian": SeveritySpec((_P("acceleration", "x", 1.0, 8.0),)),
    "radial_undersamp": SeveritySpec(
        (_P("acceleration", "x", 1.0, 8.0),),
        notes="golden-angle spokes on a Cartesian grid; the spoke count is chosen so "
        "the realised kept fraction is 1/R. R=1 is a fully-sampled radial "
        "acquisition, i.e. the identity (#246 — the axis previously ramped a raw "
        "spoke count 200 -> 16, and 200 spokes do not cover the grid).",
    ),
    # NOTE: retained fraction *decreases* with severity.
    "partial_fourier": SeveritySpec(
        (_P("retained_fraction", "fraction of ky", 1.0, 0.625),),
        notes="asymmetric ky truncation; theta=1 keeps the 5/8 minimum.",
    ),
    # --- motion ------------------------------------------------------------
    # Two knobs move together. This is a severity ramp, not an ablation -- but the
    # coupling is declared rather than hidden behind whichever one got reported.
    "rigid_motion": SeveritySpec(
        (
            _P("max_translation", "px", 0.0, 10.0),
            _P("max_rotation", "rad", 0.0, 0.1),
        ),
        notes="per-PE-line SE(2) pose; translation and rotation ramp together.",
    ),
    "pulsatile_motion": SeveritySpec(
        (_P("max_amplitude", "px", 0.0, 30.0),),
        notes="non-rigid deformation at 1.2 Hz, TR = 0.5 s.",
    ),
    "translation_drift": SeveritySpec(
        (_P("max_drift", "px", 0.0, 10.0),),
        notes="coherent linear bulk drift along readout across the acquisition: a "
        "single low-order k_y phase gradient -> directional smear, distinct from the "
        "broadband jitter of rigid_motion.",
    ),
    "sudden_motion": SeveritySpec(
        (_P("shift", "px", 0.0, 12.0),),
        notes="single abrupt bulk shift at the k-space centre (a mid-scan jerk); the "
        "step in k_y yields a discrete double-image ghost.",
    ),
    "random_motion": SeveritySpec(
        (_P("max_shift", "px", 0.0, 3.0),),
        notes="i.i.d. per-PE-line random translational jitter -> broadband ghosting; "
        "the pure-translation sibling of rigid_motion (no rotation).",
    ),
    # --- field -------------------------------------------------------------
    "nyquist_ghost": SeveritySpec(
        (_P("phase_offset", "rad", 0.0, math.pi),),
        notes="EPI even/odd phase offset; theta=1 is a pi offset (max N/2 ghost).",
    ),
    # NOTE: retained fraction *decreases* with severity.
    "gibbs": SeveritySpec(
        (_P("truncation_radius", "fraction of the max k-space radius", 1.0, 0.085),),
        notes="symmetric circular k-space truncation; less retained = stronger "
        "ringing. The radius is normalised by its own maximum (the grid CORNER), "
        "so 1.0 is a genuine identity — before #246 the window compared an "
        "un-normalised radius (max sqrt(2)) against the fraction and zeroed 22.8% "
        "of k-space at the nominal 'no truncation' setting.",
    ),
    "b0": SeveritySpec(
        (_P("strength", "normalised field-map scale", 0.0, 1.0),),
        notes="dimensionless scale on the polynomial field map (NOT Hz): drives "
        "both a phase ramp and intra-voxel-dephasing dropout.",
    ),
    "b0_geometric_distortion": SeveritySpec(
        (_P("b0_strength", "normalised field-map scale", 0.0, 1.0),),
        notes="voxel-shift distortion at a fixed 5e-5 s dwell time.",
    ),
    "bias": SeveritySpec(
        (_P("max_log_bias", "log multiplicative", 0.0, 1.2),),
        notes="smooth multiplicative bias field.",
    ),
    "t2star_blur": SeveritySpec((_P("sigma", "px", 0.0, 4.0),)),
    "slice_pve": SeveritySpec(
        (_P("sigma", "px", 0.0, 5.0),),
        notes="slice-thickness partial-volume proxy (max_kernel_px 20 / 4).",
    ),
    "resolution_snr": SeveritySpec(
        (
            _P("eff_fwhm", "px", 0.0, 2.5),
            _P("noise_sigma", "fraction of clean k-space RMS", 0.0, 0.0707),
        ),
        notes="coupled effective-resolution loss (k-space Gaussian apodization, no resampling) + noise on ONE theta; the resolution/SNR trade (SNR ~ voxel_volume). noise_sigma at theta=1 is the 20 dB SNR floor (10^(-20/20)/sqrt(2)).",
    ),
    "eddy": SeveritySpec(
        (
            _P("linear_phase_scale", "randn std, x pi rad", 0.0, 5.0),
            _P("quadratic_phase_scale", "randn std, x pi rad", 0.0, 2.5),
        ),
        notes="per-PE-line phase = pi*(alpha*u + beta*u^2), u in [-1, 1], "
        "alpha ~ N(0, scale), beta ~ N(0, 0.5*scale). The LINEAR term is a global PE "
        "translation (a reconstructor recovers it); the QUADRATIC (chirp) term is the "
        "genuine shear/ghost. The declared value is the randn STD of the pi-scaled "
        "coefficients, NOT a per-line rad increment (~0.12 rad/line at scale=5) -- the "
        "prior 'rad per PE line' label was ~40x off (advisor).",
    ),
    "spike": SeveritySpec(
        (_P("probability", "per k-space sample", 0.0, 0.05),),
        notes="RF spike / herringbone.",
    ),
    "chemical_shift": SeveritySpec((_P("shift", "px", 0.0, 14.0),)),
    "susceptibility": SeveritySpec(
        (
            _P("phase_scale", "dimensionless", 0.0, 1.0),
            _P("max_dropout", "fraction of signal", 0.0, 0.98),
        ),
        notes="air-tissue interface dipole field; phase and dropout ramp together.",
    ),
    "coil_miscal": SeveritySpec(
        (_P("relative_error", "fraction", 0.0, 1.0),),
        notes="coil-sensitivity miscalibration.",
    ),
    "magic_angle": SeveritySpec(
        (_P("t2_magic", "ms", 10.0, 30.0),),
        notes="collagen-fibre orientation T2 lengthening at TE = 20 ms. Fibre "
        "patches are drawn from the SUPPORT of the anatomy and cover ~30% of it "
        "(an MSK slice through a tendon-rich region); before #223 they were drawn "
        "uniformly over the FOV and at seed 0 both landed in signal-free air, "
        "making the axis an exact no-op.",
    ),
    "concomitant_phase": SeveritySpec(
        (_P("gradient_amplitude", "mT/m", 0.0, 30.0),),
        notes="Maxwell/concomitant-field phase at B0 = 0.3 T.",
    ),
    "iq_ghost": SeveritySpec(
        (
            _P("dc_offset", "fraction", 0.0, 0.20),
            _P("iq_imbalance", "fraction", 0.0, 0.35),
        ),
        notes="receiver DC offset and I/Q gain imbalance ramp together.",
    ),
    "stimulated_echo": SeveritySpec(
        (
            _P("amplitude", "fraction", 0.0, 0.40),
            _P("shift", "px", 0.0, 6.0),
        ),
        notes="spurious stimulated-echo pathway; amplitude and shift ramp together.",
    ),
    # --- geometric ---------------------------------------------------------
    "geometric_warp": SeveritySpec(
        (_P("per_mode_amplitude", "normalised-coord scale", 0.0, 8.0),),
        notes="smooth 5-mode elastic deformation. The declared value is the PER-MODE "
        "amplitude scale in normalised coordinates, NOT the realized peak pixel "
        "displacement (which is smaller and depends on the random mode superposition).",
    ),
    "gradient_nonlinearity": SeveritySpec(
        (_P("severity", "polynomial coeff scale", 0.0, 7e-5),),
        notes="~5% FOV peripheral distortion at theta=1, matching clinical Halbach "
        "systems (Koolstra et al. 2021).",
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Declared image components each degradation perturbs (issue #223).
#
# The sweep scores ``|D(x, theta)|``. An axis that moves ONLY the phase is
# therefore handed to the metrics as a constant, and every "sensitivity to axis
# X" number computed from it is the sensitivity of a metric to nothing.
# ``concomitant_phase`` (D23) is exactly that axis, and — verified empirically
# against all 27 — it is the ONLY one: at theta=1 its magnitude rel-L2 is 4.6e-08
# while its complex rel-L2 is 0.71.
#
# The flags below are measured, not asserted:
# ``tests/unit/infrastructure/physics/test_digital_twin_extensions.py::
# TestAffectsDeclaration`` re-derives every entry from the operator itself, so a
# flag cannot silently go stale when a degrader is retuned.
# ──────────────────────────────────────────────────────────────────────

_BOTH = frozenset({MAGNITUDE, PHASE})
_MAG = frozenset({MAGNITUDE})
_PHASE = frozenset({PHASE})

_AFFECTS: Mapping[str, frozenset[str]] = {
    # --- noise -------------------------------------------------------------
    "complex_gaussian": _BOTH,
    "rician": _MAG,  # returns a real magnitude image by construction
    "spike": _BOTH,
    # --- sampling ----------------------------------------------------------
    "cartesian_undersamp": _BOTH,
    "vd_undersamp": _BOTH,
    "vd_cartesian": _BOTH,
    "radial_undersamp": _BOTH,
    "partial_fourier": _BOTH,
    # --- motion ------------------------------------------------------------
    "rigid_motion": _BOTH,
    "pulsatile_motion": _BOTH,
    "translation_drift": _BOTH,
    "sudden_motion": _BOTH,
    "random_motion": _BOTH,
    # --- field -------------------------------------------------------------
    "nyquist_ghost": _BOTH,
    "gibbs": _BOTH,
    "b0": _BOTH,  # phase ramp AND intra-voxel-dephasing magnitude dropout
    "b0_geometric_distortion": _BOTH,
    "bias": _MAG,  # real positive multiplicative field
    "t2star_blur": _BOTH,
    "slice_pve": _BOTH,
    "resolution_snr": _BOTH,
    "eddy": _BOTH,
    "chemical_shift": _BOTH,
    "susceptibility": _BOTH,  # dipole phase AND dropout
    "coil_miscal": _MAG,  # real non-negative ratio map
    "magic_angle": _MAG,  # real positive T2 signal boost
    # THE pure-phase axis. Magnitude-scoring it yields a constant.
    "concomitant_phase": _PHASE,
    "iq_ghost": _BOTH,
    "stimulated_echo": _BOTH,
    # --- geometric ---------------------------------------------------------
    "geometric_warp": _BOTH,
    "gradient_nonlinearity": _BOTH,
}


# ──────────────────────────────────────────────────────────────────────
# Artifact origin per degradation — where in the imaging chain it arises.
#
# SCANNER: hardware / field / receiver-chain imperfections. ACQUISITION:
# sequence + sampling choices. PATIENT: subject-induced. This is what makes
# `list_degradations(origin=...)` a real query rather than a label, and it is
# indexed by `_reg` with `[name]` (not `.get`) so a new degradation cannot be
# added without classifying it — same declare-or-fail-at-import contract as
# _SEVERITY and _AFFECTS.
#
# NOTE: `slice_profile` (D27) is deliberately absent — it was removed as a
# duplicate of D12 (t2star_blur), issue #245.
# ──────────────────────────────────────────────────────────────────────

_ORIGIN: Mapping[str, ArtifactOrigin] = {
    # --- SCANNER: thermal/coil noise, B0/B1 field, gradients, receiver chain.
    "complex_gaussian": ArtifactOrigin.SCANNER,
    "rician": ArtifactOrigin.SCANNER,
    "b0": ArtifactOrigin.SCANNER,
    "b0_geometric_distortion": ArtifactOrigin.SCANNER,
    "bias": ArtifactOrigin.SCANNER,
    "eddy": ArtifactOrigin.SCANNER,
    "spike": ArtifactOrigin.SCANNER,
    "coil_miscal": ArtifactOrigin.SCANNER,
    "gradient_nonlinearity": ArtifactOrigin.SCANNER,
    "geometric_warp": ArtifactOrigin.SCANNER,
    "concomitant_phase": ArtifactOrigin.SCANNER,
    "iq_ghost": ArtifactOrigin.SCANNER,
    # --- ACQUISITION: sampling pattern, k-space truncation, sequence timing.
    "cartesian_undersamp": ArtifactOrigin.ACQUISITION,
    "vd_undersamp": ArtifactOrigin.ACQUISITION,
    "vd_cartesian": ArtifactOrigin.ACQUISITION,
    "radial_undersamp": ArtifactOrigin.ACQUISITION,
    "nyquist_ghost": ArtifactOrigin.ACQUISITION,
    "gibbs": ArtifactOrigin.ACQUISITION,
    "t2star_blur": ArtifactOrigin.ACQUISITION,
    "slice_pve": ArtifactOrigin.ACQUISITION,
    "resolution_snr": ArtifactOrigin.ACQUISITION,
    "chemical_shift": ArtifactOrigin.ACQUISITION,
    "magic_angle": ArtifactOrigin.ACQUISITION,
    "partial_fourier": ArtifactOrigin.ACQUISITION,
    "stimulated_echo": ArtifactOrigin.ACQUISITION,
    # --- PATIENT: subject motion and tissue-induced field perturbation.
    "rigid_motion": ArtifactOrigin.PATIENT,
    "pulsatile_motion": ArtifactOrigin.PATIENT,
    "translation_drift": ArtifactOrigin.PATIENT,
    "sudden_motion": ArtifactOrigin.PATIENT,
    "random_motion": ArtifactOrigin.PATIENT,
    "susceptibility": ArtifactOrigin.PATIENT,
}


def _reg(
    code: str,
    name: str,
    family: str,
    fn: Callable[..., torch.Tensor],
    description: str,
) -> DegradationSpec:
    """Build a DegradationSpec, resolving its declared severity range and components.

    A degradation with no ``_SEVERITY`` entry raises **at import**: you cannot add
    a degradation without writing down what its severity means physically. That is
    the whole point of the SSOT -- an undeclared severity axis is one that cannot
    be reported, and a sweep whose range lives only in a function body is not
    reproducible from the results.

    The same holds for ``_AFFECTS``: a degradation that does not declare whether it
    moves magnitude, phase, or both cannot be routed to the right metrics, and a
    pure-phase axis scored on ``|x|`` is scored as a constant (issue #223).
    """
    try:
        severity = _SEVERITY[name]
    except KeyError:
        raise KeyError(
            f"degradation {name!r} ({code}) has no declared SeveritySpec. Add one to "
            "_SEVERITY stating what theta=0 and theta=1 mean physically (see "
            "infrastructure/physics/severity.py)."
        ) from None
    try:
        affects = _AFFECTS[name]
    except KeyError:
        raise KeyError(
            f"degradation {name!r} ({code}) does not declare which image components "
            "it perturbs. Add an _AFFECTS entry -- a pure-phase axis scored on the "
            "magnitude image is scored as a constant (issue #223)."
        ) from None
    if not affects or not affects <= {MAGNITUDE, PHASE}:
        raise ValueError(
            f"_AFFECTS[{name!r}] = {set(affects)!r} must be a non-empty subset of "
            f"{{{MAGNITUDE!r}, {PHASE!r}}}."
        )
    try:
        origin = _ORIGIN[name]
    except KeyError:
        raise KeyError(
            f"degradation {name!r} ({code}) does not declare where in the imaging "
            "chain it arises. Add an _ORIGIN entry (scanner / acquisition / patient) "
            "-- an unclassified degradation makes list_degradations(origin=...) lie."
        ) from None
    return DegradationSpec(code, name, family, fn, description, severity, affects, origin)


DEGRADATION_REGISTRY: Mapping[str, DegradationSpec] = {
    "complex_gaussian": _reg(
        "D1",
        "complex_gaussian",
        "noise",
        degrade_complex_gaussian_noise,
        "Complex Gaussian k-space noise (→Rician)",
    ),
    "rician": _reg(
        "D2",
        "rician",
        "noise",
        degrade_rician_noise,
        "Image-space Rician magnitude noise",
    ),
    "cartesian_undersamp": _reg(
        "D3",
        "cartesian_undersamp",
        "sampling",
        degrade_cartesian_undersampling,
        "Cartesian PE-line undersampling",
    ),
    "vd_undersamp": _reg(
        "D4",
        "vd_undersamp",
        "sampling",
        degrade_variable_density_2d,
        "Variable-density 2-D Bernoulli",
    ),
    "vd_cartesian": _reg(
        "D4b",
        "vd_cartesian",
        "sampling",
        degrade_variable_density_cartesian,
        "Variable-density Cartesian PE-line",
    ),
    "radial_undersamp": _reg(
        "D5",
        "radial_undersamp",
        "sampling",
        degrade_radial_undersampling,
        "Radial spoke undersampling (Cartesian-grid)",
    ),
    "rigid_motion": _reg("D6", "rigid_motion", "motion", None, "Rigid motion (delegated)"),  # type: ignore[arg-type]
    "pulsatile_motion": _reg(
        "D7",
        "pulsatile_motion",
        "motion",
        degrade_pulsatile_motion,
        "Pulsatile non-rigid deformation",
    ),
    "translation_drift": _reg(
        "D29",
        "translation_drift",
        "motion",
        degrade_translation_drift,
        "Coherent translational drift",
    ),
    "sudden_motion": _reg(
        "D30",
        "sudden_motion",
        "motion",
        degrade_sudden_motion,
        "Single abrupt bulk shift",
    ),
    "random_motion": _reg(
        "D31",
        "random_motion",
        "motion",
        degrade_random_motion,
        "Random per-PE-line jitter",
    ),
    "nyquist_ghost": _reg("D8", "nyquist_ghost", "field", degrade_nyquist_ghost, "EPI N/2 ghost"),
    "gibbs": _reg("D9", "gibbs", "field", None, "Gibbs ringing (delegated)"),  # type: ignore[arg-type]
    "b0": _reg("D10", "b0", "field", None, "B0 inhomogeneity (delegated)"),  # type: ignore[arg-type]
    "b0_geometric_distortion": _reg(
        "D10b", "b0_geometric_distortion", "field", None, "B0 voxel-shift distortion (delegated)"
    ),  # type: ignore[arg-type]
    "bias": _reg("D11", "bias", "field", degrade_bias_field, "Multiplicative bias field"),
    "t2star_blur": _reg(
        "D12", "t2star_blur", "field", degrade_t2star_blur, "T2*-induced Gaussian blur"
    ),
    "slice_pve": _reg(
        "D13",
        "slice_pve",
        "field",
        degrade_slice_thickness_pve,
        "Slice-thickness PVE proxy",
    ),
    "eddy": _reg("D14", "eddy", "field", None, "Eddy current (delegated)"),  # type: ignore[arg-type]
    "spike": _reg("D15", "spike", "field", None, "Spike noise (delegated)"),  # type: ignore[arg-type]
    "chemical_shift": _reg("D16", "chemical_shift", "field", None, "Chemical shift (delegated)"),  # type: ignore[arg-type]
    "susceptibility": _reg(
        "D17",
        "susceptibility",
        "field",
        degrade_susceptibility,
        "Susceptibility (dipole + dropout)",
    ),
    "coil_miscal": _reg(
        "D18",
        "coil_miscal",
        "field",
        degrade_coil_sensitivity_miscal,
        "Coil-sensitivity miscalibration",
    ),
    "geometric_warp": _reg(
        "D20",
        "geometric_warp",
        "geometric",
        degrade_geometric_warp,
        "Low-frequency geometric warp",
    ),
    "gradient_nonlinearity": _reg(
        "D21", "gradient_nonlinearity", "geometric", None, "Gradient non-linearity (delegated)"
    ),  # type: ignore[arg-type]
    "magic_angle": _reg(
        "D22", "magic_angle", "field", None, "Magic-angle T2 modulation (delegated; MSK only)"
    ),  # type: ignore[arg-type]
    "concomitant_phase": _reg(
        "D23",
        "concomitant_phase",
        "field",
        degrade_concomitant_phase,
        "Maxwell concomitant phase (low-field)",
    ),
    "partial_fourier": _reg(
        "D24",
        "partial_fourier",
        "sampling",
        degrade_partial_fourier,
        "Partial-Fourier acquisition",
    ),
    "iq_ghost": _reg(
        "D25",
        "iq_ghost",
        "field",
        degrade_iq_ghost,
        "Quadrature-receiver DC + I/Q ghost",
    ),
    "stimulated_echo": _reg(
        "D26",
        "stimulated_echo",
        "field",
        degrade_stimulated_echo,
        "Stimulated-echo TSE ghost",
    ),
    "resolution_snr": _reg(
        "D28",
        "resolution_snr",
        "field",
        degrade_resolution_snr,
        "Coupled effective-resolution loss + noise (no resampling)",
    ),
    # D19 (ULF→HF domain gap) is data-paired: handled outside the simulator.
}


def _isolated_rng(device: torch.device, seed: int):
    """Context manager that seeds the global RNG without leaking state.

    The simulator primitives (``simulate_b0_inhomogeneity``,
    ``simulate_eddy_current``, ``simulate_spike_noise``,
    ``simulate_rigid_motion_kspace``) use the *global* PyTorch RNG via
    ``torch.randn``/``torch.rand``/``randn_like``. To keep the
    ``(theta, seed) -> identical output`` contract from polluting any
    other RNG consumer in the same process (C3 in the digital-twin
    review), we fork the RNG around each delegated call.

    Forking on CUDA additionally requires listing the device.
    """
    devices: list[torch.device] = []
    if device.type == "cuda":
        devices = [device]
    ctx = torch.random.fork_rng(devices=devices)
    return ctx


def _delegate_rigid_motion(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    g = _generator(x.device, seed)
    kspace = simulate_rigid_motion_kspace(
        fft2c(x),
        max_translation=float(kw.get("max_translation", float(theta) * 10.0)),
        max_rotation=float(kw.get("max_rotation", float(theta) * 0.1)),
        generator=g,
    )
    return ifft2c(kspace)


def _delegate_gibbs(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    # theta=1 retains 8.5% of the k-space RADIUS, producing heavy truncation
    # ringing. The floor moved 0.12 -> 0.085 when the window's radius was
    # normalised by its true maximum (#246): 0.12 / sqrt(2) = 0.085 is the same
    # physical window as before, so the theta=1 severity is unchanged while
    # theta=0 becomes a genuine identity instead of a 22.8%-of-k-space pedestal.
    trunc = float(kw.get("truncation_fraction", 1.0 - float(theta) * 0.915))
    return ifft2c(simulate_gibbs_ringing(fft2c(x), truncation_fraction=trunc))


def _delegate_b0(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        strength = float(kw.get("strength", float(theta)))
        out = ifft2c(simulate_b0_inhomogeneity(fft2c(x), x.shape[-2:], strength=strength))
    return out


def _delegate_eddy(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        # Strength controls per-PE-line phase accumulation (radians).
        # Bumped 0.5 -> 5.0 so theta=1 produces a visible shearing
        # ghost rather than the prior near-imperceptible phase tilt.
        strength = float(kw.get("strength", float(theta) * 5.0))
        out = ifft2c(simulate_eddy_current(fft2c(x), strength=strength))
    return out


def _delegate_spike(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        prob = float(kw.get("probability", float(theta) * 0.05))
        out = ifft2c(simulate_spike_noise(fft2c(x), probability=prob))
    return out


def _delegate_chemical_shift(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    # Bumped 6 -> 14 px so the fat-water ghost is unambiguous at
    # theta=1 (was barely > 1 voxel for many phantom features).
    # When b0_T+bw are supplied the function ignores shift_pixels
    # and computes the physical shift -- that path was the
    # "theta-invariant" bug; the gallery now omits those kwargs.
    shift = float(kw.get("shift_pixels", float(theta) * 14.0))
    # Default fat_fraction lowered 0.35 -> 0.15: 35 % was non-physical for brain
    # (parenchyma is < 1 % fat) and produced a whole-image fat ghost at 35 %
    # intensity; 0.15 keeps the ghost visible while being a more plausible
    # phantom-scale global fraction. Anatomy callers should still override via the
    # (wired-through) fat_fraction / fat_fraction_map kwarg for a localized fat mask.
    ff = float(kw.get("fat_fraction", 0.15))
    return ifft2c(
        simulate_chemical_shift(
            fft2c(x),
            x.shape[-2:],
            shift_pixels=shift,
            fat_fraction=ff,
            **{k: kw[k] for k in ("b0_T", "bw_per_pixel_hz", "fat_fraction_map") if k in kw},
        )
    )


# Mod6 (digital-twin review): compile the delegated callables once at
# module import. The previous `def _delegated(name): ...` form
# rebuilt closures on every call.
_DELEGATED_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "rigid_motion": _delegate_rigid_motion,
    "gibbs": _delegate_gibbs,
    "b0": _delegate_b0,
    "eddy": _delegate_eddy,
    "spike": _delegate_spike,
    "chemical_shift": _delegate_chemical_shift,
    "b0_geometric_distortion": _delegate_b0_geometric,
    "gradient_nonlinearity": _delegate_gradient_nl,
    "magic_angle": _delegate_magic_angle,
}


def _delegated(name: str):
    """Return the precompiled delegated callable for ``name``."""
    try:
        return _DELEGATED_FNS[name]
    except KeyError as exc:
        raise KeyError(name) from exc


def derive_axis_seed(master_seed: int, axis: str) -> int:
    """Deterministic per-axis seed, constant across the severity grid.

    Distinct axes get distinct seeds, so two axes never share one artefact
    realisation; but each axis's seed does not depend on theta, so theta stays
    the only free variable (the #244 property).

    The axis hash is ``zlib.crc32``, **not** the builtin ``hash``: the latter is
    salted by ``PYTHONHASHSEED`` and differs between interpreter runs, which
    would reintroduce cross-process non-determinism through the side door.

    .. note::
       ``scripts/sim2rank/degradation.py`` carries a byte-identical copy that
       predates this one. It should import from here instead; kept separate for
       now so this change cannot perturb a sweep mid-flight.
    """
    return (int(master_seed) * 1_000_003 + zlib.crc32(axis.encode("utf-8"))) % (2**31 - 1)


def apply_degradation(
    name: str,
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Single entry point: apply ``name`` degradation at severity ``theta``.

    Args:
        name:  Registry key from :data:`DEGRADATION_REGISTRY`.
        x:     Input image ``(B, C, H, W)`` (real or complex).
        theta: Severity ``in [0, 1]`` (clamped).
        seed:  Per-call deterministic seed.

    Returns:
        Degraded image, same shape as input. Complex-valued unless the
        specific degradation is intrinsically real (D2 / D12 with real
        input).
    """
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(
            f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY.keys())}"
        )
    spec = DEGRADATION_REGISTRY[name]
    theta = float(max(0.0, min(1.0, theta)))
    fn = spec.fn
    if fn is None:
        fn = _delegated(name)
    return fn(x, theta, seed=seed, **kwargs)


def list_degradations(origin: ArtifactOrigin | None = None) -> list[str]:
    """The canonical ordered degradation names, optionally filtered by origin.

    Args:
        origin: if given, keep only degradations arising at that point in the
            imaging chain (scanner / acquisition / patient). ``None`` = all.
    """
    if origin is None:
        return list(DEGRADATION_REGISTRY.keys())
    return [n for n, spec in DEGRADATION_REGISTRY.items() if spec.origin is origin]


def get_degradation_origin(name: str) -> ArtifactOrigin:
    """Where in the imaging chain ``name`` arises. Raises on an unknown name."""
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY)}")
    return DEGRADATION_REGISTRY[name].origin


def affected_components(name: str) -> frozenset[str]:
    """Which image components ``name`` perturbs — a subset of {magnitude, phase}."""
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY)}")
    return DEGRADATION_REGISTRY[name].affects


def is_phase_only(name: str) -> bool:
    """True when ``name`` leaves ``|x|`` untouched (issue #223).

    A caller that scores the sweep on ``.abs()`` MUST route a phase-only axis to
    the phase metrics against the phase-bearing complex tensor instead — scoring
    it on the magnitude image scores a constant.
    """
    return affected_components(name) == frozenset({PHASE})


def phase_only_degradations() -> list[str]:
    """The registered axes that move phase and nothing else."""
    return [n for n, s in DEGRADATION_REGISTRY.items() if s.phase_only]


# ──────────────────────────────────────────────────────────────────────
# Declared non-orthogonal axis pairs (issue #245)
#
# The taxonomy is *presented* as a set of orthogonal degradation axes, and the
# per-family aggregations (Borda sum, CDRS variance penalty) assume as much. The
# pairs below are the ones that measurably are NOT orthogonal — |cos| of their
# theta=1 residuals exceeds 0.9 — and cannot be made so without fabricating
# physics the 2-D magnitude data cannot carry. They are declared here rather than
# hidden in a test's skip list, so a reader of the results can see the double
# counting, and so a NEW collinear pair fails the orthogonality test loudly.
#
# The one pair that WAS a genuine duplicate (t2star_blur / slice_profile, |cos| =
# 0.998 — the same isotropic Gaussian blur registered twice) is not on this list:
# it was deleted. See the D27 note above.
# ──────────────────────────────────────────────────────────────────────

# The LOW-PASS FAMILY. Four axes with genuinely different physics and genuinely
# different operators — a Gaussian PSF (D12), an anisotropic through-plane PVE
# proxy (D13), a circular k-space window (D9), and a centre-peaked undersampling
# mask that at high R keeps almost only the k-space centre (D4) — but whose
# theta=1 residuals are all, to leading order, "the high-frequency content,
# removed". They cannot be made orthogonal; they can only be declared. Their
# pairwise |cos| runs 0.79-0.95 and is anatomy- and grid-size-dependent, so the
# pairs that cross the 0.9 guard on one image sit just under it on another.
_LOW_PASS_FAMILY = (
    "Both are low-pass operators. D9 (gibbs) is a circular k-space window, D12 "
    "(t2star_blur) an isotropic Gaussian PSF, D13 (slice_pve) an anisotropic "
    "readout-axis blur, and D4 (vd_undersamp) at high R keeps little but the "
    "k-space centre. Different physics, different operators — but every one of "
    "their residuals is dominated by the same thing: the missing high frequencies. "
    "This is a real limit of the taxonomy, not a bug to be tuned away, and it is "
    "declared so the per-family aggregations can be read with it in mind."
)

KNOWN_COLLINEAR_PAIRS: Mapping[tuple[str, str], str] = {
    ("slice_pve", "t2star_blur"): _LOW_PASS_FAMILY,
    ("gibbs", "t2star_blur"): _LOW_PASS_FAMILY,
    ("gibbs", "vd_undersamp"): _LOW_PASS_FAMILY,
    ("iq_ghost", "stimulated_echo"): (
        "Both add a low-amplitude replica of the object to itself — D25 a "
        "conjugate mirror, D26 a PE-shifted copy. On near-symmetric anatomy, and "
        "with D26's shift capped at 6 px, both residuals reduce towards a scalar "
        "multiple of the object. Raising D26's displacement to a clinically "
        "realistic fraction of the FOV would separate them; that is a severity "
        "recalibration, tracked separately."
    ),
    ("cartesian_undersamp", "vd_cartesian"): (
        "Both are incoherent Cartesian PE-line undersampling with a dense k-space "
        "centre: D4b draws the outer lines with a variable-density profile "
        "(1+|ky|)^-alpha, D3 draws them uniformly at random, but at matched R the "
        "aliasing residual of each is the same broadband incoherent-Cartesian "
        "pattern (|cos| ~0.99). Steepening D4b's density toward a low-pass profile "
        "would separate them but re-introduce the resolution-loss facade this axis "
        "was repaired (advisor) to remove, so the near-duplication is declared "
        "rather than manufactured away."
    ),
}


def is_known_collinear(a: str, b: str) -> bool:
    """True when ``(a, b)`` is a declared non-orthogonal pair (order-insensitive)."""
    return tuple(sorted((a, b))) in KNOWN_COLLINEAR_PAIRS


# ──────────────────────────────────────────────────────────────────────
# Declared non-identity clean anchors (issue #246)
#
# ``D(x, theta=0)`` must equal ``x``. Every statistic in the sweep is anchored on
# that clean baseline, so a degrader that quietly corrupts its own theta=0 output
# biases everything downstream. ``gibbs`` (a k-space window that zeroed 22.8% of
# the samples at its nominal "no truncation" setting) and ``radial_undersamp``
# (200 spokes that do not cover the grid) both did exactly that, with pedestals
# of 0.055 and 0.069 — larger than the ENTIRE theta=1 distortion of
# ``partial_fourier``. Both are fixed; the invariant is now pinned for every axis
# by ``TestCleanAnchor``.
#
# One axis has a legitimately non-identity theta=0, and it is DECLARED here
# rather than silently tolerated by the test: ``complex_gaussian``'s severity IS
# the SNR, and its SeveritySpec declares theta=0 to be **40 dB**, not infinity —
# a clean clinical scanner, not a noiseless one. The residual is bounded and
# intentional, so the bound is written down.
# ──────────────────────────────────────────────────────────────────────

DECLARED_THETA_ZERO_PEDESTAL: Mapping[str, float] = {
    # name -> max tolerated rel-L2 of D(x, 0) against x
    "complex_gaussian": 0.02,  # 40 dB SNR at theta=0, per its SeveritySpec
}


def is_identity_at_theta_zero(name: str) -> bool:
    """True when ``D(x, 0) == x`` is required of ``name`` (all but the declared)."""
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY)}")
    return name not in DECLARED_THETA_ZERO_PEDESTAL


__all__ = [
    "DECLARED_THETA_ZERO_PEDESTAL",
    "DEGRADATION_REGISTRY",
    "KNOWN_COLLINEAR_PAIRS",
    "MAGNITUDE",
    "PHASE",
    "DegradationSpec",
    "affected_components",
    "apply_degradation",
    "degrade_bias_field",
    "degrade_cartesian_undersampling",
    "degrade_coil_sensitivity_miscal",
    "degrade_complex_gaussian_noise",
    "degrade_concomitant_phase",
    "degrade_geometric_warp",
    "degrade_iq_ghost",
    "degrade_nyquist_ghost",
    "degrade_partial_fourier",
    "degrade_pulsatile_motion",
    "degrade_radial_undersampling",
    "degrade_random_motion",
    "degrade_resolution_snr",
    "degrade_rician_noise",
    "degrade_slice_thickness_pve",
    "degrade_stimulated_echo",
    "degrade_sudden_motion",
    "degrade_susceptibility",
    "degrade_t2star_blur",
    "degrade_translation_drift",
    "degrade_variable_density_2d",
    "degrade_variable_density_cartesian",
    "degrade_variable_density_undersampling",  # alias
    "get_degradation_origin",
    "is_identity_at_theta_zero",
    "is_known_collinear",
    "is_phase_only",
    "list_degradations",
    "phase_only_degradations",
]
