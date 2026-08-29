r"""Equivariance-defect non-conformity score (B-1 / Idea 1).

A *reference-free and marker-free* non-conformity score for split-conformal
prediction. Where the existing VF-residual score
(:mod:`mriforge.infrastructure.calibration`) measures
:math:`\lVert P_M\hat x - m\rVert` against a known marker subspace, this score
needs neither a clean reference nor an embedded marker — it measures how far a
frozen reconstructor :math:`G` deviates from being rotation-equivariant on the
acquired measurement :math:`y`.

Definition
----------
For a frozen reconstructor :math:`G` and a rotation group :math:`\{R_\theta\}`
(planar ``SO2``, or ``SE2`` adding a small integer pixel shift), the
**equivariance-defect score** is the root-mean-square commutator over
:math:`K` sampled group elements

.. math::

    s(y) \;=\; \Bigl(\tfrac{1}{K}\sum_{k=1}^{K}
        \bigl\lVert G(R_{\theta_k}\, y) - R_{\theta_k}\, G(y) \bigr\rVert_2^2
        \Bigr)^{1/2}.

Rationale
---------
A perfectly equivariant reconstructor commutes with the group action, so
:math:`s(y)=0`. Real networks break equivariance most where they hallucinate
or are unsure, so :math:`s` is a data-driven uncertainty proxy that is
**exchangeable across calibration and test points** (the rotations are drawn
from a fixed, data-independent distribution with a per-sample deterministic
seed). Feeding the empirical distribution of :math:`s` over a calibration split
to the standard split-conformal quantile yields a distribution-free
:math:`1-\alpha` coverage guarantee.

Reference: Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random
World* (split conformal); group-equivariance defect as a confidence signal is
the B-1 contribution of the VF campaign
(``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from mriforge.core.metrics.registry import register_metric

SUPPORTED_GROUPS = ("SO2", "SE2")


def _to_real_image(x: torch.Tensor) -> torch.Tensor:
    """Coerce a complex / real-stacked tensor to a real magnitude image.

    Rotations and the L2 commutator are evaluated on a real-valued image so the
    grid-sample interpolation path is well-defined for both complex64 (M4Raw)
    and interleaved real/imag inputs. We keep the channel axis so multi-coil
    tensors round-trip without collapsing coils.
    """
    if x.is_complex():
        return x.abs()
    return x


def _rotate_image(
    x: torch.Tensor,
    angle_rad: float,
    shift_px: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    r"""Apply :math:`R_\theta` (planar rotation + optional integer shift).

    Uses an affine ``grid_sample`` so the rotation is differentiable and works
    on ``[B, C, H, W]``. The optional integer pixel shift (the translation part
    of ``SE2``) is applied with :func:`torch.roll` so it is exactly invertible
    and introduces no interpolation error.

    Args:
        x: Real-valued ``[B, C, H, W]`` tensor.
        angle_rad: Rotation angle in radians (counter-clockwise).
        shift_px: ``(dy, dx)`` integer pixel translation applied after rotation.

    Returns:
        The rotated tensor, same shape and dtype as ``x``.
    """
    if x.ndim != 4:
        raise ValueError(f"_rotate_image expects [B, C, H, W]; got shape {tuple(x.shape)}.")
    b = x.shape[0]
    cos, sin = torch.cos, torch.sin
    a = torch.tensor(angle_rad, dtype=x.dtype, device=x.device)
    # 2x3 affine for grid_sample (rotation about image center, no scaling).
    theta = torch.zeros(b, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = cos(a)
    theta[:, 0, 1] = -sin(a)
    theta[:, 1, 0] = sin(a)
    theta[:, 1, 1] = cos(a)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    rotated = torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    dy, dx = int(shift_px[0]), int(shift_px[1])
    if dy != 0 or dx != 0:
        rotated = torch.roll(rotated, shifts=(dy, dx), dims=(-2, -1))
    return rotated


def _sample_group_elements(
    num_rotations: int,
    group: str,
    seed: int,
    max_shift_px: int = 4,
) -> list[tuple[float, tuple[int, int]]]:
    r"""Deterministically sample ``num_rotations`` group elements.

    The sampler is seeded **per-call** so the same ``seed`` yields the same set
    of rotations — this is what makes the score exchangeable across calibration
    and test points (the rotation distribution is identical and data-
    independent). For ``SO2`` the shift is always zero; for ``SE2`` an integer
    pixel shift in ``[-max_shift_px, max_shift_px]`` is drawn per element.

    Raises:
        ValueError: if ``group`` is not one of :data:`SUPPORTED_GROUPS`
            (no silent fallback, per CLAUDE.md #9).
    """
    if group not in SUPPORTED_GROUPS:
        raise ValueError(
            f"Unsupported equivariance group '{group}'. Expected one of {SUPPORTED_GROUPS}."
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    # Uniform angles on [0, 2*pi). Drawn (not evenly spaced) so the empirical
    # score is a genuine Monte-Carlo estimate whose variance shrinks with K.
    angles = torch.rand(num_rotations, generator=gen) * (2.0 * torch.pi)
    elements: list[tuple[float, tuple[int, int]]] = []
    for k in range(num_rotations):
        if group == "SE2":
            dy = int(torch.randint(-max_shift_px, max_shift_px + 1, (1,), generator=gen).item())
            dx = int(torch.randint(-max_shift_px, max_shift_px + 1, (1,), generator=gen).item())
            shift = (dy, dx)
        else:
            shift = (0, 0)
        elements.append((float(angles[k].item()), shift))
    return elements


def equivariance_defect_score(
    recon_fn: Callable[..., torch.Tensor],
    y: torch.Tensor,
    *,
    num_rotations: int = 8,
    group: str = "SO2",
    seed: int = 0,
    coil_sensitivities: torch.Tensor | None = None,
    max_shift_px: int = 4,
) -> torch.Tensor:
    r"""Monte-Carlo equivariance-defect score :math:`s(y)`.

    Computes the root-mean-square equivariance commutator per batch element:
    ``sqrt( mean_k mean_pixels( (G(R_k y) - R_k G(y))^2 ) )``. The inner average
    is over elements (a per-pixel mean, not a squared L2 sum), so the score is
    invariant to image resolution.

    Args:
        recon_fn: Frozen reconstructor. Called as ``recon_fn(y)`` or, when
            ``coil_sensitivities`` is given, ``recon_fn(y, coil_sensitivities)``.
            Must return an image-space tensor broadcastable to ``[B, C, H, W]``.
        y: Measurement tensor ``[B, C, H, W]`` (complex64 k-space/image or
            real-stacked). Rotations act on its real magnitude.
        num_rotations: ``K`` group elements in the MC estimate (``>= 1``).
        group: ``"SO2"`` or ``"SE2"``.
        seed: Per-sample seed driving the (deterministic) rotation sampler.
        coil_sensitivities: Optional CSM forwarded to ``recon_fn``.
        max_shift_px: Max integer pixel shift for the ``SE2`` translation part.

    Returns:
        A 1-D tensor of length ``B`` with the per-sample score (non-negative).

    Raises:
        ValueError: bad ``num_rotations``, unknown ``group``, or non-4D ``y``.
    """
    if num_rotations < 1:
        raise ValueError(f"num_rotations must be >= 1; got {num_rotations}.")
    if y.ndim != 4:
        raise ValueError(f"y must be [B, C, H, W]; got shape {tuple(y.shape)}.")

    elements = _sample_group_elements(num_rotations, group, seed, max_shift_px)

    def _call(arg: torch.Tensor) -> torch.Tensor:
        out = recon_fn(arg, coil_sensitivities) if coil_sensitivities is not None else recon_fn(arg)
        return _to_real_image(out)

    y_real = _to_real_image(y)
    g_y = _call(y)  # G(y) in image space (real magnitude)

    b = y_real.shape[0]
    sq_acc = torch.zeros(b, dtype=g_y.dtype, device=g_y.device)
    for angle, shift in elements:
        # G(R_theta y): rotate the *measurement* then reconstruct.
        ry = _rotate_image(y_real, angle, shift)
        g_ry = _call(ry)
        # R_theta G(y): rotate the reconstruction of the original.
        r_gy = _rotate_image(g_y, angle, shift)
        diff = (g_ry - r_gy).reshape(b, -1)
        sq_acc = sq_acc + diff.pow(2).mean(dim=1)
    mean_sq = sq_acc / float(num_rotations)
    return torch.sqrt(mean_sq.clamp_min(0.0))


def conformal_quantile(
    scores: torch.Tensor | Sequence[float],
    alpha: float,
) -> float:
    r"""Split-conformal quantile :math:`q_\alpha` from calibration scores.

    Returns the finite-sample-corrected empirical quantile

    .. math::

        q_\alpha = \text{Quantile}\Bigl(\{s_i\},\;
            \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\Bigr),

    which guarantees marginal coverage :math:`\ge 1-\alpha` under
    exchangeability (Vovk et al., 2005). When the corrected level exceeds 1
    (tiny ``n``) the max score is returned (conservative, infinite interval).

    Args:
        scores: Calibration non-conformity scores (1-D).
        alpha: Target miscoverage rate in ``(0, 1)``.

    Returns:
        The conformal threshold as a Python float.

    Raises:
        ValueError: empty ``scores`` or ``alpha`` outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    n = s.numel()
    if n == 0:
        raise ValueError("conformal_quantile requires at least one score.")
    level = (torch.ceil(torch.tensor((n + 1) * (1.0 - alpha))) / n).item()
    if level >= 1.0:
        return float(s.max().item())
    # torch.quantile uses linear interpolation; clamp level into [0, 1].
    q = torch.quantile(s, max(0.0, min(1.0, level)))
    return float(q.item())


@register_metric(name="equivariance_defect", aliases=["equivariance_defect_score"])
class EquivarianceDefectMetric:
    r"""Registered metric wrapping :func:`equivariance_defect_score`.

    Dual-mode, to fit both the generic ``IMetric`` contract and the
    score-function role described in the B-1 spec:

    * **As an :class:`~mriforge.core.metrics.registry.IMetric`** —
      ``metric(prediction, target)`` treats ``prediction`` as the measurement
      ``y`` and uses a frozen reconstructor supplied at construction
      (defaults to identity, for which the score is exactly 0). ``target`` is
      accepted for interface symmetry but unused (the score is reference-free).
    * **As an explicit scorer** — :meth:`score` / :meth:`forward` accept an
      arbitrary ``recon_fn`` and return the per-sample score tensor.

    Lower is better (a smaller defect = more equivariant = more confident).
    """

    def __init__(
        self,
        num_rotations: int = 8,
        group: str = "SO2",
        seed: int = 0,
        recon_fn: Callable[..., torch.Tensor] | None = None,
        max_shift_px: int = 4,
    ) -> None:
        if group not in SUPPORTED_GROUPS:
            raise ValueError(
                f"Unsupported equivariance group '{group}'. Expected one of {SUPPORTED_GROUPS}."
            )
        self.num_rotations = int(num_rotations)
        self.group = group
        self.seed = int(seed)
        self.max_shift_px = int(max_shift_px)
        # Identity reconstructor is the natural default: it is perfectly
        # equivariant, so the metric reports 0 unless a real G is supplied.
        self.recon_fn = recon_fn if recon_fn is not None else (lambda t: t)

    def score(
        self,
        recon_fn: Callable[..., torch.Tensor],
        y: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sample equivariance-defect score for an explicit ``recon_fn``."""
        return equivariance_defect_score(
            recon_fn,
            y,
            num_rotations=self.num_rotations,
            group=self.group,
            seed=self.seed,
            coil_sensitivities=coil_sensitivities,
            max_shift_px=self.max_shift_px,
        )

    # ``forward`` mirrors the nn.Module-style entry point named in the spec.
    def forward(
        self,
        recon_fn: Callable[..., torch.Tensor],
        y_kspace: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias of :meth:`score` (spec-named entry point)."""
        return self.score(recon_fn, y_kspace, coil_sensitivities)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """IMetric entry point: mean defect of ``self.recon_fn`` on ``prediction``."""
        recon_fn = kwargs.get("recon_fn", self.recon_fn)
        csm = kwargs.get("coil_sensitivities")
        per_sample = equivariance_defect_score(
            recon_fn,  # type: ignore[arg-type]
            prediction,
            num_rotations=self.num_rotations,
            group=self.group,
            seed=self.seed,
            coil_sensitivities=csm,  # type: ignore[arg-type]
            max_shift_px=self.max_shift_px,
        )
        return float(per_sample.mean().item())

    @property
    def name(self) -> str:
        return "equivariance_defect"

    @property
    def higher_is_better(self) -> bool:
        return False


__all__ = [
    "SUPPORTED_GROUPS",
    "EquivarianceDefectMetric",
    "conformal_quantile",
    "equivariance_defect_score",
]
