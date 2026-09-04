r"""Matrix-free generators of the MRI degradation catalog (Proposal 1).

Each registered degradation *mode* is exposed here as a single Lie-algebra
generator :math:`L_k` acting on a complex image tensor ``[B, C, H, W]``,
together with a *verified Hermitian adjoint* :math:`L_k^H`. The composite
operator is then :math:`\exp(\Omega(s))` with
:math:`\Omega(s)=\sum_k s_k L_k + \cdots` assembled in
:mod:`spectramr.infrastructure.physics.magnus_exponential`; the per-mode
severity :math:`s_k` scales a *fixed* generator pattern, so identification
reduces to estimating the severities.

Generator catalog
-----------------
Two kinds of mode are distinguished:

* ``deterministic`` — enters the exponential :math:`\exp(\Omega)`:

  - ``motion_rigid``           : advection generator :math:`-a\cdot\nabla`
    (skew-Hermitian; central circular differences).
  - ``bias_field_multiplicative`` : :math:`\operatorname{diag}(b)` with a
    smooth real field :math:`b` (Hermitian).
  - ``b0_phase``              : :math:`i\,\operatorname{diag}(\phi)` with a
    linear off-resonance ramp :math:`\phi` (anti-Hermitian).
  - ``gibbs_truncation``      : :math:`P_\Gamma - I` with the centred
    low-pass k-space projection :math:`P_\Gamma=\mathcal F^{-1}M\mathcal F`
    (Hermitian).

* ``noise`` — enters the structured covariance :math:`\Sigma` of the noise
  operator :math:`\mathcal N_\Sigma`, **not** the exponential:

  - ``rician_noise`` / ``gaussian_noise``.

The deterministic adapters return a :class:`LinearOperator`; the noise
adapters return ``None`` from :meth:`GeneratorAdapter.build` (they carry
no generator) and are consumed by
:mod:`spectramr.infrastructure.physics.structured_covariance`.

This module is the single source of truth for the operator basis; the
Tier-1 audit check ``operator_basis_registered`` resolves every YAML
``mode_dictionary`` entry against :func:`get_generator_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fft_ops import fft2c, ifft2c
from .magnus_exponential import LinearOperator

__all__ = [
    "GENERATOR_ADAPTERS",
    "GeneratorAdapter",
    "advection_generator",
    "build_generators",
    "diag_generator",
    "get_generator_adapter",
    "list_generator_modes",
    "register_generator_adapter",
    "truncation_generator",
]


# ─────────────────────────────────────────────────────────────────────────
# Core matrix-free generator constructors
# ─────────────────────────────────────────────────────────────────────────
def diag_generator(diag: torch.Tensor, *, name: str = "diag") -> LinearOperator:
    r"""Diagonal generator :math:`v \mapsto \operatorname{diag}(d)\,v`.

    ``diag`` is broadcast over the operand. The adjoint multiplies by
    :math:`\bar d`, so a real ``diag`` is self-adjoint and a purely
    imaginary ``diag`` (e.g. :math:`i\phi`) is anti-Hermitian.
    """
    diag_c = diag.conj()
    return LinearOperator(
        lambda v: diag * v,
        lambda v: diag_c * v,
        name=name,
    )


def _central_diff(v: torch.Tensor, dim: int) -> torch.Tensor:
    """Circular central difference along ``dim`` — a skew-symmetric operator."""
    return 0.5 * (torch.roll(v, -1, dims=dim) - torch.roll(v, 1, dims=dim))


def advection_generator(a_h: float, a_w: float, *, name: str = "advection") -> LinearOperator:
    r"""Advection (rigid-motion) generator :math:`-(a_h\partial_h + a_w\partial_w)`.

    Implemented with circular central differences, whose matrix is exactly
    skew-symmetric; the continuous-direction generator is therefore
    skew-Hermitian and its adjoint is :math:`+(a_h\partial_h+a_w\partial_w)`.
    ``exp(s·L)`` is a (sub-pixel) circular shift — the generator of rigid
    in-plane translation.
    """

    def _fwd(v: torch.Tensor) -> torch.Tensor:
        return -(a_h * _central_diff(v, dim=-2) + a_w * _central_diff(v, dim=-1))

    def _adj(v: torch.Tensor) -> torch.Tensor:
        # D^H = -D ⇒ L^H = -(-(a·D))^... = +(a_h D_h + a_w D_w)
        return a_h * _central_diff(v, dim=-2) + a_w * _central_diff(v, dim=-1)

    return LinearOperator(_fwd, _adj, name=name)


def truncation_generator(mask: torch.Tensor, *, name: str = "gibbs") -> LinearOperator:
    r"""Soft-truncation / Gibbs generator :math:`P_\Gamma - I`.

    :math:`P_\Gamma = \mathcal F^{-1}\operatorname{diag}(m)\mathcal F` is the
    centred low-pass k-space projection (``mask`` is real, in ``[0, 1]``,
    broadcast over the operand). Because the centred FFT is unitary and the
    k-space mask is real-diagonal, :math:`P_\Gamma` is Hermitian, hence
    :math:`P_\Gamma - I` is self-adjoint.
    """

    def _proj(v: torch.Tensor) -> torch.Tensor:
        return ifft2c(mask * fft2c(v))

    def _fwd(v: torch.Tensor) -> torch.Tensor:
        return _proj(v) - v

    # Self-adjoint ⇒ adjoint == forward.
    return LinearOperator(_fwd, _fwd, name=name)


# ─────────────────────────────────────────────────────────────────────────
# Fixed canonical patterns (severity scales these)
# ─────────────────────────────────────────────────────────────────────────
def _grid(h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised coordinate grids in ``[-1, 1]`` of shape ``[H, W]``."""
    real_dtype = torch.float64 if dtype in (torch.complex128,) else torch.float32
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=real_dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=real_dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gy, gx


def _bias_field(h: int, w: int, device, dtype) -> torch.Tensor:
    """Smooth, zero-mean quadratic bias field ``b`` (real), shape ``[1, 1, H, W]``."""
    gy, gx = _grid(h, w, device, dtype)
    b = gx * gx + gy * gy
    b = b - b.mean()
    return b.to(dtype).reshape(1, 1, h, w)


def _b0_ramp(h: int, w: int, device, dtype) -> torch.Tensor:
    """Linear off-resonance phase ramp ``i·phi``, shape ``[1, 1, H, W]`` (imaginary)."""
    gy, gx = _grid(h, w, device, dtype)
    phi = 0.5 * (gx + gy)
    diag = (1j * phi.to(dtype)).reshape(1, 1, h, w)
    return diag


def _gibbs_mask(h: int, w: int, device, dtype, keep_fraction: float = 0.5) -> torch.Tensor:
    """Centred low-pass k-space mask keeping the central ``keep_fraction``."""
    real_dtype = torch.float64 if dtype in (torch.complex128,) else torch.float32
    mh = max(1, round(h * keep_fraction))
    mw = max(1, round(w * keep_fraction))
    mask = torch.zeros(h, w, device=device, dtype=real_dtype)
    h0 = (h - mh) // 2
    w0 = (w - mw) // 2
    mask[h0 : h0 + mh, w0 : w0 + mw] = 1.0
    return mask.reshape(1, 1, h, w)


# ─────────────────────────────────────────────────────────────────────────
# Adapter registry
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GeneratorAdapter:
    """Registry entry mapping a YAML mode name to a matrix-free generator.

    Attributes:
        name: The registered mode name (matches ``mode_dictionary`` YAML).
        kind: ``"deterministic"`` (enters :math:`\\exp(\\Omega)`) or
            ``"noise"`` (enters the covariance :math:`\\Sigma`).
        builder: Callable ``(H, W, device, dtype) -> LinearOperator`` for
            deterministic modes; ``None`` for noise modes.
        description: Human-readable summary for audit / docs.
    """

    name: str
    kind: str
    builder: object  # Callable[[int, int, Any, torch.dtype], LinearOperator] | None
    description: str = ""

    def build(self, h: int, w: int, device, dtype: torch.dtype) -> LinearOperator | None:
        """Materialise the generator for an ``H x W`` image, or ``None`` for noise modes."""
        if self.kind == "noise" or self.builder is None:
            return None
        return self.builder(h, w, device, dtype)  # type: ignore[operator]


GENERATOR_ADAPTERS: dict[str, GeneratorAdapter] = {}


def register_generator_adapter(adapter: GeneratorAdapter) -> None:
    """Register (or refuse to silently overwrite) a generator adapter."""
    existing = GENERATOR_ADAPTERS.get(adapter.name)
    if existing is not None and existing is not adapter:
        raise ValueError(
            f"Generator mode {adapter.name!r} already registered "
            f"(kind={existing.kind!r}); refusing to overwrite. Rename one."
        )
    GENERATOR_ADAPTERS[adapter.name] = adapter


def get_generator_adapter(name: str) -> GeneratorAdapter:
    """Resolve a mode name to its adapter, raising on unknown modes (no fallback)."""
    if name not in GENERATOR_ADAPTERS:
        raise KeyError(
            f"Unknown degradation mode {name!r}. Registered modes: "
            f"{sorted(GENERATOR_ADAPTERS)}. Register an adapter in "
            "degradation_generators.py or remove the mode from mode_dictionary."
        )
    return GENERATOR_ADAPTERS[name]


def list_generator_modes() -> list[str]:
    """All registered mode names."""
    return sorted(GENERATOR_ADAPTERS)


def build_generators(
    mode_dictionary: list[str] | tuple[str, ...],
    h: int,
    w: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> tuple[list[LinearOperator], list[str], list[str]]:
    r"""Build the deterministic generator basis for a ``mode_dictionary``.

    Args:
        mode_dictionary: Ordered mode names (the operator basis).
        h, w: Image height / width.
        device, dtype: Device and complex dtype of the generators.

    Returns:
        ``(generators, deterministic_modes, noise_modes)`` where
        ``generators[i]`` is the :class:`LinearOperator` for
        ``deterministic_modes[i]`` (order preserved from
        ``mode_dictionary``) and ``noise_modes`` lists the noise-kind
        entries (consumed by the structured covariance).
    """
    generators: list[LinearOperator] = []
    det_modes: list[str] = []
    noise_modes: list[str] = []
    for name in mode_dictionary:
        adapter = get_generator_adapter(name)
        if adapter.kind == "noise":
            noise_modes.append(name)
            continue
        op = adapter.build(h, w, device, dtype)
        assert op is not None  # deterministic adapters always build
        generators.append(op)
        det_modes.append(name)
    return generators, det_modes, noise_modes


# ── Built-in catalog registration ────────────────────────────────────────
register_generator_adapter(
    GeneratorAdapter(
        name="motion_rigid",
        kind="deterministic",
        builder=lambda h, w, device, dtype: advection_generator(1.0, 0.0, name="motion_rigid"),
        description="Rigid in-plane translation (advection, skew-Hermitian).",
    )
)
register_generator_adapter(
    GeneratorAdapter(
        name="bias_field_multiplicative",
        kind="deterministic",
        builder=lambda h, w, device, dtype: diag_generator(
            _bias_field(h, w, device, dtype), name="bias_field_multiplicative"
        ),
        description="Smooth multiplicative bias field (Hermitian diagonal).",
    )
)
register_generator_adapter(
    GeneratorAdapter(
        name="b0_phase",
        kind="deterministic",
        builder=lambda h, w, device, dtype: diag_generator(
            _b0_ramp(h, w, device, dtype), name="b0_phase"
        ),
        description="Off-resonance phase ramp i·diag(phi) (anti-Hermitian).",
    )
)
register_generator_adapter(
    GeneratorAdapter(
        name="gibbs_truncation",
        kind="deterministic",
        builder=lambda h, w, device, dtype: truncation_generator(
            _gibbs_mask(h, w, device, dtype), name="gibbs_truncation"
        ),
        description="Centred k-space low-pass truncation P_Γ - I (Hermitian).",
    )
)
register_generator_adapter(
    GeneratorAdapter(
        name="rician_noise",
        kind="noise",
        builder=None,
        description="Rician acquisition noise — realised through the covariance Σ.",
    )
)
register_generator_adapter(
    GeneratorAdapter(
        name="gaussian_noise",
        kind="noise",
        builder=None,
        description="Additive Gaussian noise — realised through the covariance Σ.",
    )
)
