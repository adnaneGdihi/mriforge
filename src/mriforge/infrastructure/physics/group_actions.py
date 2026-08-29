r"""Group actions on (complex) MRI images for Equivariant Imaging (EI).

Equivariant Imaging (Chen et al., ICCV 2021) supervises a reconstruction
network *without ground truth* by enforcing the self-consistency identity

.. math::
    f\big(A\, T_g\, f(A^{*} y)\big) \;=\; T_g\, f(A^{*} y)

for elements :math:`g` of a symmetry group :math:`G` that the *signal set* is
invariant under but the *forward operator* :math:`A` is **not** equivariant to.
This module supplies the :math:`T_g` (and :math:`T_g^{-1}`) used by
:class:`mriforge.infrastructure.training.strategies.equivariant_imaging_strategy.EquivariantImagingStrategy`.

Identifiability (Tachella et al., sensing theorems, JMLR 2023) requires the
operator to *break* the group symmetry — otherwise the group only moves signal
*within* the measured subspace and EI learns nothing new. :func:`sensing_margin`
quantifies this, and the ``ei_sensing_margin`` audit refuses an arm whose
operator is (near-)equivariant to the chosen group (margin :math:`\to 0`).

For brain MRI the appropriate group is a *small-rotation* / dihedral subset,
**not** full ``SO(2)``: the brain is only approximately rotation-invariant, so a
large equivariance weight under continuous rotation would bias anatomy. Two
groups are provided:

* :class:`DihedralGroup` — the dihedral group :math:`D_4` (4 rotations by
  multiples of 90 degrees times an axis flip), realised by ``torch.rot90`` /
  ``torch.flip``, so it is **exact** (and exactly invertible) even on complex
  tensors. This is the default for square Cartesian grids.
* :class:`SmallRotationGroup` — a finite set of small continuous rotations
  realised by ``grid_sample``; complex tensors are rotated by acting on the
  real and imaginary parts jointly. The round-trip is only *approximate*
  (bilinear interpolation), as expected for a continuous action.

Actions are defined on image-domain tensors ``[B, C, H, W]`` and apply
identically to real and complex tensors, so they compose with the complex
k-space forward operator :math:`A = M F` through the physics SSOT
(:func:`mriforge.infrastructure.physics.fft_ops.fft2c` / ``ifft2c``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn.functional as F

__all__ = [
    "GROUP_REGISTRY",
    "DihedralGroup",
    "GroupAction",
    "SmallRotationGroup",
    "get_group_action",
    "sensing_margin",
]


class GroupAction(ABC):
    r"""A finite (sampled) group acting on image-domain tensors.

    Concrete subclasses implement :meth:`apply` (:math:`T_g`) and
    :meth:`inverse` (:math:`T_g^{-1}`) for element index ``g`` in
    ``range(num_elements)``, with ``g == 0`` always the identity.
    """

    #: Human-readable identifier (matches the registry key).
    name: str = "group"

    @property
    @abstractmethod
    def num_elements(self) -> int:
        """Number of group elements, including the identity at index 0."""

    @abstractmethod
    def apply(self, x: torch.Tensor, g: int) -> torch.Tensor:
        r"""Apply :math:`T_g` to ``x`` (``[B, C, H, W]``, real or complex)."""

    @abstractmethod
    def inverse(self, x: torch.Tensor, g: int) -> torch.Tensor:
        r"""Apply :math:`T_g^{-1}` to ``x``."""

    def sample(self, generator: torch.Generator | None = None) -> int:
        """Sample a uniform **non-identity** element index in ``[1, num_elements)``.

        EI draws a non-trivial transform each step; returning the identity would
        make the equivariance term vanish (a silent facade, pitfall #16).
        """
        n = self.num_elements
        if n <= 1:
            raise ValueError(f"{self.name!r} has no non-identity elements to sample from.")
        return int(torch.randint(1, n, (1,), generator=generator).item())


class DihedralGroup(GroupAction):
    r"""The dihedral group :math:`D_4` on a square grid (exact, complex-safe).

    Element ``g`` decomposes as ``k = g // 2`` quarter-turns followed by ``f =
    g % 2`` horizontal flips: :math:`T_g = F^{f} R^{k}`. Both ``rot90`` and
    ``flip`` are exact permutations of the lattice, so :meth:`inverse` recovers
    the input bit-exactly — including for complex tensors, which ``torch.rot90``
    / ``torch.flip`` handle natively.
    """

    name = "dihedral"

    @property
    def num_elements(self) -> int:
        return 8

    @staticmethod
    def _decompose(g: int) -> tuple[int, int]:
        if not 0 <= g < 8:
            raise ValueError(f"DihedralGroup element must be in [0, 8), got {g}.")
        return g // 2, g % 2

    def apply(self, x: torch.Tensor, g: int) -> torch.Tensor:
        k, f = self._decompose(g)
        out = torch.rot90(x, k, dims=(-2, -1))
        if f:
            out = torch.flip(out, dims=(-1,))
        return out

    def inverse(self, x: torch.Tensor, g: int) -> torch.Tensor:
        k, f = self._decompose(g)
        # T_g = F^f R^k  =>  T_g^{-1} = R^{-k} F^f  (flip is its own inverse).
        out = torch.flip(x, dims=(-1,)) if f else x
        return torch.rot90(out, -k, dims=(-2, -1))


class SmallRotationGroup(GroupAction):
    r"""A finite set of *small* continuous rotations (brain-appropriate).

    Index 0 is the identity; indices ``1 .. num_angles`` are equispaced angles
    in ``[-max_angle_deg, +max_angle_deg]`` (excluding 0). Rotation uses
    ``affine_grid`` / ``grid_sample`` (bilinear), so the action is approximate
    near the FOV edges. Complex inputs are rotated by acting on the real and
    imaginary channels jointly.
    """

    name = "small_rotation"

    def __init__(self, max_angle_deg: float = 10.0, num_angles: int = 4) -> None:
        if max_angle_deg <= 0.0:
            raise ValueError(f"max_angle_deg must be > 0, got {max_angle_deg}.")
        if num_angles < 1:
            raise ValueError(f"num_angles must be >= 1, got {num_angles}.")
        self.max_angle_deg = float(max_angle_deg)
        self.num_angles = int(num_angles)
        # Non-identity angles, symmetric about 0, excluding 0 itself.
        if num_angles == 1:
            self._angles = [self.max_angle_deg]
        else:
            step = (2.0 * self.max_angle_deg) / (self.num_angles + 1)
            self._angles = [-self.max_angle_deg + step * i for i in range(1, self.num_angles + 1)]

    @property
    def num_elements(self) -> int:
        return self.num_angles + 1

    def _angle_deg(self, g: int) -> float:
        if not 0 <= g < self.num_elements:
            raise ValueError(
                f"SmallRotationGroup element must be in [0, {self.num_elements}), got {g}."
            )
        return 0.0 if g == 0 else self._angles[g - 1]

    @staticmethod
    def _rotate(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        if angle_deg == 0.0:
            return x
        if x.is_complex():
            return torch.complex(
                SmallRotationGroup._rotate(x.real, angle_deg),
                SmallRotationGroup._rotate(x.imag, angle_deg),
            )
        rad = angle_deg * math.pi / 180.0
        cos, sin = math.cos(rad), math.sin(rad)
        b = x.shape[0]
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=x.dtype, device=x.device)
        theta = theta.unsqueeze(0).expand(b, 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def apply(self, x: torch.Tensor, g: int) -> torch.Tensor:
        return self._rotate(x, self._angle_deg(g))

    def inverse(self, x: torch.Tensor, g: int) -> torch.Tensor:
        return self._rotate(x, -self._angle_deg(g))


GROUP_REGISTRY: dict[str, type[GroupAction]] = {
    "dihedral": DihedralGroup,
    "small_rotation": SmallRotationGroup,
}


def get_group_action(name: str, **kwargs: object) -> GroupAction:
    """Resolve a registered group action by name (no silent fallback, pitfall #9).

    Args:
        name: registry key (``"dihedral"`` / ``"small_rotation"``).
        **kwargs: forwarded to the group constructor (e.g. ``max_angle_deg``).

    Raises:
        ValueError: if ``name`` is not registered.
    """
    if name not in GROUP_REGISTRY:
        raise ValueError(f"Unknown group action {name!r}; available: {sorted(GROUP_REGISTRY)}.")
    return GROUP_REGISTRY[name](**kwargs)  # type: ignore[arg-type]


def sensing_margin(
    forward_op: Callable[[torch.Tensor], torch.Tensor],
    adjoint_op: Callable[[torch.Tensor], torch.Tensor],
    group: GroupAction,
    x: torch.Tensor,
    *,
    n_samples: int = 4,
    generator: torch.Generator | None = None,
) -> float:
    r"""Identifiability gate for Equivariant Imaging.

    Returns the mean over sampled non-identity elements :math:`g` of the
    fraction of :math:`T_g x_r` that leaves the measured subspace
    ``range(A^* A)``:

    .. math::
        \text{margin} = \operatorname*{mean}_g
        \frac{\lVert (I - A^{*}A)\, T_g x_r \rVert}{\lVert T_g x_r \rVert},
        \qquad x_r = A^{*}A\,x ,

    where ``forward_op`` is :math:`A` and ``adjoint_op`` is :math:`A^{*}`. The
    input is first projected into ``range(A^* A)`` so the diagnostic measures
    only the *new* signal the group exposes. A margin of 0 means the operator is
    equivariant to the group (the group keeps signal inside the measured
    subspace) and EI adds nothing; a positive margin means the group rotates
    signal into the null space, which is exactly what EI learns to fill.

    Uses the first ``min(n_samples, num_elements - 1)`` *distinct* non-identity
    elements (deterministic when ``generator`` is omitted).
    """
    n = min(n_samples, group.num_elements - 1)
    if n < 1:
        raise ValueError("group has no non-identity elements to probe.")
    if generator is None:
        elements = list(range(1, n + 1))
    else:
        elements = []
        while len(elements) < n:
            g = int(torch.randint(1, group.num_elements, (1,), generator=generator).item())
            if g not in elements:
                elements.append(g)

    with torch.no_grad():
        x_range = adjoint_op(forward_op(x))
        margins: list[float] = []
        for g in elements:
            tg = group.apply(x_range, g)
            tg_proj = adjoint_op(forward_op(tg))
            num = torch.linalg.vector_norm(tg - tg_proj)
            den = torch.linalg.vector_norm(tg).clamp_min(1e-12)
            margins.append(float(num / den))
    return sum(margins) / len(margins)
