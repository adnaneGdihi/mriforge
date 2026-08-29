"""Per-pixel Bernoulli Fisher-Rao geometry — the geodesic primitive (MICCAI B-3.4).

Treats each magnitude intensity ``x in [0,1]`` as a Bernoulli distribution ``(x, 1-x)``. By
the textbook information-geometry result, the Fisher-Rao metric on the Bernoulli manifold,
under the sqrt-embedding ``(sqrt(x), sqrt(1-x))``, is the round metric of the unit CIRCLE:
``(cos theta, sin theta)`` with ``theta = arccos(sqrt(x)) in [0, pi/2]`` and ``x = cos^2 theta``.
The Fisher-Rao GEODESIC between two intensities is then the constant-speed arc in ``theta``
(spherical interpolation), and geodesic SHOOTING is ``x_t = cos^2(theta_s + delta)`` for an
angular velocity ``delta``.

Why this realisation: it is genuinely Fisher-Rao (the Bernoulli manifold is THE statistical
manifold of a [0,1] intensity), per-pixel so the output is image-valued, and ``cos^2`` keeps
every pixel in ``[0,1]`` BY CONSTRUCTION (no clamp) and smoothly. The Fisher metric primitives
for the Bloch likelihood live in
:mod:`mriforge.infrastructure.physics.fisher_rao_geometry` (the physics SSOT); this block is the
complementary Bernoulli-sphere geodesic used by the cross-field translator B-3.4.

Distinct from B-3.1 (Euclidean velocity ODE on raw images), B-3.3 (stochastic bridge) and
B-3.9 (curl-free Brenier displacement): here the transport is the SPHERICAL (information-
geometry) geodesic, not a Euclidean one — the genuine distinction is the smooth ANGULAR step
vs a clamped linear one.

HONEST SCOPE (read before interpreting results):
- "Bounded [0,1] without a clamp" is the real structural property; it is NOT "norm
  preservation" in any non-trivial sense (``x + (1-x) = 1`` holds for ANY ``x in [0,1]``), and
  the Euclidean ablation also lands in ``[0,1]`` via ``.clamp`` — the two differ in HOW
  (smooth angular vs clamped linear), not in the range.
- BOUNDARY-GRADIENT VANISHING: ``d(geodesic_step)/d(delta) = -sin(2(theta+delta))`` equals
  ``-2 sqrt(x(1-x))`` at ``delta=0``, which VANISHES at intensity boundaries (``x->0``
  background, ``x->1`` saturated tissue) — the learned velocity gets ~500x weaker signal there
  than at midtones. The Euclidean ablation has a constant gradient 1, so the two arms differ in
  boundary gradient CONDITIONING; an SSIM/dice delta should not be read as pure
  information-geometry effect. (The math is correct; the per-pixel CNN's receptive field lets a
  boundary pixel's loss still reach weights via neighbours, so the impact is bounded but real.)
- FOLDING: for a spatially-uniform ``delta`` the step is non-monotone near the boundary
  (``theta`` hits ``pi/2`` and ``cos^2`` folds); the per-pixel velocity net compensates.
"""

from __future__ import annotations

import torch


def intensity_to_angle(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Bernoulli sqrt-embedding angle ``theta = arccos(sqrt(x)) in [0, pi/2]`` for ``x in [0,1]``."""
    xc = x.clamp(eps, 1.0 - eps)
    return torch.arccos(xc.sqrt())


def angle_to_intensity(theta: torch.Tensor) -> torch.Tensor:
    """Inverse embedding ``x = cos^2(theta) in [0,1]`` (bounded for ANY theta — norm-preserving)."""
    return torch.cos(theta) ** 2


def fisher_rao_geodesic_step(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Geodesic shooting on the Bernoulli circle: ``x_t = cos^2(arccos(sqrt(x)) + delta)``.

    ``delta`` is the per-pixel angular velocity. ``delta=0`` is the identity. The output is in
    ``[0,1]`` by construction (no clamp needed). NOTE: ``d/d(delta) = -2 sqrt(x(1-x))`` at
    ``delta=0`` vanishes at the intensity boundaries (a learned-velocity trainability caveat;
    see the module docstring), and the step folds non-monotonically once ``theta`` passes
    ``pi/2``.
    """
    return angle_to_intensity(intensity_to_angle(x) + delta)


def fisher_rao_geodesic_distance(
    x_a: torch.Tensor, x_b: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    """Fisher-Rao (great-circle) distance between two intensities = ``|theta_a - theta_b|``.

    Equivalently ``arccos(sqrt(x_a x_b) + sqrt((1-x_a)(1-x_b)))`` (the Bhattacharyya angle).
    """
    xa = x_a.clamp(eps, 1.0 - eps)
    xb = x_b.clamp(eps, 1.0 - eps)
    bc = (xa * xb).sqrt() + ((1.0 - xa) * (1.0 - xb)).sqrt()
    return torch.arccos(bc.clamp(-1.0, 1.0))
