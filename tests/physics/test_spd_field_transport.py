r"""Physics tests for SPD-manifold field-transport harmonization (SPD-FT).

Targets ``spectramr.infrastructure.physics.spd_field_transport``.

Manifold-valued MRI estimands (diffusion / structure / conductivity tensors) live
on the cone :math:`\mathrm{Sym}^+(n)` and are field-dependent. SPD-FT models the
field effect as a congruence :math:`\Sigma\mapsto E\,\Sigma\,E^\top` and harmonises
by affine-invariant transport, with :math:`E=\Sigma_{B_0'}^{1/2}\Sigma_{B_0}^{-1/2}`
estimated from traveling-subject / phantom pairs. Properties checked:

- **PD preservation.** Transported tensors stay on the cone (unlike Euclidean
  ComBat, which can leave it).
- **Mean mapping.** The estimated :math:`E` maps the source tissue mean exactly
  onto the target mean.
- **ComBat limit.** For commuting scalar tensors the transport reduces to a
  multiplicative scalar correction (a log-domain ComBat shift).
- **Identity.** Same-field transport is the identity.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.physics


def _spd(n: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    torch.manual_seed(seed)
    a = torch.randn(n, n)
    return scale * (a @ a.T + n * torch.eye(n))


def test_transport_preserves_positive_definiteness() -> None:
    from spectramr.infrastructure.physics.spd_field_transport import (
        affine_invariant_transport_operator,
        spd_field_transport,
    )

    src_mean = _spd(3, 0)
    tgt_mean = _spd(3, 1, scale=2.0)
    e = affine_invariant_transport_operator(src_mean, tgt_mean)

    sigma = _spd(3, 5)
    out = spd_field_transport(sigma, e)
    eigvals = torch.linalg.eigvalsh(out)
    assert torch.all(eigvals > 0), "transported tensor must remain SPD"


def test_operator_maps_source_mean_to_target_mean() -> None:
    from spectramr.infrastructure.physics.spd_field_transport import (
        affine_invariant_transport_operator,
        spd_field_transport,
    )

    src_mean = _spd(3, 2)
    tgt_mean = _spd(3, 3, scale=1.7)
    e = affine_invariant_transport_operator(src_mean, tgt_mean)
    mapped = spd_field_transport(src_mean, e)
    assert torch.allclose(mapped, tgt_mean, atol=1e-4)


def test_combat_scalar_limit() -> None:
    """Commuting scalar tensors: transport is a multiplicative scalar correction."""
    from spectramr.infrastructure.physics.spd_field_transport import (
        affine_invariant_transport_operator,
        spd_field_transport,
    )

    sigma_s, sigma_t = 2.0, 5.0
    src_mean = (sigma_s**2) * torch.eye(3)
    tgt_mean = (sigma_t**2) * torch.eye(3)
    e = affine_invariant_transport_operator(src_mean, tgt_mean)

    x = 9.0 * torch.eye(3)  # scalar tensor tau^2 I
    out = spd_field_transport(x, e)
    expected = (sigma_t / sigma_s) ** 2 * x
    assert torch.allclose(out, expected, atol=1e-4)


def test_same_field_transport_is_identity() -> None:
    from spectramr.infrastructure.physics.spd_field_transport import (
        affine_invariant_transport_operator,
        spd_field_transport,
    )

    mean = _spd(3, 7)
    e = affine_invariant_transport_operator(mean, mean)
    sigma = _spd(3, 8)
    assert torch.allclose(spd_field_transport(sigma, e), sigma, atol=1e-4)


def test_estimate_transport_from_pairs() -> None:
    from spectramr.infrastructure.physics.spd_field_transport import (
        estimate_transport_from_pairs,
        spd_field_transport,
    )

    # Pairs related by a known congruence G.
    torch.manual_seed(11)
    g = torch.randn(3, 3) + 2.0 * torch.eye(3)
    src = torch.stack([_spd(3, k) for k in range(20, 30)])
    tgt = torch.stack([g @ s @ g.T for s in src])
    e = estimate_transport_from_pairs(src, tgt)
    # E maps the source mean onto the target mean.
    mapped_mean = spd_field_transport(src.mean(0), e)
    assert torch.allclose(mapped_mean, tgt.mean(0), atol=1e-3)
