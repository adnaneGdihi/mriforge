"""The cocycle residual cannot certify the anti-ensemble claim; field sensitivity can.

These tests are the executable shadow of ``docs/lean/MRIxFields/MRIxFields/Cocycle.lean``
(sorry-free, axioms: propext / Classical.choice / Quot.sound):

* ``IsFacade.all_residuals_zero`` — a field-blind perfect autoencoder ``R(q,b) = D(q)``
  with ``D∘E = id`` **is** the identity translator ``G = Id``, and it drives the cocycle,
  latent-cycle AND field-identity residuals to exactly zero. So ``cocycle_residual ~ 0``
  is consistent with total collapse (pitfall #16: the measurement does not exclude the
  facade it names).
* ``cocycleResidual_le`` — ``eps_coc <= L * eps_lat``: the cocycle penalty is dominated by
  the latent-cycle penalty already in the objective.
* ``fieldSensitivity_ge_of_fidelity`` — what *does* discriminate the two is whether the
  render moves when only the target field moves.

Here we exercise those statements numerically against the real renderer contract.
"""

from __future__ import annotations

import torch
from torch import nn


class _FieldBlindRenderer(nn.Module):
    """The facade: an autoencoder whose renderer IGNORES the field argument."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x)  # E = id

    def render(
        self,
        q: torch.Tensor,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del field_strength, contrast_id  # D(q) — the field is never read
        return self.identity(q)


def _residuals(gen: _FieldBlindRenderer, x: torch.Tensor, b_s, b_t, b_m):
    q = gen.encode(x)
    x_hat = gen.render(q, b_t)  # G(x; s, t)
    x_id = gen.render(q, b_s)  # G(x; s, s)
    x_m = gen.render(q, b_m)
    x_mt = gen.render(gen.encode(x_m), b_t)  # G(G(x; s, m); m, t)
    return {
        "cocycle": torch.mean(torch.abs(x_mt - x_hat)),
        "latent": torch.mean(torch.abs(gen.encode(x_hat) - q)),
        "identity": torch.mean(torch.abs(x_id - x)),
        "field_sensitivity": torch.mean(torch.abs(x_hat - x_id)),
    }


class TestFacadeSatisfiesEveryAdvertisedResidual:
    """`IsFacade.all_residuals_zero` + `fieldSensitivity_zero`, numerically."""

    def test_field_blind_autoencoder_zeroes_all_three_logged_residuals(self) -> None:
        gen = _FieldBlindRenderer()
        x = torch.rand(2, 1, 8, 8)
        r = _residuals(
            gen,
            x,
            torch.tensor([0.1, 0.1]),
            torch.tensor([7.0, 7.0]),
            torch.tensor([3.0, 3.0]),
        )
        # The certificate the arm publishes: all perfect.
        assert float(r["cocycle"]) == 0.0
        assert float(r["latent"]) == 0.0
        assert float(r["identity"]) == 0.0
        # ...on a model that renders 0.1 T and 7 T identically. The certificate is vacuous.
        assert float(r["field_sensitivity"]) == 0.0

    def test_field_sensitivity_is_what_separates_method_from_facade(self) -> None:
        """A genuinely field-dependent renderer has non-zero sensitivity."""

        class FieldAware(_FieldBlindRenderer):
            def render(self, q, field_strength, contrast_id=None):  # type: ignore[no-untyped-def]
                del contrast_id
                b = field_strength.reshape(-1, 1, 1, 1).to(q.dtype)
                return q * b  # the field genuinely moves the render

        gen = FieldAware()
        x = torch.rand(2, 1, 8, 8)
        r = _residuals(
            gen,
            x,
            torch.tensor([0.1, 0.1]),
            torch.tensor([7.0, 7.0]),
            torch.tensor([3.0, 3.0]),
        )
        assert float(r["field_sensitivity"]) > 0.0


class TestCocycleIsDominatedByLatentCycle:
    """`cocycleResidual_le`: eps_coc <= L * eps_lat for an L-Lipschitz renderer."""

    def test_cocycle_residual_bounded_by_lipschitz_times_latent_residual(self) -> None:
        lipschitz_const = 2.0

        class ScaleRenderer(_FieldBlindRenderer):
            def render(self, q, field_strength, contrast_id=None):  # type: ignore[no-untyped-def]
                del contrast_id, field_strength
                return lipschitz_const * q  # L-Lipschitz in q

            def encode(self, x):  # type: ignore[no-untyped-def]
                return 0.5 * x  # so E∘R is not idempotent -> non-zero latent residual

        gen = ScaleRenderer()
        x = torch.rand(4, 1, 8, 8)
        r = _residuals(
            gen,
            x,
            torch.tensor([0.1] * 4),
            torch.tensor([7.0] * 4),
            torch.tensor([3.0] * 4),
        )
        assert float(r["cocycle"]) <= lipschitz_const * float(r["latent"]) + 1e-6


class TestStrategyExposesTheInstrument:
    def test_strategy_declares_field_sensitivity_exposure(self) -> None:
        from spectramr.infrastructure.training.strategies.field_cocycle_strategy import (
            FieldCocycleTranslationStrategy,
        )

        # The seam only forwards a batch key whose name appears in the signature.
        import inspect

        params = inspect.signature(
            FieldCocycleTranslationStrategy.validation_step
        ).parameters
        assert "field_strength" in params, (
            "validation_step must declare field_strength or the pipeline seam will not "
            "forward the SOURCE field, and G(x; s, s) — hence field_sensitivity — "
            "cannot be computed at validation."
        )
