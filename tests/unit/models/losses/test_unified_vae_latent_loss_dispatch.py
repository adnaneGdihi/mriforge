"""WS1-core-04 (b): ``losses.latent.latent_loss_type`` selects the VAE latent
regulariser, dispatched through the loss registry like any other loss. The
default ``kl_divergence`` keeps the byte-identical inline-KL path; alternatives
(``latent_regularization``, ``mmd``) resolve via ``create_loss``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.computers.unified_vae import UnifiedVAELossComputer


def _stub(latent_loss_type):
    return SimpleNamespace(
        _latent_loss_cache=None,
        config=SimpleNamespace(
            losses=SimpleNamespace(
                latent=SimpleNamespace(latent_loss_type=latent_loss_type)
            )
        ),
    )


def test_default_kl_uses_inline_path() -> None:
    fn, name = UnifiedVAELossComputer._configured_latent_loss(_stub("kl_divergence"))
    assert fn is None
    assert name == "kl_divergence"


def test_unset_defaults_to_inline_kl() -> None:
    fn, _ = UnifiedVAELossComputer._configured_latent_loss(_stub(None))
    assert fn is None


def test_nondefault_resolves_via_registry() -> None:
    fn, name = UnifiedVAELossComputer._configured_latent_loss(
        _stub("latent_regularization")
    )
    assert fn is not None
    assert name == "latent_regularization"


def test_kl_family_called_with_mu_logvar() -> None:
    captured = {}

    def loss_fn(a, b):
        captured["args"] = (a, b)
        return torch.tensor(1.0)

    mu, logvar = torch.zeros(2, 4), torch.zeros(2, 4)
    UnifiedVAELossComputer._apply_latent_loss(loss_fn, "kl", mu, logvar)
    assert captured["args"][0] is mu and captured["args"][1] is logvar


def test_latent_regularization_called_with_reparam_sample() -> None:
    seen = {}

    def loss_fn(z):
        seen["shape"] = tuple(z.shape)
        return torch.tensor(0.0)

    UnifiedVAELossComputer._apply_latent_loss(
        loss_fn, "latent_regularization", torch.zeros(2, 4), torch.zeros(2, 4)
    )
    assert seen["shape"] == (2, 4)


def test_unknown_latent_loss_raises() -> None:
    with pytest.raises(ConfigurationError):
        UnifiedVAELossComputer._apply_latent_loss(
            lambda *a: None, "bogus", torch.zeros(2, 4), torch.zeros(2, 4)
        )


def test_schema_accepts_supported_latent_loss_types() -> None:
    from spectramr.config.schemas.loss import LatentLossesConfig

    for v in ("kl_divergence", "latent_regularization", "mmd", "vq_kl"):
        assert LatentLossesConfig(latent_loss_type=v).latent_loss_type == v


def test_schema_rejects_unsupported_latent_loss_type() -> None:
    from pydantic import ValidationError

    from spectramr.config.schemas.loss import LatentLossesConfig

    with pytest.raises(ValidationError):
        LatentLossesConfig(latent_loss_type="not_a_loss")
