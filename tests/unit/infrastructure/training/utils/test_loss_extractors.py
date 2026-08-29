"""Unit tests for loss extractor classes.

Targets ``mriforge.infrastructure.training.utils.loss_extractors``:
- ``BaseLossExtractor.get_loss`` / ``get_loss_float``
- ``GANLossExtractor``, ``VAELossExtractor``, ``VQVAELossExtractor``,
  ``DiffusionLossExtractor``, ``ReconstructionLossExtractor``,
  ``MAELossExtractor``
- ``create_loss_extractor`` factory

Properties verified:
- Missing key returns zero-tensor default.
- Present key returns the stored tensor.
- ``get_loss_float`` converts tensors to float.
- ``extract_all_losses`` returns a dict with the expected keys.
- ``create_loss_extractor`` raises ``ValueError`` for unknown strategy type.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _t(v: float) -> torch.Tensor:
    return torch.tensor(v)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_base_extractor() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import BaseLossExtractor

    ext = BaseLossExtractor({"my_loss": _t(3.0)})
    result = ext.get_loss("my_loss")
    assert isinstance(result, torch.Tensor)
    assert float(result.item()) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# BaseLossExtractor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,present,expected",
    [
        ("g_total_loss", True, 5.0),
        ("g_total_loss", False, 0.0),
    ],
)
def test_base_get_loss_presence(key, present, expected) -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import BaseLossExtractor

    losses = {key: _t(5.0)} if present else {}
    ext = BaseLossExtractor(losses)
    val = ext.get_loss(key)
    assert float(val.item()) == pytest.approx(expected)


def test_base_get_loss_float_from_tensor() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import BaseLossExtractor

    ext = BaseLossExtractor({"lx": _t(2.5)})
    f = ext.get_loss_float("lx")
    assert isinstance(f, float)
    assert f == pytest.approx(2.5)


def test_base_get_loss_float_from_float() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import BaseLossExtractor

    ext = BaseLossExtractor({"lx": 9.9})
    f = ext.get_loss_float("lx")
    assert f == pytest.approx(9.9)


def test_base_get_loss_float_default_for_missing() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import BaseLossExtractor

    ext = BaseLossExtractor({})
    f = ext.get_loss_float("__missing__")
    assert f == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GANLossExtractor
# ---------------------------------------------------------------------------


def test_gan_extract_all_losses_keys() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import GANLossExtractor

    losses = {
        "g_total_loss": _t(1.0),
        "d_total_loss": _t(0.5),
        "g_loss_l1": _t(0.3),
        "g_loss_reconstruction": _t(0.2),
        "g_loss_adv": _t(0.1),
        "g_loss_perceptual": _t(0.05),
        "d_loss_real": _t(0.25),
        "d_loss_fake": _t(0.25),
    }
    ext = GANLossExtractor(losses)
    all_losses = ext.extract_all_losses()
    required_keys = {
        "g_total_loss", "d_total_loss", "g_loss_l1",
        "g_loss_reconstruction", "g_loss_adversarial", "g_loss_perceptual",
        "d_loss_real", "d_loss_fake",
    }
    assert required_keys.issubset(set(all_losses.keys()))


def test_gan_missing_keys_return_zero() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import GANLossExtractor

    ext = GANLossExtractor({})
    assert float(ext.get_g_total_loss().item()) == pytest.approx(0.0)
    assert float(ext.get_d_total_loss().item()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VAELossExtractor
# ---------------------------------------------------------------------------


def test_vae_kl_and_reconstruction() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import VAELossExtractor

    losses = {
        "g_total_loss": _t(1.5),
        "g_loss_reconstruction": _t(1.0),
        "g_loss_kl": _t(0.5),
    }
    ext = VAELossExtractor(losses)
    assert float(ext.get_total_loss().item()) == pytest.approx(1.5)
    assert float(ext.get_reconstruction_loss().item()) == pytest.approx(1.0)
    assert float(ext.get_kl_divergence().item()) == pytest.approx(0.5)


def test_vae_kl_weight_default_is_tensor() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import VAELossExtractor

    ext = VAELossExtractor({})
    w = ext.get_kl_weight()
    assert isinstance(w, torch.Tensor)


# ---------------------------------------------------------------------------
# create_loss_extractor factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_type", ["gan", "reconstruction", "diffusion", "vae", "vqvae", "mae"])
def test_factory_returns_extractor_for_known_types(strategy_type: str) -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import (
        BaseLossExtractor,
        create_loss_extractor,
    )

    ext = create_loss_extractor(strategy_type, {})
    assert isinstance(ext, BaseLossExtractor)


def test_factory_raises_for_unknown_type() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import create_loss_extractor

    with pytest.raises(ValueError, match="Unknown strategy_type"):
        create_loss_extractor("__bogus__", {})


# ---------------------------------------------------------------------------
# Edge: None losses dict
# ---------------------------------------------------------------------------


def test_edge_none_losses_treated_as_empty() -> None:
    from mriforge.infrastructure.training.utils.loss_extractors import BaseLossExtractor

    ext = BaseLossExtractor(None)  # type: ignore[arg-type]
    assert float(ext.get_loss("any_key").item()) == pytest.approx(0.0)
