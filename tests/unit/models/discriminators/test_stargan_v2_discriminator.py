"""Unit tests for the StarGAN v2 multi-domain discriminator (Task 9).

Covers :class:`~spectramr.models.discriminators.stargan_v2_discriminator.StarGANv2Discriminator`:

* Output shape: ``[B]`` (one real/fake logit per sample, from its own domain head).
* Domain-head selection: different ``y`` values select different heads, so same-input
  logits differ across domains (anti-facade, pitfall #16).
* Out-of-range domain index raises ``ValueError`` (pitfall #9 — no silent clamp).
* Registry: ``"stargan_v2_discriminator"`` is registered and resolvable.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

from spectramr.models.discriminators.stargan_v2_discriminator import StarGANv2Discriminator


def test_discriminator_returns_per_domain_logit() -> None:
    d = StarGANv2Discriminator(img_channels=1, num_domains=5)
    out = d(torch.rand(4, 1, 64, 64), torch.tensor([0, 1, 2, 4]))
    assert out.shape == (4,)


def test_discriminator_different_y_selects_different_head() -> None:
    """Two distinct domain labels must yield different logits on the same input."""
    d = StarGANv2Discriminator(img_channels=1, num_domains=5)
    d.eval()
    x = torch.rand(2, 1, 64, 64)
    logits_0 = d(x, torch.tensor([0, 0]))
    logits_4 = d(x, torch.tensor([4, 4]))
    # Different heads -> different logit values (not identical).
    assert (logits_0 - logits_4).abs().mean() > 1e-6


def test_out_of_range_domain_index_raises() -> None:
    """Pitfall #9: an unknown domain index must raise, never silently clamp."""
    d = StarGANv2Discriminator(img_channels=1, num_domains=5)
    with pytest.raises(ValueError):
        d(torch.rand(1, 1, 64, 64), torch.tensor([5]))
    with pytest.raises(ValueError):
        d(torch.rand(1, 1, 64, 64), torch.tensor([-1]))


def test_discriminator_registered_and_resolvable() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY, get_model_class

    populate_model_registry()
    assert "stargan_v2_discriminator" in MODEL_REGISTRY
    assert get_model_class("stargan_v2_discriminator") is StarGANv2Discriminator
