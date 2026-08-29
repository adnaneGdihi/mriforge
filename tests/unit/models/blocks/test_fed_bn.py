r"""Tests for ``FedBN`` (idea 8 §8.2).

Targets ``mriforge.models.blocks.fed_bn``.

Categories:

- Construction: rejects non-positive num_features / momentum / eps,
  rejects unsupported ndim
- Forward at training time normalises input across the batch
- Forward at eval time uses the running stats
- Output shape and channel count preserved
- Weight / bias are real ``nn.Parameter`` (trainable)
- ``state_dict()`` excludes the running buffers (FedBN semantic) but
  includes ``weight`` and ``bias``
- ``is_local_only`` flag is True (federated-aggregator gate)
- ndim variants: 1-D, 2-D, 3-D inputs
- Channel-count mismatch raises
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.fed_bn import FedBN


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_zero_num_features_rejected() -> None:
    """``num_features ≤ 0`` raises."""
    with pytest.raises(ValueError, match="num_features"):
        FedBN(num_features=0)


def test_zero_momentum_rejected() -> None:
    """``momentum ≤ 0`` raises."""
    with pytest.raises(ValueError, match="momentum"):
        FedBN(num_features=4, momentum=0.0)


def test_unsupported_ndim_rejected() -> None:
    """``ndim`` outside {1, 2, 3} raises."""
    with pytest.raises(ValueError, match="ndim"):
        FedBN(num_features=4, ndim=4)


def test_zero_eps_rejected() -> None:
    """``eps ≤ 0`` raises."""
    with pytest.raises(ValueError, match="eps"):
        FedBN(num_features=4, eps=0.0)


# ---------------------------------------------------------------------------
# Forward — training mode normalises across the batch
# ---------------------------------------------------------------------------


def test_training_mode_normalises_to_zero_mean_unit_var() -> None:
    """In training mode, output has ≈ 0 mean and ≈ 1 var per channel."""
    layer = FedBN(num_features=4, ndim=2).train()
    torch.manual_seed(0)
    x = torch.randn(32, 4, 8, 8) * 5.0 + 3.0
    out = layer(x)
    # Check per-channel statistics across the batch.
    mean = out.mean(dim=(0, 2, 3))
    var = out.var(dim=(0, 2, 3), unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_eval_mode_uses_running_stats() -> None:
    """``eval()`` uses running mean / var, not the batch's."""
    layer = FedBN(num_features=4, ndim=2)
    # Run a few training steps to populate running stats.
    layer.train()
    for _ in range(8):
        layer(torch.randn(16, 4, 8, 8))
    saved_mean = layer.running_mean.clone()
    saved_var = layer.running_var.clone()
    layer.eval()
    layer(torch.randn(16, 4, 8, 8))
    # Running stats unchanged in eval mode.
    assert torch.equal(layer.running_mean, saved_mean)
    assert torch.equal(layer.running_var, saved_var)


# ---------------------------------------------------------------------------
# Shape & channel contract
# ---------------------------------------------------------------------------


def test_output_shape_matches_input_2d() -> None:
    """``[B, C, H, W]`` input → same shape output."""
    layer = FedBN(num_features=4, ndim=2)
    x = torch.randn(2, 4, 8, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_output_shape_matches_input_3d() -> None:
    """``[B, C, D, H, W]`` input → same shape output."""
    layer = FedBN(num_features=4, ndim=3)
    x = torch.randn(2, 4, 4, 8, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_channel_count_mismatch_rejected() -> None:
    """Input channel count must equal ``num_features``."""
    layer = FedBN(num_features=4, ndim=2)
    with pytest.raises(ValueError, match="channel mismatch"):
        layer(torch.zeros(2, 7, 8, 8))


def test_wrong_dim_input_rejected() -> None:
    """``ndim=2`` requires 4-D input."""
    layer = FedBN(num_features=4, ndim=2)
    with pytest.raises(ValueError, match="expected"):
        layer(torch.zeros(2, 4, 8))


# ---------------------------------------------------------------------------
# FedBN semantics: state_dict excludes running stats
# ---------------------------------------------------------------------------


def test_state_dict_includes_affine_params() -> None:
    """``weight`` and ``bias`` are part of the state dict (so they're
    trainable per-site but you can also persist them locally if needed)."""
    layer = FedBN(num_features=4, ndim=2)
    sd = layer.state_dict()
    assert "weight" in sd
    assert "bias" in sd


def test_state_dict_excludes_running_stats() -> None:
    """The running buffers are non-persistent — the FedBN semantic.

    Federated aggregators that average ``state_dict()`` therefore
    skip them, leaving each site with its own statistics.
    """
    layer = FedBN(num_features=4, ndim=2)
    sd = layer.state_dict()
    assert "running_mean" not in sd
    assert "running_var" not in sd
    assert "num_batches_tracked" not in sd


def test_is_local_only_flag_true() -> None:
    """The audit gate consults ``is_local_only`` to skip aggregation."""
    layer = FedBN(num_features=4)
    assert layer.is_local_only is True


def test_weight_and_bias_are_parameters() -> None:
    """γ and β are real ``nn.Parameter`` objects (trainable)."""
    layer = FedBN(num_features=4)
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert isinstance(layer.bias, torch.nn.Parameter)
    assert layer.weight.requires_grad
    assert layer.bias.requires_grad


def test_running_stats_update_during_training() -> None:
    """Running mean / var move toward batch stats over multiple training steps."""
    layer = FedBN(num_features=4, ndim=2).train()
    # Use a non-zero-mean input so the running mean has somewhere to move.
    initial_mean = layer.running_mean.clone()
    for _ in range(20):
        layer(torch.randn(16, 4, 8, 8) + 5.0)
    assert not torch.equal(layer.running_mean, initial_mean)
