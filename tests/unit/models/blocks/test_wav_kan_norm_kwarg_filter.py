"""Regression test for :class:`WavKANConvNDLayer` norm-kwarg leakage.

Audit anchor: 2026-05-14 E13 / F8 (``_BatchNorm.__init__() got an
unexpected keyword argument 'grid_size'``).

Background
----------
The caller chain ``RefinedKANUNet → DoubleConvWithKAN →
WavKANConv2DLayer`` forwards the full YAML ``model_kwargs`` blob via
``**norm_kwargs``. Pre-2026-05-15 those KAN-specific options
(``grid_size``, ``spline_order``, ``scale_noise``, …) landed on
``BatchNorm2d.__init__`` and raised a confusing ``TypeError``.

The fix in ``wav_kan.py`` introspects ``norm_class`` with
``inspect.signature`` and keeps only kwargs the norm constructor
accepts. This test pins that behaviour.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from spectramr.models.layers.kan.kan_convs.wav_kan import (
    WavKANConv2DLayer,
    WavKANConvNDLayer,
)


# The full set of KAN options that *can* arrive via a YAML
# ``model_kwargs`` blob and that have appeared in real experiment
# configs. None of these are accepted by ``nn.BatchNorm2d``; they must
# all be filtered out by ``WavKANConvNDLayer.__init__``.
KAN_LEAK_KWARGS: dict[str, object] = {
    "grid_size": 5,
    "spline_order": 3,
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "grid_update_freq": 10,
    "grid_range": [-1.0, 1.0],
    "grid_update_decay": 0.95,
    "base_filters": 32,
    "features": 64,
    "bottleneck_only": False,
}


def test_wav_kan_accepts_kan_kwargs_without_typeerror() -> None:
    """A YAML ``model_kwargs`` blob with KAN options must not crash construction."""
    layer = WavKANConv2DLayer(
        input_dim=4,
        output_dim=8,
        kernel_size=3,
        groups=1,
        padding=1,
        norm_layer=nn.BatchNorm2d,
        **KAN_LEAK_KWARGS,
    )
    assert isinstance(layer, WavKANConvNDLayer)
    # The KAN-specific keys must NOT survive on ``self.norm_kwargs``;
    # otherwise they would still leak into the norm constructor below.
    for k in KAN_LEAK_KWARGS:
        assert k not in layer.norm_kwargs, (
            f"KAN leak: {k!r} survived the norm_kwargs filter. "
            "Re-check the inspect.signature filter in WavKANConvNDLayer.__init__."
        )


def test_wav_kan_keeps_valid_batchnorm_kwargs() -> None:
    """Valid BN kwargs (``eps``, ``momentum``, ``affine``) survive the filter."""
    layer = WavKANConv2DLayer(
        input_dim=4,
        output_dim=8,
        kernel_size=3,
        padding=1,
        norm_layer=nn.BatchNorm2d,
        eps=1e-3,
        momentum=0.05,
        affine=False,
        **KAN_LEAK_KWARGS,
    )
    assert layer.norm_kwargs == {
        "eps": 1e-3,
        "momentum": 0.05,
        "affine": False,
    }
    # Sanity: those kwargs are in fact accepted by BatchNorm2d.
    valid = {
        k for k, p in inspect.signature(nn.BatchNorm2d).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    for k in layer.norm_kwargs:
        assert k in valid


def test_wav_kan_forward_pass_after_filter() -> None:
    """Construction succeeds *and* forward pass works with KAN kwargs."""
    layer = WavKANConv2DLayer(
        input_dim=4,
        output_dim=8,
        kernel_size=3,
        padding=1,
        norm_layer=nn.BatchNorm2d,
        **KAN_LEAK_KWARGS,
    )
    x = torch.randn(2, 4, 8, 8)
    y = layer(x)
    assert y.shape[:2] == (2, 8), f"unexpected output channels: {y.shape}"


def test_wav_kan_with_instance_norm() -> None:
    """Filter must also work with a different norm class (``nn.InstanceNorm2d``)."""
    layer = WavKANConv2DLayer(
        input_dim=4,
        output_dim=8,
        kernel_size=3,
        padding=1,
        norm_layer=nn.InstanceNorm2d,
        **KAN_LEAK_KWARGS,
    )
    for k in KAN_LEAK_KWARGS:
        assert k not in layer.norm_kwargs


@pytest.mark.parametrize("bad_kwarg", list(KAN_LEAK_KWARGS.keys()))
def test_unfiltered_kwarg_would_break_batchnorm(bad_kwarg: str) -> None:
    """Sanity: BatchNorm2d genuinely rejects each leak kwarg.

    If this fails (i.e. BatchNorm accepts a KAN kwarg in some future torch
    release), then the filter test above can no longer prove the bug —
    update the test set to the kwargs that still reject.
    """
    with pytest.raises(TypeError):
        nn.BatchNorm2d(4, **{bad_kwarg: KAN_LEAK_KWARGS[bad_kwarg]})  # type: ignore[arg-type]
