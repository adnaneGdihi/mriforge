"""Regression: unconditional diffusion sampling must NOT silently default the
noise shape.

``preprocess_input(..., unconditional=True)`` generates the initial noise from
``in_channels`` / ``image_size`` in the diffusion config. These were read with
``.get(key, default)`` (→ 1 channel, 256 px), so a config that omits them (or a
model trained on a different shape — 2-channel complex, 320 px M4Raw, …) silently
produced mis-shaped noise the model cannot consume. They are now required (raise)
in the unconditional path.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.inference.diffusion_inference_strategy import (
    DiffusionInferenceStrategy,
)


def _bare_strategy(diffusion_config: dict) -> DiffusionInferenceStrategy:
    """Construct just enough of the strategy to call preprocess_input, without
    the heavy mixin/config initialisation."""
    strat = DiffusionInferenceStrategy.__new__(DiffusionInferenceStrategy)
    strat.diffusion_config = diffusion_config
    strat.device = torch.device("cpu")
    return strat


def test_unconditional_missing_shape_keys_raises():
    strat = _bare_strategy({})  # no in_channels / image_size
    try:
        strat.preprocess_input(torch.zeros(1, 1, 8, 8), unconditional=True)
    except ValueError as exc:
        assert "in_channels" in str(exc) and "image_size" in str(exc)
    else:
        raise AssertionError(
            "unconditional sampling silently defaulted the noise shape instead "
            "of raising on missing in_channels/image_size"
        )


def test_unconditional_with_configured_shape_produces_that_shape():
    strat = _bare_strategy({"in_channels": 2, "image_size": 32})
    out = strat.preprocess_input(
        torch.zeros(1, 1, 8, 8), unconditional=True, batch_size=3
    )
    assert out.shape == (3, 2, 32, 32)


def test_conditional_path_uses_input_tensor_shape():
    """The conditional path (the common case) is unaffected — it passes the real
    input tensor through, no shape defaulting involved."""
    strat = _bare_strategy({})
    x = torch.randn(2, 1, 16, 16)
    out = strat.preprocess_input(x)  # unconditional defaults to False
    assert out.shape == x.shape
