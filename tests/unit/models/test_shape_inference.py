"""Unit tests for :mod:`spectramr.models.shape_inference`.

The module provides analytical (no-tensor) shape inference for conv2d
and maxpool2d stacks plus a couple of parameter validators. These are
pure-Python integer arithmetic — perfect for CPU-only testing.

Targets:

* ``infer_conv2d_output_size`` matches the PyTorch formula for every
  stride / padding / dilation combination.
* ``infer_maxpool2d_output_size`` ``stride=None`` defaults to
  ``kernel_size`` (PyTorch convention).
* ``infer_conv_stack_output`` composes both ops correctly, threading
  channel counts through.
* ``infer_conv_stack_output`` raises on unknown layer types
  (CLAUDE.md #9 loud-fail).
* ``compute_flattened_size`` is just ``H × W × C``.
* The three validators (``validate_conv2d_params``,
  ``validate_batch_size``, ``validate_image_dimensions``) raise on
  every documented invalid input.
* Pre-defined layer constants (``WAVELET_KAN_ENCODER_LAYERS``,
  ``FOURIER_KAN_ENCODER_LAYERS``) are well-formed and produce
  consistent shapes when fed through ``infer_conv_stack_output``.

These tests don't import torch — verifying you can use shape inference
in CI-time validation passes that don't have CUDA available.
"""

from __future__ import annotations

import pytest

from spectramr.models.shape_inference import (
    FOURIER_KAN_ENCODER_LAYERS,
    WAVELET_KAN_ENCODER_LAYERS,
    compute_flattened_size,
    infer_conv2d_output_size,
    infer_conv_stack_output,
    infer_maxpool2d_output_size,
    validate_batch_size,
    validate_conv2d_params,
    validate_image_dimensions,
)


# ── infer_conv2d_output_size ────────────────────────────────────────


class TestInferConv2dOutputSize:
    def test_same_padding_3x3(self) -> None:
        # 3x3 conv with padding=1 preserves spatial dims (the "same"
        # convention).
        assert infer_conv2d_output_size(64, 64, kernel_size=3, padding=1) == (64, 64)

    def test_no_padding_3x3(self) -> None:
        # 3x3 conv, no padding → H - 2.
        assert infer_conv2d_output_size(64, 64, kernel_size=3) == (62, 62)

    def test_stride_two_halves_dims(self) -> None:
        # stride=2 with padding=1, kernel=3 halves dims (PyTorch
        # convention; verified by hand: (64 + 2 - 2 - 1) // 2 + 1 = 32).
        assert infer_conv2d_output_size(
            64, 64, kernel_size=3, stride=2, padding=1
        ) == (32, 32)

    def test_dilation_two_with_padding_three(self) -> None:
        # Dilated 3x3 conv has effective kernel size = 5; needs
        # padding=2 to preserve dims.
        out = infer_conv2d_output_size(
            32, 32, kernel_size=3, padding=2, dilation=2
        )
        assert out == (32, 32)

    def test_non_square_input(self) -> None:
        # The formula must apply independently to H and W.
        out = infer_conv2d_output_size(
            16, 32, kernel_size=3, padding=1
        )
        assert out == (16, 32)


# ── infer_maxpool2d_output_size ─────────────────────────────────────


class TestInferMaxpool2dOutputSize:
    def test_kernel_2_default_stride_halves(self) -> None:
        # stride=None → stride = kernel_size = 2.
        assert infer_maxpool2d_output_size(64, 64, kernel_size=2) == (32, 32)

    def test_explicit_stride_one(self) -> None:
        # stride=1, kernel=2 → out = in - 1.
        assert infer_maxpool2d_output_size(
            64, 64, kernel_size=2, stride=1
        ) == (63, 63)

    def test_padding_preserves_dims_with_stride_one(self) -> None:
        # padding=1, kernel=3, stride=1 → preserved.
        assert infer_maxpool2d_output_size(
            32, 32, kernel_size=3, stride=1, padding=1
        ) == (32, 32)


# ── infer_conv_stack_output ─────────────────────────────────────────


class TestInferConvStackOutput:
    def test_single_conv_layer(self) -> None:
        layers = [
            {
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 16,
                "kernel_size": 3,
                "padding": 1,
            }
        ]
        assert infer_conv_stack_output(32, 32, layers) == (32, 32, 16)

    def test_conv_then_pool(self) -> None:
        layers = [
            {
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 16,
                "kernel_size": 3,
                "padding": 1,
            },
            {"type": "maxpool2d", "kernel_size": 2},
        ]
        # Conv keeps 32x32 (same padding), pool halves to 16x16.
        assert infer_conv_stack_output(32, 32, layers) == (16, 16, 16)

    def test_maxpool_does_not_change_channels(self) -> None:
        layers = [
            {
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
            },
            {"type": "maxpool2d", "kernel_size": 2},
        ]
        h, w, c = infer_conv_stack_output(64, 64, layers)
        assert c == 32  # channel count preserved through pool

    def test_empty_stack_falls_back_to_input(self) -> None:
        # Empty list → fall back to (h, w, 1).
        assert infer_conv_stack_output(32, 32, []) == (32, 32, 1)

    def test_unknown_layer_type_raises(self) -> None:
        layers = [{"type": "unknown_layer", "kernel_size": 3}]
        with pytest.raises(ValueError, match="Unsupported layer type"):
            infer_conv_stack_output(32, 32, layers)


# ── compute_flattened_size ──────────────────────────────────────────


class TestComputeFlattenedSize:
    def test_simple_product(self) -> None:
        assert compute_flattened_size(8, 8, 16) == 1024

    def test_zero_dim_returns_zero(self) -> None:
        assert compute_flattened_size(0, 8, 16) == 0


# ── validate_conv2d_params ──────────────────────────────────────────


class TestValidateConv2dParams:
    def test_valid_params_no_raise(self) -> None:
        validate_conv2d_params(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )

    @pytest.mark.parametrize(
        "kwargs, msg",
        [
            ({"in_channels": 0, "out_channels": 1, "kernel_size": 3}, "in_channels"),
            (
                {"in_channels": 1, "out_channels": -1, "kernel_size": 3},
                "out_channels",
            ),
            (
                {"in_channels": 1, "out_channels": 1, "kernel_size": 0},
                "kernel_size",
            ),
            (
                {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_size": 3,
                    "stride": 0,
                },
                "stride",
            ),
            (
                {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_size": 3,
                    "padding": -1,
                },
                "padding",
            ),
        ],
    )
    def test_invalid_params_raise_with_field_named(
        self, kwargs: dict, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            validate_conv2d_params(**kwargs)


# ── validate_batch_size ─────────────────────────────────────────────


class TestValidateBatchSize:
    def test_valid_batch_size(self) -> None:
        validate_batch_size(32)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_batch_size(0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_batch_size(-1)

    def test_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_batch_size(10_000, max_batch_size=1024)

    def test_at_max_accepted(self) -> None:
        # Exactly at the limit must NOT raise.
        validate_batch_size(1024, max_batch_size=1024)


# ── validate_image_dimensions ───────────────────────────────────────


class TestValidateImageDimensions:
    def test_valid_dimensions(self) -> None:
        validate_image_dimensions(256, 256)

    @pytest.mark.parametrize("h, w", [(0, 256), (256, 0)])
    def test_too_small_raises(self, h: int, w: int) -> None:
        with pytest.raises(ValueError, match="must be between"):
            validate_image_dimensions(h, w)

    @pytest.mark.parametrize("h, w", [(8192, 256), (256, 8192)])
    def test_too_large_raises(self, h: int, w: int) -> None:
        with pytest.raises(ValueError, match="must be between"):
            validate_image_dimensions(h, w, max_dim=4096)


# ── Pre-defined layer constants ─────────────────────────────────────


class TestPredefinedLayerStacks:
    def test_wavelet_kan_encoder_layers_well_formed(self) -> None:
        # Every layer entry has a "type" key, conv layers have all
        # required fields.
        for layer in WAVELET_KAN_ENCODER_LAYERS:
            assert "type" in layer
            if layer["type"] == "conv2d":
                for k in ("in_channels", "out_channels", "kernel_size"):
                    assert k in layer

    def test_fourier_kan_encoder_layers_well_formed(self) -> None:
        for layer in FOURIER_KAN_ENCODER_LAYERS:
            assert "type" in layer

    def test_wavelet_stack_infers_consistent_output(self) -> None:
        # 64x64 input → final pre-conv layer halves three times to 8x8,
        # final conv keeps spatial dims. Channels end at 512.
        h, w, c = infer_conv_stack_output(64, 64, WAVELET_KAN_ENCODER_LAYERS)
        assert h == 8
        assert w == 8
        assert c == 512

    def test_fourier_and_wavelet_share_topology(self) -> None:
        # Both encoders are designed to have the same in/out shape
        # contract — a regression guard against accidental drift.
        w_out = infer_conv_stack_output(64, 64, WAVELET_KAN_ENCODER_LAYERS)
        f_out = infer_conv_stack_output(64, 64, FOURIER_KAN_ENCODER_LAYERS)
        assert w_out == f_out
