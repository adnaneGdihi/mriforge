"""Unit tests for :mod:`spectramr.shared.utils.onnx_dynamic_axes`.

The module is a thin helper for ONNX export: given a sample input /
output shape, produce a ``dynamic_axes`` dict for ``torch.onnx.export``.

The two public functions are:

* :func:`extract_tensor_shape` — walk an arbitrary sample (tensor,
  dict, list / tuple) and return the shape of the *first* tensor it
  finds.
* :func:`build_dynamic_axes` — given an ``(in_shape, out_shape)`` pair
  return the standard ``{"input": {0: "batch", ...}, "output": ...}``
  mapping expected by ``torch.onnx.export``.

The shape-classification helper (``_axes_for_shape``) handles rank-3
(``[B, C, L]``), rank-4 (``[B, C, H, W]``), and rank-5+
(``[B, C, D, H, W, ...]``) layouts; we test each branch.

Uses real ``torch.zeros`` for the tensor input — small and CPU-safe.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.shared.utils.onnx_dynamic_axes import (
    _axes_for_shape,
    build_dynamic_axes,
    extract_tensor_shape,
)


# ── extract_tensor_shape ───────────────────────────────────────────


class TestExtractTensorShape:
    def test_bare_tensor(self) -> None:
        t = torch.zeros(2, 3, 16, 16)
        assert extract_tensor_shape(t) == (2, 3, 16, 16)

    def test_none_returns_none(self) -> None:
        assert extract_tensor_shape(None) is None

    def test_scalar_returns_none(self) -> None:
        # Plain ints / floats are not tensors, dicts, or iterables → None.
        assert extract_tensor_shape(42) is None
        assert extract_tensor_shape(1.5) is None

    def test_dict_finds_first_tensor(self) -> None:
        sample = {"a": "skip", "image": torch.zeros(1, 3, 8, 8), "noise": "skip"}
        out = extract_tensor_shape(sample)
        assert out == (1, 3, 8, 8)

    def test_nested_dict(self) -> None:
        sample = {"inputs": {"image": torch.zeros(2, 1, 4, 4)}}
        out = extract_tensor_shape(sample)
        assert out == (2, 1, 4, 4)

    def test_list_finds_first_tensor(self) -> None:
        sample = ["skip", None, torch.zeros(1, 2, 3)]
        assert extract_tensor_shape(sample) == (1, 2, 3)

    def test_tuple_iterates(self) -> None:
        sample = (None, torch.zeros(4, 1))
        assert extract_tensor_shape(sample) == (4, 1)

    def test_string_is_not_an_iterable_for_this_purpose(self) -> None:
        """Strings are technically iterable but treating them as such
        would explode the search — the implementation filters strings
        out and returns None for a bare string."""
        assert extract_tensor_shape("hello") is None

    def test_bytes_is_not_an_iterable_for_this_purpose(self) -> None:
        assert extract_tensor_shape(b"\x00\x01") is None

    def test_empty_dict_returns_none(self) -> None:
        assert extract_tensor_shape({}) is None

    def test_empty_list_returns_none(self) -> None:
        assert extract_tensor_shape([]) is None


# ── _axes_for_shape ────────────────────────────────────────────────


class TestAxesForShape:
    def test_rank_2_only_batch_axis(self) -> None:
        # [B, C] — only the batch dim is dynamic.
        out = _axes_for_shape((2, 3))
        assert out == {0: "batch"}

    def test_rank_3_adds_length(self) -> None:
        # 1-D signal [B, C, L].
        out = _axes_for_shape((1, 3, 16))
        assert out == {0: "batch", 2: "length"}

    def test_rank_4_adds_height_and_width(self) -> None:
        # 2-D image [B, C, H, W].
        out = _axes_for_shape((1, 3, 64, 64))
        assert out == {0: "batch", 2: "height", 3: "width"}

    def test_rank_5_adds_depth_height_width(self) -> None:
        # 3-D volume [B, C, D, H, W].
        out = _axes_for_shape((1, 3, 8, 64, 64))
        assert out == {
            0: "batch",
            2: "depth",
            3: "height",
            4: "width",
        }

    def test_rank_above_5_uses_generic_dim_names(self) -> None:
        out = _axes_for_shape((1, 3, 8, 64, 64, 4, 2))
        assert out[0] == "batch"
        assert out[2] == "depth"
        assert out[3] == "height"
        assert out[4] == "width"
        # Beyond rank 5 → "dim5", "dim6", …
        assert out[5] == "dim5"
        assert out[6] == "dim6"

    def test_rank_0_scalar_raises(self) -> None:
        # WS-C C4: a rank-0 (scalar) shape has no axis 0 to mark dynamic;
        # silently returning {0: "batch"} would emit an invalid axes map.
        with pytest.raises(ValueError, match="rank-0"):
            _axes_for_shape(())


# ── build_dynamic_axes ─────────────────────────────────────────────


class TestBuildDynamicAxes:
    def test_both_shapes_set(self) -> None:
        out = build_dynamic_axes((1, 3, 64, 64), (1, 1, 64, 64))
        assert set(out) == {"input", "output"}
        assert out["input"] == {0: "batch", 2: "height", 3: "width"}
        assert out["output"] == {0: "batch", 2: "height", 3: "width"}

    def test_input_only_defaults_output_to_batch_axis(self) -> None:
        out = build_dynamic_axes((1, 3, 64, 64), None)
        assert out["input"] == {0: "batch", 2: "height", 3: "width"}
        # Output has only the batch dim documented.
        assert out["output"] == {0: "batch"}

    def test_output_only_defaults_input_to_batch_axis(self) -> None:
        out = build_dynamic_axes(None, (1, 3, 64, 64))
        assert out["input"] == {0: "batch"}
        assert out["output"] == {0: "batch", 2: "height", 3: "width"}

    def test_both_none_returns_batch_only(self) -> None:
        # No shape info available — fall back to batch-only on both
        # roles. This is the safe default for ``torch.onnx.export``.
        out = build_dynamic_axes(None, None)
        assert out == {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    def test_mixed_rank_3_input_4_output(self) -> None:
        out = build_dynamic_axes((1, 3, 16), (1, 1, 64, 64))
        assert out["input"] == {0: "batch", 2: "length"}
        assert out["output"] == {0: "batch", 2: "height", 3: "width"}
