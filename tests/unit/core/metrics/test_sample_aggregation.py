"""Tests for :mod:`spectramr.core.metrics.sample_aggregation` (issue #1347).

The module owns one invariant -- a published metric is a mean over *samples* --
in two pieces: the sample-axis rule a per-sample reduction needs, and the
constants that name the convention in ``provenance.json``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.sample_aggregation import (
    PSNR_REDUCTION,
    SAMPLE_AXIS_MIN_NDIM,
    VALIDATION_EPOCH_WEIGHTING,
    aggregation_provenance,
    per_sample_flat,
    per_sample_peak,
)


class TestPerSampleFlat:
    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ((4, 1, 8, 8), (4, 64)),
            ((2, 3, 8, 8), (2, 192)),
            ((2, 1, 4, 8, 8), (2, 256)),  # 5-D volume
        ],
    )
    def test_a_tensor_with_a_sample_axis_keeps_dim_zero(self, shape, expected):
        assert tuple(per_sample_flat(torch.zeros(shape)).shape) == expected

    @pytest.mark.parametrize("shape", [(8, 8), (3, 8, 8), (64,)])
    def test_a_tensor_below_the_threshold_is_one_sample(self, shape):
        """A bare image must not have its channel or row axis read as a batch --
        that would re-create #1347 one layer down, in the metric that fixed it."""
        flat = per_sample_flat(torch.zeros(shape))
        assert flat.shape[0] == 1
        assert flat.numel() == torch.zeros(shape).numel()

    def test_the_threshold_is_the_image_pair_contract(self):
        """``(B, C, H, W)`` is ``BaseMetric.INPUT_SIGNATURE == "image_pair"``."""
        assert SAMPLE_AXIS_MIN_NDIM == 4

    def test_it_is_a_view_not_a_copy(self):
        """This runs per range-sensitive metric per validation batch."""
        source = torch.zeros(4, 1, 8, 8)
        assert per_sample_flat(source).data_ptr() == source.data_ptr()


class TestPerSamplePeak:
    def test_each_sample_gets_its_own_peak(self):
        flat = torch.tensor([[1.0, 2.0, 3.0], [10.0, 0.5, 0.5]])
        assert per_sample_peak(flat, floor=0.0, empty_fallback=1.0).tolist() == [
            3.0,
            10.0,
        ]

    def test_a_complex_sample_uses_its_magnitude(self):
        flat = torch.tensor([[3 + 4j, 0 + 0j]])
        assert per_sample_peak(flat, floor=0.0, empty_fallback=1.0).tolist() == [5.0]

    def test_an_all_zero_sample_takes_the_fallback_not_zero(self):
        flat = torch.tensor([[0.0, 0.0], [2.0, 1.0]])
        assert per_sample_peak(flat, floor=0.0, empty_fallback=1.0).tolist() == [
            1.0,
            2.0,
        ]

    def test_the_floor_binds_per_sample(self):
        flat = torch.tensor([[0.001, 0.002], [5.0, 1.0]])
        assert per_sample_peak(flat, floor=0.01, empty_fallback=1.0).tolist() == (
            pytest.approx([0.01, 5.0])
        )

    def test_it_does_not_sync_to_the_host(self):
        """The scalar spellings this replaces each called ``.item()`` -- one GPU
        sync per range-sensitive metric per validation batch (non-negotiable 9)."""
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(per_sample_peak)))
        syncs = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"item", "tolist", "cpu", "numpy"}
        ]
        assert syncs == []


class TestAggregationProvenance:
    def test_it_names_both_halves(self):
        """The convention restates recorded numbers, so the artifact has to say
        which convention produced them."""
        record = aggregation_provenance()
        assert record == {
            "psnr_reduction": PSNR_REDUCTION,
            "validation_epoch_weighting": VALIDATION_EPOCH_WEIGHTING,
        }
        assert record["psnr_reduction"] == "per_sample_mean"
        assert record["validation_epoch_weighting"] == "sample"

    def test_it_is_json_serialisable(self):
        import json

        assert json.loads(json.dumps(aggregation_provenance()))
