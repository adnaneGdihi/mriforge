"""T2 -- the load-bearing invariant: the map tier did not invent a new metric.

For every map-tier metric::

    reduce(maps(pred, target), ones_mask)  ==  registry_metric(pred, target)

If this fails, the region tier is computing a *different* metric under the same
name -- and nobody would notice, because the numbers would still look plausible.
"Region-restricted SSIM" that is silently not SSIM is the exact facade this whole
package exists to prevent (pitfall #16).

This is where a region-local normalisation would sneak in: PSNR's peak, SSIM's
``data_range`` (a static assumption, NOT the data max), NMSE's denominator.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.regions.reductions import (  # noqa: E402
    MAP_REDUCERS,
    masked_mean,
)
from spectramr.core.metrics.registry import get_metric  # noqa: E402


@pytest.fixture
def pair():
    g = torch.Generator().manual_seed(11)
    target = torch.rand(1, 1, 64, 64, generator=g)
    pred = (target + 0.08 * torch.randn(1, 1, 64, 64, generator=g)).clamp(0, 1)
    return pred, target


# The T2 identity has to be checked on shapes where it can actually FAIL. Every
# fixture here used to be [1, 1, 64, 64] -- and B*C == 1 there, which is exactly the
# one shape on which a mask that is summed as [1, 1, H, W] while the values broadcast
# over [B, C, H, W] gives the right answer by accident. On [4, 2, H, W] the map tier
# returned an SSIM of 7.62. C=2 is the ordinary interleaved real/imag layout and any
# batched evaluation has B>1, so the passing shape was the unrepresentative one.
_SHAPES = [(1, 1, 64, 64), (2, 1, 64, 64), (1, 2, 64, 64), (3, 2, 64, 64)]


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("key", sorted(MAP_REDUCERS))
def test_map_reduce_over_ones_equals_the_unrestricted_metric(key, shape) -> None:
    """T2. An all-ones mask must reproduce the registry metric exactly, at ANY shape."""
    g = torch.Generator().manual_seed(11)
    target = torch.rand(*shape, generator=g)
    pred = (target + 0.08 * torch.randn(*shape, generator=g)).clamp(0, 1)

    reducer = MAP_REDUCERS[key]
    ones = torch.ones(shape[-2], shape[-1], dtype=torch.bool)

    restricted = reducer.reduce(reducer.maps(pred, target), ones)
    reference = float(get_metric(key)(pred, target))

    assert restricted == pytest.approx(reference, rel=1e-4, abs=1e-5), (
        f"{key} at {shape}: map-then-mask over an all-ones mask gives {restricted}, "
        f"but the unrestricted metric gives {reference}. The map tier is computing a "
        f"DIFFERENT metric under the same name."
    )


def test_masked_mean_denominator_counts_batch_and_channel() -> None:
    """The mean of a constant is that constant -- at every shape.

    Regression: the denominator counted only the mask's H*W pixels while the numerator
    broadcast over B and C, so the result was inflated by exactly B*C (a constant 3.0
    over a [4, 2, 8, 8] tensor came back as 24.0).
    """
    for b, c in [(1, 1), (2, 1), (1, 2), (4, 2)]:
        values = torch.full((b, c, 8, 8), 3.0)
        mask = torch.ones(8, 8, dtype=torch.bool)
        got = float(masked_mean(values, mask))
        assert got == pytest.approx(3.0), (
            f"masked_mean of a constant 3.0 at [{b},{c},8,8] gave {got} (inflated by B*C = {b * c})"
        )


@pytest.mark.parametrize("key", sorted(MAP_REDUCERS))
def test_restricting_to_a_subregion_changes_the_value(key, pair) -> None:
    """A region restriction that changes nothing is not a restriction."""
    pred, target = pair
    reducer = MAP_REDUCERS[key]
    maps = reducer.maps(pred, target)

    ones = torch.ones(64, 64, dtype=torch.bool)
    quadrant = torch.zeros(64, 64, dtype=torch.bool)
    quadrant[:32, :32] = True

    assert reducer.reduce(maps, ones) != pytest.approx(reducer.reduce(maps, quadrant), rel=1e-6)


class TestMaskedMean:
    def test_reduces_over_the_mask_only(self) -> None:
        values = torch.zeros(1, 1, 4, 4)
        values[0, 0, 0, 0] = 4.0
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[0, 0] = True
        # Over the single masked pixel the mean is 4.0, not 4/16.
        assert float(masked_mean(values, mask)) == pytest.approx(4.0)

    def test_an_empty_mask_raises_rather_than_returning_zero(self) -> None:
        """0/0 is not 0. An empty region is a data bug, not a score of zero."""
        with pytest.raises(ValueError, match="empty mask"):
            masked_mean(torch.ones(1, 1, 4, 4), torch.zeros(4, 4, dtype=torch.bool))


class TestNormalisationComesFromTheFullSlice:
    """The trap: a region-local normalisation makes regions incomparable."""

    def test_psnr_peak_is_not_taken_from_the_roi(self) -> None:
        """A dark ROI has a small local max.

        Deriving the peak from it would inflate that region's PSNR for every metric
        and every artifact -- a region effect manufactured out of the FOV.
        """
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, :16, :] = 1.0  # bright half
        target[0, 0, 16:, :] = 0.05  # dark half
        pred = target + 0.01

        reducer = MAP_REDUCERS["psnr"]
        maps = reducer.maps(pred, target)
        # The peak map is a full-slice scalar: identical regardless of the region.
        dark = torch.zeros(32, 32, dtype=torch.bool)
        dark[16:, :] = True
        bright = torch.zeros(32, 32, dtype=torch.bool)
        bright[:16, :] = True

        # Same absolute error in both halves => same PSNR, because the peak is shared.
        assert reducer.reduce(maps, dark) == pytest.approx(reducer.reduce(maps, bright), rel=1e-4)

    def test_ssim_data_range_is_the_registry_static_assumption(self) -> None:
        """SSIM's data_range is 1.0 (or 2.0 with negatives), NOT the data max.

        Recomputing it per-ROI would give different C1/C2 constants per region,
        making "SSIM on GM" and "SSIM on the lesion bbox" incomparable.
        """
        g = torch.Generator().manual_seed(3)
        target = 0.2 * torch.rand(1, 1, 64, 64, generator=g)  # max well below 1.0
        pred = (target + 0.02 * torch.randn(1, 1, 64, 64, generator=g)).clamp(0, 1)

        reducer = MAP_REDUCERS["ssim"]
        ones = torch.ones(64, 64, dtype=torch.bool)
        restricted = reducer.reduce(reducer.maps(pred, target), ones)
        # If we had used data_range = target.max() (~0.2) the constants would differ
        # and this would not match the registry metric.
        assert restricted == pytest.approx(float(get_metric("ssim")(pred, target)), rel=1e-4)


class TestComplexInputs:
    def test_the_error_is_the_complex_difference_not_the_magnitude_difference(
        self,
    ) -> None:
        """|p - t|, never |p| - |t|.

        Taking magnitudes first would make a pure phase error invisible: two complex
        images with identical magnitude but opposite phase would score a perfect MSE
        of 0.
        """
        target = torch.ones(1, 1, 8, 8, dtype=torch.complex64)
        pred = -target  # same magnitude, opposite phase
        ones = torch.ones(8, 8, dtype=torch.bool)

        reducer = MAP_REDUCERS["mse"]
        mse = reducer.reduce(reducer.maps(pred, target), ones)
        # |1 - (-1)|^2 = 4. If we had used |p| - |t| this would be 0.
        assert mse == pytest.approx(4.0)
