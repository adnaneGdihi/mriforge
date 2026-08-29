"""Sim2RankSlice: the invariants that keep a region attached to the right pixels."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.data.sources.sim2rank_source import (  # noqa: E402
    Sim2RankSlice,
    Sim2RankVolumeSource,
)


def _slice(**kw) -> Sim2RankSlice:
    defaults = {
        "content_id": "f#4",
        "clean": torch.rand(1, 32, 32),
        "contrast": "AXT2",
        "file_id": "f",
        "slice_index": 4,
    }
    return Sim2RankSlice(**{**defaults, **kw})


class TestSliceWiseInvariant:
    def test_a_volume_is_rejected(self) -> None:
        """The digital-twin simulator is slice-wise. A [S, H, W] tensor here would
        be silently degraded as a BATCH of independent slices -- each getting its
        own motion trajectory, its own noise draw -- which is not what a volume is."""
        with pytest.raises(ValueError, match=r"must be \[1, H, W\]"):
            _slice(clean=torch.rand(16, 32, 32))

    def test_a_bare_2d_slice_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be \[1, H, W\]"):
            _slice(clean=torch.rand(32, 32))


class TestRegionGeometryIsChecked:
    def test_a_mask_from_another_geometry_is_rejected(self) -> None:
        """Regions are computed on the clean reference. A geometry mismatch means
        the mask belongs to a different slice, and resampling it would move the
        region boundary."""
        with pytest.raises(ValueError, match="another geometry"):
            _slice(region_masks={"full": torch.ones(64, 64, dtype=torch.bool)})

    def test_a_float_mask_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool mask"):
            _slice(region_masks={"full": torch.ones(32, 32)})

    def test_matching_masks_are_accepted(self) -> None:
        s = _slice(
            region_masks={
                "full": torch.ones(32, 32, dtype=torch.bool),
                "path:lesion_any": torch.zeros(32, 32, dtype=torch.bool).index_fill_(
                    0, torch.tensor([1, 2]), True
                ),
            }
        )
        assert set(s.region_masks) == {"full", "path:lesion_any"}


class TestProtocol:
    def test_a_minimal_source_satisfies_the_protocol(self) -> None:
        class Fake:
            name = "fake"
            contrasts = ("AXT2",)

            def __iter__(self):
                yield _slice()

            def __len__(self) -> int:
                return 1

        assert isinstance(Fake(), Sim2RankVolumeSource)
        assert next(iter(Fake())).contrast == "AXT2"
