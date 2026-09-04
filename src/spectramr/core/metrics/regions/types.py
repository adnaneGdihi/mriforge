"""Region masks: the geometry a metric is restricted to.

A :class:`RegionMask` is a boolean support on one slice, plus enough provenance to
say *where it came from*. Regions are computed on the **clean reference only** --
re-segmenting each degraded frame would let the region boundary move with the
artifact, so you would be measuring "how much did the artifact confuse the
segmenter" rather than "how well does this metric track degradation inside this
region".

Everything here validates and raises. An empty region is a data bug, not a region
to score: a metric averaged over zero pixels is 0/0, and the one thing this package
exists to prevent is a number that was never measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "FULL_REGION_ID",
    "RegionMask",
    "RegionSet",
    "RegionSource",
]

FULL_REGION_ID = "full"


class RegionSource:
    """Where a region came from. Free-form, but these are the known producers."""

    FULL = "full"  # the whole slice (the identity region)
    BRAIN = "brain"  # foreground / brain mask
    PATHOLOGY = "fastmri_plus"  # a fastMRI+ annotation bbox
    CONTROL = "matched_control"  # the size-matched contralateral control
    TISSUE = "synthseg"  # GM / WM / CSF parcellation
    STRUCTURAL = "structural"  # anatomical tier


@dataclass(frozen=True, slots=True)
class RegionMask:
    """A boolean support on one ``[H, W]`` slice.

    Args:
        mask: bool tensor ``[H, W]``. **Must contain at least one True.**
        region_id: stable identifier, e.g. ``"path:wm_lesion#0"`` or ``"synthseg:gm"``.
        source: one of :class:`RegionSource`.
        provenance: how this mask was produced. Required for every non-full region
            -- a mask with no provenance cannot be reproduced, and an ROI you cannot
            reproduce is not a result.
    """

    mask: torch.Tensor
    region_id: str
    source: str
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("a RegionMask needs a region_id")
        if self.mask.ndim != 2:
            raise ValueError(f"{self.region_id}: mask must be [H, W], got {tuple(self.mask.shape)}")
        if self.mask.dtype != _bool_dtype():
            raise TypeError(
                f"{self.region_id}: mask must be a bool tensor, got {self.mask.dtype}. "
                "A float mask invites silent partial weighting -- threshold it upstream."
            )
        if not bool(self.mask.any()):
            raise ValueError(
                f"{self.region_id}: region is empty. A metric averaged over zero "
                "pixels is 0/0 -- an empty region is a data bug, not a region to score."
            )
        if self.source != RegionSource.FULL and not self.provenance:
            raise ValueError(
                f"{self.region_id}: non-full regions need provenance (an ROI you "
                "cannot reproduce is not a result)."
            )

    # -- geometry ---------------------------------------------------------

    @property
    def n_px(self) -> int:
        return int(self.mask.sum().item())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Tight ``(y0, x0, y1, x1)`` bounding box, ``y1``/``x1`` exclusive."""
        rows = self.mask.any(dim=1).nonzero().flatten()
        cols = self.mask.any(dim=0).nonzero().flatten()
        return (
            int(rows[0]),
            int(cols[0]),
            int(rows[-1]) + 1,
            int(cols[-1]) + 1,
        )

    @property
    def bbox_shape(self) -> tuple[int, int]:
        y0, x0, y1, x1 = self.bbox
        return (y1 - y0, x1 - x0)

    @property
    def min_side(self) -> int:
        """Shortest side of the tight bbox.

        Distinct from ``n_px``: a 1x3000 sliver has a large area and no usable
        neighbourhood. Metrics with a receptive field need this; metrics fitting
        patch statistics need the area. Both floors exist because neither implies
        the other.
        """
        h, w = self.bbox_shape
        return min(h, w)

    @property
    def is_rectangular(self) -> bool:
        """True when the mask fills its bbox exactly.

        Load-bearing for the crop tier: for a non-rectangular region (GM, WM) the
        bbox is a strict *superset* of the region, so cropping to it computes the
        metric over the bbox -- not over GM. Crop-tier metrics are therefore
        ineligible on non-rectangular regions rather than quietly scoring the wrong
        support.
        """
        return self.n_px == self.bbox_shape[0] * self.bbox_shape[1]

    def to(self, device: object) -> RegionMask:
        return RegionMask(
            mask=self.mask.to(device),
            region_id=self.region_id,
            source=self.source,
            provenance=self.provenance,
        )


def _bool_dtype():  # small indirection so the module imports without torch at doc time
    import torch

    return torch.bool


@dataclass(frozen=True, slots=True)
class RegionSet:
    """Every region scored for one slice. Always contains the ``full`` identity region.

    The ``full`` region is what makes the region axis a *comparison*: without a
    whole-slice control there is nothing to say the lesion-ROI ranking diverges
    *from*.
    """

    regions: tuple[RegionMask, ...]

    def __post_init__(self) -> None:
        ids = [r.region_id for r in self.regions]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate region_id(s): {dupes}")
        if FULL_REGION_ID not in ids:
            raise ValueError(
                f"a RegionSet must contain the {FULL_REGION_ID!r} identity region -- "
                "it is the control the regional rankings are compared against."
            )

    def __iter__(self):
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, region_id: str) -> RegionMask:
        for r in self.regions:
            if r.region_id == region_id:
                return r
        raise KeyError(f"no region {region_id!r} (have: {[r.region_id for r in self]})")

    @staticmethod
    def full_region(height: int, width: int) -> RegionMask:
        """The whole-slice identity region."""
        import torch

        return RegionMask(
            mask=torch.ones(height, width, dtype=torch.bool),
            region_id=FULL_REGION_ID,
            source=RegionSource.FULL,
        )
