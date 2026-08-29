"""The region axis: one place that says which regions exist and what they mean.

Every consumer -- the eligibility gate, the leaderboard cube, the figure tree, the
xlsx -- resolves regions from here. A new region is one entry in ``REGION_SPECS``,
not an edit to ``figure_manifest.py`` and three table writers.

Region tiers
------------
``full``
    The whole slice. The identity region, and the **control condition** for the
    entire study: without it there is nothing for a regional ranking to diverge
    *from*, and the region axis stops being a comparison.
``brain``
    Everything SynthSeg did not call background. Note this is already a
    non-trivial restriction -- background and air are roughly 40% of a 320x320
    fastMRI brain slice, so a global metric spends 40% of its average on
    whether the *air* was reconstructed well.
``tissue:*``
    SynthSeg-derived. Non-rectangular, so crop-tier metrics are ineligible on
    them (see ``core/metrics/regions/eligibility.py``).
``path:*``
    fastMRI+ pathology boxes. Rectangular by construction.
``control:*``
    Size-matched contralateral-mirror controls. Rectangular by construction.

Why ``control:`` exists at all
------------------------------
A lesion ROI is small; the brain is large. Any rank shift between ``full`` and
``path:lesion_any`` could be a **size** effect rather than a **lesion** effect --
pitfall #17 (confounded ablation) applied to the region axis. The matched control
is the same size, in normal-appearing tissue, so ``lesion_vs_control`` isolates
the one variable that matters. It is the only comparison that answers the actual
scientific question.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from mriforge.data.regions.synthseg_labels import TissueClass

__all__ = [
    "REGION_SPECS",
    "RegionSpec",
    "RegionTier",
    "control_id_for",
    "is_control_id",
    "lesion_region_id",
    "region_spec",
    "tissue_region_ids",
]


class RegionTier(StrEnum):
    """Which machinery produces the region -- and therefore what can gate it."""

    IDENTITY = "identity"
    ANATOMY = "anatomy"
    TISSUE = "tissue"
    PATHOLOGY = "pathology"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class RegionSpec:
    region_id: str
    tier: RegionTier
    description: str
    tissues: tuple[TissueClass, ...] = ()
    rectangular: bool = False
    """True only when the region is a filled box.

    Load-bearing: a crop-tier metric (LPIPS, NIQE, ...) on a NON-rectangular
    region would score the region's *bounding box* -- which for grey matter
    contains white matter, CSF and background -- and report it as "LPIPS on grey
    matter". The eligibility gate reads this to decline instead.
    """

    requires_synthseg: bool = False
    requires_annotations: bool = False


FULL = "full"
BRAIN = "brain"

_SPECS: tuple[RegionSpec, ...] = (
    RegionSpec(
        region_id=FULL,
        tier=RegionTier.IDENTITY,
        description="The whole slice, background included. The control condition.",
        rectangular=True,
    ),
    RegionSpec(
        region_id=BRAIN,
        tier=RegionTier.ANATOMY,
        description=(
            "Non-background. Already a real restriction: air/background is ~40% of "
            "a fastMRI brain slice, so a global metric spends 40% of its average "
            "on the air."
        ),
        tissues=tuple(t for t in TissueClass if t is not TissueClass.BACKGROUND),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="tissue:gm",
        tier=RegionTier.TISSUE,
        description="Cortical + subcortical + cerebellar grey matter.",
        tissues=(
            TissueClass.CORTICAL_GM,
            TissueClass.SUBCORTICAL_GM,
            TissueClass.CEREBELLAR_GM,
        ),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="tissue:wm",
        tier=RegionTier.TISSUE,
        description="Cerebral + cerebellar white matter.",
        tissues=(TissueClass.CEREBRAL_WM, TissueClass.CEREBELLAR_WM),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="tissue:csf",
        tier=RegionTier.TISSUE,
        description="Ventricles and extra-ventricular CSF.",
        tissues=(TissueClass.CSF,),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="tissue:cortical_gm",
        tier=RegionTier.TISSUE,
        description="Cortical ribbon only -- the thin, high-curvature structure.",
        tissues=(TissueClass.CORTICAL_GM,),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="tissue:subcortical_gm",
        tier=RegionTier.TISSUE,
        description="Deep grey nuclei -- compact and central.",
        tissues=(TissueClass.SUBCORTICAL_GM,),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="tissue:brainstem",
        tier=RegionTier.TISSUE,
        description="Brainstem.",
        tissues=(TissueClass.BRAINSTEM,),
        requires_synthseg=True,
    ),
    RegionSpec(
        region_id="path:lesion_any",
        tier=RegionTier.PATHOLOGY,
        description=(
            "Union of every fastMRI+ box on the slice. The DEFAULT pathology "
            "region: ~16 classes x 4 contrasts leaves most per-class cells below "
            "30 lesions, so a per-class default would be underpowered by "
            "construction."
        ),
        rectangular=False,  # a union of boxes need not be a box
        requires_annotations=True,
    ),
    RegionSpec(
        region_id="control:lesion_any",
        tier=RegionTier.CONTROL,
        description=(
            "Size-matched contralateral-mirror control for path:lesion_any, in "
            "normal-appearing tissue. Without it, a rank shift at the lesion could "
            "be a SIZE effect rather than a LESION effect (pitfall #17)."
        ),
        rectangular=False,
        requires_annotations=True,
        requires_synthseg=True,
    ),
)

REGION_SPECS: Mapping[str, RegionSpec] = {s.region_id: s for s in _SPECS}


def region_spec(region_id: str) -> RegionSpec:
    """Look up a region, or raise. No default -- an undeclared region is a typo."""
    try:
        return REGION_SPECS[region_id]
    except KeyError:
        raise KeyError(
            f"unknown region {region_id!r}. Declared regions: "
            f"{sorted(REGION_SPECS)}. Declare new ones in "
            "mriforge/data/regions/region_registry.py -- it is the region-axis SSOT, "
            "so the figure tree and the leaderboard pick them up for free."
        ) from None


def tissue_region_ids() -> tuple[str, ...]:
    """Regions that need the SynthSeg cache. Skipped wholesale when it is unsound."""
    return tuple(s.region_id for s in _SPECS if s.tier is RegionTier.TISSUE)


def lesion_region_id(group: str | None = None) -> str:
    """``path:lesion_any`` (default) or the opt-in per-group region."""
    return "path:lesion_any" if group is None else f"path:lesion_{group}"


def control_id_for(lesion_region_id_: str) -> str:
    """The matched control paired with a pathology region.

    One naming rule, in one place: a mismatch here would silently pair a lesion
    with the *wrong* control, and the paired test would still produce a number.
    """
    if not lesion_region_id_.startswith("path:"):
        raise ValueError(
            f"{lesion_region_id_!r} is not a pathology region; only pathology "
            "regions get matched controls."
        )
    return "control:" + lesion_region_id_.removeprefix("path:")


def is_control_id(region_id: str) -> bool:
    return region_id.startswith("control:")
