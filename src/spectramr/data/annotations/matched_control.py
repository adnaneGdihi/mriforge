"""Size-matched contralateral-mirror controls for fastMRI+ lesion boxes.

Why this exists
---------------
A lesion ROI is ~40x40 px; the brain is ~200x200. If the metric leaderboard
inside ``path:lesion_any`` differs from the leaderboard on the whole slice, there
are two candidate explanations and the study cannot tell them apart:

* the metrics genuinely behave differently **on pathology**; or
* the metrics behave differently **on any small region**, and the lesion is small.

That is pitfall #17 (confounded ablation) applied to the region axis. The fix is
the standard one: hold size constant and vary only the thing you claim to be
testing. Every lesion box gets a **same-size** box in normal-appearing
contralateral tissue, and ``lesion_vs_control`` is the only comparison that
answers the actual question.

An unpaired lesion is dropped
-----------------------------
If no valid control can be placed (the lesion is midline, or the mirror lands on
another lesion, or it falls off the brain), **the lesion is excluded too** -- and
the drop is counted. Keeping it would silently re-introduce the very size confound
the control exists to remove: the lesion pool would then contain regions with no
size-matched counterpart, and the paired comparison would quietly become unpaired
for part of the cohort.

Determinism
-----------
The seed is ``blake2b(file_id | slice | annotation_id | global_seed)`` -- a pure
function of *what is being sampled*, never of iteration order or worker count. An
``enumerate()``-derived seed would give a different control depending on how many
dataloader workers happened to be running, which makes the study irreproducible in
a way that is almost impossible to notice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from spectramr.data.annotations.fastmri_plus import LesionBox

__all__ = [
    "DEFAULT_LESION_CLEARANCE_PX",
    "DEFAULT_MIN_BRAIN_COVERAGE",
    "MatchedControl",
    "MatchedControlReport",
    "sample_matched_control",
    "sample_matched_controls",
]

DEFAULT_MIN_BRAIN_COVERAGE = 0.95
"""A control must be ~entirely inside the brain.

A control that is 60% air would make the "normal tissue" baseline mostly a
background patch -- on which every metric agrees, because there is nothing there.
That would manufacture a rank difference between lesion and control that is really
a difference between tissue and air.
"""

DEFAULT_LESION_CLEARANCE_PX = 2
"""Dilate every lesion by this much before testing control overlap.

Zero-overlap is not enough: a control abutting a lesion picks up its edge
gradient, and the artifact's effect on that edge is exactly what the lesion ROI is
supposed to be measuring. A 2 px cordon keeps the control genuinely normal.
"""

_MAX_REJECTION_ATTEMPTS = 64


@dataclass(frozen=True, slots=True)
class MatchedControl:
    """A control box, same size as its lesion, in normal-appearing tissue."""

    region_id: str
    paired_with: str
    x: int
    y: int  # already in TOP-LEFT array coordinates
    width: int
    height: int
    brain_coverage: float
    strategy: str  # "contralateral_mirror" | "rejection_sample"
    seed: int

    def to_bool_mask(self, image_height: int, image_width: int) -> torch.Tensor:
        mask = torch.zeros(image_height, image_width, dtype=torch.bool)
        mask[self.y : self.y + self.height, self.x : self.x + self.width] = True
        return mask

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "paired_with": self.paired_with,
            "strategy": self.strategy,
            "brain_coverage": round(self.brain_coverage, 4),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class MatchedControlReport:
    """Which lesions got a control, and which were dropped for lack of one."""

    controls: tuple[MatchedControl, ...] = field(default_factory=tuple)
    dropped_lesions: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """``(lesion_region_id, why)``. These lesions are excluded from the study."""

    @property
    def n_paired(self) -> int:
        return len(self.controls)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_lesions)

    def to_dict(self) -> dict[str, object]:
        return {
            "paired": self.n_paired,
            "dropped_unpaired": self.n_dropped,
            "dropped_detail": [{"lesion": lid, "reason": why} for lid, why in self.dropped_lesions],
            "by_strategy": {
                s: sum(c.strategy == s for c in self.controls)
                for s in {c.strategy for c in self.controls}
            },
        }


def _seed_for(file_id: str, slice_index: int, annotation_id: str, global_seed: int) -> int:
    """A pure function of what is sampled -- never of iteration order."""
    key = f"{file_id}|{slice_index}|{annotation_id}|{global_seed}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    padded = mask[None, None].float()
    grown = torch.nn.functional.max_pool2d(padded, kernel_size=k, stride=1, padding=radius)
    return grown[0, 0] > 0


def _accept(
    x: int,
    y: int,
    w: int,
    h: int,
    brain: torch.Tensor,
    forbidden: torch.Tensor,
    *,
    min_brain_coverage: float,
) -> tuple[bool, float, str]:
    """The three acceptance criteria, all of which must hold."""
    ih, iw = brain.shape
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        return False, 0.0, "out_of_bounds"

    patch_brain = brain[y : y + h, x : x + w]
    coverage = float(patch_brain.sum()) / float(w * h)
    if coverage < min_brain_coverage:
        return False, coverage, "insufficient_brain_coverage"

    if bool(forbidden[y : y + h, x : x + w].any()):
        return False, coverage, "overlaps_a_lesion"

    return True, coverage, "ok"


def sample_matched_control(
    box: LesionBox,
    *,
    brain: torch.Tensor,
    lesion_union: torch.Tensor,
    global_seed: int,
    min_brain_coverage: float = DEFAULT_MIN_BRAIN_COVERAGE,
    lesion_clearance_px: int = DEFAULT_LESION_CLEARANCE_PX,
) -> tuple[MatchedControl | None, str]:
    """Place one control. Returns ``(control, reason)``; ``control`` is ``None`` on failure.

    Args:
        box: The lesion to match.
        brain: ``[H, W]`` bool brain mask from the SynthSeg cache (clean reference).
        lesion_union: ``[H, W]`` bool union of **every** lesion on this slice --
            not just ``box``. A control must avoid all of them, or it is not a
            control.
        global_seed: Study-level seed, mixed into the per-box seed.

    Strategy: mirror about the brain's column centroid first (an anatomically
    meaningful counterpart), and fall back to seeded rejection sampling only if the
    mirror is invalid.
    """
    h_img, w_img = brain.shape
    y = box.top_left_y(h_img)
    w, h = box.width, box.height
    forbidden = _dilate(lesion_union, lesion_clearance_px)

    annotation_id = f"{box.x},{box.y_raw},{box.width},{box.height},{box.label}"
    seed = _seed_for(box.file_id, box.slice_index, annotation_id, global_seed)
    region_id = f"control:{box.file_id}#{box.slice_index}#{box.x},{box.y_raw}"
    paired_with = f"path:{box.file_id}#{box.slice_index}#{box.x},{box.y_raw}"

    if not bool(brain.any()):
        return None, "empty_brain_mask"

    # 1. Contralateral mirror about the brain's column centroid.
    cols = torch.nonzero(brain.any(dim=0)).flatten()
    centroid = float(cols.float().mean())
    mirrored_x = round(2 * centroid - (box.x + w))

    ok, coverage, why = _accept(
        mirrored_x, y, w, h, brain, forbidden, min_brain_coverage=min_brain_coverage
    )
    if ok:
        return (
            MatchedControl(
                region_id=region_id,
                paired_with=paired_with,
                x=mirrored_x,
                y=y,
                width=w,
                height=h,
                brain_coverage=coverage,
                strategy="contralateral_mirror",
                seed=seed,
            ),
            "ok",
        )

    # 2. Seeded rejection sampling anywhere in normal-appearing tissue.
    gen = torch.Generator().manual_seed(seed % (2**63))
    last = why
    for _ in range(_MAX_REJECTION_ATTEMPTS):
        cx = int(torch.randint(0, max(1, w_img - w), (1,), generator=gen).item())
        cy = int(torch.randint(0, max(1, h_img - h), (1,), generator=gen).item())
        ok, coverage, last = _accept(
            cx, cy, w, h, brain, forbidden, min_brain_coverage=min_brain_coverage
        )
        if ok:
            return (
                MatchedControl(
                    region_id=region_id,
                    paired_with=paired_with,
                    x=cx,
                    y=cy,
                    width=w,
                    height=h,
                    brain_coverage=coverage,
                    strategy="rejection_sample",
                    seed=seed,
                ),
                "ok",
            )

    return None, f"no_valid_control_after_{_MAX_REJECTION_ATTEMPTS}_attempts:{last}"


def sample_matched_controls(
    boxes: Sequence[LesionBox],
    *,
    brain: torch.Tensor,
    global_seed: int,
    min_brain_coverage: float = DEFAULT_MIN_BRAIN_COVERAGE,
    lesion_clearance_px: int = DEFAULT_LESION_CLEARANCE_PX,
) -> MatchedControlReport:
    """Pair every lesion on a slice, dropping the ones that cannot be paired.

    A lesion with no control is **excluded from the study**, not kept unpaired --
    see the module docstring.
    """
    if not boxes:
        return MatchedControlReport()

    h_img, w_img = brain.shape
    union = torch.zeros(h_img, w_img, dtype=torch.bool)
    for b in boxes:
        union |= b.to_bool_mask(h_img, w_img)

    controls: list[MatchedControl] = []
    dropped: list[tuple[str, str]] = []
    for b in boxes:
        control, why = sample_matched_control(
            b,
            brain=brain,
            lesion_union=union,
            global_seed=global_seed,
            min_brain_coverage=min_brain_coverage,
            lesion_clearance_px=lesion_clearance_px,
        )
        if control is None:
            dropped.append((f"path:{b.file_id}#{b.slice_index}#{b.x},{b.y_raw}", why))
        else:
            controls.append(control)

    return MatchedControlReport(controls=tuple(controls), dropped_lesions=tuple(dropped))
