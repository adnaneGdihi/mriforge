"""fastMRI+ bounding-box annotation parser.

The CSV ships one row per annotated box::

    file,slice,study_level,x,y,width,height,label

The y-origin: the boxes are annotated on a VERTICALLY FLIPPED image
-------------------------------------------------------------------
**The fastMRI+ boxes do not index the raw ``reconstruction_rss`` array directly.**
They were drawn on an up/down-flipped copy of it, so a box's row in *our* (unflipped)
array is ``H - y_raw - height``. That is :data:`ESTABLISHED_Y_ORIGIN`
(:attr:`YOrigin.BOTTOM_LEFT`), and it is the value to pass.

Evidence -- upstream, primary, not inferred. The fastMRI+ example notebook flips the
image before drawing a single box (``ExampleScripts/example.ipynb``)::

    img_data = f['reconstruction_rss'][:]
    img_data = img_data[:, ::-1, :]        # flipped up down
    ...
    plotted_image.rectangle(((x0, y0), (x1, y1)), outline="white")   # PIL: y DOWN

and the upstream README gives the cause: *"In the process of converting the images to
DICOM, the pixel arrays were flipped (up/down) to provide a view that was closer to
DICOM orientation. This should be taken into consideration when using the labels."*
https://github.com/microsoft/fastmri-plus

So ``(x, y)`` is a top-left-origin coordinate **in the flipped frame**, which is
arithmetically identical to a bottom-left origin in the unflipped array: the flip maps
flipped rows ``[y, y+h)`` onto array rows ``[H-y-h, H-1-y]``, whose top row is
``H - y - h``. We move the *box*, never the image -- the image must stay canonical so
that the SynthSeg region masks (segmented from the same unflipped volume) stay aligned
with it.

This was previously pinned to ``TOP_LEFT`` on the strength of a prose assertion with no
citation and no test. The cost of that being wrong is uniquely nasty, and is the reason
for every guard below: a flipped box on an axial brain slice lands on *mirror-image
normal-appearing tissue* -- same size, still brain, still fully in bounds, so
:class:`BoxOutOfBoundsError` never fires. Worse, :mod:`mriforge.data.annotations.matched_control`
mirrors the *control* too, so lesion and control stay internally consistent and
``lesion_vs_control`` yields a clean, tight-CI, **entirely meaningless** table in which
both arms sit on healthy tissue. A crash would have been a kindness.

Three guards therefore stay, even though the answer is now cited:

1. :class:`YOrigin` remains a **required** argument to :func:`parse_annotations` --
   so it is always *explicit* and always *stamped into the manifest's provenance*
   (pitfall #15). A default would let the convention go invisible again the moment
   someone points this parser at a different annotation export.
2. The conversion to array coordinates happens in exactly one place --
   :meth:`LesionBox.top_left_y` -- because it needs the image height, which the CSV
   does not carry. One conversion site, one place for a bug to hide.
3. ``scripts/data/build_fastmri_plus_manifest.py`` scores *both* conventions, **raises**
   if the configured one is not the better-scoring one, and renders the QC overlays
   **side by side under both** so a human can see the choice rather than ratify it.
   That gate also catches failures with nothing to do with the y-origin -- an off-by-one
   slice index, a mis-joined ``file_id`` -- which produce the same plausible-null
   signature.

Collection is total; the gate is elsewhere
------------------------------------------
This parser **never raises on an unmapped label**. It counts every row it saw,
every row it dropped, and every label it could not map, and hands that back as
an :class:`AnnotationParseReport`. The build gate is a single explicit call to
:meth:`AnnotationParseReport.raise_if_unmapped`. Keeping collection total is what
makes the ``--report-labels`` discovery mode possible: you read the label strings
off the real CSV instead of guessing them.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import torch

from mriforge.data.annotations.fastmri_plus_classes import (
    LesionGroup,
    UnmappedLabelError,
    group_for,
)

__all__ = [
    "ESTABLISHED_Y_ORIGIN",
    "AnnotationParseReport",
    "BoxOutOfBoundsError",
    "LesionBox",
    "UnmappedLabelsError",
    "YOrigin",
    "parse_annotations",
    "pooled_lesion_mask",
]

_EXPECTED_COLUMNS = (
    "file",
    "slice",
    "study_level",
    "x",
    "y",
    "width",
    "height",
    "label",
)

# A study-level row carries no box -- "this study shows X, but not on any one slice".
#
# The AUTHORITY is the `study_level` column, which the shipped CSV fills with Yes/No.
# It was previously ignored, in favour of a `-1`-in-every-geometry-column sentinel that
# **does not exist**: the real CSV leaves x/y/width/height *empty*. So all 643 study-level
# rows failed `int("")`, landed in `malformed`, and the parse report -- whose entire
# promise is that no row vanishes uncounted -- reported them under the wrong heading.
#
# Verified 2026-07-13 against microsoft/fastmri-plus @ 67ed9a6: 8,213 rows = 7,570 boxes
# (study_level=No) + 643 study-level (Yes), matching the paper's stated counts.
_STUDY_LEVEL_TRUE = frozenset({"yes", "true", "1"})

# Kept as a secondary guard only: some re-exports do use -1 rather than an empty field,
# and a -1 box would otherwise rasterise as a real ROI at a negative offset.
_STUDY_LEVEL_SENTINEL = -1


class YOrigin(StrEnum):
    """Which edge the CSV's ``y`` is measured from, relative to the **raw** array.

    For fastMRI+ this is :data:`ESTABLISHED_Y_ORIGIN` (``BOTTOM_LEFT``). The enum keeps
    both members because the value must stay an *explicit, stamped* choice -- see the
    module docstring -- and because ``TOP_LEFT`` is what the scoring code has to be
    able to express in order to be *wrong*, which is what the tests check.
    """

    TOP_LEFT = "top_left"
    """``y`` counts downward from row 0 -- the numpy/array convention.

    **Not fastMRI+.** Assuming this silently mirrors every box onto the contralateral
    hemisphere; see the module docstring.
    """

    BOTTOM_LEFT = "bottom_left"
    """``y`` counts upward from the last row: array row ``H - y - height``. **fastMRI+.**

    Equivalently, and this is what actually happened: ``y`` is a top-left coordinate in
    the up/down-**flipped** image that the fastMRI+ radiologists annotated. The two
    descriptions are the same arithmetic.
    """


ESTABLISHED_Y_ORIGIN = YOrigin.BOTTOM_LEFT
"""fastMRI+ boxes index a vertically flipped copy of ``reconstruction_rss``.

Cited, not asserted: upstream's own example notebook applies ``img[:, ::-1, :]``
("flipped up down") before drawing any box, and the upstream README states the pixel
arrays were flipped up/down during DICOM conversion. This repo never flips the image
(``FastMRIH5Strategy``), so the box must be flipped instead:
``array_row = H - y_raw - height``. See the module docstring for the full quote and link.

Corrected 2026-07-13. It was previously ``TOP_LEFT`` -- a prose claim with no citation
and no test -- which would have placed every lesion box on mirror-image healthy tissue
while the matched control mirrored with it, keeping the whole pathology axis green,
self-consistent, and void.

The CSV does not record the convention, so it lives here -- one constant, one place.
The manifest still stamps the resolved value into its provenance block, so a run's
artifacts stay self-describing even if this constant later changes.
"""


class BoxOutOfBoundsError(ValueError):
    """A box that does not intersect the image at all.

    Almost always means the wrong :class:`YOrigin`: a flip pushes boxes near one
    edge clean off the other side.
    """


class UnmappedLabelsError(ValueError):
    """One or more CSV labels had no declared :class:`LesionGroup`."""


@dataclass(frozen=True, slots=True)
class LesionBox:
    """One fastMRI+ box, holding the **raw** CSV coordinates.

    ``y_raw`` is verbatim from the CSV and is meaningless without ``y_origin``.
    Nothing downstream may read ``y_raw`` directly -- go through
    :meth:`top_left_y`, which is the single conversion site.
    """

    file_id: str
    slice_index: int
    x: int
    y_raw: int
    width: int
    height: int
    label: str
    group: LesionGroup
    y_origin: YOrigin

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"box {self.label!r} on {self.file_id} slice {self.slice_index} has "
                f"non-positive extent ({self.width}x{self.height}); a zero-area ROI "
                "is a data bug, not a region to score."
            )

    def top_left_y(self, image_height: int) -> int:
        """Convert ``y_raw`` to the array (top-left) convention.

        The **only** place the y-origin is applied. Under ``BOTTOM_LEFT`` the box
        spans rows ``[y_raw, y_raw + height)`` counted up from the bottom, whose
        top row in array coordinates is ``H - y_raw - height``.
        """
        if self.y_origin is YOrigin.TOP_LEFT:
            return self.y_raw
        return image_height - self.y_raw - self.height

    def to_bool_mask(self, image_height: int, image_width: int) -> torch.Tensor:
        """Rasterise to a ``[H, W]`` bool mask, clipped to the image.

        A box that is *partially* outside is clipped (and the clip is counted by
        the builder -- a high clip rate is itself a y-origin smell). A box that is
        *entirely* outside raises: there is no honest ROI to score, and it is the
        loudest available signal that the origin is wrong.
        """
        y0 = self.top_left_y(image_height)
        y1, x0, x1 = y0 + self.height, self.x, self.x + self.width

        cy0, cy1 = max(0, y0), min(image_height, y1)
        cx0, cx1 = max(0, x0), min(image_width, x1)
        if cy0 >= cy1 or cx0 >= cx1:
            raise BoxOutOfBoundsError(
                f"box {self.label!r} on {self.file_id} slice {self.slice_index} "
                f"maps to rows [{y0}, {y1}) x cols [{x0}, {x1}), which does not "
                f"intersect the {image_height}x{image_width} image under "
                f"y_origin={self.y_origin.value!r}. The other y-origin is the first "
                "thing to check."
            )

        mask = torch.zeros(image_height, image_width, dtype=torch.bool)
        mask[cy0:cy1, cx0:cx1] = True
        return mask

    def is_clipped(self, image_height: int, image_width: int) -> bool:
        """True when the box crosses an image edge."""
        y0 = self.top_left_y(image_height)
        return (
            y0 < 0
            or self.x < 0
            or y0 + self.height > image_height
            or self.x + self.width > image_width
        )


def pooled_lesion_mask(
    boxes: Sequence[LesionBox], image_height: int, image_width: int
) -> torch.Tensor:
    """Union of every box on a slice -- the default ``lesion_any`` region.

    Pooling is the default because per-class regions are underpowered: ~16 raw
    classes x 4 contrasts leaves most per-class cells below 30 lesions.
    """
    if not boxes:
        raise ValueError("cannot pool an empty box list into a region")
    mask = torch.zeros(image_height, image_width, dtype=torch.bool)
    for box in boxes:
        mask |= box.to_bool_mask(image_height, image_width)
    return mask


@dataclass(frozen=True, slots=True)
class AnnotationParseReport:
    """Everything the parse saw, kept or thrown away.

    Every dropped row is counted. A parser that silently skips rows makes the
    annotation count unfalsifiable -- you cannot tell "this cohort has 40 lesions"
    from "this cohort has 300 lesions and the parser ate 260".
    """

    csv_path: Path
    y_origin: YOrigin
    total_rows: int
    boxes: tuple[LesionBox, ...]
    study_level_dropped: int
    malformed: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    labels_seen: Mapping[str, int] = field(default_factory=dict)
    unmapped_labels: Mapping[str, int] = field(default_factory=dict)

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    @property
    def n_files(self) -> int:
        return len({b.file_id for b in self.boxes})

    @property
    def n_annotated_slices(self) -> int:
        return len({(b.file_id, b.slice_index) for b in self.boxes})

    def raise_if_unmapped(self) -> None:
        """The build gate. Called by the manifest builder, never by the parser."""
        if not self.unmapped_labels:
            return
        listing = ", ".join(
            f"{label!r} (x{n})"
            for label, n in sorted(self.unmapped_labels.items(), key=lambda kv: -kv[1])
        )
        raise UnmappedLabelsError(
            f"{len(self.unmapped_labels)} label(s) in {self.csv_path} have no "
            f"declared LesionGroup, covering "
            f"{sum(self.unmapped_labels.values())} annotation row(s): {listing}. "
            "Declare each one in mriforge/data/annotations/fastmri_plus_classes.py. "
            "Building the manifest anyway would drop these pathologies without "
            "trace."
        )

    def by_group(self) -> Mapping[LesionGroup, int]:
        return Counter(b.group for b in self.boxes)

    def to_dict(self) -> dict[str, object]:
        """Provenance block for the manifest."""
        return {
            "csv_path": str(self.csv_path),
            "y_origin": self.y_origin.value,
            "total_rows": self.total_rows,
            "boxes_parsed": self.n_boxes,
            "files": self.n_files,
            "annotated_slices": self.n_annotated_slices,
            "study_level_dropped": self.study_level_dropped,
            "malformed_dropped": len(self.malformed),
            "malformed_detail": [{"line": ln, "reason": why} for ln, why in self.malformed],
            "labels_seen": dict(self.labels_seen),
            "by_group": {g.value: n for g, n in self.by_group().items()},
        }


def _slices_by_file(
    boxes: Iterable[LesionBox],
) -> dict[tuple[str, int], list[LesionBox]]:
    out: dict[tuple[str, int], list[LesionBox]] = {}
    for box in boxes:
        out.setdefault((box.file_id, box.slice_index), []).append(box)
    return out


def group_boxes_by_slice(
    boxes: Iterable[LesionBox],
) -> dict[tuple[str, int], list[LesionBox]]:
    """``(file_id, slice_index) -> [LesionBox]`` -- the lookup the sweep needs."""
    return _slices_by_file(boxes)


def parse_annotations(csv_path: str | Path, *, y_origin: YOrigin) -> AnnotationParseReport:
    """Parse a fastMRI+ annotation CSV.

    Args:
        csv_path: The fastMRI+ ``brain.csv`` / ``knee.csv``.
        y_origin: **Required.** Which edge ``y`` is measured from. Determine it
            empirically -- see the module docstring.

    Returns:
        An :class:`AnnotationParseReport`. This function does **not** raise on an
        unmapped label; call :meth:`AnnotationParseReport.raise_if_unmapped` to
        gate the build.
    """
    path = Path(csv_path)
    boxes: list[LesionBox] = []
    malformed: list[tuple[int, str]] = []
    labels_seen: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    study_level = 0
    total = 0

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _EXPECTED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is missing expected fastMRI+ column(s) {missing}; "
                f"found {reader.fieldnames}. Refusing to guess the layout."
            )

        for lineno, row in enumerate(reader, start=2):  # start=2: line 1 is the header
            total += 1
            label = (row["label"] or "").strip()
            if label:
                labels_seen[label] += 1

            # Study-level rows carry no box, and the CSV SAYS SO in its own column.
            # Check it BEFORE parsing geometry: the real file leaves x/y/w/h empty, so
            # `int("")` would raise and the row would be filed as "malformed" -- a true
            # statement about the geometry and a false one about the row.
            #
            # Expanding a study-level row into a whole-slice box would be worse still:
            # that is just the brain region wearing a lesion's name, answering a
            # different question while looking like this one.
            if (row.get("study_level") or "").strip().casefold() in _STUDY_LEVEL_TRUE:
                study_level += 1
                continue

            try:
                x, y, w, h = (
                    int(row["x"]),
                    int(row["y"]),
                    int(row["width"]),
                    int(row["height"]),
                )
                slice_index = int(row["slice"])
            except (TypeError, ValueError):
                malformed.append((lineno, f"non-integer geometry: {row}"))
                continue

            # Secondary guard: a re-export that encodes "no box" as -1 rather than an
            # empty field. Without this, a -1 box rasterises as a real ROI at a negative
            # offset.
            if _STUDY_LEVEL_SENTINEL in (x, y, w, h):
                study_level += 1
                continue

            if not label:
                malformed.append((lineno, "empty label"))
                continue

            try:
                group = group_for(label)
            except UnmappedLabelError:
                unmapped[label] += 1
                continue

            try:
                boxes.append(
                    LesionBox(
                        file_id=row["file"].strip(),
                        slice_index=slice_index,
                        x=x,
                        y_raw=y,
                        width=w,
                        height=h,
                        label=label,
                        group=group,
                        y_origin=y_origin,
                    )
                )
            except ValueError as exc:
                malformed.append((lineno, str(exc)))

    return AnnotationParseReport(
        csv_path=path,
        y_origin=y_origin,
        total_rows=total,
        boxes=tuple(boxes),
        study_level_dropped=study_level,
        malformed=tuple(malformed),
        labels_seen=dict(labels_seen),
        unmapped_labels=dict(unmapped),
    )
