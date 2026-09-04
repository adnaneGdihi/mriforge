"""Annotation ingest (fastMRI+ pathology bounding boxes).

Annotations are a **data** concern, so parsing lives here under
``src/spectramr/data/`` rather than in ``scripts/sim2rank/`` -- ``src/`` may never
import ``scripts/`` (CLAUDE.md non-negotiable #5), and the sweep, the manifest
generator and the QC renderer all need the same parser.
"""

from __future__ import annotations

from spectramr.data.annotations.fastmri_plus import (
    ESTABLISHED_Y_ORIGIN,
    AnnotationParseReport,
    BoxOutOfBoundsError,
    LesionBox,
    UnmappedLabelsError,
    YOrigin,
    group_boxes_by_slice,
    parse_annotations,
    pooled_lesion_mask,
)
from spectramr.data.annotations.fastmri_plus_classes import (
    LesionGroup,
    UnmappedLabelError,
    declared_labels,
    group_for,
)
from spectramr.data.annotations.manifest_qc import (
    ApprovalDigestMismatchError,
    approval_digest,
    verify_approval,
)

__all__ = [
    "ESTABLISHED_Y_ORIGIN",
    "AnnotationParseReport",
    "ApprovalDigestMismatchError",
    "BoxOutOfBoundsError",
    "LesionBox",
    "LesionGroup",
    "UnmappedLabelError",
    "UnmappedLabelsError",
    "YOrigin",
    "approval_digest",
    "declared_labels",
    "group_boxes_by_slice",
    "group_for",
    "parse_annotations",
    "pooled_lesion_mask",
    "verify_approval",
]
