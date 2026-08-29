"""Bind a manifest's QC approval to the thing that was actually approved.

``qc.approved = true`` means "a human looked at the overlays and confirmed the boxes
land on pathology". That claim is *about a specific y-origin*. Nothing stopped someone
from approving a manifest and then hand-editing ``qc.y_origin`` -- which silently flips
every box onto the contralateral hemisphere while the approval stamp stays green.

So the approval carries a digest of the facts it was made about. The source recomputes
it and refuses a manifest whose approval no longer describes its contents.

This is a tamper/drift guard, not a proof of diligence: nothing can force a human to
open the PNGs. What it does guarantee is that an approval cannot be *transplanted* onto
a different configuration.

Lives in ``src/`` (not beside the builder in ``scripts/``) because both the builder and
:mod:`mriforge.data.sources.fastmri_brain_source` need it, and ``src/`` may never import
from ``scripts/``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

__all__ = ["ApprovalDigestMismatchError", "approval_digest", "verify_approval"]


class ApprovalDigestMismatchError(ValueError):
    """The manifest was approved, then its QC-relevant contents changed."""


def approval_digest(manifest: Mapping[str, object]) -> str:
    """Digest the facts a QC approval is a statement about.

    Deliberately **not** the whole manifest: re-approving after an unrelated field moves
    would be pure friction. These are the fields that, if changed, invalidate what the
    reviewer saw:

    - ``qc.y_origin`` -- the one that flips every box.
    - the overlay basenames -- *which* slices were reviewed.
    - the annotation source + box count -- a different CSV, or a different parse.
    """
    qc = manifest.get("qc") or {}
    annotation = manifest.get("annotation") or {}
    if not isinstance(qc, Mapping) or not isinstance(annotation, Mapping):
        raise TypeError("manifest 'qc' and 'annotation' must be objects")

    overlays = qc.get("overlays") or []
    payload = {
        "y_origin": qc.get("y_origin"),
        "overlays": sorted(Path(str(o)).name for o in overlays),
        "csv_path": annotation.get("csv_path"),
        "boxes_parsed": annotation.get("boxes_parsed"),
        "total_records": manifest.get("total_records"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


def verify_approval(manifest: Mapping[str, object], *, source: str) -> None:
    """Raise when an approval no longer describes the manifest it sits in.

    A no-op on an unapproved manifest -- the ``approved is not True`` gate in the
    caller is what rejects those, with a message about running the QC step.
    """
    qc = manifest.get("qc") or {}
    if not isinstance(qc, Mapping) or qc.get("approved") is not True:
        return

    recorded = qc.get("approved_digest")
    if not recorded:
        raise ApprovalDigestMismatchError(
            f"{source} is approved but carries no 'qc.approved_digest'. It was approved "
            "by an older generator that could not bind the approval to a y-origin. "
            "Re-run scripts/data/build_fastmri_plus_manifest.py and re-approve after "
            "looking at the A/B overlays -- the y-origin was CORRECTED on 2026-07-13 "
            "(it was inverted), so a pre-correction approval is exactly the thing that "
            "must not be trusted."
        )

    expected = approval_digest(manifest)
    if recorded != expected:
        raise ApprovalDigestMismatchError(
            f"{source}: qc.approved_digest={recorded!r} but the manifest now digests to "
            f"{expected!r}. The approval was made about different contents -- most "
            "likely qc.y_origin was edited after approval, which mirrors every lesion "
            "box onto the contralateral hemisphere without changing anything else. "
            "Rebuild, re-review the overlays, re-approve."
        )
