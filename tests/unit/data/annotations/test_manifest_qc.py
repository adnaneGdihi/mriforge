"""The approval digest: an approval cannot be transplanted onto a different y-origin."""

from __future__ import annotations

import re

import pytest

from mriforge.data.annotations.manifest_qc import (
    ApprovalDigestMismatchError,
    approval_digest,
    verify_approval,
)


def _manifest(**qc_overrides) -> dict:
    qc = {
        "y_origin": "bottom_left",
        "overlays": ["/qc/a_sl001.png", "/qc/b_sl007.png"],
        "approved": True,
        "approved_digest": None,
    }
    qc.update(qc_overrides)
    m = {
        "annotation": {"csv_path": "brain.csv", "boxes_parsed": 7570},
        "total_records": 1001,
        "qc": qc,
    }
    if qc["approved_digest"] is None:
        m["qc"]["approved_digest"] = approval_digest(m)
    return m


class TestTheDigestCoversWhatMatters:
    def test_the_y_origin_changes_the_digest(self) -> None:
        """The whole point. Flipping this one field mirrors every lesion box onto the
        contralateral hemisphere -- and changes nothing else about the manifest."""
        a = _manifest()
        b = _manifest()
        b["qc"]["y_origin"] = "top_left"
        assert approval_digest(a) != approval_digest(b)

    def test_the_reviewed_slice_set_changes_the_digest(self) -> None:
        """An approval says 'I looked at THESE slices'. A different set is a different
        claim -- you cannot inherit someone else's review of other images."""
        b = _manifest(overlays=["/qc/a_sl001.png"])
        assert approval_digest(_manifest()) != approval_digest(b)

    def test_a_different_annotation_csv_changes_the_digest(self) -> None:
        b = _manifest()
        b["annotation"]["csv_path"] = "knee.csv"
        assert approval_digest(_manifest()) != approval_digest(b)

    def test_the_digest_is_stable_across_overlay_ordering_and_directory(self) -> None:
        """Basenames, sorted. Re-rendering the same slices into a different scratch dir
        must not invalidate a review -- only the CONTENT of the claim may."""
        b = _manifest(overlays=["/elsewhere/b_sl007.png", "/elsewhere/a_sl001.png"])
        assert approval_digest(_manifest()) == approval_digest(b)


class TestVerifyApproval:
    def test_a_matching_digest_passes(self) -> None:
        verify_approval(_manifest(), source="m.json")

    def test_an_unapproved_manifest_is_not_this_gate_s_business(self) -> None:
        """`approved is not True` is rejected by the caller, with a message about
        running the QC step. This gate stays silent so the two do not fight."""
        verify_approval({"qc": {"approved": False}}, source="m.json")

    def test_a_tampered_y_origin_raises(self) -> None:
        m = _manifest()
        m["qc"]["y_origin"] = "top_left"  # digest was minted for bottom_left
        with pytest.raises(ApprovalDigestMismatchError, match="edited after approval"):
            verify_approval(m, source="m.json")

    def test_an_approval_with_no_digest_raises(self) -> None:
        """Every approval predating 2026-07-13 was made under the INVERTED y-origin.
        Honouring it is exactly the failure this module exists to prevent."""
        m = _manifest()
        del m["qc"]["approved_digest"]
        with pytest.raises(ApprovalDigestMismatchError, match="y-origin was CORRECTED"):
            verify_approval(m, source="m.json")

    def test_the_error_names_the_manifest(self) -> None:
        m = _manifest(approved_digest="deadbeef")
        with pytest.raises(ApprovalDigestMismatchError, match=re.escape("fastmri_plus_brain.json")):
            verify_approval(m, source="fastmri_plus_brain.json")
