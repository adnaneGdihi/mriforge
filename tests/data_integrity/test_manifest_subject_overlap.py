"""Audit on-disk manifest artifacts for subject overlap across splits.

This complements :mod:`tests.data_integrity.test_no_data_leaks` (which
tests the split *algorithm* via mocks). This file tests the real
manifest *artifacts* under ``data/manifests/`` — catching the class of
bug surfaced in the 2026-05-16 round-9 audit, where
``cross_site_pretrain.json`` was found to contain 100% of the subjects
from ``m4raw_multicoil_val.json`` (a hard data leak that no algorithmic
test would catch because the manifests are pre-built).

Subject identity is extracted as the leading digit-run of each
record's ``file_id`` field, which is the convention used across m4raw
manifests (e.g. ``2022061001_FLAIR01`` → subject ``2022061001``).

Tests skip cleanly when manifests are absent (e.g. on cluster nodes
with different path layouts) — they only fire when the file exists
locally.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

# Subject-id pattern: leading digit-run of file_id / filename / subject_id.
_SUBJ_RE = re.compile(r"^(\d+)")


def _extract_subjects(manifest_path: Path) -> set[str]:
    """Read a manifest and return the set of subject IDs it covers.

    Handles three record schemas seen in this repo:

    - top-level ``records`` list (m4raw_*, cross_site_*, pathology_*),
    - list-of-records at the top level (older manifests),
    - ULF paired manifests with a ``records`` list plus a
      ``split_hint`` field per record.
    """
    if not manifest_path.exists():
        return set()
    payload = json.loads(manifest_path.read_text())
    records = (
        payload.get("records", payload)
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(records, list):
        return set()
    subjects: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        fid = (
            rec.get("file_id")
            or rec.get("filename")
            or rec.get("subject_id")
            or rec.get("subject")
            or ""
        )
        match = _SUBJ_RE.match(str(fid))
        if match:
            subjects.add(match.group(1))
    return subjects


def _subjects_with_split(manifest_path: Path) -> dict[str, set[str]]:
    """For manifests that carry an internal ``split_hint`` (ULF paired),
    return a {split → subjects} mapping. Empty dict for flat manifests."""
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text())
    records = payload.get("records", []) if isinstance(payload, dict) else []
    splits: dict[str, set[str]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        split = rec.get("split_hint") or rec.get("split")
        if split is None:
            continue
        sid = rec.get("subject_id") or rec.get("subject", "")
        if sid:
            splits.setdefault(str(split), set()).add(str(sid))
    return splits


# ── train ↔ val ↔ test disjointness across all m4raw splits ─────────────────


@pytest.mark.parametrize(
    "train_name,val_name",
    [
        ("m4raw_train.json", "m4raw_multicoil_val.json"),
        ("m4raw_train.json", "m4raw_multicoil_test.json"),
        ("m4raw_multi_contrast_train.json", "m4raw_multi_contrast_val.json"),
        ("m4raw_multi_contrast_train.json", "m4raw_multi_contrast_test.json"),
        ("m4raw_multicoil_train_image_reconstructed.json",
         "m4raw_multicoil_val_image_reconstructed.json"),
        ("m4raw_calibration.json", "m4raw_multicoil_val.json"),
        ("m4raw_calibration.json", "m4raw_multicoil_test.json"),
    ],
)
def test_m4raw_split_pair_is_subject_disjoint(train_name: str, val_name: str) -> None:
    train = _extract_subjects(MANIFEST_DIR / train_name)
    val = _extract_subjects(MANIFEST_DIR / val_name)
    if not train or not val:
        pytest.skip(f"Manifest(s) missing locally: {train_name} or {val_name}")
    overlap = train & val
    assert not overlap, (
        f"DATA LEAK: {len(overlap)} shared subjects between {train_name} and "
        f"{val_name}: {sorted(overlap)[:10]}…"
    )


# ── pathology + special manifests ───────────────────────────────────────────


def test_pathology_test_set_disjoint_from_m4raw_val() -> None:
    pathology = _extract_subjects(MANIFEST_DIR / "pathology_test_set.json")
    val = _extract_subjects(MANIFEST_DIR / "m4raw_multicoil_val.json")
    if not pathology or not val:
        pytest.skip("Manifest(s) missing locally")
    overlap = pathology & val
    assert not overlap, (
        f"DATA LEAK: pathology_test_set ∩ m4raw_multicoil_val: {sorted(overlap)}"
    )


# ── cross_site_pretrain regression guard ────────────────────────────────────


def test_cross_site_pretrain_versus_val_splits() -> None:
    """Pin the round-9 finding: ``cross_site_pretrain.json`` contains
    every subject of ``m4raw_multicoil_val.json`` and ``m4raw_multi_contrast_val.json``.

    The plan-level fix (round-9 §5h) re-routed
    ``exp_pac_bayes_certificate.yaml`` to use ``m4raw_multicoil_test``
    as val. This test guards against a future YAML re-using
    ``cross_site_pretrain`` + ``m4raw_multicoil_val`` without fixing
    the pretrain manifest first.
    """
    cross = _extract_subjects(MANIFEST_DIR / "cross_site_pretrain.json")
    mc_val = _extract_subjects(MANIFEST_DIR / "m4raw_multicoil_val.json")
    mc_test = _extract_subjects(MANIFEST_DIR / "m4raw_multicoil_test.json")
    if not cross:
        pytest.skip("cross_site_pretrain.json missing locally")
    # The known-bad pair: cross_site_pretrain ∩ multicoil_val == multicoil_val
    if mc_val:
        overlap_val = cross & mc_val
        # Document the current leak rather than asserting it away — the
        # PAC-Bayes YAML now uses multicoil_test, so any consumer of
        # cross_site_pretrain + multicoil_val is a regression.
        assert overlap_val, (
            "cross_site_pretrain unexpectedly disjoint from "
            "m4raw_multicoil_val — was the pretrain manifest re-built? "
            "If so, the round-9 PAC-Bayes redirect to multicoil_test "
            "is no longer needed and the YAML may be reverted."
        )
    if mc_test:
        overlap_test = cross & mc_test
        assert not overlap_test, (
            f"DATA LEAK: cross_site_pretrain ∩ m4raw_multicoil_test: "
            f"{sorted(overlap_test)}. The PAC-Bayes redirect target is "
            "no longer safe — pick a different val manifest."
        )


# ── ULF paired manifest internal splits ─────────────────────────────────────


@pytest.mark.parametrize(
    "manifest_name",
    [
        "ulf_paired_brain_isotropic_v5_fixed.json",
        "ulf_paired_brain_isotropic_v5.json",
        "ulf_paired_brain_v5.json",
        "ulf_paired_brain_registered_v5.json",
    ],
)
def test_ulf_paired_internal_splits_disjoint(manifest_name: str) -> None:
    splits = _subjects_with_split(MANIFEST_DIR / manifest_name)
    if not splits:
        pytest.skip(
            f"{manifest_name} missing or has no split_hint field"
        )
    keys = list(splits.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = splits[keys[i]] & splits[keys[j]]
            assert not overlap, (
                f"DATA LEAK in {manifest_name}: "
                f"{keys[i]} ∩ {keys[j]}: {sorted(overlap)[:10]}…"
            )
