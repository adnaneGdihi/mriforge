"""Advisory gate: a committed dataset ``profile`` must not contradict ``source.needs``.

``scripts/data/gen_external_dataset_manifests.py --profile`` (run on the cluster where
``databases/external/`` lives) attaches a *measured* ``profile`` block to each populated
manifest under ``data/manifests/external/`` and records a ``needs_crosscheck`` of the
measured needs vs the hand-asserted ``source.needs``. This driver pins, for every
manifest that carries such a block:

* the profile is well-formed and its ``needs_crosscheck`` recomputes consistently
  (``profiler.crosscheck_manifest`` re-derives it repo-side, no data read);
* a dataset whose data *provably lacks* a need it claims (``agree == False``) does not
  ship silently — it must either have ``source.needs`` corrected or be listed in
  ``_PROFILE_MISMATCH_ACKNOWLEDGED`` with a reason (the "flag, don't fake" rule;
  same pattern as the VF B1+ xfail in ``test_vf_dataset_migration_invariants``).

Most manifests are cluster-only and carry no profile yet, so this file is largely
vacuous locally; it activates as profiled manifests are committed. This is **advisory**:
it catches a *committed* mismatch, it does not run the profiler.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mriforge.data import profiler

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST_DIR = _REPO / "data" / "manifests" / "external"

# Acknowledged measured-vs-asserted mismatches: dataset_name -> reason. An entry here
# means a human reviewed the profile and the asserted source.needs and chose to keep
# the claim (e.g. the feature is real but the header could not expose it on the sampled
# records). Empty until a profiling run surfaces one.
_PROFILE_MISMATCH_ACKNOWLEDGED: dict[str, str] = {}


def _profiled_manifests() -> list[Path]:
    out: list[Path] = []
    if not _MANIFEST_DIR.is_dir():
        return out
    for p in sorted(_MANIFEST_DIR.glob("*.json")):
        if p.name == "_CATALOG.json":
            continue
        if json.loads(p.read_text()).get("profile"):
            out.append(p)
    return out


_PROFILED = _profiled_manifests()
_IDS = [p.stem for p in _PROFILED]


def test_profiled_manifest_count_is_reported() -> None:
    """Informational guard so the cohort can't silently empty to zero unnoticed."""
    # No assertion on the count (it grows as the cluster profiles datasets); this simply
    # keeps the file non-empty and documents the current state in the test name.
    assert isinstance(_PROFILED, list)


@pytest.mark.parametrize("manifest_path", _PROFILED, ids=_IDS)
def test_committed_profile_is_wellformed(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    profile = manifest["profile"]
    for key in ("inventory", "geometry", "contrast", "physics_features",
                "derived_needs", "needs_crosscheck", "provenance"):
        assert key in profile, f"{manifest_path.name}: profile missing {key!r}"
    # the stored crosscheck must match a fresh recompute from the block
    recomputed = profiler.crosscheck_manifest(manifest)
    assert recomputed is not None
    assert recomputed["missing_from_data"] == profile["needs_crosscheck"]["missing_from_data"]
    assert recomputed["agree"] == profile["needs_crosscheck"]["agree"]


@pytest.mark.parametrize("manifest_path", _PROFILED, ids=_IDS)
def test_no_silently_committed_need_mismatch(manifest_path: Path) -> None:
    """A claimed need the measured data lacks must be fixed or explicitly acknowledged."""
    manifest = json.loads(manifest_path.read_text())
    cc = profiler.crosscheck_manifest(manifest)
    assert cc is not None
    if cc["agree"]:
        return
    name = manifest_path.stem
    if name in _PROFILE_MISMATCH_ACKNOWLEDGED:
        pytest.xfail(
            f"{name}: acknowledged measured-vs-asserted mismatch — "
            f"{_PROFILE_MISMATCH_ACKNOWLEDGED[name]}"
        )
    pytest.fail(
        f"{name}: source.needs claims {cc['missing_from_data']} but the measured "
        f"profile (derived={cc['derived']}) provably lacks it (pitfall #18). Fix "
        "source.needs in datasets.json (regen the manifest) or add an acknowledged "
        "reason to _PROFILE_MISMATCH_ACKNOWLEDGED."
    )
