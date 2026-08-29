"""Guards for the ulf_paired (64mT to 3T) manifest generator.

`data/manifests/` is gitignored wholesale, so the manifest never reaches the
cluster; only the generator does (`data.md`). These tests therefore exercise the
GENERATOR against the raw BIDS tree rather than asserting on a checked-in
artifact, and skip cleanly when the raw data is absent (CI, fresh clone).

Two classes of regression are pinned:

1. Discovery pointed at the wrong root. The default `--raw-root` matched only one
   of the two extraction layouts in use, and a miss is silent: discovery finds
   zero pairs and still writes a structurally valid manifest. The stale
   `ulf_paired_v6.json` on disk was generated that way and lists 32 records with
   no T2w at all, against the 43 the correct root yields.

2. Native geometry going unrecorded. `process_pair` resamples the 64mT volume
   onto the 3T grid, so after preprocessing both files report HF spacing and the
   ULF's true resolution survives only as an effective PSF. Every fiducial-based
   arm sizes its marker against the NATIVE ULF voxel; without these stamps that
   is not recoverable from the outputs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MRIFORGE_SUPPRESS_CLINICAL_WARNING", "1")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "preprocessing" / "preprocess_ulf_paired.py"


def _load_module():
    """`scripts/` is outside the package (layer rule), so import it by path."""
    spec = importlib.util.spec_from_file_location("preprocess_ulf_paired", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT.exists():
        pytest.skip("preprocess_ulf_paired.py not present")
    return _load_module()


@pytest.fixture(scope="module")
def raw_root(mod):
    try:
        return mod.resolve_raw_root(None)
    except FileNotFoundError:
        pytest.skip("raw ulf_paired BIDS tree not present on this host")


def test_raw_root_autodetect_finds_both_bids_trees(mod, raw_root) -> None:
    assert (raw_root / "3T_data").is_dir()
    assert (raw_root / "64mT_data").is_dir()


def test_explicit_bad_raw_root_raises_rather_than_finding_nothing(mod, tmp_path) -> None:
    """The silent-empty-manifest failure mode, made loud."""
    with pytest.raises(FileNotFoundError, match="does not contain both"):
        mod.resolve_raw_root(str(tmp_path))


def test_all_four_paired_contrasts_are_discovered(mod, raw_root, tmp_path) -> None:
    """T2w was dropped for a while by a false 'the 3T scanner has no T2w'
    premise. Eleven subjects pair on T1w/T2w/FLAIR and ten on ADC."""
    pairs = mod.discover_pairs(raw_root, tmp_path, list(mod.PAIRED_CONTRASTS))
    by_contrast: dict[str, int] = {}
    for p in pairs:
        by_contrast[p.contrast] = by_contrast.get(p.contrast, 0) + 1
    assert set(by_contrast) == {"T1w", "T2w", "FLAIR", "ADC"}
    assert by_contrast["T2w"] > 0, "T2w silently dropped again"
    assert len(pairs) >= 40


def test_every_paired_record_stamps_native_voxel_sizes(mod, raw_root, tmp_path) -> None:
    pairs = mod.discover_pairs(raw_root, tmp_path, list(mod.PAIRED_CONTRASTS))
    for p in pairs[:6]:
        rec = mod.make_record(p, "train", None)
        meta = rec["metadata"]
        assert meta["ulf_voxel_mm"] is not None
        assert meta["hf_voxel_mm"] is not None
        assert len(meta["ulf_voxel_mm"]) == 3
        assert all(v > 0 for v in meta["ulf_voxel_mm"])


def test_ulf_is_coarser_in_plane_than_hf(mod, raw_root, tmp_path) -> None:
    """The premise the whole ULF-to-HF task rests on. Through-plane is NOT
    checked: ULF is 5.0 mm against HF FLAIR's 5.5 mm, so that axis is a
    denoising/contrast-transfer problem, not super-resolution."""
    pairs = mod.discover_pairs(raw_root, tmp_path, list(mod.PAIRED_CONTRASTS))
    for p in pairs:
        ulf = mod.voxel_mm(p.ulf_src)
        hf = mod.voxel_mm(p.hf_src)
        if ulf is None or hf is None:
            continue
        assert min(ulf[:2]) >= min(hf[:2]), f"{p.subject}/{p.contrast} in-plane"


def test_geometry_minority_subjects_are_kept_out_of_validation(mod, raw_root, tmp_path) -> None:
    """This cohort runs three distinct 64mT protocols. With 11 subjects a 20%
    validation fraction is two subjects, so one out-of-protocol subject would be
    half of validation and the val metric would measure protocol mismatch."""
    pairs = mod.discover_pairs(raw_root, tmp_path, list(mod.PAIRED_CONTRASTS))
    sigs = mod.geometry_signature(pairs)
    groups: dict[tuple, list[str]] = {}
    for sub, sig in sigs.items():
        groups.setdefault(sig, []).append(sub)
    if len(groups) == 1:
        pytest.skip("cohort is geometrically homogeneous on this host")

    majority = max(groups.values(), key=len)
    minority = {s for members in groups.values() if members is not majority for s in members}
    splits = mod.assign_splits(sorted(sigs), val_frac=0.2, keep_out_of_val=minority)
    val = {s for s, sp in splits.items() if sp == "val"}
    assert val, "validation split must not be empty"
    assert not (val & minority), f"geometry outliers leaked into val: {val & minority}"


def test_split_is_subject_level(mod) -> None:
    """No subject may appear on both sides."""
    splits = mod.assign_splits(["a", "b", "c", "d", "e"], val_frac=0.4)
    train = {s for s, sp in splits.items() if sp == "train"}
    val = {s for s, sp in splits.items() if sp == "val"}
    assert not (train & val)
    assert train | val == {"a", "b", "c", "d", "e"}


def test_unpaired_records_carry_no_hf_target(mod, raw_root, tmp_path) -> None:
    """The unpaired 64mT cohort is inference-only: no 3T truth exists, so no
    supervised metric is computable on it and nothing may claim otherwise."""
    unpaired = mod.discover_unpaired_ulf(raw_root, tmp_path)
    assert unpaired, "expected a held-out unpaired ULF cohort"
    rec = mod.make_unpaired_record(unpaired[0], None)
    assert rec["target_path"] is None
    assert rec["split_hint"] == "test"
    assert rec["metadata"]["hf_voxel_mm"] is None
    assert rec["metadata"]["ulf_voxel_mm"] is not None
