"""Manifest-driven dataset catalog for the dataset EDA.

Discovers every manifest under ``data/manifests/**`` (JSON only — never the ``.pkl``
mirrors, to respect the pickle ban in CLAUDE.md pitfall #11), resolves external→local
pointers, and classifies each dataset's modality + tier from manifest metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Order matters: non-Cartesian (spiral/radial) is checked before temporal so a
# spiral-fMRI set such as ``kasper_7t_spiral_fmri`` classifies as ``noncartesian``,
# not ``temporal`` (it is both, but the acquisition geometry is the dominant axis
# for understanding the data).
_NONCARTESIAN_HINTS = ("spiral", "radial", "4dflow", "fary", "ocmr", "trajectory")
_TEMPORAL_HINTS = ("cine", "fmri", "realtime", "5d", "temporal", "4d")
_PAIRED_HINTS = ("paired", "ulf_to_hf", "ulf_paired", "64mt", "harmoni")


@dataclass(frozen=True)
class DatasetEntry:
    """One discoverable dataset, resolved from its manifest.

    Attributes:
        dataset_id: Manifest filename stem (e.g. ``fastmri_knee_singlecoil``).
        aliases: Other manifest stems that resolved to the same populated data.
        manifest_path: The discovered manifest file.
        data_root: Resolved root for the records (absolute, or ``None`` if absent).
        file_type: ``h5`` | ``nifti`` | ``ismrmrd`` | ``bart`` | ``mat`` | ``twix`` |
            ``png`` | ``unknown``.
        records: Populated ``records[]`` (possibly empty for absent datasets).
        modality: ``kspace`` | ``noncartesian`` | ``image`` | ``paired`` | ``map`` |
            ``temporal`` | ``unknown``.
        tier: ``present`` (records to plot) or ``absent`` (card only).
        status: ``downloaded`` | ``not_downloaded`` | ``defective`` | ``unknown``.
        source: External ``source`` metadata block, or ``{}`` for local manifests.
    """

    dataset_id: str
    aliases: tuple[str, ...]
    manifest_path: Path
    data_root: Path | None
    file_type: str
    records: tuple[dict, ...]
    modality: str
    tier: str
    status: str
    source: dict


def classify_modality(*, role: str, raw_kspace: bool, file_type: str, name: str) -> str:
    """Map manifest metadata to a figure-suite modality (no voxel read needed)."""
    n = name.lower()
    ft = (file_type or "").lower()
    if role.upper() == "MAP":
        return "map"
    if role.upper() == "RAW" or raw_kspace:
        # ISMRMRD / .mrd raw acquisitions are a list of readouts (n_acq, coil, samples), like
        # BART/TWIX: a naive ifft2c fabricates a meaningless image, so they get the non-Cartesian
        # treatment (acquired-samples log-magnitude only) until trajectory-based gridding exists.
        if any(h in n for h in _NONCARTESIAN_HINTS) or ft in ("bart", "twix", "ismrmrd", "mrd"):
            return "noncartesian"
        if any(h in n for h in _TEMPORAL_HINTS):
            return "temporal"
        return "kspace"
    if any(h in n for h in _PAIRED_HINTS):
        return "paired"
    if any(h in n for h in _TEMPORAL_HINTS):
        return "temporal"  # image-domain cine / 4D-flow / 5D → the temporal figure suite
    if role.lower() == "img" or ft in ("png", "nifti"):
        return "image"
    return "unknown"


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _resolve_path(raw: str | None, base: Path) -> Path | None:
    """Resolve a manifest path string.

    Absolute or already-existing paths are returned as-is; otherwise the (repo-relative)
    path is resolved against ``base`` — the EDA ``data_root`` (repo root by default), so a
    manifest ``data_root`` of ``databases/fastmri/...`` lands under ``<base>/databases/...``.
    """
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() or p.exists():
        return p
    return Path(base) / p


def _resolve_records(manifest: dict, base: Path) -> tuple[dict, list[dict]]:
    """Follow a ``local_manifest`` pointer to the populated records, if present."""
    local = manifest.get("local_manifest")
    if local:
        cand = _resolve_path(local, base)
        if cand and cand.exists():
            populated = _load_json(cand)
            return populated, list(populated.get("records", []))
    records = list(manifest.get("records", []))
    if not records and manifest.get("files"):
        # legacy files[] / nifti_paired schema (ulf_paired_brain_v5) — treat each file as a record
        # so the dataset is discoverable; its input_path locates the data like primary_path.
        records = list(manifest.get("files", []))
    return manifest, records


_PHYSICAL_SKIP = {"external", "processed"}  # ledger-only / symlink duplicate
# Data extensions the EDA can read (kept in sync with ``loader._DATA_EXTS``). ``.cfl`` is the
# BART k-space payload (radial/spiral); ``.nii.gz`` is matched by name (double suffix).
_DATA_EXTS = (
    ".h5",
    ".nii.gz",
    ".nii",
    ".png",
    ".pt",
    ".npy",
    ".mat",
    ".mrd",
    ".cfl",
    ".dcm",
    ".ima",
)
# Archive formats whose presence means the dataset IS downloaded — the loader reads an image
# straight out of them (kirby_kki_multimodal ships its NIfTI inside MMRR-*.tar.gz). Counted as
# data so a downloaded-but-not-unpacked dataset is tier=present rather than 'absent'.
_ARCHIVE_EXTS = (".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar")


def _is_archive(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(e) for e in _ARCHIVE_EXTS)


# Shared-container directory names that hold *many* datasets — the scan-dir climb STOPS at these
# so a genuinely-absent set (its own dir missing) cannot walk up to ``external/`` and false-match
# a *sibling's* data (the historical calgary→zenodo false-match). Dataset-OWNED subdirs
# (``raw``/``extracted``/``Data``/…) are NOT here — the climb may legitimately anchor on them.
_SHARED_CONTAINERS = {"external", "processed", "databases", "datasets"}


def _suffix(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".gz" and path.stem.lower().endswith(".nii"):
        return ".nii.gz"
    return s


def _dir_has_data(d: Path, limit: int = 4000, include_archives: bool = True) -> bool:
    if not d.is_dir():
        return False
    for seen, p in enumerate(d.rglob("*")):
        if p.is_file() and (_suffix(p) in _DATA_EXTS or (include_archives and _is_archive(p))):
            return True
        if seen > limit:
            break
    return False


def resolve_scan_dir(root: Path | None) -> Path | None:
    """Best on-disk directory to scan for a dataset's files (trust the filesystem over the
    manifest). Shared by :func:`discover` (the tier decision) and the EDA loader.

    Resolution order — first candidate that actually contains data wins:

    1. ``data_root`` itself;
    2. a sibling ``extracted/`` when ``data_root`` ends in ``raw/`` — external sets unpacked to
       ``…/<x>/extracted`` while the manifest stub still points at ``…/<x>/raw``;
    3. the nearest **dataset-owned ancestor** that holds data — the climb walks up the
       ``databases`` tree (so a ``data_root`` pointing deep into a *never-generated* subpath,
       ``…/Data/64mT_data_image/isotropic``, still finds the real data at ``…/Data``) but STOPS
       at a :data:`_SHARED_CONTAINERS` name or on leaving ``databases``, so an absent set never
       anchors on a sibling's data.

    Falls back to the first existing candidate (so the loader can emit a precise
    "no readable files under <dir>" note), or ``None`` when nothing resolves.
    """
    if root is None:
        return None
    root = Path(root)
    candidates: list[Path] = []
    if root.is_dir():
        candidates.append(root)
    if root.name == "raw":
        sib = root.parent / "extracted"
        if sib.is_dir():
            candidates.append(sib)
    for anc in root.parents:
        if anc.name in _SHARED_CONTAINERS or "databases" not in str(anc):
            break  # never anchor at/above a shared container that holds many datasets
        if anc.is_dir():
            candidates.append(anc)
    # Prefer a dir with real (loadable) data over one that only holds archives — mrf_100mt_optimum
    # keeps its .npy dictionary in extracted/ while raw/ has only a .zip; the extracted data must win.
    for c in candidates:
        if _dir_has_data(c, include_archives=False):
            return c
    for c in candidates:
        if _dir_has_data(c):  # archive-only (kirby's MMRR-*.tar.gz) — the loader unpacks it
            return c
    return candidates[0] if candidates else None


def _derive_root_from_records(records: list[dict], base: Path) -> Path | None:
    """Common-ancestor directory of the records' declared paths (existing or not).

    Used when a manifest omits ``data_root`` but its records carry ``primary_path`` /
    ``input_path`` values (the ``ulf_paired_brain_*_v5`` family). The leaf may point at a
    never-generated path — :func:`resolve_scan_dir` climbs from the returned ancestor to the real
    data. Returns the common ancestor directory, or ``None`` when no record path is present.
    """
    import os

    paths: list[str] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        cand = _resolve_path(r.get("primary_path") or r.get("path") or r.get("input_path"), base)
        if cand is not None:
            paths.append(str(cand if cand.is_dir() else cand.parent))
    if not paths:
        return None
    return Path(os.path.commonpath(paths))


def discover_physical(databases_dir: Path, covered: list[DatasetEntry]) -> list[DatasetEntry]:
    """Add entries for ``databases/<subdir>`` trees with real data but no covering manifest.

    A subdir is "covered" when some present manifest entry's resolved ``data_root`` already
    lives under it (so its files are reached via :func:`discover` + the loader's ancestor
    scan). The uncovered remainder — e.g. ``kasper`` (spiral) and ``mock_paired`` (paired
    NIfTI), which have stale or no manifest — is surfaced here so no on-disk data is neglected.
    """
    databases_dir = Path(databases_dir)
    if not databases_dir.is_dir():
        return []
    covered_roots = [str(e.data_root) for e in covered if e.tier == "present" and e.data_root]
    # Candidate dataset dirs: top-level ``databases/<x>``, plus one level *into* the shared
    # containers (``external/<x>``, ``processed/<x>``). Descending there surfaces data whose
    # manifest is stale, mis-named (``kirby_kki`` vs ``kirby_kki_multimodal``), or unpopulated —
    # which the old wholesale ``external/`` skip rendered invisible.
    candidates: list[Path] = []
    for sub in sorted(p for p in databases_dir.iterdir() if p.is_dir()):
        if sub.name in _PHYSICAL_SKIP:
            candidates.extend(sorted(p for p in sub.iterdir() if p.is_dir()))
        else:
            candidates.append(sub)
    entries: list[DatasetEntry] = []
    seen: set[str] = set()
    for sub in candidates:
        if sub.name in seen or f"disk_{sub.name}" in seen:
            continue
        if any(f"/{sub.name}/" in r or r.endswith(f"/{sub.name}") for r in covered_roots):
            continue
        if not _dir_has_data(sub):
            continue
        seen.add(f"disk_{sub.name}")
        has_pair = (sub / "source").is_dir() and (sub / "target").is_dir()
        name = sub.name
        is_noncart = (
            any(h in name.lower() for h in _NONCARTESIAN_HINTS)
            or "kasper" in name.lower()
            or "spifi" in name.lower()
        )
        if has_pair:
            role, modality = "img", "paired"
        elif is_noncart:
            role, modality = "RAW", "noncartesian"
        else:
            role = "img"
            modality = classify_modality(
                role=role, raw_kspace=False, file_type="unknown", name=name
            )
        entries.append(
            DatasetEntry(
                dataset_id=f"disk_{name}",
                aliases=(),
                manifest_path=sub,
                data_root=sub,
                file_type="unknown",
                records=(),
                modality=modality,
                tier="present",
                status="downloaded",
                source={
                    "role": role,
                    "anatomy": "?",
                    "provides": f"on-disk files under {sub}",
                    "links": [],
                    "note": "discovered by directory scan (no manifest)",
                },
            )
        )
    return entries


def discover(manifests_root: Path, data_root: Path) -> list[DatasetEntry]:
    """Discover and classify every JSON manifest under ``manifests_root``.

    Args:
        manifests_root: Directory holding manifests (recursively); ``_CATALOG.json`` skipped.
        data_root: Base for resolving repo-relative manifest ``data_root`` / ``local_manifest``
            paths (repo root by default; absolute manifest roots ignore it).

    Returns:
        One :class:`DatasetEntry` per manifest, in sorted-path order.
    """
    manifests_root = Path(manifests_root)
    base = Path(data_root)
    paths = sorted(p for p in manifests_root.rglob("*.json") if p.name != "_CATALOG.json")

    entries: list[DatasetEntry] = []
    for mpath in paths:
        try:
            manifest = _load_json(mpath)
        except Exception:
            continue
        source = manifest.get("source", {}) or {}
        populated, records = _resolve_records(manifest, base)
        file_type = (populated.get("file_type") or manifest.get("file_type") or "unknown").lower()
        root = _resolve_path(populated.get("data_root") or manifest.get("data_root"), base)
        if root is None and records:
            # data_root omitted but records carry absolute paths (ulf_*_v5) → derive it so the
            # loader has a directory to scan instead of emitting a 'no scan dir' card.
            root = _derive_root_from_records(list(records), base)
        status = (
            manifest.get("status")
            or source.get("status")
            or ("downloaded" if records else "unknown")
        )
        # Presence trusts the filesystem, not just a (possibly-unpopulated) manifest stub: a
        # descriptor with empty records[] whose data is on disk — at data_root, a sibling
        # extracted/ tree, or a dataset-specific ancestor — still renders. A manifest explicitly
        # flagged 'defective' is never auto-promoted (controlled-access sets carry only sidecars).
        if str(status).lower() == "defective":
            tier = "absent"
        elif records:
            tier = "present"
        else:
            scan = resolve_scan_dir(root)
            if scan is not None and _dir_has_data(scan):
                tier, status = "present", "downloaded"  # found on disk despite the stub status
            else:
                tier = "absent"
        modality = classify_modality(
            role=str(source.get("role", "RAW" if "kspace" in mpath.stem else "img")),
            raw_kspace=bool(source.get("raw_kspace", "kspace" in mpath.stem)),
            file_type=file_type,
            name=mpath.stem,
        )
        entries.append(
            DatasetEntry(
                dataset_id=mpath.stem,
                aliases=(),
                manifest_path=mpath,
                data_root=root,
                file_type=file_type,
                records=tuple(records),
                modality=modality,
                tier=tier,
                status=str(status),
                source=source,
            )
        )
    return entries
