"""Measured dataset profiler — what a dataset physically *contains*.

The external-dataset manifests (``data/manifests/external/<id>.json``) record which
files exist and what the dataset is *claimed* to provide (``source.needs``, hand
curated). This module produces the **measured** complement: a header-only ``profile``
block reporting inventory, geometry, contrast, and — most importantly — the physics
features actually present in the data, three-state (``present`` / ``absent`` /
``unknown``) with cited evidence. From those it derives the dataset's *measured* needs
and cross-checks them against the asserted ``source.needs``, closing the
measured-capability gap (CLAUDE.md pitfall #18: metric/claim ↔ data mismatch).

Design invariants:

* **Header-only.** Extractors read ``.hdr`` text, ISMRMRD/HDF5 XML headers, NIfTI
  affines, and DICOM tags — never voxel payloads (the queue-build OOM lesson).
* **Three-state honesty (pitfall #9).** A feature is ``present`` only with concrete
  evidence; ``absent`` only when the header was read and the feature is provably not
  there; ``unknown`` when the header cannot answer or the reader library is missing.
  We never guess ``absent`` from silence, and the cross-check treats ``unknown`` as
  "not contradicted".

This file holds the pure logic (feature/need maps, ``derive_needs``,
``crosscheck_needs``); the per-format extractors and the dataset aggregator are added
alongside it. Data inspection is a ``data/`` SSOT responsibility.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

import numpy as np

# The 10 needs (DATASETS_compilation.md): 1 multi-echo B0/QSM · 2 B1+ · 3 phase-cycled
# bSSFP · 4 non-Cartesian+trajectory · 5 MRF · 6 fMRI/4D/cine · 7 qMRI T1/T2/PD ·
# 8 ULF->HF · 9 multi-vendor · 10 multi-coil+calibration.
NEED_LABELS: dict[int, str] = {
    1: "multi-echo B0/QSM",
    2: "B1+",
    3: "phase-cycled bSSFP",
    4: "non-Cartesian+trajectory",
    5: "MRF",
    6: "fMRI/4D/cine",
    7: "qMRI T1/T2/PD",
    8: "ULF->HF",
    9: "multi-vendor",
    10: "multi-coil+calibration",
}

# Physics-feature key -> the numbered need it satisfies. Needs 8/9/10 are derived from
# geometry/contrast/inventory rather than a single feature, so they are handled in
# ``derive_needs`` directly. ``phase`` and ``b1_minus`` are descriptive (they inform a
# need — e.g. b1_minus feeds multi-coil — but are not needs by themselves).
NEED_TO_FEATURE: dict[int, str] = {
    1: "b0_field",
    2: "b1_plus",
    3: "phase_cycled_bssfp",
    4: "trajectory",
    5: "mrf",
    6: "temporal_4d",
    7: "qmri",
}

# Every physics-feature key a profile reports (mapped needs + descriptive features).
PHYSICS_FEATURE_KEYS: tuple[str, ...] = (
    "b0_field",
    "b1_plus",
    "phase_cycled_bssfp",
    "trajectory",
    "mrf",
    "qmri",
    "temporal_4d",
    "phase",
    "b1_minus",
)

# Three-state feature values.
PRESENT = "present"
ABSENT = "absent"
UNKNOWN = "unknown"

# Field strength (Tesla) at/below which a dataset is ultra-low-field (need 8). 64 mT
# Hyperfine = 0.064 T is ULF; 0.3 T M4Raw is not.
_ULF_FIELD_T = 0.1


def present(evidence: str) -> dict[str, str | None]:
    """A ``present`` feature carrying the concrete evidence that proved it."""
    return {"state": PRESENT, "evidence": evidence}


def absent() -> dict[str, str | None]:
    """An ``absent`` feature — the header was read and the feature is provably not there."""
    return {"state": ABSENT, "evidence": None}


def unknown(reason: str) -> dict[str, str | None]:
    """An ``unknown`` feature — the header could not answer (never guessed ``absent``)."""
    return {"state": UNKNOWN, "evidence": reason}


def _state(features: dict, key: str) -> str:
    return str(features.get(key, {}).get("state", UNKNOWN))


def derive_needs(profile: dict) -> dict[str, list[int]]:
    """Compute the dataset's *measured* needs from its profile.

    Returns ``{"present": [...], "unknown": [...]}`` — sorted, disjoint need-id lists.
    A need is ``present`` only when its evidence is measured, ``unknown`` when the
    relevant header could not be read, and otherwise (provably absent) appears in
    neither. ``present`` always wins over ``unknown`` if both are somehow triggered.
    """
    features = profile.get("physics_features", {})
    geometry = profile.get("geometry", {})
    contrast = profile.get("contrast", {})
    inventory = profile.get("inventory", {})

    present_needs: set[int] = set()
    unknown_needs: set[int] = set()

    # Feature-mapped needs (1,2,3,4,5,6,7).
    for need, feat in NEED_TO_FEATURE.items():
        st = _state(features, feat)
        if st == PRESENT:
            present_needs.add(need)
        elif st == UNKNOWN:
            unknown_needs.add(need)

    # Need 10 — multi-coil (n_coils > 1).
    n_coils = geometry.get("n_coils")
    if n_coils is None:
        unknown_needs.add(10)
    elif n_coils > 1:
        present_needs.add(10)

    # Need 8 — ultra-low-field (field strength <= 0.1 T).
    field_t = contrast.get("field_strength_t")
    if field_t is None:
        unknown_needs.add(8)
    elif field_t <= _ULF_FIELD_T:
        present_needs.add(8)

    # Need 9 — multi-vendor (>= 2 distinct vendors in the headers).
    vendors = inventory.get("vendors")
    if not vendors:
        unknown_needs.add(9)
    elif len({v for v in vendors if v}) >= 2:
        present_needs.add(9)

    unknown_needs -= present_needs
    return {"present": sorted(present_needs), "unknown": sorted(unknown_needs)}


def _normalize_needs(needs: list) -> tuple[list[int], list[str]]:
    """Split a raw ``source.needs`` list into (sorted-unique ints, non-numeric labels).

    ``source.needs`` may mix numbered needs with descriptive strings, e.g.
    ``["phase", 10, "7"]`` — coerce numeric ints/strings, keep the rest verbatim.
    """
    numeric: set[int] = set()
    non_numeric: list[str] = []
    for n in needs:
        if isinstance(n, bool):  # bool is an int subclass — exclude
            non_numeric.append(str(n))
        elif isinstance(n, int):
            numeric.add(n)
        elif isinstance(n, str) and n.strip().lstrip("-").isdigit():
            numeric.add(int(n.strip()))
        else:
            non_numeric.append(str(n))
    return sorted(numeric), non_numeric


def crosscheck_needs(
    asserted: list,
    derived_present: list[int],
    derived_unknown: list[int],
) -> dict:
    """Compare the asserted ``source.needs`` against the measured needs.

    The dangerous mismatch (pitfall #18) is ``missing_from_data``: a need the manifest
    *claims* but the data *provably lacks*. A claimed need that merely could not be
    measured (``derived_unknown``) is NOT a mismatch — it is "not contradicted". A need
    the data has but the manifest did not claim is informative (``extra_in_data``), not
    a disagreement. ``agree`` is ``True`` iff ``missing_from_data`` is empty.
    """
    asserted_numeric, non_numeric = _normalize_needs(asserted)
    present_set = set(derived_present)
    unknown_set = set(derived_unknown)

    missing = [n for n in asserted_numeric if n not in present_set and n not in unknown_set]
    extra = [n for n in derived_present if n not in asserted_numeric]
    return {
        "asserted": asserted_numeric,
        "non_numeric": non_numeric,
        "derived": sorted(present_set),
        "missing_from_data": sorted(missing),
        "extra_in_data": sorted(extra),
        "agree": not missing,
    }


# ============================ per-format header extractors ============================
#
# Each ``profile_*`` reads ONE record's HEADER (never voxel payloads) and returns a
# record-level contribution::
#
#     {"reader": str, "geometry": {...}, "contrast": {...},
#      "features": {<key>: <three-state>}, "vendor": str | None}
#
# An extractor reports a feature only when its header can speak to it: ``present`` with
# evidence, or ``absent`` when it positively rules it out. A feature outside the format's
# competence is OMITTED (it stays ``unknown`` at the dataset aggregate) — never guessed.

# filename-substring -> (feature key, evidence). A name match is positive evidence only;
# the ABSENCE of a match never implies the feature is absent.
_NAME_FEATURE_PATTERNS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("b1map", "b1_map", "b1plus", "b1+", "afi", "double_angle", "bloch_siegert"),
        "b1_plus",
        "filename names a B1+ map",
    ),
    (
        (
            "t1map",
            "t2map",
            "pdmap",
            "t2star",
            "r1map",
            "r2map",
            "r2star",
            "t1_map",
            "t2_map",
            "pd_map",
        ),
        "qmri",
        "filename names a qMRI map",
    ),
    (
        ("b0map", "b0_map", "fieldmap", "field_map", "b0field", "_b0"),
        "b0_field",
        "filename names a B0 field map",
    ),
    (("mrf", "dictionary", "_dict"), "mrf", "filename names MRF/dictionary data"),
    (
        ("phasecycled", "phase_cycled", "pcbssfp", "pc_bssfp", "pc-bssfp"),
        "phase_cycled_bssfp",
        "filename names phase-cycled bSSFP",
    ),
)


def _features_from_name(path: str | _Path) -> dict[str, dict]:
    """Detect physics features asserted by a record's filename (positive evidence only)."""
    stem = _Path(path).name.lower()
    out: dict[str, dict] = {}
    for needles, key, evidence in _NAME_FEATURE_PATTERNS:
        if any(n in stem for n in needles):
            out[key] = present(evidence)
    return out


def _parse_bart_dims(text: str) -> list[int]:
    """First non-comment, non-blank line of a BART ``.hdr`` is the dim vector."""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return [int(tok) for tok in s.split()]
    raise ValueError("BART header contains no dimension line.")


def profile_bart(path: str | _Path) -> dict:
    """Profile a BART ``.cfl``/``.hdr`` record from its header (pure stdlib).

    BART ``.hdr`` gives the dimension vector but not its semantics — the repo configures
    coil/echo/time axes per-dataset via ``bart_dim_map``. So this reports the first two
    dims as the matrix and the third as slices, but leaves ``n_coils`` ``None`` (honest
    unknown rather than assuming the BART COIL_DIM convention). A ``*_traj`` sidecar
    proves a non-Cartesian trajectory (need 4); the payload is always complex (phase).
    """
    p = _Path(path)
    base = p.with_suffix("") if p.suffix in (".cfl", ".hdr") else p
    hdr = base.with_suffix(".hdr")
    dims = _parse_bart_dims(hdr.read_text())

    geometry = {
        "matrix": dims[:2] if len(dims) >= 2 else None,
        "n_slices": dims[2] if len(dims) >= 3 else None,
        "n_coils": None,  # BART .hdr cannot disambiguate coil from echo/time — unknown
        "voxel_spacing_mm": None,
        "domain": "kspace",
        "bart_dims": dims,
    }
    features: dict[str, dict] = {"phase": present("BART payload is complex64")}
    if (base.parent / (base.name + "_traj.cfl")).is_file() or (
        base.parent / (base.name + "_traj.hdr")
    ).is_file():
        features["trajectory"] = present("BART _traj sidecar present (non-Cartesian)")
    features.update(_features_from_name(base))
    return {
        "reader": "bart-hdr",
        "geometry": geometry,
        "contrast": {"complex": True, "field_strength_t": None},
        "features": features,
        "vendor": None,
    }


def profile_nifti(path: str | _Path) -> dict:
    """Profile a NIfTI record from its header (voxel spacing, matrix, 4D temporality)."""
    import nibabel as nib

    img: Any = nib.load(str(path))  # nibabel's union return type hides shape/header
    shape = [int(s) for s in img.shape]
    zooms = [float(z) for z in img.header.get_zooms()]
    is_4d = len(shape) >= 4 and shape[3] > 1

    geometry = {
        "matrix": list(shape[:2]) if len(shape) >= 2 else None,
        "n_slices": shape[2] if len(shape) >= 3 else None,
        "voxel_spacing_mm": zooms[:3] if zooms else None,
        "n_coils": None,
        "domain": "image",
    }
    features: dict[str, dict] = {
        "temporal_4d": present(f"4D volume, {shape[3]} frames") if is_4d else absent()
    }
    is_complex = bool(np.iscomplexobj(np.empty(0, dtype=img.get_data_dtype())))
    if is_complex:
        features["phase"] = present("complex NIfTI dtype")
    features.update(_features_from_name(path))
    return {
        "reader": "nifti-header",
        "geometry": geometry,
        "contrast": {"field_strength_t": None, "complex": is_complex},
        "features": features,
        "vendor": None,
    }


_ISMRMRD_NS = "{http://www.ismrm.org/ISMRMRD}"


def _xml_text(value) -> str:
    """Decode an h5py-stored XML header (bytes / np.bytes_ / str / 1-elem array)."""
    if isinstance(value, np.ndarray):
        value = value[0] if value.size else b""
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8", "replace")
    return str(value)


def _parse_ismrmrd_xml(text: str) -> dict:
    """Extract field strength, coils, vendor, matrix, FOV, trajectory, TE from the XML."""
    import xml.etree.ElementTree as ET

    ns = _ISMRMRD_NS
    out: dict = {
        "trajectory": None,
        "field_strength_t": None,
        "n_coils": None,
        "vendor": None,
        "matrix": None,
        "fov_mm": None,
        "te_ms": None,
    }
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out

    asi = root.find(f"{ns}acquisitionSystemInformation")
    if asi is not None:
        out["vendor"] = asi.findtext(f"{ns}systemVendor") or None
        fs = asi.findtext(f"{ns}systemFieldStrength_T")
        out["field_strength_t"] = float(fs) if fs else None
        rc = asi.findtext(f"{ns}receiverChannels")
        out["n_coils"] = int(rc) if rc else None

    enc = root.find(f"{ns}encoding")
    if enc is not None:
        out["trajectory"] = enc.findtext(f"{ns}trajectory") or None
        mx = enc.find(f"{ns}encodedSpace/{ns}matrixSize")
        if mx is not None:
            out["matrix"] = [int(mx.findtext(f"{ns}{a}", "0")) for a in ("x", "y")]
        fov = enc.find(f"{ns}encodedSpace/{ns}fieldOfView_mm")
        if fov is not None:
            out["fov_mm"] = [float(fov.findtext(f"{ns}{a}", "0")) for a in ("x", "y", "z")]

    seq = root.find(f"{ns}sequenceParameters")
    if seq is not None:
        tes = [float(e.text) for e in seq.findall(f"{ns}TE") if e.text]
        out["te_ms"] = tes or None
    return out


def profile_ismrmrd(path: str | _Path) -> dict:
    """Profile an ISMRMRD ``.mrd`` / FastMRI ``.h5`` record from its embedded header.

    Reads the ISMRMRD XML (via ``h5py`` — no ``ismrmrd`` dependency) for matrix, FOV,
    field strength, coils, vendor, trajectory, and echo count. For a FastMRI-style file
    (``kspace`` + ``ismrmrd_header`` datasets) it also reads the k-space shape for the
    coil count. Trajectory and multi-echo B0 are positively rule-out-able from the
    header; raw k-space is complex (phase present).
    """
    import h5py

    def _value(node: Any) -> Any:  # read an h5py Dataset's value (weak h5py stubs)
        return node[()]

    xml_text = ""
    kspace_shape = None
    with h5py.File(path, "r") as f:
        for key in f:
            item = f[key]
            if isinstance(item, h5py.Group) and "xml" in item:
                xml_text = _xml_text(_value(item["xml"]))
                break
        if not xml_text and "ismrmrd_header" in f:
            xml_text = _xml_text(_value(f["ismrmrd_header"]))
        if "kspace" in f:
            ks: Any = f["kspace"]
            kspace_shape = tuple(int(s) for s in ks.shape)

    h = _parse_ismrmrd_xml(xml_text) if xml_text else {}
    n_coils = h.get("n_coils")
    if n_coils is None and kspace_shape is not None and len(kspace_shape) >= 4:
        n_coils = kspace_shape[1]  # FastMRI layout [slices, coils, H, W]

    geometry = {
        "matrix": h.get("matrix"),
        "n_slices": kspace_shape[0] if kspace_shape else None,
        "n_coils": n_coils,
        "fov_mm": h.get("fov_mm"),
        "voxel_spacing_mm": None,
        "domain": "kspace",
    }
    te = h.get("te_ms")
    n_echoes = len(te) if te else None
    contrast = {
        "field_strength_t": h.get("field_strength_t"),
        "te_ms": te,
        "n_echoes": n_echoes,
        "complex": True,
    }

    features: dict[str, dict] = {"phase": present("raw complex k-space")}
    traj = h.get("trajectory")
    if traj:
        features["trajectory"] = (
            absent() if traj.lower() == "cartesian" else present(f"ISMRMRD trajectory={traj}")
        )
    if n_echoes is not None:
        features["b0_field"] = (
            present(f"multi-echo ({n_echoes} TEs)") if n_echoes >= 2 else absent()
        )
    features.update(_features_from_name(path))
    return {
        "reader": "ismrmrd-xml",
        "geometry": geometry,
        "contrast": contrast,
        "features": features,
        "vendor": h.get("vendor"),
    }


def profile_dicom(path: str | _Path) -> dict:
    """Profile a DICOM record from its tags (TR/TE/flip/field/vendor/matrix)."""
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

    def _f(tag):
        v = ds.get(tag)
        return float(v) if v is not None else None

    field_t = _f("MagneticFieldStrength")
    spacing = ds.get("PixelSpacing")
    geometry = {
        "matrix": [int(ds.Rows), int(ds.Columns)] if "Rows" in ds and "Columns" in ds else None,
        "n_slices": None,
        "n_coils": None,
        "voxel_spacing_mm": [float(s) for s in spacing] if spacing else None,
        "domain": "image",
    }
    contrast = {
        "field_strength_t": field_t,
        "tr_ms": [_f("RepetitionTime")] if "RepetitionTime" in ds else None,
        "te_ms": [_f("EchoTime")] if "EchoTime" in ds else None,
        "flip_deg": [_f("FlipAngle")] if "FlipAngle" in ds else None,
        "complex": False,
    }
    features = _features_from_name(path)
    return {
        "reader": "dicom-tags",
        "geometry": geometry,
        "contrast": contrast,
        "features": features,
        "vendor": ds.get("Manufacturer"),
    }


def profile_twix(path: str | _Path) -> dict:
    """Profile a Siemens TWIX ``.dat`` record from its protocol header (header-only).

    Delegates to the dependency-free :func:`mriforge.data.twix.parse_twix_header` (no
    k-space read). The vendor is always Siemens. Multi-contrast (``lContrasts >= 2``)
    proves multi-echo B0 capability (need 1), matching the ISMRMRD logic; raw k-space is
    complex (phase). bSSFP/TrueFISP is recorded as ``contrast.sequence`` but
    ``phase_cycled_bssfp`` (need 3) is left *unknown* unless the filename names it — a
    single header cannot prove the RF phase-cycling, so it must not be guessed absent.
    """
    from mriforge.data.twix import parse_twix_header

    hdr = parse_twix_header(path)
    geometry = {
        "matrix": hdr.matrix,
        "n_slices": None,
        "n_coils": hdr.n_coils,
        "voxel_spacing_mm": None,
        "domain": "kspace",
    }
    contrast = {
        "field_strength_t": hdr.field_strength_t,
        "sequence": hdr.sequence,
        "n_echoes": hdr.n_contrasts,
        "te_ms": hdr.te_ms,
        "tr_ms": hdr.tr_ms,
        "flip_deg": hdr.flip_deg,
        "complex": True,
    }
    features: dict[str, dict] = {"phase": present("TWIX raw complex k-space")}
    if hdr.n_contrasts is not None:
        features["b0_field"] = (
            present(f"multi-contrast TWIX ({hdr.n_contrasts} echoes)")
            if hdr.n_contrasts >= 2
            else absent()
        )
    features.update(_features_from_name(path))
    return {
        "reader": "twix-header",
        "geometry": geometry,
        "contrast": contrast,
        "features": features,
        "vendor": hdr.vendor,
    }


# file_type (manifest) / suffix -> extractor. ``.ima`` is Siemens' DICOM export
# extension (uppercase on disk; the dispatch lookup lower-cases the suffix).
_SUFFIX_DISPATCH: dict[str, str] = {
    ".cfl": "bart",
    ".hdr": "bart",
    ".nii": "nifti",
    ".gz": "nifti",
    ".mrd": "ismrmrd",
    ".h5": "ismrmrd",
    ".dcm": "dicom",
    ".ima": "dicom",
    ".dat": "twix",
}
_EXTRACTORS = {
    "bart": profile_bart,
    "nifti": profile_nifti,
    "ismrmrd": profile_ismrmrd,
    "h5": profile_ismrmrd,
    "dicom": profile_dicom,
    "twix": profile_twix,
}


def profile_record(path: str | _Path, file_type: str | None = None) -> dict:
    """Dispatch one record to its header extractor (by ``file_type`` or suffix).

    Raises on an unknown format rather than guessing (pitfall #9: no silent fallback).
    """
    if file_type is not None:
        extractor = _EXTRACTORS.get(file_type)
        if extractor is None:
            raise ValueError(
                f"profile_record: unknown file_type {file_type!r} "
                f"(known: {', '.join(_EXTRACTORS)})."
            )
        return extractor(path)
    name = _Path(path).name.lower()
    kind = (
        "nifti"
        if name.endswith(".nii.gz")
        else _SUFFIX_DISPATCH.get(_Path(path).suffix.lower())  # .IMA -> .ima
    )
    if kind is None:
        raise ValueError(
            f"profile_record: cannot dispatch {_Path(path).name!r} — unknown suffix "
            f"(known: {', '.join(sorted(_SUFFIX_DISPATCH))}). Refusing to guess (pitfall #9)."
        )
    return _EXTRACTORS[kind](path)


# ============================ dataset-level aggregation ============================


def _mode(values: list) -> Any:
    """Most common non-None value (first-seen wins ties); None if all None."""
    seen: dict = {}
    order: list = []
    for v in values:
        if v is None:
            continue
        k = tuple(v) if isinstance(v, list) else v
        if k not in seen:
            seen[k] = 0
            order.append((k, v))
        seen[k] += 1
    if not order:
        return None
    best = max(order, key=lambda kv: seen[kv[0]])
    return best[1]


def aggregate_profiles(record_profiles: list[dict]) -> dict:
    """Merge per-record contributions into a dataset-level profile.

    Physics features merge by precedence ``present`` > ``absent`` > ``unknown``: a
    feature any record proves present is present for the dataset (the first such
    evidence is kept); one that some record positively rules out and none proves is
    absent; one no record could speak to stays unknown. Geometry/contrast scalars take
    the modal observed value; vendors are the observed set.
    """
    feats: dict[str, dict] = {}
    for key in PHYSICS_FEATURE_KEYS:
        states = [rp.get("features", {}).get(key) for rp in record_profiles]
        states = [s for s in states if s]
        chosen = unknown("no record header could observe this feature")
        for st in states:
            if st["state"] == PRESENT:
                chosen = st
                break
            if st["state"] == ABSENT:
                chosen = absent()
        feats[key] = chosen

    def _modal(section: str, field: str) -> Any:
        return _mode([rp.get(section, {}).get(field) for rp in record_profiles])

    geometry = {
        "matrix": _modal("geometry", "matrix"),
        "n_slices": _modal("geometry", "n_slices"),
        "n_coils": _modal("geometry", "n_coils"),
        "voxel_spacing_mm": _modal("geometry", "voxel_spacing_mm"),
        "domain": _modal("geometry", "domain"),
    }
    contrast = {
        "field_strength_t": _modal("contrast", "field_strength_t"),
        "sequence": _modal("contrast", "sequence"),
        "n_echoes": _modal("contrast", "n_echoes"),
        "te_ms": _modal("contrast", "te_ms"),
        "tr_ms": _modal("contrast", "tr_ms"),
        "flip_deg": _modal("contrast", "flip_deg"),
        "complex": _modal("contrast", "complex"),
    }
    vendors = sorted({rp.get("vendor") for rp in record_profiles if rp.get("vendor")})
    return {
        "geometry": geometry,
        "contrast": contrast,
        "physics_features": feats,
        "vendors": vendors,
    }


def _record_path(data_root: _Path, relative_path: str, file_type: str) -> _Path:
    """Absolute path of a manifest record (BART records are suffix-stripped basenames)."""
    return data_root / relative_path


def _inventory_file_types(records: list[dict], file_type: str) -> dict[str, int]:
    """Count file extensions across the records (BART = paired .cfl/.hdr per basename)."""
    if file_type == "bart":
        n = len(records)
        return {".cfl": n, ".hdr": n}
    counts: dict[str, int] = {}
    for r in records:
        name = str(r.get("relative_path", "")).lower()
        suffix = ".nii.gz" if name.endswith(".nii.gz") else _Path(name).suffix
        if suffix:
            counts[suffix] = counts.get(suffix, 0) + 1
    return counts


def profile_dataset(
    data_root: str | _Path,
    file_type: str,
    records: list[dict],
    asserted_needs: list | tuple = (),
    sample: int = 8,
    stamp: str = "",
    host: str = "",
) -> dict:
    """Build the measured ``profile`` block for one dataset (header-only).

    Samples up to ``sample`` records, profiles each header, aggregates, and computes
    ``derived_needs`` + ``needs_crosscheck`` against ``asserted_needs``. Records whose
    header cannot be read are skipped (counted), never fatal — so a single corrupt file
    does not abort the profile. Deterministic sampling (first ``sample`` records).
    """
    root = _Path(data_root)
    sampled = list(records)[: max(sample, 0)]
    record_profiles: list[dict] = []
    skipped = 0
    for r in sampled:
        path = _record_path(root, str(r.get("relative_path", "")), file_type)
        try:
            record_profiles.append(profile_record(path, file_type=file_type))
        except Exception:  # corrupt / missing / unreadable header — skip, count, continue
            skipped += 1

    agg = aggregate_profiles(record_profiles)
    inventory = {
        "file_types": _inventory_file_types(records, file_type),
        "n_records_total": len(records),
        "n_records_profiled": len(record_profiles),
        "n_records_skipped": skipped,
        "vendors": agg["vendors"],
    }
    profile = {
        "inventory": inventory,
        "geometry": agg["geometry"],
        "contrast": agg["contrast"],
        "physics_features": agg["physics_features"],
    }
    needs = derive_needs(profile)
    profile["derived_needs"] = needs["present"]
    profile["unknown_needs"] = needs["unknown"]
    profile["needs_crosscheck"] = crosscheck_needs(
        list(asserted_needs), needs["present"], needs["unknown"]
    )
    profile["provenance"] = {
        "profiled_at": stamp,
        "host": host,
        "profiler_version": "1",
        "sample": sample,
        "readers": sorted({rp["reader"] for rp in record_profiles}),
    }
    return profile


def crosscheck_manifest(manifest: dict) -> dict | None:
    """Re-verify a manifest's committed ``profile`` block against its ``source.needs``.

    Returns ``None`` if the manifest has no profile yet (cluster-only / not profiled).
    Otherwise recomputes :func:`crosscheck_needs` from the profile's measured
    ``derived_needs`` / ``unknown_needs`` — so a committed profile can be audited
    repo-side without re-reading the (cluster-only) data.
    """
    profile = manifest.get("profile")
    if not profile:
        return None
    asserted = manifest.get("source", {}).get("needs", [])
    return crosscheck_needs(
        list(asserted),
        profile.get("derived_needs", []),
        profile.get("unknown_needs", []),
    )
