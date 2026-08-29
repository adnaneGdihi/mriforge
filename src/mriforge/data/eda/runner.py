"""Orchestrate dataset EDA: classify → load → probe → render PNGs to ``<out>/<id>/``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch

from mriforge.data.eda import plots, probes
from mriforge.data.eda.catalog import DatasetEntry, discover, discover_physical
from mriforge.data.eda.loader import DatasetSample, sample


def default_output_dir(
    base: str | Path = "experiments/results", now: datetime | None = None
) -> Path:
    """A timestamped run directory ``<base>/eda_<YYYYMMDD_HHMMSS>`` (preserves run history)."""
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"eda_{ts}"


def _safe(produced: list[Path], notes: list[str], fn, *args, **kwargs) -> None:
    """Render one figure; a failure records a note and continues (EDA never aborts a suite)."""
    try:
        produced.append(fn(*args, **kwargs))
    except Exception as exc:
        notes.append(f"figure {getattr(fn, '__name__', fn)} failed: {type(exc).__name__}: {exc}")


# Note fragments that mean a dataset was *not read / not shown correctly* — a figure render or
# probe crashed, the loader found nothing, a file threw, or a "present" dataset had no resolvable
# scan dir. Surfacing these into coverage.json stops a degraded dataset from masquerading as a
# clean ``error=null`` row (the silent-degradation surface CLAUDE.md #10 forbids). The by-design
# non-Cartesian "…probes need NUFFT gridding — skipped" note is intentionally NOT a marker.
_ISSUE_MARKERS = (
    "failed:",  # figure render / probe crash (render_montage, kspace probes, profile)
    "no readable files",  # loader globbed the scan dir and read nothing
    "error reading",  # a specific file raised while being read
    "skipped malformed",  # a degenerate 1-D / non-plane read was dropped (osi2one wrong-key bug)
    "absent or no scan dir",  # absent tier, or present-tier with an unresolvable data_root
)


def _issues(notes: list[str]) -> list[str]:
    """Subset of ``notes`` that signal a genuine read/render failure (see ``_ISSUE_MARKERS``)."""
    return [n for n in notes if any(m in n for m in _ISSUE_MARKERS)]


def _read_notes(profile_path: Path) -> list[str]:
    """Best-effort read of the per-dataset ``profile.json`` notes (empty list if unreadable)."""
    if not profile_path.exists():
        return []
    try:
        return json.loads(profile_path.read_text()).get("notes", []) or []
    except Exception:
        return []


def _quality(images: list[torch.Tensor]) -> dict:
    return {
        "records": len(images),
        "nan_slices": sum(int(torch.isnan(im).any()) for im in images),
        "zero_slices": sum(int((im == 0).all()) for im in images),
    }


def _inventory(entry: DatasetEntry, s: DatasetSample) -> dict[str, dict]:
    counts: dict[str, dict] = {"contrast": {}, "field_T": {}}
    contrast = (s.profile.get("contrast") or {}) if s.profile else {}
    if isinstance(contrast, dict):
        for k, v in contrast.items():
            if isinstance(v, (int, float)):
                counts["contrast"][str(k)] = v
    for ft in entry.source.get("field_T") or []:
        counts["field_T"][str(ft)] = counts["field_T"].get(str(ft), 0) + 1
    return counts


def run_dataset(entry: DatasetEntry, out_root: Path, budget: int = 8) -> list[Path]:
    out_dir = Path(out_root) / entry.dataset_id
    produced: list[Path] = []
    s = sample(entry, budget=budget)
    n = s.notes
    _safe(produced, n, plots.render_card, entry, s.profile, out_dir)

    # Card-only early-out when there is nothing to visualize. Non-Cartesian datasets have raw
    # k-space but (intentionally) no recon image, so s.kspace must count as "something to show".
    if entry.tier != "present" or (not s.images and not s.pairs and not s.kspace and not s.frames):
        _write_provenance(entry, s, produced, out_dir)
        return produced

    norm_sample: list[float] = []
    if s.images:
        dist = probes.normalized_distributions(s.images)
        norm_sample = dist["max"].astype(float).tolist()[:2000]
        _safe(produced, n, plots.render_quality, _quality(s.images), out_dir)
        _safe(produced, n, plots.render_inventory, _inventory(entry, s), out_dir)
        _safe(
            produced,
            n,
            plots.render_montage,
            s.images,
            out_dir,
            "10_sample_montage.png",
            entry.dataset_id,
        )
        _safe(produced, n, plots.render_norm_voxel_dist, dist, out_dir)
        _safe(produced, n, plots.render_percentile_table, probes.percentiles(s.images), out_dir)

    if entry.modality in ("kspace", "noncartesian") and s.kspace and s.kspace[0].ndim >= 2:
        try:
            ksp = s.kspace[0]
            # The raw k-space log-magnitude is the honest acquired-samples view for both grids.
            _safe(produced, n, plots.render_kspace_logmag, ksp, out_dir)
            if entry.modality == "kspace":
                # Cartesian-only probes: radial binning, phase-encode sampling, ifft recon and
                # ESPIRiT all assume a Cartesian grid. Running them on radial/spiral samples
                # yields physically meaningless figures, so they are gated to Cartesian k-space.
                radii, energy, cumulative = probes.radial_energy(ksp)
                _safe(produced, n, plots.render_radial_energy, radii, energy, cumulative, out_dir)
                info = probes.detect_sampling(ksp)
                _safe(produced, n, plots.render_sampling_pattern, info["mask"], info, out_dir)
                csm = probes.coil_maps(ksp)
                if csm is not None:
                    _safe(produced, n, plots.render_coil_maps, csm, out_dir)
                if s.complex_images:
                    ph = probes.phase_stats(s.complex_images[0])
                    _safe(produced, n, plots.render_phase, ph["phase"], ph["grad_mag"], out_dir)
            else:
                n.append(
                    "non-Cartesian: image recon, sampling, radial-energy and ESPIRiT probes need "
                    "NUFFT gridding (trajectory) — skipped; showing raw k-space log-magnitude only"
                )
        except Exception as exc:
            n.append(f"kspace probes failed: {type(exc).__name__}: {exc}")

    if entry.modality == "paired" and s.pairs:
        _safe(produced, n, plots.render_paired_montage, s.pairs, out_dir)
        _safe(produced, n, plots.render_paired_bland_altman, s.pairs[0][0], s.pairs[0][1], out_dir)

    if entry.modality == "map" and s.images:
        _safe(produced, n, plots.render_map_montage, s.images, out_dir, "a.u.")

    if entry.modality == "temporal" and s.frames:
        # cine / 4D-flow / 5D: the dedicated time-axis view (sampled frames + temporal std).
        _safe(produced, n, plots.render_temporal_strip, s.frames, out_dir)

    _write_provenance(entry, s, produced, out_dir, norm_sample)
    return produced


def _write_provenance(
    entry, s, produced, out_dir: Path, norm_sample: list[float] | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.json").write_text(
        json.dumps(
            {
                "dataset_id": entry.dataset_id,
                "modality": entry.modality,
                "tier": entry.tier,
                "status": entry.status,
                "scan_dir": str(s.scan_dir) if s.scan_dir else None,
                "figures": [p.name for p in produced],
                "notes": s.notes,
                "norm_intensity": norm_sample or [],
                "profile": s.profile,
            },
            indent=1,
            default=str,
        )
    )


def run_all(
    manifests_root: Path,
    data_root: Path,
    out_root: Path,
    budget: int = 8,
    databases_dir: Path | None = None,
) -> list[dict]:
    """Run EDA over every manifest plus any on-disk datasets without a covering manifest."""
    entries = discover(Path(manifests_root), Path(data_root))
    if databases_dir is not None:
        entries = entries + discover_physical(Path(databases_dir), entries)

    rows: list[dict] = []
    overview_rows: list[dict] = []
    for entry in entries:
        try:
            produced = run_dataset(entry, out_root, budget=budget)
            notes = _read_notes(out_root / entry.dataset_id / "profile.json")
            row = {
                "dataset_id": entry.dataset_id,
                "modality": entry.modality,
                "tier": entry.tier,
                "n_figures": len(produced),
                "figures": [p.name for p in produced],
                "error": None,
                "notes": notes,
                "issues": _issues(notes),
            }
        except Exception as exc:
            row = {
                "dataset_id": entry.dataset_id,
                "modality": entry.modality,
                "tier": entry.tier,
                "n_figures": 0,
                "figures": [],
                "error": repr(exc),
                "notes": [],
                "issues": [repr(exc)],
            }
        rows.append(row)
        overview_rows.append(_overview_row(entry, row, Path(out_root)))

    from mriforge.data.eda.overview import render_overview

    render_overview(overview_rows, Path(out_root) / "_overview")
    return rows


def _overview_row(entry: DatasetEntry, row: dict, out_root: Path) -> dict:
    """Build a cross-dataset row, reading the normalized-intensity sample written during the
    per-dataset pass (no second voxel load)."""
    norm: list[float] = []
    prof_path = out_root / entry.dataset_id / "profile.json"
    if prof_path.exists():
        try:
            norm = json.loads(prof_path.read_text()).get("norm_intensity", []) or []
        except Exception:
            norm = []
    return {
        **row,
        "field_T": entry.source.get("field_T"),
        "anatomy": entry.source.get("anatomy"),
        "role": entry.source.get("role"),
        "status": entry.status,
        "n_records": len(entry.records),
        "norm_intensity": norm,
    }
