r"""Reconstruct plotter case-payloads from a recorded ``report_cases/`` dir.

Two data sources, tried in order:

1. ``report_cases/cases_index.json`` + ``case_*.npz`` — the canonical recorder
   output (``input``/``prediction``/``target`` magnitude arrays + per-case
   metrics).
2. Downloaded validation PNG pairs (``real_images``/``fake_images``) — the
   fallback for the standalone ``mriforge report`` path when a run's npz cases
   were not recorded but its validation images were. ``real`` = target,
   ``fake`` = prediction; the latest epoch/step group is used.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mriforge.infrastructure.run_layout import drop_profiling_artifacts

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"epoch(\d+)_step(\d+)")

_TASK_CASE_KEY = {
    "reconstruction": "cases_kspace",
    "super_resolution": "cases_sr",
    "synthesis": "cases_synth",
    "default": "cases_kspace",
    "gan": "cases_synth",
    "diffusion": "cases_kspace",
    "vae": "cases_synth",
}

# Aliases bridging the recorder's canonical array names (``input`` /
# ``prediction`` / ``target``) onto the per-plotter key vocabulary. The MRI
# qualitative plotters (e.g. ``mri_a7_kspace_recon``) read ``zero_filled`` /
# ``baseline`` / ``pred`` / ``fs_ref`` rather than the canonical names, so we
# add these as non-destructive aliases — the canonical keys are preserved too.
_PLOTTER_ALIASES = {
    # kspace recon plotter (mri_a7_kspace_recon)
    "zero_filled": "input",
    "pred": "prediction",
    "fs_ref": "target",
    # ``baseline`` has no recorded counterpart; fall back to zero-filled input
    # so the comparison column renders rather than crashing on a missing key.
    "baseline": "input",
    # super-resolution triptych plotter (mri_a1_sr_triptych) reads lr/pred/hr;
    # the synthesis plotter reads input/pred/target (input & target canonical).
    "lr": "input",
    "hr": "target",
}


def _apply_plotter_aliases(arrays: dict) -> dict:
    """Add plotter-facing key aliases without clobbering existing keys."""
    out = dict(arrays)
    for alias, source in _PLOTTER_ALIASES.items():
        if alias not in out and source in arrays:
            out[alias] = arrays[source]
    return out


def _read_gray(path: Path) -> np.ndarray:
    """Read a PNG as a 2-D float32 magnitude array (RGB collapsed to luminance)."""
    import matplotlib.image as mpimg

    arr = np.asarray(mpimg.imread(str(path)), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.float32)


def _find_image_dirs(run_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate the ``real_images`` / ``fake_images`` dirs for a downloaded run."""
    for base in (run_dir / "metrics", run_dir / "logs" / "metrics", run_dir, run_dir / "logs"):
        real, fake = base / "real_images", base / "fake_images"
        if real.is_dir() and fake.is_dir():
            return real, fake
    # Recursive, so it reaches into `profiles/<run_id>/run/` — where `mriforge
    # profile` files a THROWAWAY run of this same arm, PNGs and all. Those are a
    # measurement of the arm, not the arm's validation output, and this fallback
    # only fires when the arm has none of its own — exactly when the throwaway
    # would be adopted as the report's cases. One owner: `run_layout`.
    reals = drop_profiling_artifacts(list(run_dir.rglob("real_images")))
    fakes = drop_profiling_artifacts(list(run_dir.rglob("fake_images")))
    if reals and fakes:
        return reals[0], fakes[0]
    return None, None


def load_image_pair_cases(run_dir: str | Path, *, max_cases: int = 12) -> list[dict]:
    """Synthesize image cases from downloaded validation PNG pairs.

    ``real`` PNGs are targets, ``fake`` PNGs predictions. Uses the latest
    epoch/step group so the mosaics reflect the most-trained validation pass.
    Returns [] when no pairs are found (metrics are empty — PNGs carry none).
    """
    run_dir = Path(run_dir)
    real_dir, fake_dir = _find_image_dirs(run_dir)
    if real_dir is None or fake_dir is None:
        return []
    reals = sorted(real_dir.glob("*.png"))
    if not reals:
        return []

    def _grp(p: Path) -> tuple[int, int]:
        m = _STEP_RE.search(p.name)
        return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

    groups: dict[tuple[int, int], list[Path]] = {}
    for p in reals:
        groups.setdefault(_grp(p), []).append(p)
    latest = max(groups)
    epoch, step = latest
    cases: list[dict] = []
    for i, rp in enumerate(sorted(groups[latest])[:max_cases]):
        fp = fake_dir / rp.name.replace("_real_", "_fake_")
        if not fp.exists():
            continue
        arrays = _apply_plotter_aliases(
            {"target": _read_gray(rp), "prediction": _read_gray(fp), "input": _read_gray(rp)}
        )
        cases.append(
            {
                **arrays,
                "metrics": {},
                "domain": {},
                "rank": "case",
                "case_id": f"epoch{epoch:03d}_step{step:06d}_{i:03d}",
            }
        )
    if cases:
        logger.info("report cases: recovered %d from PNG pairs under %s", len(cases), real_dir)
    return cases


def load_report_cases(run_dir: str | Path, *, task: str, subdir: str = "report_cases") -> dict:
    """Load recorded cases and route them into plotter kwargs.

    Returns a dict possibly containing ``cases_kspace`` / ``cases_sr`` /
    ``cases_synth`` (image payloads), ``cases_failure`` (worst-ranked),
    ``predictions_df`` and ``stratified_df`` (per-case metric frames).
    Empty dict when no cases were recorded (and no PNG pairs recoverable).
    """
    cases_dir = Path(run_dir) / subdir
    index_path = cases_dir / "cases_index.json"
    cases: list[dict] = []
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for row in index:
            npz_path = cases_dir / row["npz"]
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                arrays = {k: data[k] for k in data.files}
            arrays = _apply_plotter_aliases(arrays)
            cases.append(
                {
                    **arrays,
                    "metrics": row["metrics"],
                    "domain": row["domain"],
                    "rank": row["rank"],
                    "case_id": row["case_id"],
                }
            )
    if not cases:
        # Fallback: recover cases from downloaded validation PNG pairs.
        cases = load_image_pair_cases(run_dir)
    if not cases:
        return {}

    image_key = _TASK_CASE_KEY.get(str(task).lower(), "cases_kspace")
    failures = [c for c in cases if c["rank"] == "worst"] or cases[-1:]

    rows = []
    for c in cases:
        for m, v in c["metrics"].items():
            rows.append(
                {
                    "method": "model",
                    "metric": m,
                    "value": float(v),
                    "subject_id": c["case_id"],
                    "split": "test",
                    **{f"strat_{k}": val for k, val in c["domain"].items()},
                }
            )
    pred_df = pd.DataFrame(rows)

    payload = {
        image_key: cases,
        "cases_failure": failures,
        "predictions_df": pred_df,
        # independent copy so a plotter mutating one frame can't alias the other
        "stratified_df": pred_df.copy(),
    }
    return payload
