"""Split one Scalene profile into separate training / validation reports.

Scalene writes **one outfile per process**, and its ``--profile-only`` narrows by
*filename substring* — so neither mechanism can separate the training phase from
the validation phase. The two interleave inside a single run and share their
callees: a UNet forward costs the same ``models/…py:N`` whether it runs under
``_execute_training_loop`` or under ``_run_validation``. Filtering by file can
only ever split the *frames*, never the *phases*.

What makes a real split possible is that scalene 2.3.0 records full call stacks by
default (top-level ``stacks``), each with a ``cpu_samples`` *fraction* of the run's
CPU (they sum to 1.0) -- never seconds, hence ``cpu_share`` throughout. Attributing
a sample by **which markers appear in its stack** captures callee time: a forward
under validation is validation time though its frame is in ``models/``. Hence
``stacks``, not the ``files`` section.

Three limits, stated rather than left to be discovered. **CPU only:** the payload
carries ``cpu_samples``/``c_time``/``python_time``/``count`` and nothing else, so
memory and GPU **cannot** be phase-split — hence the ``metric``/``excludes`` fields
on every report, without which a reader finds no memory column and concludes
validation allocates nothing. **Whole-run scope:** ``stacks`` is *not* filtered by
``--profile-only`` (verified against 2.3.0), so the split covers the whole run even
under ``--focus``. **Child processes are unseen:** Scalene profiles only the process
it launches, so with ``num_workers > 0`` batch preparation lands in NO bucket — a
data-bound run can look compute-bound. Both are carried in ``summary.json`` so the
artifact states its own scope.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Hotspot rows kept per phase — capped so a 2,000-frame run does not write a
#: report nobody reads, and never silently: ``hotspots_truncated`` records the cut.
MAX_HOTSPOTS = 40


@dataclass(frozen=True)
class PhaseMarker:
    """One (file, function) pair whose presence in a stack names a phase.

    Matched as a **path suffix plus an exact function name**, never as a bare
    substring of the raw frame: ``'_run_validation' in frame`` would also match a
    future ``_run_validation_cascade`` and misfile its time.
    """

    phase: str
    file_suffix: str
    function: str


#: Marker table, **most specific first** — :func:`classify_stack` returns the first
#: phase that matches, and that order is load-bearing. Every in-training validation
#: stack *also* contains ``_execute_training_loop``, because that is its caller, so
#: validation must be tested first or the whole phase is absorbed into training and
#: the split reports a plausible, entirely wrong answer. Pinned by
#: ``test_profile_phases.py::test_a_stack_with_both_markers_is_validation``.
#: ``_run_validation`` lives in ``pipelines/train.py`` (not ``training_loop.py``) and
#: is what both the in-training gate and ``TrainingLoop.evaluate`` drive.
PHASE_MARKERS: tuple[PhaseMarker, ...] = (
    PhaseMarker("validation", "pipelines/train.py", "_run_validation"),
    PhaseMarker("train", "pipelines/training_loop.py", "_execute_training_loop"),
)

#: Where a sample lands when no marker appears in its stack. Setup, loader
#: startup, checkpoint writes and teardown belong to neither phase; a two-bucket
#: split would force them into one and overstate it. Reported, never dropped.
UNATTRIBUTED = "other"

PHASES: tuple[str, ...] = ("train", "validation", UNATTRIBUTED)


@dataclass(frozen=True)
class Frame:
    """One parsed stack frame: ``"<file> <func>:<lineno>;"``."""

    filename: str
    function: str
    lineno: int


def parse_frame(raw: str) -> Frame | None:
    """Parse one scalene stack frame, or ``None`` if it is not that shape.

    ``None`` rather than raising: one odd frame must not sink a whole profile, and
    a stack that parses nowhere classifies as :data:`UNATTRIBUTED` — visible.
    """
    text = raw.strip().rstrip(";")
    head, sep, lineno = text.rpartition(":")
    if not sep:
        return None
    filename, sep, function = head.rpartition(" ")
    if not sep:
        return None
    try:
        return Frame(filename=filename, function=function, lineno=int(lineno))
    except ValueError:
        return None


def classify_stack(frames: list[str]) -> str:
    """Name the phase a sampled stack belongs to.

    Marker order decides — see :data:`PHASE_MARKERS` for why validation first.
    """
    parsed = [f for f in (parse_frame(r) for r in frames) if f is not None]
    for marker in PHASE_MARKERS:
        for frame in parsed:
            if frame.function == marker.function and frame.filename.endswith(marker.file_suffix):
                return marker.phase
    return UNATTRIBUTED


def _leaf(frames: list[str]) -> Frame | None:
    """The innermost parseable frame — where the time was actually spent."""
    for raw in reversed(frames):
        parsed = parse_frame(raw)
        if parsed is not None:
            return parsed
    return None


@dataclass
class PhaseTotals:
    """Accumulated cost for one phase."""

    cpu_share: float = 0.0
    python_share: float = 0.0
    c_share: float = 0.0
    samples: int = 0
    #: ``(file, func, lineno)`` -> cpu share, keyed on the leaf frame.
    hotspots: dict[tuple[str, str, int], float] | None = None

    def __post_init__(self) -> None:
        if self.hotspots is None:
            self.hotspots = {}


def split_stacks(profile: dict[str, Any]) -> dict[str, PhaseTotals] | None:
    """Partition a loaded scalene profile's samples by phase.

    Returns ``None`` when the profile carries no ``stacks`` — a state to report,
    never one to infer: the caller records *why* the split is unavailable instead
    of writing an empty report that reads as "no validation time".
    """
    stacks = profile.get("stacks")
    if not stacks:
        return None

    totals = {phase: PhaseTotals() for phase in PHASES}
    for entry in stacks:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        frames, payload = entry
        if not isinstance(frames, list) or not isinstance(payload, dict):
            continue
        bucket = totals[classify_stack(frames)]
        cpu = float(payload.get("cpu_samples", 0.0) or 0.0)
        bucket.cpu_share += cpu
        bucket.python_share += float(payload.get("python_time", 0.0) or 0.0)
        bucket.c_share += float(payload.get("c_time", 0.0) or 0.0)
        bucket.samples += int(payload.get("count", 0) or 0)
        leaf = _leaf(frames)
        if leaf is not None and bucket.hotspots is not None:
            key = (leaf.filename, leaf.function, leaf.lineno)
            bucket.hotspots[key] = bucket.hotspots.get(key, 0.0) + cpu
    return totals


def build_phase_report(phase: str, totals: PhaseTotals, *, run_cpu_share: float) -> dict[str, Any]:
    """Render one phase's findings as the dict written to ``phases/<phase>.json``.

    ``metric``/``excludes`` ride in the payload on purpose — see the module
    docstring on why a file named ``validation.json`` must say what it is not.
    """
    ranked = sorted((totals.hotspots or {}).items(), key=lambda kv: kv[1], reverse=True)
    kept = ranked[:MAX_HOTSPOTS]
    return {
        "phase": phase,
        "attribution": "call-stack membership (captures callee time)",
        "metric": "share_of_profiled_cpu (scalene cpu_samples; NOT seconds)",
        "excludes": ["memory", "gpu"],
        "note": "Memory/GPU: see scalene-profile.json. Whole run, even under --focus.",
        "marker": next(
            (
                {"file_suffix": m.file_suffix, "function": m.function}
                for m in PHASE_MARKERS
                if m.phase == phase
            ),
            None,
        ),
        "totals": {
            "cpu_share": round(totals.cpu_share, 6),
            "python_share": round(totals.python_share, 6),
            "c_share": round(totals.c_share, 6),
            "samples": totals.samples,
            "share_of_run": (round(totals.cpu_share / run_cpu_share, 6) if run_cpu_share else None),
        },
        "hotspots_truncated": len(ranked) - len(kept),
        "hotspots": [
            {
                "file": filename,
                "function": function,
                "lineno": lineno,
                "cpu_share": round(cpu, 6),
                "share_of_phase": (round(cpu / totals.cpu_share, 6) if totals.cpu_share else None),
            }
            for (filename, function, lineno), cpu in kept
        ],
    }


def write_phase_reports(profile_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """Write ``phases/<phase>.json`` + ``phases/summary.json``; return the record.

    The returned dict is folded into ``profile_manifest.json`` so the manifest
    always states whether the split ran and, when it did not, why.
    """
    totals = split_stacks(profile)
    if totals is None:
        logger.warning(
            "Scalene profile carries no per-sample stacks — no train/validation "
            "split written. The full profile is unaffected."
        )
        return {"status": "unavailable", "reason": "profile contains no 'stacks'"}

    run_cpu = sum(t.cpu_share for t in totals.values())
    phase_dir = profile_dir / "phases"
    phase_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for phase, phase_totals in totals.items():
        report = build_phase_report(phase, phase_totals, run_cpu_share=run_cpu)
        path = phase_dir / f"{phase}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        written[phase] = str(path)

    summary = {
        "attribution": "call-stack membership (captures callee time)",
        "metric": "share_of_profiled_cpu (scalene cpu_samples; NOT seconds)",
        "excludes": ["memory", "gpu"],
        "caveats": [
            "Child processes are not profiled: with num_workers > 0, batch "
            "preparation appears in no bucket.",
        ],
        "attributed_cpu_share": round(run_cpu, 6),
        "elapsed_time_sec": profile.get("elapsed_time_sec"),
        "phases": {
            phase: {
                "cpu_share": round(t.cpu_share, 6),
                "share_of_run": round(t.cpu_share / run_cpu, 6) if run_cpu else None,
            }
            for phase, t in totals.items()
        },
    }
    summary_path = phase_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for phase, t in totals.items():
        share = (t.cpu_share / run_cpu * 100) if run_cpu else 0.0
        logger.info("  %-11s %5.1f%% of profiled CPU", phase, share)

    # A validation bucket of exactly zero is ambiguous — validation never fired,
    # or the marker drifted after a rename. Both are real, so neither is asserted.
    if totals["validation"].cpu_share == 0.0:
        logger.warning(
            "No validation time attributed. Either validation never fired in this "
            "run, or PHASE_MARKERS drifted from the source (see profile_phases)."
        )

    return {"status": "written", "summary": str(summary_path), "reports": written}


__all__ = [
    "MAX_HOTSPOTS",
    "PHASES",
    "PHASE_MARKERS",
    "UNATTRIBUTED",
    "Frame",
    "PhaseMarker",
    "PhaseTotals",
    "build_phase_report",
    "classify_stack",
    "parse_frame",
    "split_stacks",
    "write_phase_reports",
]
