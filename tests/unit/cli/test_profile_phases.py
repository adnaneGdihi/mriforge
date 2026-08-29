"""Tests for the train/validation phase split of a Scalene profile.

The frame strings here are the real 2.3.0 shape (``"<file> <func>:<line>;"``),
taken from an actual profile rather than invented, because a fixture that agrees
with the docstring instead of with the producer is green and blind.
"""

from __future__ import annotations

import json
from pathlib import Path

import mriforge
from mriforge.cli.profile_phases import (
    MAX_HOTSPOTS,
    PHASE_MARKERS,
    UNATTRIBUTED,
    build_phase_report,
    classify_stack,
    parse_frame,
    split_stacks,
    write_phase_reports,
)

SRC = "/repo/src/mriforge"
TRAIN_FRAME = f"{SRC}/pipelines/training_loop.py _execute_training_loop:812;"
VAL_FRAME = f"{SRC}/pipelines/train.py _run_validation:1955;"
FORWARD_FRAME = f"{SRC}/models/generators/unet.py forward:214;"
SETUP_FRAME = f"{SRC}/bootstrap.py build_container:88;"


def _stack(frames, cpu=1.0, count=1):
    return [list(frames), {"cpu_samples": cpu, "c_time": 0.0, "python_time": cpu, "count": count}]


# ------------------------------------------------------------------ parsing


def test_parse_frame_reads_the_real_scalene_shape():
    frame = parse_frame(TRAIN_FRAME)
    assert frame is not None
    assert frame.filename == f"{SRC}/pipelines/training_loop.py"
    assert frame.function == "_execute_training_loop"
    assert frame.lineno == 812


def test_parse_frame_returns_none_for_junk_rather_than_raising():
    """One odd frame must not sink a whole profile."""
    for junk in ("", "no-colon-here", "file.py func:notanint", ":"):
        assert parse_frame(junk) is None


# -------------------------------------------------- the precedence landmine


def test_a_stack_with_both_markers_is_validation():
    """THE planted violation: every in-training validation stack contains BOTH.

    ``_run_validation`` is called *by* ``_execute_training_loop``, so a sampled
    validation stack always carries the training marker too. If ``PHASE_MARKERS``
    is ever reordered — or the loop is rewritten to test train first — the entire
    validation phase silently folds into the training bucket and the split
    reports a plausible, completely wrong answer. This test is the only thing
    standing between that edit and a believable number.
    """
    assert classify_stack([TRAIN_FRAME, VAL_FRAME, FORWARD_FRAME]) == "validation"


def test_marker_order_puts_validation_before_train():
    """The property the test above depends on, asserted directly."""
    phases = [m.phase for m in PHASE_MARKERS]
    assert phases.index("validation") < phases.index("train")


# --------------------------------------------------------- marker matching


def test_train_only_stack_is_train():
    assert classify_stack([TRAIN_FRAME, FORWARD_FRAME]) == "train"


def test_unmarked_stack_is_other_not_forced_into_a_phase():
    assert classify_stack([SETUP_FRAME]) == UNATTRIBUTED


def test_function_name_is_matched_exactly_not_as_a_substring():
    """``'_run_validation' in frame`` would also match a cousin name."""
    cousin = f"{SRC}/pipelines/train.py _run_validation_cascade:2100;"
    assert classify_stack([cousin]) == UNATTRIBUTED


def test_file_suffix_must_match_too():
    """A same-named helper in another module is not the validation entry point."""
    impostor = f"{SRC}/scratch/other.py _run_validation:5;"
    assert classify_stack([impostor]) == UNATTRIBUTED


# ------------------------------------------------------- the point: callees


def test_callee_time_lands_in_the_calling_phase():
    """Why stacks and not ``--profile-only``.

    The same ``models/…/unet.py forward`` line runs under both phases. Filename
    filtering cannot tell the two apart; stack membership can, and must.
    """
    profile = {
        "stacks": [
            _stack([TRAIN_FRAME, FORWARD_FRAME], cpu=8.0),
            _stack([TRAIN_FRAME, VAL_FRAME, FORWARD_FRAME], cpu=2.0),
        ]
    }
    totals = split_stacks(profile)
    assert totals is not None
    assert totals["train"].cpu_share == 8.0
    assert totals["validation"].cpu_share == 2.0
    # The forward frame is the leaf of both, and appears as a hotspot in each.
    key = (f"{SRC}/models/generators/unet.py", "forward", 214)
    assert totals["train"].hotspots[key] == 8.0
    assert totals["validation"].hotspots[key] == 2.0


# --------------------------------------------------------------- absence


def test_a_profile_without_stacks_is_unavailable_not_empty():
    """Absent is a state to report, never one to infer (non-negotiable 18)."""
    assert split_stacks({"files": {}}) is None
    assert split_stacks({"stacks": []}) is None


def test_write_records_why_the_split_did_not_run(tmp_path):
    record = write_phase_reports(tmp_path, {"files": {}})
    assert record["status"] == "unavailable"
    assert "stacks" in record["reason"]
    assert not (tmp_path / "phases").exists()


# ---------------------------------------------------------------- reports


def test_write_emits_one_report_per_phase_plus_a_summary(tmp_path):
    profile = {
        "stacks": [
            _stack([TRAIN_FRAME, FORWARD_FRAME], cpu=6.0),
            _stack([TRAIN_FRAME, VAL_FRAME, FORWARD_FRAME], cpu=3.0),
            _stack([SETUP_FRAME], cpu=1.0),
        ]
    }
    record = write_phase_reports(tmp_path, profile)
    assert record["status"] == "written"

    phases = tmp_path / "phases"
    for name in ("train", "validation", UNATTRIBUTED, "summary"):
        assert (phases / f"{name}.json").exists(), name

    summary = json.loads((phases / "summary.json").read_text())
    assert summary["attributed_cpu_share"] == 10.0
    assert summary["phases"]["train"]["share_of_run"] == 0.6
    assert summary["phases"]["validation"]["share_of_run"] == 0.3
    assert summary["phases"][UNATTRIBUTED]["share_of_run"] == 0.1


def test_a_phase_report_says_what_it_does_not_cover(tmp_path):
    """A file named ``validation.json`` must not read as "the validation profile"."""
    profile = {"stacks": [_stack([TRAIN_FRAME, VAL_FRAME], cpu=1.0)]}
    write_phase_reports(tmp_path, profile)
    report = json.loads((tmp_path / "phases" / "validation.json").read_text())
    assert report["metric"].startswith("share_of_profiled_cpu")
    assert set(report["excludes"]) == {"memory", "gpu"}


def test_hotspot_truncation_is_reported_never_silent(tmp_path):
    """A capped list that does not say it was capped reads as "that was all"."""
    stacks = [
        _stack([TRAIN_FRAME, f"{SRC}/models/m.py f{i}:{i};"], cpu=float(i + 1))
        for i in range(MAX_HOTSPOTS + 7)
    ]
    totals = split_stacks({"stacks": stacks})
    assert totals is not None
    report = build_phase_report("train", totals["train"], run_cpu_share=1.0)
    assert len(report["hotspots"]) == MAX_HOTSPOTS
    assert report["hotspots_truncated"] == 7


def test_hotspots_are_ranked_by_cost(tmp_path):
    stacks = [
        _stack([TRAIN_FRAME, f"{SRC}/models/m.py cheap:1;"], cpu=1.0),
        _stack([TRAIN_FRAME, f"{SRC}/models/m.py dear:2;"], cpu=9.0),
    ]
    totals = split_stacks({"stacks": stacks})
    assert totals is not None
    report = build_phase_report("train", totals["train"], run_cpu_share=10.0)
    assert [h["function"] for h in report["hotspots"]] == ["dear", "cheap"]
    assert report["hotspots"][0]["share_of_phase"] == 0.9


def test_malformed_stack_entries_are_skipped_not_fatal():
    profile = {"stacks": [["not-a-pair"], _stack([TRAIN_FRAME], cpu=2.0), 42]}
    totals = split_stacks(profile)
    assert totals is not None
    assert totals["train"].cpu_share == 2.0


# ------------------------------------------------------------- marker drift


def test_the_marked_functions_still_exist_where_the_markers_say():
    """Guards the one failure this split cannot detect at runtime.

    A rename or a move of either entry point leaves ``classify_stack`` matching
    nothing: the validation bucket reads 0.0, which is indistinguishable from
    "validation never fired". Pin the markers to the source so the drift fails
    here — loudly, in CI — instead of silently in a report someone trusts.
    """
    package_root = Path(mriforge.__file__).parent
    for marker in PHASE_MARKERS:
        # file_suffix is already package-relative ("pipelines/train.py").
        source = (package_root / marker.file_suffix).read_text(encoding="utf-8")
        assert f"def {marker.function}(" in source, (
            f"{marker.function} is gone from {marker.file_suffix} — the "
            f"'{marker.phase}' phase would silently measure 0.0. Update "
            f"PHASE_MARKERS in mriforge/cli/profile_phases.py."
        )


# ------------------------------------------------------------------ the unit


def test_summary_carries_the_runs_wall_clock_as_the_only_seconds_field(tmp_path):
    """``cpu_share`` is a fraction; the one real duration must survive to the report.

    Without this the summary has no seconds anywhere and a reader reaches for
    ``cpu_share`` as if it were one — which is exactly the mislabelling this
    field exists to prevent.
    """
    profile = {"elapsed_time_sec": 412.88, "stacks": [_stack([TRAIN_FRAME], cpu=1.0)]}
    write_phase_reports(tmp_path, profile)
    summary = json.loads((tmp_path / "phases" / "summary.json").read_text())
    assert summary["elapsed_time_sec"] == 412.88
    assert "cpu_seconds" not in json.dumps(summary)


def test_producer_shaped_shares_summing_to_one_partition_exactly(tmp_path):
    """Real scalene output normalizes: every stack's ``cpu_samples`` sums to 1.0.

    The other tests use whole numbers to prove the arithmetic makes no
    normalization assumption. This one uses the shape the producer actually
    emits, so the fixture family covers both (see the E2E figures in the PR:
    0.799 train / 0.201 validation on a 3:1 workload).
    """
    profile = {
        "elapsed_time_sec": 0.4475691318511963,
        "stacks": [
            _stack([TRAIN_FRAME, FORWARD_FRAME], cpu=0.798869),
            _stack([TRAIN_FRAME, VAL_FRAME, FORWARD_FRAME], cpu=0.201131),
        ],
    }
    totals = split_stacks(profile)
    assert totals is not None
    assert round(sum(t.cpu_share for t in totals.values()), 6) == 1.0

    write_phase_reports(tmp_path, profile)
    summary = json.loads((tmp_path / "phases" / "summary.json").read_text())
    assert summary["attributed_cpu_share"] == 1.0
    assert summary["phases"]["train"]["share_of_run"] == 0.798869
    assert summary["phases"]["validation"]["share_of_run"] == 0.201131
    assert summary["phases"][UNATTRIBUTED]["share_of_run"] == 0.0
