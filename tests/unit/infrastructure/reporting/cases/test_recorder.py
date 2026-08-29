import json

import numpy as np

from mriforge.infrastructure.reporting.cases.recorder import ReportCaseRecorder


def test_recorder_keeps_best_median_worst_and_writes(tmp_path):
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
    )
    for i in range(10):
        rec.observe(
            case_id=f"s{i}",
            arrays={
                "input": np.zeros((4, 4), dtype=np.float32),
                "prediction": np.full((4, 4), float(i), dtype=np.float32),
                "target": np.ones((4, 4), dtype=np.float32),
            },
            metrics={"psnr": float(i)},
            domain={"acceleration": 4},
        )
    out = rec.write(tmp_path)
    index = json.loads((out / "cases_index.json").read_text())
    assert len(index) == 3
    ranks = {row["rank"] for row in index}
    assert ranks == {"best", "median", "worst"}
    by_rank = {row["rank"]: row for row in index}
    assert by_rank["best"]["metrics"]["psnr"] == 9.0
    assert by_rank["worst"]["metrics"]["psnr"] == 0.0
    for row in index:
        assert (out / row["npz"]).exists()


def test_recorder_ranks_when_metrics_carry_monitor_prefix(tmp_path):
    """The feed seam stores ``val_<metric>`` keys; ranking must still work.

    Regression for the silent inversion: ``validation.primary_metric`` is a bare
    name (``psnr``) but ``feed_report_case_recorder`` stores the validation dict
    verbatim (``val_psnr``). An exact-match lookup never fires, so every case
    scores 0.0 and best/median/worst collapse to insertion order — a 14.5 dB case
    was labelled ``best`` while a 17.2 dB case was labelled ``worst``.
    """
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
    )
    for i in range(10):
        rec.observe(
            case_id=f"s{i}",
            arrays={"prediction": np.full((4, 4), float(i), dtype=np.float32)},
            metrics={"val_psnr": float(i), "val_robust_mri_psnr": float(i) - 3.0},
            domain={},
        )
    out = rec.write(tmp_path)
    index = json.loads((out / "cases_index.json").read_text())
    by_rank = {row["rank"]: row for row in index}
    assert by_rank["best"]["metrics"]["val_psnr"] == 9.0
    assert by_rank["worst"]["metrics"]["val_psnr"] == 0.0


def test_recorder_disabled_when_zero_cases(tmp_path):
    rec = ReportCaseRecorder(
        n_cases=0, selection="first", primary_metric="psnr", higher_is_better=True
    )
    rec.observe(
        case_id="s0",
        arrays={"prediction": np.zeros((2, 2), dtype=np.float32)},
        metrics={"psnr": 1.0},
        domain={},
    )
    out = rec.write(tmp_path)
    assert not (out / "cases_index.json").exists()


def test_recorder_persists_volume_arrays(tmp_path):
    """A recorder handed a 3-D ``*_volume`` array writes it into the npz."""
    rec = ReportCaseRecorder(
        n_cases=1,
        selection="first",
        primary_metric="psnr",
        higher_is_better=True,
        record_volumes=True,
    )
    assert rec.record_volumes is True
    rec.observe(
        case_id="s0",
        arrays={
            "prediction": np.zeros((4, 4), np.float32),
            "prediction_volume": np.ones((5, 4, 4), np.float32),
        },
        metrics={"psnr": 1.0},
        domain={},
    )
    out = rec.write(tmp_path)
    index = json.loads((out / "cases_index.json").read_text())
    with np.load(out / index[0]["npz"]) as data:
        assert "prediction_volume" in data.files
        assert data["prediction_volume"].shape == (5, 4, 4)


def test_recorder_pool_is_bounded_and_keeps_extremes(tmp_path):
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
        max_pool=10,
    )
    # feed 200 cases with a wide metric spread; pool must stay bounded
    for i in range(200):
        rec.observe(
            case_id=f"s{i}",
            arrays={"prediction": np.full((2, 2), float(i), np.float32)},
            metrics={"psnr": float(i)},
            domain={},
        )
    assert len(rec._cases) <= 10
    # the global best (199) and worst (0) must survive eviction
    psnrs = [c["metrics"]["psnr"] for c in rec._cases]
    assert max(psnrs) == 199.0
    assert min(psnrs) == 0.0


def test_eviction_survives_colliding_case_ids():
    """Regression for #617: eviction crashed when two cases shared a ``case_id``.

    ``_evict_median`` used ``list.remove``, which short-circuits on identity but
    otherwise falls back to ``==``. A case dict holds ``{str: np.ndarray}``, so
    dict equality compares ``case_id`` first and normally short-circuits to
    False -- but cascading validation feeds this recorder once per acceleration
    rung at the SAME training iteration, and the feed seam labelled them all
    ``val_step<N>``. On that tie the comparison reached the arrays and
    ``bool(ndarray == ndarray)`` raised "The truth value of an array with more
    than one element is ambiguous", killing the caller's entire validation
    image-logging block for the rest of the run.

    ``test_recorder_pool_is_bounded_and_keeps_extremes`` above misses it because
    it feeds unique ids; the collision is the whole trigger.
    """
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
        max_pool=4,
    )
    rng = np.random.default_rng(0)
    for i in range(12):
        rec.observe(
            case_id="val_step9000",  # every cascade rung of one validation event
            arrays={"prediction": rng.random((8, 8), dtype=np.float32)},
            metrics={"psnr": float(i)},
            domain={},
        )
    assert len(rec._cases) <= 4
    # extremes still survive: eviction dropped the median, not the newest
    psnrs = [c["metrics"]["psnr"] for c in rec._cases]
    assert max(psnrs) == 11.0
    assert min(psnrs) == 0.0


def test_eviction_removes_exactly_one_case_by_identity():
    """Equal-valued duplicates must not both be dropped by a single eviction."""
    rec = ReportCaseRecorder(
        n_cases=2,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
        max_pool=3,
    )
    shared = np.ones((3, 3), dtype=np.float32)
    for psnr in (10.0, 20.0, 20.0, 30.0):
        rec.observe(
            case_id="dup",
            arrays={"prediction": shared.copy()},
            metrics={"psnr": psnr},
            domain={},
        )
    assert len(rec._cases) == 3, "eviction removed more than the single victim"
