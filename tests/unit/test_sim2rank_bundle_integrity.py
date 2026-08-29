"""Validate a REAL sim2rank output bundle: is every number in it a measured number?

Opt-in, and pointed at a run::

    SIM2RANK_BUNDLE=/path/to/results pytest tests/unit/test_sim2rank_bundle_integrity.py -v

Without ``SIM2RANK_BUNDLE`` every test skips, so ``make test`` stays green and this file
never grades a stale artefact that happens to be sitting in the tree.

**Why an artefact test and not only unit tests.** Everything else in ``tests/unit``
grades the code on synthetic inputs. That is necessary and it is not sufficient: the
2026-07-27 ``rician`` defect needed the *real* p99 (~1e-4 on fastMRI brain) to fire at
all, because the operator only misbehaved when the caller's units differed from the unit
scale a synthetic phantom happens to have. A unit test on a p99 ~ 1 phantom passed
throughout. These checks read what the run actually produced, so a mis-scaled axis, a
metric pinned on its clamp, a duplicated ranker or a fabricated zero cannot reach a
figure or a paper unchallenged.

Run this after every cluster sweep, before reading the leaderboard.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_ENV = "SIM2RANK_BUNDLE"


def _bundle() -> Path:
    root = os.environ.get(_ENV)
    if not root:
        pytest.skip(
            f"set {_ENV}=/path/to/sim2rank/results to validate a run "
            "(skipped by default so this never grades a stale artefact)"
        )
    path = Path(root).expanduser()
    if not path.is_dir():
        pytest.fail(f"{_ENV}={path} is not a directory")
    return path


def _load_json(name: str) -> dict[str, Any]:
    path = _bundle() / name
    if not path.is_file():
        pytest.skip(f"{name} absent from the bundle — that stage did not run")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def multigen() -> dict[str, Any]:
    return _load_json("run_all_generations_results.json")


@pytest.fixture(scope="module")
def sim() -> dict[str, Any]:
    return _load_json("sim2rank_results.json")


@pytest.fixture(scope="module")
def metrics_long() -> pd.DataFrame:
    path = _bundle() / "metrics_long.csv.gz"
    if not path.is_file():
        pytest.skip(
            "metrics_long.csv.gz absent — no per-severity trajectories to check"
        )
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def tables() -> Path:
    path = _bundle() / "tables"
    if not path.is_dir():
        pytest.skip("tables/ absent — export_tables did not run")
    return path


#: Columns of a wide per-axis table that are provenance, not axes.
_META_COLS = frozenset(
    {"registry_key", "category", "method", "method_generation", "status", "blocked_on"}
)


def _wide_per_axis_tables(tables: Path) -> list[Path]:
    """The one-method-per-file wide CSVs, excluding the long master.

    ``per_axis_all_methods_long.csv`` matches the same glob but is long-format; reading
    its ``method_generation`` column as an axis produces a nonsense comparison.
    """
    return [
        p
        for p in sorted(tables.glob("per_axis_*.csv"))
        if p.name != "per_axis_all_methods_long.csv"
    ]


def _axis_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


# ──────────────────────────────────────────────────────────────────────
# The degradation actually degraded, and by a comparable amount
# ──────────────────────────────────────────────────────────────────────


def test_no_axis_is_wildly_mis_scaled(metrics_long: pd.DataFrame) -> None:
    """NMSE at the MILDEST severity must be in family across axes.

    NMSE is scale-free, so its value at the gentlest step is directly comparable
    between axes: it says "how much of the reference energy did the mildest nudge
    destroy". A single axis orders of magnitude above the rest is not a harsh
    degradation, it is a unit error.

    This is the artefact-side detector for the 2026-07-27 ``rician`` defect, where
    the mildest step already realised **NMSE = 4.5e5** against ~0.88 on all 29 other
    axes — 5e5x the cross-axis median. The tolerance is deliberately enormous (100x)
    so a genuinely aggressive axis passes and only a scaling error fails.
    """
    nmse = metrics_long[metrics_long["metric"] == "nmse"]
    if nmse.empty:
        pytest.skip("nmse not in the sweep")
    mildest = nmse.sort_values("corruption_factor").groupby("axis").first()["value"]
    median = float(np.median(mildest.to_numpy()))
    assert median > 0, "every axis reports NMSE 0 at the mildest step"

    offenders = {a: float(v) for a, v in mildest.items() if v > 100.0 * median}
    assert not offenders, (
        f"axes mis-scaled at the mildest severity (cross-axis median NMSE = "
        f"{median:.4g}): {offenders}. An NMSE far above 1 at the gentlest step means "
        "the perturbation carries more energy than the reference — the operator is "
        "reading an absolute intensity instead of a signal-relative one."
    )


def test_every_axis_moves_its_metrics(metrics_long: pd.DataFrame) -> None:
    """An axis on which no live metric varies did not fire (#223)."""
    live = metrics_long[metrics_long["health"] == "live"]
    if live.empty:
        pytest.skip("no live metrics recorded")
    varied = live.groupby("axis")["value"].nunique()
    dead = sorted(varied[varied <= 1].index)
    assert not dead, (
        f"axes on which not one live metric changed across the whole severity sweep: "
        f"{dead} — the degradation is a no-op there"
    )


def test_no_axis_saturates_an_outlying_number_of_metrics(
    metrics_long: pd.DataFrame,
) -> None:
    """A metric pinned at one value across a whole sweep is on a clamp, not measuring.

    Some pinning is normal (a metric genuinely blind to one artefact), so this
    compares each axis against the cross-axis median rather than to zero. On the
    2026-07-27 brain run ``rician`` pinned 27 metrics against a median of 6.
    """
    live = metrics_long[metrics_long["health"] == "live"]
    if live.empty:
        pytest.skip("no live metrics recorded")
    nuniq = live.groupby(["axis", "metric"])["value"].nunique().reset_index()
    pinned = nuniq[nuniq["value"] <= 1].groupby("axis").size()
    if pinned.empty:
        return
    pinned = pinned.reindex(nuniq["axis"].unique(), fill_value=0)
    median = float(np.median(pinned.to_numpy()))
    limit = max(3.0 * median, median + 10.0)
    offenders = {a: int(n) for a, n in pinned.items() if n > limit}
    assert not offenders, (
        f"axes pinning far more metrics than the median ({median:.0f}): {offenders}. "
        "Metrics stuck on a clamp bound at every severity level are reporting their "
        "floor, not a measurement."
    )


# ──────────────────────────────────────────────────────────────────────
# The rankers produced rankings
# ──────────────────────────────────────────────────────────────────────


def test_every_declared_method_has_scores(multigen: dict[str, Any]) -> None:
    methods = multigen.get("method_names") or []
    subs = multigen.get("sub_scores") or {}
    assert methods, "the run declared no methods"
    missing = sorted(set(methods) - set(subs))
    assert not missing, f"methods declared but never scored: {missing}"

    widths = {m: len(subs[m]) for m in methods}
    assert len(set(widths.values())) == 1, (
        f"rankers scored different numbers of metrics: {widths} — the positional zip "
        "against the registry keys relabels every score in the short ones"
    )


def test_no_ranker_score_is_nan(multigen: dict[str, Any]) -> None:
    """NaN is never a documented sentinel (``+inf`` and ``-inf`` are)."""
    bad: dict[str, list[str]] = {}
    for method, scores in (multigen.get("sub_scores") or {}).items():
        nan_keys = [k for k, v in scores.items() if v is not None and np.isnan(v)]
        if nan_keys:
            bad[method] = nan_keys[:10]
    assert not bad, f"NaN scores (a crashed metric reported as a result): {bad}"


def test_non_finite_scores_are_confined_to_the_rankers_that_define_them(
    multigen: dict[str, Any],
) -> None:
    """``+inf``/``-inf`` are legitimate for SCVR and the anchor, suspicious elsewhere.

    ``ranks_from_scores`` ranks ``+inf`` best (SCVR: zero estimable content nuisance)
    and ``-inf`` worst (anchor: never calibrated). A ranker with no such sentinel in
    its definition emitting one has divided by zero.
    """
    allowed = ("SCVR", "Anchor")
    offenders: dict[str, list[str]] = {}
    for method, scores in (multigen.get("sub_scores") or {}).items():
        if any(tag in method for tag in allowed):
            continue
        hits = [
            k for k, v in scores.items() if v is not None and np.isinf(np.float64(v))
        ]
        if hits:
            offenders[method] = hits[:10]
    assert (
        not offenders
    ), f"non-finite scores from rankers with no sentinel semantics: {offenders}"


def test_no_two_rankers_are_identical(multigen: dict[str, Any]) -> None:
    """Two names for one statistic inflate the apparent agreement between methods."""
    subs = multigen.get("sub_scores") or {}
    names = sorted(subs)
    dupes = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if subs[a] == subs[b]:
                dupes.append((a, b))
    assert not dupes, f"rankers with byte-identical scores: {dupes}"


def test_no_two_per_axis_surfaces_are_identical(multigen: dict[str, Any]) -> None:
    """The Gen1_MinimaxBorda-was-a-copy-of-Gen1_ADR_mean regression, on real output."""
    per_axis = multigen.get("per_axis_sub_scores") or {}
    names = sorted(per_axis)
    dupes = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if per_axis[a] == per_axis[b]:
                dupes.append((a, b))
    assert not dupes, (
        f"per-axis surfaces that are byte-identical: {dupes} — the exported "
        "per_axis_*.csv files for these methods are duplicates of each other"
    )


def test_per_axis_methods_cover_every_swept_axis(multigen: dict[str, Any]) -> None:
    declared = set((multigen.get("config") or {}).get("axes") or [])
    if not declared:
        pytest.skip("config.axes not recorded")
    for method, axis_map in (multigen.get("per_axis_sub_scores") or {}).items():
        missing = sorted(declared - set(axis_map))
        assert not missing, f"{method} has no per-axis scores for: {missing}"


def test_bootstrap_stability_is_absent_for_never_resampled_methods(
    multigen: dict[str, Any],
) -> None:
    """A nominal-only method has no stability to report; 1.0 would read as perfect."""
    nominal = set(multigen.get("nominal_only_methods") or [])
    if not nominal:
        pytest.skip("every method was bootstrapped")
    stability = multigen.get("bootstrap_stability") or {}
    tested = {m: stability[m] for m in stability if m not in nominal}
    assert tested, "no bootstrapped method to compare against"
    # The JSON legitimately stores 1.0 for them; what must not happen is a
    # never-resampled method outscoring every resampled one in the exported table.
    # That assertion lives in the tables section below.
    assert set(nominal) <= set(stability), (
        f"nominal-only methods missing from bootstrap_stability: "
        f"{sorted(set(nominal) - set(stability))}"
    )


# ──────────────────────────────────────────────────────────────────────
# Metric health is recorded, and the tables honour it
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def unmeasured_keys(sim: dict[str, Any]) -> set[str]:
    health = sim.get("metric_health") or {}
    if not health:
        pytest.skip("no metric_health block — 0.0 cannot be told from 'never ran'")
    summary = health.get("summary") or {}
    return {
        k
        for state in ("never_computed", "not_applicable")
        for k in (summary.get(state) or [])
    }


def test_never_run_metrics_are_nan_in_every_table(
    tables: Path, unmeasured_keys: set[str]
) -> None:
    """A metric that never ran must read NaN, never 0.0, on EVERY surface.

    On an ADR scale 0.0 is a legal floor, so a placeholder zero is indistinguishable
    from a measured "no discriminative power". Until 2026-07-27 five of the six
    per-axis surfaces shipped the zeros while the sixth masked them.
    """
    if not unmeasured_keys:
        pytest.skip("this run computed every metric")
    offenders: dict[str, list[str]] = {}
    for csv in _wide_per_axis_tables(tables):
        df = pd.read_csv(csv)
        if "registry_key" not in df.columns:
            continue
        axis_cols = _axis_columns(df)
        rows = df[df["registry_key"].isin(sorted(unmeasured_keys))]
        if rows.empty:
            continue
        block = rows[axis_cols].apply(pd.to_numeric, errors="coerce")
        zeros = int((block == 0).sum().sum())
        if zeros:
            offenders[csv.name] = [f"{zeros} placeholder zeros"]
    assert not offenders, (
        f"never-run metrics written as 0.0 rather than NaN: {offenders}. Those cells "
        "compete in every mean as a measured worst-case."
    )


def test_the_long_table_and_the_wide_tables_agree(tables: Path) -> None:
    """Every wide cell must appear in the long master with the same value.

    They are separate builders over the same payload; a divergence means one of them
    picked up a transform the other did not.
    """
    long_path = tables / "per_axis_all_methods_long.csv"
    if not long_path.is_file():
        pytest.skip("per_axis_all_methods_long.csv absent")
    long_df = pd.read_csv(long_path)
    lookup = long_df.set_index(["method", "axis", "registry_key"])["score"]

    checked = 0
    for csv in _wide_per_axis_tables(tables):
        df = pd.read_csv(csv)
        if "method" not in df.columns:
            continue
        method = str(df["method"].iloc[0])
        if method not in set(long_df["method"]):
            continue
        for axis in _axis_columns(df):
            for key, wide_val in zip(df["registry_key"], df[axis], strict=True):
                try:
                    long_val = lookup.loc[(method, axis, key)]
                except KeyError:
                    pytest.fail(f"{method}/{axis}/{key} is in {csv.name}, not in long")
                both_nan = pd.isna(wide_val) and pd.isna(long_val)
                assert both_nan or float(wide_val) == pytest.approx(
                    float(long_val), rel=1e-9
                ), f"{csv.name} and the long table disagree on {method}/{axis}/{key}"
                checked += 1
    assert checked > 0, "no wide table could be cross-checked against the long one"


def test_anchor_verdicts_partition_cleanly(tables: Path) -> None:
    """Every metric gets exactly one anchor verdict, and 'not calibrated' is its own.

    Reporting an un-calibrated metric as ``passes_anchor=False`` turns a missing
    measurement into a measured failure.
    """
    path = tables / "anchor_calibration_per_metric.csv"
    if not path.is_file():
        pytest.skip("anchor_calibration_per_metric.csv absent")
    df = pd.read_csv(path)
    assert "anchor_status" in df.columns, (
        "anchor table has no anchor_status column — a bare passes_anchor boolean "
        "cannot distinguish 'failed' from 'never calibrated'"
    )
    allowed = {"pass", "fail", "uninformative", "not_calibrated"}
    unknown = sorted(set(df["anchor_status"]) - allowed)
    assert not unknown, f"unknown anchor_status values: {unknown}"

    not_cal = df[df["anchor_status"] == "not_calibrated"]
    assert (
        not_cal["anchor_mae"].isna().all()
    ), "a metric marked not_calibrated carries an MAE"
    assert (
        not_cal["passes_anchor"].isna().all()
    ), "an un-calibrated metric is reported as a measured pass/fail"


def test_nominal_only_methods_report_no_stability_in_the_table(tables: Path) -> None:
    """The exported column must not print 1.0 for a method that was never resampled."""
    path = tables / "algorithm_performance_per_axis.csv"
    if not path.is_file():
        pytest.skip("algorithm_performance_per_axis.csv absent")
    df = pd.read_csv(path)
    if "nominal_only" not in df.columns:
        pytest.skip("table predates the nominal_only column")
    nominal = df[df["nominal_only"].astype(str).str.lower() == "true"]
    if nominal.empty:
        pytest.skip("every method was bootstrapped")
    assert nominal["bootstrap_stability"].isna().all(), (
        "a never-resampled method reports a bootstrap stability; sorting that column "
        "puts every untested ranker above the ones that were actually stress-tested"
    )
