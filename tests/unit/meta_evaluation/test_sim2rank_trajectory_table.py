"""Differential tests for the batched trajectory-table helpers (#1518).

``_build_trajectory_table`` / ``_per_family_average`` are shared by seven rankers,
so the batched rewrite has to be *indistinguishable* from the per-family loop it
replaced, not merely close. The pre-#1518 implementation is kept verbatim below as
the oracle and every case is compared at the bit level — ``torch.equal`` would
accept ``-0.0`` for ``0.0``, and would reject two NaNs that the old code also
produced, so neither direction of that is good enough.
"""

from __future__ import annotations

import random
import struct

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.rankers.sim2rank import (
    _build_trajectory_table,
    _dataset_axes,
    _per_family_average,
)
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
)

NAN = float("nan")
TINY = torch.zeros((1, 2, 2))


# --------------------------------------------------------------------------- #
# Oracle: the implementation as it stood before #1518, copied verbatim.
# --------------------------------------------------------------------------- #
def _build_trajectory_table_ref(dataset, metric_name):
    severities = sorted({float(s.severity.get("theta", 0.0)) for s in dataset.samples})
    contents = sorted({s.content_id for s in dataset.samples})
    families = sorted({s.degradation_family for s in dataset.samples})
    sev_index = {v: i for i, v in enumerate(severities)}

    traj = {(c, f): [NAN] * len(severities) for c in contents for f in families}
    for s, v in zip(dataset.samples, dataset.metric_values[metric_name], strict=True):
        key = (s.content_id, s.degradation_family)
        traj[key][sev_index[float(s.severity.get("theta", 0.0))]] = float(v)
    return traj, severities, contents, families


def _per_family_average_ref(traj, families, contents):
    out = {}
    for fam in families:
        rows = [traj[(c, fam)] for c in contents if traj.get((c, fam)) is not None]
        if not rows:
            continue
        t = torch.tensor(rows, dtype=torch.float64)
        finite = torch.isfinite(t).float()
        denom = finite.sum(dim=0).clamp(min=1.0)
        avg = torch.where(torch.isfinite(t), t, torch.zeros_like(t)).sum(dim=0) / denom
        out[fam] = avg
    return out


def _bits(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().numpy().tobytes()


def _make(
    rng: random.Random,
    n_c: int,
    n_f: int,
    n_t: int,
    *,
    drop_prob: float = 0.0,
    nan_prob: float = 0.0,
    dead_family: bool = False,
    dup_theta: bool = False,
) -> MetricEvaluationDataset | None:
    families = [f"f{i}" for i in range(n_f)]
    samples = []
    for ci in range(n_c):
        for f in families:
            if dead_family and f == families[-1]:
                continue
            for t in range(n_t):
                if rng.random() < drop_prob:
                    continue
                samples.append(
                    DegradationSample(
                        clean=TINY,
                        degraded=TINY,
                        degradation_family=f,
                        severity={"theta": 0.0 if dup_theta else float(t)},
                        content_id=f"c{ci}",
                        seed=0,
                    )
                )
    if not samples:
        return None
    vals = [NAN if rng.random() < nan_prob else rng.uniform(-1e6, 1e6) for _ in samples]
    return MetricEvaluationDataset(samples=samples, metric_values={"m": vals})


def _assert_matches_oracle(ds: MetricEvaluationDataset) -> None:
    traj_r, sev_r, con_r, fam_r = _build_trajectory_table_ref(ds, "m")
    traj_n, sev_n, con_n, fam_n = _build_trajectory_table(ds, "m")
    assert (sev_r, con_r, fam_r) == (sev_n, con_n, fam_n)
    assert traj_r.keys() == traj_n.keys()
    for key, ref_row in traj_r.items():
        new_row = traj_n[key]
        assert len(ref_row) == len(new_row)
        assert all(
            struct.pack("<d", a) == struct.pack("<d", b)
            for a, b in zip(ref_row, new_row, strict=True)
        ), f"trajectory {key} differs"

    avg_r = _per_family_average_ref(traj_r, fam_r, con_r)
    avg_n = _per_family_average(traj_n, fam_n, con_n)
    assert avg_r.keys() == avg_n.keys()
    for fam, ref_t in avg_r.items():
        assert ref_t.shape == avg_n[fam].shape
        assert _bits(ref_t) == _bits(avg_n[fam]), f"family {fam} differs bitwise"


# --------------------------------------------------------------------------- #
# Differential coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("plain", {"n_c": 4, "n_f": 5, "n_t": 6}),
        ("nan_holes", {"n_c": 5, "n_f": 4, "n_t": 7, "nan_prob": 0.35}),
        ("missing_rows", {"n_c": 6, "n_f": 5, "n_t": 4, "drop_prob": 0.4}),
        ("missing_and_nan", {"n_c": 6, "n_f": 5, "n_t": 4, "drop_prob": 0.3, "nan_prob": 0.3}),
        ("dead_family", {"n_c": 3, "n_f": 4, "n_t": 5, "dead_family": True}),
        ("duplicate_theta", {"n_c": 3, "n_f": 3, "n_t": 4, "dup_theta": True}),
        ("single_content", {"n_c": 1, "n_f": 4, "n_t": 6}),
        ("single_severity", {"n_c": 4, "n_f": 4, "n_t": 1}),
        ("single_family", {"n_c": 4, "n_f": 1, "n_t": 5}),
        ("all_nan", {"n_c": 4, "n_f": 3, "n_t": 5, "nan_prob": 1.0}),
    ],
)
def test_matches_pre_1518_oracle(label: str, kwargs: dict) -> None:
    ds = _make(random.Random(hash(label) & 0xFFFF), **kwargs)
    assert ds is not None
    _assert_matches_oracle(ds)


def test_matches_oracle_over_randomized_shapes() -> None:
    rng = random.Random(20260826)
    checked = 0
    for _ in range(40):
        ds = _make(
            rng,
            n_c=rng.randint(1, 7),
            n_f=rng.randint(1, 8),
            n_t=rng.randint(1, 7),
            drop_prob=rng.choice([0.0, 0.2, 0.5]),
            nan_prob=rng.choice([0.0, 0.3, 0.9]),
            dead_family=rng.random() < 0.3,
        )
        if ds is None:
            continue
        _assert_matches_oracle(ds)
        checked += 1
    assert checked >= 30, "randomized sweep degenerated to too few shapes"


# --------------------------------------------------------------------------- #
# The two behaviours a naive batch would silently break
# --------------------------------------------------------------------------- #
def test_family_with_no_rows_is_absent_not_zeros() -> None:
    """NaN-filling a wholly-absent family would emit zeros; it must be dropped."""
    traj = {("c0", "present"): [1.0, 2.0]}
    out = _per_family_average(traj, ["present", "ghost"], ["c0"])
    assert set(out) == {"present"}
    assert "ghost" not in out


def test_all_nan_column_averages_to_zero_not_nan() -> None:
    """The finite count is clamped to 1, so an empty column yields 0.0."""
    traj = {("c0", "f"): [NAN, 3.0], ("c1", "f"): [NAN, 5.0]}
    out = _per_family_average(traj, ["f"], ["c0", "c1"])
    assert out["f"].tolist() == [0.0, 4.0]
    assert not torch.isnan(out["f"]).any()


def test_missing_content_row_equals_nan_row() -> None:
    """A padded NaN row must contribute nothing — numerator and count alike."""
    present = {("c0", "f"): [2.0, 4.0], ("c1", "f"): [6.0, 8.0]}
    with_gap = dict(present)
    with_nan = {**present, ("c2", "f"): [NAN, NAN]}
    contents = ["c0", "c1", "c2"]
    gap = _per_family_average(with_gap, ["f"], contents)["f"]
    nan_row = _per_family_average(with_nan, ["f"], contents)["f"]
    assert _bits(gap) == _bits(nan_row)
    assert gap.tolist() == [4.0, 6.0]


def test_ragged_widths_across_families_still_work() -> None:
    """Each family used to get its own tensor, so widths could differ per family.

    The batch groups by width to keep that legal — a single ``(F, C, T)`` tensor
    would raise on this input.
    """
    traj = {("c0", "short"): [1.0, 3.0], ("c0", "long"): [2.0, 4.0, 6.0]}
    out = _per_family_average(traj, ["short", "long"], ["c0"])
    assert out["short"].tolist() == [1.0, 3.0]
    assert out["long"].tolist() == [2.0, 4.0, 6.0]


@pytest.mark.parametrize(("families", "contents"), [([], ["c0"]), (["f"], []), ([], [])])
def test_empty_axes_return_empty_mapping(families: list, contents: list) -> None:
    assert _per_family_average({("c0", "f"): [1.0]}, families, contents) == {}


# --------------------------------------------------------------------------- #
# The memoised axes
# --------------------------------------------------------------------------- #
def test_axes_cache_is_reused_and_metric_independent() -> None:
    ds = _make(random.Random(7), n_c=3, n_f=3, n_t=4)
    assert ds is not None
    ds.metric_values["m2"] = [v * 2.0 for v in ds.metric_values["m"]]

    first = _dataset_axes(ds)
    assert _dataset_axes(ds) is first, "axes recomputed for the same dataset"

    t1, *_ = _build_trajectory_table(ds, "m")
    t2, *_ = _build_trajectory_table(ds, "m2")
    key = next(iter(t1))
    assert t1[key] != t2[key], "cache leaked one metric's values into another"


def test_axes_cache_invalidates_on_resized_samples() -> None:
    ds = _make(random.Random(8), n_c=2, n_f=2, n_t=3)
    assert ds is not None
    _, _, contents_before, _ = _build_trajectory_table(ds, "m")

    ds.samples.append(
        DegradationSample(
            clean=TINY,
            degraded=TINY,
            degradation_family="f0",
            severity={"theta": 0.0},
            content_id="cNEW",
            seed=0,
        )
    )
    ds.metric_values["m"].append(1.0)
    _, _, contents_after, _ = _build_trajectory_table(ds, "m")
    assert "cNEW" not in contents_before
    assert "cNEW" in contents_after


def test_axes_cache_invalidates_on_reassigned_samples() -> None:
    ds = _make(random.Random(9), n_c=2, n_f=2, n_t=3)
    assert ds is not None
    _build_trajectory_table(ds, "m")

    replacement = [
        DegradationSample(
            clean=TINY,
            degraded=TINY,
            degradation_family="only",
            severity={"theta": 0.0},
            content_id="solo",
            seed=0,
        )
    ]
    ds.samples = replacement
    ds.metric_values["m"] = [1.0]
    _, _, contents, families = _build_trajectory_table(ds, "m")
    assert contents == ["solo"]
    assert families == ["only"]


def test_duplicate_severity_slot_resolves_last_wins() -> None:
    """Two samples on one (content, family, theta) slot: the later value stands."""
    samples = [
        DegradationSample(
            clean=TINY,
            degraded=TINY,
            degradation_family="f",
            severity={"theta": 0.0},
            content_id="c",
            seed=0,
        )
        for _ in range(2)
    ]
    ds = MetricEvaluationDataset(samples=samples, metric_values={"m": [11.0, 22.0]})
    traj, _, _, _ = _build_trajectory_table(ds, "m")
    assert traj[("c", "f")] == [22.0]
