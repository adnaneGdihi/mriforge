"""Paired tests for ``infrastructure/reporting/inference_artifacts.py``.

The load-bearing case here is the *skip* path, not the compute path. Every arm in
the ``kspace_filling`` cohort declares only full-reference metrics, so on the
inference path as it stands today all of them skip -- and a report that silently
renders empty is indistinguishable from a report of a model that scored nothing.
These tests pin that the reason is recorded, that it is recorded somewhere the
aggregator can survive reading, and that the same declared set becomes fully
computable the moment a reference exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.reporting.inference_artifacts import (
    FINAL_EVAL_JSON,
    FINAL_EVAL_MANIFEST_JSON,
    METRICS_CSV,
    REASON_NEEDS_CONTEXT,
    REASON_NO_REFERENCE,
    REASON_UNREGISTERED,
    STATUS_COMPUTED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    InferenceEvaluator,
    classify_metric,
    write_inference_run_summary,
)

# The real declared set of every arm in experiments/inprogress/kspace_filling/.
# Measured across all 58 arms: `mae` and `robust_mri_psnr` on 58, the other three
# on 57. All five are full-reference, which is the whole point of this fixture --
# it is not a hand-picked worst case, it is the cohort.
COHORT_DECLARED = ["mae", "robust_mri_psnr", "hfen", "kspace_error", "phase_mse"]

# A registered metric that genuinely runs on predictions alone. Needed because
# the cohort set never exercises the compute branch, which would otherwise ship
# untested.
NO_REFERENCE_METRIC = "brisque"


@pytest.fixture
def pred() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 1, 32, 32)


class TestClassificationPartitionsTheDeclaredSet:
    def test_unregistered_name_is_named_as_such(self):
        assert classify_metric("not_a_metric", has_reference=True) == REASON_UNREGISTERED

    def test_unregistered_is_checked_before_anything_reads_its_flags(self):
        # Ordering guard: an unregistered name has no metadata, so a
        # requires_reference() lookup on it would answer from a default rather
        # than from the metric. The reason must be UNREGISTERED either way.
        assert classify_metric("not_a_metric", has_reference=False) == REASON_UNREGISTERED

    def test_full_reference_metric_without_a_target_is_skipped_with_the_reason(self):
        assert classify_metric("mae", has_reference=False) == REASON_NO_REFERENCE

    def test_reference_free_metric_is_computable(self):
        assert classify_metric(NO_REFERENCE_METRIC, has_reference=False) is None


class TestTheCohortIsTargetReadyNotUncomputable:
    """The finding this module exists to record, as an executable claim.

    Without a reference every declared metric skips; *with* one, every declared
    metric computes. So the gap is the missing reference on the inference data
    path -- not the evaluation machinery, and not the metrics.
    """

    def test_every_declared_metric_skips_without_a_reference(self):
        reasons = {m: classify_metric(m, has_reference=False) for m in COHORT_DECLARED}
        assert set(reasons.values()) == {REASON_NO_REFERENCE}, reasons

    def test_every_declared_metric_becomes_computable_with_a_reference(self):
        blocked = {
            m: r
            for m in COHORT_DECLARED
            if (r := classify_metric(m, has_reference=True)) is not None
        }
        assert not blocked, (
            "these declared metrics would still not compute even once the "
            f"inference path supplies a target: {blocked}"
        )

    def test_a_context_consuming_metric_is_distinguished_from_a_missing_target(self):
        # Two different causes must not collapse into one reason: supplying a
        # target fixes the first and does nothing for the second.
        from spectramr.core.metrics.registry import MetricsRegistry

        ctx = next(
            (
                n
                for n in MetricsRegistry.list_available()
                if MetricsRegistry.is_registered(n)
                and not MetricsRegistry.requires_reference(n)
                and (MetricsRegistry.needs_context(n) or MetricsRegistry.needs(n))
            ),
            None,
        )
        if ctx is None:
            pytest.skip("no context-consuming reference-free metric registered")
        assert classify_metric(ctx, has_reference=True) == REASON_NEEDS_CONTEXT


class TestArtifactsAreWrittenEvenWhenNothingIsComputable:
    def test_final_eval_json_is_written_empty_rather_than_omitted(self, tmp_path, pred):
        ev = InferenceEvaluator(COHORT_DECLARED)
        ev.observe(case_id="subj01", prediction=pred)  # no target
        ev.write(tmp_path)
        path = tmp_path / FINAL_EVAL_JSON
        assert path.exists(), (
            "an absent final_eval.json cannot be told apart from an inference run "
            "that never evaluated at all"
        )
        assert json.loads(path.read_text()) == {}

    def test_every_declared_metric_appears_in_the_manifest_with_a_reason(self, tmp_path, pred):
        ev = InferenceEvaluator(COHORT_DECLARED)
        ev.observe(case_id="subj01", prediction=pred)
        ev.write(tmp_path)
        payload = json.loads((tmp_path / FINAL_EVAL_MANIFEST_JSON).read_text())
        assert [m["name"] for m in payload["metrics"]] == COHORT_DECLARED
        assert all(m["status"] == STATUS_SKIPPED for m in payload["metrics"])
        assert all(m["reason"] == REASON_NO_REFERENCE for m in payload["metrics"])
        assert payload["reference_available"] is False
        assert payload["cases_observed"] == 1

    def test_skips_are_visible_as_csv_rows_not_absent_columns(self, tmp_path, pred):
        ev = InferenceEvaluator(COHORT_DECLARED)
        ev.observe(case_id="subj01", prediction=pred)
        ev.write(tmp_path)
        rows = list(csv.DictReader((tmp_path / METRICS_CSV).open()))
        assert {r["metric"] for r in rows} == set(COHORT_DECLARED)
        assert all(r["status"] == STATUS_SKIPPED for r in rows)


class TestComputeBranch:
    def test_a_reference_free_metric_is_computed_per_subject(self, tmp_path, pred):
        ev = InferenceEvaluator([NO_REFERENCE_METRIC])
        ev.observe(case_id="subj01", prediction=pred)
        ev.observe(case_id="subj02", prediction=pred * 0.5)
        ev.write(tmp_path)
        payload = json.loads((tmp_path / FINAL_EVAL_JSON).read_text())
        assert set(payload) == {NO_REFERENCE_METRIC}
        assert set(payload[NO_REFERENCE_METRIC]) == {"subj01", "subj02"}
        assert all(isinstance(v, float) for v in payload[NO_REFERENCE_METRIC].values())
        manifest = json.loads((tmp_path / FINAL_EVAL_MANIFEST_JSON).read_text())
        assert manifest["metrics"][0]["status"] == STATUS_COMPUTED
        assert manifest["metrics"][0]["reason"] is None

    def test_a_full_reference_metric_is_computed_when_a_target_is_supplied(self, tmp_path, pred):
        ev = InferenceEvaluator(["mae"])
        ev.observe(case_id="subj01", prediction=pred, target=torch.zeros_like(pred))
        ev.write(tmp_path)
        payload = json.loads((tmp_path / FINAL_EVAL_JSON).read_text())
        # mae against a zero target is the mean of a uniform [0,1) draw.
        assert payload["mae"]["subj01"] == pytest.approx(pred.mean().item(), abs=1e-5)

    def test_a_raising_metric_is_recorded_not_swallowed(self, tmp_path, pred, monkeypatch):
        from spectramr.core.metrics import registry as registry_mod

        def _boom(name, **kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(registry_mod.MetricsRegistry, "get", _boom)
        ev = InferenceEvaluator([NO_REFERENCE_METRIC])
        ev.observe(case_id="subj01", prediction=pred)
        ev.write(tmp_path)
        payload = json.loads((tmp_path / FINAL_EVAL_MANIFEST_JSON).read_text())
        entry = payload["metrics"][0]
        assert entry["status"] == STATUS_FAILED
        assert "kaboom" in entry["reason"]


class TestArtifactsRoundTripThroughTheAggregator:
    """The artifacts are only worth writing if ``report`` can actually read them."""

    def _aggregate(self, run_dir: Path):
        from spectramr.infrastructure.reporting.aggregator import aggregate

        return aggregate(run_dir)

    def test_an_all_skipped_run_yields_a_frame_the_aggregator_can_build(self, tmp_path, pred):
        ev = InferenceEvaluator(COHORT_DECLARED)
        ev.observe(case_id="subj01", prediction=pred)
        ev.write(tmp_path)
        df = self._aggregate(tmp_path)  # must not raise
        got = set(df["metric"]) if "metric" in df else set()
        assert not (got & set(COHORT_DECLARED)), (
            "a skipped metric must not reach the frame as a value row -- a row "
            f"here would be a number nothing computed: {got & set(COHORT_DECLARED)}"
        )

    def test_a_reason_in_final_eval_json_would_break_the_aggregator(self, tmp_path):
        """The negative control for where skip reasons are allowed to live.

        This is why the manifest exists as a separate file. ``_flatten_eval_json``
        coerces every value with ``float()``, so parking a human-readable reason
        in ``final_eval.json`` -- the obvious first design -- does not degrade the
        report, it destroys it: a metric that merely could not be computed takes
        the whole report down with it.
        """
        (tmp_path / FINAL_EVAL_JSON).write_text(json.dumps({"mae": REASON_NO_REFERENCE}))
        with pytest.raises(ValueError):
            self._aggregate(tmp_path)

    def test_a_computed_metric_reaches_the_tidy_frame_as_a_test_split_row(self, tmp_path, pred):
        ev = InferenceEvaluator([NO_REFERENCE_METRIC])
        ev.observe(case_id="subj01", prediction=pred)
        ev.write(tmp_path)
        df = self._aggregate(tmp_path)
        rows = df[df["metric"] == NO_REFERENCE_METRIC]
        assert len(rows) == 1, f"expected one row, frame was:\n{df}"
        assert rows.iloc[0]["split"] == "test"
        assert rows.iloc[0]["subject_id"] == "subj01"

    def test_run_summary_facts_reach_the_frame_as_run_split_rows(self, tmp_path, pred):
        model = torch.nn.Linear(4, 4)  # 20 params
        write_inference_run_summary(tmp_path, model=model, duration_sec=120.0, effective_batch=2)
        InferenceEvaluator([]).write(tmp_path)
        df = self._aggregate(tmp_path)
        run_rows = df[df["split"] == "run"].set_index("metric")["value"].to_dict()
        assert run_rows["params_m"] == pytest.approx(20e-6)
        assert run_rows["duration_min"] == pytest.approx(2.0)
        assert run_rows["effective_batch"] == pytest.approx(2.0)

    def test_iterations_per_sec_is_absent_rather_than_relabelled(self, tmp_path):
        # files/sec is not the training iteration rate the card labels it as.
        write_inference_run_summary(tmp_path, duration_sec=1.0)
        payload = json.loads((tmp_path / "run_summary.json").read_text())
        assert "iterations_per_sec" not in payload
