import numpy as np

from mriforge.infrastructure.reporting.cases.metric_sink import PerCallMetricSink
from mriforge.infrastructure.reporting.cases.recorder import ReportCaseRecorder
from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
    feed_report_case_recorder,
)


def test_feed_report_case_recorder_extracts_first_in_batch():
    rec = ReportCaseRecorder(n_cases=2, selection="first",
                             primary_metric="psnr", higher_is_better=True)
    pred = np.random.rand(2, 1, 8, 8).astype(np.float32)
    tgt = np.ones((2, 1, 8, 8), np.float32)
    inp = np.zeros((2, 1, 8, 8), np.float32)
    feed_report_case_recorder(rec, predictions=pred, targets=tgt, inputs=inp,
                              metrics={"psnr": 25.0}, step=3)
    assert len(rec._cases) >= 1
    assert "prediction" in rec._cases[0]["arrays"]
    assert rec._cases[0]["arrays"]["prediction"].ndim == 2


def test_feed_noop_when_recorder_disabled():
    rec = ReportCaseRecorder(n_cases=0, selection="first",
                             primary_metric="psnr", higher_is_better=True)
    feed_report_case_recorder(rec, predictions=np.zeros((1, 1, 2, 2), np.float32),
                              targets=None, inputs=None, metrics={}, step=0)
    assert rec._cases == []


def test_feed_also_populates_metric_sink():
    """The same seam feeds the per-case sink from the batch metrics dict."""
    rec = ReportCaseRecorder(n_cases=2, selection="first",
                             primary_metric="psnr", higher_is_better=True)
    sink = PerCallMetricSink(enabled=True)
    pred = np.random.rand(2, 1, 4, 4).astype(np.float32)
    feed_report_case_recorder(rec, predictions=pred, targets=pred, inputs=pred,
                              metrics={"psnr": 25.0, "ssim": 0.8}, step=7, sink=sink)
    assert sink.n_rows == 1
    assert sink._rows[0]["case_id"] == "val_step7"
    assert sink._rows[0]["psnr"] == 25.0
    assert sink._rows[0]["split"] == "val"


def test_feed_populates_sink_even_when_recorder_disabled():
    """Sink fires independently of the (bounded) image recorder."""
    rec = ReportCaseRecorder(n_cases=0, selection="first",
                             primary_metric="psnr", higher_is_better=True)
    sink = PerCallMetricSink(enabled=True)
    feed_report_case_recorder(rec, predictions=np.zeros((1, 1, 2, 2), np.float32),
                              targets=None, inputs=None, metrics={"psnr": 30.0},
                              step=1, sink=sink)
    assert rec._cases == []
    assert sink.n_rows == 1


def test_feed_noop_when_both_disabled():
    sink = PerCallMetricSink(enabled=False)
    feed_report_case_recorder(None, predictions=np.zeros((1, 1, 2, 2), np.float32),
                              targets=None, inputs=None, metrics={"psnr": 1.0},
                              step=0, sink=sink)
    assert sink.n_rows == 0


def test_record_volumes_preserves_3d_stack_from_5d_input():
    """record_volumes=True keeps a [Z,H,W] volume from a 5-D [B,C,Z,H,W] tensor."""
    rec = ReportCaseRecorder(n_cases=2, selection="first", primary_metric="psnr",
                             higher_is_better=True, record_volumes=True)
    pred = np.random.rand(2, 1, 6, 8, 8).astype(np.float32)  # B,C,Z,H,W
    feed_report_case_recorder(rec, predictions=pred, targets=pred, inputs=pred,
                              metrics={"psnr": 25.0}, step=1)
    arrays = rec._cases[0]["arrays"]
    # 2-D representative kept AND the volume preserved (coils RSS'd, Z kept).
    assert arrays["prediction"].ndim == 2
    assert arrays["prediction_volume"].shape == (6, 8, 8)
    assert arrays["target_volume"].shape == (6, 8, 8)


def test_record_volumes_off_by_default_stores_no_volume():
    rec = ReportCaseRecorder(n_cases=2, selection="first", primary_metric="psnr",
                             higher_is_better=True)
    pred = np.random.rand(2, 1, 6, 8, 8).astype(np.float32)
    feed_report_case_recorder(rec, predictions=pred, targets=pred, inputs=pred,
                              metrics={"psnr": 25.0}, step=1)
    assert not any(k.endswith("_volume") for k in rec._cases[0]["arrays"])


def test_record_volumes_skips_4d_slice_with_coils():
    """A 4-D [B,C,H,W] is a 2-D slice + coils, NOT a volume: no *_volume stored."""
    rec = ReportCaseRecorder(n_cases=2, selection="first", primary_metric="psnr",
                             higher_is_better=True, record_volumes=True)
    pred = np.random.rand(2, 4, 8, 8).astype(np.float32)  # B,C,H,W (coils)
    feed_report_case_recorder(rec, predictions=pred, targets=pred, inputs=pred,
                              metrics={"psnr": 25.0}, step=1)
    assert not any(k.endswith("_volume") for k in rec._cases[0]["arrays"])


# ── batch identity: what a per-case row actually averaged over ───────────────


def test_summarize_batch_identity_reads_the_collated_per_sample_lists():
    """`ImageCollateStrategy` leaves non-tensor values as a per-sample list."""
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        summarize_batch_identity,
    )

    out = summarize_batch_identity(
        {"file_id": ["a_T101", "b_T102"], "contrast": ["T1", "T1"]}
    )
    assert out == {"file_id": "a_T101|b_T102", "batch_size": 2, "contrast": "T1"}


def test_a_mixed_contrast_batch_names_every_contrast_present():
    """The validation loader shuffles, so one batch can hold T1 AND FLAIR.

    The metrics are a MEAN over the batch. Naming one contrast would attribute
    a number to a contrast that produced only half of it — a wrong label reads
    as a measurement, where a joined one reads as the mixture it is.
    """
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        summarize_batch_identity,
    )

    out = summarize_batch_identity(
        {"file_id": ["a_T101", "b_FLAIR01"], "contrast": ["T1", "FLAIR"]}
    )
    assert out["contrast"] == "FLAIR|T1"
    assert out["batch_size"] == 2


def test_batch_size_one_yields_a_single_volume_row():
    """`validation.loader.batch_size: 1` makes each row exactly one volume."""
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        summarize_batch_identity,
    )

    out = summarize_batch_identity({"file_id": ["a_T101"], "contrast": ["T1"]})
    assert out == {"file_id": "a_T101", "batch_size": 1, "contrast": "T1"}


def test_a_dataset_publishing_no_identity_yields_no_columns():
    """Absent, not placeholder — an "unknown" string would read as data."""
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        summarize_batch_identity,
    )

    assert summarize_batch_identity({"input": 1}) == {}
    assert summarize_batch_identity(None) == {}


def test_identity_is_read_off_an_object_batch_too():
    from types import SimpleNamespace

    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        summarize_batch_identity,
    )

    batch = SimpleNamespace(file_id=["a_T101"], contrast=["T1"])
    assert summarize_batch_identity(batch)["contrast"] == "T1"


def test_context_reaches_the_sink_verbatim():
    """The feed helper forwards context to the sink and NOT into metrics.

    Merging it into `metrics` would send it through the float coercion, which
    is what drops `contrast` and turns `heldout` into `1.0`.
    """
    from mriforge.infrastructure.reporting.cases.metric_sink import PerCallMetricSink
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        feed_report_case_recorder,
    )

    sink = PerCallMetricSink(enabled=True)
    feed_report_case_recorder(
        None,
        predictions=np.zeros((1, 1, 2, 2), np.float32),
        targets=np.zeros((1, 1, 2, 2), np.float32),
        inputs=np.zeros((1, 1, 2, 2), np.float32),
        metrics={"val_psnr": 18.9},
        step=1500,
        sink=sink,
        cascade_level=8.0,
        context={"acceleration_level": 8.0, "timestep": 27, "contrast": "FLAIR"},
    )
    row = sink._rows[0]
    assert row["case_id"] == "val_step1500_R8x"
    assert row["contrast"] == "FLAIR"
    assert row["timestep"] == 27
    assert row["val_psnr"] == 18.9


def test_context_is_optional_for_strategies_that_do_not_supply_it():
    """The non-diffusion feed site passes no context and must be unaffected."""
    from mriforge.infrastructure.reporting.cases.metric_sink import PerCallMetricSink
    from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import (
        feed_report_case_recorder,
    )

    sink = PerCallMetricSink(enabled=True)
    feed_report_case_recorder(
        None,
        predictions=np.zeros((1, 1, 2, 2), np.float32),
        targets=np.zeros((1, 1, 2, 2), np.float32),
        inputs=np.zeros((1, 1, 2, 2), np.float32),
        metrics={"val_psnr": 30.0},
        step=1,
        sink=sink,
    )
    assert sink._rows[0] == {
        "case_id": "val_step1",
        "split": "val",
        "step": 1,
        "val_psnr": 30.0,
    }
