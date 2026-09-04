"""The single TensorBoard writer: directory resolution, gating, and cadence.

Driven against a recording double rather than a real ``SummaryWriter`` so the
feature surface is checked without an event file or a torch build carrying
tensorboard. The one thing a double cannot check -- that the constructor raises
when tensorboard is genuinely absent -- is exercised by patching the module
symbol to ``None``, which is exactly the state an ImportError leaves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.config.schemas.enums import TrackingService
from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.infrastructure.services import tensorboard_writer as tbw
from spectramr.infrastructure.services.tensorboard_writer import (
    TensorBoardWriter,
    resolve_event_dir,
)


class RecordingWriter:
    """Stands in for ``SummaryWriter``, remembering every call."""

    def __init__(self, log_dir: str, purge_step: int | None = None) -> None:
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.calls: list[tuple] = []

    def add_scalar(self, tag, value, global_step=None):
        self.calls.append(("scalar", tag, value, global_step))

    def add_scalars(self, tag, values, global_step=None):
        self.calls.append(("scalars", tag, dict(values), global_step))

    def add_images(self, tag, tensor, global_step=None):
        self.calls.append(("images", tag, global_step))

    def add_histogram(self, tag, values, step):
        self.calls.append(("histogram", tag, step))

    def add_text(self, tag, body, global_step=None):
        self.calls.append(("text", tag, body, global_step))

    def add_hparams(self, hparams, metrics):
        self.calls.append(("hparams", dict(hparams), dict(metrics)))

    def flush(self):
        self.calls.append(("flush",))

    def close(self):
        self.calls.append(("close",))

    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def recording(monkeypatch):
    made: list[RecordingWriter] = []

    def factory(log_dir, purge_step=None):
        made.append(RecordingWriter(log_dir, purge_step))
        return made[-1]

    monkeypatch.setattr(tbw, "SummaryWriter", factory)
    return made


def _config(**tracking) -> LoggingConfigSchema:
    base = {"enabled": True, "service": TrackingService.TENSORBOARD}
    base.update(tracking)
    return LoggingConfigSchema(tracking=base)


class TestResolveEventDir:
    """Every declaration in the corpus must land inside the run directory."""

    def test_unset_uses_the_run_relative_default(self) -> None:
        assert resolve_event_dir("/runs/a", None) == Path("/runs/a/tensorboard")

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [("./tensorboard", "/runs/a/tensorboard"), ("runs/", "/runs/a/runs")],
    )
    def test_the_corpus_declarations_stay_inside_the_run_dir(
        self, declared: str, expected: str
    ) -> None:
        """All 21 committed declarations are RELATIVE.

        Resolving them against the process CWD instead would put every run's
        events in one shared directory, where event files interleave. The 20
        arms declaring ``./tensorboard`` must be byte-identical to the previous
        hardcoded behaviour.
        """
        assert resolve_event_dir("/runs/a", declared) == Path(expected)

    def test_an_absolute_declaration_still_overrides(self) -> None:
        assert resolve_event_dir("/runs/a", "/tmp/tb") == Path("/tmp/tb")


class TestGating:
    """Disabled is a state, not an error -- but never a SILENT one."""

    @pytest.mark.parametrize(
        "tracking",
        [
            {"enabled": False},
            {"enable_tensorboard": False},
            {"service": TrackingService.NONE},
        ],
        ids=["tracking-off", "tensorboard-off", "service-none"],
    )
    def test_each_off_switch_disables_the_writer(self, tracking, recording) -> None:
        writer = TensorBoardWriter(Path("/runs/a"), _config(**tracking))
        assert not writer.enabled
        assert not recording, "a disabled writer must not construct a SummaryWriter"

    def test_a_non_zero_rank_gets_no_writer(self, recording) -> None:
        """DDP: one event dir per run, owned by rank 0, or the scalars interleave."""
        writer = TensorBoardWriter(Path("/runs/a"), _config(), is_rank_zero=False)
        assert not writer.enabled
        assert not recording

    def test_a_disabled_writer_is_falsy(self) -> None:
        """The call sites guard with `if tb_writer:`; that must keep meaning
        'this rank owns the writer', not 'the object exists'."""
        assert not TensorBoardWriter(Path("/runs/a"), _config(enabled=False))

    def test_an_enabled_writer_is_truthy(self, recording, tmp_path) -> None:
        assert TensorBoardWriter(tmp_path, _config())

    def test_every_method_is_a_no_op_when_disabled(self) -> None:
        """Call sites should not need to guard each call individually."""
        writer = TensorBoardWriter(Path("/runs/a"), _config(enabled=False))
        writer.scalars({"loss": 1.0}, 1, "train")
        writer.grouped_scalars("adv", {"g": 1.0}, 1)
        writer.images({"val": object()}, 1)
        writer.text("config", "body")
        writer.record_hparam_metric("psnr", 30.0)
        writer.hparams({"lr": 1e-4})
        writer.flush()
        writer.close()

    def test_missing_tensorboard_raises_when_the_arm_declared_tracking(self, monkeypatch) -> None:
        """A run that ASKED for tracking and got none must not exit 0.

        Non-negotiable #3: the previous code logged a warning and carried on,
        which is the 'passed-with-warnings' outcome pitfall #10 calls the most
        dangerous one.
        """
        monkeypatch.setattr(tbw, "SummaryWriter", None)
        with pytest.raises(RuntimeError, match="tensorboard"):
            TensorBoardWriter(Path("/runs/a"), _config())

    def test_missing_tensorboard_only_warns_when_tracking_was_defaulted(
        self, monkeypatch, caplog
    ) -> None:
        """`tensorboard` is in the `viz` EXTRA but `service` defaults to it.

        So raising on an arm that never mentioned `logging.tracking` would
        reject the library default for every run in a core-only install -- a
        different failure from the one above, and one this must not conflate.
        """
        monkeypatch.setattr(tbw, "SummaryWriter", None)
        writer = TensorBoardWriter(Path("/runs/a"), LoggingConfigSchema())
        assert not writer.enabled
        assert any("tensorboard" in r.message.lower() for r in caplog.records)

    def test_the_two_cases_differ_only_by_what_the_yaml_declared(self, monkeypatch) -> None:
        """Pins the discriminator: an explicit value EQUAL to the default still
        counts as declared, which a value comparison could not tell."""
        monkeypatch.setattr(tbw, "SummaryWriter", None)
        assert not LoggingConfigSchema().tracking.model_fields_set
        declared = LoggingConfigSchema(tracking={"service": TrackingService.TENSORBOARD})
        assert declared.tracking.model_fields_set == {"service"}
        with pytest.raises(RuntimeError):
            TensorBoardWriter(Path("/runs/a"), declared)

    def test_it_does_not_raise_when_tracking_is_declared_off(self, monkeypatch) -> None:
        """Anti-vacuity: absence only matters if tracking was actually wanted."""
        monkeypatch.setattr(tbw, "SummaryWriter", None)
        assert not TensorBoardWriter(Path("/runs/a"), _config(service=TrackingService.NONE)).enabled


class TestResume:
    def test_purge_step_is_the_resume_iteration(self, recording, tmp_path) -> None:
        """Without it a resumed run keeps the pre-crash tail and every chart
        folds back on itself at the resume point."""
        TensorBoardWriter(tmp_path, _config(), start_iteration=5000)
        assert recording[0].purge_step == 5000

    def test_a_fresh_run_passes_no_purge_step(self, recording, tmp_path) -> None:
        """`purge_step=0` is not the same as None to SummaryWriter."""
        TensorBoardWriter(tmp_path, _config(), start_iteration=0)
        assert recording[0].purge_step is None


class TestFeatures:
    @pytest.fixture
    def writer(self, recording, tmp_path):
        return TensorBoardWriter(tmp_path, _config()), recording

    def test_scalars_are_prefixed_and_non_numerics_dropped(self, writer) -> None:
        w, made = writer
        w.scalars({"loss": 0.5, "note": "text"}, 7, "train")
        assert made[0].calls == [("scalar", "train/loss", 0.5, 7)]

    def test_grouped_scalars_share_one_axis(self, writer) -> None:
        """`add_scalars` answers 'is D winning'; separate charts cannot."""
        w, made = writer
        w.grouped_scalars("adv", {"g": 1.0, "d": 2.0, "bad": None}, 3)
        assert made[0].calls == [("scalars", "adv", {"g": 1.0, "d": 2.0}, 3)]

    def test_grouped_scalars_skip_an_all_non_numeric_group(self, writer) -> None:
        w, made = writer
        w.grouped_scalars("adv", {"bad": None}, 3)
        assert made[0].calls == []

    def test_hparams_stringifies_rather_than_dropping(self, writer) -> None:
        """An omitted hyper-parameter is the confound this dashboard exists to
        surface (pitfall #17), so a non-scalar is recorded, not skipped."""
        w, made = writer
        w.record_hparam_metric("psnr", 31.5)
        w.hparams({"lr": 1e-4, "mode": "gan", "amp": True, "shape": [1, 2]})
        kind, hp, metrics = made[0].calls[-1]
        assert kind == "hparams"
        assert hp == {"lr": 1e-4, "mode": "gan", "amp": True, "shape": "[1, 2]"}
        assert metrics == {"psnr": 31.5}

    def test_hparams_without_metrics_still_writes(self, writer) -> None:
        """TensorBoard drops an hparams entry with no metric dict at all."""
        w, made = writer
        w.hparams({"lr": 1e-4})
        assert made[0].calls[-1][2] == {"hparam/noop": 0.0}


class TestHistogramCadence:
    """`add_histogram` copies each parameter to the host -- a GPU sync per
    tensor. The cadence gate is a training-loop performance invariant
    (non-negotiable #9), not a preference."""

    class _Param:
        def __init__(self) -> None:
            self.requires_grad = True
            self.grad = None

    class _Model:
        def __init__(self, touched: list) -> None:
            self._touched = touched

        def named_parameters(self):
            self._touched.append(True)
            return iter([("enc.w", TestHistogramCadence._Param())])

    def test_an_off_cadence_step_returns_before_touching_a_tensor(
        self, recording, tmp_path
    ) -> None:
        touched: list = []
        cfg = LoggingConfigSchema(intervals={"histogram": 100})
        w = TensorBoardWriter(tmp_path, cfg)
        w.histograms(self._Model(touched), step=37, prefix="train")
        assert not touched, "named_parameters() must not be reached off-cadence"
        assert recording[0].calls == []

    def test_an_on_cadence_step_writes_weights(self, recording, tmp_path) -> None:
        touched: list = []
        cfg = LoggingConfigSchema(intervals={"histogram": 100})
        w = TensorBoardWriter(tmp_path, cfg)
        w.histograms(self._Model(touched), step=200, prefix="train")
        assert touched
        assert recording[0].kinds() == ["histogram"]
        assert recording[0].calls[0][1] == "train/weights/enc/w"

    def test_the_default_cadence_is_coarse(self) -> None:
        """A per-step default would put a sync in the hot loop for every arm."""
        assert LoggingConfigSchema().intervals.histogram == 1000
