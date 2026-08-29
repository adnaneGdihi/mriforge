"""Paired tests for ``infrastructure/reporting/run_hook.py``.

The hook's *content* is already guarded by source-text pins in
``tests/unit/pipelines/test_train_reporting_hook.py``, and those keep working
through the move because ``inspect.getsource`` resolves an object to its real
definition. What no test covered before is the thing the move actually changed:
that the hook is one shared object rather than two drifting copies, and that its
three config-driven branches behave as documented from a non-training caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from mriforge.infrastructure.reporting import run_hook


@pytest.fixture
def logger_():
    return logging.getLogger("test_run_hook")


class TestTrainReExportsRatherThanCopies:
    """The failure mode this guards is a silent fork.

    If someone later pastes the body back into ``train.py`` instead of importing
    it, both verbs keep working and nothing fails -- while the training report
    and the inference report quietly diverge, which is the exact defect this
    whole change set exists to remove. Identity is the only cheap way to notice.
    """

    def test_train_hook_is_the_same_object(self):
        from mriforge.pipelines import train

        assert train._maybe_run_reporting is run_hook.maybe_run_reporting
        assert train._run_unconfigured_report is run_hook.run_unconfigured_report
        assert train._UNCONFIGURED_REPORT_TABLES is run_hook.UNCONFIGURED_REPORT_TABLES

    def test_the_hook_does_not_live_in_the_training_pipeline_any_more(self):
        import inspect

        from mriforge.pipelines import train

        src_file = inspect.getsourcefile(train._maybe_run_reporting)
        assert src_file is not None
        assert Path(src_file).name == "run_hook.py", (
            "the hook resolved back into pipelines/, which would put `report` "
            "behind `train` again for infer/predict"
        )


class TestBranchSelection:
    def test_absent_reporting_block_takes_the_tables_only_floor(
        self, tmp_path, logger_, monkeypatch
    ):
        seen = {}
        monkeypatch.setattr(
            run_hook,
            "run_unconfigured_report",
            lambda run_dir, lg: seen.update(run_dir=run_dir),
        )
        run_hook.maybe_run_reporting(
            SimpleNamespace(reporting=None), run_dir=tmp_path, logger_=logger_
        )
        assert seen["run_dir"] == tmp_path

    def test_disabled_reporting_block_also_takes_the_floor(self, tmp_path, logger_, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            run_hook,
            "run_unconfigured_report",
            lambda run_dir, lg: seen.update(run_dir=run_dir),
        )
        run_hook.maybe_run_reporting(
            SimpleNamespace(reporting=SimpleNamespace(enabled=False)),
            run_dir=tmp_path,
            logger_=logger_,
        )
        assert seen["run_dir"] == tmp_path

    def test_enabled_reporting_reaches_generate_report_with_the_run_dir(
        self, tmp_path, logger_, monkeypatch
    ):
        import mriforge.infrastructure.reporting as reporting_pkg

        seen = {}

        def _fake(run_dir, **kwargs):
            seen["run_dir"] = run_dir
            seen["kwargs"] = kwargs
            return {"figures": {}, "tables": {}, "tikz": {}, "out_dir": run_dir}

        monkeypatch.setattr(reporting_pkg, "generate_report", _fake, raising=False)
        cfg = SimpleNamespace(
            reporting=SimpleNamespace(enabled=True, task="default"),
            run=SimpleNamespace(seed=7),
        )
        run_hook.maybe_run_reporting(cfg, run_dir=tmp_path, logger_=logger_)
        assert seen["run_dir"] == tmp_path
        # `run.seed` is the canonical location -- the flat read stamped None.
        assert seen["kwargs"]["seed"] == 7
        # method_name defaults to the run directory's name, not an empty string.
        assert seen["kwargs"]["method_name"] == tmp_path.name


class TestFailurePolicy:
    def _cfg(self, *, fail_on_error: bool):
        return SimpleNamespace(
            reporting=SimpleNamespace(enabled=True, task="default", fail_on_error=fail_on_error),
            run=SimpleNamespace(seed=0),
        )

    def _boom(self, *a, **k):
        raise RuntimeError("plot exploded")

    def test_soft_fails_by_default_so_a_long_run_still_wraps_up(
        self, tmp_path, logger_, monkeypatch
    ):
        import mriforge.infrastructure.reporting as reporting_pkg

        monkeypatch.setattr(reporting_pkg, "generate_report", self._boom, raising=False)
        run_hook.maybe_run_reporting(
            self._cfg(fail_on_error=False), run_dir=tmp_path, logger_=logger_
        )  # must not raise

    def test_fail_on_error_propagates(self, tmp_path, logger_, monkeypatch):
        import mriforge.infrastructure.reporting as reporting_pkg

        monkeypatch.setattr(reporting_pkg, "generate_report", self._boom, raising=False)
        with pytest.raises(RuntimeError, match="plot exploded"):
            run_hook.maybe_run_reporting(
                self._cfg(fail_on_error=True), run_dir=tmp_path, logger_=logger_
            )
