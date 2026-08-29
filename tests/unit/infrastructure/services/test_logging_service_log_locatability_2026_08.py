"""A run's log must be locatable, and must not vanish silently.

The `experiment_11_attention_none` run of 2026-08-18 wrote `provenance.json`, a
`resolved_config.json`, a TensorBoard event file, eight `debug_snapshots/` and
sixteen PNGs -- and **not one log line**. Two mechanisms, both measured:

* `logging.sinks.dir` is authoritative over the run directory. That precedence is
  CORRECT per non-negotiable 3b (a declared value must not be replaced by a
  caller default); the defect was that the resolved path never reached any
  artifact, so a log could not be found from the run it belonged to.
* The relocation fallback was silent *by construction*. Its `except` branch
  touched neither `self._logger` nor `warnings`, so the single event that moves a
  run's entire log announced itself nowhere -- and on a compute node the
  `tempfile.mkdtemp` destination is wiped at job teardown, so the log did not
  merely move, it ceased to exist while the run reported success.

These tests exercise the real `setup` against a real unwritable directory rather
than mocking the failure, because the thing under test is whether the branch can
speak at all.
"""

from __future__ import annotations

import logging as std_logging
import os
import warnings

import pytest

from mriforge.infrastructure.services.logging_service import LoggingService


def _service(name: str) -> LoggingService:
    svc = LoggingService(logger_name=name)
    # Each test gets a fresh logger; `setup` skips file-handler installation when
    # one already exists, so a leaked handler would make the next test vacuous.
    for handler in list(svc._logger.handlers):
        svc._logger.removeHandler(handler)
    return svc


class TestTheResolvedLogPathIsRecorded:
    """`self._log_dir` holds the INTENDED directory and survives relocation, so
    it cannot answer "where is this run's log".
    """

    def test_a_successful_setup_records_the_file_it_opened(self, tmp_path):
        svc = _service("locatability_ok")
        svc.setup(str(tmp_path), log_level="WARNING")
        assert svc.resolved_log_path == str(tmp_path / "locatability_ok.log")
        assert os.path.exists(svc.resolved_log_path)
        assert svc.log_dir_relocated_from is None

    def test_the_attributes_exist_even_with_file_logging_off(self, tmp_path):
        """A consumer must never have to tell "no attribute" from "no file log".
        `train.py` reads these to stamp provenance and runs for every arm,
        including those with `sinks.to_file: false`."""
        svc = _service("locatability_nofile")
        svc.setup("", log_level="WARNING")
        assert svc.resolved_log_path is None
        assert svc.log_dir_relocated_from is None


class TestRelocationAnnouncesItself:
    """The `except (PermissionError, OSError)` branch was structurally mute."""

    @staticmethod
    def _unwritable(tmp_path):
        target = tmp_path / "readonly"
        target.mkdir()
        os.chmod(target, 0o500)  # r-x: makedirs(exist_ok=True) passes, open fails
        return target

    @pytest.fixture
    def unwritable(self, tmp_path):
        target = self._unwritable(tmp_path)
        yield target
        os.chmod(target, 0o700)  # let tmp_path cleanup remove it

    def test_it_warns_naming_both_the_intended_and_actual_directory(self, unwritable, caplog):
        svc = _service("locatability_relocated")
        with caplog.at_level(std_logging.WARNING), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            svc.setup(str(unwritable), log_level="WARNING")

        if svc.log_dir_relocated_from is None:
            pytest.skip("this filesystem/user can write to a 0o500 directory")

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert str(unwritable) in messages, (
            "the relocation warning does not name the directory that failed"
        )
        assert svc.resolved_log_path in messages, (
            "the relocation warning does not name where the log actually went"
        )
        assert "wiped at job teardown" in messages, (
            "the warning must say the temp destination is not durable; a log that "
            "silently disappears at teardown is the failure being reported"
        )

    def test_it_also_raises_a_python_warning(self, unwritable):
        """A LOG warning about the log sink failing is the message most likely to
        be lost, so the failure is reported through both channels."""
        svc = _service("locatability_pywarn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svc.setup(str(unwritable), log_level="WARNING")

        if svc.log_dir_relocated_from is None:
            pytest.skip("this filesystem/user can write to a 0o500 directory")
        assert any(
            issubclass(w.category, RuntimeWarning) and "relocated" in str(w.message) for w in caught
        ), f"no RuntimeWarning about the relocation: {[str(w.message) for w in caught]}"

    def test_the_relocation_is_recorded_for_provenance(self, unwritable):
        svc = _service("locatability_record")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svc.setup(str(unwritable), log_level="WARNING")

        if svc.log_dir_relocated_from is None:
            pytest.skip("this filesystem/user can write to a 0o500 directory")
        assert svc.log_dir_relocated_from == str(unwritable)
        assert svc.resolved_log_path and str(unwritable) not in svc.resolved_log_path
        # Logging still works -- refusing to fall back is NOT the fix here; the
        # run should keep going, it just must not do so silently.
        assert any(isinstance(h, std_logging.FileHandler) for h in svc._logger.handlers)

    def test_a_writable_directory_does_not_warn(self, tmp_path):
        """Anti-vacuity: the warning must not fire on the normal path."""
        svc = _service("locatability_quiet")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svc.setup(str(tmp_path), log_level="WARNING")
        assert not [w for w in caught if "relocated" in str(w.message)]
        assert svc.log_dir_relocated_from is None
