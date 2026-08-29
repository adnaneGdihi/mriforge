"""Tests for ``LossLoggingConfigSchema``.

Targets ``mriforge.config.schemas.loss_logging``. A minimal Pydantic schema whose
legacy keys are silently ignored (``extra="ignore"``).

**Which fields are actually consumed, measured 2026-08-09** — the previous claim
here ("two consumed fields, ``frequency`` and ``output_dir``") was half wrong,
and a duplicate of this file at ``tests/unit/config/test_loss_logging_schema.py``
repeated it verbatim. That file has been deleted; this one is on the canonical
pairing path for ``src/mriforge/config/schemas/loss_logging.py``.

===============  ==========================================================
``output_dir``   read — ``strategies/pinn_strategy.py:109``; written by
                 ``pipelines/hpo.py:439``
``csv_path``     written by ``pipelines/hpo.py:438``, checked by
                 ``validation/paired_arms_audit.py:111``
``frequency``    **no reader anywhere in src/.** Its named consumer,
                 ``execution_engine.py``, was deleted in the 2026-06
                 dead-orchestration sweep, so this is an orphaned field
                 rather than an unwired knob (pitfall #15). Filed as an issue
                 rather than removed here: dropping it would change what an
                 existing YAML validates against.
===============  ==========================================================
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.loss_logging import LossLoggingConfigSchema

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default frequency = 1, output_dir = None."""
    cfg = LossLoggingConfigSchema()
    assert cfg.frequency == 1
    assert cfg.output_dir is None


def test_explicit_construction() -> None:
    """Explicit values are stored."""
    cfg = LossLoggingConfigSchema(frequency=10, output_dir="/scratch/logs")
    assert cfg.frequency == 10
    assert cfg.output_dir == "/scratch/logs"


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_frequency_ge_one() -> None:
    """``frequency >= 1``."""
    with pytest.raises(ValidationError):
        LossLoggingConfigSchema(frequency=0)


def test_frequency_negative_raises() -> None:
    """Negative frequency rejected."""
    with pytest.raises(ValidationError):
        LossLoggingConfigSchema(frequency=-5)


# ---------------------------------------------------------------------------
# Frozen + extra='ignore'
# ---------------------------------------------------------------------------


def test_schema_is_frozen() -> None:
    """Cannot mutate after construction."""
    cfg = LossLoggingConfigSchema()
    with pytest.raises(ValidationError):
        cfg.frequency = 99


def test_reader_less_legacy_fields_are_silently_ignored() -> None:
    """Fields removed in 2026-05 that genuinely have no reader don't raise.

    They stay undeclared on purpose: declaring an unread knob is pitfall #15,
    and ``extra='ignore'`` is what keeps the ~417 arms still setting ``enabled``
    loadable while that corpus debt is paid down.
    """
    cfg = LossLoggingConfigSchema(
        enabled=False,
        include_metrics=True,
        compute_psnr=True,
    )

    assert cfg.frequency == 1
    assert not hasattr(cfg, "enabled")


def test_csv_path_survives_the_load() -> None:
    """``csv_path`` has a producer and a consumer, so it must transport.

    ``hpo.py:438`` writes it into every trial YAML and
    ``paired_arms_audit.py:111`` audits it. The 2026-05 sweep removed it as
    "never wired", after which ``extra='ignore'`` discarded the write on load --
    the audit then read a key that could not survive the trip (#795). This test
    asserts the transport, so a re-removal fails loudly instead of going quiet.
    """
    cfg = LossLoggingConfigSchema(csv_path="/runs/trial_7/logs/loss_log.csv")

    assert cfg.csv_path == "/runs/trial_7/logs/loss_log.csv"


def test_csv_path_defaults_to_none() -> None:
    """An arm that does not set it must not acquire a fabricated path."""
    assert LossLoggingConfigSchema().csv_path is None


def test_output_dir_accepts_none_and_string() -> None:
    """``output_dir`` is ``str | None``."""
    cfg_none = LossLoggingConfigSchema(output_dir=None)
    cfg_str = LossLoggingConfigSchema(output_dir="/x")
    assert cfg_none.output_dir is None
    assert cfg_str.output_dir == "/x"
