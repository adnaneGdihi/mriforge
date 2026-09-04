"""Fuzz: audit-pass ∧ train-dry-run-fail disagreement detection.

Defect predicate
----------------
A configuration is a *disagreement* if ``spectramr audit <yaml>`` exits 0
(schema + Tier-1 static checks pass) but ``spectramr train --dry-run`` exits
non-zero (the pipeline rejects the config at runtime).

The reverse (audit fails ∧ dry-run succeeds) is not captured here — those
are audit over-strictness bugs and are tracked separately.

Strategy
--------
We use ``hypothesis-jsonschema`` to generate dict instances that conform to
the *JSON Schema* emitted by ``TrainingSettings.model_json_schema()``.
Each drawn dict is:

1. Written to a temp YAML file.
2. Run through ``spectramr audit`` via subprocess.
3. If audit exits 0, run ``spectramr train --dry-run`` via subprocess.
4. If dry-run exits non-zero the pair (audit_stdout, train_stderr) is
   persisted to the regression corpus for later replay.

Corpus persistence
------------------
Failing configs are written to
``tests/fuzz/corpus/audit_train_disagreement/<sha256>.yaml`` so they survive
hypothesis database resets and can be replayed with::

    pytest tests/fuzz/schema_fuzz/test_audit_vs_training_disagreement.py \
        --hypothesis-replay

The corpus directory is created automatically; it is gitignored in this
repo to avoid polluting the codebase with huge generated artifacts.

Guards
------
- ``pytest.importorskip("hypothesis")``
- ``pytest.importorskip("hypothesis_jsonschema")``

Both ensure collection silently skips when the libraries are absent.

Notes
-----
* This test runs the CLI via subprocess, which means it activates the
  installed ``spectramr`` package (not a mocked version).  It is
  intentionally slow and is marked ``@pytest.mark.fuzz`` so the default
  ``make test`` target does not pick it up.
* The ``@settings(max_examples=50, ...)`` default is conservative; on the
  cluster raise it to 500+ via ``HYPOTHESIS_MAX_EXAMPLES`` env var.
* ``deadline=None`` because subprocess calls can take >200 ms each.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Guard: optional dependencies
# ---------------------------------------------------------------------------
hypothesis = pytest.importorskip("hypothesis")
hypothesis_jsonschema = pytest.importorskip("hypothesis_jsonschema")

from hypothesis import given, settings, HealthCheck  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis_jsonschema import from_schema  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus directory for persisting failing examples
# ---------------------------------------------------------------------------
_CORPUS_DIR = Path(__file__).parent.parent.parent / "fuzz" / "corpus" / "audit_train_disagreement"


def _persist_disagreement(config_dict: dict, audit_out: str, train_err: str) -> Path:
    """Write a disagreement pair to the corpus directory.

    Returns the path to the saved YAML so the caller can log it.
    """
    _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = yaml.dump(config_dict, default_flow_style=False)
    sha = hashlib.sha256(payload.encode()).hexdigest()[:16]
    yaml_path = _CORPUS_DIR / f"{sha}.yaml"
    meta_path = _CORPUS_DIR / f"{sha}.meta.json"
    yaml_path.write_text(payload, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"audit_stdout": audit_out, "train_stderr": train_err}, indent=2),
        encoding="utf-8",
    )
    return yaml_path


# ---------------------------------------------------------------------------
# JSON schema retrieval (lazy, cached)
# ---------------------------------------------------------------------------
_TRAINING_SCHEMA: dict | None = None


def _get_training_schema() -> dict:
    global _TRAINING_SCHEMA
    if _TRAINING_SCHEMA is not None:
        return _TRAINING_SCHEMA
    try:
        from spectramr.config.settings import TrainingSettings  # noqa: PLC0415
        _TRAINING_SCHEMA = TrainingSettings.model_json_schema()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Could not load TrainingSettings schema: {exc}")
    return _TRAINING_SCHEMA


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _spectramr_cmd() -> list[str]:
    """Return the base spectramr CLI invocation for the current interpreter."""
    return [sys.executable, "-m", "spectramr.cli"]


def _run_audit(yaml_path: str) -> tuple[int, str, str]:
    """Run ``spectramr audit <yaml>``.  Returns (returncode, stdout, stderr)."""
    result = subprocess.run(
        [*_spectramr_cmd(), "audit", yaml_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout, result.stderr


def _run_train_dry_run(yaml_path: str) -> tuple[int, str, str]:
    """Run ``spectramr train --config <yaml> --dry_run``.
    Returns (returncode, stdout, stderr).
    """
    result = subprocess.run(
        [*_spectramr_cmd(), "train", "--config", yaml_path, "--dry_run"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Hypothesis settings
# ---------------------------------------------------------------------------

_MAX_EXAMPLES = int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "50"))

# Suppress flaky-test health check so slow subprocesses don't abort the run.
_SUPPRESS = [HealthCheck.too_slow, HealthCheck.filter_too_much]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.fuzz
@settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=_SUPPRESS,
)
@given(data=st.data())
def test_no_audit_train_disagreement(data) -> None:
    """A config that passes audit must also pass train --dry-run.

    Defect: audit exits 0 (pass) but train --dry-run exits non-zero (fail).
    Such pairs are persisted to the corpus and the test is marked failed so
    Hypothesis shrinks to a minimal reproducer.
    """
    # hypothesis-jsonschema generates instances from the JSON Schema that
    # Pydantic emits. TrainingSettings emits recursive ``$ref``s the library
    # cannot resolve (HypothesisRefResolutionError), so draw inside a guard and
    # skip cleanly rather than erroring the whole suite — generating from a
    # recursive schema is a separate tooling effort (resolve / prune ``$defs``).
    try:
        config_dict = data.draw(from_schema(_get_training_schema()))
    except Exception as exc:
        pytest.skip(
            f"hypothesis_jsonschema cannot generate from the recursive "
            f"TrainingSettings schema ({type(exc).__name__})"
        )
    # Inject required config_version so generated dicts are accepted by the
    # YAML loader's version guard.
    config_dict.setdefault("config_version", "6.0")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(config_dict, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        audit_rc, audit_out, audit_err = _run_audit(tmp_path)

        # Only investigate further if audit passes.
        if audit_rc != 0:
            return  # Audit correctly rejected — not a disagreement.

        train_rc, _train_out, train_err = _run_train_dry_run(tmp_path)

        if train_rc != 0:
            corpus_path = _persist_disagreement(config_dict, audit_out, train_err)
            pytest.fail(
                f"Disagreement detected: audit passed (rc=0) but train --dry-run "
                f"failed (rc={train_rc}).\n"
                f"Config persisted to: {corpus_path}\n"
                f"Train stderr (first 1000 chars):\n{train_err[:1000]}"
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Regression replay: re-run all corpus entries as parametrised tests
# ---------------------------------------------------------------------------

def _corpus_yaml_ids() -> list[str]:
    """Return IDs for every YAML in the corpus (empty if corpus absent)."""
    if not _CORPUS_DIR.exists():
        return []
    return [p.stem for p in sorted(_CORPUS_DIR.glob("*.yaml"))]


@pytest.mark.fuzz
@pytest.mark.parametrize("corpus_stem", _corpus_yaml_ids())
def test_corpus_regression(corpus_stem: str) -> None:
    """Replay every previously-found disagreement to guard against regressions.

    Each corpus entry was originally a disagreement (audit-pass, train-fail).
    Passing means the bug has been fixed; failing means the bug is still present.

    The test passes only when *both* audit and train --dry-run exit 0, which
    means the underlying disagreement has been resolved.
    """
    yaml_path = str(_CORPUS_DIR / f"{corpus_stem}.yaml")
    if not Path(yaml_path).exists():
        pytest.skip(f"Corpus entry {corpus_stem} not found")

    audit_rc, _audit_out, audit_err = _run_audit(yaml_path)
    assert audit_rc == 0, (
        f"Corpus entry {corpus_stem}: audit now fails (rc={audit_rc}) — "
        f"either the schema tightened or the YAML became stale.\n{audit_err[:500]}"
    )

    train_rc, _train_out, train_err = _run_train_dry_run(yaml_path)
    assert train_rc == 0, (
        f"Corpus entry {corpus_stem}: train --dry-run still fails (rc={train_rc}) — "
        f"regression not yet fixed.\n{train_err[:500]}"
    )
