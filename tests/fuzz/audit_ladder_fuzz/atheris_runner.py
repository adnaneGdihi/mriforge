"""Atheris coverage-guided fuzzer for ``ConfigHealthChecker.run_all_checks``.

Standalone script — NOT a pytest module.  Run directly::

    source .venv/bin/activate
    python tests/fuzz/audit_ladder_fuzz/atheris_runner.py      # finite run
    python tests/fuzz/audit_ladder_fuzz/atheris_runner.py -runs=100000

Or via the Makefile target::

    make fuzz-audit-ladder

Seeds
-----
The fuzzer seeds its corpus from every YAML in ``experiments/inprogress/**/*.yaml``
(relative to repo root, auto-detected from this file's path).  The YAML bytes
are used verbatim as initial corpus entries so the fuzzer starts from realistic
inputs rather than empty buffers.

TestOneInput contract
---------------------
For each fuzzed byte sequence we:

1. Attempt to decode as UTF-8 YAML.
2. Parse into a raw ``dict`` via ``yaml.safe_load``.
3. Inject ``config_version: '1.0'`` if missing (so the settings loader
   doesn't fail on the version guard before we reach the health checker).
4. Instantiate ``TrainingSettings(**data)`` — expected to raise
   ``ValidationError`` / ``ValueError`` / ``TypeError`` / ``KeyError``
   on malformed inputs.
5. If instantiation succeeds, run ``ConfigHealthChecker().run_all_checks(settings)``
   and assert the return value is a ``HealthCheckReport`` instance.

Non-Crash contract (the fuzzer's oracle)
----------------------------------------
Only exceptions that are NOT in the expected set
``{ValueError, TypeError, KeyError, pydantic.ValidationError, yaml.YAMLError,
   UnicodeDecodeError}`` are re-raised and treated as crashes.  Everything else
is silently swallowed — those are schema-rejection paths, not bugs.

The fuzzer therefore finds:

* Unhandled ``AttributeError`` / ``IndexError`` / ``AssertionError`` / etc.
  inside ``run_all_checks`` that the normal test suite never reaches.
* Crashes from the checker inspecting a field that Pydantic allowed through
  but the checker assumed would be absent.

Guards
------
``import atheris`` is guarded:  if atheris is not installed the script prints
a warning and exits 0 (so ``make fuzz-audit-ladder`` doesn't break CI).
"""

from __future__ import annotations

import sys
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard: atheris required
# ---------------------------------------------------------------------------
try:
    import atheris  # type: ignore[import-not-found]
    _ATHERIS_AVAILABLE = True
except ImportError:
    _ATHERIS_AVAILABLE = False

if not _ATHERIS_AVAILABLE:
    print(
        "WARNING: atheris is not installed. "
        "Install it with: pip install atheris\n"
        "Exiting 0 — fuzz-audit-ladder target is a no-op without atheris.",
        file=sys.stderr,
    )
    sys.exit(0)

# ---------------------------------------------------------------------------
# Imports (only reached when atheris is available)
# ---------------------------------------------------------------------------
import yaml  # noqa: E402

# pydantic ValidationError — we need the class for the expected-exception guard.
try:
    from pydantic import ValidationError as _PydanticValidationError
except ImportError:
    _PydanticValidationError = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Repo root detection
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
# Walk up until we find pyproject.toml (repo root marker).
_REPO_ROOT = _THIS_DIR
for _ in range(8):
    if (_REPO_ROOT / "pyproject.toml").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent

# Ensure src/ is on the path for imports.
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Lazy import of MRIForge modules (inside TestOneInput to avoid import-time
# crashes when the package is partially initialised)
# ---------------------------------------------------------------------------

def _import_mriforge():
    """Return (TrainingSettings, ConfigHealthChecker) — raises ImportError if unavailable."""
    from mriforge.config.settings import TrainingSettings  # noqa: PLC0415
    from mriforge.infrastructure.validation.config_health_checker import (  # noqa: PLC0415
        ConfigHealthChecker,
        HealthCheckReport,
    )
    return TrainingSettings, ConfigHealthChecker, HealthCheckReport


# Populate once at startup (but after atheris is confirmed available).
try:
    _TrainingSettings, _ConfigHealthChecker, _HealthCheckReport = _import_mriforge()
    _MRIFORGE_AVAILABLE = True
except Exception as _e:
    logger.warning("MRIForge modules unavailable: %s", _e)
    _MRIFORGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Expected exceptions (swallowed by the fuzzer — not crashes)
# ---------------------------------------------------------------------------
_EXPECTED: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    yaml.YAMLError,
    UnicodeDecodeError,
)
if _PydanticValidationError is not None:
    _EXPECTED = _EXPECTED + (_PydanticValidationError,)


# ---------------------------------------------------------------------------
# TestOneInput — the atheris fuzz target
# ---------------------------------------------------------------------------

def TestOneInput(data: bytes) -> None:  # noqa: N802 (atheris convention: PascalCase)
    """Fuzz target invoked by atheris for each generated byte sequence."""
    if not _MRIFORGE_AVAILABLE:
        return

    fdp = atheris.FuzzedDataProvider(data)  # type: ignore[attr-defined]

    try:
        raw_bytes = fdp.ConsumeBytes(len(data))
        text = raw_bytes.decode("utf-8", errors="replace")
        parsed = yaml.safe_load(text)
    except _EXPECTED:
        return

    if not isinstance(parsed, dict):
        return

    # Inject required version field.
    parsed.setdefault("config_version", "6.0")

    try:
        settings_obj = _TrainingSettings(**parsed)
    except _EXPECTED:
        return
    except Exception:
        # Unexpected exception from Pydantic internals — re-raise for atheris.
        raise

    # If we successfully built a settings object, run the health checker.
    try:
        checker = _ConfigHealthChecker()
        report = checker.run_all_checks(settings_obj)
    except _EXPECTED:
        # Health checker may legitimately raise for pathological configs.
        return
    except Exception:
        # Unhandled exception inside the health checker — this is a bug.
        raise

    # Soft assert: run_all_checks should always return a HealthCheckReport.
    if not isinstance(report, _HealthCheckReport):
        raise AssertionError(
            f"run_all_checks returned {type(report).__name__!r} instead of "
            f"HealthCheckReport. This is a health-checker contract violation."
        )


# ---------------------------------------------------------------------------
# Seed corpus from experiments/inprogress/**/*.yaml
# ---------------------------------------------------------------------------

def _collect_seed_corpus() -> list[bytes]:
    """Read every inprogress YAML and return their raw bytes as seed entries."""
    corpus: list[bytes] = []
    inprogress_dir = _REPO_ROOT / "experiments" / "inprogress"
    if not inprogress_dir.exists():
        logger.warning("Seed corpus dir not found: %s", inprogress_dir)
        return corpus
    for yaml_path in sorted(inprogress_dir.rglob("*.yaml")):
        try:
            corpus.append(yaml_path.read_bytes())
        except OSError as exc:
            logger.warning("Could not read seed %s: %s", yaml_path, exc)
    logger.info("Loaded %d seed corpus entries from %s", len(corpus), inprogress_dir)
    return corpus


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    seeds = _collect_seed_corpus()
    # atheris requires the fuzz target to be wrapped.
    atheris.instrument_imports()  # type: ignore[attr-defined]
    atheris.Setup(  # type: ignore[attr-defined]
        sys.argv,
        atheris.instrument_func(TestOneInput),  # type: ignore[attr-defined]
        # Pass seed corpus entries; atheris copies them into its internal pool.
        # We write them to a temp dir that atheris can read.
        # If running with -corpus_dir=<path>, atheris will also save minimised
        # crashes there for later replay.
    )
    if seeds:
        import tempfile
        import shutil
        _seed_dir = Path(tempfile.mkdtemp(prefix="atheris_seeds_"))
        try:
            for i, s in enumerate(seeds):
                (_seed_dir / f"seed_{i:06d}").write_bytes(s)
            # Prepend seed dir to argv so atheris picks it up.
            sys.argv.insert(1, str(_seed_dir))
            atheris.Fuzz()  # type: ignore[attr-defined]
        finally:
            shutil.rmtree(_seed_dir, ignore_errors=True)
    else:
        atheris.Fuzz()  # type: ignore[attr-defined]
