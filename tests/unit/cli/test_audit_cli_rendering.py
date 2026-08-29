"""Round-11 regression: rich-console rendering must not leak ANSI codes when piped.

Audit anchor: TODO/audit/smoke_audit_20260516.md §F-COLOR.

The round-11 retrofit (2026-05-17) added a rich ``Console`` to the
audit CLI so ``python -m mriforge.cli audit foo.yaml`` is colored on a
TTY. The hard invariant downstream parsers depend on: **when stdout
is NOT a TTY, the output must contain ZERO ANSI escape codes** and
be byte-identical to the pre-round-11 plain-text rendering.

These tests:

1. Pin the ``HealthCheckResult.__rich__`` API so a future refactor
   that removes it gets caught.
2. Verify ``__rich__`` returns a styled object (the cluster won't
   see colors otherwise).
3. Run the audit CLI as a subprocess with stdout piped and assert
   no ANSI codes leak through. The smoke wrapper / CI pipe the audit
   output to log files; a regression here would corrupt them with
   escape garbage.
4. Verify ``--json`` output is still valid JSON.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_YAML = REPO_ROOT / "experiments" / "active" / "dummy_basic.yaml"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _audit(*extra_args: str, force_color: bool = False) -> subprocess.CompletedProcess:
    """Run ``python -m mriforge.cli audit <yaml>`` as a subprocess (always non-TTY here)."""
    env = os.environ.copy()
    if force_color:
        env["FORCE_COLOR"] = "1"
    else:
        # Defensive: strip any inherited FORCE_COLOR / CLICOLOR.
        env.pop("FORCE_COLOR", None)
        env.pop("CLICOLOR_FORCE", None)
        # NO_COLOR is widely respected. Set it so rich definitely
        # disables ANSI even when a downstream lib forces TTY.
        env["NO_COLOR"] = "1"
    return subprocess.run(
        ["python", "-m", "mriforge.cli", "audit", str(EXAMPLE_YAML), *extra_args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env=env, timeout=120,
    )


# ── 1. HealthCheckResult.__rich__ surface ────────────────────────────


def test_health_check_result_has_rich_method() -> None:
    """The __rich__ method is the contract rich-console relies on."""
    from mriforge.infrastructure.validation.config_health_checker import (
        HealthCheckResult,
    )
    r = HealthCheckResult(
        passed=False, check_name="dummy", message="m", severity="error",
    )
    assert hasattr(r, "__rich__"), (
        "F-COLOR regression: HealthCheckResult must define __rich__ so "
        "the audit CLI emits color when stdout is a TTY. See "
        "TODO/audit/smoke_audit_20260516.md §F-COLOR."
    )


def test_health_check_result_rich_returns_styled_text() -> None:
    """``__rich__`` returns a rich ``Text`` with at least one styled span."""
    from mriforge.infrastructure.validation.config_health_checker import (
        HealthCheckResult,
    )
    from rich.text import Text
    r = HealthCheckResult(
        passed=False, check_name="x", message="boom", severity="error",
    )
    rendered = r.__rich__()
    assert isinstance(rendered, Text), (
        "F-COLOR: __rich__ must return rich.text.Text for proper "
        "TTY-aware rendering."
    )
    # The Text must carry at least one styled span — otherwise the
    # output is indistinguishable from the plain __str__ and adding
    # rich was pointless.
    has_style = any(span.style for span in rendered.spans if hasattr(span, "style"))
    assert has_style, (
        "F-COLOR: __rich__ returned a Text with no styled spans. "
        "Colors will never appear on the cluster TTY."
    )


def test_health_check_result_rich_pass_uses_green_icon() -> None:
    """Pass results render with a green ✅ icon."""
    from mriforge.infrastructure.validation.config_health_checker import (
        HealthCheckResult,
    )
    r = HealthCheckResult(passed=True, check_name="x", message="ok")
    rendered = r.__rich__()
    plain = rendered.plain
    assert plain.startswith("✅"), (
        f"Pass icon ✅ should be the first character of __rich__ "
        f"output; got {plain[:20]!r}"
    )
    # Find a green-styled span:
    spans_with_green = [
        s for s in rendered.spans
        if hasattr(s, "style") and s.style and "green" in str(s.style)
    ]
    assert spans_with_green, (
        "Pass result must include at least one span with 'green' "
        "style — otherwise the cluster sees a plain ✅ with no color."
    )


def test_health_check_result_rich_error_uses_red_icon() -> None:
    """Error results render with a red ❌ icon."""
    from mriforge.infrastructure.validation.config_health_checker import (
        HealthCheckResult,
    )
    r = HealthCheckResult(
        passed=False, check_name="x", message="boom", severity="error",
    )
    rendered = r.__rich__()
    assert rendered.plain.startswith("❌")
    spans_with_red = [
        s for s in rendered.spans
        if hasattr(s, "style") and s.style and "red" in str(s.style)
    ]
    assert spans_with_red, (
        "Error result must include at least one 'red' span — without "
        "it the cluster sees a plain ❌ with no urgency cue."
    )


# ── 2. Console module ────────────────────────────────────────────────


def test_console_module_exports_shared_console() -> None:
    """``mriforge.cli._rendering`` must export a module-level ``console``."""
    from mriforge.cli._rendering import console
    from rich.console import Console
    assert isinstance(console, Console), (
        "F-COLOR: mriforge.cli._rendering.console must be a rich.Console "
        "instance. Without it, the audit CLI can't reliably emit "
        "ANSI markup."
    )


def test_console_soft_wrap_enabled() -> None:
    """``soft_wrap=True`` so long messages don't get hard-broken (grep stays usable)."""
    from mriforge.cli._rendering import console
    assert getattr(console, "soft_wrap", False) is True, (
        "Console must have soft_wrap=True; otherwise rich inserts "
        "hard line breaks when the terminal is narrow and breaks "
        "grep-friendly output."
    )


# ── 3. End-to-end: non-TTY output is ANSI-free ───────────────────────


@pytest.mark.skipif(
    not EXAMPLE_YAML.exists(),
    reason=f"sample YAML missing: {EXAMPLE_YAML}",
)
def test_audit_cli_emits_no_ansi_when_piped() -> None:
    """The hard CI invariant: piped audit output must contain ZERO ANSI codes.

    The smoke wrapper, CI, and the strict-real CI gates all pipe the
    audit output. If round-11 leaked ANSI here, every downstream log
    file would be corrupted with escape sequences. Rich's auto-TTY
    detection is what protects us — this test pins it.
    """
    proc = _audit()
    assert proc.returncode in (0, 1, 2), (
        f"Unexpected audit exit code {proc.returncode}; stderr:\n"
        f"{proc.stderr[-500:]}"
    )
    leaks = ANSI_RE.findall(proc.stdout)
    assert not leaks, (
        f"F-COLOR regression: audit CLI leaked {len(leaks)} ANSI "
        f"escape sequence(s) when stdout was piped (not a TTY). "
        f"The smoke wrapper pipes audit output to log files; any "
        f"escape codes there would corrupt grep / process_smoke_log "
        f"parsing. First leak: {leaks[0]!r}. See "
        f"TODO/audit/smoke_audit_20260516.md §F-COLOR."
    )


@pytest.mark.skipif(
    not EXAMPLE_YAML.exists(),
    reason=f"sample YAML missing: {EXAMPLE_YAML}",
)
def test_audit_cli_json_still_valid_json() -> None:
    """``--json`` output is bytes the consumer can parse with ``json.loads``."""
    proc = _audit("--json")
    # JSON may be preceded by CUDA deprecation warnings on stderr,
    # but stdout should be clean JSON.
    stdout = proc.stdout.strip()
    assert stdout, "audit --json emitted no stdout"
    # Sometimes a leading FutureWarning leaks to stdout; tolerate it
    # by finding the first {.
    json_start = stdout.find("{")
    assert json_start >= 0, (
        f"No JSON object in --json output; got:\n{stdout[:500]}"
    )
    try:
        data = json.loads(stdout[json_start:])
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"F-COLOR regression: --json output is no longer valid "
            f"JSON. Likely cause: rich markup leaked into the JSON "
            f"path. Error: {exc}\nFirst 500 chars:\n{stdout[:500]}"
        )
    # Spot-check the schema.
    assert "passed" in data
    assert "config_path" in data


@pytest.mark.skipif(
    not EXAMPLE_YAML.exists(),
    reason=f"sample YAML missing: {EXAMPLE_YAML}",
)
def test_audit_cli_force_color_does_emit_ansi() -> None:
    """``FORCE_COLOR=1`` should bring back the ANSI escape codes.

    This is the cluster-fix path: if the cluster shell suppresses TTY
    detection (e.g., ``sbatch`` redirects stdout), the user sets
    ``FORCE_COLOR=1`` and gets colors back. Pin the behavior.
    """
    proc = _audit(force_color=True)
    assert ANSI_RE.search(proc.stdout), (
        "F-COLOR: FORCE_COLOR=1 should re-enable rich's ANSI output "
        "even when stdout is not a TTY. Without this knob, cluster "
        "users running `sbatch` have no way to see colors."
    )
