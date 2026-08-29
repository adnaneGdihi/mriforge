"""Regression test for the audit-13-F11 fix (D7).

``src/core/metrics/__init__.py`` used to import 14 metric modules by
hand. The list drifted: ``hallucination_metrics`` was missing for
months (audit 13 F2). Replacing the hand list with
``pkgutil.walk_packages`` removes the drift surface.

This test pins the walk-discovery path so a future refactor can't
silently revert to a hand list, and confirms the previously-tricky
``hallucination_metrics`` decorators fire end-to-end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def test_init_uses_pkgutil_walk_packages() -> None:
    src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "mriforge"
        / "core"
        / "metrics"
        / "__init__.py"
    ).read_text()
    assert "pkgutil.walk_packages" in src, (
        "src/mriforge/core/metrics/__init__.py reverted to a hand-maintained "
        "import list. Use pkgutil.walk_packages — audit 13 F11 (D7) — "
        "so adding a new metric module doesn't require editing __init__."
    )


def test_init_has_no_silent_passthrough_on_import_error() -> None:
    src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "mriforge"
        / "core"
        / "metrics"
        / "__init__.py"
    ).read_text()
    # The old code had ``try: <bulk import>; except ImportError: pass``
    # which deleted metrics with no diagnostic. The new code logs the
    # failing module name.
    #
    # Match the handler BODY, not a text split on the first "except
    # ImportError" in the file: that phrase now also appears in a docstring,
    # so the split-based form was satisfied by prose rather than by code.
    assert "except ImportError" in src
    assert not re.search(r"except ImportError[^:\n]*:\s*\n\s*pass\b", src), (
        "metrics/__init__.py reintroduced silent ImportError passthrough. "
        "Log the failing module so missing decorators are diagnosable."
    )


@pytest.mark.parametrize(
    "metric_name",
    [
        "feature_fidelity_index",
        "fabrication_rate",
        "hfen",
    ],
)
def test_decorators_fired_under_walk_discovery(metric_name: str) -> None:
    """The walk must reach hallucination_metrics + hfen (both were
    historically flaky)."""
    from mriforge.core.metrics import list_available

    assert metric_name in list_available(), (
        f"@register_metric on '{metric_name}' did not fire. Check the "
        "walk-discovery path in src/core/metrics/__init__.py — audit 13 F11."
    )


def test_no_dead_metrics_initialized_guard() -> None:
    """The pre-fix ``_METRICS_INITIALIZED`` guard was a no-op (local
    variable initialised to False on every import, so the ``or not``
    branch never short-circuited). Keep it gone — the module cache
    handles idempotency."""
    import mriforge.core.metrics as metrics_pkg

    assert not hasattr(metrics_pkg, "_METRICS_INITIALIZED"), (
        "The dead _METRICS_INITIALIZED guard is back. Python's module "
        "cache already provides per-process idempotency for imports; the "
        "guard added no value and confused the contract — audit 13 F11."
    )


# ---------------------------------------------------------------------------
# Walk-discovery error split (6.1): in-repo poisoning re-raises; a missing
# third-party optional dep is warned + recorded in MISSING_OPTIONAL_DEPS.
# ---------------------------------------------------------------------------


def test_missing_optional_deps_is_a_public_dict() -> None:
    """The introspection surface exists and is a mapping so audits/tests can
    read the skipped-module set instead of parsing warning logs."""
    import mriforge.core.metrics as metrics_pkg

    assert isinstance(metrics_pkg.MISSING_OPTIONAL_DEPS, dict)
    assert "MISSING_OPTIONAL_DEPS" in metrics_pkg.__all__


@pytest.mark.parametrize(
    "root_name",
    ["mriforge", "mriforge.core.metrics.some_broken_module", ""],
)
def test_walk_import_error_reraises_in_repo_and_falsy(root_name: str) -> None:
    """An in-repo (``mriforge*``) or falsy-``name`` ImportError is the
    poisoning class — it must re-raise so a broken metric module can never
    silently vanish from the registry."""
    import mriforge.core.metrics as metrics_pkg

    exc = ImportError("boom", name=root_name or None)
    before = dict(metrics_pkg.MISSING_OPTIONAL_DEPS)
    with pytest.raises(ImportError):
        metrics_pkg._record_or_raise_walk_import_error("some.module", exc)
    # A poisoning failure is never recorded as an "optional dep" skip.
    assert before == metrics_pkg.MISSING_OPTIONAL_DEPS


def test_walk_import_error_records_third_party_dep(caplog) -> None:
    """A third-party root cause warns AND is recorded in
    MISSING_OPTIONAL_DEPS (walked-module → missing-dep), no raise."""
    import logging

    import mriforge.core.metrics as metrics_pkg

    module_name = "mriforge.core.metrics._fake_optional"
    exc = ModuleNotFoundError("No module named 'torchmetrics'", name="torchmetrics")
    metrics_pkg.MISSING_OPTIONAL_DEPS.pop(module_name, None)
    try:
        with caplog.at_level(logging.WARNING, logger="mriforge.core.metrics"):
            # Must not raise.
            metrics_pkg._record_or_raise_walk_import_error(module_name, exc)
        assert metrics_pkg.MISSING_OPTIONAL_DEPS[module_name] == "torchmetrics"
        assert any("missing optional dependency" in rec.getMessage() for rec in caplog.records)
    finally:
        metrics_pkg.MISSING_OPTIONAL_DEPS.pop(module_name, None)


def test_discover_reraises_when_a_fake_in_repo_module_is_broken(monkeypatch) -> None:
    """End-to-end: the discovery loop re-raises when a walked in-repo module
    fails to import (monkeypatched fake submodule), proving the package would
    refuse to import silently-incomplete rather than drop the module."""
    import mriforge.core.metrics as metrics_pkg

    fake = f"{metrics_pkg._PACKAGE_NAME}.fake_broken"

    def fake_walk(path, prefix, onerror=None):
        yield (None, fake, False)

    def fake_import(name):
        raise ImportError("in-repo boom", name=fake)

    monkeypatch.setattr(metrics_pkg.pkgutil, "walk_packages", fake_walk)
    monkeypatch.setattr(metrics_pkg.importlib, "import_module", fake_import)

    with pytest.raises(ImportError):
        metrics_pkg._discover_and_import_metric_modules()


def test_discover_records_when_a_fake_module_misses_third_party(monkeypatch) -> None:
    """End-to-end: a walked module whose only failure is a missing third-party
    dep is skipped (recorded, no raise) so the rest of the walk proceeds."""
    import mriforge.core.metrics as metrics_pkg

    fake = f"{metrics_pkg._PACKAGE_NAME}.fake_needs_piq"

    def fake_walk(path, prefix, onerror=None):
        yield (None, fake, False)

    def fake_import(name):
        raise ModuleNotFoundError("No module named 'piq'", name="piq")

    monkeypatch.setattr(metrics_pkg.pkgutil, "walk_packages", fake_walk)
    monkeypatch.setattr(metrics_pkg.importlib, "import_module", fake_import)
    metrics_pkg.MISSING_OPTIONAL_DEPS.pop(fake, None)
    try:
        metrics_pkg._discover_and_import_metric_modules()  # must not raise
        assert metrics_pkg.MISSING_OPTIONAL_DEPS[fake] == "piq"
    finally:
        metrics_pkg.MISSING_OPTIONAL_DEPS.pop(fake, None)


# ---------------------------------------------------------------------------
# Sub-PACKAGE failures (2026-08-28). ``walk_packages`` recurses by *importing*
# each sub-package, and that import happens inside pkgutil -- not in the loop
# body -- so the loop's own ``except ImportError`` never sees it. With the
# default ``onerror=None`` pkgutil discards the error and abandons the entire
# sub-tree: exit 0, no warning, and every @register_metric underneath gone.
# Measured before the fix: planting a raise in ``connectivity`` took the
# registry from 211 metrics to 210 while the process still exited 0.
# ---------------------------------------------------------------------------


def test_discovery_walk_is_given_an_onerror_callback() -> None:
    """The wiring itself: a walk without ``onerror`` cannot report a broken
    sub-package at all, so the callback existing is not enough -- the walk has
    to be handed it."""
    import mriforge.core.metrics as metrics_pkg

    captured: dict[str, object] = {}

    def fake_walk(path, prefix, onerror=None):
        captured["onerror"] = onerror
        return iter(())

    original = metrics_pkg.pkgutil.walk_packages
    metrics_pkg.pkgutil.walk_packages = fake_walk  # type: ignore[assignment]
    try:
        metrics_pkg._discover_and_import_metric_modules()
    finally:
        metrics_pkg.pkgutil.walk_packages = original  # type: ignore[assignment]

    assert captured.get("onerror") is metrics_pkg._on_walk_package_error, (
        "pkgutil.walk_packages was called without onerror=_on_walk_package_error. "
        "Its default swallows a sub-package ImportError and silently drops every "
        "metric beneath it."
    )


def test_onerror_reraises_an_in_repo_subpackage_failure() -> None:
    """A broken in-repo sub-package is the poisoning class: loud, like the
    per-module path. pkgutil passes the name only, so the callback reads the
    live exception from the active ``except`` block -- it must therefore be
    exercised from inside one."""
    import mriforge.core.metrics as metrics_pkg

    name = f"{metrics_pkg._PACKAGE_NAME}.connectivity"
    before = dict(metrics_pkg.MISSING_OPTIONAL_DEPS)
    try:
        raise ImportError("broken sub-package", name=name)
    except ImportError:
        with pytest.raises(ImportError):
            metrics_pkg._on_walk_package_error(name)
    assert before == metrics_pkg.MISSING_OPTIONAL_DEPS


def test_onerror_records_a_third_party_subpackage_failure() -> None:
    """A sub-package whose only problem is a missing optional dependency
    degrades exactly as a module does -- skipped, but *recorded*. The
    recording is the point: pre-fix, this case was indistinguishable from
    nothing having gone wrong."""
    import mriforge.core.metrics as metrics_pkg

    name = f"{metrics_pkg._PACKAGE_NAME}.fake_optional_subpackage"
    metrics_pkg.MISSING_OPTIONAL_DEPS.pop(name, None)
    try:
        try:
            raise ModuleNotFoundError("No module named 'piq'", name="piq")
        except ImportError:
            metrics_pkg._on_walk_package_error(name)  # must not raise
        assert metrics_pkg.MISSING_OPTIONAL_DEPS[name] == "piq"
    finally:
        metrics_pkg.MISSING_OPTIONAL_DEPS.pop(name, None)


def test_onerror_does_not_widen_what_is_tolerated() -> None:
    """Setting ``onerror`` routes *every* walk exception to the callback,
    including ones ``onerror=None`` would have propagated. Non-ImportError
    must still propagate, or this fix would silence errors that are loud
    today."""
    import mriforge.core.metrics as metrics_pkg

    try:
        raise RuntimeError("not an import problem")
    except RuntimeError:
        with pytest.raises(RuntimeError):
            metrics_pkg._on_walk_package_error("mriforge.core.metrics.whatever")


def test_a_broken_subpackage_is_silent_without_onerror_and_loud_with_it(
    tmp_path, monkeypatch
) -> None:
    """The defect and the fix, against real ``pkgutil`` rather than a stub.

    A stubbed walk cannot show this: the swallow lives inside pkgutil's own
    recursion, so only a real package tree exercises it.
    """
    import pkgutil

    import mriforge.core.metrics as metrics_pkg

    root = tmp_path / "walkprobe"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sub" / "__init__.py").write_text(
        "raise ImportError('broken', name='mriforge.core.metrics.sub')\n"
    )
    (root / "sub" / "leaf.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    # Default onerror=None: no error, no warning -- the sub-tree simply is not
    # yielded. Nothing downstream can tell this from an empty sub-package.
    seen = [name for _, name, _ in pkgutil.walk_packages([str(root)], "walkprobe.")]
    assert "walkprobe.sub" in seen, "the sub-package is still discovered"
    assert "walkprobe.sub.leaf" not in seen, (
        "expected pkgutil to abandon the sub-tree silently; if this now fails, "
        "the swallow this callback guards against has changed shape"
    )

    with pytest.raises(ImportError):
        list(
            pkgutil.walk_packages(
                [str(root)],
                "walkprobe.",
                onerror=metrics_pkg._on_walk_package_error,
            )
        )
