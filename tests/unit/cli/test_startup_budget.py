"""Startup import-budget regression guard.

The ``spectramr`` CLI cold start (notably ``--help``) historically pulled
torch, the full SOTA model catalogue, wandb and scipy *before* argparse
even built the parser — ~8.7 s on ``--help``. The fixes (see
``docs/cli_startup_budget.rst``) keep the parser-construction
path torch-free; these tests are the executable guard.

Each check runs in a **clean subprocess** on purpose: pytest sibling
tests routinely import torch, so asserting ``"torch" not in sys.modules``
in-process would be non-deterministic. A fresh interpreter gives a
truthful answer regardless of suite ordering or whether torch is even
installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

#: The import budget, and the ONE owner of it. Every check below interpolates
#: this tuple into its subprocess rather than restating the set: the parser
#: guard used to carry its own inline copy, which meant this constant was dead
#: and widening it changed nothing (non-negotiable 17). ``pandas`` is on the
#: list because the metric-discovery walk pulled it in via ``core/__init__``.
HEAVY_MODULES = ("torch", "timm", "wandb", "scipy", "pandas")


def _run_in_clean_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that can import the package."""
    env = os.environ.copy()
    # Let the child import the exact same package the parent resolved,
    # whether installed (CI) or run from a source checkout.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_build_parser_is_torch_free() -> None:
    result = _run_in_clean_subprocess(
        """
        import sys
        import spectramr.cli.app as app

        app.build_parser()

        heavy = set(HEAVY) & set(sys.modules)
        assert not heavy, f'heavy imports leaked into --help path: {sorted(heavy)}'
        print('OK')
        """.replace("HEAVY", repr(HEAVY_MODULES))
    )
    assert result.returncode == 0, (
        f"build_parser() pulled heavy imports.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_importing_models_does_not_eagerly_load_sota_registry() -> None:
    result = _run_in_clean_subprocess(
        """
        import sys
        import spectramr.models

        assert 'spectramr.models.sota_registry' not in sys.modules, (
            'spectramr.models eagerly imported the SOTA catalogue'
        )
        # The lazy PEP-562 hook must still resolve the submodule on demand.
        assert hasattr(spectramr.models, '__getattr__')
        print('OK')
        """
    )
    assert result.returncode == 0, (
        f"import spectramr.models is no longer lean.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_cli_help_exits_zero_and_lists_subcommands() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")
    result = subprocess.run(
        [sys.executable, "-m", "spectramr.cli", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    # The extra-CLI subcommands must still be attached on the torch-free path.
    for sub in ("train", "audit", "audit-ksd", "design-mrf-sequence"):
        assert sub in result.stdout, f"{sub!r} missing from --help output"


def test_importing_core_does_not_eagerly_load_the_metric_registry() -> None:
    """``spectramr.core`` must not drag the metric-discovery walk in (#1130).

    ``core/metrics/__init__.py`` walks its package and imports torch; the
    parser path only ever wanted ``core.env``. Because importing any submodule
    runs its parent ``__init__`` first, an eager ``from . import metrics`` here
    was charged to every ``spectramr --help``.
    """
    result = _run_in_clean_subprocess(
        """
        import sys
        import spectramr.core

        assert 'spectramr.core.metrics' not in sys.modules, (
            'spectramr.core eagerly imported the metric registry'
        )
        heavy = set(HEAVY) & set(sys.modules)
        assert not heavy, f'import spectramr.core pulled heavy imports: {sorted(heavy)}'
        # The lazy PEP-562 hook must still resolve both exported submodules.
        assert hasattr(spectramr.core, '__getattr__')
        print('OK')
        """.replace("HEAVY", repr(HEAVY_MODULES))
    )
    assert result.returncode == 0, (
        f"import spectramr.core is no longer lean.\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_core_lazy_exports_still_resolve_on_demand() -> None:
    """Laziness must move *who* triggers the walk, never whether it happens.

    Pins the "nothing is lost" half of the change: touching ``core.metrics``
    resolves a real module and the registry still fills, and the submodules
    that are *not* exported keep working through ``from spectramr.core import X``
    (the import system falls back to a submodule import when ``__getattr__``
    raises). Six modules use that form.
    """
    result = _run_in_clean_subprocess(
        """
        import types

        import spectramr.core

        # 1. Attribute access resolves the exported submodules.
        assert isinstance(spectramr.core.env, types.ModuleType)
        assert isinstance(spectramr.core.metrics, types.ModuleType)

        # 2. Touching metrics still runs walk-discovery: the registry fills.
        from spectramr.core.metrics.registry import MetricsRegistry

        assert len(MetricsRegistry._metrics) > 100, (
            f'metric discovery did not run: {len(MetricsRegistry._metrics)} metrics'
        )

        # 3. Non-exported submodules keep resolving through `from ... import`.
        from spectramr.core import compute_device, execution_ledger, resources

        assert isinstance(execution_ledger, types.ModuleType)
        assert isinstance(compute_device, types.ModuleType)
        assert isinstance(resources, types.ModuleType)

        # 4. A genuinely unknown name still raises, never silently resolves.
        #    The message must name the attribute: accepting any AttributeError
        #    cannot tell a correct refusal from an unrelated blow-up inside the
        #    hook, and a planted silent-fallback slipped through on exactly that.
        try:
            spectramr.core.not_a_real_module
        except AttributeError as exc:
            assert 'not_a_real_module' in str(exc), (
                f'AttributeError did not name the attribute: {exc}'
            )
        else:
            raise AssertionError('unknown attribute did not raise AttributeError')
        print('OK')
        """
    )
    assert result.returncode == 0, (
        f"lazy spectramr.core lost a resolution path.\nstdout={result.stdout}\nstderr={result.stderr}"
    )
