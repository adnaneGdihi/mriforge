"""Tests for :mod:`spectramr.pipelines`'s PEP 562 lazy export surface.

``pipelines/__init__`` used to eagerly ``from .infer import
run_inference_pipeline``, which made *every* import under the package -- notably
``spectramr.pipelines.fit``, the one ``spectramr.api`` needs -- drag in
``infer -> inference_factory -> cold_diffusion_inference_strategy ->
models.diffusion.kspace_process``.

That is fatal only in one window, which is why nothing caught it: plugin
discovery runs at module-import time, so an out-of-tree plugin importing
``spectramr.api`` did so while ``spectramr`` was still initialising and hit a
partially-initialised module. These tests pin the two properties that keep the
window closed -- every name still resolves, and the heavy submodules are not
pulled in to do it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import spectramr.pipelines as pipelines


def test_every_declared_export_resolves():
    """``__all__`` is a promise: each name must actually resolve."""
    for name in pipelines.__all__:
        assert getattr(pipelines, name) is not None, f"{name} did not resolve"


def test_all_matches_the_lazy_table():
    """``__all__`` and ``_LAZY_EXPORTS`` must not drift apart."""
    assert set(pipelines.__all__) == set(pipelines._LAZY_EXPORTS)


def test_dir_lists_the_lazy_names():
    """``dir()`` must advertise the lazily-resolved names (REPL/tab-completion)."""
    assert set(pipelines.__all__) <= set(dir(pipelines))


def test_unknown_attribute_raises_attribute_error():
    """An unknown name raises rather than resolving to something (NN3)."""
    try:
        _ = pipelines.definitely_not_an_export
    except AttributeError as exc:
        assert "definitely_not_an_export" in str(exc)
    else:  # pragma: no cover - the assertion above is the point
        raise AssertionError("expected AttributeError for an unknown export")


def test_importing_the_package_does_not_pull_the_inference_chain():
    """The regression guard: ``import spectramr.pipelines`` must stay light.

    Subprocess, because ``sys.modules`` is process-global and the rest of the
    suite will already have imported ``infer`` for its own reasons.
    """
    probe = textwrap.dedent(
        """
        import sys
        import spectramr.pipelines  # noqa: F401
        heavy = [m for m in ("spectramr.pipelines.infer",
                             "spectramr.infrastructure.inference") if m in sys.modules]
        print("HEAVY:", heavy)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"},
        timeout=600,
    )
    assert "HEAVY: []" in out.stdout, (
        "importing spectramr.pipelines eagerly pulled the inference chain again — "
        f"the plugin-import window is open.\nstdout={out.stdout[-2000:]}\n"
        f"stderr={out.stderr[-2000:]}"
    )


def test_fit_import_does_not_pull_the_inference_chain():
    """``spectramr.pipelines.fit`` is what ``spectramr.api`` resolves; keep it light."""
    probe = textwrap.dedent(
        """
        import sys
        import spectramr.pipelines.fit  # noqa: F401
        print("INFER_LOADED:", "spectramr.pipelines.infer" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"},
        timeout=600,
    )
    assert "INFER_LOADED: False" in out.stdout, (
        f"stdout={out.stdout[-2000:]}\nstderr={out.stderr[-2000:]}"
    )
