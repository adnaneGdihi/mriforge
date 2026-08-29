"""A package imported at module top level must be a CORE dependency (#941).

An optional extra is a promise that the package works without it. A module-level
``import`` breaks that promise for every module that transitively reaches the
importing one — the failure is an ``ImportError`` at load, not a graceful
degradation at the point of use.

``torchvision`` was the case that surfaced this. It sat in ``viz`` while three
modules imported it at top level::

    models/losses/perceptual_loss.py       from torchvision import models
    models/losses/gan_loss_library.py      from torchvision import models
    infrastructure/training/strategies/diffusion.py
                                           from torchvision.utils import save_image

and ``populate_model_registry()`` — which CLAUDE.md makes mandatory and
load-bearing — imports all three. So a core install could not build its own
registry. It went unnoticed because ``.[dev]`` resolves to ``.[all]``, which
includes ``viz``: every developer and CI environment had it incidentally.

All three sites above were deferred into their call sites by #963 (merged
four minutes after this file, independently, as the OTHER valid fix for the
same issue). torchvision stayed core anyway, per the owner's ruling in #971
-- see ``test_registry_populate_path_still_needs_a_working_torchvision``
below for why that is still honest, and #972 for the collision this caused.

**This is a ratchet, not a clean gate.** Five packages are still in this state
and each needs its own decision — promote to core, or defer the import into the
function that needs it, as every ``torchvision`` site now does (all 19,
including the three above). Tracked separately; the job here is that the
list does not grow.
"""

import ast
import os
import pathlib
import re
import subprocess
import sys
import tomllib

# Import root -> distribution name, for the ones that differ.
_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "dateutil": "python-dateutil",
    "mpl_toolkits": "matplotlib",
}

# Declared only in an extra, yet imported at module top level. Measured
# 2026-08-12. REMOVE entries as they are fixed; do not add. A new entry means a
# core install just lost the ability to import that code path.
_KNOWN = {
    "matplotlib",  # 54 sites, `viz`
    "nibabel",  # 2 sites,  `mri`
    "optuna",  # 7 sites,  `hpo`
    "seaborn",  # 3 sites,  `viz`
    "torchio",  # 48 sites, `mri` -- includes data/builders/dataset_instantiator.py
}

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _normalise(spec: str) -> str:
    return re.split(r"[<>=!\[;\s]", spec.strip())[0].lower().replace("_", "-")


def _declared():
    with (_REPO / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]
    core = {_normalise(x) for x in project["dependencies"]}
    extras: dict[str, set[str]] = {}
    for extra, specs in project["optional-dependencies"].items():
        for spec in specs:
            name = _normalise(spec)
            if name.startswith("mriforge"):  # self-referential extra
                continue
            extras.setdefault(name, set()).add(extra)
    return core, extras


def _top_level_imports():
    """Module-scope imports only — a deferred import inside a function is fine."""
    found: dict[str, set[str]] = {}
    for path in (_REPO / "src" / "mriforge").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:  # tree.body == module scope
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.ImportFrom) and (node.level or 0):
                continue  # relative import
            module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if not module:
                continue
            root = module.split(".")[0]
            if root in sys.stdlib_module_names or root == "mriforge":
                continue
            dist = _DISTRIBUTION.get(root, root.lower().replace("_", "-"))
            found.setdefault(dist, set()).add(
                f"{path.relative_to(_REPO)}:{node.lineno}"
            )
    return found


def test_no_new_extras_only_top_level_imports():
    """The ratchet: the extras-only set must not grow."""
    core, extras = _declared()
    offenders = {
        dist: sites
        for dist, sites in _top_level_imports().items()
        if dist not in core and dist in extras
    }

    new = set(offenders) - _KNOWN
    assert not new, "\n".join(
        [
            "Imported at module top level but declared ONLY in an optional extra:",
            "",
            *(
                f"  {d} (extras={sorted(extras[d])}) — {len(offenders[d])} site(s), "
                f"e.g. {sorted(offenders[d])[0]}"
                for d in sorted(new)
            ),
            "",
            "Either promote it to [project.dependencies], or defer the import",
            "into the function that needs it. An extra that is imported at module",
            "scope is not optional.",
        ]
    )


def test_the_known_list_has_no_stale_entries():
    """Fixing one must shrink the list, or the ratchet stops ratcheting."""
    core, extras = _declared()
    offenders = {
        dist
        for dist, _ in _top_level_imports().items()
        if dist not in core and dist in extras
    }
    stale = _KNOWN - offenders
    assert not stale, (
        f"{sorted(stale)} no longer violate — remove them from _KNOWN so the "
        "ratchet keeps its grip"
    )


def test_torchvision_is_core_because_the_registry_imports_it():
    """The case that produced this file (#941) -- disposition, not mechanism.

    This only asserts the *decision* (torchvision stays in
    ``[project.dependencies]``). The mechanism that justifies it changed
    after this file was written and is re-verified by
    ``test_registry_populate_path_still_needs_a_working_torchvision`` below
    -- see that test's docstring for the current, checked reason. Do not
    restate the old "three files import it at top level" claim here; it is
    false as of #963/#971 (see #972).
    """
    core, _ = _declared()
    assert "torchvision" in core, (
        "torchvision must stay a core dependency -- see "
        "test_registry_populate_path_still_needs_a_working_torchvision for "
        "the current, verified reason (#972)"
    )


def test_registry_populate_path_still_needs_a_working_torchvision():
    """Guard the guard (#972): the ORIGINAL claim here was wrong and taught a
    false lesson, so this replacement is deliberately narrower than it looks.

    History: this file originally asserted that perceptual_loss.py,
    gan_loss_library.py and diffusion.py import torchvision at module scope,
    and that is why torchvision has to be core. #963 (merged four minutes
    after #968 wrote that claim) deferred all three imports into their call
    sites -- confirmed here: zero module-scope torchvision imports remain
    anywhere in ``src/mriforge`` (see ``test_no_new_extras_only_top_level_imports``
    and this test's own site list below). That half of the original
    justification is simply gone.

    What is verified, current, and load-bearing instead:
    ``src/mriforge/core/metrics/evaluation_metrics.py`` -- the metrics SSOT,
    unconditionally reached by ``populate_model_registry()`` via
    ``mriforge.core`` (imported by nearly everything, very early) --
    does ``import torchvision`` at module scope, guarded only by
    ``except ImportError``. Two verified, DIFFERENT outcomes follow:

    * torchvision genuinely absent (not installed) -> ``ModuleNotFoundError``
      is an ``ImportError`` subclass -> caught cleanly, ``torchvision = None``,
      and ``populate_model_registry()`` SUCCEEDS (measured: all 590 models
      register). Torchvision is NOT unconditionally required by the code.
    * torchvision present but ABI-mismatched with torch (the failure
      pyproject.toml documents: ``RuntimeError: operator torchvision::nms
      does not exist``) -> NOT an ``ImportError``, so the guard does not
      catch it, and ``populate_model_registry()`` crashes hard.

    So: the code does NOT strictly require torchvision -- a clean install
    without it works today. Keeping it core is a policy decision (#941/#971),
    not a code necessity, and the honest arguments for that decision are
    softer than "the registry needs it":

    * The features that use torchvision -- the perceptual/VGG loss family,
      FID/Inception in this metrics SSOT -- are ordinary training/eval
      components reached by common arms, not a plotting extra; declaring
      that as core is a more honest signal than ``viz`` ever was.
    * Core means torch and torchvision resolve in ONE install transaction.
      ``[tool.uv.sources]`` pins both to the same index regardless of core
      vs. extra, and pip honours neither -- so core does not itself
      guarantee ABI compatibility. But an extra reopens an installation
      order where torchvision is added in a SEPARATE transaction against an
      already-installed torch (e.g. ``pip install mriforge`` now, ``pip
      install mriforge[viz]`` later) -- exactly the two-step path that
      produced the mismatch pyproject.toml documents. Core reduces the
      chance of landing there; it does not eliminate it.

    Neither argument is "populate_model_registry() cannot run without it" --
    that claim is false today, empirically. What this test actually guards:
    if evaluation_metrics.py's guard is ever widened to also catch
    ``RuntimeError`` (deferring gracefully, the way
    ``pretrained_vae_adapter.py`` already does for ``diffusers``), or if this
    site stops being reachable from ``populate_model_registry()``, the
    present-but-broken crash goes away too, and this test will fail --
    which means the core-vs-extra question needs re-asking, not that the
    test is broken.
    """
    # Half 1 (static): the three original sites really are all deferred now
    # -- the failure mode #972 was filed for, and the reason the old test's
    # literal claim can never be un-broken by editing it back.
    sites = _top_level_imports().get("torchvision", set())
    original_three = {
        "src/mriforge/models/losses/perceptual_loss.py",
        "src/mriforge/models/losses/gan_loss_library.py",
        "src/mriforge/infrastructure/training/strategies/diffusion.py",
    }
    files = {s.rsplit(":", 1)[0] for s in sites}
    assert not (original_three & files), (
        "one of the three originally-top-level torchvision importers is "
        "top-level again -- the mechanism this test now documents (evaluation_"
        "metrics.py) may no longer be the only reason torchvision is core; "
        "re-read this test's docstring and update it rather than reverting"
    )

    # Half 1b (static): the real site's guard is still the narrow one that
    # makes it load-bearing. If this ever widens to catch RuntimeError too,
    # the dynamic half below will also start passing "clean" -- both must
    # keep agreeing.
    src = (
        _REPO / "src/mriforge/core/metrics/evaluation_metrics.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    torchvision_try = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(n, ast.Import) and n.names[0].name == "torchvision"
                for n in node.body
            )
        ),
        None,
    )
    assert torchvision_try is not None, (
        "core/metrics/evaluation_metrics.py no longer imports torchvision at "
        "module scope at all -- the mechanism this test documents is gone; "
        "re-derive whether torchvision still needs to be core (#972)"
    )
    def _handled_names(expr: ast.expr | None) -> set[str]:
        if isinstance(expr, ast.Name):
            return {expr.id}
        if isinstance(expr, ast.Tuple):
            return {n.id for n in expr.elts if isinstance(n, ast.Name)}
        return set()

    handled: set[str] = set()
    for h in torchvision_try.handlers:
        handled |= _handled_names(h.type)
    assert handled == {"ImportError"}, (
        f"evaluation_metrics.py's torchvision import now handles {sorted(handled)}, "
        "not just ImportError -- if RuntimeError is now caught too, the "
        "present-but-broken crash this test relies on no longer happens; "
        "re-derive whether torchvision still needs to be core (#972)"
    )

    # Half 2 (dynamic): the two scenarios the docstring claims, run for real
    # in subprocesses so a poisoned sys.modules cannot leak into other tests.
    absent = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torchvision'] = None\n"
            "import mriforge.core.metrics.evaluation_metrics as m\n"
            "assert m.torchvision is None\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
    )
    assert absent.returncode == 0, (
        "importing evaluation_metrics.py with torchvision genuinely absent "
        f"no longer succeeds cleanly:\n{absent.stderr[-2000:]}"
    )

    broken = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, builtins\n"
            "real_import = builtins.__import__\n"
            "def _blocked(name, *a, **k):\n"
            "    if name == 'torchvision' or name.startswith('torchvision.'):\n"
            "        raise RuntimeError('operator torchvision::nms does not exist')\n"
            "    return real_import(name, *a, **k)\n"
            "builtins.__import__ = _blocked\n"
            "import mriforge.core.metrics.evaluation_metrics\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
    )
    assert broken.returncode != 0 and "RuntimeError" in broken.stderr, (
        "importing evaluation_metrics.py with torchvision present-but-ABI-"
        "broken no longer crashes -- the mechanism that currently justifies "
        f"keeping torchvision core may have changed (#972):\n{broken.stderr[-2000:]}"
    )
