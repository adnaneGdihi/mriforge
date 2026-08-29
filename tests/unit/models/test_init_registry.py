"""Guard ``populate_model_registry`` idempotency + completeness.

Regression for the CLI cold-start optimization (PR #158). An earlier
revision gated the discovery walk in ``TrainingSettings.from_yaml`` on
``MODEL_REGISTRY`` being empty, on the assumption that a non-empty
registry meant "already fully populated". That assumption is false:
``@register_model`` decorators fire eagerly as model packages are
imported (populating a partial subset), while ~200 models AND all the
compatibility aliases (``pinn``, ``patch_gan``, ``mae``, ...) register
ONLY through ``populate_model_registry``'s ``pkgutil`` walk. So the
emptiness check skipped the walk and dropped those names, making valid
configs fail ``Invalid model_type`` at load time.

The fix moves idempotency into ``populate_model_registry`` itself (a
module-level flag the walk owns), so it runs at most once but always at
least once. These tests assert the walk is complete and idempotent.
"""

from __future__ import annotations

import json
import subprocess
import sys

import mriforge.models.init_registry as init_registry
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY


def test_populate_is_idempotent_via_internal_flag() -> None:
    """Repeat calls are guarded by the module-level flag, not re-walked."""
    populate_model_registry()
    assert init_registry._REGISTRY_POPULATED is True

    keys_after_first = set(MODEL_REGISTRY)
    # A second call must be a no-op (flag short-circuits before the walk)
    # and must not change the registry contents.
    populate_model_registry()
    assert set(MODEL_REGISTRY) == keys_after_first


def test_walk_registers_walk_only_models_and_aliases() -> None:
    """Walk-only models + compat aliases must be present after populate.

    These names are NOT registered by eager ``@register_model`` import
    side-effects — only by the discovery walk. If a future change gates
    the walk on registry-emptiness again, this fails.
    """
    populate_model_registry()

    # Walk-only models (register via pkgutil walk, not eager import).
    for name in ("diff_varnet", "cycle_gan", "swin_ir", "latent_diffusion"):
        assert name in MODEL_REGISTRY, f"walk-only model '{name}' missing"

    # Compatibility aliases installed at the tail of the walk.
    # ``sde_diffusion`` left this list 2026-08-12 (diffusion cleanup, phase 1.3):
    # zero arms declared it and it was an exact dict-entry copy of
    # ``score_based_diffusion``, which is asserted directly instead.
    for alias in ("pinn", "patch_gan", "mae"):
        assert alias in MODEL_REGISTRY, f"compat alias '{alias}' missing"
    assert "score_based_diffusion" in MODEL_REGISTRY


def test_force_reruns_walk() -> None:
    """``force=True`` re-runs the walk even when the flag is set."""
    populate_model_registry()
    assert init_registry._REGISTRY_POPULATED is True

    # force must repopulate without error and keep the full catalogue.
    populate_model_registry(force=True)
    assert "diff_varnet" in MODEL_REGISTRY
    assert "pinn" in MODEL_REGISTRY


# ──────────────────────────────────────────────────────────────────────
# Import-failure surfacing (2026-07-12)
#
# The discovery walk used to swallow every ImportError at DEBUG. A module that
# fails to import registers nothing, so the catalog silently shrinks and a valid
# `model_type` is then rejected as unknown. That is exactly what happened in CI:
# the yaml-audit lane installed no torchvision, `vf_reconstruction_generators`
# (which needs it) never loaded, and the audit rejected `neural_complex_sum` --
# a model that exists and carries @register_model -- as an invalid model_type.
# A dropped module must be loud and must be nameable by the error that it causes.
# ──────────────────────────────────────────────────────────────────────


def test_import_failure_is_recorded_and_warned(monkeypatch, caplog) -> None:
    """A module that will not import is recorded, not swallowed at DEBUG."""
    import importlib
    import logging

    real_import = importlib.import_module
    victim = "mriforge.models.generators.vf_reconstruction_generators"

    def _boom(name: str, *a, **kw):
        if name == victim:
            raise ImportError("No module named 'torchvision'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(init_registry.importlib, "import_module", _boom)
    init_registry._IMPORT_FAILURES.clear()

    with caplog.at_level(logging.WARNING, logger=init_registry.__name__):
        populate_model_registry(force=True)

    failures = init_registry.get_registry_import_failures()
    assert victim in failures, "a failed module import must be recorded, not swallowed"
    assert "torchvision" in failures[victim]
    assert any(victim in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING), (
        "a dropped module must WARN, not whisper at DEBUG"
    )

    # Undo the patch FIRST. monkeypatch tears down after the test body, so a restore
    # done here would still run under `_boom` -- it would re-raise the fake ImportError
    # and re-record the failure it is meant to clear. `_IMPORT_FAILURES` is a module
    # global, so that leaks into every later test: the completeness guard in
    # tests/unit/models/encoders/test_divov3_encoder.py then fails on a failure this
    # test manufactured.
    monkeypatch.undo()
    populate_model_registry(force=True)
    assert not init_registry.get_registry_import_failures()


def test_get_registry_import_failures_returns_a_copy() -> None:
    """Callers cannot mutate the recorded failures out from under the registry."""
    populate_model_registry()
    snapshot = init_registry.get_registry_import_failures()
    snapshot["mriforge.models.bogus"] = "nope"
    assert "mriforge.models.bogus" not in init_registry.get_registry_import_failures()


# ---------------------------------------------------------------------------
# Sub-package failures during model discovery (2026-08-28).
#
# The walk's ``except Exception -> _record_import_failure`` wraps the loop
# body's own ``import_module``. pkgutil imports a SUB-package itself, while
# recursing, so that handler never runs for one -- and pkgutil's default
# discards the error along with the entire sub-tree. Every listed model package
# is flat today, which makes the hole latent rather than absent: the first
# nested package added would drop its models with no error and exit 0.
#
# That the walk is HANDED an onerror is owned repo-wide by
# tests/architecture/test_discovery_walks_report_errors.py. This test owns the
# other half -- that the callback, when pkgutil actually calls it, records the
# loss instead of raising a TypeError in a path no green run ever executes.
# ---------------------------------------------------------------------------


def test_a_broken_subpackage_is_recorded_not_dropped(tmp_path, monkeypatch) -> None:
    import pkgutil

    from mriforge.models import init_registry

    root = tmp_path / "modelwalkprobe"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sub" / "__init__.py").write_text("raise ImportError('broken sub-package')\n")
    (root / "sub" / "leaf.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    # Without onerror pkgutil abandons the sub-tree in silence: no exception,
    # and the leaf simply never appears. Nothing downstream can tell this from
    # an empty sub-package.
    seen = [n for _, n, _ in pkgutil.walk_packages([str(root)], "modelwalkprobe.")]
    assert "modelwalkprobe.sub.leaf" not in seen

    before = init_registry.get_registry_import_failures()
    try:
        list(
            pkgutil.walk_packages(
                [str(root)],
                "modelwalkprobe.",
                onerror=init_registry._on_walk_package_error,
            )
        )
        after = init_registry.get_registry_import_failures()
        new = set(after) - set(before)
        assert new == {"modelwalkprobe.sub"}, (
            "the broken sub-package must land in get_registry_import_failures(), "
            "which is what makes an incomplete model catalog introspectable "
            f"rather than invisible; got {sorted(new)}"
        )
        assert "ImportError" in after["modelwalkprobe.sub"]
    finally:
        init_registry._IMPORT_FAILURES.pop("modelwalkprobe.sub", None)


# ---------------------------------------------------------------------------
# Alias timing: populate owns it, an import cannot promise it
# ---------------------------------------------------------------------------
#
# ``stubs.register_aliases`` used to run as an import-time side effect, and every
# alias in it is guarded by ``if <canonical> in MODEL_REGISTRY``. Run the pass
# before the generators register and the guards simply fail -- silently, with no
# fallback and no warning -- and ``sys.modules`` caching then makes it permanent,
# because ``populate_model_registry``'s import of the module becomes a no-op.
#
# Measured on this tree before the fix, by import order:
#
#     populate alone           586 models
#     ``import stubs`` first   533 models   (53 lost)
#     full walk-import first   585 models   (1 lost)
#
# The 53 include ``standard_unet``, ``vit``, ``cycle_gan``, ``pinn``, ``vq_vae``
# and ``latent_diffusion``, while ``config/validation_constants.py`` went on
# listing all of them as valid -- so an arm naming one passed validation and then
# could not be built. The middle row is not hypothetical either: a full
# walk-import is the release procedure's own verification command.
#
# One subprocess, ~6 s, deliberately NOT marked ``slow``. ``slow`` is opt-in, and
# a regression guard that never runs in the default lane is the facade shape this
# repo calls pitfall #16.

_ORDER_SENSITIVE_ALIASES = (
    "standard_unet",
    "vit",
    "cycle_gan",
    "pinn",
    "vq_vae",
    "latent_diffusion",
    "rl_scanner",
    "swin_ir",
    "world_model",
    "enhanced_unet",
)


def test_aliases_survive_an_early_import_of_the_stubs_module() -> None:
    """A cold process that imports stubs FIRST must still get every alias.

    Cold subprocess, not this process: the suite has already imported half the
    package, so an in-process check cannot reproduce the ordering that breaks.
    """
    code = (
        "import json, warnings; warnings.filterwarnings('ignore')\n"
        "import mriforge.models.stubs\n"  # the adversarial early import
        "from mriforge.models.init_registry import populate_model_registry\n"
        "from mriforge.models.registry import MODEL_REGISTRY\n"
        "populate_model_registry()\n"
        "print(json.dumps(sorted(MODEL_REGISTRY)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"cold populate failed:\n{proc.stdout}\n{proc.stderr}"
    names = set(json.loads(proc.stdout.strip().splitlines()[-1]))

    missing = [a for a in _ORDER_SENSITIVE_ALIASES if a not in names]
    assert not missing, (
        f"{len(missing)} model name(s) vanished because ``mriforge.models.stubs`` "
        f"was imported before populate_model_registry(): {missing}.\n"
        "register_aliases() must be CALLED by populate, not left to fire on "
        "import -- see the comment above and stubs.py's own tail comment."
    )
