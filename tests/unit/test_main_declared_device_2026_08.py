"""``main._declared_device``: the one owner of "what device did the config ask for?".

D01#4. ``run.device`` carries a schema default of ``"cuda"``, so reading the
attribute cannot tell a declaration from a default -- and
``resolve_compute_device`` treats the two differently (an explicit ``"cuda"`` is
a hard requirement ``FORCE_CPU`` does not relax; ``None`` becomes ``"auto"``).
Before this, the train path's third fallback read the LEGACY top-level
``settings.device`` -- a spelling ``RENAMES`` retired with a raise posture -- so
the canonical knob had no consumer on the train path and the inference family
never consulted the config at all.
"""

import argparse
import types

import pytest

from spectramr import main as main_mod
from spectramr.config.settings import TrainingSettings
from spectramr.main import _declared_device

# Module-level ``__``-prefixed name: no mangling applies outside a class body,
# but ``getattr`` states that explicitly rather than relying on it.
_common_train_setup = getattr(main_mod, "__common_train_setup")

_BASE = {"model": {}, "data": {}, "optimization": {}, "logging": {}}


def _settings(**kw) -> TrainingSettings:
    return TrainingSettings(**_BASE, **kw)


def test_declared_device_returns_the_declared_value():
    assert _declared_device(_settings(run={"device": "cpu"})) == "cpu"


def test_declared_cuda_is_reported_as_declared():
    """An explicit ``cuda`` must NOT be confused with the default of the same
    spelling: it is the one value FORCE_CPU may not relax."""
    assert _declared_device(_settings(run={"device": "cuda"})) == "cuda"


def test_undeclared_device_is_none_not_the_schema_default():
    settings = _settings()
    assert settings.run.device == "cuda", "schema default changed; update this test"
    assert _declared_device(settings) is None


def test_a_run_block_without_device_is_still_undeclared():
    """Declaring a sibling key must not make ``device`` look declared."""
    assert _declared_device(_settings(run={"seed": 7})) is None


def test_missing_run_block_is_tolerated():
    class _NoRun:
        pass

    assert _declared_device(_NoRun()) is None


def test_legacy_top_level_device_is_rejected_by_the_schema():
    """The leg this helper replaces could never resolve: the top-level spelling
    raises rather than folding, so ``getattr(settings, "device", None)`` was
    permanently ``None``."""
    with pytest.raises(Exception, match=r"run\.device"):
        TrainingSettings(**_BASE, device="cpu")


# ---------------------------------------------------------------------------
# The CALL SITES, not just the helper.
#
# The planted-violation battery scored the two chain legs GREEN-BLIND: deleting
# ``_declared_device(settings)`` from ``begin_inference_run`` and from
# ``__common_train_setup`` left every test above green, because they all pin the
# helper in isolation. A helper nobody calls is exactly the shape non-negotiable
# 16 is about -- "a capability is not delivered until the production path calls
# it" -- so the fix is not delivered until a test watches the production path
# consume it. These four do.
# ---------------------------------------------------------------------------

def _stub_config_load(monkeypatch, settings):
    """Make ``TrainingSettings.from_yaml`` yield ``settings`` and mute the ledger."""
    from spectramr.core.execution_ledger import ExecutionLedger

    monkeypatch.setattr(
        main_mod.TrainingSettings, "from_yaml", classmethod(lambda _cls, _p: settings)
    )
    monkeypatch.setattr(ExecutionLedger, "begin_run", classmethod(lambda _cls, **_k: None))


def _capture_accelerator(monkeypatch):
    seen: dict = {}

    def _fake(device, seed, *, deterministic, pipeline):
        seen.update(
            device=device, seed=seed, deterministic=deterministic, pipeline=pipeline
        )
        return "torch.device(cpu)"

    monkeypatch.setattr(main_mod, "initialize_accelerator", _fake)
    return seen


def _train_args(**kw) -> argparse.Namespace:
    base = {
        "config": "unused.yaml",
        "seed": None,
        "device": None,
        "override": None,
        "dry_run": False,
        "resume": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_begin_inference_run_forwards_the_declared_device(monkeypatch):
    """``infer``/``predict`` must consult the config, not only ``--device``.

    This is the regression the row exists for: before it, the inference family
    never read the YAML at all, so the CPU opt-out that
    ``resolve_compute_device``'s own error message advertises was unreachable
    from those verbs.
    """
    settings = _settings(run={"device": "cpu"})
    _stub_config_load(monkeypatch, settings)
    seen = _capture_accelerator(monkeypatch)

    main_mod.begin_inference_run("unused.yaml", None, pipeline="infer")

    assert seen["device"] == "cpu", "the declared device never reached the accelerator"


def test_cli_device_still_outranks_the_declared_one(monkeypatch):
    settings = _settings(run={"device": "cpu"})
    _stub_config_load(monkeypatch, settings)
    seen = _capture_accelerator(monkeypatch)

    main_mod.begin_inference_run("unused.yaml", "cuda", pipeline="infer")

    assert seen["device"] == "cuda"


def test_train_chain_third_leg_reads_the_declared_device(monkeypatch):
    """CLI > ``training.device`` > declared ``run.device``.

    With the first two legs empty, the third must produce the declared value --
    and ``None`` (not the schema default ``"cuda"``) when nothing was declared,
    or ``FORCE_CPU`` becomes inert for every undeclared arm.
    """
    settings = _settings(run={"device": "cpu"})
    _stub_config_load(monkeypatch, settings)
    seen = _capture_accelerator(monkeypatch)
    monkeypatch.setattr(
        "spectramr.bootstrap.build_container", lambda *_a, **_k: types.SimpleNamespace()
    )

    _common_train_setup(_train_args(dry_run=True))

    assert seen["device"] == "cpu", "run.device has no consumer on the train path"


def test_train_chain_is_none_when_nothing_is_declared(monkeypatch):
    settings = _settings()
    _stub_config_load(monkeypatch, settings)
    seen = _capture_accelerator(monkeypatch)
    monkeypatch.setattr(
        "spectramr.bootstrap.build_container", lambda *_a, **_k: types.SimpleNamespace()
    )

    _common_train_setup(_train_args(dry_run=True))

    assert seen["device"] is None, "the schema default leaked in as a declaration"


def test_dry_run_validates_the_device_the_live_run_would_use(monkeypatch):
    """A dry run that builds the container on a different device green-lights a
    config the live path would reject."""
    settings = _settings(run={"device": "cpu"})
    _stub_config_load(monkeypatch, settings)
    _capture_accelerator(monkeypatch)
    seen: dict = {}
    monkeypatch.setattr(
        "spectramr.bootstrap.build_container",
        lambda _cfg, device=None, pipeline=None: seen.update(
            device=device, pipeline=pipeline
        ),
    )

    _common_train_setup(_train_args(dry_run=True))

    assert seen["device"] == "cpu"
    assert seen["pipeline"] == "train"


def test_live_train_run_forwards_the_resolved_device_to_the_pipeline(monkeypatch):
    """``run_training_pipeline`` re-resolves through ``build_container``; handing
    it the raw CLI string makes the accelerator and the container two independent
    owners of one question (non-negotiable 17)."""
    settings = _settings(run={"device": "cpu"})
    _stub_config_load(monkeypatch, settings)
    _capture_accelerator(monkeypatch)
    seen: dict = {}

    def _fake_pipeline(_cfg, device=None, **_kw):
        seen["device"] = device
        return {"success": True}

    monkeypatch.setattr("spectramr.pipelines.run_training_pipeline", _fake_pipeline)

    _common_train_setup(_train_args())

    assert seen["device"] == "cpu", "the live pipeline got the raw --device instead"
