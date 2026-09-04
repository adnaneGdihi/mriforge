"""Tests for the retired ``spectramr.models.sota_registry`` shim (2026-07-02).

History: this module used to fire ~41 dynamic ``register_model(...)`` calls,
and once clobbered ``bloch_mamba_v2``'s decorator-declared
``ModelCapabilities`` with an empty re-registration (silently disabling the
audit's ``check_data_model_compatibility`` — CLAUDE.md pitfall #9/#10). All
registrations migrated to per-class ``@register_model`` decorators fired by
the ``populate_model_registry()`` pkgutil walk; the shim stays importable
because callers/tests reference it by name.

These tests pin the new reality:

* the shim is importable and ``register_sota_models()`` is a callable no-op
  that adds NOTHING to the registry;
* the roster names now resolve via the walk alone (spot-checked here; the
  full 42-name roster is pinned by ``test_registry_census.py``);
* ``bloch_mamba_v2``'s decorator-declared capabilities survive end-to-end
  (the original clobber regression, now guarded at the decorator itself by
  the capability-downgrade guard in ``models/registry.py``).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def test_shim_register_sota_models_is_a_noop() -> None:
    """Calling the retired entry point must not add or change registrations."""
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    before = dict(MODEL_REGISTRY)

    sota = importlib.import_module("spectramr.models.sota_registry")
    result = sota.register_sota_models()

    assert result is None
    assert dict(MODEL_REGISTRY) == before, (
        "the retired sota_registry shim must not mutate MODEL_REGISTRY"
    )


def test_shim_module_carries_no_dynamic_registrations() -> None:
    """The shim must stay inert — no model classes, no register calls at import."""
    sota = importlib.import_module("spectramr.models.sota_registry")
    public = [n for n in vars(sota) if not n.startswith("_")]
    assert public == ["register_sota_models"], (
        f"sota_registry shim grew unexpected attributes {public}; registrations "
        "belong on the model class as @register_model + the walk list."
    )


def test_sota_names_resolve_via_walk_alone() -> None:
    """Spot-check former sota-registered names now owned by decorators+walk."""
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    from spectramr.models.registry import get_model_class, get_model_mode

    # (name, mode, class name) — one per former sota_registry phase.
    for name, mode, cls_name in (
        ("pa_pcfm", "diffusion", "PAPCFM"),
        ("d2_mamba", "reconstruction", "D2Mamba"),
        ("gen_3dgs", "synthesis", "Gen3DGS"),
        ("cold_diffusion_process", "diffusion", "ColdDiffusion"),
        ("rl_mri", "acquisition", "ActiveScanner"),
        ("neural_ode_recon", "reconstruction", "NeuralODERecon"),
        ("diff_seq", "reconstruction", "DiffSEQ"),
        ("q_fno", "reconstruction", "QFNO"),
    ):
        assert get_model_mode(name) == mode
        assert get_model_class(name).__name__ == cls_name


def test_bloch_mamba_v2_capabilities_survive_migration() -> None:
    """The decorator-declared caps survive the decorator+walk path.

    Original scar: a bare re-registration in the old register_sota_models()
    overwrote the entry with ModelCapabilities() (all None), so
    get_model_capabilities returned None and the audit's
    check_data_model_compatibility was silently skipped.
    """
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    from spectramr.models.registry import get_model_capabilities

    caps = get_model_capabilities("bloch_mamba_v2")
    assert caps is not None, (
        "bloch_mamba_v2 decorator-declared caps lost during the sota migration"
    )
    assert caps.input_domain == "image"
    assert caps.output_domain == "image"
    assert caps.spatial_dims == (2,)
    assert caps.requires_paired_data is True
    assert caps.accepts_complex is False


def test_bloch_mamba_v2_registered_once_with_correct_mode() -> None:
    """The model is still registered and resolvable under reconstruction mode."""
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    from spectramr.models.registry import get_model_class, get_model_mode

    assert get_model_mode("bloch_mamba_v2") == "reconstruction"
    cls = get_model_class("bloch_mamba_v2")
    assert cls.__name__ == "BlochMambaV2"


def test_spec_card_exposes_bloch_mamba_v2_capabilities() -> None:
    """The audit surface (spec_card._derive_model_form) sees non-None caps."""
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    from spectramr.infrastructure.validation.spec_card import _derive_model_form

    config = SimpleNamespace(model=SimpleNamespace(model_type="bloch_mamba_v2"))
    form = _derive_model_form(config)

    assert form["present"] is True
    assert form["registered"] is True
    assert form["model_type"] == "bloch_mamba_v2"
    assert form["capabilities"] is not None
    assert form["capabilities"].spatial_dims == (2,)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
