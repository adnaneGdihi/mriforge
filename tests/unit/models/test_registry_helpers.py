"""Unit tests for the registry helpers in :mod:`spectramr.models.registry`.

The integrity / contract tests in
:mod:`tests.unit.models.test_registry_integrity` exercise the full
production registry. This file instead targets the *helper* functions in
isolation:

* ``register_model`` decorator semantics (success + overwrite-rejection)
* ``get_model_class`` / ``get_model_mode`` error messages
* ``model_supports`` returning False on unknown models / capabilities
* ``get_model_capabilities`` distinguishing "unannotated" (returns None)
  from "annotated as not-supported" (returns the dataclass)
* ``list_models_with_capability``

We back the registry up around each test so we don't poison the
production singleton.
"""

from __future__ import annotations

import pytest

import spectramr.models.registry as reg_mod
from spectramr.models.capabilities import ModelCapabilities
from spectramr.models.registry import (
    MODEL_REGISTRY,
    get_model_capabilities,
    get_model_class,
    get_model_mode,
    list_models,
    list_models_with_capability,
    model_supports,
    register_model,
)


# ── Fixture: clean registry per-test ───────────────────────────────


@pytest.fixture
def clean_registry() -> dict:
    """Snapshot MODEL_REGISTRY, clear, then restore — keeps the
    production registry untouched after every test in this module.

    Triggers auto-discovery before snapshotting so the backup actually
    contains the production model set. Without this, if a test in this
    module is the first to access ``MODEL_REGISTRY`` (test ordering is
    not deterministic), the backup captures an empty dict and the
    "restore" leaves the registry empty for the rest of the session —
    poisoning every later test that depends on auto-discovered models
    (e.g. ``test_registry_contract.py``, ~1500 parametrized tests).
    """
    # Trigger auto-discovery side-effects exactly once before backing
    # up. The import is a no-op on subsequent calls thanks to
    # ``sys.modules`` caching, so this only pays the discovery cost
    # on the first invocation.
    import spectramr.models.init_registry  # noqa: F401
    backup = dict(MODEL_REGISTRY)
    MODEL_REGISTRY.clear()
    try:
        yield MODEL_REGISTRY
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(backup)


def _new_cls(name: str = "_FakeModel") -> type:
    """Build a unique, throwaway class for registration tests."""
    return type(name, (), {})


# ── register_model success path ────────────────────────────────────


class TestRegisterModelSuccess:
    def test_registration_stores_class_mode_and_capabilities(
        self, clean_registry: dict
    ) -> None:
        cls = _new_cls()
        decorated = register_model("alpha", training_mode="gan")(cls)
        assert decorated is cls  # decorator returns the class unchanged
        assert "alpha" in clean_registry
        entry = clean_registry["alpha"]
        assert entry["class"] is cls
        assert entry["mode"] == "gan"
        assert entry["supports_contrast_conditioning"] is False
        # Capabilities dataclass is always present, default-empty here.
        assert isinstance(entry["capabilities"], ModelCapabilities)

    def test_capability_flags_round_trip_through_register(
        self, clean_registry: dict
    ) -> None:
        cls = _new_cls()
        register_model(
            "beta",
            training_mode="diffusion",
            supports_contrast_conditioning=True,
            spatial_dims=(2,),
            input_domain="kspace",
            output_domain="image",
            accepts_complex=True,
            expects_real_imag_interleaved=False,
            requires_paired_data=True,
        )(cls)
        entry = clean_registry["beta"]
        assert entry["supports_contrast_conditioning"] is True
        caps = entry["capabilities"]
        assert caps.spatial_dims == (2,)
        assert caps.input_domain == "kspace"
        assert caps.output_domain == "image"
        assert caps.accepts_complex is True
        assert caps.expects_real_imag_interleaved is False
        assert caps.requires_paired_data is True

    def test_same_class_re_registration_is_idempotent(
        self, clean_registry: dict
    ) -> None:
        """Re-decorating the *same* class is allowed — common in test
        reloads / import cycles."""
        cls = _new_cls()
        register_model("gamma", training_mode="gan")(cls)
        # Decorate again — should not raise.
        register_model("gamma", training_mode="gan")(cls)
        assert clean_registry["gamma"]["class"] is cls


# ── register_model duplicate rejection ─────────────────────────────


class TestRegisterModelDuplicateRejection:
    def test_overwriting_with_different_class_raises(
        self, clean_registry: dict
    ) -> None:
        cls_a = _new_cls("ModelA")
        cls_b = _new_cls("ModelB")
        register_model("dup", training_mode="gan")(cls_a)
        with pytest.raises(ValueError) as ei:
            register_model("dup", training_mode="diffusion")(cls_b)
        msg = str(ei.value)
        # Error names both classes and modes for debuggability.
        assert "ModelA" in msg
        assert "ModelB" in msg
        # And explicitly says it's refusing.
        assert "refusing to overwrite" in msg


# ── get_model_class / get_model_mode ───────────────────────────────


class TestLookups:
    def test_get_model_class_returns_class(self, clean_registry: dict) -> None:
        cls = _new_cls()
        register_model("alpha", training_mode="gan")(cls)
        assert get_model_class("alpha") is cls

    def test_get_model_class_unknown_lists_available(
        self, clean_registry: dict
    ) -> None:
        register_model("alpha", training_mode="gan")(_new_cls())
        with pytest.raises(ValueError) as ei:
            get_model_class("does_not_exist")
        msg = str(ei.value)
        assert "does_not_exist" in msg
        assert "alpha" in msg

    def test_get_model_mode(self, clean_registry: dict) -> None:
        register_model("alpha", training_mode="reconstruction")(_new_cls())
        assert get_model_mode("alpha") == "reconstruction"

    def test_get_model_mode_unknown_raises(self, clean_registry: dict) -> None:
        with pytest.raises(ValueError):
            get_model_mode("does_not_exist")


# ── model_supports / list_models_with_capability ───────────────────


class TestCapabilityFlags:
    def test_model_supports_returns_true_for_registered_flag(
        self, clean_registry: dict
    ) -> None:
        register_model(
            "supports_cc",
            training_mode="gan",
            supports_contrast_conditioning=True,
        )(_new_cls("SupportsCC"))
        assert model_supports("supports_cc", "supports_contrast_conditioning") is True

    def test_model_supports_returns_false_for_disabled_flag(
        self, clean_registry: dict
    ) -> None:
        register_model(
            "no_cc",
            training_mode="gan",
            supports_contrast_conditioning=False,
        )(_new_cls("NoCC"))
        assert model_supports("no_cc", "supports_contrast_conditioning") is False

    def test_model_supports_unknown_model_returns_false(
        self, clean_registry: dict
    ) -> None:
        # Per the docstring, unknown models silently return False — this
        # is the *only* place such silent behaviour is allowed because
        # the audit layer is responsible for the loud failure.
        assert model_supports("ghost", "anything") is False

    def test_model_supports_unknown_capability_returns_false(
        self, clean_registry: dict
    ) -> None:
        register_model("alpha", training_mode="gan")(_new_cls())
        # Unknown flag → False (forward-compatible).
        assert model_supports("alpha", "supports_quantum_field_theory") is False

    def test_list_models_with_capability_filters(self, clean_registry: dict) -> None:
        register_model(
            "a", training_mode="gan", supports_contrast_conditioning=True
        )(_new_cls("A"))
        register_model(
            "b", training_mode="gan", supports_contrast_conditioning=False
        )(_new_cls("B"))
        register_model(
            "c", training_mode="gan", supports_contrast_conditioning=True
        )(_new_cls("C"))

        out = list_models_with_capability("supports_contrast_conditioning")
        assert set(out) == {"a", "c"}


# ── get_model_capabilities ─────────────────────────────────────────


class TestGetModelCapabilities:
    def test_unannotated_returns_none(self, clean_registry: dict) -> None:
        """A model registered with no capability hints is treated as
        unannotated — get_model_capabilities returns None so the audit
        ladder can opt out gracefully."""
        register_model("unannotated", training_mode="gan")(_new_cls())
        assert get_model_capabilities("unannotated") is None

    def test_annotated_returns_dataclass(self, clean_registry: dict) -> None:
        register_model(
            "annotated",
            training_mode="gan",
            spatial_dims=(2,),
            input_domain="image",
            output_domain="image",
            accepts_complex=False,
        )(_new_cls())
        caps = get_model_capabilities("annotated")
        assert caps is not None
        assert caps.spatial_dims == (2,)
        assert caps.input_domain == "image"

    def test_unknown_model_returns_none(self, clean_registry: dict) -> None:
        assert get_model_capabilities("ghost") is None


# ── list_models ────────────────────────────────────────────────────


class TestListModels:
    def test_list_models_is_a_copy(self, clean_registry: dict) -> None:
        register_model("alpha", training_mode="gan")(_new_cls())
        snapshot = list_models()
        # Mutating the snapshot must not mutate the registry.
        snapshot.clear()
        assert "alpha" in MODEL_REGISTRY

    def test_list_models_returns_all_entries(self, clean_registry: dict) -> None:
        register_model("a", training_mode="gan")(_new_cls("A"))
        register_model("b", training_mode="diffusion")(_new_cls("B"))
        out = list_models()
        assert set(out) == {"a", "b"}
        assert out["a"]["mode"] == "gan"
        assert out["b"]["mode"] == "diffusion"


# ── Sanity: module-level constants ─────────────────────────────────


def test_module_exports_required_names() -> None:
    """The module's public surface includes the symbols the audit
    layer imports — keep this guard in place so refactors that rename
    a helper get caught immediately."""
    for name in (
        "MODEL_REGISTRY",
        "register_model",
        "get_model_class",
        "get_model_mode",
        "list_models",
        "model_supports",
        "get_model_capabilities",
        "list_models_with_capability",
    ):
        assert hasattr(reg_mod, name), f"Missing public name: {name}"
