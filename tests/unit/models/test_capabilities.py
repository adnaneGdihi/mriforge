"""Unit tests for the capability metadata system.

Covers:
- ModelCapabilities / LossCapabilities / StrategyCapabilities / AdapterCapabilities
  dataclass round-trip and frozen-ness.
- register_model(...) capability kwargs reach the registry.
- get_model_capabilities() distinguishes "unannotated" from "annotated as None".
- register_adapter(...) validation of insertion_points.
- Adapter chain composition + form matching.

Locked to the design at
docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md.
"""

from __future__ import annotations

import pytest

from mriforge.data.adapters.registry import (
    ADAPTER_REGISTRY,
    get_adapter_capabilities,
    register_adapter,
)
from mriforge.infrastructure.validation.adapter_composition import (
    candidate_adapters_for,
    compose_chain,
    forms_match,
)
from mriforge.models.capabilities import (
    AdapterCapabilities,
    LossCapabilities,
    ModelCapabilities,
    StrategyCapabilities,
    VALID_INSERTION_POINTS,
)
from mriforge.models.registry import (
    MODEL_REGISTRY,
    get_model_capabilities,
    register_model,
)

# ---------------------------------------------------------------------------
# Capability dataclasses
# ---------------------------------------------------------------------------


class TestCapabilitiesDataclasses:
    def test_model_capabilities_default_all_none(self):
        caps = ModelCapabilities()
        for f in caps.__dataclass_fields__:
            assert getattr(caps, f) is None

    def test_model_capabilities_frozen(self):
        caps = ModelCapabilities(spatial_dims=(2,))
        with pytest.raises(Exception):
            caps.spatial_dims = (3,)  # type: ignore[misc]

    def test_loss_capabilities_default_to_unannotated(self):
        """Every declarative field defaults to "nothing declared".

        ``domain_agnostic`` is the one non-``None`` field and defaults to
        ``False``, which is the same statement in bool form: no claim has been
        made. It is a flag rather than a ``Domain`` member because "the loss
        does not care" is a different kind of statement from "the tensor is a
        k-space", and only the second belongs in the domain vocabulary.
        """
        caps = LossCapabilities()
        assert caps.domain_agnostic is False
        for f in caps.__dataclass_fields__:
            if f == "domain_agnostic":
                continue
            assert getattr(caps, f) is None

    def test_strategy_capabilities_defaults(self):
        caps = StrategyCapabilities()
        assert caps.required_config_fields == ()
        assert caps.supported_paradigms == frozenset()

    def test_model_capabilities_field_units_and_trajectory_param(self):
        # The two field/trajectory metadata fields the VF field-domain metrics
        # read to enforce the parametrization guard (pitfall #16).
        caps = ModelCapabilities(
            output_field_units="Hz",
            trajectory_parametrization="spiral",
        )
        assert caps.output_field_units == "Hz"
        assert caps.trajectory_parametrization == "spiral"

    def test_model_capabilities_new_fields_default_none(self):
        caps = ModelCapabilities()
        assert caps.output_field_units is None
        assert caps.trajectory_parametrization is None

    def test_adapter_capabilities_holds_side_effects_tuple(self):
        caps = AdapterCapabilities(
            bridges_from={"spatial_dims": 3},
            bridges_to={"spatial_dims": 2},
            insertion_points=("pre_model",),
            side_effects=("a", "b"),
        )
        assert caps.side_effects == ("a", "b")
        assert "pre_model" in caps.insertion_points


# ---------------------------------------------------------------------------
# register_model capability kwargs
# ---------------------------------------------------------------------------


class TestRegisterModelCapabilities:
    def test_legacy_decorator_unannotated(self):
        @register_model(name="__test_legacy_model", training_mode="gan")
        class _Legacy:
            pass

        assert get_model_capabilities("__test_legacy_model") is None
        # Cleanup
        MODEL_REGISTRY.pop("__test_legacy_model", None)

    def test_annotated_decorator_round_trip(self):
        @register_model(
            name="__test_annotated_model",
            training_mode="gan",
            spatial_dims=(2, 3),
            input_domain="image",
            output_domain="kspace",
            accepts_complex=False,
            expects_real_imag_interleaved=True,
            requires_paired_data=True,
        )
        class _Annotated:
            pass

        caps = get_model_capabilities("__test_annotated_model")
        assert caps is not None
        assert caps.spatial_dims == (2, 3)
        assert caps.input_domain == "image"
        assert caps.output_domain == "kspace"
        assert caps.accepts_complex is False
        assert caps.expects_real_imag_interleaved is True
        assert caps.requires_paired_data is True
        MODEL_REGISTRY.pop("__test_annotated_model", None)

    def test_field_units_and_trajectory_param_round_trip(self):
        @register_model(
            name="__test_field_units",
            training_mode="multi_acquisition",
            input_domain="image",
            output_domain="image",
            output_field_units="Hz",
            trajectory_parametrization="spiral",
        )
        class _FieldModel:
            pass

        caps = get_model_capabilities("__test_field_units")
        assert caps is not None
        assert caps.output_field_units == "Hz"
        assert caps.trajectory_parametrization == "spiral"
        MODEL_REGISTRY.pop("__test_field_units", None)

    def test_partially_annotated_returns_caps(self):
        # Even one non-None field counts as "annotated".
        @register_model(
            name="__test_partial",
            training_mode="diffusion",
            spatial_dims=(2,),
        )
        class _Partial:
            pass

        caps = get_model_capabilities("__test_partial")
        assert caps is not None
        assert caps.spatial_dims == (2,)
        assert caps.input_domain is None
        MODEL_REGISTRY.pop("__test_partial", None)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


class TestRegisterAdapter:
    def test_round_trip(self):
        @register_adapter(
            name="__test_adapter_a",
            bridges_from={"domain": "image"},
            bridges_to={"domain": "kspace"},
            insertion_points=("pre_model",),
            invertible=True,
            side_effects=("loses_phase",),
        )
        class _A:
            def __call__(self, x):
                return x

        caps = get_adapter_capabilities("__test_adapter_a")
        assert caps is not None
        assert caps.bridges_from == {"domain": "image"}
        assert caps.bridges_to == {"domain": "kspace"}
        assert caps.invertible is True
        assert caps.side_effects == ("loses_phase",)
        ADAPTER_REGISTRY.pop("__test_adapter_a", None)

    def test_invalid_insertion_point_raises(self):
        with pytest.raises(ValueError, match="unknown insertion points"):

            @register_adapter(
                name="__test_bad",
                bridges_from={},
                bridges_to={},
                insertion_points=("not_a_real_hook",),
            )
            class _Bad:
                pass

    def test_all_canonical_insertion_points_accepted(self):
        for hook in sorted(VALID_INSERTION_POINTS):
            name = f"__test_hook_{hook}"

            @register_adapter(
                name=name,
                bridges_from={"x": 1},
                bridges_to={"x": 2},
                insertion_points=(hook,),
            )
            class _OK:
                pass

            assert get_adapter_capabilities(name) is not None
            ADAPTER_REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------
# Adapter chain composition
# ---------------------------------------------------------------------------


class TestFormsMatch:
    def test_none_is_unconstrained(self):
        ok, m = forms_match(produced={"a": None}, required={"a": 5})
        assert ok and not m

    def test_required_tuple_membership(self):
        ok, _ = forms_match(produced={"d": 2}, required={"d": (2, 3)})
        assert ok
        ok, _ = forms_match(produced={"d": 1}, required={"d": (2, 3)})
        assert not ok

    def test_missing_keys_treated_as_unconstrained(self):
        ok, m = forms_match(produced={"a": 1}, required={"b": 2})
        assert ok and not m


class TestComposeChain:
    @pytest.fixture(autouse=True)
    def _register(self):
        # Pre-register two adapters that bridge X→Y→Z.
        @register_adapter(
            name="__chain_xy",
            bridges_from={"d": "x"},
            bridges_to={"d": "y"},
            insertion_points=("pre_model",),
        )
        class _XY:
            pass

        @register_adapter(
            name="__chain_yz",
            bridges_from={"d": "y"},
            bridges_to={"d": "z"},
            insertion_points=("pre_model",),
        )
        class _YZ:
            pass

        yield
        ADAPTER_REGISTRY.pop("__chain_xy", None)
        ADAPTER_REGISTRY.pop("__chain_yz", None)

    def test_valid_chain(self):
        result = compose_chain({"d": "x"}, ["__chain_xy", "__chain_yz"])
        assert result.bridged
        assert result.output_form["d"] == "z"

    def test_broken_chain_reports_first_break(self):
        # __chain_yz can't consume "x" directly.
        result = compose_chain({"d": "x"}, ["__chain_yz", "__chain_xy"])
        assert not result.bridged
        assert "expects" in result.error and "__chain_yz" in result.error

    def test_unknown_adapter(self):
        result = compose_chain({"d": "x"}, ["__not_registered"])
        assert not result.bridged
        assert "not registered" in result.error


class TestCandidateSuggestions:
    def test_single_step_candidate_found(self):
        @register_adapter(
            name="__cand_only",
            bridges_from={"d": "a"},
            bridges_to={"d": "b"},
            insertion_points=("pre_model",),
        )
        class _C:
            pass

        cands = candidate_adapters_for(
            upstream_form={"d": "a"},
            required_form={"d": "b"},
            hook="pre_model",
        )
        assert "__cand_only" in cands
        ADAPTER_REGISTRY.pop("__cand_only", None)

    def test_hook_filter_excludes_wrong_hook(self):
        @register_adapter(
            name="__cand_wrong_hook",
            bridges_from={"d": "a"},
            bridges_to={"d": "b"},
            insertion_points=("pre_loss_target",),
        )
        class _C:
            pass

        # Asking for pre_model should exclude this candidate.
        cands = candidate_adapters_for(
            upstream_form={"d": "a"},
            required_form={"d": "b"},
            hook="pre_model",
        )
        assert "__cand_wrong_hook" not in cands
        ADAPTER_REGISTRY.pop("__cand_wrong_hook", None)


class TestLossCapabilitiesDomainAgnostic:
    """``domain_agnostic`` is a flag, deliberately not a ``Domain`` member.

    A signal domain says what the tensor IS; "agnostic" says the loss does not
    care. Admitting the second into the literal would put a non-domain into the
    vocabulary ``SignalDomain`` exists to keep closed -- the four-copies-diverge
    defect one level up.
    """

    def test_it_defaults_to_false(self) -> None:
        from mriforge.models.capabilities import LossCapabilities

        assert LossCapabilities().domain_agnostic is False

    def test_it_is_independent_of_domain(self) -> None:
        """``domain=None`` means UNANNOTATED; ``domain_agnostic=True`` means
        deliberately generic. Collapsing them is what let 104 of 214
        registrations read as intentional while being merely un-audited."""
        from mriforge.models.capabilities import LossCapabilities

        unannotated = LossCapabilities()
        generic = LossCapabilities(domain_agnostic=True)
        assert unannotated.domain is None and generic.domain is None
        assert unannotated != generic

    def test_agnostic_is_absent_from_the_domain_literal(self) -> None:
        import typing

        from mriforge.models.capabilities import Domain

        assert "agnostic" not in typing.get_args(Domain)
