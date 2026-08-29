import pytest
from pydantic import ValidationError

from mriforge.config.schemas.model import ModelComponentSchema, ModelConfigSchema


class TestModelConfigSchema:
    def test_defaults(self):
        schema = ModelConfigSchema()
        assert schema.in_channels == 1
        assert schema.out_channels == 1
        assert schema.model_type == "unet"
        assert schema.z_dim == 256
        assert schema.kan_type == "BSpline"
        assert schema.input_type == "image"

    def test_custom_values(self):
        schema = ModelConfigSchema(
            in_channels=3,
            out_channels=3,
            model_type="vit",
            z_dim=128,
            kan_type="Fourier",
            input_type="latent",
        )
        assert schema.in_channels == 3
        assert schema.out_channels == 3
        assert schema.model_type == "vit"
        assert schema.z_dim == 128
        assert schema.kan_type == "Fourier"
        assert schema.input_type == "latent"

    def test_checkpoint_path_is_a_typed_field_not_dropped(self):
        """``model.checkpoint_path`` must be a first-class field. It is the
        sequential-campaign injection target (``inject_as: model.checkpoint_path``):
        ``ModelConfigSchema`` is ``extra='ignore'``, so without a declared field
        the injected value is silently dropped on re-validation and the warm-start
        from another model's checkpoint never happens (pitfall #15/#16)."""
        schema = ModelConfigSchema(checkpoint_path="experiments/results/stage0/best.pt")
        assert schema.checkpoint_path == "experiments/results/stage0/best.pt"
        # Default is None (no init checkpoint).
        assert ModelConfigSchema().checkpoint_path is None

    def test_validation_model_type(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(model_type="invalid_model")

    def test_validation_kan_type(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(kan_type="invalid_kan")

    def test_validation_input_type(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(input_type="invalid_input")

    def test_components(self):
        gen = ModelComponentSchema(name="gen", kwargs={"a": 1})
        disc = ModelComponentSchema(name="disc", kwargs={"b": 2})
        schema = ModelConfigSchema(
            generator_component=gen, discriminator_component=disc
        )
        assert schema.generator_component.name == "gen"
        assert schema.discriminator_component.name == "disc"

    def test_model_domain_and_output_type_fields(self):
        # H4 (audit 2026-05-28): these are declared explicitly so they receive
        # typed validation rather than being passed through untyped.
        schema = ModelConfigSchema(model_domain="kspace", output_type="complex")
        assert schema.model_domain == "kspace"
        assert schema.output_type == "complex"

    def test_extra_keys_are_ignored_not_forbidden(self):
        # H4: the model block is deliberately extra="ignore" (NOT forbid) — it
        # is polymorphic and arch-specific knobs belong in model_kwargs. The
        # forbid migration is tracked in
        # TODO/backlog_model_block_strictness_2026_05_28.md. Until it lands,
        # unknown top-level keys must be silently dropped (status quo), not
        # rejected. See ModelConfigSchema docstring.
        schema = ModelConfigSchema(channel_multipliers=[1, 2, 4])
        assert not hasattr(schema, "channel_multipliers")


class TestModelComponentSchema:
    def test_defaults(self):
        schema = ModelComponentSchema()
        assert schema.name == ""
        assert schema.kwargs == {}

    def test_custom_values(self):
        schema = ModelComponentSchema(name="test", kwargs={"x": 10})
        assert schema.name == "test"
        assert schema.kwargs == {"x": 10}


class TestVirtualFiducialModelTypes:
    """Regression tests for Virtual Fiducial experiment model types."""

    def test_hyper_mamba_unet_is_valid(self):
        """hyper_mamba_unet must pass Pydantic validation (VF experiments B & TTO)."""
        schema = ModelConfigSchema(model_type="hyper_mamba_unet")
        assert schema.model_type == "hyper_mamba_unet"

    def test_cross_attention_oracle_unet_is_valid(self):
        """cross_attention_oracle_unet must pass Pydantic validation (VF Method A & C)."""
        schema = ModelConfigSchema(model_type="cross_attention_oracle_unet")
        assert schema.model_type == "cross_attention_oracle_unet"

    def test_hyper_mamba_bridge_is_valid(self):
        """hyper_mamba_bridge must pass Pydantic validation."""
        schema = ModelConfigSchema(model_type="hyper_mamba_bridge")
        assert schema.model_type == "hyper_mamba_bridge"


class TestModelConditioningBlock:
    """Adaptive-conditioning block on the model schema (2026-05-26)."""

    def test_conditioning_defaults_disabled(self):
        assert ModelConfigSchema().conditioning.enabled is False

    def test_conditioning_block_is_parsed(self):
        schema = ModelConfigSchema(
            conditioning={"enabled": True, "sources": ["diffusion_t", "severity_vec"]}
        )
        assert schema.conditioning.sources == ["diffusion_t", "severity_vec"]

    def test_conditioning_rejects_unknown_source(self):
        with pytest.raises(ValidationError):
            ModelConfigSchema(conditioning={"enabled": True, "sources": ["banana"]})


class TestModelTypeHasOneAuthority:
    """The ``VALID_MODEL_TYPES`` fallback is gone (#978).

    It was documented as covering "the bootstrap window before
    ``populate_model_registry()`` runs", but it could not fire on the real load
    path (``from_yaml`` populates first), and where it *could* fire it gave the
    wrong answer -- the whitelist is a strict subset of the registry, so it only
    ever rejected names that are legitimately registered.
    """

    def test_a_registered_but_unadvertised_name_is_accepted(self):
        """The 130 names in MODEL_REGISTRY but not in VALID_MODEL_TYPES.

        Directly constructing the schema used to reject these, because the
        unpopulated registry sent the value to the whitelist. This is the
        regression the fallback caused, and the reason deleting it is a fix
        rather than only a simplification.
        """
        from mriforge.config.schemas.model import ModelConfigSchema
        from mriforge.config.validation_constants import VALID_MODEL_TYPES
        from mriforge.models.init_registry import populate_model_registry
        from mriforge.models.registry import MODEL_REGISTRY

        populate_model_registry()
        only_registered = sorted(set(MODEL_REGISTRY) - set(VALID_MODEL_TYPES))
        assert only_registered, "precondition: the registry must exceed the whitelist"
        assert ModelConfigSchema(model_type=only_registered[0]).model_type

    def test_an_unregistered_name_still_raises(self):
        from pydantic import ValidationError

        from mriforge.config.schemas.model import ModelConfigSchema

        with pytest.raises(ValidationError, match="not found in MODEL_REGISTRY"):
            ModelConfigSchema(model_type="definitely_not_a_model")

    def test_the_error_no_longer_advertises_a_second_authority(self):
        """A message naming two sources invites adding a name to the wrong one."""
        from pydantic import ValidationError

        from mriforge.config.schemas.model import ModelConfigSchema

        with pytest.raises(ValidationError) as exc:
            ModelConfigSchema(model_type="definitely_not_a_model")
        assert "VALID_MODEL_TYPES" not in str(exc.value)

    def test_validate_model_type_does_not_read_the_whitelist(self):
        """Source-level: the constant must not be reachable from this module."""
        import mriforge.config.schemas.model as mod

        assert not hasattr(mod, "VALID_MODEL_TYPES")
