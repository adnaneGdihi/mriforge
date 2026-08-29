from pydantic import BaseModel, Field

from mriforge.config.schemas.defaults_provider import ConfigDefaultsProvider


class TestDefaultsProvider:
    def test_initialization(self):
        provider = ConfigDefaultsProvider(environment="development")
        assert provider.environment == "development"

        provider = ConfigDefaultsProvider()
        assert provider.environment == "base"

    def test_get_defaults_base(self):
        provider = ConfigDefaultsProvider(environment="base")
        defaults = provider.get_defaults()

        assert defaults["epochs"] == 200
        assert defaults["training_mode"] == "reconstruction"
        assert defaults["optimization"]["learning_rate"] == 1e-5

    def test_get_defaults_development(self):
        provider = ConfigDefaultsProvider(environment="development")
        defaults = provider.get_defaults()

        assert defaults["epochs"] == 50
        assert defaults["logging"]["silent"] == False
        assert defaults["services"]["enable_profiling"] == True

    def test_get_defaults_production(self):
        provider = ConfigDefaultsProvider(environment="production")
        defaults = provider.get_defaults()

        assert defaults["epochs"] == 500
        assert defaults["logging"]["silent"] == True
        assert defaults["services"]["enable_profiling"] == False

    def test_get_defaults_testing(self):
        provider = ConfigDefaultsProvider(environment="testing")
        defaults = provider.get_defaults()

        assert defaults["epochs"] == 1
        assert defaults["data"]["batch_size"] == 1

    def test_get_defaults_reconstruction_mode(self):
        provider = ConfigDefaultsProvider()
        defaults = provider.get_defaults(training_mode="reconstruction")

        assert "enable_data_consistency" in defaults
        assert defaults["enable_data_consistency"] == False

    def test_get_defaults_other_mode(self):
        provider = ConfigDefaultsProvider()
        defaults = provider.get_defaults(training_mode="gan")

        assert "enable_data_consistency" not in defaults

    def test_get_default_for_field(self):
        provider = ConfigDefaultsProvider()
        assert provider.get_default_for_field("epochs") == 200
        assert provider.get_default_for_field("non_existent") is None

    def test_merge_with_defaults(self):
        provider = ConfigDefaultsProvider()
        user_config = {"epochs": 10, "new_field": "value"}

        merged = provider.merge_with_defaults(user_config)

        assert merged["epochs"] == 10
        assert merged["new_field"] == "value"
        assert merged["training_mode"] == "reconstruction"  # Default preserved

    def test_extract_defaults_from_schema(self):
        class MockSchema(BaseModel):
            field1: int = 10
            field2: str = Field(default="test")
            field3: int  # Required

        defaults = ConfigDefaultsProvider.extract_defaults_from_schema(MockSchema)

        assert defaults["field1"] == 10
        assert defaults["field2"] == "test"
        assert "field3" not in defaults

    def test_extract_nested_defaults_from_schema(self):
        class InnerSchema(BaseModel):
            inner_field: int = 5

        class OuterSchema(BaseModel):
            outer_field: int = 10
            nested: InnerSchema = Field(default_factory=InnerSchema)

        defaults = ConfigDefaultsProvider.extract_nested_defaults_from_schema(
            OuterSchema
        )

        assert defaults["outer_field"] == 10
        assert defaults["nested.inner_field"] == 5


def test_base_default_blocks_are_loadable_by_their_own_schemas():
    """A defaults provider that emits a config its schema rejects is worse
    than no provider at all.

    The ``ema`` block drifted exactly that way: it kept the ``enable_ema``
    alias that v5.1 removed (``extra="forbid"`` rejects it) and declared
    ``enable_adaptive_ema`` alongside the two stability knobs that have no
    consumer (#1294).
    """
    from mriforge.config.schemas.ema import EMAConfigSchema

    defaults = ConfigDefaultsProvider(environment="base").get_defaults()
    ema = EMAConfigSchema.model_validate(defaults["ema"])
    assert ema.enabled is True
    assert ema.enable_adaptive_ema is False
