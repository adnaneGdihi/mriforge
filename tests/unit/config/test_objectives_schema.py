from mriforge.config.schemas.base import DiffusionObjectiveConfig
from mriforge.config.schemas.objectives import ObjectiveConfigSchema


class TestObjectiveConfigSchema:
    def test_defaults(self):
        schema = ObjectiveConfigSchema()
        assert schema.gan is None
        assert schema.diffusion is None
        assert schema.reconstruction is None
        assert schema.latent is None
        assert schema.domain_adaptation is None
        assert schema.biophysical_flow is None
        assert schema.ssl is None

    def test_custom_values(self):
        # Note: Timesteps and latent_dim have been moved to training schemas (SSOT consolidation)
        # objectives now contains only remaining paradigm-specific hyperparameters
        # (timesteps no longer accepted in DiffusionObjectiveConfig - it's in training.diffusion)
        diffusion_config = DiffusionObjectiveConfig(cond_drop_prob=0.05)

        schema = ObjectiveConfigSchema(diffusion=diffusion_config)

        assert schema.diffusion == diffusion_config
        assert schema.diffusion.cond_drop_prob == 0.05
