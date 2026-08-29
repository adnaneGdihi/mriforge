"""Unit tests for Domain Adaptation Training Strategy.

Tests cover:
- Initialization
- Domain discriminator setup
- Loss computation
- Validation metrics
"""

from unittest.mock import MagicMock

import pytest

from mriforge.config.schemas.loss import LossConfigSchema
from tests.utils.config_block_stub import block_stub
import torch
import torch.nn as nn

from mriforge.config.schemas.loss import (
    DiffusionLossesConfig,
    GANLossesConfig,
    LatentLossesConfig,
    PhysicsLossesConfig,
    ReconstructionLossesConfig,
    SSLLossesConfig,
)
from mriforge.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from tests.utils.mock_environment import create_mock_training_env


class MockGenerator(nn.Module):
    """Mock generator for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class MockDiscriminator(nn.Module):
    """Mock discriminator for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3), keepdim=True)

    def extract_features(self, x):
        return self.conv(x)


class MockDomainDiscriminator(nn.Module):
    """Mock domain discriminator for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32 * 32, 1)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


@pytest.fixture
def mock_env():
    """Create mock training environment for domain adaptation."""
    generator = MockGenerator()
    discriminator = MockDiscriminator()

    # Create optimizer using builder for structure correctness
    optimizer_g = OptimizationBuilder.create_single_optimizer(
        generator.parameters(), learning_rate=0.001, optimizer_type="adam"
    )
    optimizer_d = OptimizationBuilder.create_single_optimizer(
        discriminator.parameters(), learning_rate=0.001, optimizer_type="adam"
    )

    config = MagicMock()
    config.training_mode = "domain_adaptation"
    config.deep_supervision_weight = 0.0
    config.model.model_type = "test"

    config.training = MagicMock()
    config.training.training_mode = "domain_adaptation"
    config.training.latent_dim = 128

    # Configure optimization config
    config.optimization = MagicMock()
    config.optimization.precision.enabled = False
    # resolve_amp_precision() validates amp_dtype against {float16,bfloat16,float32}
    # and raises on an unknown value (pitfall #9); a bare MagicMock is "unknown".
    config.optimization.precision.dtype = "float16"
    config.optimization.mixed_precision = "fp16"
    config.optimization.gradient.clip.value = 1.0
    config.optimization.max_grad_norm = 1.0
    config.optimization.gradient.accumulation_steps = 1
    config.optimization.gradient.clip.enabled = False
    # StandardOptimizerStepper validates this against {"norm", "value"} and raises
    # on an unknown value (pitfall #9). A bare MagicMock attribute is "unknown", so
    # the stepper rejected the env at construction — mirror the schema default.
    config.optimization.gradient.clip.method = "norm"
    config.optimization.loss_scaling = 1024.0

    # Configure acceleration
    config.acceleration = MagicMock()
    config.acceleration.gradient_accumulation_steps = 1

    # Physics config
    config.physics = MagicMock()
    config.physics.pinn = MagicMock()
    config.physics.pinn = MagicMock()
    config.physics.pinn.enabled = False

    # Configure losses. The whole block is the REAL schema, not a MagicMock:
    # `losses` auto-vivifies every sub-block a reader asks for, so a retired one
    # (`adversarial`, `regularization` — neither is a field on LossConfigSchema)
    # arrived at `enabled_losses.get("adversarial", 0) > 0` as a Mock and failed
    # as a TypeError instead of being absent, and `losses.output_domain` (folded
    # to `losses.policy.output_domain`) arrived where production validates it
    # against {kspace, image, complex_image, latent}.
    config.losses = LossConfigSchema(
        policy={"output_domain": "image"},
        reconstruction=ReconstructionLossesConfig(
            enable_l1=True,
            lambda_l1=1.0,
            enable_perceptual=False,
            enable_ssim=False,
        ),
        gan=GANLossesConfig(enable_adversarial=False, enable_r1=False),
        physics=PhysicsLossesConfig(),
        diffusion=DiffusionLossesConfig(enable_diffusion=False),
        # Constructor kwargs, not post-hoc assignment: the schema is frozen like
        # a live config, so `config.losses.latent = ...` raises
        # "Instance is frozen" once `losses` stops being a MagicMock.
        latent=LatentLossesConfig(
            enable_reconstruction=False,
            enable_kl=False,
            enable_codebook=False,
            enable_commitment=False,
        ),
        ssl=SSLLossesConfig(),
    )

    # Configure training config for PaDNet/Diffusion
    # Use MagicMock with explicit values to avoid schema validation issues (deprecated objectives)
    config.training.num_timesteps = 1000
    config.training.noise_schedule = "linear"
    config.training.prediction_type = "epsilon"
    config.training.beta_start = 0.0001
    config.training.beta_end = 0.02

    # Also set on config.training.diffusion because some strategies might access it there
    config.training.diffusion = MagicMock()
    config.training.diffusion.num_timesteps = 1000
    config.training.diffusion.timesteps = (
        1000  # Explicitly set timesteps for validation
    )
    config.training.diffusion.noise_schedule = "linear"
    config.training.diffusion.prediction_type = "epsilon"
    config.training.diffusion.beta_start = 0.0001
    config.training.diffusion.beta_end = 0.02

    # Data config
    config.data = MagicMock()
    config.data.prior_loading = MagicMock()
    config.data.prior_loading.enabled = False

    return create_mock_training_env(
        config=config,
        device="cpu",
        generator=generator,
        discriminator=discriminator,
        opt_g=optimizer_g,
        opt_d=optimizer_d,
        model_type="test",
    )


@pytest.fixture
def sample_batch():
    """Create sample batch tensors."""
    lr_batch = torch.randn(2, 1, 32, 32)
    hr_batch = torch.randn(2, 1, 32, 32)
    return lr_batch, hr_batch


class TestDomainAdaptationTrainingStrategy:
    """Test DomainAdaptationTrainingStrategy."""

    @pytest.fixture
    def strategy(self, mock_env):
        """Create domain adaptation strategy."""
        from mriforge.infrastructure.training.strategies import (
            DomainAdaptationTrainingStrategy,
        )

        return DomainAdaptationTrainingStrategy(env=mock_env)

    def test_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy is not None
        assert strategy.domain_discriminator is None  # Not initialized yet
        assert strategy.lambda_domain == 1.0

    def test_initialize_domain_adaptation(self, strategy):
        """Test domain adaptation initialization."""
        domain_disc = MockDomainDiscriminator()
        domain_opt = torch.optim.Adam(domain_disc.parameters(), lr=0.001)

        strategy.initialize_domain_adaptation(
            domain_discriminator=domain_disc,
            domain_optimizer=domain_opt,
            lambda_domain=0.5,
        )

        assert strategy.domain_discriminator is not None
        assert strategy.domain_optimizer is not None
        assert strategy.lambda_domain == 0.5
        assert strategy.domain_loss_fn is not None

    def test_compute_domain_loss_without_discriminator(self, strategy, sample_batch):
        """Test domain loss computation without discriminator."""
        lr_batch, hr_batch = sample_batch
        hr_fakes = strategy.env.models["generator"](lr_batch)

        loss = strategy.compute_domain_loss(hr_fakes, hr_batch)

        # Should return 0 when no domain discriminator
        assert loss.item() == 0.0

    def test_compute_domain_loss_with_discriminator(self, strategy, sample_batch):
        """Test domain loss computation with discriminator."""
        lr_batch, hr_batch = sample_batch

        # Initialize domain discriminator
        domain_disc = MockDomainDiscriminator()
        domain_opt = torch.optim.Adam(domain_disc.parameters(), lr=0.001)
        strategy.initialize_domain_adaptation(domain_disc, domain_opt)

        hr_fakes = strategy.env.models["generator"](lr_batch)
        loss = strategy.compute_domain_loss(hr_fakes, hr_batch)

        # Should return non-zero loss
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_compute_metrics(self, strategy, sample_batch):
        """Test metric computation."""
        lr_batch, hr_batch = sample_batch
        hr_fakes = strategy.env.models["generator"](lr_batch)

        metrics = strategy.compute_metrics(hr_fakes, hr_batch)

        assert isinstance(metrics, dict)

    def test_validation_step(self, strategy, mock_env, sample_batch):
        """Test validation step."""
        lr_batch, hr_batch = sample_batch

        metrics = strategy.validation_step(lr_batch, hr_batch)

        assert isinstance(metrics, dict)

    def test_validation_with_domain_discriminator(self, strategy, sample_batch):
        """Test validation includes domain loss when discriminator present."""
        lr_batch, hr_batch = sample_batch

        # Initialize domain discriminator
        domain_disc = MockDomainDiscriminator()
        domain_opt = torch.optim.Adam(domain_disc.parameters(), lr=0.001)
        strategy.initialize_domain_adaptation(domain_disc, domain_opt)

        metrics = strategy.validation_step(lr_batch, hr_batch)

        assert "domain_loss" in metrics
        # SS-003 regression: domain_loss must be a Python float (not a Tensor) so
        # downstream metric logging / JSON provenance serialization works.
        assert isinstance(metrics["domain_loss"], float)
        assert not isinstance(metrics["domain_loss"], torch.Tensor)


class TestDisentangledTrainingStrategy:
    """Test DisentangledTrainingStrategy."""

    @pytest.fixture
    def strategy(self, mock_env):
        """Create disentangled strategy."""
        from mriforge.infrastructure.training.strategies import DisentangledTrainingStrategy

        return DisentangledTrainingStrategy(env=mock_env)

    def test_initialization(self, strategy):
        """Test disentangled strategy initializes correctly."""
        assert strategy is not None


class TestPaDNetTrainingStrategy:
    """Test PaDNetTrainingStrategy."""

    @pytest.fixture
    def diffusion_env(self, mock_env):
        """Create mock state for diffusion-based strategies."""
        mock_env.config.training_mode = "diffusion"
        return mock_env

    @pytest.fixture
    def strategy(self, diffusion_env):
        """Create PaDNet strategy."""
        from mriforge.infrastructure.training.strategies import PaDNetTrainingStrategy

        return PaDNetTrainingStrategy(env=diffusion_env)

    def test_initialization(self, strategy):
        """Test PaDNet strategy initializes correctly."""
        assert strategy is not None


class TestMetaLearningTrainingStrategy:
    """Test MetaLearningTrainingStrategy."""

    @pytest.fixture
    def strategy(self, mock_env):
        """Create meta-learning strategy."""
        from mriforge.infrastructure.training.strategies import MetaLearningTrainingStrategy

        return MetaLearningTrainingStrategy(env=mock_env)

    def test_initialization(self, strategy):
        """Test meta-learning strategy initializes correctly."""
        assert strategy is not None


class TestMaskedPretrainingStrategy:
    """Test MaskedPretrainingStrategy."""

    @pytest.fixture
    def mae_env(self, mock_env):
        """Create mock state for MAE strategies."""
        mock_env.config.training_mode = "mae_pretraining"
        return mock_env

    @pytest.fixture
    def strategy(self, mae_env):
        """Create masked pretraining strategy."""
        from mriforge.infrastructure.training.strategies import MaskedPretrainingStrategy

        return MaskedPretrainingStrategy(env=mae_env)

    def test_initialization(self, strategy):
        """Test masked pretraining strategy initializes correctly."""
        assert strategy is not None


class TestTRELLISTrainingStrategy:
    """Test TRELLISTrainingStrategy (volumetric)."""

    @pytest.fixture
    def volumetric_env(self, mock_env):
        """Create mock state for volumetric strategies."""
        mock_env.config.training_mode = "3d_generation"
        return mock_env

    @pytest.fixture
    def strategy(self, volumetric_env):
        """Create TRELLIS strategy."""
        from mriforge.infrastructure.training.strategies import TRELLISTrainingStrategy

        return TRELLISTrainingStrategy(env=volumetric_env)

    def test_initialization(self, strategy):
        """Test TRELLIS strategy initializes correctly."""
        assert strategy is not None


class TestTttAdaptationStrategy:
    """Test TttAdaptationStrategy (test-time adaptation)."""

    @pytest.fixture
    def strategy(self, mock_env):
        """Create TTT strategy."""
        from mriforge.infrastructure.training.strategies import TttAdaptationStrategy

        return TttAdaptationStrategy(env=mock_env)

    def test_initialization(self, strategy):
        """Test TTT strategy initializes correctly."""
        assert strategy is not None

    def test_ttt_closure_zero_grad_uses_set_to_none(self, strategy):
        """SS-005 regression: zero_grad inside ttt_closure must pass set_to_none=True.

        Both the intermediate inner-loop step and the final step call
        ``self.env.opt_g.zero_grad(set_to_none=True)`` (NN#9: avoid redundant
        memory-write transactions in the hot training path).
        """
        # Pin a small, concrete number of adaptation steps so the closure runs a
        # deterministic loop (the mock config leaves num_adaptation_steps a Mock).
        strategy.num_adaptation_steps = 2
        strategy.consistency_weight = 1.0

        # Wrap the real optimizer's zero_grad to record how it was called while
        # preserving the actual gradient-reset behaviour.
        real_opt = strategy.env.opt_g
        zero_grad_calls: list[dict] = []
        original_zero_grad = real_opt.zero_grad

        def _spy_zero_grad(*args, **kwargs):
            zero_grad_calls.append({"args": args, "kwargs": kwargs})
            return original_zero_grad(*args, **kwargs)

        real_opt.zero_grad = _spy_zero_grad  # type: ignore[method-assign]

        input_batch = torch.randn(2, 1, 32, 32)
        target_batch = torch.randn(2, 1, 32, 32)

        # No mask/kspace in kwargs -> TV-loss branch, fully CPU-differentiable.
        steps = strategy.train_step(
            batch=None,
            epoch=0,
            input_batch=input_batch,
            target_batch=target_batch,
        )
        closure = steps[0]["closure"]
        # Drive the closure (inner-loop intermediate steps + final step).
        closure()

        # One zero_grad per intermediate step (num_adaptation_steps - 1) plus the
        # final step == num_adaptation_steps total.
        assert len(zero_grad_calls) == strategy.num_adaptation_steps
        for call in zero_grad_calls:
            assert call["kwargs"].get("set_to_none") is True, (
                "ttt_closure must call zero_grad(set_to_none=True)"
            )
