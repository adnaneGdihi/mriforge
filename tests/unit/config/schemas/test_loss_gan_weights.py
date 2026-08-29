"""Tests for translation/StarGAN loss weights on GANLossesConfig."""

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.loss import GANLossesConfig


class TestGANLossTranslationWeights:
    """Verify that cycle/identity/nce/style/diversity weights exist and are validated."""

    def test_gan_loss_weights_defaults(self):
        c = GANLossesConfig()
        assert c.lambda_cycle == 10.0
        assert c.lambda_identity == 0.5
        assert c.lambda_nce == 1.0
        assert c.lambda_style == 1.0
        assert c.lambda_diversity == 1.0

    def test_gan_loss_weight_custom_values(self):
        c = GANLossesConfig(
            lambda_cycle=5.0,
            lambda_identity=0.1,
            lambda_nce=2.0,
            lambda_style=0.5,
            lambda_diversity=3.0,
        )
        assert c.lambda_cycle == 5.0
        assert c.lambda_identity == 0.1
        assert c.lambda_nce == 2.0
        assert c.lambda_style == 0.5
        assert c.lambda_diversity == 3.0

    def test_gan_loss_weight_negative_rejected(self):
        with pytest.raises(ValidationError):
            GANLossesConfig(lambda_cycle=-1.0)

    def test_gan_loss_weight_negative_identity_rejected(self):
        with pytest.raises(ValidationError):
            GANLossesConfig(lambda_identity=-0.1)

    def test_gan_loss_weight_negative_nce_rejected(self):
        with pytest.raises(ValidationError):
            GANLossesConfig(lambda_nce=-1.0)

    def test_gan_loss_weight_negative_style_rejected(self):
        with pytest.raises(ValidationError):
            GANLossesConfig(lambda_style=-1.0)

    def test_gan_loss_weight_negative_diversity_rejected(self):
        with pytest.raises(ValidationError):
            GANLossesConfig(lambda_diversity=-1.0)

    def test_gan_loss_weight_zero_accepted(self):
        """Zero is valid (disables the loss term)."""
        c = GANLossesConfig(
            lambda_cycle=0.0,
            lambda_identity=0.0,
            lambda_nce=0.0,
            lambda_style=0.0,
            lambda_diversity=0.0,
        )
        assert c.lambda_cycle == 0.0
        assert c.lambda_diversity == 0.0
