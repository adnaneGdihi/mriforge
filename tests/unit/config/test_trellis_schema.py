import pytest
from pydantic import ValidationError

from spectramr.config.schemas.trellis import (
    TrellisConfigSchema,
    TrellisDecoderConfigSchema,
    TrellisEncoderConfigSchema,
    TrellisTrainerConfigSchema,
    TrellisVAEConfigSchema,
)


class TestTrellisConfigSchema:
    def test_defaults(self):
        schema = TrellisConfigSchema()
        assert schema.representation == "gaussian"
        assert schema.input_resolution == (224, 224)
        assert schema.output_resolution == (64, 128, 128)
        assert schema.feature_dim == 256
        assert schema.num_views == 4
        assert schema.num_layers == 12
        assert schema.num_heads == 8
        assert schema.mlp_ratio == 4.0
        assert schema.num_stages == 3
        assert schema.stride == 8
        assert schema.patch_size == 8
        assert schema.gaussian_num_gaussians == 32768
        assert schema.gaussian_resolution == 32
        assert schema.gaussian_latent_dim == 64
        assert schema.mesh_num_vertices == 50000
        assert schema.mesh_num_faces == 100000
        assert schema.mesh_latent_dim == 64

    def test_validation(self):
        with pytest.raises(ValidationError):
            TrellisConfigSchema(representation="invalid")
        with pytest.raises(ValidationError):
            TrellisConfigSchema(input_resolution=(224,))
        with pytest.raises(ValidationError):
            TrellisConfigSchema(output_resolution=(64, 128))
        with pytest.raises(ValidationError):
            TrellisConfigSchema(input_resolution=(-1, 224))


class TestTrellisEncoderConfigSchema:
    def test_defaults(self):
        schema = TrellisEncoderConfigSchema()
        assert schema.name == "ElasticSLatEncoder"
        assert schema.resolution == 64
        assert schema.in_channels == 1024
        assert schema.model_channels == 768
        assert schema.latent_channels == 8
        assert schema.num_blocks == 12
        assert schema.num_heads == 12
        assert schema.mlp_ratio == 4.0
        assert schema.attn_mode == "swin"
        assert schema.window_size == 8
        assert schema.use_fp16 is False


class TestTrellisDecoderConfigSchema:
    def test_defaults(self):
        schema = TrellisDecoderConfigSchema()
        assert schema.name == "ElasticSLatGaussianDecoder"
        assert schema.resolution == 64
        assert schema.model_channels == 768
        assert schema.latent_channels == 8
        assert schema.num_blocks == 12
        assert schema.num_heads == 12
        assert schema.mlp_ratio == 4.0
        assert schema.attn_mode == "swin"
        assert schema.window_size == 8
        assert schema.use_fp16 is False
        assert schema.output_channels == 1
        assert schema.output_resolution == (64, 64, 64)
        assert schema.representation_config == {}

    def test_validation(self):
        with pytest.raises(ValidationError):
            TrellisDecoderConfigSchema(output_resolution=(64, 64))


class TestTrellisTrainerConfigSchema:
    def test_defaults(self):
        schema = TrellisTrainerConfigSchema()
        assert schema.max_steps == 100000
        assert schema.batch_size_per_gpu == 1
        assert schema.batch_split == 1
        assert schema.optimizer["name"] == "AdamW"
        assert schema.ema_rate == [0.999]
        assert schema.fp16_mode == "off"
        assert schema.i_log == 500
        assert schema.loss_type == "l1"
        assert schema.lambda_ssim == 0.0
        assert schema.lambda_kl == 1e-4


class TestTrellisVAEConfigSchema:
    def test_defaults(self):
        schema = TrellisVAEConfigSchema()
        assert isinstance(schema.encoder, TrellisEncoderConfigSchema)
        assert isinstance(schema.decoder, TrellisDecoderConfigSchema)
        assert isinstance(schema.trainer, TrellisTrainerConfigSchema)
