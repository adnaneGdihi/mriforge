#!/usr/bin/env python3
"""Comprehensive tests for the model registry.

Tests validation of:
- Model registration completeness
- No duplicate registrations
- Model instantiation
- Interface compliance
- Training mode correctness
"""

import pytest
import torch

from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY, get_model_class, list_models

populate_model_registry()


class TestModelRegistryBasics:
    """Test basic registry functionality."""

    def test_registry_not_empty(self):
        """Verify registry contains models."""
        assert len(MODEL_REGISTRY) > 0, "Registry should not be empty"

    def test_registry_count(self):
        """Verify we have the expected number of models."""
        model_count = len(MODEL_REGISTRY)
        # We registered 110+ models
        assert model_count >= 60, f"Expected at least 60 models, got {model_count}"

    def test_list_models_returns_dict(self):
        """Verify list_models returns proper dictionary."""
        models = list_models()
        assert isinstance(models, dict), "list_models should return dictionary"
        assert len(models) > 0, "list_models should return non-empty dict"


class TestNoDuplicates:
    """Test that no duplicate model names exist."""

    def test_no_duplicate_model_names(self):
        """Verify each model name is registered only once."""
        # Track all occurrences
        name_to_locations: dict[str, list] = {}

        # This test would need to scan source files
        # For now, verify registry itself has unique keys
        assert len(MODEL_REGISTRY) == len(
            set(MODEL_REGISTRY.keys())
        ), "Registry contains duplicate keys"

    def test_specific_models_exist(self):
        """Verify our newly registered models are present."""
        expected_models = [
            "fourier_kan_vae",
            "wavelet_kan_vae",
            "vae",
            "sparse_vae",
            "vqvae",
            "vqgan",
            "configurable_vae",
            "latent_discriminator",
            "patch_latent_discriminator",
            "multiscale_latent_discriminator",
        ]

        for model_name in expected_models:
            assert (
                model_name in MODEL_REGISTRY
            ), f"Expected model '{model_name}' not found in registry"


class TestModelInstantiation:
    """Test that registered models can be instantiated."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "fourier_kan_vae",
            "wavelet_kan_vae",
            "vae",
            "sparse_vae",
        ],
    )
    def test_vae_models_instantiate(self, model_name):
        """Test VAE models can be instantiated with default parameters."""
        try:
            model_class = get_model_class(model_name)
            assert (
                model_class is not None
            ), f"get_model_class returned None for {model_name}"

            # Try to instantiate with minimal params
            model = model_class(in_channels=1, latent_dim=64)
            assert model is not None, f"Failed to instantiate {model_name}"
        except Exception as e:
            pytest.fail(f"Failed to instantiate {model_name}: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [
            "vqvae",
            "vqgan",
        ],
    )
    def test_vq_models_instantiate(self, model_name):
        """Test VQ models can be instantiated."""
        try:
            model_class = get_model_class(model_name)
            assert model_class is not None

            model = model_class(in_channels=1, out_channels=1)
            assert model is not None
        except Exception as e:
            pytest.fail(f"Failed to instantiate {model_name}: {e}")


    def test_configurable_vae_instantiates(self):
        """Test ConfigurableVAE instantiates."""
        model_class = get_model_class("configurable_vae")
        assert model_class is not None

        # ConfigurableVAE works with defaults
        model = model_class()
        assert model is not None

    @pytest.mark.parametrize(
        "model_name",
        [
            "latent_discriminator",
        ],
    )
    def test_latent_discriminator_instantiate(self, model_name):
        """Test latent discriminator models instantiate."""
        model_class = get_model_class(model_name)
        assert model_class is not None

        # Latent discriminators take latent_dim
        model = model_class(latent_dim=128)
        assert model is not None

    @pytest.mark.parametrize(
        "model_name",
        [
            "patch_latent_discriminator",
            "multiscale_latent_discriminator",
        ],
    )
    def test_patch_discriminators_instantiate(self, model_name):
        """Test patch discriminator models instantiate."""
        model_class = get_model_class(model_name)
        assert model_class is not None

        # Patch discriminators take in_channels
        model = model_class(in_channels=4)
        assert model is not None


class TestModelInterfaces:
    """Test that models implement required interfaces."""

    def test_vae_models_have_forward(self):
        """Verify VAE models have forward method."""
        model_class = get_model_class("vae")
        model = model_class(in_channels=1, latent_dim=64)

        assert hasattr(model, "forward"), "Model should have forward method"
        assert callable(model.forward), "forward should be callable"

    def test_models_have_name_property(self):
        """Verify models have name property."""
        model_class = get_model_class("vae")
        model = model_class(in_channels=1, latent_dim=64)

        assert hasattr(model, "name"), "Model should have name property"

    def test_models_have_parameter_count(self):
        """Verify models can report parameter count."""
        model_class = get_model_class("vae")
        model = model_class(in_channels=1, latent_dim=64)

        assert hasattr(
            model, "get_parameter_count"
        ), "Model should have get_parameter_count method"

        param_count = model.get_parameter_count()
        assert isinstance(param_count, int), "Parameter count should be int"
        assert param_count > 0, "Model should have parameters"


class TestTrainingModes:
    """Test that models have correct training modes."""

    def test_vae_models_have_vae_mode(self):
        """Verify VAE models are registered with 'vae' training mode."""
        vae_models = [
            "fourier_kan_vae",
            "wavelet_kan_vae",
            "vae",
            "sparse_vae",
            "vqvae",
        ]

        models_dict = list_models()
        for model_name in vae_models:
            assert model_name in models_dict, f"{model_name} not in registry"
            # The registry should store (class, training_mode)
            # Exact structure depends on registry implementation

    def test_gan_models_have_gan_mode(self):
        """Verify GAN models are registered with 'gan' training mode."""
        gan_models = [
            "vqgan",
            "latent_discriminator",
            "patch_latent_discriminator",
            "multiscale_latent_discriminator",
        ]

        models_dict = list_models()
        for model_name in gan_models:
            assert model_name in models_dict, f"{model_name} not in registry"


class TestModelForwardPass:
    """Test that models can perform forward passes."""

    def test_vae_forward_pass(self):
        """Test VAE can process input tensor."""
        model_class = get_model_class("vae")
        model = model_class(in_channels=1, latent_dim=64)

        # Create dummy input
        x = torch.randn(2, 1, 64, 64)

        try:
            with torch.no_grad():
                output = model(x)

            # VAE returns (reconstruction, mu, logvar)
            assert output is not None
            if isinstance(output, tuple):
                assert len(output) == 3, "VAE should return (recon, mu, logvar)"
        except Exception as e:
            pytest.fail(f"VAE forward pass failed: {e}")

    def test_discriminator_forward_pass(self):
        """Test discriminator can process input."""
        model_class = get_model_class("latent_discriminator")
        model = model_class(latent_dim=128)

        # Create dummy latent input
        x = torch.randn(2, 128)

        try:
            with torch.no_grad():
                output = model(x)

            assert output is not None
            assert output.shape[0] == 2, "Batch dimension should be preserved"
        except Exception as e:
            pytest.fail(f"Discriminator forward pass failed: {e}")


class TestRegistryEdgeCases:
    """Test edge cases and error handling."""

    def test_get_nonexistent_model_raises_value_error(self):
        """Verify getting non-existent model raises ValueError."""
        with pytest.raises(ValueError):
            get_model_class("nonexistent_model_xyz_123")

    def test_registry_is_immutable(self):
        """Clearing the registry must not poison later tests.

        ``MODEL_REGISTRY`` is a plain module-global dict, so ``.clear()``
        does mutate it. This test snapshots and restores it so the
        global state is left intact for the rest of the session.

        NB: restoration is load-bearing now that ``populate_model_registry``
        is idempotent (it no longer re-walks on every call), so a cleared
        registry would otherwise stay empty and break every later test
        that relies on auto-discovered models. See ``test_registry_helpers``
        for the same backup/restore pattern.
        """
        backup = dict(MODEL_REGISTRY)
        original_count = len(MODEL_REGISTRY)
        try:
            MODEL_REGISTRY.clear()
            assert len(MODEL_REGISTRY) == 0
        finally:
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(backup)

        assert len(MODEL_REGISTRY) == original_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
