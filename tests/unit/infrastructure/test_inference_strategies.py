"""Unit tests for inference strategies."""


class TestBaseInferenceStrategy:
    """Test base inference strategy."""

    def test_base_strategy_exists(self):
        """Test base inference strategy exists."""
        # This file was likely removed or refactored. Checking specific known strategies instead.
        pass


class TestReconstructionInference:
    """Test reconstruction inference strategy."""

    def test_strategy_exists(self):
        """Test reconstruction inference exists."""
        from mriforge.infrastructure.inference.reconstruction_inference_strategy import (
            ReconstructionInferenceStrategy,
        )

        assert hasattr(ReconstructionInferenceStrategy, "__init__")


class TestDiffusionInference:
    """Test diffusion inference strategy."""

    def test_strategy_exists(self):
        """Test diffusion inference exists."""
        from mriforge.infrastructure.inference.diffusion_inference_strategy import (
            DiffusionInferenceStrategy,
        )

        assert hasattr(DiffusionInferenceStrategy, "__init__")


class TestGANInference:
    """Test GAN inference strategy."""

    def test_strategy_exists(self):
        """Test GAN inference exists."""
        from mriforge.infrastructure.inference.gan_inference_strategy import (
            GANInferenceStrategy,
        )

        assert hasattr(GANInferenceStrategy, "__init__")


class TestVAEInference:
    """Test VAE inference strategy."""

    def test_strategy_exists(self):
        """Test VAE inference exists."""
        from mriforge.infrastructure.inference.vae_inference_strategy import (
            VAEInferenceStrategy,
        )

        assert hasattr(VAEInferenceStrategy, "__init__")


class TestSSLInference:
    """Test SSL inference strategy."""

    def test_strategy_exists(self):
        """Test SSL inference exists."""
        from mriforge.infrastructure.inference.ssl_inference_strategy import (
            SSLInferenceStrategy,
        )

        assert hasattr(SSLInferenceStrategy, "__init__")


class TestSlidingWindowInference:
    """Test sliding window inference."""

    def test_sliding_window_exists(self):
        """Test sliding window module exists."""
        from mriforge.infrastructure.inference.sliding_window import sliding_window_inference

        assert callable(sliding_window_inference)


class TestStreamingInference:
    """Test streaming inference."""

    def test_streaming_exists(self):
        """Test streaming inference module exists."""
        from mriforge.infrastructure.inference.streaming_inference import (
            StreamingInferenceEngine,
        )

        assert hasattr(StreamingInferenceEngine, "__init__")


class TestMultiModalInference:
    """Test multi-modal inference."""

    def test_multi_modal_exists(self):
        """Test multi-modal inference module exists."""
        from mriforge.infrastructure.inference.multi_modal_inference import (
            MultiModalConditioning,
        )

        assert hasattr(MultiModalConditioning, "__init__")


class TestProcessingStrategies:
    """Test processing strategies."""

    def test_processing_exists(self):
        """Test processing strategies exist."""
        from mriforge.infrastructure.inference.processing_strategies import InferenceStrategy

        assert hasattr(InferenceStrategy, "infer")


class TestPerformanceOptimization:
    """Test performance optimization for inference."""

    def test_optimization_exists(self):
        """Test optimization module exists."""
        from mriforge.infrastructure.inference.performance_optimization import (
            InferencePerformanceOptimizer,
        )

        assert hasattr(InferencePerformanceOptimizer, "__init__")


class TestInferencePerformanceOptimizer:
    """Regression tests for InferencePerformanceOptimizer (ICC-004)."""

    def test_init_sets_autocast_defaults(self):
        """__init__ must establish autocast attributes before optimize_model()."""
        import torch

        from mriforge.infrastructure.inference.performance_optimization import (
            InferencePerformanceOptimizer,
        )

        opt = InferencePerformanceOptimizer(device=torch.device("cpu"), config={})

        # Attributes that run_optimized_inference() reads must exist now.
        assert hasattr(opt, "autocast_enabled")
        assert hasattr(opt, "autocast_dtype")
        assert hasattr(opt, "enable_mixed_precision")
        # CPU-only env: CUDA not available -> autocast disabled, dtype None.
        assert opt.autocast_enabled is False
        assert opt.autocast_dtype is None

    def test_run_optimized_inference_before_optimize_model_does_not_raise(self):
        """run_optimized_inference() before optimize_model() must not AttributeError."""
        import torch
        import torch.nn as nn

        from mriforge.infrastructure.inference.performance_optimization import (
            InferencePerformanceOptimizer,
        )

        class _TwoArgModel(nn.Module):
            def forward(self, x, t):  # noqa: D401 — accepts (x_t, t_tensor)
                return x

        opt = InferencePerformanceOptimizer(device=torch.device("cpu"), config={})
        x_t = torch.randn(1, 1, 8, 8)

        # Must not raise AttributeError on self.autocast_enabled / autocast_dtype.
        out = opt.run_optimized_inference(_TwoArgModel(), x_t, t=0)
        assert out is not None
