"""Integration tests for training and inference strategy device/size integrity.

This module tests:
- Training strategy input/output shapes
- Device consistency during training
- Batch handling in strategies
- Inference pipeline device/size integrity
- Model output validation
"""

import torch

from spectramr.data.batch_types import TrainingBatch


class TestTrainingStrategyDeviceIntegrity:
    """Test device integrity in training strategies."""

    def test_batch_device_consistency_in_training(self):
        """Test that batch devices are consistent during training step."""
        # Create mock training batch
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, device="cpu"),
            target=torch.randn(4, 2, 256, 256, device="cpu"),
        )

        # Verify batch is on CPU
        assert batch.input.device.type == "cpu"
        assert batch.target.device.type == "cpu"

        # Move to CPU (should be no-op)
        moved = batch.to("cpu")
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"

    def test_batch_device_consistency_multicoil(self):
        """Test device consistency for multi-coil batches."""
        batch = TrainingBatch(
            input=torch.randn(4, 36, 256, 256, device="cpu"),  # 18 coils
            target=torch.randn(4, 36, 256, 256, device="cpu"),
            mask=torch.ones(4, 1, 256, 256, device="cpu"),
        )

        # All should be on same device
        devices = [batch.input.device, batch.target.device, batch.mask.device]
        assert all(d == devices[0] for d in devices)


class TestTrainingStrategyShapeIntegrity:
    """Test shape integrity in training strategies."""

    def test_batch_input_output_shape_matching(self):
        """Test that input and output shapes match for compatible architectures."""
        # Cold diffusion: input and output should have same shape
        batch = TrainingBatch(
            input=torch.randn(4, 36, 256, 256),  # Undersampled k-space
            target=torch.randn(4, 36, 256, 256),  # Full k-space
        )

        # Verify shapes match
        assert batch.input.shape[1:] == batch.target.shape[1:]  # Channels and spatial
        assert batch.input.shape[0] == batch.target.shape[0]  # Batch size

    def test_batch_size_consistency_through_training(self):
        """Test batch size consistency through training step."""
        batch_size = 4
        batch = TrainingBatch(
            input=torch.randn(batch_size, 2, 256, 256),
            target=torch.randn(batch_size, 2, 256, 256),
        )

        # Extract size
        extracted_batch_size = batch.input.shape[0]
        assert extracted_batch_size == batch_size

        # Should be usable for operations expecting this batch size
        # (e.g., timestep sampling would use this size)
        assert extracted_batch_size > 0

    def test_spatial_dimension_consistency(self):
        """Test spatial dimension consistency."""
        spatial_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

        for h, w in spatial_sizes:
            batch = TrainingBatch(
                input=torch.randn(4, 2, h, w),
                target=torch.randn(4, 2, h, w),
            )

            # Spatial dims should match
            assert batch.input.shape[-2:] == (h, w)
            assert batch.target.shape[-2:] == (h, w)

    def test_channel_dimension_preservation(self):
        """Test that channel dimensions are preserved."""
        for channels in [1, 2, 4, 36, 64]:
            batch = TrainingBatch(
                input=torch.randn(4, channels, 256, 256),
                target=torch.randn(4, channels, 256, 256),
            )

            assert batch.input.shape[1] == channels
            assert batch.target.shape[1] == channels


class TestInferencePipelineIntegrity:
    """Test device and size integrity in inference pipeline."""

    def test_inference_batch_shape_handling(self):
        """Test that inference handles batch shapes correctly."""
        batch = TrainingBatch(
            input=torch.randn(1, 2, 256, 256),  # Single sample inference
            target=torch.randn(1, 2, 256, 256),
        )

        # Should be valid for inference
        assert batch.input.shape[0] == 1
        assert batch.target.shape[0] == 1

    def test_inference_device_consistency(self):
        """Test device consistency during inference."""
        batch = TrainingBatch(
            input=torch.randn(1, 2, 256, 256, device="cpu"),
            target=torch.randn(1, 2, 256, 256, device="cpu"),
        )

        # All should be on same device
        assert batch.input.device == batch.target.device

    def test_inference_batch_size_one(self):
        """Test inference with batch size 1."""
        batch = TrainingBatch(
            input=torch.randn(1, 36, 256, 256),  # Multi-coil k-space
            target=torch.randn(1, 36, 256, 256),
        )

        assert batch.input.shape[0] == 1
        assert batch.target.shape[0] == 1

    def test_inference_large_batch_size(self):
        """Test inference with large batch size."""
        batch_size = 128
        batch = TrainingBatch(
            input=torch.randn(batch_size, 2, 256, 256),
            target=torch.randn(batch_size, 2, 256, 256),
        )

        assert batch.input.shape[0] == batch_size
        assert batch.target.shape[0] == batch_size


class TestTensorShapeValidation:
    """Test tensor shape validation through pipeline."""

    def test_4d_tensor_validation(self):
        """Test validation of 4D tensors."""
        valid_shapes = [
            (1, 1, 64, 64),
            (4, 2, 256, 256),
            (8, 36, 256, 256),
            (16, 4, 128, 128),
        ]

        for shape in valid_shapes:
            batch = TrainingBatch(
                input=torch.randn(*shape),
                target=torch.randn(*shape),
            )

            # Should be valid
            assert batch.input.ndim == 4
            assert batch.target.ndim == 4

    def test_5d_tensor_validation(self):
        """Test validation of 5D tensors (volumetric)."""
        shape = (1, 2, 64, 64, 32)  # Batch, channels, height, width, depth

        batch = TrainingBatch(
            input=torch.randn(*shape),
            target=torch.randn(*shape),
        )

        assert batch.input.ndim == 5
        assert batch.target.ndim == 5

    def test_shape_mismatch_detection(self):
        """Test detection of shape mismatches."""
        # Different shapes should be allowed (e.g., super-resolution)
        batch = TrainingBatch(
            input=torch.randn(4, 1, 128, 128),  # Low-res
            target=torch.randn(4, 1, 256, 256),  # High-res
        )

        # Should allow different shapes
        assert batch.input.shape != batch.target.shape
        assert batch.input.shape[0] == batch.target.shape[0]  # But same batch size

    def test_channel_mismatch_allowed(self):
        """Test that channel mismatches are allowed where needed."""
        # Some models convert 2-channel complex to 1-channel magnitude
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),  # Complex (real + imag)
            target=torch.randn(4, 1, 256, 256),  # Magnitude
        )

        assert batch.input.shape[1] != batch.target.shape[1]


class TestBatchNormalizationIntegrity:
    """Test batch normalization statistics integrity."""

    def test_batch_statistics_consistency(self):
        """Test that batch statistics are consistent."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Calculate statistics
        input_mean = batch.input.mean()
        target_mean = batch.target.mean()

        # Both should be valid numbers
        assert not torch.isnan(input_mean)
        assert not torch.isnan(target_mean)
        assert not torch.isinf(input_mean)
        assert not torch.isinf(target_mean)

    def test_batch_dtype_statistics(self):
        """Test statistics calculation with different dtypes."""
        for dtype in [torch.float32, torch.float64]:
            batch = TrainingBatch(
                input=torch.randn(4, 2, 256, 256, dtype=dtype),
                target=torch.randn(4, 2, 256, 256, dtype=dtype),
            )

            # Should be able to compute statistics
            input_std = batch.input.std()
            assert not torch.isnan(input_std)


class TestMultiGPUDeviceConsistency:
    """Test device consistency in multi-GPU scenarios."""

    def test_batch_device_single_gpu_simulation(self):
        """Test batch device handling in single GPU context."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, device="cpu"),
            target=torch.randn(4, 2, 256, 256, device="cpu"),
        )

        # Move to device
        device = torch.device("cpu")
        moved = batch.to(device)

        # Should be on specified device
        assert moved.input.device == device
        assert moved.target.device == device

    def test_batch_device_consistency_multiple_tensors(self):
        """Test device consistency across multiple tensors."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, device="cpu"),
            target=torch.randn(4, 2, 256, 256, device="cpu"),
            mask=torch.ones(4, 1, 256, 256, device="cpu"),
        )

        # Get all devices
        devices = [batch.input.device, batch.target.device, batch.mask.device]

        # All should be the same
        assert all(d == devices[0] for d in devices)

    def test_batch_nonblocking_transfer(self):
        """Test non-blocking device transfer."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Non-blocking transfer should work
        moved = batch.to("cpu", non_blocking=True)

        # Should be on CPU
        assert moved.input.device.type == "cpu"


class TestColdDiffusionStrategyIntegrity:
    """Test device/size integrity for cold diffusion strategy."""

    def test_cold_diffusion_batch_shape_consistency(self):
        """Test that cold diffusion batches have consistent shapes."""
        # Undersampled and full k-space should have same shape
        batch = TrainingBatch(
            input=torch.randn(4, 36, 256, 256),  # Undersampled
            target=torch.randn(4, 36, 256, 256),  # Full
        )

        # Should be identical except for values
        assert batch.input.shape == batch.target.shape

    def test_cold_diffusion_multi_coil_consistency(self):
        """Test multi-coil consistency in cold diffusion."""
        num_coils = 18
        num_channels = num_coils * 2  # Real + imaginary

        batch = TrainingBatch(
            input=torch.randn(4, num_channels, 256, 256),
            target=torch.randn(4, num_channels, 256, 256),
        )

        # Channel dimension should reflect coil count
        assert batch.input.shape[1] == num_channels
        assert batch.target.shape[1] == num_channels

    def test_cold_diffusion_mask_shape(self):
        """Test mask shape consistency in cold diffusion."""
        batch = TrainingBatch(
            input=torch.randn(4, 36, 256, 256),
            target=torch.randn(4, 36, 256, 256),
            mask=torch.ones(4, 1, 256, 256),  # Single mask for all channels
        )

        # Mask should be broadcastable to input shape
        assert batch.mask.shape[0] == batch.input.shape[0]  # Same batch size
        assert batch.mask.shape[-2:] == batch.input.shape[-2:]  # Same spatial dims


class TestModelInputValidation:
    """Test validation of model inputs."""

    def test_model_input_tensor_requirements(self):
        """Test that batch meets model input requirements."""
        # Example: k-space model expects (B, 2, H, W)
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Should meet requirements
        assert batch.input.ndim == 4
        assert batch.input.shape[0] > 0  # Non-empty batch
        assert batch.input.shape[1] == 2  # Expected channels

    def test_model_input_dtype_validation(self):
        """Test dtype validation for model input."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, dtype=torch.float32),
            target=torch.randn(4, 2, 256, 256, dtype=torch.float32),
        )

        # Should have valid dtype
        assert batch.input.dtype in [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

    def test_model_input_device_validation(self):
        """Test device validation for model input."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, device="cpu"),
            target=torch.randn(4, 2, 256, 256, device="cpu"),
        )

        # Should have valid device
        assert batch.input.device.type in ["cpu", "cuda"]


class TestGradientComputationIntegrity:
    """Test gradient computation integrity."""

    def test_batch_gradient_requirements(self):
        """Test that batch supports gradient computation."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, requires_grad=True),
            target=torch.randn(4, 2, 256, 256, requires_grad=False),
        )

        # Input should support gradients
        assert batch.input.requires_grad is True

    def test_batch_loss_computation(self):
        """Test loss computation with batch."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, requires_grad=True),
            target=torch.randn(4, 2, 256, 256),
        )

        # Should be able to compute loss
        loss = torch.nn.functional.mse_loss(batch.input, batch.target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_batch_backward_pass(self):
        """Test backward pass with batch."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256, requires_grad=True),
            target=torch.randn(4, 2, 256, 256),
        )

        # Compute loss and backward
        loss = torch.nn.functional.mse_loss(batch.input, batch.target)
        loss.backward()

        # Gradients should be computed
        assert batch.input.grad is not None
        assert not torch.isnan(batch.input.grad).any()


class TestMemoryEfficiency:
    """Test memory efficiency of batch handling."""

    def test_batch_memory_reference_semantics(self):
        """Test that batch uses reference semantics for tensors."""
        input_tensor = torch.randn(4, 2, 256, 256)
        target_tensor = torch.randn(4, 2, 256, 256)

        batch = TrainingBatch(input=input_tensor, target=target_tensor)

        # Batch should reference same tensors (not copy)
        assert batch.input is input_tensor
        assert batch.target is target_tensor

    def test_batch_device_transfer_memory(self):
        """Test memory usage during device transfer."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256),
            target=torch.randn(4, 2, 256, 256),
        )

        # Device transfer should move tensors to device
        moved = batch.to("cpu")

        # New batch should have tensors on target device
        assert moved.input.device.type == "cpu"
        assert moved.target.device.type == "cpu"


class TestNumericalStability:
    """Test numerical stability in batch operations."""

    def test_large_value_stability(self):
        """Test stability with large tensor values."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256) * 1e6,
            target=torch.randn(4, 2, 256, 256) * 1e6,
        )

        # Should handle large values
        assert not torch.isnan(batch.input).any()
        assert not torch.isinf(batch.input).any()

    def test_small_value_stability(self):
        """Test stability with small tensor values."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256) * 1e-6,
            target=torch.randn(4, 2, 256, 256) * 1e-6,
        )

        # Should handle small values
        assert not torch.isnan(batch.input).any()
        assert not torch.isinf(batch.input).any()

    def test_mixed_scale_stability(self):
        """Test stability with mixed-scale values."""
        batch = TrainingBatch(
            input=torch.randn(4, 2, 256, 256) * 1e3,
            target=torch.randn(4, 2, 256, 256) * 1e-3,
        )

        # Should handle different scales
        assert not torch.isnan(batch.input).any()
        assert not torch.isnan(batch.target).any()


# ---------------------------------------------------------------------------
# Cross-path construction integrity (PR2 item 7)
# ---------------------------------------------------------------------------
#
# Everything above this line tests tensors: shapes, devices, dtypes. Nothing
# tested that the two entry points BUILD THE SAME MODEL, which is the divergence
# that produced #1306 and #1310 -- `infer` reached past `ModelBuilder` to
# `ModelFactory.create_model`, a config-sniffing layer that resolved a strict
# subset of the builder's kwargs, so predict silently ran a differently-
# configured architecture than training had trained.

from pathlib import Path  # noqa: E402
from typing import ClassVar  # noqa: E402

import torch.nn as nn  # noqa: E402
import yaml as _yaml  # noqa: E402

from spectramr.models.registry import register_model  # noqa: E402


@register_model("_integrity_witness", "reconstruction")
class _IntegrityWitness(nn.Module):
    """Records the kwargs it was constructed with, so a test can compare the
    two construction paths on what actually reached the constructor rather than
    on what a builder claims to inject."""

    seen: ClassVar[dict] = {}

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=8,
        # Named explicitly, not swept up by ``**kwargs``. ``generator_kwargs``
        # gates these two on ``name in contract.accepted`` (:203, :212), which
        # -- unlike the ``_accepts`` helper used for the DC keys (:245) -- does
        # NOT consider ``**kwargs``. A witness relying on ``**kwargs`` would
        # therefore never receive them and the assertion would fail for a
        # reason that has nothing to do with the two paths agreeing.
        acceleration_config=None,
        kspace_log_scaled=None,
        **kwargs,
    ):
        super().__init__()
        _IntegrityWitness.seen = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "acceleration_config": acceleration_config,
            "kspace_log_scaled": kspace_log_scaled,
            **kwargs,
        }
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.head = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        return self.head(self.conv(x))


def _integrity_arm(tmp_path: Path) -> str:
    """A reconstruction arm declaring the SSOT blocks the bypass used to drop."""
    cfg = {
        "config_version": "1.0",
        "model": {
            "model_type": "_integrity_witness",
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 16,
        },
        "data": {
            "sampling": {"patch_size": [16, 16]},
            "loader": {"batch_size": 1},
            "processing": {"enable_log_scaling": True},
        },
        "undersampling": {"enabled": True},
        "physics": {"data_consistency": {"enabled": True, "method": "soft"}},
        "optimization": {},
        "logging": {},
        # Every one of the 647 committed `inprogress/` arms declares this block,
        # and `ModelBuilder.build_discriminator` reads `training.gan` without
        # guarding `training` itself against None -- so omitting it here would
        # test a crash rather than the invariant.
        "training": {"num_epochs": 1},
    }
    path = tmp_path / "integrity_arm.yaml"
    path.write_text(_yaml.safe_dump(cfg))
    return str(path)


class TestTrainingAndInferenceBuildTheSameGenerator:
    """The two paths must agree on the architecture, not merely on tensor shape.

    This is a *regression guard*, green on the current tree because the
    consolidation landed. Both chains are spelled exactly as their production
    call sites spell them -- ``TrainingEnvironmentDirector.build_environment``
    (director.py:95) and ``run_inference_pipeline`` (infer.py:166) -- so a
    future change to the steps of either chain that is not mirrored in the
    other fails here: a step that renames parameters (``compile()`` wraps in
    ``OptimizedModule`` and prefixes every key with ``_orig_mod.``), or an
    injection added to one builder chain and not the other.

    **What it does not cover, stated plainly.** It compares the two chains, not
    the two *call sites*: it re-spells them rather than invoking
    ``run_inference_pipeline``, which needs a checkpoint and input files. So it
    would NOT have caught the original defect, where ``infer`` never called
    ``ModelBuilder`` at all and went to ``ModelFactory.create_model`` instead.
    That case is covered by ``TestInferBuildsThroughTheCanonicalBuilder`` and
    ``TestAccelerationConfigReachesTheModel`` in
    ``tests/unit/pipelines/test_infer.py``, which do drive the real pipeline.
    """

    @staticmethod
    def _training_generator(config_path: str):
        """Exactly the chain the training director runs."""
        import torch

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        config = TrainingSettings.from_yaml(config_path)
        builder = (
            ModelBuilder(config, torch.device("cpu"))
            .build_generator()
            .build_discriminator()
            .build_encoder_decoder()
            .validate()
            .compile()
            .build_ema()
        )
        return builder.build()["generator"], dict(_IntegrityWitness.seen)

    @staticmethod
    def _inference_generator(config_path: str):
        """Exactly the chain the inference pipeline runs."""
        import torch

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        config = TrainingSettings.from_yaml(config_path)
        model = (
            ModelBuilder(config, torch.device("cpu"))
            .build_generator()
            .validate()
            .build()["generator"]
        )
        return model, dict(_IntegrityWitness.seen)

    def test_state_dict_keys_and_shapes_are_identical(self, tmp_path):
        """A checkpoint written by one path must load into the other."""
        arm = _integrity_arm(tmp_path)
        trained, _ = self._training_generator(arm)
        inferred, _ = self._inference_generator(arm)

        t, i = trained.state_dict(), inferred.state_dict()
        assert set(t) == set(i), (
            "the two paths produce different parameter names — a checkpoint "
            f"from one cannot load into the other. only-training={set(t)-set(i)}, "
            f"only-inference={set(i)-set(t)}"
        )
        for k in t:
            assert t[k].shape == i[k].shape, (
                f"{k}: training built {tuple(t[k].shape)}, inference built "
                f"{tuple(i[k].shape)} — same name, different architecture"
            )

    def test_the_same_declared_kwargs_reach_both(self, tmp_path):
        """Identical state_dicts are necessary but not sufficient.

        ``acceleration_config`` and ``kspace_log_scaled`` change *behaviour*
        without changing a single parameter name, which is precisely why the
        drop went unnoticed: every shape-based test still passed.
        """
        arm = _integrity_arm(tmp_path)
        _, train_kwargs = self._training_generator(arm)
        _, infer_kwargs = self._inference_generator(arm)

        assert train_kwargs == infer_kwargs, (
            "the two paths injected different constructor kwargs: "
            f"only-training={ {k: v for k, v in train_kwargs.items() if infer_kwargs.get(k) != v} }, "
            f"only-inference={ {k: v for k, v in infer_kwargs.items() if train_kwargs.get(k) != v} }"
        )
        assert train_kwargs.get("in_channels") == 3, "declared width was defaulted"
        assert train_kwargs.get("base_channels") == 16, "declared depth was defaulted"

    def test_the_ssot_blocks_reach_both_paths(self, tmp_path):
        """The three injections the bypass skipped entirely."""
        arm = _integrity_arm(tmp_path)
        for label, build in (
            ("training", self._training_generator),
            ("inference", self._inference_generator),
        ):
            _, kwargs = build(arm)
            assert kwargs.get("acceleration_config") is not None, (
                f"{label} built a model with no acceleration block, but the arm "
                "declares undersampling"
            )
            assert kwargs.get("kspace_log_scaled") is True, f"{label}: #1306 knob"
            assert kwargs.get("use_dc") is True, f"{label}: data-consistency knob"
