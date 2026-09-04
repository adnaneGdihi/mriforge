"""Regression tests for the 2026-06 audit fixes to ``pinn_strategy``.

Covers three BROKEN_STRATEGY_ONLY findings, all confined to
``src/spectramr/infrastructure/training/strategies/pinn_strategy.py``:

1. Dead ``image_size`` config read replaced with canonical ``data.patch_size``.
2. Broad ``except Exception`` around TensorBoard image logging narrowed to
   ``(OSError, RuntimeError)`` so unexpected programming errors propagate.
3. ``B > 1`` k-space batch now collapses the LEADING batch axis (not the
   trailing W axis), keeping the resulting tensor shaped ``(Coils, H, W)``.

The strategy is exercised via ``object.__new__`` with minimal fakes (mirroring
``test_pinn_csm_saving.py``) so no real config / model / DI container is needed
on the tested paths.
"""

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.strategies.pinn_strategy import (
    ConcretePINNSensitivityStrategy,
)


# ---------------------------------------------------------------------------
# Finding 1: dead image_size read -> canonical patch_size
# ---------------------------------------------------------------------------
class TestPatchSizePrecompute:
    """The dead ``image_size`` probe must be gone and ``patch_size`` consulted."""

    def test_source_no_longer_references_image_size(self) -> None:
        """The non-existent ``data.image_size`` field must not be read.

        Assert the *exact buggy constructs* are gone (a bare ``"image_size"``
        grep is brittle — the explanatory comment legitimately names the field
        it replaced); the canonical ``patch_size`` read must be present.
        """
        src = inspect.getsource(ConcretePINNSensitivityStrategy)
        # The exact buggy assignment + probe must be gone (behaviour is also
        # covered by test_setup_reads_patch_size_w_h_order).
        assert "self.H, self.W = self.config.data.image_size" not in src
        # The canonical patch_size read must be present. Note the canonical path
        # is now `data.sampling.patch_size` -- this literal was left at the
        # pre-decomposition spelling, so the grep failed against source that was
        # already correct (the companion behavioural test below is what actually
        # proves the read; this line only guards the buggy construct's return).
        assert "self.config.data.sampling.patch_size" in src

    def test_setup_reads_patch_size_w_h_order(self, tmp_path) -> None:
        """``_setup_strategy_specific_components`` must seed H/W from patch_size.

        ``patch_size`` is normalized to ``(W, H, D)`` by the data schema, so the
        strategy must set ``self.W = patch_size[0]`` and ``self.H = patch_size[1]``.
        """
        strategy = object.__new__(ConcretePINNSensitivityStrategy)
        strategy.device = torch.device("cpu")

        # Minimal config double exposing only what the setup path touches.
        cfg = MagicMock()
        cfg.physics.pinn.warmup_epochs = 5
        cfg.physics.pinn.alpha_ema = 0.9
        cfg.physics.pinn.map_save_interval = 10
        cfg.losses.pinn.lambda_unit_norm_coil = 1.0
        cfg.losses.pinn.lambda_pde = 1.0
        cfg.losses.pinn.lambda_magnitude_tv = 0.0
        cfg.losses.pinn.lambda_pinn_dc = 1.0
        # patch_size normalized form is (W, H, D). Set on the canonical
        # sub-block: a MagicMock answers `data.patch_size` happily, so the flat
        # spelling left the reader taking a Mock into `torch.linspace`.
        cfg.data.sampling.patch_size = (200, 128, 1)
        cfg.training.output_dir = str(tmp_path)
        cfg.loss_logging.output_dir = None
        strategy.config = cfg

        # ``generator_model`` (base property) resolves ``self.env.generator``.
        fake_model = MagicMock()
        fake_model.num_coils = 4
        fake_env = MagicMock()
        fake_env.generator = fake_model
        strategy.env = fake_env

        # Avoid touching the registry: stub the schema check + loss creation.
        strategy._verify_strategy_config = lambda **_: None  # type: ignore[method-assign]

        import spectramr.infrastructure.training.strategies.pinn_strategy as mod

        orig_create_loss = mod.create_loss
        mod.create_loss = lambda *a, **k: MagicMock(  # type: ignore[assignment]
            to=lambda *aa, **kk: MagicMock()
        )
        try:
            strategy._setup_strategy_specific_components()
        finally:
            mod.create_loss = orig_create_loss

        # W from patch_size[0]=200, H from patch_size[1]=128.
        assert strategy.W == 200
        assert strategy.H == 128
        # Grid must have H*W coordinate rows with W (cols) intact.
        assert strategy.coords_full.shape == (200 * 128, 2)


# ---------------------------------------------------------------------------
# Finding 2: narrowed except around TB image logging
# ---------------------------------------------------------------------------
class TestTensorboardLoggingExceptNarrowed:
    """IO/runtime logging errors are downgraded; programming errors propagate."""

    def _make_strategy(self, raising_exc: BaseException) -> ConcretePINNSensitivityStrategy:
        strategy = object.__new__(ConcretePINNSensitivityStrategy)
        strategy.num_coils = 2
        logging_service = MagicMock()
        logging_service.log_images_batch.side_effect = raising_exc
        strategy.logging_service = logging_service
        return strategy

    def _maps(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude = torch.rand(2, 8, 8)
        phase = torch.rand(2, 8, 8) * 2 * torch.pi - torch.pi
        rss = torch.rand(8, 8)
        return magnitude, phase, rss

    def test_runtime_error_is_swallowed_to_warning(self) -> None:
        """A RuntimeError from the TB backend is downgraded to a warning."""
        strategy = self._make_strategy(RuntimeError("TB writer failed"))
        magnitude, phase, rss = self._maps()

        # Must not raise.
        strategy._log_maps_to_tensorboard(magnitude, phase, rss, step=0)
        strategy.logging_service.log_warning.assert_called_once()

    def test_os_error_is_swallowed_to_warning(self) -> None:
        """An OSError (e.g. disk full) is downgraded to a warning."""
        strategy = self._make_strategy(OSError("disk full"))
        magnitude, phase, rss = self._maps()

        strategy._log_maps_to_tensorboard(magnitude, phase, rss, step=0)
        strategy.logging_service.log_warning.assert_called_once()

    def test_unexpected_error_propagates(self) -> None:
        """An unexpected programming error (ValueError) must NOT be swallowed."""
        strategy = self._make_strategy(ValueError("shape bug"))
        magnitude, phase, rss = self._maps()

        with pytest.raises(ValueError, match="shape bug"):
            strategy._log_maps_to_tensorboard(magnitude, phase, rss, step=0)
        strategy.logging_service.log_warning.assert_not_called()

    def test_attribute_error_propagates(self) -> None:
        """An AttributeError (wiring bug) must NOT be swallowed."""
        strategy = self._make_strategy(AttributeError("missing attr"))
        magnitude, phase, rss = self._maps()

        with pytest.raises(AttributeError, match="missing attr"):
            strategy._log_maps_to_tensorboard(magnitude, phase, rss, step=0)


# ---------------------------------------------------------------------------
# Finding 3: B>1 collapses the batch axis, not W
# ---------------------------------------------------------------------------
class TestBatchDimCollapse:
    """``_compute_losses_impl`` must keep W intact when collapsing B>1."""

    def _make_strategy(
        self, num_coils: int, height: int, width: int
    ) -> ConcretePINNSensitivityStrategy:
        strategy = object.__new__(ConcretePINNSensitivityStrategy)
        strategy.device = torch.device("cpu")
        strategy.num_coils = num_coils
        # Start the precomputed grid at a deliberately WRONG size so the
        # runtime self-correction from k-space shape is what we observe.
        strategy.H, strategy.W = 1, 1
        strategy.coords_full = torch.zeros(1, 2)

        strategy.warmup_epochs = 100  # force the simple DC-only path (epoch=0)
        strategy.alpha_ema = 0.9
        strategy.ema_grad_pde = None
        strategy.lambda_norm = 1.0
        strategy.lambda_pde_max = 1.0
        strategy.lambda_tv = 0.0
        strategy.lambda_dc = 1.0

        # Real SIREN forward contract: coords (N, 2) -> two (N, num_coils).
        def fake_model(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = coords.shape[0]
            return torch.zeros(n, num_coils), torch.zeros(n, num_coils)

        strategy.model = fake_model
        strategy.norm_criterion = lambda *a, **k: torch.tensor(0.0)
        strategy.tv_criterion = lambda *a, **k: torch.tensor(0.0)
        strategy.pde_criterion = lambda *a, **k: torch.tensor(0.0)
        return strategy

    def test_batch_two_keeps_width(self) -> None:
        """A ``(B=2, C, H, W)`` k-space resolves to H/W with W intact (no crash)."""
        num_coils, height, width = 4, 12, 20
        strategy = self._make_strategy(num_coils, height, width)

        # Complex k-space [B, Coils, H, W] with distinct H != W so a W-collapse
        # (the pre-fix bug) would corrupt the spatial reshape and crash.
        kspace = torch.randn(2, num_coils, height, width, dtype=torch.complex64)
        batch = {"measured_kspace": kspace, "mask": None}

        losses = strategy._compute_losses_impl(
            input_batch=kspace,
            target_batch=kspace,
            epoch=0,
            batch=batch,
        )

        # Grid self-corrected to the true spatial dims; W stayed as the column
        # axis (=20), not collapsed.
        assert strategy.H == height
        assert strategy.W == width
        assert "g_total_loss" in losses
        assert torch.isfinite(losses["g_total_loss"]).all()

    def test_batch_one_still_works(self) -> None:
        """``B=1`` continues to collapse to ``(C, H, W)`` correctly."""
        num_coils, height, width = 4, 16, 16
        strategy = self._make_strategy(num_coils, height, width)

        kspace = torch.randn(1, num_coils, height, width, dtype=torch.complex64)
        batch = {"measured_kspace": kspace, "mask": None}

        losses = strategy._compute_losses_impl(
            input_batch=kspace,
            target_batch=kspace,
            epoch=0,
            batch=batch,
        )
        assert strategy.H == height
        assert strategy.W == width
        assert torch.isfinite(losses["g_total_loss"]).all()
