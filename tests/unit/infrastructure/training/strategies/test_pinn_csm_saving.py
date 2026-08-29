"""Tests for PINN strategy sensitivity map saving functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from mriforge.infrastructure.training.strategies.pinn_strategy import (
    ConcretePINNSensitivityStrategy,
)


class TestSaveMapAsPng:
    """Tests for _save_map_as_png static method."""

    def test_grayscale_saves_png(self, tmp_path: Path) -> None:
        """Grayscale map should save as 8-bit L-mode PNG."""
        tensor = torch.rand(64, 64)
        out_path = tmp_path / "test_gray.png"

        ConcretePINNSensitivityStrategy._save_map_as_png(tensor, out_path)

        assert out_path.exists()
        with Image.open(out_path) as img:
            assert img.mode == "L"
            assert img.size == (64, 64)

    def test_grayscale_min_max_normalization(self, tmp_path: Path) -> None:
        """Pixel values should be min-max normalized to [0, 255]."""
        tensor = torch.tensor([[0.0, 10.0], [5.0, 20.0]])
        out_path = tmp_path / "test_norm.png"

        ConcretePINNSensitivityStrategy._save_map_as_png(tensor, out_path)

        img = Image.open(out_path)
        arr = np.array(img)
        assert arr.min() == 0
        assert arr.max() == 255

    def test_constant_tensor_produces_black_image(self, tmp_path: Path) -> None:
        """A constant tensor (vmax - vmin < eps) should produce all-zero pixels."""
        tensor = torch.ones(32, 32) * 5.0
        out_path = tmp_path / "test_const.png"

        ConcretePINNSensitivityStrategy._save_map_as_png(tensor, out_path)

        img = Image.open(out_path)
        arr = np.array(img)
        assert arr.max() == 0

    def test_hsv_colormap_saves_rgb(self, tmp_path: Path) -> None:
        """HSV colormap should produce RGB PNG."""
        tensor = torch.rand(32, 32)
        out_path = tmp_path / "test_hsv.png"

        ConcretePINNSensitivityStrategy._save_map_as_png(
            tensor, out_path, colormap="hsv"
        )

        assert out_path.exists()
        with Image.open(out_path) as img:
            assert img.mode == "RGB"

    def test_hsv_fallback_without_matplotlib(self, tmp_path: Path) -> None:
        """If matplotlib is not available, fallback to grayscale."""
        tensor = torch.rand(32, 32)
        out_path = tmp_path / "test_fallback.png"

        with patch.dict("sys.modules", {"matplotlib.cm": None}):
            # The try/except inside should catch ImportError and fallback
            ConcretePINNSensitivityStrategy._save_map_as_png(
                tensor, out_path, colormap="hsv"
            )

        assert out_path.exists()

    def test_gpu_tensor_handled(self, tmp_path: Path) -> None:
        """GPU tensors should be moved to CPU before saving."""
        tensor = torch.rand(16, 16)  # On CPU (can't test CUDA in unit)
        out_path = tmp_path / "test_cpu.png"

        ConcretePINNSensitivityStrategy._save_map_as_png(tensor, out_path)

        assert out_path.exists()


class TestNormalizeForDisplay:
    """Tests for _normalize_for_display static method."""

    def test_returns_zero_one_range(self) -> None:
        """Output should be in [0, 1]."""
        tensor = torch.tensor([[1.0, 5.0], [3.0, 9.0]])
        result = ConcretePINNSensitivityStrategy._normalize_for_display(tensor)
        assert result.min().item() == pytest.approx(0.0)
        assert result.max().item() == pytest.approx(1.0)

    def test_constant_tensor_returns_zeros(self) -> None:
        """A constant tensor should return all zeros."""
        tensor = torch.ones(8, 8) * 42.0
        result = ConcretePINNSensitivityStrategy._normalize_for_display(tensor)
        assert result.max().item() == 0.0


class TestPhaseToRgb:
    """Tests for _phase_to_rgb static method."""

    def test_output_shape(self) -> None:
        """Output should be (3, H, W)."""
        phase = torch.rand(32, 32) * 2 * torch.pi - torch.pi  # [-π, π]
        result = ConcretePINNSensitivityStrategy._phase_to_rgb(phase)
        assert result.shape == (3, 32, 32)

    def test_output_range(self) -> None:
        """Output should be in [0, 1]."""
        phase = torch.rand(16, 16) * 2 * torch.pi - torch.pi
        result = ConcretePINNSensitivityStrategy._phase_to_rgb(phase)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0


class TestLogMapsToTensorboard:
    """Tests for _log_maps_to_tensorboard."""

    @pytest.fixture()
    def strategy_with_logging(self) -> ConcretePINNSensitivityStrategy:
        """Create a minimal mock strategy with a mock logging_service."""
        strategy = object.__new__(ConcretePINNSensitivityStrategy)
        strategy.num_coils = 4
        strategy.logging_service = MagicMock()
        return strategy

    def test_logs_expected_number_of_images(
        self, strategy_with_logging: ConcretePINNSensitivityStrategy
    ) -> None:
        """Should log 4 mag + 4 phase + 1 RSS = 9 entries."""
        magnitude = torch.rand(4, 16, 16)
        phase = torch.rand(4, 16, 16) * 2 * torch.pi - torch.pi
        rss = torch.rand(16, 16)

        strategy_with_logging._log_maps_to_tensorboard(magnitude, phase, rss, step=100)

        strategy_with_logging.logging_service.log_images_batch.assert_called_once()
        images_dict = strategy_with_logging.logging_service.log_images_batch.call_args[
            0
        ][0]

        assert len(images_dict) == 9  # 4 mag + 4 phase + 1 RSS

    def test_tensorboard_tags_follow_convention(
        self, strategy_with_logging: ConcretePINNSensitivityStrategy
    ) -> None:
        """Tags should follow csm/ prefix convention."""
        magnitude = torch.rand(4, 16, 16)
        phase = torch.rand(4, 16, 16)
        rss = torch.rand(16, 16)

        strategy_with_logging._log_maps_to_tensorboard(magnitude, phase, rss, step=0)

        images_dict = strategy_with_logging.logging_service.log_images_batch.call_args[
            0
        ][0]
        tags = list(images_dict.keys())

        assert "csm/magnitude_coil0" in tags
        assert "csm/phase_coil3" in tags
        assert "csm/rss_combined" in tags

    def test_magnitude_tensors_have_correct_shape(
        self, strategy_with_logging: ConcretePINNSensitivityStrategy
    ) -> None:
        """Magnitude images should be (1, 1, H, W) for TensorBoard."""
        magnitude = torch.rand(4, 32, 32)
        phase = torch.rand(4, 32, 32)
        rss = torch.rand(32, 32)

        strategy_with_logging._log_maps_to_tensorboard(magnitude, phase, rss, step=0)

        images_dict = strategy_with_logging.logging_service.log_images_batch.call_args[
            0
        ][0]
        assert images_dict["csm/magnitude_coil0"].shape == (1, 1, 32, 32)

    def test_phase_tensors_have_rgb_shape(
        self, strategy_with_logging: ConcretePINNSensitivityStrategy
    ) -> None:
        """Phase images should be (1, 3, H, W) for TensorBoard (RGB)."""
        magnitude = torch.rand(4, 32, 32)
        phase = torch.rand(4, 32, 32) * 2 * torch.pi - torch.pi
        rss = torch.rand(32, 32)

        strategy_with_logging._log_maps_to_tensorboard(magnitude, phase, rss, step=0)

        images_dict = strategy_with_logging.logging_service.log_images_batch.call_args[
            0
        ][0]
        assert images_dict["csm/phase_coil0"].shape == (1, 3, 32, 32)

    def test_no_logging_service_does_not_crash(self) -> None:
        """If logging_service is None, should return silently."""
        strategy = object.__new__(ConcretePINNSensitivityStrategy)
        strategy.num_coils = 4
        strategy.logging_service = None

        # Should not raise
        strategy._log_maps_to_tensorboard(
            torch.rand(4, 8, 8), torch.rand(4, 8, 8), torch.rand(8, 8), step=0
        )


class TestSaveSensitivityMaps:
    """Integration-style tests for _save_sensitivity_maps."""

    @pytest.fixture()
    def mock_strategy(self, tmp_path: Path) -> ConcretePINNSensitivityStrategy:
        """Create a minimal mock strategy with a fake SIREN model."""
        strategy = object.__new__(ConcretePINNSensitivityStrategy)

        # Minimal attributes needed by _save_sensitivity_maps
        strategy.H = 16
        strategy.W = 16
        strategy.num_coils = 4
        strategy.device = torch.device("cpu")
        strategy._csm_output_dir = tmp_path / "csm"
        strategy._csm_output_dir.mkdir(parents=True, exist_ok=True)

        # Coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing="ij",
        )
        strategy.coords_full = torch.stack([x.flatten(), y.flatten()], dim=-1)

        # Mock SIREN model
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.rand(256, 4),  # S_r: (H*W, num_coils)
            torch.rand(256, 4),  # S_i
        )
        strategy.model = mock_model

        # Mock logging_service for TensorBoard
        strategy.logging_service = MagicMock()

        return strategy

    def test_saves_raw_pt_file(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """Should save a .pt file with complex, magnitude, and phase tensors."""
        mock_strategy._save_sensitivity_maps(epoch=3, step=100)

        pt_files = list(mock_strategy._csm_output_dir.glob("*.pt"))
        assert len(pt_files) == 1
        data = torch.load(pt_files[0], weights_only=False)
        assert "complex" in data
        assert "magnitude" in data
        assert "phase" in data
        assert data["complex"].shape == (4, 16, 16)

    def test_saves_per_coil_magnitude_pngs(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """Should produce magnitude_coil{c}.png for each coil."""
        mock_strategy._save_sensitivity_maps(epoch=1, step=50)

        epoch_dir = mock_strategy._csm_output_dir / "epoch0001_step000050"
        for c in range(4):
            assert (epoch_dir / f"magnitude_coil{c}.png").exists()

    def test_saves_per_coil_phase_pngs(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """Should produce phase_coil{c}.png for each coil."""
        mock_strategy._save_sensitivity_maps(epoch=1, step=50)

        epoch_dir = mock_strategy._csm_output_dir / "epoch0001_step000050"
        for c in range(4):
            assert (epoch_dir / f"phase_coil{c}.png").exists()

    def test_saves_rss_combined(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """Should produce rss_combined.png overview."""
        mock_strategy._save_sensitivity_maps(epoch=0, step=0)

        epoch_dir = mock_strategy._csm_output_dir / "epoch0000_step000000"
        assert (epoch_dir / "rss_combined.png").exists()

    def test_model_set_to_eval_then_train(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """Model should be set to eval() then back to train()."""
        mock_strategy._save_sensitivity_maps(epoch=0)

        mock_strategy.model.eval.assert_called_once()
        mock_strategy.model.train.assert_called_once()

    def test_total_file_count(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """4 coils → 4 mag + 4 phase + 1 RSS = 9 PNGs + 1 .pt = 10 files."""
        mock_strategy._save_sensitivity_maps(epoch=5, step=200)

        pt_files = list(mock_strategy._csm_output_dir.glob("*.pt"))
        epoch_dir = mock_strategy._csm_output_dir / "epoch0005_step000200"
        png_files = list(epoch_dir.glob("*.png"))

        assert len(pt_files) == 1
        assert len(png_files) == 9  # 4 mag + 4 phase + 1 RSS

    def test_calls_tensorboard_logging(
        self, mock_strategy: ConcretePINNSensitivityStrategy
    ) -> None:
        """Should call log_images_batch on logging_service."""
        mock_strategy._save_sensitivity_maps(epoch=0, step=50)

        mock_strategy.logging_service.log_images_batch.assert_called_once()
        images_dict = mock_strategy.logging_service.log_images_batch.call_args[0][0]
        step = mock_strategy.logging_service.log_images_batch.call_args[0][1]

        assert len(images_dict) == 9
        assert step == 50
