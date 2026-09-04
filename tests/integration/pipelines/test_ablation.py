"""Integration tests for Ablation Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from spectramr.pipelines.ablation import run_ablation_pipeline


@patch("spectramr.pipelines.ablation.Path.exists", return_value=True)
@patch("spectramr.pipelines.ablation.TrainingSettings.from_yaml")
@patch("spectramr.pipelines.ablation.DataBuilder")
def test_run_ablation_pipeline_success(
    mock_builder_cls, mock_settings_cls, mock_exists
):
    """Test successful execution of ablation pipeline."""
    # Mock settings
    mock_settings = MagicMock()
    mock_settings_cls.return_value = mock_settings

    # Mock DataBuilder
    mock_builder = mock_builder_cls.return_value
    mock_loader = MagicMock()
    mock_loader.dataset = MagicMock()
    mock_loader.dataset.__len__.return_value = 100
    mock_loader.__len__.return_value = 10
    mock_loader.__iter__.return_value = iter([{"image": "tensor"}])

    # Return dictionary with ablation loaders
    mock_builder.build_ablation_subsets.return_value.build.return_value = {
        "ablation_10pct": mock_loader,
        "ablation_50pct": mock_loader,
    }

    results = run_ablation_pipeline(Path("config.yaml"))

    assert results["success"] is True
    assert "ablation_10pct" in results["fractions"]
    assert "ablation_50pct" in results["fractions"]
    assert results["fractions"]["ablation_10pct"]["status"] == "completed"


@patch("spectramr.pipelines.ablation.Path.exists", return_value=True)
@patch("spectramr.pipelines.ablation.TrainingSettings.from_yaml")
@patch("spectramr.pipelines.ablation.DataBuilder")
def test_run_ablation_pipeline_failure(
    mock_builder_cls, mock_settings_cls, mock_exists
):
    """Test handling of ablation failure."""
    # Mock settings
    mock_settings = MagicMock()
    mock_settings_cls.return_value = mock_settings

    # Mock DataBuilder failure
    mock_builder = mock_builder_cls.return_value
    mock_loader = MagicMock()
    # Raise error when iterating
    mock_loader.__iter__.side_effect = Exception("Loader failure")

    mock_builder.build_ablation_subsets.return_value.build.return_value = {
        "ablation_10pct": mock_loader,
    }

    results = run_ablation_pipeline(Path("config.yaml"))

    assert results["success"] is False
    assert results["fractions"]["ablation_10pct"]["status"] == "failed"
    assert "Loader failure" in results["fractions"]["ablation_10pct"]["error"]
