from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from spectramr.data.io_strategies import (
    AutoDetectStrategy,
    DicomStrategy,
    FastMRIH5Strategy,
    IOStrategyFactory,
    NiftiStrategy,
)

# ==============================================================================
# Regular Behavior (Happy Path) & Integration
# ==============================================================================


@patch("h5py.File")
def test_fastmri_h5_strategy_regular(mock_h5_file):
    """Test loading a mock FastMRI H5 file."""
    # Setup mock h5py file structure
    mock_file = MagicMock()
    mock_h5_file.return_value.__enter__.return_value = mock_file

    mock_file.__contains__.side_effect = lambda key: (
        key
        in [
            "kspace",
            "reconstruction_rss",
            "ismrmrd_header",
        ]
    )

    def mock_getitem(key):
        if key == "kspace":
            return [np.zeros((2, 10, 10), dtype=np.complex64)]
        elif key == "reconstruction_rss":
            return [np.ones((10, 10), dtype=np.float32)]
        elif key == "ismrmrd_header":
            mock_hdr = MagicMock()
            mock_hdr.__getitem__.return_value = "mock_xml_header"
            return mock_hdr
        raise KeyError(key)

    mock_file.__getitem__.side_effect = mock_getitem
    mock_file.attrs = {"mock_attr": 42}

    strategy = FastMRIH5Strategy()
    result = strategy.load("mock_fastmri.h5", metadata={"slice_index": 0})

    assert "kspace" in result
    assert "data" in result
    assert "header" in result
    assert "metadata" in result

    assert result["header"] == "mock_xml_header"
    assert result["metadata"]["mock_attr"] == 42
    assert result["metadata"]["slice_index"] == 0
    assert torch.is_tensor(result["kspace"])
    assert torch.is_tensor(result["data"])


@patch("nibabel.load")
def test_nifti_strategy_regular(mock_nib_load):
    """Test loading a NIfTI volume slice."""
    mock_img = MagicMock()
    # Let's mock a standard 3D NIfTI (W, H, D)
    mock_img.get_fdata.return_value = np.ones((20, 20, 10))
    mock_img.affine = np.eye(4)
    mock_img.shape = (20, 20, 10)

    mock_nib_load.return_value = mock_img

    strategy = NiftiStrategy()
    result = strategy.load("mock_brain.nii.gz")

    assert "data" in result
    assert "affine" in result

    tensor = result["data"]
    # Since it was 3D (W, H, D) it gets unsqueezed to (1, W, H, D)
    assert tensor.shape == (1, 20, 20, 10)
    assert torch.all(tensor == 1.0)
    assert np.array_equal(result["affine"], np.eye(4))


@patch("pathlib.Path.is_dir")
@patch("pathlib.Path.is_file")
@patch("pydicom.dcmread")
def test_dicom_strategy_single_file(mock_dcmread, mock_is_file, mock_is_dir):
    """Test loading a single DICOM file."""
    mock_is_file.return_value = True
    mock_is_dir.return_value = False

    mock_ds = MagicMock()
    mock_ds.pixel_array = np.ones((128, 128))
    mock_ds.RescaleSlope = 2.0
    mock_ds.RescaleIntercept = -10.0
    mock_ds.PatientID = "12345"
    mock_dcmread.return_value = mock_ds

    strategy = DicomStrategy()
    result = strategy.load("mock_image.dcm")

    tensor = result["data"]
    # Expected rescale: 1.0 * 2.0 - 10.0 = -8.0
    # Expected shape: (1, 128, 128) because 2D
    assert tensor.shape == (1, 128, 128)
    assert torch.all(tensor == -8.0)
    assert result["metadata"]["PatientID"] == "12345"


# ==============================================================================
# Edge Cases & Special Handling
# ==============================================================================


@patch("pathlib.Path.is_dir")
@patch("pathlib.Path.is_file")
def test_dicom_strategy_not_found(mock_is_file, mock_is_dir):
    mock_is_file.return_value = False
    mock_is_dir.return_value = False

    strategy = DicomStrategy()
    with pytest.raises(FileNotFoundError, match="DICOM path not found"):
        strategy.load("ghost_dir")


@patch("h5py.File")
def test_fastmri_strategy_missing_header_warning(mock_h5_file, caplog):
    """Ensure FastMRI handles missing headers gracefully with a warning."""
    mock_file = MagicMock()
    mock_h5_file.return_value.__enter__.return_value = mock_file

    mock_file.__contains__.side_effect = lambda key: (
        key
        in [
            "kspace",
            "reconstruction_rss",
        ]
    )
    mock_file.__getitem__.side_effect = lambda key: {
        "kspace": np.zeros((10, 10)),
        "reconstruction_rss": np.ones((10, 10)),
    }[key]
    mock_file.attrs = {}

    strategy = FastMRIH5Strategy()
    result = strategy.load("missing_hdr.h5")

    assert "header" not in result
    assert "Missing 'ismrmrd_header'" in caplog.text


@patch.object(FastMRIH5Strategy, "load")
@patch.object(NiftiStrategy, "load")
@patch.object(DicomStrategy, "load")
def test_auto_detect_strategy_routing(mock_dicom, mock_nifti, mock_fastmri):
    """Ensure AutoDetect routes to the right strategy based on extension."""
    strategy = AutoDetectStrategy()

    strategy.load("file.h5")
    mock_fastmri.assert_called_once()

    strategy.load("file.nii.gz")
    mock_nifti.assert_called_once()

    strategy.load("file.dcm")
    mock_dicom.assert_called_once()

    # An unrecognised extension raises (no silent NIfTI fallback — pitfall #9/#10).
    with pytest.raises(ValueError, match=r"[Uu]nrecognised|[Uu]nknown"):
        strategy.load("file.unknown")
    assert mock_nifti.call_count == 1  # not re-routed to NIfTI


def test_io_strategy_factory_registry():
    """Test retrieving known strategies and unknown handling."""
    strat = IOStrategyFactory.get("fastmri")
    assert isinstance(strat, FastMRIH5Strategy)

    strat = IOStrategyFactory.get("nifti")
    assert isinstance(strat, NiftiStrategy)

    with pytest.raises(ValueError, match="Unknown IO Strategy"):
        IOStrategyFactory.get("magic_format")


# ==============================================================================
# Gradient Flow Audit (CRITICAL)
# ==============================================================================


@patch("nibabel.load")
def test_gradient_flow_audit_io_loads_detached(mock_nib_load):
    """
    Ensure that tensors coming off disk strictly NEVER have requires_grad=True.
    Data loaders should produce detached tensors.
    """
    mock_img = MagicMock()
    mock_img.get_fdata.return_value = np.random.rand(10, 10, 10).astype(np.float32)
    mock_img.affine = np.eye(4)
    mock_img.shape = (10, 10, 10)
    mock_nib_load.return_value = mock_img

    strategy = NiftiStrategy()
    result = strategy.load("mock_brain_grad_audit.nii.gz")

    tensor = result["data"]

    # Tensors loaded from IO strategies MUST NOT have gradients natively
    assert tensor.requires_grad is False
    assert tensor.grad_fn is None
