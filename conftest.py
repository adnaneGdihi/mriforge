import os
import sys

# Initialize remediation if needed (before other imports)
# This allows running tests in environments without torch/cuda
try:
    import torch
except ImportError:
    # Add project root to path if not present
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Remediation module is missing, create mocks
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    mock_torch.__path__ = []  # Mark as package
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device = MagicMock
    mock_torch.randn = MagicMock(return_value=MagicMock())
    mock_torch.manual_seed = MagicMock()
    mock_torch.seed = MagicMock()

    # Create submodules that are also packages
    mock_nn = MagicMock()
    mock_nn.__path__ = []

    mock_optim = MagicMock()
    mock_optim.__path__ = []

    mock_utils = MagicMock()
    mock_utils.__path__ = []

    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_torch.cuda
    sys.modules["torch.backends"] = MagicMock()
    sys.modules["torch.backends.cudnn"] = MagicMock()
    sys.modules["torch.nn"] = mock_nn
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.optim"] = mock_optim
    sys.modules["torch.optim.optimizer"] = MagicMock()
    sys.modules["torch.utils"] = mock_utils
    sys.modules["torch.utils.data"] = MagicMock()
    sys.modules["torch.autograd"] = MagicMock()
    sys.modules["torch.distributed"] = MagicMock()
    sys.modules["torch.multiprocessing"] = MagicMock()
    sys.modules["torch.distributions"] = MagicMock()
    sys.modules["torch.amp"] = MagicMock()

    import torch

import pytest
from _pytest.nodes import Item

_GPU_SKIP_MESSAGE = "Test requires a CUDA-capable GPU."
_MEMORY_SKIP_MESSAGE = "Test marked as memory_hungry requires a CUDA-capable GPU."

# Configure pytest-asyncio
pytest_plugins = ["tests.fixtures.builders"]


# Set asyncio mode to auto to work with existing event loops
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (deselect with '-m \"not asyncio\"')"
    )
    # MambaBlock fails loud without the mamba_ssm CUDA kernel (so a GRU is never
    # silently trained under the "Mamba" label). CI / CPU dev boxes lack the
    # kernel, so allow the wiring-only GRU fallback for the test session.
    # setdefault lets a developer force the raise path with
    # SPECTRAMR_ALLOW_MAMBA_FALLBACK=0 (test_mamba_block deletes it per-test).
    os.environ.setdefault("SPECTRAMR_ALLOW_MAMBA_FALLBACK", "1")


def pytest_runtest_setup(item: Item) -> None:
    if "torch" in globals() and torch.cuda.is_available():
        return

    if item.get_closest_marker("gpu"):
        pytest.skip(_GPU_SKIP_MESSAGE)

    if item.get_closest_marker("memory_hungry"):
        pytest.skip(_MEMORY_SKIP_MESSAGE)
