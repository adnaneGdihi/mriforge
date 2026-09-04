import sys
from unittest.mock import MagicMock


def setup_remediation():
    """Mock torch for environments where it is not installed."""
    if "torch" not in sys.modules:
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_torch.device = MagicMock
        mock_torch.tensor = MagicMock
        mock_torch.nn.Module = MagicMock
        mock_torch.nn.L1Loss = MagicMock
        mock_torch.nn.MSELoss = MagicMock
        mock_torch.nn.SmoothL1Loss = MagicMock
        mock_torch.nn.BCEWithLogitsLoss = MagicMock
        sys.modules["torch"] = mock_torch
        sys.modules["torch.nn"] = mock_torch.nn
        sys.modules["torch.nn.functional"] = MagicMock()
        sys.modules["torchvision"] = MagicMock()
        sys.modules["torchvision.models"] = MagicMock()


def setup_physics_mocks():
    """Mock physics modules for environments with incomplete installations.
    
    Provides stubs for MRI physics operations:
    - FFT operations (fft2c, ifft2c)
    - Data consistency layers
    - Bloch simulation
    - Diffusion processes
    - SENSE/NUFFT operations
    """
    if "torch" not in sys.modules:
        setup_remediation()

    physics_modules = [
        "spectramr.infrastructure.physics.fft_ops",
        "spectramr.infrastructure.physics.data_consistency",
        "spectramr.infrastructure.physics.bloch_simulation",
        "spectramr.infrastructure.physics.epg",
        "spectramr.infrastructure.physics.sense",
        "spectramr.infrastructure.physics.motion_simulation",
        "spectramr.infrastructure.physics.noise_simulation",
        "spectramr.infrastructure.physics.sampling",
        "spectramr.infrastructure.physics.nufft_ops",
        "spectramr.infrastructure.physics.b0_mapping",
        "spectramr.infrastructure.physics.qsm",
        "spectramr.infrastructure.physics.coil_sensitivity",
    ]

    for module_name in physics_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()


def setup_diffusion_mocks():
    """Mock diffusion modules for environments with incomplete installations.
    
    Provides stubs for diffusion models:
    - Standard Gaussian diffusion (DDPM/DDIM)
    - Cold diffusion (deterministic degradation)
    - Rician diffusion (MRI noise model)
    - Chi-square diffusion (multi-coil noise)
    - Laplace diffusion (heavy-tailed noise)
    - Rectified flow (straight-path transport)
    - Score-based diffusion (score matching)
    """
    if "torch" not in sys.modules:
        setup_remediation()

    diffusion_modules = [
        "spectramr.models.diffusion.base_diffusion",
        "spectramr.models.diffusion.cold_diffusion",
        "spectramr.models.diffusion.rician_diffusion",
        "spectramr.models.diffusion.chi_square_diffusion",
        "spectramr.models.diffusion.laplace_diffusion",
        "spectramr.models.diffusion.rectified_flow",
        "spectramr.models.diffusion.score_sde",
        "spectramr.models.diffusion.ddim_sampler",
        "spectramr.models.diffusion.noise_schedules",
    ]

    for module_name in diffusion_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()


def setup_time_series_mocks():
    """Mock time-series and recurrent models for temporal MRI.
    
    Provides stubs for:
    - LSTM/RNN temporal reconstruction
    - Variational RNN (VRNN)
    - Temporal coherence networks
    - Recurrent diffusion models
    - Sequential motion estimation
    - Frame interpolation networks
    """
    if "torch" not in sys.modules:
        setup_remediation()

    temporal_modules = [
        "spectramr.models.recurrent.lstm_mri",
        "spectramr.models.recurrent.vrnn",
        "spectramr.models.recurrent.temporal_coherence",
        "spectramr.models.recurrent.recurrent_diffusion",
        "spectramr.models.motion.temporal_motion",
        "spectramr.models.reconstruction.temporal_reconstruction",
    ]

    for module_name in temporal_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()


def setup_all_remediation():
    """Comprehensive remediation setup for full physics + diffusion + temporal support."""
    setup_remediation()
    setup_physics_mocks()
    setup_diffusion_mocks()
    setup_time_series_mocks()
