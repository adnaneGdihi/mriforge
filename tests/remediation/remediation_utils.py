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
        "mriforge.infrastructure.physics.fft_ops",
        "mriforge.infrastructure.physics.data_consistency",
        "mriforge.infrastructure.physics.bloch_simulation",
        "mriforge.infrastructure.physics.epg",
        "mriforge.infrastructure.physics.sense",
        "mriforge.infrastructure.physics.motion_simulation",
        "mriforge.infrastructure.physics.noise_simulation",
        "mriforge.infrastructure.physics.sampling",
        "mriforge.infrastructure.physics.nufft_ops",
        "mriforge.infrastructure.physics.b0_mapping",
        "mriforge.infrastructure.physics.qsm",
        "mriforge.infrastructure.physics.coil_sensitivity",
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
        "mriforge.models.diffusion.base_diffusion",
        "mriforge.models.diffusion.cold_diffusion",
        "mriforge.models.diffusion.rician_diffusion",
        "mriforge.models.diffusion.chi_square_diffusion",
        "mriforge.models.diffusion.laplace_diffusion",
        "mriforge.models.diffusion.rectified_flow",
        "mriforge.models.diffusion.score_sde",
        "mriforge.models.diffusion.ddim_sampler",
        "mriforge.models.diffusion.noise_schedules",
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
        "mriforge.models.recurrent.lstm_mri",
        "mriforge.models.recurrent.vrnn",
        "mriforge.models.recurrent.temporal_coherence",
        "mriforge.models.recurrent.recurrent_diffusion",
        "mriforge.models.motion.temporal_motion",
        "mriforge.models.reconstruction.temporal_reconstruction",
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
