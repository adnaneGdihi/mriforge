from . import (
    implementations,  # Import to register operators
    registry,
)

# Grand Unified VF pipeline physics
from .bloch_simulation import ContinuousBlochODE
from .coil_sensitivity import (
    coil_combine_rss,
    coil_combine_sense,
    estimate_csm_rss,
    load_csm_from_file,
    validate_csm,
)
from .data_consistency import SimpleDataConsistency
from .dc_navigator import DCNavigatorExtractor
from .differentiable_bloch import DifferentiableBlochLayer, SPGRBlochSimulator
from .digital_twin_simulator import CornerFiducialEmbedder, DigitalTwinSimulator
from .engine import PhysicsEngine, get_physics_engine, image_to_kspace, kspace_to_image
from .fft_ops import (
    NUFFTOperators,
    apply_mask,
    fft2c,
    fft2c_masked,
    get_fft_ops,
    ifft2c,
    ifft2c_masked,
    sense_adjoint,
    sense_forward,
)
from .kinematic_forward import KinematicForwardOperator
from .multi_acquisition import (
    MultiAcquisitionSimulator,
    invert_afi,
    invert_bloch_siegert,
    invert_double_angle,
    invert_dual_echo,
    invert_mrf,
)
from .multi_physics_bloch import MultiPhysicsBlochLayer
from .nufft_ops import GoldenAngleTrajectory, NUFFTForwardModel
from .null_space_erasure import NullSpaceMarkerEraser
from .phase_steganography import PhaseSteganographicMarker
from .spamm_grid import DiracNotchFilter, SPAMMGridInjector
from .subpixel_registration import estimate_subpixel_shifts, fourier_shift
from .virtual_fiducial import MotionTrajectory, VirtualFiducial

__all__ = [
    "estimate_subpixel_shifts",
    "fourier_shift",
    # Engine facade (preferred)
    "PhysicsEngine",
    "get_physics_engine",
    "image_to_kspace",
    "kspace_to_image",
    # FFT operations
    "NUFFTOperators",
    "apply_mask",
    "fft2c",
    "fft2c_masked",
    "get_fft_ops",
    "ifft2c",
    "ifft2c_masked",
    # SENSE operations
    "sense_adjoint",
    "sense_forward",
    # Coil sensitivity
    "coil_combine_rss",
    "coil_combine_sense",
    "create_synthetic_csm",
    "estimate_csm_rss",
    "load_csm_from_file",
    "validate_csm",
    # Data consistency
    "SimpleDataConsistency",
    # Differentiable Bloch forward model
    "DifferentiableBlochLayer",
    "SPGRBlochSimulator",
    # Faithful multi-acquisition synthesis (VF field-mapping arms)
    "MultiAcquisitionSimulator",
    "invert_afi",
    "invert_bloch_siegert",
    "invert_double_angle",
    "invert_dual_echo",
    "invert_mrf",
    "MultiPhysicsBlochLayer",
    # Motion correction
    "KinematicForwardOperator",
    "VirtualFiducial",
    "MotionTrajectory",
    # Digital Twin simulator
    "DigitalTwinSimulator",
    "CornerFiducialEmbedder",
    # K-space trajectory generators
    "kspace_trajectory_generator",
    # Registry
    "implementations",
    "registry",
    # Grand Unified VF pipeline physics
    "ContinuousBlochODE",
    "DCNavigatorExtractor",
    "DiracNotchFilter",
    "GoldenAngleTrajectory",
    "NUFFTForwardModel",
    "NullSpaceMarkerEraser",
    "PhaseSteganographicMarker",
    "SPAMMGridInjector",
]
