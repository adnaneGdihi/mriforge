"""Implicit Neural Representations (INR) for Universal MRI Representation
========================================================================

This package implements disentangled Implicit Neural Representations that
separate anatomical structure from contrast/physics mechanisms.

Key Components:
- DisentangledEncoder: Extracts z_geo (anatomy) and z_phy (physics) codes
- Hypernetwork: Generates SIREN weights conditioned on latent codes
- LocalINR: Patch-based continuous representation of volumes
- TenF-INR: Tensor-factorized INR for dynamic 4D MRI

Mathematical Foundation:
    θ_i = Φ(z_geo^i, z_phy)  # Hypernetwork generates local MLP weights
    I(x) = SIREN_θ(x)        # Query intensity at coordinate x
"""

from .disentangled_encoder import AnatomyEncoder, DisentangledEncoder, PhysicsEncoder
from .hypernetwork import Hypernetwork, HyperSIREN, ShiftModulation
from .local_inr import INRVolume, LocalINR, PatchCoordinator
from .tenf_inr import (
    CPTensorINR,
    SpatialBasisINR,
    TemporalBasisINR,
    TensorFactorizedINR,
)

__all__ = [
    # Encoders
    "AnatomyEncoder",
    "PhysicsEncoder",
    "DisentangledEncoder",
    # Hypernetwork
    "Hypernetwork",
    "ShiftModulation",
    "HyperSIREN",
    # INR
    "LocalINR",
    "INRVolume",
    "PatchCoordinator",
    # TenF-INR
    "TensorFactorizedINR",
    "CPTensorINR",
    "SpatialBasisINR",
    "TemporalBasisINR",
]
