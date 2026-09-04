"""Physics Parameter Registry for MRI Sequences.

Provides canonical physics parameters for standard MRI sequences.
Used as fallback when physics metadata not available in batch.

Reference: Bernstein et al. (2004) "Handbook of MRI Pulse Sequences"
"""

from dataclasses import dataclass

import torch


@dataclass
class PhysicsParameters:
    """Physics parameters for MRI sequence.

    Attributes:
        TR: Repetition time (ms)
        TE: Echo time (ms)
        TI: Inversion time (ms) - 0 if not applicable
        B0: Field strength (Tesla)
        contrast_type: Sequence type (T1w, T2w, FLAIR, DWI, etc.)
    """

    TR: float  # milliseconds
    TE: float  # milliseconds
    TI: float  # milliseconds (0 if not IR sequence)
    B0: float  # Tesla
    contrast_type: str

    def to_normalized_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Convert to normalized tensor [0, 1] for network prediction.

        Normalization ranges:
        - TR: [0, 10000] ms
        - TE: [0, 200] ms
        - TI: [0, 3000] ms
        - B0: [0, 7.0] T

        Returns:
            Tensor of shape [4] with normalized values
        """
        # TR ceiling chosen to cover clinical FLAIR (TR up to ~9000 ms at 3T)
        # without saturating the network input.
        TR_MAX = 10000.0
        TE_MAX = 200.0
        TI_MAX = 3000.0
        B0_MAX = 7.0

        return torch.tensor(
            [
                self.TR / TR_MAX,
                self.TE / TE_MAX,
                self.TI / TI_MAX,
                self.B0 / B0_MAX,
            ],
            device=device,
        )

    def to_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Convert to raw tensor [TR, TE, TI, B0] in physical units.

        Returns:
            Tensor of shape [4] with raw physics values
        """
        return torch.tensor([self.TR, self.TE, self.TI, self.B0], device=device)


# Canonical physics parameters for standard sequences
# Reference: Clinical MRI protocols at 3T
PHYSICS_REGISTRY: dict[str, PhysicsParameters] = {
    # === T1-WEIGHTED (SPGR/FLASH) ===
    "T1w": PhysicsParameters(
        TR=8.0,  # Short TR for T1 contrast
        TE=4.0,  # Short TE to minimize T2 weighting
        TI=0.0,  # Not an inversion recovery
        B0=3.0,  # 3 Tesla
        contrast_type="T1w",
    ),
    "T1": PhysicsParameters(  # Alias
        TR=8.0, TE=4.0, TI=0.0, B0=3.0, contrast_type="T1"
    ),
    # === T2-WEIGHTED (SPIN ECHO) ===
    "T2w": PhysicsParameters(
        TR=3000.0,  # Long TR to allow full T1 recovery
        TE=100.0,  # Long TE for T2 contrast
        TI=0.0,  # Not an inversion recovery
        B0=3.0,
        contrast_type="T2w",
    ),
    "T2": PhysicsParameters(  # Alias
        TR=3000.0, TE=100.0, TI=0.0, B0=3.0, contrast_type="T2"
    ),
    # === FLAIR (FLUID ATTENUATED INVERSION RECOVERY) ===
    "FLAIR": PhysicsParameters(
        TR=9000.0,  # Very long TR
        TE=120.0,  # Long TE for T2 weighting
        TI=2500.0,  # TI chosen to null CSF signal at 3T
        B0=3.0,
        contrast_type="FLAIR",
    ),
    # === DIFFUSION-WEIGHTED IMAGING ===
    "DWI": PhysicsParameters(
        TR=3000.0,  # Similar to T2w
        TE=80.0,  # Moderate TE
        TI=0.0,  # Not IR
        B0=3.0,
        contrast_type="DWI",
    ),
    # === PROTON DENSITY ===
    "PD": PhysicsParameters(
        TR=2000.0,  # Moderate TR
        TE=15.0,  # Short TE to minimize T2 weighting
        TI=0.0,
        B0=3.0,
        contrast_type="PD",
    ),
    # === ULTRA-LOW FIELD (64mT) T1-weighted ===
    "ULF_T1w": PhysicsParameters(
        TR=300.0,  # Shorter TR feasible at low field (faster T1 recovery)
        TE=20.0,  # Longer TE due to hardware constraints
        TI=0.0,
        B0=0.064,  # 64 mT
        contrast_type="ULF_T1w",
    ),
    # === ULTRA-LOW FIELD T2-weighted ===
    "ULF_T2w": PhysicsParameters(
        TR=2000.0,  # Long TR
        TE=80.0,  # Long TE for T2 contrast
        TI=0.0,
        B0=0.064,
        contrast_type="ULF_T2w",
    ),
    # === GENERIC ULF (fallback) ===
    "ULF": PhysicsParameters(TR=300.0, TE=20.0, TI=0.0, B0=0.064, contrast_type="ULF"),
    "ULF_64mT": PhysicsParameters(  # Alias
        TR=300.0, TE=20.0, TI=0.0, B0=0.064, contrast_type="ULF_64mT"
    ),
    # === HIGH-FIELD GENERIC (fallback) ===
    "HF": PhysicsParameters(TR=8.0, TE=4.0, TI=0.0, B0=3.0, contrast_type="HF"),
    "HF_3T": PhysicsParameters(  # Alias
        TR=8.0, TE=4.0, TI=0.0, B0=3.0, contrast_type="HF_3T"
    ),
}


def infer_physics_from_name(name: str) -> PhysicsParameters:
    """Infer physics parameters from contrast/filename.

    Args:
        name: Contrast name, filename, or metadata string
            Examples: "T1w", "FLAIR", "ulf_registered_sub01_T2w.nii.gz"

    Returns:
        PhysicsParameters instance

    Raises:
        KeyError: If contrast type cannot be inferred

    Examples:
        >>> params = infer_physics_from_name("T1w")
        >>> params.TR, params.TE
        (8.0, 4.0)

        >>> params = infer_physics_from_name("ulf_registered_sub01_T2w.nii.gz")
        >>> params.contrast_type
        'ULF_T2w'
    """
    name_upper = name.upper()

    # Check for ULF first (more specific)
    if "ULF" in name_upper or "64MT" in name_upper or "LF" in name_upper.split("_"):
        # Check for specific contrast within ULF
        if "T1" in name_upper or "T1W" in name_upper:
            return PHYSICS_REGISTRY["ULF_T1w"]
        elif "T2" in name_upper or "T2W" in name_upper:
            return PHYSICS_REGISTRY["ULF_T2w"]
        else:
            # Generic ULF
            return PHYSICS_REGISTRY["ULF"]

    # Check for high-field contrasts
    if "FLAIR" in name_upper:
        return PHYSICS_REGISTRY["FLAIR"]
    elif "DWI" in name_upper or "DIFFUSION" in name_upper:
        return PHYSICS_REGISTRY["DWI"]
    elif "T2" in name_upper or "T2W" in name_upper:
        return PHYSICS_REGISTRY["T2w"]
    elif "T1" in name_upper or "T1W" in name_upper:
        return PHYSICS_REGISTRY["T1w"]
    elif "PD" in name_upper:
        return PHYSICS_REGISTRY["PD"]
    elif "HF" in name_upper or "3T" in name_upper:
        return PHYSICS_REGISTRY["HF"]

    # Fallback: Generic 3T T1w
    return PHYSICS_REGISTRY["T1w"]


def normalize_physics_tensor(physics: torch.Tensor) -> torch.Tensor:
    """Normalize raw physics tensor [TR, TE, TI, B0] to [0, 1].

    Args:
        physics: Tensor of shape [..., 4] with raw physics values (ms, ms, ms, T)

    Returns:
        Normalized tensor of same shape with values in [0, 1]
    """
    # TR ceiling chosen to cover clinical FLAIR (TR up to ~9000 ms at 3T)
    # without saturating the network input.
    TR_MAX = 10000.0
    TE_MAX = 200.0
    TI_MAX = 3000.0
    B0_MAX = 7.0

    # Clone to avoid in-place modification
    physics_norm = physics.clone()
    physics_norm[..., 0] = physics_norm[..., 0] / TR_MAX  # TR
    physics_norm[..., 1] = physics_norm[..., 1] / TE_MAX  # TE
    physics_norm[..., 2] = physics_norm[..., 2] / TI_MAX  # TI
    physics_norm[..., 3] = physics_norm[..., 3] / B0_MAX  # B0

    return physics_norm


def denormalize_physics_tensor(physics_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize physics tensor from [0, 1] to raw values.

    Args:
        physics_norm: Tensor of shape [..., 4] with normalized values [0, 1]

    Returns:
        Denormalized tensor with raw physics values (ms, ms, ms, T)
    """
    # TR ceiling chosen to cover clinical FLAIR (TR up to ~9000 ms at 3T)
    # without saturating the network input.
    TR_MAX = 10000.0
    TE_MAX = 200.0
    TI_MAX = 3000.0
    B0_MAX = 7.0

    physics = physics_norm.clone()
    physics[..., 0] = physics[..., 0] * TR_MAX  # TR
    physics[..., 1] = physics[..., 1] * TE_MAX  # TE
    physics[..., 2] = physics[..., 2] * TI_MAX  # TI
    physics[..., 3] = physics[..., 3] * B0_MAX  # B0

    return physics
