"""Physics Metadata Registry.

Defines the acquisition parameters for 64mT (Hyperfine) and 3T (Philips) protocols
as extracted from "Paired 64mT and 3T Brain MRI Scans" data descriptor (2025).

Source: van den Broek et al., paired ULF-HF brain MRI dataset.

Used to generate the Ground Truth Physics Vector P_GT = [TR, TE, TI, B0, b-value, flip]
for Physics-Informed Disentanglement.

.. mermaid::

    flowchart LR
        File[Filename] --> Parser{Get Physics Vector}
        Parser -->|64mT/3T| Field[Field Strength]
        Parser -->|T1/T2/FLAIR| Contrast[Contrast Type]

        Field & Contrast --> Key[Registry Key]
        Key --> Registry{Lookup}
        Registry --> Raw[TR, TE, TI, B0]
        Raw --> Norm{Normalize}
        Norm --> Vec[Normalized Vector]

Values are normalized to [0, 1] range based on global max values:
Max TR: ~11000 ms
Max TE: ~200 ms
Max TI: ~2800 ms
Max B0: 3.0 T
"""

import torch

# Normalization Constants (Max Values from Paired Dataset)
MAX_TR = 11000.0  # From 3T FLAIR
MAX_TE = 200.0  # From 64mT T2w (194.8 ms)
MAX_TI = 3000.0  # From 3T FLAIR (2800 ms)
MAX_B0 = 3.0  # 3 Tesla
MAX_B_VALUE = 1000.0  # s/mm² (DWI)
MAX_FLIP = 90.0  # degrees


def normalize_physics(
    tr: float, te: float, ti: float, b0: float, b_value: float = 0.0, flip: float = 90.0
) -> tuple[float, float, float, float, float, float]:
    """Normalize physics parameters to [0, 1].

    Args:
        tr: Repetition time (ms)
        te: Echo time (ms)
        ti: Inversion time (ms), 0 if N/A
        b0: Magnetic field strength (T)
        b_value: b-value for DWI (s/mm²), 0 for non-DWI
        flip: Flip angle (degrees)

    Returns:
        Tuple of 6 normalized values
    """
    return (
        tr / MAX_TR,
        te / MAX_TE,
        ti / MAX_TI,
        b0 / MAX_B0,
        b_value / MAX_B_VALUE,
        flip / MAX_FLIP,
    )


# Raw Parameters from Paired Dataset Tables
# Source: Table 1 (64mT Hyperfine) and Table 2 (3T Philips Low-res)
# Note: TI = 0 if N/A, b_value = 0 for non-DWI
# 64mT = 0.064 T
# 3T = 3.0 T

# Format: "Key_Substring": (TR, TE, TI, B0, b_value, flip_angle)
PHYSICS_REGISTRY: dict[str, tuple[float, float, float, float, float, float]] = {
    # --- 64mT (Low Field - Hyperfine Swoop) ---
    # T1w Standard (Table 1)
    "64mT_T1w": (880.0, 5.03, 340.0, 0.064, 0.0, 90.0),
    # T2w FAST (Table 1)
    "64mT_T2w": (2000.0, 194.8, 0.0, 0.064, 0.0, 90.0),
    # FLAIR (Table 1)
    "64mT_FLAIR": (3500.0, 175.1, 1290.0, 0.064, 0.0, 90.0),
    # DWI (Table 1) - b=0/900 s/mm²
    "64mT_DWI": (1000.0, 76.22, 100.0, 0.064, 900.0, 90.0),
    # --- 3T (High Field - Philips Achieva) ---
    # T1w 3D SPGR Low-res (Table 2) - flip angle ~12° for SPGR
    "3T_T1w": (10.0, 4.58, 0.0, 3.0, 0.0, 12.0),
    # T2w 2D Low-res (Table 2)
    "3T_T2w": (4645.0, 80.0, 0.0, 3.0, 0.0, 90.0),
    # FLAIR 2D Low-res (Table 2)
    "3T_FLAIR": (11000.0, 125.0, 2800.0, 3.0, 0.0, 90.0),
    # DWI 2D Low-res (Table 2) - b=0/1000 s/mm²
    "3T_DWI": (2827.0, 73.0, 0.0, 3.0, 1000.0, 90.0),
}


def get_physics_vector(filename: str) -> torch.Tensor:
    """
    Heuristic determination of physics vector from filename.

    Args:
        filename: Name of the file (e.g., 'sub-01_ses-01_acq-lowres_T1w.nii.gz')

    Returns:
        Tensor of shape [6] containing normalized [TR, TE, TI, B0, b_value, flip].
        Returns zeros if no match found.
    """
    # 1. Determine Field Strength
    is_64mt = (
        "64mT" in filename or "low_field" in filename or "swoop" in filename or "LF" in filename
    )
    is_3t = (
        "3T" in filename or "high_field" in filename or "philips" in filename or "HF" in filename
    )

    # 2. Determine Contrast
    is_t1 = "T1w" in filename or "T1_" in filename
    is_t2 = "T2w" in filename or "T2_" in filename
    is_flair = "FLAIR" in filename or "flair" in filename
    is_dwi = "DWI" in filename or "dwi" in filename or "diffusion" in filename

    key = None

    if is_64mt:
        if is_t1:
            key = "64mT_T1w"
        elif is_t2:
            key = "64mT_T2w"
        elif is_flair:
            key = "64mT_FLAIR"
        elif is_dwi:
            key = "64mT_DWI"
    elif is_3t:
        if is_t1:
            key = "3T_T1w"
        elif is_t2:
            key = "3T_T2w"
        elif is_flair:
            key = "3T_FLAIR"
        elif is_dwi:
            key = "3T_DWI"

    # Check for specific ULF/HF pair naming conventions if generic fails
    # Example: "input" -> 64mT T1 (standard ULF scan), "target" -> 3T T1 (standard HF scan) for Exp 32a
    if key is None:
        # Fallback heuristics for specific experiment structures
        # Assuming Exp 32a (T1 ULF -> T1 HF)
        pass

    if key and key in PHYSICS_REGISTRY:
        params = PHYSICS_REGISTRY[key]
        norm_params = normalize_physics(*params)
        return torch.tensor(norm_params, dtype=torch.float32)

    # Default/Unknown: Return zeros
    # This might signify "learnable" or "unknown" to the model
    return torch.zeros(6, dtype=torch.float32)
