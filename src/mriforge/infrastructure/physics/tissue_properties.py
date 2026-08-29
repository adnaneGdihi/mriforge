"""Tissue Properties Registry for Bloch Simulation.

Provides physics-based T1, T2, and Proton Density (PD) values for common brain tissues
at various magnetic field strengths (B0).

Values are approximate and based on literature for:
- White Matter (WM)
- Gray Matter (GM)
- Cerebrospinal Fluid (CSF)

References:
- Bottomley et al. "A review of normal tissue hydrogen NMR relaxation times..." (1984)
- Stanisz et al. "T1, T2 relaxation and magnetization transfer in tissue at 3T" (2005)
- Hyperfine Swoop clinical documentation (64mT)
"""

from dataclasses import dataclass
from enum import Enum

import torch


class TissueType(Enum):
    """TissueType class."""

    BACKGROUND = 0
    CSF = 1
    GM = 2
    WM = 3
    # Add more as needed (FAT, MUSCLE, etc.)


@dataclass(frozen=True)
class RelaxationProperties:
    """RelaxationProperties class."""

    t1: float  # ms
    t2: float  # ms
    pd: float  # Relative Proton Density [0-1]


class TissueFieldStrength(Enum):
    """String-tag enum keying the :class:`TissuePropertyRegistry` lookup table.

    Distinct from :class:`mriforge.infrastructure.physics.field_simulation.FieldStrength`,
    which carries a numeric T-value used in off-resonance physics. The two
    enums are kept separate per CLAUDE.md pitfall #12: operator-dispatch
    enums co-located with their dispatch table need not migrate to
    ``src/config/schemas/enums.py`` as long as they are never YAML-facing.
    See ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5.
    """

    mT64 = "64mT"
    T1_5 = "1.5T"
    T3_0 = "3.0T"


# Backward-compat alias — the enum was renamed from ``FieldStrength`` to
# ``TissueFieldStrength`` in an earlier physics refactor (to avoid
# collision with the numeric-T-value enum in ``field_simulation``).
# The previous ``FieldStrength`` alias was deliberately removed —
# ``tests/unit/infrastructure/physics/test_field_strength_enum_split.py::
# test_tissue_field_strength_is_not_re_exported_as_field_strength``
# enforces the split, since the whole point of the rename was to make
# the two enums distinguishable at import time. Use
# ``TissueFieldStrength`` (here) vs ``FieldStrength`` (in
# ``field_simulation`` — numeric tesla values) explicitly.


class TissuePropertyRegistry:
    """Registry for looking up tissue properties by field strength."""

    # Dictionary mapping TissueFieldStrength -> {TissueType -> Properties}
    _REGISTRY: dict[str, dict[TissueType, RelaxationProperties]] = {
        TissueFieldStrength.mT64.value: {
            TissueType.BACKGROUND: RelaxationProperties(t1=0.0, t2=0.0, pd=0.0),
            TissueType.CSF: RelaxationProperties(t1=4000.0, t2=2000.0, pd=1.0),
            # Low field: Short T1s
            TissueType.GM: RelaxationProperties(t1=500.0, t2=100.0, pd=0.8),
            TissueType.WM: RelaxationProperties(t1=350.0, t2=80.0, pd=0.7),
        },
        TissueFieldStrength.T1_5.value: {
            TissueType.BACKGROUND: RelaxationProperties(t1=0.0, t2=0.0, pd=0.0),
            TissueType.CSF: RelaxationProperties(t1=4000.0, t2=2000.0, pd=1.0),
            # Mid field
            TissueType.GM: RelaxationProperties(t1=950.0, t2=100.0, pd=0.8),
            TissueType.WM: RelaxationProperties(t1=600.0, t2=80.0, pd=0.7),
        },
        TissueFieldStrength.T3_0.value: {
            TissueType.BACKGROUND: RelaxationProperties(t1=0.0, t2=0.0, pd=0.0),
            TissueType.CSF: RelaxationProperties(t1=4000.0, t2=2000.0, pd=1.0),
            # High field: Long T1s
            TissueType.GM: RelaxationProperties(t1=1350.0, t2=110.0, pd=0.8),
            TissueType.WM: RelaxationProperties(t1=850.0, t2=80.0, pd=0.7),
        },
    }

    @classmethod
    def get_properties(
        cls, field_strength: str | TissueFieldStrength = TissueFieldStrength.mT64
    ) -> dict[TissueType, RelaxationProperties]:
        """Get all tissue properties for a specific field strength."""
        if isinstance(field_strength, TissueFieldStrength):
            fs_key = field_strength.value
        else:
            fs_key = field_strength

        if fs_key not in cls._REGISTRY:
            raise ValueError(f"Unknown field strength: {fs_key}")

        return cls._REGISTRY[fs_key]

    @classmethod
    def get_map_tensor(
        cls, segmentation_map: torch.Tensor, field_strength: str = "64mT"
    ) -> torch.Tensor:
        """Convert a segmentation map (integers) to a (B, 3, H, W) property map.

        Args:
            segmentation_map: (B, 1, H, W) integer tensor (0=BG, 1=CSF, 2=GM, 3=WM)
            field_strength: "64mT", "1.5T", "3.0T"

        Returns:
            maps: (B, 3, H, W) float tensor corresponding to (PD, T1, T2)
        """
        props = cls.get_properties(field_strength)

        B, _, H, W = segmentation_map.shape
        device = segmentation_map.device

        # Output tensor: 3 channels (PD, T1, T2)
        out_maps = torch.zeros(B, 3, H, W, device=device)

        # Naive implementation: iterate tissues and apply masks
        # Ideally simpler with scatter but this is readable and reasonably fast for 4 classes
        seg_flat = segmentation_map.long()

        for tissue_type, p in props.items():
            mask = seg_flat == tissue_type.value

            # Fill PD (channel 0)
            out_maps[:, 0:1][mask] = p.pd

            # Fill T1 (channel 1)
            out_maps[:, 1:2][mask] = p.t1

            # Fill T2 (channel 2)
            out_maps[:, 2:3][mask] = p.t2

        return out_maps
