"""Scanner-conditioned degradation profiles for the sim2rank sweep.

A :class:`ScannerProfile` bundles the acquisition parameters that determine WHICH
artifacts a scanner produces and HOW STRONG they are, and derives the per-axis
degradation kwargs from them. It lives in the physics layer (next to the formulas it
drives and the gyromagnetic constant); ``scripts/sim2rank`` injects it into the
operator-library closures, so it never reaches ``core/`` (inward-only deps).

Why the derivation is **theta-aware**, and the trap it avoids
------------------------------------------------------------

Each axis's worst-case physical amount is computed from the scanner parameters, then
**scaled by theta**, so the axis stays monotone in severity *and* differs between
scanners (0.3 T fat-water shift is ~10x smaller than 3 T). Passing the raw scanner
parameters straight to a degrader is the trap that made ``chemical_shift``
theta-INVARIANT: ``_delegate_chemical_shift`` ignores ``shift_pixels`` and computes a
fixed physical shift the moment ``b0_T``/``bw`` are supplied, flattening the severity
ramp every ranker depends on. This module therefore passes a theta-scaled
``shift_pixels`` and never the raw ``b0_T``/``bw``.

Honest scope
------------

Only the axes with a clear physical formula are conditioned (``chemical_shift`` via
the fat-water shift, ``b0``/``b0_geometric_distortion`` via off-resonance ~ B0); every
other axis keeps its default theta ramp (``axis_kwargs`` returns ``{}``). B1-transmit
scales as ``(B0/3)^0.7`` but has no registry axis yet, so it cannot be swept here --
adding a ``b1`` axis is the declared follow-up.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Proton gyromagnetic ratio (Hz / T).
GAMMA_HZ_PER_T = 42.577e6

#: Field strength the derivation treats as the un-scaled reference: at 3 T the
#: conditioned severity equals the default ramp, and only other fields rescale.
_REFERENCE_B0_T = 3.0

#: Undersampling axes -- only present when the protocol accelerates (R > 1).
_UNDERSAMPLING_AXES = frozenset(
    {
        "cartesian_undersamp",
        "vd_undersamp",
        "vd_cartesian",
        "radial_undersamp",
        "partial_fourier",
    }
)

#: Axes produced by an EPI readout specifically (the classic N/2 ghost).
_EPI_ONLY_AXES = frozenset({"nyquist_ghost"})


@dataclass(frozen=True, slots=True)
class ScannerProfile:
    """Acquisition parameters that condition the degradation sweep for one scanner."""

    name: str
    b0_T: float  # noqa: N815 -- B0 field strength in Tesla (MRI notation)
    readout_bw_hz_per_px: float = 250.0
    voxel_size_mm: float | None = None
    tr_ms: float = 0.0
    te_ms: float = 0.0
    ti_ms: float = 0.0
    fa_deg: float = 0.0
    acceleration_R: float = 1.0  # noqa: N815 -- R = parallel-imaging reduction factor
    n_coils: int = 1
    chi_ppm: float = 3.5  # fat-water chemical-shift difference
    fat_fraction: float = 0.1
    sequence: str = "spin_echo"
    #: Explicit artifact allow-list; ``None`` derives it from the protocol.
    active_axes: tuple[str, ...] | None = None

    def chemical_shift_max_px(self) -> float:
        """Worst-case fat-water displacement (pixels) for this scanner.

        ``delta_f = gamma * B0 * chi``; pixels = ``delta_f / readout_bw_per_px``.
        Linear in B0, so 0.3 T is ~1/10 of 3 T at the same bandwidth.
        """
        delta_f_hz = GAMMA_HZ_PER_T * self.b0_T * self.chi_ppm * 1e-6
        return delta_f_hz / max(self.readout_bw_hz_per_px, 1e-6)

    def b0_field_scale(self) -> float:
        """Off-resonance scales ~linearly with B0; 3.0 T is the reference (== 1.0)."""
        return self.b0_T / _REFERENCE_B0_T

    def to_meta(self) -> dict[str, object]:
        """Dependency-free provenance dict (safe to stamp into SimulatorConfig/JSON)."""
        return {
            "name": self.name,
            "b0_T": self.b0_T,
            "readout_bw_hz_per_px": self.readout_bw_hz_per_px,
            "voxel_size_mm": self.voxel_size_mm,
            "acceleration_R": self.acceleration_R,
            "n_coils": self.n_coils,
            "chi_ppm": self.chi_ppm,
            "fat_fraction": self.fat_fraction,
            "sequence": self.sequence,
        }


#: M4Raw: 0.3 T ultra-low-field brain, 4-channel, single fully-sampled acquisition.
M4RAW_03T = ScannerProfile(
    name="m4raw_0.3T",
    b0_T=0.3,
    readout_bw_hz_per_px=200.0,
    voxel_size_mm=1.0,
    acceleration_R=1.0,
    n_coils=4,
    fat_fraction=0.1,
    sequence="spin_echo",
)

#: fastMRI brain: 3 T, ~15-channel, accelerated.
FASTMRI_3T = ScannerProfile(
    name="fastmri_3T",
    b0_T=3.0,
    readout_bw_hz_per_px=250.0,
    voxel_size_mm=0.5,
    acceleration_R=4.0,
    n_coils=15,
    fat_fraction=0.1,
    sequence="spin_echo",
)

PRESETS: dict[str, ScannerProfile] = {p.name: p for p in (M4RAW_03T, FASTMRI_3T)}


def resolve_profile(name: str) -> ScannerProfile:
    """Look up a preset by name; raise on an unknown one (never a silent default)."""
    if name not in PRESETS:
        raise KeyError(f"Unknown scanner profile {name!r}. Known presets: {sorted(PRESETS)}.")
    return PRESETS[name]


def axis_kwargs(profile: ScannerProfile, name: str, theta: float) -> dict[str, float]:
    """Theta-aware degradation kwargs for axis ``name`` under ``profile``.

    Returns ``{}`` for axes this module does not condition, so the degrader keeps its
    default theta ramp. Every returned value is theta-scaled, keeping the axis
    monotone in severity (the theta-invariance trap in the module docstring).
    """
    t = float(theta)
    if name == "chemical_shift":
        return {
            "shift_pixels": t * profile.chemical_shift_max_px(),
            "fat_fraction": profile.fat_fraction,
        }
    if name == "b0":
        return {"strength": t * profile.b0_field_scale()}
    if name == "b0_geometric_distortion":
        # The delegate _delegate_b0_geometric reads ``b0_strength`` (its SeveritySpec
        # param), NOT ``strength``: emitting ``strength`` here is swallowed by **kw and
        # falls back to theta, making the field-strength conditioning inert for this
        # axis (a #16 facade -- 0.3 T and 3 T produced identical output). Emit the
        # axis-correct kwarg name.
        return {"b0_strength": t * profile.b0_field_scale()}
    return {}


def active_axes(profile: ScannerProfile, candidates: Iterable[str]) -> list[str]:
    """The subset of ``candidates`` a scanner with this profile actually produces.

    Gated by protocol facts (which refine the SCANNER/ACQUISITION/PATIENT artifact
    origin taxonomy): chemical shift needs fat present; undersampling needs
    acceleration; the N/2 ghost needs an EPI readout. An explicit
    ``profile.active_axes`` overrides the derivation.
    """
    cand = list(candidates)
    if profile.active_axes is not None:
        allowed = set(profile.active_axes)
        return [a for a in cand if a in allowed]
    out: list[str] = []
    for a in cand:
        if a == "chemical_shift" and profile.fat_fraction <= 0.0:
            continue
        if a in _UNDERSAMPLING_AXES and profile.acceleration_R <= 1.0:
            continue
        if a in _EPI_ONLY_AXES and profile.sequence != "epi":
            continue
        out.append(a)
    return out
