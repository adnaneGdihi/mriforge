"""``undersampling_block_is_applied`` (cohort review 2026-09-02, T0.6).

An ``undersampling:`` block reaches the data on exactly these routes:

* the loader, when ``data.trajectory`` is declared (``PhysicsInformedMasking``
  for Cartesian, the non-Cartesian simulator otherwise; without a trajectory
  ``_build_physics_transform`` is a no-op) or ``data.image_undersampling: true``;
* the strategy, when the resolved class says ``applies_undersampling = True``
  (the twin-driven strategies, ``virtual_fiducial`` and ``vf_admm``, do NOT say so:
  their twin undersamples at ``physics.digital_twin.acceleration`` and reads none
  of this block, so a block on their arms is an error -- VF review 2026-09-03)
  (cold-diffusion schedules, the k-space mixin's batch masks, the VF twin) or
  the block itself declares ``enable_dynamic_mask``;
* the digital twin transform (``physics.digital_twin.apply_as_transform``).

Measured on 2026-09-02: 106 image-domain arms declared the block with none
of those, the builder logged "Dataloader will use acceleration=4x", nothing
used it, and the audit's tag census counted them as accelerated. For an
image-domain dataset that is an error; for a k-space dataset whose strategy
has not declared the flag yet it is reported as unverified (the census step
of the ratchet), because an under-declared masking strategy must not fail an
arm that works today.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    Tier,
    WitnessVerdict,
    register_witness,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

__all__ = ["undersampling_block_is_applied", "undersampling_consumers"]

_NAME = "undersampling_block_is_applied"
_CATEGORY = "advertised_but_inert"

#: ``dataset_type`` values that serve images: the loader's k-space bridge is
#: the only way an undersampling block can touch them.
_IMAGE_ROUTES = frozenset(
    {
        "nifti",
        "nifti_paired",
        "paired_nifti",
        "paired_mri",
        "bids",
        "bids_paired",
        "image",
        "npy_slice",
        "synthetic",
        "mrixfields",
        "png_paired",
        "preprocessed",
        "dicom",
    }
)


def undersampling_consumers(settings: object, strategy_cls: type | None) -> list[str]:
    """Every route through which the block reaches the data, as human-readable names."""
    data = getattr(settings, "data", None)
    block = getattr(settings, "undersampling", None)
    physics = getattr(settings, "physics", None)
    out: list[str] = []
    if getattr(data, "trajectory", None):
        out.append(f"loader: data.trajectory={data.trajectory!r}")
    if getattr(data, "image_undersampling", False):
        out.append("loader: data.image_undersampling")
    if getattr(block, "enable_dynamic_mask", False):
        out.append("strategy: undersampling.enable_dynamic_mask")
    twin = getattr(physics, "digital_twin", None)
    if twin is not None and getattr(twin, "apply_as_transform", False):
        out.append("transform: physics.digital_twin.apply_as_transform")
    if strategy_cls is not None:
        # Scan the MRO: ``class X(BaseTrainingStrategy, KspaceMixin)`` finds the
        # base's ``False`` before the mixin's ``True`` on a plain attribute read,
        # so a mixin-declared masking route would be invisible.
        declaring = [
            base.__name__
            for base in strategy_cls.__mro__
            if base.__dict__.get("applies_undersampling", False) is True
        ]
        if declaring:
            out.append(
                f"strategy: {declaring[0]}.applies_undersampling (via {strategy_cls.__name__})"
            )
    return out


def _resolve_strategy(settings: object) -> type | None:
    try:
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        return TrainingStrategyFactory().get_strategy_class(settings)
    except Exception:  # the witness reports; resolution failures are the factory's checks
        return None


@register_witness(
    _NAME,
    category=_CATEGORY,
    stage=Stage.DECLARE,
    tiers=(Tier.T1,),
    subjects=(Subject.SETTINGS,),
    severity=Severity.ERROR,
    description="A declared undersampling block reaches the data through some route",
    fix_hint=(
        "Declare data.trajectory (loader masking) or data.image_undersampling: true, "
        "use a strategy that masks (applies_undersampling), or delete the block: an "
        "acceleration nothing applies is advertised in every tag and applied nowhere."
    ),
)
def undersampling_block_is_applied(subject: WitnessSubject) -> WitnessVerdict:
    """Error (image data) / advisory (k-space data) when no route applies the block."""
    settings = subject.settings
    block = getattr(settings, "undersampling", None)
    raw_block = (subject.raw_config or {}).get("undersampling") if subject.raw_config else None
    if block is not None and not raw_block:
        # ``undersampling: {}`` -- the schema fills defaults, but the arm declared
        # nothing, so nothing is advertised. The drain deletes the empty block.
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="undersampling block is empty ({}): nothing declared, nothing advertised",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    if block is None:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="no undersampling block declared",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    data = getattr(settings, "data", None)
    dataset_type = str(getattr(data, "dataset_type", "") or "").lower()
    strategy_cls = _resolve_strategy(settings)
    consumers = undersampling_consumers(settings, strategy_cls)
    if consumers:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="applied via " + "; ".join(consumers),
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    if getattr(block, "declares_no_acceleration", False):
        # The explicit fully-sampled declaration (base and max 1.0, see
        # AccelerationConfigSchema.declares_no_acceleration): nothing to apply.
        return WitnessVerdict(
            witness_name=_NAME,
            passed=True,
            message="base and max acceleration 1.0 declare no acceleration; nothing to apply",
            severity=Severity.INFO,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
        )
    accel = getattr(block, "base_acceleration", None)
    if dataset_type in _IMAGE_ROUTES:
        return WitnessVerdict(
            witness_name=_NAME,
            passed=False,
            message=(
                f"undersampling block (base_acceleration={accel}) on image-domain "
                f"dataset_type={dataset_type!r} with no data.trajectory, no "
                "data.image_undersampling, no dynamic mask and a strategy that does not "
                "mask: nothing applies it, and the audit tags count it as accelerated."
            ),
            severity=Severity.ERROR,
            category=_CATEGORY,
            stage=Stage.DECLARE,
            tier=Tier.T1,
            yaml_keys=("undersampling", "data.trajectory", "data.image_undersampling"),
        )
    strategy_name = strategy_cls.__name__ if strategy_cls is not None else "unresolved"
    return WitnessVerdict(
        witness_name=_NAME,
        passed=True,
        message=(
            f"UNVERIFIED: k-space dataset_type={dataset_type!r} with no declared route; "
            f"strategy {strategy_name} does not declare applies_undersampling. If it masks "
            "in-strategy, declare the flag; if not, this block is inert."
        ),
        severity=Severity.INFO,
        category=_CATEGORY,
        stage=Stage.DECLARE,
        tier=Tier.T1,
    )
