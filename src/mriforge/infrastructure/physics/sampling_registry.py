"""The single owner of k-space sampling-pattern names.

Four maps used to name these patterns -- ``sampling._ACCELERATOR_REGISTRY``, the
``MaskType`` enum, two private alias dicts in ``training/utils/kspace_masks.py``,
and a fourth registry in ``clinical_sampling.py`` that no module imported. They
overlapped without agreeing, so a name could mean different things depending on
which one a caller happened to reach, and ``cartesian_vd`` meant nothing at all
on the live path (issues #953, #954).

This module is the one place a pattern name is defined. Everything else asks it.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mriforge.infrastructure.physics.sampling import _ACCELERATOR_REGISTRY


class SamplingPatternRegistry:
    """Canonical sampling-pattern names, plus the aliases the corpus uses."""

    #: alias -> canonical. Every entry exists because arms in the corpus or a
    #: retired vocabulary spell a pattern this way; none is invented for
    #: symmetry. Adding one is a claim that the two names mean the same physics,
    #: so measure before you add (see tools/physics/compare_sampling_hierarchies.py).
    ALIASES: ClassVar[dict[str, str]] = {
        # historic spellings carried over from _PATTERN_TO_ACCELERATION
        "uniform": "uniform_cartesian",
        "gaussian": "variable_density_2d_gaussian",
        "poisson": "poisson_disk",
        "random": "random_cartesian",
        "fractional": "fractional_variable_density",
        "fractional_vd": "fractional_variable_density",
        "multi": "multi_mask",
        # from the retired clinical_sampling registry (issue #953). cartesian_vd
        # is the live one: 19 inprogress arms declare it. The mapping onto
        # variable_density_1d is measured, not assumed -- the two agree on 95% of
        # lines at the accelerated end.
        "cartesian_vd": "variable_density_1d",
        "cartesian_equispaced": "uniform_cartesian",
        "cartesian_random": "random_cartesian",
        "radial_golden": "golden_angle",
        "spiral_vds": "spiral",
    }

    @classmethod
    def list_canonical(cls) -> tuple[str, ...]:
        """Every name that names an accelerator class directly."""
        return tuple(sorted(_ACCELERATOR_REGISTRY))

    @classmethod
    def list_accepted(cls) -> tuple[str, ...]:
        """Every name a config may legally declare."""
        return tuple(sorted(set(cls.list_canonical()) | set(cls.ALIASES)))

    @classmethod
    def resolve(cls, name: str) -> str:
        """Map any accepted spelling onto its canonical name.

        Raises:
            ValueError: If the name is not accepted. It must raise rather than
                fall back to a default: a silently-defaulted pattern trains an
                arm on physics it did not ask for (non-negotiable #3).
        """
        key = str(name).strip().lower()
        if key in _ACCELERATOR_REGISTRY:
            return key
        if key in cls.ALIASES:
            return cls.ALIASES[key]
        raise ValueError(
            f"Unknown sampling pattern {name!r}. Accepted names: {', '.join(cls.list_accepted())}."
        )

    @classmethod
    def accelerator_for(cls, name: str) -> type[Any] | dict[str, Any]:
        """The registry entry a name resolves to."""
        return _ACCELERATOR_REGISTRY[cls.resolve(name)]
