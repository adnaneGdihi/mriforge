"""Physics Builder

Creates MRI physics operators (FFT, k-space masks, data consistency, coil sensitivity).
"""

import logging
from typing import Any

from spectramr.config.settings import TrainingSettings

from .base import Builder

logger = logging.getLogger(__name__)


class PhysicsBuilder(Builder):
    """Builds MRI physics components.

    Creates physics operators needed for MRI reconstruction: FFT transforms,
    k-space undersampling masks, data consistency operators, and coil sensitivity
    estimators.

    Attributes:
        _config: Training configuration
        _device: Device where physics components will be used
        _components: Dictionary of created physics components

    Example:
        >>> builder = PhysicsBuilder(config, torch.device("cuda"))
        >>> physics = (builder
        ...     .build_fft_transformer()
        ...     .build_mask_generator()
        ...     .build_data_consistency()
        ...     .build())
        >>> fft = physics["fft"]
    """

    def __init__(self, config: TrainingSettings, device: Any):
        """Initialize PhysicsBuilder.

        Args:
            config: Immutable training configuration
            device: Device for physics components
        """
        self._config = config
        self._device = device
        self._components: dict[str, Any] = {}

    def build_fft_transformer(self) -> "PhysicsBuilder":
        """Create FFT/IFFT transformer.

        Always creates FFT transformer as it's fundamental for MRI reconstruction.

        Returns:
            self: For method chaining
        """
        try:
            from spectramr.infrastructure.physics.fft_ops import FFTTransformer

            self._components["fft"] = FFTTransformer()  # norm="ortho" by default
            logger.info("Created FFT transformer")
        except Exception as e:
            logger.error(f"Failed to create FFT transformer: {e}")
            raise

        return self

    def build_mask_generator(self) -> "PhysicsBuilder":
        """Create k-space undersampling mask generator.

        Creates mask generator for undersampling simulations.

        Returns:
            self: For method chaining
        """
        try:
            from spectramr.infrastructure.training.utils.kspace_masks import (
                KSpaceMaskGenerator,
            )
            from spectramr.models.diffusion.kspace_process import (
                accelerator_kwargs_from_config,
            )

            # Extract acceleration config if available
            accel_config = self._config.undersampling

            # The schedule length lives at `training.diffusion.timesteps`.
            # A second branch used to read `training.num_timesteps` — a path no
            # schema has ever carried, so it could not fire even when the block
            # above was absent. Removed rather than repointed: the flat legacy
            # spelling folds at load, so anything the arm declares arrives here
            # already nested.
            num_timesteps = 1000  # Default
            if (
                self._config.training
                and hasattr(self._config.training, "diffusion")
                and self._config.training.diffusion
            ):
                num_timesteps = getattr(self._config.training.diffusion, "timesteps", 1000)

            kwargs = {}
            default_pattern = "linear"

            if accel_config:
                # Was ``accel_config.model_dump()`` verbatim — every schema
                # field, defaults included, splatted into the accelerator
                # constructor, and ``mask_seed`` never translated to ``seed``.
                # Same defect as the strategy mixin had; same shared allowlist
                # fixes it, so all three generators build alike.
                default_pattern, kwargs = accelerator_kwargs_from_config(accel_config)

            self._components["mask_generator"] = KSpaceMaskGenerator(
                num_timesteps=num_timesteps,
                device=self._device,
                default_pattern=default_pattern,
                accelerator_kwargs=kwargs,
            )
            logger.info(f"Created k-space mask generator (pattern={default_pattern})")
        except Exception as e:
            logger.warning(f"Failed to create mask generator: {e}")

        return self

    def build_data_consistency(self) -> "PhysicsBuilder":
        """[DEPRECATED] Create data consistency operator.

        DC is now integrated directly into the models (e.g. KSpaceColdDiffusionGenerator)
        to support learnable parameters and maintain architectural SSOT.

        Returns:
            self: For method chaining
        """
        logger.debug("PhysicsBuilder.build_data_consistency skipped: DC is model-integrated")
        return self

    def build_coil_sensitivity(self) -> "PhysicsBuilder":
        """No-op. Coil sensitivity is owned by the DATA path, not by this builder.

        This method was unreachable dead code, not a live failure. It read
        ``self._config.physics.parallel_imaging.enabled``, but ``parallel_imaging``
        is not a field on ANY config schema (verified across every class in
        ``config/schemas/``) and ``settings.physics`` is ``None`` unless a config
        supplies a ``physics:`` block. Both guards therefore returned early on
        every call, and the body below them never ran::

            from ...coil_sensitivity import ESPIRiTSensitivity   # never existed
            self._components["coil_sens"] = ESPIRiTSensitivity()

        So the broken import never raised and its ``except Exception`` never
        warned -- the defect was invisible from the outside precisely because the
        dead knob above it kept the dead import unreachable.

        Restoring the import would fix nothing: ``_components["coil_sens"]`` was
        the ONLY reference to that key tree-wide, so a working estimator would be
        constructed and discarded.

        The owner is already elected. ``estimate_smaps`` is called live from
        ``data_pipeline_director.py:290``, smaps reach the strategies on the batch
        as ``_current_smaps`` / ``gen_kwargs["smaps"]``, and
        ``tests/unit/test_coil_sensitivity.py`` records the older
        ``CoilSensitivityEstimationService`` as "deprecated (superseded by
        estimate_smaps)". That is a duplicate owner, so this route goes rather
        than being wired (non-negotiable 17).

        Kept as an explicit no-op rather than deleted because ``director.py:171``
        chains it, and a builder step that vanishes from a fluent chain is harder
        to notice than one that says why it does nothing.
        """
        return self

    def validate(self) -> "PhysicsBuilder":
        """Validate that required physics components are created.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If FFT transformer is missing
        """
        if "fft" not in self._components:
            raise ValueError("FFT transformer is required but not created")

        logger.info(f"Physics validation passed ({len(self._components)} components created)")
        return self

    def build(self) -> dict[str, Any]:
        """Return all physics components.

        Returns:
            Dict[str, Any]: Copy of physics components dictionary

        Raises:
            ValueError: If validation fails
        """
        if "fft" not in self._components:
            raise ValueError("FFT transformer not created. Call build_fft_transformer() first.")

        return dict(self._components)  # Return copy for immutability
