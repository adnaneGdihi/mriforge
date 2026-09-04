"""Phase 1: Physics Builders for MRI Reconstruction Components

Implements fluent builders for creating MRI physics operators:
- FFTBuilder: Creates FFT/IFFT transformers
- MaskBuilder: Creates k-space undersampling masks
- DataConsistencyBuilder: Creates data consistency projectors
- PhysicsBuilder: Aggregates all physics operators

Each builder provides a fluent API for configuration and instantiation.
"""

import logging
from typing import Any

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)
from spectramr.infrastructure.builders.core import FluentBuilder
from spectramr.infrastructure.physics.data_consistency import SimpleDataConsistency
from spectramr.infrastructure.physics.fft_ops import FFTTransformer
from spectramr.infrastructure.physics.sampling import MaskGenerator

logger = logging.getLogger(__name__)


class FFTBuilder(FluentBuilder[FFTTransformer]):
    """Builder for creating FFT operators for k-space transformations.

    Configures normalized FFT/IFFT operations with proper k-space centering.

    Example:
        >>> builder = FFTBuilder(config)
        >>> fft = (builder
        ...     .with_norm("ortho")
        ...     .with_centering(True)
        ...     .validate()
        ...     .build())
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize FFT builder.

        Args:
            config: Training configuration
        """
        config = ctx.config
        super().__init__()
        self._config = config
        self._norm: str = "ortho"
        self._centering: bool = True
        self._device: str = "cuda"
        logger.info("FFTBuilder initialized")

    def with_norm(self, norm: str) -> "FFTBuilder":
        """Set FFT normalization.

        Args:
            norm: Normalization mode ('ortho' or '')

        Returns:
            self for chaining
        """
        # FFTTransformer is hardcoded to norm="ortho" (the physics SSOT); any
        # other value would be silently dropped at build() — an inert knob
        # (#15). Reject it instead of accepting-then-ignoring.
        if norm != "ortho":
            raise ValueError(
                f"Unsupported FFT norm {norm!r}: FFTTransformer only implements "
                "'ortho' (centered, norm-preserving)."
            )
        self._norm = norm
        return self

    def with_centering(self, centering: bool = True) -> "FFTBuilder":
        """Enable/disable k-space centering.

        Args:
            centering: Whether to center k-space

        Returns:
            self for chaining
        """
        # This builder wires only the CENTRED entry points (``fft2c`` / ``ifft2c``
        # / ``fftnc`` / ``ifftnc``), which is the convention non-negotiable 2
        # requires for complex MRI k-space. ``FFTTransformer`` does also expose an
        # uncentered family, but it is reachable only by naming it explicitly
        # (``fft2_uncentered`` & co, #1350) -- never by flipping a builder flag.
        # So ``centering=False`` has nothing to build: raise rather than accept a
        # knob that would be silently ignored (non-negotiable 8, pitfall #15).
        if centering is not True:
            raise ValueError(
                "Unsupported: this builder wires only the centred FFT entry "
                "points; centering must be True. The uncentered family is "
                "reachable only by calling FFTTransformer.fft2_uncentered() "
                "and friends by name."
            )
        self._centering = centering
        return self

    def with_device(self, device: str) -> "FFTBuilder":
        """Set device for FFT operations.

        Args:
            device: Device name ('cuda', 'cpu')

        Returns:
            self for chaining
        """
        self._device = device
        return self

    def validate(self) -> "FFTBuilder":
        """Validate builder state.

        Returns:
            self for chaining
        """
        super().validate()
        return self

    def build(self) -> FFTTransformer:
        """Build and return FFT transformer.

        Returns:
            Configured FFT operator

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            fft = FFTTransformer(
                device=self._device,
            )

            self._product = fft
            logger.info(f"FFT builder built: (norm={self._norm}, centering={self._centering})")

            return fft

        except Exception as e:
            raise RuntimeError(f"Failed to create FFT operator: {e}") from e


class MaskBuilder(FluentBuilder[torch.Tensor]):
    """Builder for creating k-space undersampling masks.

    Generates masks for Cartesian, radial, and non-Cartesian sampling patterns.

    Example:
        >>> builder = MaskBuilder(config, device="cuda")
        >>> mask = (builder
        ...     .with_shape((320, 256))
        ...     .with_sampling_pattern("cartesian")
        ...     .with_acceleration(4)
        ...     .with_center_fraction(0.08)
        ...     .validate()
        ...     .build())
    """

    def __init__(self, config: TrainingSettings, device: str = "cuda"):
        """Initialize mask builder.

        Args:
            config: Training configuration
            device: Device for mask creation
        """
        super().__init__()
        self._config = config
        self._device = torch.device(device)
        self._shape: tuple | None = None
        # Must be a real ``MaskType`` value — ``generate_mask`` does
        # ``MaskType(pattern)`` and ``"cartesian"`` is NOT a member (the closest
        # members are ``uniform_cartesian`` / ``cartesian_lines``), so a bogus
        # default crashes the first shaped build.
        self._sampling_pattern: str = "uniform_cartesian"
        self._acceleration: int = 4
        self._center_fraction: float = 0.08
        self._seed: int | None = None
        logger.info("MaskBuilder initialized")

    def with_shape(self, shape: tuple) -> "MaskBuilder":
        """Set k-space dimensions.

        Args:
            shape: K-space shape (height, width)

        Returns:
            self for chaining
        """
        if len(shape) != 2:
            raise ValueError(f"Shape must be 2D: {shape}")
        self._shape = shape
        return self

    def with_sampling_pattern(self, pattern: str) -> "MaskBuilder":
        """Set sampling pattern type.

        Args:
            pattern: Type ('cartesian', 'radial', 'spiral', etc.)

        Returns:
            self for chaining
        """
        # ``cartesian`` is accepted as a friendly alias for the real MaskType
        # member ``uniform_cartesian`` so existing call sites keep working, but
        # it is normalised before storage so ``MaskType(pattern)`` never sees an
        # invalid value downstream (pitfall #9: no silent crash on the default).
        aliases = {"cartesian": "uniform_cartesian"}
        valid_patterns = (
            "cartesian",
            "uniform_cartesian",
            "radial",
            "spiral",
            "variable_density",
        )
        if pattern not in valid_patterns:
            raise ValueError(f"Invalid pattern: {pattern}")
        self._sampling_pattern = aliases.get(pattern, pattern)
        return self

    def with_acceleration(self, acceleration: int) -> "MaskBuilder":
        """Set undersampling acceleration factor.

        Args:
            acceleration: Acceleration rate (2, 4, 8, etc.)

        Returns:
            self for chaining
        """
        if acceleration <= 1:
            raise ValueError(f"Invalid acceleration: {acceleration}")
        self._acceleration = acceleration
        return self

    def with_center_fraction(self, fraction: float) -> "MaskBuilder":
        """Set center k-space fully sampled fraction.

        Args:
            fraction: Fraction of center to sample (0.0-1.0)

        Returns:
            self for chaining
        """
        if not (0 < fraction <= 1):
            raise ValueError(f"Invalid center_fraction: {fraction}")
        self._center_fraction = fraction
        return self

    def with_seed(self, seed: int) -> "MaskBuilder":
        """Set random seed for reproducibility.

        Args:
            seed: Random seed

        Returns:
            self for chaining
        """
        self._seed = seed
        return self

    def validate(self) -> "MaskBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._shape is None:
            raise ValueError("K-space shape must be specified")

        return self

    def build(self) -> torch.Tensor:
        """Build and return k-space mask.

        Returns:
            Undersampling mask

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            mask_gen = MaskGenerator(seed=self._seed)

            # Generate mask based on pattern. NOTE: the kwarg is
            # ``acceleration_factor`` — passing ``acceleration`` lets it fall into
            # ``**kwargs`` where it is ignored and the rate silently pins at the
            # 2.0 default (pitfall #15: a configured knob becomes a no-op).
            mask = mask_gen.generate_mask(
                shape=self._shape,
                acceleration_factor=self._acceleration,
                center_fraction=self._center_fraction,
                mask_type=self._sampling_pattern,
            )

            self._product = mask
            logger.info(
                f"Mask built: {self._sampling_pattern} "
                f"(shape={self._shape}, accel={self._acceleration})"
            )

            return mask

        except Exception as e:
            raise RuntimeError(f"Failed to create mask: {e}") from e


class DataConsistencyBuilder(FluentBuilder[SimpleDataConsistency]):
    """Builder for creating data consistency operators.

    Data consistency projects reconstructions back to measured k-space values.

    Example:
        >>> builder = DataConsistencyBuilder(config)
        >>> dc = (builder
        ...     .with_method("hard")
        ...     .with_weight(1.0)
        ...     .with_device("cuda")
        ...     .validate()
        ...     .build())
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize data consistency builder.

        Args:
            config: Training configuration
        """
        config = ctx.config
        super().__init__()
        self._config = config
        self._method: str = "hard"
        self._weight: float = 1.0
        self._device: str = "cuda"
        logger.info("DataConsistencyBuilder initialized")

    def with_method(self, method: str) -> "DataConsistencyBuilder":
        """Set DC projection method.

        Args:
            method: Method type ('hard' for projection, 'soft' for weighted)

        Returns:
            self for chaining
        """
        if method not in ("hard", "soft"):
            raise ValueError(f"Invalid method: {method}")
        self._method = method
        return self

    def with_weight(self, weight: float) -> "DataConsistencyBuilder":
        """Set DC weight for soft projection.

        Args:
            weight: Weight multiplier

        Returns:
            self for chaining
        """
        if weight < 0:
            raise ValueError(f"Invalid weight: {weight}")
        self._weight = weight
        return self

    def with_device(self, device: str) -> "DataConsistencyBuilder":
        """Set device for DC operations.

        Args:
            device: Device name ('cuda', 'cpu')

        Returns:
            self for chaining
        """
        self._device = device
        return self

    def validate(self) -> "DataConsistencyBuilder":
        """Validate builder state.

        Returns:
            self for chaining
        """
        super().validate()

        if self._method == "soft" and self._weight == 0:
            logger.warning("Soft DC with weight=0 is ineffective")

        return self

    def build(self) -> SimpleDataConsistency:
        """Build and return data consistency operator.

        Returns:
            Configured DC operator

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            dc = SimpleDataConsistency(
                method=self._method,
                weight=self._weight,
                device=self._device,
            )

            self._product = dc
            logger.info(f"Data consistency built: method={self._method}")

            return dc

        except Exception as e:
            raise RuntimeError(f"Failed to create DC operator: {e}") from e


class PhysicsBuilder(FluentBuilder[dict[str, Any]]):
    """Builder for aggregating all physics operators.

    Creates a complete physics module with FFT, masks, and DC projection.

    Example:
        >>> builder = PhysicsBuilder(config)
        >>> physics = (builder
        ...     .with_fft_norm("ortho")
        ...     .with_mask_acceleration(4)
        ...     .with_dc_method("hard")
        ...     .validate()
        ...     .build())
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize physics builder.

        Args:
            config: Training configuration
        """
        config = ctx.config
        super().__init__()
        self._config = config
        self._fft_norm: str = "ortho"
        self._fft_centering: bool = True
        self._mask_shape: tuple | None = None
        self._mask_pattern: str = "cartesian"
        self._mask_acceleration: int = 4
        self._mask_center_fraction: float = 0.08
        self._dc_method: str = "hard"
        self._dc_weight: float = 1.0
        self._device: str = "cuda"
        logger.info("PhysicsBuilder initialized")

    def with_fft_norm(self, norm: str) -> "PhysicsBuilder":
        """Set FFT normalization.

        Args:
            norm: Normalization mode

        Returns:
            self for chaining
        """
        self._fft_norm = norm
        return self

    def with_fft_centering(self, centering: bool = True) -> "PhysicsBuilder":
        """Enable/disable k-space centering.

        Args:
            centering: Whether to center k-space

        Returns:
            self for chaining
        """
        self._fft_centering = centering
        return self

    def with_mask_shape(self, shape: tuple) -> "PhysicsBuilder":
        """Set k-space mask shape.

        Args:
            shape: Mask dimensions

        Returns:
            self for chaining
        """
        self._mask_shape = shape
        return self

    def with_mask_pattern(self, pattern: str) -> "PhysicsBuilder":
        """Set mask sampling pattern.

        Args:
            pattern: Sampling pattern type

        Returns:
            self for chaining
        """
        self._mask_pattern = pattern
        return self

    def with_mask_acceleration(self, acceleration: int) -> "PhysicsBuilder":
        """Set mask acceleration factor.

        Args:
            acceleration: Undersampling acceleration

        Returns:
            self for chaining
        """
        self._mask_acceleration = acceleration
        return self

    def with_mask_center_fraction(self, fraction: float) -> "PhysicsBuilder":
        """Set mask center fraction.

        Args:
            fraction: Fraction of center to sample

        Returns:
            self for chaining
        """
        self._mask_center_fraction = fraction
        return self

    def with_dc_method(self, method: str) -> "PhysicsBuilder":
        """Set data consistency method.

        Args:
            method: DC method type

        Returns:
            self for chaining
        """
        self._dc_method = method
        return self

    def with_dc_weight(self, weight: float) -> "PhysicsBuilder":
        """Set data consistency weight.

        Args:
            weight: DC weight

        Returns:
            self for chaining
        """
        self._dc_weight = weight
        return self

    def with_device(self, device: str) -> "PhysicsBuilder":
        """Set device for all operators.

        Args:
            device: Device name

        Returns:
            self for chaining
        """
        self._device = device
        return self

    def validate(self) -> "PhysicsBuilder":
        """Validate builder state.

        Returns:
            self for chaining
        """
        super().validate()

        if self._mask_shape is None:
            logger.warning("Mask shape not specified, using default")

        return self

    def build(self) -> dict[str, Any]:
        """Build and return physics operators dictionary.

        Returns:
            Dictionary with 'fft', 'mask', 'dc' operators

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            # Create FFT operator
            fft_builder = FFTBuilder(self._config)
            fft = fft_builder.with_norm(self._fft_norm).with_centering(self._fft_centering).build()

            # Create mask operator
            mask = None
            if self._mask_shape:
                mask_builder = MaskBuilder(self._config, device=self._device)
                mask = (
                    mask_builder.with_shape(self._mask_shape)
                    .with_sampling_pattern(self._mask_pattern)
                    .with_acceleration(self._mask_acceleration)
                    .with_center_fraction(self._mask_center_fraction)
                    .build()
                )

            # Create data consistency operator
            dc_builder = DataConsistencyBuilder(self._config)
            dc = (
                dc_builder.with_method(self._dc_method)
                .with_weight(self._dc_weight)
                .with_device(self._device)
                .build()
            )

            physics_ops = {
                "fft": fft,
                "mask": mask,
                "dc": dc,
            }

            self._product = physics_ops
            logger.info("Physics operators built successfully")

            return physics_ops

        except Exception as e:
            raise RuntimeError(f"Failed to build physics operators: {e}") from e


__all__ = [
    "DataConsistencyBuilder",
    "FFTBuilder",
    "MaskBuilder",
    "PhysicsBuilder",
]
