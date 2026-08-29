"""Phase 1: Data Builders for Data Pipeline Components

Implements fluent builders for creating data loading components:
- DatasetBuilder: Creates dataset instances
- DataLoaderBuilder: Creates PyTorch DataLoaders with multiprocessing

DataPipelineBuilder was removed: its only caller was the unreachable
TrainingPipelineDirector, and full-pipeline orchestration is the job of
``directors/data_pipeline_director.py::DataPipelineDirector`` (the data SSOT).

Each builder provides a fluent API for configuration and instantiation.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mriforge.core.worker_seeding import seed_worker
from mriforge.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)
from mriforge.infrastructure.builders.core import FluentBuilder

logger = logging.getLogger(__name__)


class DatasetBuilder(FluentBuilder[Dataset]):
    """Builder for creating dataset instances.

    Supports various dataset types (FastMRI, synthetic, custom).

    Example:
        >>> builder = DatasetBuilder(config)
        >>> dataset = (builder
        ...     .with_type("fastmri")
        ...     .with_split("train")
        ...     .with_motion_artifacts(True)
        ...     .validate()
        ...     .build())
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize dataset builder.

        Args:
            config: Training configuration
        """
        config = ctx.config
        super().__init__()
        self._config = config
        self._dataset_type: str | None = None
        self._split: str = "train"
        self._data_root: str | None = None
        self._transform: Callable | None = None
        self._motion_artifacts: bool = False
        self._noise_level: float = 0.0
        self._kwargs: dict[str, Any] = {}
        logger.info("DatasetBuilder initialized")

    def with_type(self, dataset_type: str) -> "DatasetBuilder":
        """Set dataset type.

        Args:
            dataset_type: Type of dataset (fastmri, synthetic, etc.)

        Returns:
            self for chaining
        """
        self._dataset_type = dataset_type
        return self

    def with_split(self, split: str) -> "DatasetBuilder":
        """Set dataset split.

        Args:
            split: Data split ('train', 'val', 'test', etc.)

        Returns:
            self for chaining
        """
        self._split = split
        return self

    def with_data_root(self, path: str) -> "DatasetBuilder":
        """Set root directory for data.

        Args:
            path: Path to data directory

        Returns:
            self for chaining
        """
        self._data_root = path
        return self

    def with_transform(self, transform: Callable) -> "DatasetBuilder":
        """Set data transform.

        Args:
            transform: Transform function or Compose object

        Returns:
            self for chaining
        """
        self._transform = transform
        return self

    def with_motion_artifacts(self, enabled: bool = True) -> "DatasetBuilder":
        """Enable/disable motion artifact simulation.

        Args:
            enabled: Whether to add simulated motion

        Returns:
            self for chaining
        """
        self._motion_artifacts = enabled
        return self

    def with_noise(self, noise_level: float) -> "DatasetBuilder":
        """Set noise level for augmentation.

        Args:
            noise_level: Standard deviation of Gaussian noise

        Returns:
            self for chaining
        """
        self._noise_level = noise_level
        return self

    def with_parameter(self, key: str, value: Any) -> "DatasetBuilder":
        """Set custom dataset parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            self for chaining
        """
        self._kwargs[key] = value
        return self

    def validate(self) -> "DatasetBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required parameters missing
        """
        super().validate()

        if self._dataset_type is None:
            self._dataset_type = self._config.data.dataset_type

        return self

    def build(self) -> Dataset:
        """Build and return dataset instance.

        Returns:
            Configured dataset

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            # Use unified API to create dataset
            from mriforge.data.datasets.api import create_dataset

            dataset_kwargs = {
                "split": self._split,
                "transform": self._transform,
            }

            # Add optional parameters
            if self._data_root:
                dataset_kwargs["data_root"] = self._data_root
            if self._motion_artifacts:
                dataset_kwargs["motion_artifacts"] = True
            if self._noise_level > 0:
                dataset_kwargs["noise_level"] = self._noise_level

            # Merge custom parameters
            dataset_kwargs.update(self._kwargs)

            # Create dataset strictly without synthetic fallback.
            # BIDS resolution and TorchIO pipelines must be provided prior, or handled by the underlying create_dataset wrapper.
            dataset = create_dataset(self._dataset_type, config=self._config, **dataset_kwargs)

            self._product = dataset
            logger.info(
                f"Dataset built: {self._dataset_type} (split={self._split}, size={len(dataset)})"
            )

            # Fail-fast: Empty datasets should not proceed silently
            if len(dataset) == 0:
                raise RuntimeError(
                    f"Dataset '{self._dataset_type}' is empty (size=0). "
                    f"Check data paths, manifests, or dataset configuration. "
                    f"No synthetic fallback - fix the data source."
                )

            return dataset

        except Exception as e:
            raise RuntimeError(f"Failed to create dataset: {e}") from e


class DataLoaderBuilder(FluentBuilder[DataLoader]):
    """Builder for creating PyTorch DataLoader instances.

    Configures data loading with multiprocessing, pinning, and prefetching.

    Example:
        >>> builder = DataLoaderBuilder(config, dataset=dataset)
        >>> loader = (builder
        ...     .with_batch_size(32)
        ...     .with_num_workers(4)
        ...     .with_pin_memory(True)
        ...     .validate()
        ...     .build())
    """

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize data loader builder.

        Args:
            ctx: Builder context carrying ``config`` and ``dataset``.
        """
        super().__init__()
        self._config = ctx.config
        self._dataset = ctx.dataset
        self._batch_size: int | None = None
        self._num_workers: int = 0
        # Default to True only when CUDA is actually available — otherwise
        # PyTorch warns ("'pin_memory' argument is set as true but no
        # accelerator is found") and the pinned host memory is wasted.
        # See findings booklet 2026-05-05 T-6.
        self._pin_memory: bool = bool(torch.cuda.is_available())
        self._shuffle: bool = False
        self._drop_last: bool = False
        self._prefetch_factor: int = 2
        self._persistent_workers: bool = False
        self._collate_fn: Callable | None = None
        self._sampler: Any = None
        logger.info("DataLoaderBuilder initialized")

    def with_batch_size(self, batch_size: int) -> "DataLoaderBuilder":
        """Set batch size.

        Args:
            batch_size: Number of samples per batch

        Returns:
            self for chaining
        """
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size: {batch_size}")
        self._batch_size = batch_size
        return self

    def with_num_workers(self, num_workers: int) -> "DataLoaderBuilder":
        """Set number of worker processes.

        Args:
            num_workers: Number of parallel data loading workers

        Returns:
            self for chaining
        """
        if num_workers < 0:
            raise ValueError(f"Invalid num_workers: {num_workers}")
        self._num_workers = num_workers
        return self

    def with_pin_memory(self, pin_memory: bool = True) -> "DataLoaderBuilder":
        """Enable/disable pinned memory for GPU transfer.

        Args:
            pin_memory: Whether to pin memory

        Returns:
            self for chaining
        """
        self._pin_memory = pin_memory
        return self

    def with_shuffle(self, shuffle: bool = True) -> "DataLoaderBuilder":
        """Enable/disable data shuffling.

        Args:
            shuffle: Whether to shuffle data

        Returns:
            self for chaining
        """
        self._shuffle = shuffle
        return self

    def with_sampler(self, sampler: Any) -> "DataLoaderBuilder":
        """Supply an explicit ``torch.utils.data.Sampler`` for the epoch order.

        Mutually exclusive with :meth:`with_shuffle` — ``DataLoader`` raises when both are
        given, so setting a sampler clears the shuffle flag here rather than letting that
        surface as a confusing constructor error two layers away. The sampler owns the
        shuffling from then on (see
        :class:`~mriforge.data.samplers.VolumeBlockedSliceSampler`, which shuffles inside a
        resident-volume block).

        Args:
            sampler: a Sampler over dataset indices, or ``None`` to clear.

        Returns:
            self for chaining
        """
        self._sampler = sampler
        if sampler is not None:
            self._shuffle = False
        return self

    def with_drop_last(self, drop_last: bool = True) -> "DataLoaderBuilder":
        """Enable/disable dropping last incomplete batch.

        Args:
            drop_last: Whether to drop last incomplete batch

        Returns:
            self for chaining
        """
        self._drop_last = drop_last
        return self

    def with_prefetch_factor(self, factor: int) -> "DataLoaderBuilder":
        """Set prefetch factor for async loading.

        Args:
            factor: Number of batches to prefetch per worker

        Returns:
            self for chaining
        """
        if factor <= 0:
            raise ValueError(f"Invalid prefetch_factor: {factor}")
        self._prefetch_factor = factor
        return self

    def with_persistent_workers(self, persistent: bool = True) -> "DataLoaderBuilder":
        """Enable/disable persistent workers.

        Args:
            persistent: Whether to keep workers alive between epochs

        Returns:
            self for chaining
        """
        self._persistent_workers = persistent
        return self

    def with_collate_fn(self, collate_fn: Callable) -> "DataLoaderBuilder":
        """Set custom collation function.

        Args:
            collate_fn: Collation function

        Returns:
            self for chaining
        """
        self._collate_fn = collate_fn
        return self

    def validate(self) -> "DataLoaderBuilder":
        """Validate builder state.

        Returns:
            self for chaining

        Raises:
            ValueError: If invalid parameters
        """
        super().validate()

        if self._batch_size is None:
            self._batch_size = self._config.data.loader.batch_size

        if self._batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self._batch_size}")

        return self

    def build(self) -> DataLoader:
        """Build and return DataLoader instance.

        Returns:
            Configured DataLoader

        Raises:
            ValueError: If validation fails
        """
        self.validate()

        try:
            # [PHASE 2.2] Use CollationStrategySelector for centralized strategy selection
            if self._collate_fn is None:
                from mriforge.data.collation import CollateStrategyFactory
                from mriforge.data.collation.strategy_selector import (
                    CollationStrategySelector,
                )

                # Extract config parameters for selector
                dataset_type = self._config.data.dataset_type
                patch_size = self._config.data.sampling.patch_size
                enable_slab_mode = self._config.data.sampling.enable_slab_mode

                # STEP 1: Select strategy based on user config (centralized logic)
                strategy_name, strategy_kwargs = CollationStrategySelector.select_strategy(
                    config=self._config.data.collation,
                    dataset_type=dataset_type,
                    enable_slab_mode=enable_slab_mode,
                    patch_size=patch_size,
                )

                # STEP 2: Instantiate strategy with user-specified parameters
                logger.info(
                    f"[DataLoaderBuilder] Creating collation strategy: "
                    f"strategy='{strategy_name}', kwargs={strategy_kwargs}"
                )

                strategy = CollateStrategyFactory.create(strategy_name, **strategy_kwargs)
                self._collate_fn = strategy.collate

                # STEP 3: Log collation function for transparency
                logger.info(
                    f"[DataLoaderBuilder] Collation function ready: {self._collate_fn.__name__} "
                    f"(strategy={strategy_name})"
                )

            if self._sampler is not None:
                describe = getattr(self._sampler, "describe", None)
                logger.info(
                    "[DataLoaderBuilder] epoch order from %s%s",
                    type(self._sampler).__name__,
                    f" — {describe()}" if callable(describe) else "",
                )

            loader = DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                sampler=self._sampler,
                # DataLoader forbids shuffle=True alongside a sampler; with_sampler()
                # already clears the flag, this keeps the invariant local to the call.
                shuffle=self._shuffle if self._sampler is None else False,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                drop_last=self._drop_last,
                collate_fn=self._collate_fn,
                # Seed NumPy/``random`` per worker (torch auto-seeds only its own
                # RNG) so numpy-based transforms don't duplicate draws across
                # workers. No-op at num_workers=0.
                worker_init_fn=seed_worker,
                prefetch_factor=(self._prefetch_factor if self._num_workers > 0 else None),
                persistent_workers=self._persistent_workers and self._num_workers > 0,
            )

            self._product = loader
            logger.info(
                f"DataLoader built: batch_size={self._batch_size}, "
                f"workers={self._num_workers}, pin_memory={self._pin_memory}"
            )

            return loader

        except Exception as e:
            raise RuntimeError(f"Failed to create DataLoader: {e}") from e


__all__ = [
    "DataLoaderBuilder",
    "DatasetBuilder",
]
