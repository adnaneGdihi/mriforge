"""Model/Dataset Creation Pipeline - Config-Driven Factory Interface.

This pipeline provides a unified interface for creating models, optimizers,
and datasets directly from TrainingSettings configuration.

Single Source of Truth: TrainingSettings (loaded from YAML via ``from_yaml``,
or supplied in-memory by a scripting-API caller via the ``config=`` argument).
"""

import logging
from pathlib import Path
from typing import Any

import torch

from mriforge.config.settings import TrainingSettings
from mriforge.core.module_utils import resolve_state_dict
from mriforge.infrastructure.builders.leaf.optimizer_builders import OptimizerBuilder

logger = logging.getLogger(__name__)


def _resolve_settings(
    config_path: Path | None, config: TrainingSettings | None
) -> TrainingSettings:
    """Resolve a :class:`TrainingSettings` from an in-memory object or a path.

    An explicit in-memory ``config`` wins; otherwise the YAML at
    ``config_path`` is loaded via the SSOT :meth:`TrainingSettings.from_yaml`.
    Passing neither raises (no silent default — pitfall #9): the scripting-API
    in-memory path and the YAML path are the only two sources.

    Args:
        config_path: Path to a training-config YAML, or ``None``.
        config: An already-built in-memory ``TrainingSettings``, or ``None``.

    Returns:
        The resolved frozen ``TrainingSettings``.
    """
    if config is not None:
        return config
    if config_path is None:
        raise ValueError("make_*: provide either config_path or config")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    return TrainingSettings.from_yaml(str(config_path))


def _config_path_label(config_path: Path | None) -> str:
    """Provenance label for the config source (path or ``<in-memory>``)."""
    return str(config_path) if config_path else "<in-memory>"


def make_model(
    config_path: Path | None = None,
    device: str = "cuda",
    checkpoint_path: Path | None = None,
    *,
    config: TrainingSettings | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Create a model from configuration.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        device: Device to create model on ('cuda' or 'cpu').
        checkpoint_path: Optional checkpoint to load weights from.
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (model, metadata).
    """
    device_obj = torch.device(device)
    config = _resolve_settings(config_path, config)
    logger.info(f"Config loaded: model={config.model.model_type}")

    # Create model through the canonical builder -- the same one training uses.
    # This module advertises "Single Source of Truth: TrainingSettings", but the
    # `ModelFactory.create_model(config.model)` it called was handed a bare
    # ModelConfigSchema and so ran the branch that drops `acceleration_config`
    # and `kspace_log_scaled`, quietly building a different model than the same
    # config trains. ModelBuilder's injections are contract-gated, so nothing
    # reaches a constructor that does not declare it.
    logger.info(f"Creating model: {config.model.model_type}")
    from mriforge.infrastructure.training.builders.model_builder import ModelBuilder

    model = ModelBuilder(config, device_obj).build_generator().validate().build()["generator"]
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {num_params} parameters")

    # Load checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device_obj)
            # Was the narrowest checkpoint reader in the repo: it knew only
            # "model_state_dict", fell through to loading the whole envelope,
            # and -- unlike infer's -- never stripped wrapper prefixes either,
            # so a compiled/DDP checkpoint failed here too. One SSOT now.
            model.load_state_dict(
                resolve_state_dict(
                    checkpoint, model.state_dict().keys(), source=str(checkpoint_path)
                )
            )
            logger.info("Weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    metadata = {
        "config_path": _config_path_label(config_path),
        "model_type": config.model.model_type,
        "device": device,
        "num_parameters": num_params,
        "checkpoint_loaded": checkpoint_path is not None,
    }

    return model, metadata


def make_optimizer(
    config_path: Path | None = None,
    model: torch.nn.Module | None = None,
    *,
    config: TrainingSettings | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Create an optimizer from configuration.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        model: Model to optimize (required).
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (optimizer, metadata).
    """
    if model is None:
        raise ValueError("make_optimizer: 'model' is required")
    config = _resolve_settings(config_path, config)
    logger.info(f"Config loaded: optimizer={config.optimization.optimizer.type}")

    # Create optimizer
    logger.info(f"Creating optimizer: {config.optimization.optimizer.type}")
    builder = OptimizerBuilder(config, params=model.parameters())
    optimizer = builder.validate().build()
    logger.info(f"Optimizer created with lr={config.optimization.optimizer.learning_rate}")

    metadata = {
        "config_path": _config_path_label(config_path),
        "optimizer_type": config.optimization.optimizer.type,
        "learning_rate": config.optimization.optimizer.learning_rate,
    }

    return optimizer, metadata


def make_dataset(
    config_path: Path | None = None,
    split: str = "train",
    device: str = "cpu",
    *,
    config: TrainingSettings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a dataset from configuration.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        split: Dataset split ('train', 'val', 'test').
        device: Device for data operations.
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (dataset, metadata).
    """
    config = _resolve_settings(config_path, config)
    logger.info(f"Config loaded: dataset={config.data.dataset_type}, split={split}")

    from mriforge.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    director = DataPipelineDirector(config)
    train_loader, val_loader = director.build_dataloaders(num_workers=0)

    # Return the underlying dataset for the requested split
    if split in ("val", "validation", "test") and val_loader is not None:
        dataset = val_loader.dataset
    elif train_loader is not None:
        dataset = train_loader.dataset
    else:
        raise RuntimeError(f"No data loader produced for split='{split}'")

    logger.info(f"Dataset created: {len(dataset)} samples")  # type: ignore[arg-type]

    metadata = {
        "config_path": _config_path_label(config_path),
        "dataset_type": config.data.dataset_type,
        "split": split,
        "num_samples": len(dataset),  # type: ignore[arg-type]
        "batch_size": config.data.loader.batch_size,
    }

    return dataset, metadata


def make_dataloader(
    config_path: Path | None = None,
    split: str = "train",
    batch_size: int | None = None,
    num_workers: int | None = None,
    *,
    config: TrainingSettings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a dataloader from configuration.

    Delegates to :class:`~mriforge.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector`
    instead of instantiating ``torch.utils.data.DataLoader`` here. This
    keeps the pipelines layer free of direct ``DataLoader`` construction
    (CLAUDE.md pitfall #11 — data SSOT lives in
    :mod:`mriforge.data` + the director). Phase 4 of
    ``TODO/backlog_ssot_and_layering_cleanup.md``.

    Args:
        config_path: Path to training config YAML (or omit and pass ``config``).
        split: Dataset split (``"train"``, ``"val"``, or ``"test"``).
        batch_size: Currently unused — the director reads batch size from
            the config. Kept in the signature for backward-compat.
        num_workers: Override number of workers (passed to the director).
        config: An in-memory ``TrainingSettings`` (scripting-API path); takes
            precedence over ``config_path``.

    Returns:
        Tuple of (dataloader, metadata).
    """
    config = _resolve_settings(config_path, config)

    # Build via the canonical data-pipeline director (SSOT entry point).
    from mriforge.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    director = DataPipelineDirector(config)
    if num_workers is None:
        num_workers = config.data.loader.num_workers
    train_loader, val_loader = director.build_dataloaders(num_workers=num_workers)

    if split in ("val", "validation", "test") and val_loader is not None:
        dataloader = val_loader
    elif train_loader is not None:
        dataloader = train_loader
    else:
        raise RuntimeError(f"Director produced no loader for split='{split}'")

    logger.info(
        f"DataLoader resolved: {len(dataloader)} batches (num_workers={num_workers}, split={split})"
    )

    metadata = {
        "config_path": _config_path_label(config_path),
        "dataset_type": config.data.dataset_type,
        "split": split,
        "num_batches": len(dataloader),
        "batch_size": getattr(dataloader, "batch_size", config.data.loader.batch_size),
        "num_workers": num_workers,
        "shuffle": split == "train",
    }

    return dataloader, metadata
