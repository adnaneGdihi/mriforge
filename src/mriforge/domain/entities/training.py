from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from mriforge.infrastructure.training.contexts import TrainingEnvironment


@dataclass
class TrainingSessionEntity:
    """Domain entity representing a training session in the MRIForge system.

    ### Training Session Lifecycle
    ```mermaid
    classDiagram
        class TrainingSessionEntity {
            +str session_id
            +str model_id
            +dict config
            +str status
            +datetime start_time
            +dict metrics
            +list checkpoints
            +TrainingEnvironment env
            +TrainingStrategy strategy
            +int current_epoch
            +float best_loss
            +int global_step
            +generator nn.Module
            +discriminator nn.Module
            +gen_optimizer Optimizer
            +disc_optimizer Optimizer
        }
    ```
    """

    session_id: str
    model_id: str
    config: Any  # TrainingSettings at runtime, typed as Any to avoid circular import
    status: str = "initialized"  # initialized, running, completed, failed
    start_time: datetime | None = None
    end_time: datetime | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)

    env: Optional["TrainingEnvironment"] = None
    strategy: Any | None = None
    current_epoch: int = 0
    best_loss: float = float("inf")
    global_step: int = 0
    ssl_loader: Any | None = field(default=None, repr=False)
    ssl_iterator: Any | None = field(default=None, repr=False)
    iteration_counter: Any | None = field(default=None, repr=False)

    @property
    def generator(self) -> nn.Module | None:
        """Get generator model from environment."""
        return self.env.generator if self.env else None

    @property
    def discriminator(self) -> nn.Module | None:
        """Get discriminator model from environment."""
        return self.env.discriminator if self.env else None

    @property
    def gen_optimizer(self) -> torch.optim.Optimizer | None:
        """Get generator optimizer from environment."""
        return self.env.opt_g if self.env else None

    @property
    def disc_optimizer(self) -> torch.optim.Optimizer | None:
        """Get discriminator optimizer from environment."""
        return self.env.opt_d if self.env else None

    @property
    def scheduler_g(self) -> Any | None:
        """Get generator scheduler from environment."""
        return self.env.schedulers.get("scheduler_g") if self.env else None

    @property
    def scheduler_d(self) -> Any | None:
        """Get discriminator scheduler from environment."""
        return self.env.schedulers.get("scheduler_d") if self.env else None

    @property
    def device(self) -> torch.device | None:
        """device.

        Returns:
            Optional[torch.device]: Description.
        """
        return self.env.device if self.env else None

    @property
    def ema(self) -> Any | None:
        """ema.

        Returns:
            Optional[Any]: Description.
        """
        return self.env.generator.ema if self.env else None

    @property
    def train_loader(self) -> Any | None:
        """Helper to get train loader from environment."""
        return self.env.data_loaders.get("train") if self.env and self.env.data_loaders else None

    @property
    def val_loader(self) -> Any | None:
        """Helper to get validation loader from environment."""
        return self.env.data_loaders.get("val") if self.env and self.env.data_loaders else None

    @property
    def grad_scaler(self) -> Any | None:
        """Helper to get gradient scaler from environment."""
        return self.env.scaler if self.env else None

    def start_session(self) -> None:
        """Starts the training session by updating status and start time.

        This method transitions the session from 'initialized' to 'running'
        state and records the start timestamp.
        """
        self.status = "running"
        self.start_time = datetime.now()

    def end_session(self, success: bool = True) -> None:
        """Ends the training session and records the completion status.

        Args:
            success (bool): Whether the training completed successfully.
                Defaults to True.

        """
        self.status = "completed" if success else "failed"
        self.end_time = datetime.now()

    def add_metric(self, key: str, value: Any) -> None:
        """Adds a training metric to the session.

        Args:
            key (str): Metric name (e.g., 'loss', 'accuracy').
            value (Any): Metric value to store.

        """
        self.metrics[key] = value

    def add_checkpoint(self, checkpoint_path: str) -> None:
        """Adds a checkpoint file path to the session.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        """
        self.checkpoints.append(checkpoint_path)

    def increment_step(self) -> None:
        """Increments the global step counter."""
        self.global_step += 1

    def to_dict(self) -> dict[str, Any]:
        """Converts the entity to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all entity attributes
                with datetime objects converted to ISO format strings.

        """
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "config": self.config,
            "status": self.status,
            "start_time": (self.start_time.isoformat() if self.start_time else None),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "checkpoints": self.checkpoints,
        }


@dataclass
class ExperimentEntity:
    """Domain entity representing an experiment in the MRIForge system.

    An experiment groups multiple training sessions together for comparison
    and analysis. It provides organization and metadata for related training
    runs, enabling systematic evaluation of different configurations.

    Attributes:
        experiment_id (str): Unique identifier for the experiment.
        name (str): Human-readable name for the experiment.
        description (str): Detailed description of the experiment's purpose.
        config (dict[str, Any]): Base configuration for the experiment.
        sessions (list[str]): list of training session IDs in this experiment.
        created_at (datetime): When the experiment was created.
        tags (list[str]): Tags for categorization and filtering.

    Example:
        >>> exp = ExperimentEntity(
        ...     experiment_id="exp_001",
        ...     name="UNet Architecture Study",
        ...     description="Comparing different UNet variants for MRI SR"
        ... )
        >>> exp.add_session("train_001")
        >>> exp.add_tag("unet")

    """

    experiment_id: str
    name: str
    description: str
    config: dict[str, Any]
    sessions: list[str] = field(default_factory=list)  # Session IDs
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def add_session(self, session_id: str) -> None:
        """Adds a training session to the experiment.

        Args:
            session_id (str): ID of the training session to add.

        """
        if session_id not in self.sessions:
            self.sessions.append(session_id)

    def add_tag(self, tag: str) -> None:
        """Adds a tag to the experiment for categorization.

        Args:
            tag (str): Tag to add to the experiment.

        """
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> dict[str, Any]:
        """Converts the entity to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all entity attributes
                with datetime objects converted to ISO format strings.

        """
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "sessions": self.sessions,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


@dataclass
class ValidationResultEntity:
    """Domain entity representing validation results in the MRIForge system.

    This entity stores the results of model validation during or after
    training, including performance metrics and optionally sample outputs.
    It enables tracking of model performance over time and comparison
    across different training stages.

    Attributes:
        result_id (str): Unique identifier for the validation result.
        session_id (str): ID of the training session this result belongs to.
        epoch (int): Training epoch when validation was performed.
        metrics (dict[str, float]): Validation metrics (e.g., PSNR, SSIM).
        samples (Optional[torch.Tensor]): Sample outputs from validation.
        created_at (datetime): When the validation was performed.

    Example:
        >>> result = ValidationResultEntity(
        ...     result_id="val_001",
        ...     session_id="train_001",
        ...     epoch=50,
        ...     metrics={"psnr": 25.3, "ssim": 0.85}
        ... )
        >>> print(result.is_successful)
        True

    """

    result_id: str
    session_id: str
    epoch: int
    metrics: dict[str, float]
    samples: torch.Tensor | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_successful(self) -> bool:
        """Checks if validation was successful based on metrics.

        Success is determined by all metrics having positive values.
        This is a simple heuristic that can be customized based on
        domain-specific requirements.

        Returns:
            bool: True if metrics are present AND all are positive, False
                otherwise. An empty ``metrics`` dict is NOT success (``all(...)``
                on an empty iterable is vacuously True — a freshly-constructed
                result must not report success).

        """
        # Define success criteria based on domain rules
        if not self.metrics:
            return False
        return all(metric > 0 for metric in self.metrics.values())

    def to_dict(self) -> dict[str, Any]:
        """Converts the entity to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all entity attributes
                with datetime objects converted to ISO format strings.

        """
        return {
            "result_id": self.result_id,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "has_samples": self.samples is not None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(kw_only=True)
class DomainEvent:
    """Base class for domain events in the MRIForge system.

    Domain events represent significant occurrences in the domain that other
    parts of the system might be interested in. They follow the Domain Event
    pattern and enable loose coupling between domain components.

    A ``@dataclass(kw_only=True)``: the subclasses below are themselves
    dataclasses, and a plain-class base whose hand-written ``__init__`` is never
    chained by the generated subclass ``__init__`` left ``timestamp``/``event_id``
    unset (``to_dict`` then raised ``AttributeError``). ``kw_only`` lets the base
    carry a defaulted ``timestamp`` ahead of the subclasses' non-default fields
    without the "non-default follows default" error.

    Attributes:
        event_id (str): Unique identifier for the event.
        timestamp (datetime): When the event occurred.

    Example:
        >>> event = TrainingStartedEvent(
        ...     event_id="evt_001",
        ...     session_id="train_001",
        ...     model_id="unet_001",
        ...     config={"epochs": 100}
        ... )

    """

    event_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Converts the event to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing event data with
                timestamp in ISO format.

        """
        return {
            "event_id": self.event_id,
            "event_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(kw_only=True)
class TrainingStartedEvent(DomainEvent):
    """Event fired when a training session starts in the MRIForge system.

    This event signals the beginning of a training session and includes
    all the relevant information about the training configuration and
    models being used.

    Attributes:
        event_id (str): Unique identifier for the event (inherited).
        session_id (str): ID of the training session that started.
        model_id (str): ID of the model being trained.
        config (dict[str, Any]): Training configuration parameters.

    """

    session_id: str
    model_id: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts the event to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing base event data plus
                training-specific information.

        """
        base_dict = super().to_dict()
        base_dict.update(
            {
                "session_id": self.session_id,
                "model_id": self.model_id,
                "config": self.config,
            },
        )
        return base_dict


@dataclass(kw_only=True)
class TrainingCompletedEvent(DomainEvent):
    """Event fired when a training session completes in the MRIForge system.

    This event signals the end of a training session and includes the
    final performance metrics and success status.

    Attributes:
        event_id (str): Unique identifier for the event (inherited).
        session_id (str): ID of the training session that completed.
        final_metrics (dict[str, float]): Final training metrics achieved.
        success (bool): Whether the training completed successfully.

    """

    session_id: str
    final_metrics: dict[str, float]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Converts the event to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing base event data plus
                completion-specific information.

        """
        base_dict = super().to_dict()
        base_dict.update(
            {
                "session_id": self.session_id,
                "final_metrics": self.final_metrics,
                "success": self.success,
            },
        )
        return base_dict


@dataclass(kw_only=True)
class ModelValidatedEvent(DomainEvent):
    """Event fired when model validation occurs in the MRIForge system.

    This event signals that a model has been validated and includes
    the validation results and performance metrics.

    Attributes:
        model_id (str): ID of the model that was validated.
        validation_results (dict[str, Any]): Results from the validation
            process, including metrics and analysis.

    """

    model_id: str
    validation_results: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts the event to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing base event data plus
                validation-specific information.

        """
        base_dict = super().to_dict()
        base_dict.update(
            {
                "model_id": self.model_id,
                "validation_results": self.validation_results,
            },
        )
        return base_dict
