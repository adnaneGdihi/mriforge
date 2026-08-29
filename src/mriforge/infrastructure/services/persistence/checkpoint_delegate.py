import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointDelegate:
    """Delegate responsible for model checkpointing logic.

    Encapsulates naming conventions and atomic save operations,
    delegating the actual tensor serialization to ICheckpointService.
    """

    def __init__(self, checkpoint_service: Any, output_dir: str):
        """Initialize the checkpoint delegate.

        Args:
            checkpoint_service: Service that handles tensor serialization
            output_dir: Directory where checkpoints should be saved
        """
        self.checkpoint_service = checkpoint_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, session: Any, step: int, epoch: int) -> str:
        """Save a model checkpoint with standard naming.

        Constructs a filename like 'checkpoint_epoch_X_step_Y.pth' and
        delegates to the checkpoint service.

        Args:
            session: Training session containing models and optimizers
            step: Current training step
            epoch: Current training epoch

        Returns:
            Path to the saved checkpoint
        """
        # Construct standard filename
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pth"
        file_path = self.output_dir / filename

        logger.info(f"Saving checkpoint: {filename}")

        # Delegate actual saving to service
        try:
            saved_path = self.checkpoint_service.save_checkpoint(
                session=session,
                file_path=str(file_path),
                step=step,
                epoch=epoch,
            )
            return saved_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filename}: {e}")
            raise
