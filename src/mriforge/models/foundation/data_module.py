"""
Multi-Modal Data Module for UMR-FM.

Aggregates data from FastMRI, SKM-TEA, and Stanford Knee.
Currently a placeholder structure until datasets are mounted.
"""

import logging

logger = logging.getLogger(__name__)


class MultiModalDataModule:
    """DataModule for aggregating multiple MRI datasets.

    .. warning::

        This is a **scaffold** — real dataset mounts must be configured
        before use.  Calling :meth:`train_dataloader` or
        :meth:`val_dataloader` without first mounting the underlying
        datasets will raise ``NotImplementedError``.
    """

    def __init__(self, batch_size: int = 1):
        """__init__.

        Args:
            batch_size (int): Description.
        """
        self.batch_size = batch_size
        self.datasets = ["FastMRI", "SKM-TEA", "Stanford"]
        self._setup_done = False

    def setup(self):
        """setup.

        Returns:
            Any: Description.
        """
        logger.debug(f"Setting up datasets: {self.datasets}")
        # In real implementation, would mount and indexing paths.
        # self._setup_done = True  # Uncomment when real loaders are wired

    def train_dataloader(self):
        """Return a training DataLoader.

        Raises:
            NotImplementedError: Always, until real dataset mounts are configured.
        """
        raise NotImplementedError(
            "MultiModalDataModule.train_dataloader() requires real dataset "
            "mounts (FastMRI, SKM-TEA, Stanford). Configure data paths in "
            "the foundation model YAML before training."
        )

    def val_dataloader(self):
        """Return a validation DataLoader.

        Raises:
            NotImplementedError: Always, until real dataset mounts are configured.
        """
        raise NotImplementedError(
            "MultiModalDataModule.val_dataloader() requires real dataset "
            "mounts. Configure data paths in the foundation model YAML."
        )
