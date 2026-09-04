"""Tiny synthetic MRI dataset for smoke tests and CI.

Provides deterministic low-resolution/high-resolution tensor pairs so tests
can exercise data pipelines without external datasets.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset


class SyntheticMRIDataset(Dataset):
    """Generate deterministic synthetic MRI pairs.

    The dataset creates simple geometric patterns (sine gratings) with an
    optional amount of additive noise to mimic low/high resolution pairs while
    keeping computations lightweight.

    .. mermaid::

        flowchart LR
            Start[Index] --> Gen{Generate Pattern}
            Gen -->|Sin/Cos| HR[High Res Tensor]
            HR -->|Downsample| LR[Low Res Tensor]

            HR & LR --> Device[Move to Device]
            Device --> Batch[TrainingBatch]
    """

    def __init__(
        self,
        num_samples: int = 16,
        lr_shape: tuple[int, int, int] = (1, 64, 64),
        upscale_factor: int = 2,
        noise_level: float = 0.0,
        external_augmentations: Callable | None = None,
    ) -> None:
        """__init__.

        Args:
            num_samples (int): Description.
            lr_shape (tuple[int, int, int]): Description.
            upscale_factor (int): Description.
            noise_level (float): Description.
            external_augmentations (callable | None): Description.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if len(lr_shape) != 3:
            raise ValueError("lr_shape must be (C, H, W)")
        if upscale_factor <= 0:
            raise ValueError("upscale_factor must be positive")

        self.num_samples = num_samples
        self.lr_shape = lr_shape
        self.upscale_factor = upscale_factor
        self.noise_level = max(0.0, float(noise_level))
        self.external_augmentations = external_augmentations

        channels, height, width = lr_shape
        self.hr_shape = (
            channels,
            height * upscale_factor,
            width * upscale_factor,
        )

        self._base_grid = self._create_normalised_grid(self.hr_shape[1:])

    @staticmethod
    def _create_normalised_grid(shape: tuple[int, int]) -> torch.Tensor:
        """_create_normalised_grid.

        Args:
            shape (tuple[int, int]): Description.
        Returns:
            torch.Tensor: Description.
        """
        h, w = shape
        y = torch.linspace(-math.pi, math.pi, h)
        x = torch.linspace(-math.pi, math.pi, w)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_y, grid_x], dim=0)

    def __len__(self) -> int:
        """__len__.

        Returns:
            int: Description.
        """
        return self.num_samples

    def dry_iter(self) -> list[tio.Subject]:
        """Cheap Subject shells for ``tio.Queue`` length probing (no voxel read).

        ``tio.Queue.iterations_per_epoch`` calls ``subjects_dataset.dry_iter()``
        and only needs an indexable ``Sequence[Subject]`` of the right length.
        Without it a queued arm crashes at step 0 with ``'SyntheticMRIDataset'
        object has no attribute 'dry_iter'`` (the 2026-06 oracle_bssfp /
        exp_vf_29 incident class). Mirrors :meth:`OracleBssfpDataset.dry_iter`.
        """
        stub = torch.zeros(1, 1, 1, 1)
        return [tio.Subject(input=tio.ScalarImage(tensor=stub)) for _ in range(len(self))]

    def _generate_hr_sample(self, index: int) -> torch.Tensor:
        """_generate_hr_sample.

        Args:
            index (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        channels, _, _ = self.hr_shape
        base_pattern = torch.sin(self._base_grid[0] * (index + 1))
        texture = torch.cos(self._base_grid[1] * (index + 1))
        image = (base_pattern + texture) / 2.0
        image = image.unsqueeze(0).repeat(channels, 1, 1)
        if self.noise_level > 0:
            noise = torch.randn_like(image) * self.noise_level
            image = image + noise
        return image.clamp(-1.0, 1.0)

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """_downsample.

        Args:
            tensor (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=self.lr_shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def __getitem__(self, index: int) -> tio.Subject:
        """__getitem__.

        Args:
            index (int): Description.
        Returns:
            tio.Subject: Description.
        """
        hr_tensor = self._generate_hr_sample(index % self.num_samples)
        lr_tensor = self._downsample(hr_tensor)

        if self.external_augmentations is not None:
            lr_tensor, hr_tensor = self.external_augmentations(
                lr_tensor,
                hr_tensor,
            )

        # TorchIO ScalarImage expects (C, H, W, D); add trailing depth dim = 1.
        return tio.Subject(
            input=tio.ScalarImage(tensor=lr_tensor.unsqueeze(-1), affine=np.eye(4)),
            target=tio.ScalarImage(tensor=hr_tensor.unsqueeze(-1), affine=np.eye(4)),
            index=index,
        )

    def get_metadata(self) -> dict[str, object]:
        """get_metadata.

        Returns:
            dict[str, object]: Description.
        """
        return {
            "dataset_type": "synthetic",
            "num_samples": self.num_samples,
            "lr_shape": self.lr_shape,
            "hr_shape": self.hr_shape,
            "noise_level": self.noise_level,
        }
