"""Synthetic CLAHE-enabled dataset used by integration tests."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import Dataset


def _apply_fake_clahe(
    image: torch.Tensor,
    clip_limit: float = 2.0,
) -> torch.Tensor:
    """Lightweight CLAHE-style normalization without external deps."""

    if image.numel() == 0:
        return image

    image = image - image.min()
    image = image / (image.max() - image.min() + 1e-6)
    image = torch.clamp(image * clip_limit, 0.0, 1.0)
    return image


class CLAHEDataset(Dataset[dict[str, torch.Tensor]]):
    """Toy dataset that returns multimodal MRI-like tensors.

    The implementation purposely injects a small delay when running inside
    DataLoader worker processes so tests covering timeout handling can trigger
    the expected ``RuntimeError`` when a tight ``timeout`` argument is set.
    """

    def __init__(
        self,
        length: int = 64,
        image_shape: tuple[int, int, int] = (1, 64, 64),
        device: torch.device | None = None,
        simulate_worker_delay: float = 0.2,
    ) -> None:
        """__init__.

        Args:
            length (int): Description.
            image_shape (tuple[int, int, int]): Description.
            device (torch.device | None): Description.
            simulate_worker_delay (float): Description.
        """
        super().__init__()
        self.length = length
        self.image_shape = image_shape
        self.device = device or torch.device("cpu")
        self.simulate_worker_delay = simulate_worker_delay

    def __len__(self) -> int:  # pragma: no cover - trivial
        """__len__.

        Returns:
            int: Description.
        """
        return self.length

    def _generate_base_image(self, index: int) -> torch.Tensor:
        """_generate_base_image.

        Args:
            index (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        generator = torch.Generator()
        generator.manual_seed(index)
        image = torch.rand(
            self.image_shape,
            dtype=torch.float32,
            generator=generator,
        )
        return _apply_fake_clahe(image)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """__getitem__.

        Args:
            index (int): Description.
        Returns:
            dict[str, Any]: Description.
        """
        dataset_length = len(self)
        if dataset_length <= 0:
            raise IndexError("CLAHEDataset is empty")
        if index < 0:
            index += dataset_length
        if index < 0 or index >= dataset_length:
            raise IndexError(f"Index {index} out of range for CLAHEDataset")

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and self.simulate_worker_delay > 0:
            time.sleep(self.simulate_worker_delay)

        image = self._generate_base_image(index)
        t1 = image.clone()
        t2 = torch.flip(image, dims=(-1,))
        mask = torch.ones_like(image)

        sample = {
            "images": image,
            "t1": t1,
            "t2": t2,
            "mask": mask,
            "metadata": {"index": index},
        }

        if self.device:
            sample = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }

        return sample
