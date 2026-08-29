import torch


class CoordinateSampler:
    """
    Utility for sampling coordinates for Implicit Neural Representation (INR) training.

    Provides methods to sample regular grids or random points within the normalized
    [-1, 1] k-space/image domain.
    """

    @staticmethod
    def get_full_grid(
        batch_size: int, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """
        Generate a regular grid for all pixels in an image.

        Args:
            batch_size: Number of samples in batch.
            shape: (H, W) of the grid.
            device: Target device.

        Returns:
            coords: (B, H*W, 2) tensor in [-1, 1].
        """
        H, W = shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        # Flatten to (H*W, 2)
        coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
        # Expand to batch size
        return coords.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def sample_random_points(batch_size: int, n_points: int, device: torch.device) -> torch.Tensor:
        """
        Sample random continuous points in the domain [-1, 1].

        Args:
            batch_size: Number of samples in batch.
            n_points: Number of points to sample per batch member.
            device: Target device.

        Returns:
            coords: (B, n_points, 2) tensor in [-1, 1].
        """
        return torch.rand(batch_size, n_points, 2, device=device) * 2.0 - 1.0

    @staticmethod
    def sample_roi(
        batch_size: int,
        n_points: int,
        center: tuple[float, float],
        scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample points within a specific Region of Interest (ROI).

        Args:
            batch_size: Batch size.
            n_points: Points per batch item.
            center: (x, y) center of ROI in [-1, 1].
            scale: Size of the ROI relative to the whole domain.
            device: Target device.
        """
        offsets = (torch.rand(batch_size, n_points, 2, device=device) - 0.5) * 2.0 * scale
        center_tensor = torch.tensor(center, device=device).view(1, 1, 2)
        return (center_tensor + offsets).clamp(-1, 1)
