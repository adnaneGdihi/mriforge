"""Bloch Dictionary for MRF-DiPh
================================

Pre-computed dictionary of Bloch responses for fast tissue parameter
estimation via dot-product matching.

The dictionary stores signal intensities for a grid of (T1, T2, PD, B0) values,
enabling O(1) lookup of the closest tissue parameters for any observed signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlochDictionary(nn.Module):
    """Pre-computed dictionary of Bloch responses for fast matching.

    This dictionary pre-computes MRI signal responses for a grid of tissue
    parameters (T1, T2, PD). During inference, we find the best-matching
    entry via normalized dot product.

    Physical Basis:
    - Spin Echo: S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)
    - Gradient Echo: S = PD * sin(α) * (1-E1)/(1-cos(α)*E1) * exp(-TE/T2*)

    Args:
        T1_range: (min, max) for T1 in milliseconds
        T2_range: (min, max) for T2 in milliseconds
        PD_range: (min, max) for proton density (normalized)
        T1_steps: Number of T1 values in grid
        T2_steps: Number of T2 values in grid
        PD_steps: Number of PD values in grid
        sequence_type: "SE" (Spin Echo) or "GRE" (Gradient Recalled Echo)
        device: Compute device
    """

    def __init__(
        self,
        T1_range: tuple[float, float] = (200.0, 4000.0),
        T2_range: tuple[float, float] = (10.0, 500.0),
        PD_range: tuple[float, float] = (0.1, 1.0),
        T1_steps: int = 50,
        T2_steps: int = 50,
        PD_steps: int = 10,
        sequence_type: str = "SE",
        device: torch.device | None = None,
    ):
        """__init__.

        Args:
            T1_range (tuple[float, float]): Description.
            T2_range (tuple[float, float]): Description.
            PD_range (tuple[float, float]): Description.
            T1_steps (int): Description.
            T2_steps (int): Description.
            PD_steps (int): Description.
            sequence_type (str): Description.
            device (Optional[torch.device]): Description.
        """
        super().__init__()

        self.T1_range = T1_range
        self.T2_range = T2_range
        self.PD_range = PD_range
        self.T1_steps = T1_steps
        self.T2_steps = T2_steps
        self.PD_steps = PD_steps
        self.sequence_type = sequence_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Total dictionary size
        self.num_atoms = T1_steps * T2_steps * PD_steps

        # Generate parameter grids
        self._generate_parameter_grids()

    def _generate_parameter_grids(self) -> None:
        """Generate the parameter lookup grids."""
        # Create logarithmically-spaced T1 and T2 (better coverage of biological range)
        T1_values = torch.logspace(
            torch.log10(torch.tensor(self.T1_range[0])),
            torch.log10(torch.tensor(self.T1_range[1])),
            self.T1_steps,
        )
        T2_values = torch.logspace(
            torch.log10(torch.tensor(self.T2_range[0])),
            torch.log10(torch.tensor(self.T2_range[1])),
            self.T2_steps,
        )
        PD_values = torch.linspace(self.PD_range[0], self.PD_range[1], self.PD_steps)

        # Create meshgrid of all parameter combinations
        T1_grid, T2_grid, PD_grid = torch.meshgrid(T1_values, T2_values, PD_values, indexing="ij")

        # Flatten to [num_atoms, 3] parameter matrix
        self.register_buffer(
            "params",
            torch.stack(
                [
                    T1_grid.flatten(),
                    T2_grid.flatten(),
                    PD_grid.flatten(),
                ],
                dim=1,
            ),  # [num_atoms, 3] -> (T1, T2, PD)
        )

        # Store individual grids for reverse lookup
        self.register_buffer("T1_values", T1_values)
        self.register_buffer("T2_values", T2_values)
        self.register_buffer("PD_values", PD_values)

    def compute_dictionary(
        self,
        TR: float,
        TE: float,
        alpha: float = 90.0,
    ) -> torch.Tensor:
        """Compute dictionary signals for given sequence parameters.

        Args:
            TR: Repetition time in ms
            TE: Echo time in ms
            alpha: Flip angle in degrees (for GRE)

        Returns:
            Dictionary signals [num_atoms, 1]
        """
        T1 = self.params[:, 0]
        T2 = self.params[:, 1]
        PD = self.params[:, 2]

        # Avoid division by zero
        T1 = T1 + 1e-6
        T2 = T2 + 1e-6

        E1 = torch.exp(-TR / T1)
        E2 = torch.exp(-TE / T2)

        if self.sequence_type == "SE":
            # Spin Echo: S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)
            signal = PD * (1 - E1) * E2
        elif self.sequence_type == "GRE":
            # Gradient Echo with Ernst angle formula
            alpha_rad = alpha * 3.14159 / 180.0
            sin_alpha = torch.sin(torch.tensor(alpha_rad))
            cos_alpha = torch.cos(torch.tensor(alpha_rad))
            signal = PD * sin_alpha * (1 - E1) / (1 - cos_alpha * E1 + 1e-8) * E2
        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")

        return signal.unsqueeze(1)  # [num_atoms, 1]

    def match(
        self,
        signal: torch.Tensor,
        TR: float,
        TE: float,
        alpha: float = 90.0,
    ) -> torch.Tensor:
        """Find tissue parameters that best match observed signal.

        Uses L2 distance matching.

        Args:
            signal: Observed signal [B, 1, H, W]
            TR: Repetition time in ms
            TE: Echo time in ms
            alpha: Flip angle in degrees

        Returns:
            Tissue maps [B, 3, H, W] with channels (T1, T2, PD)
        """
        B, C, *spatial = signal.shape
        device = signal.device

        # Compute dictionary for current sequence parameters
        dictionary = self.compute_dictionary(TR, TE, alpha).to(device)  # [num_atoms, 1]
        dictionary = dictionary.squeeze(1)  # [num_atoms]

        # Flatten spatial dimensions
        signal_flat = signal.view(B, -1)  # [B, SpatialElements]

        # Compute L2 distance: [B, H*W, 1] - [1, 1, num_atoms] -> [B, H*W, num_atoms]
        diff = signal_flat.unsqueeze(2) - dictionary.unsqueeze(0).unsqueeze(0)
        dist = diff**2  # [B, H*W, num_atoms]

        # Find best matching atom for each pixel (minimum distance)
        best_idx = dist.argmin(dim=2)  # [B, H*W]

        # Lookup tissue parameters
        params = self.params.to(device)  # [num_atoms, 3]
        tissue_maps = params[best_idx.view(-1)]  # [B*Spatial, 3]
        # Reshape to [B, *spatial, 3] then permute to [B, 3, *spatial]
        tissue_maps = tissue_maps.view(B, *spatial, 3).permute(0, -1, *range(1, 1 + len(spatial)))

        return tissue_maps

    def match_soft(
        self,
        signal: torch.Tensor,
        TR: float,
        TE: float,
        alpha: float = 90.0,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Soft matching with temperature-scaled softmax.

        Returns weighted average of parameters (differentiable).

        Args:
            signal: Observed signal [B, 1, H, W]
            TR: Repetition time in ms
            TE: Echo time in ms
            alpha: Flip angle in degrees
            temperature: Softmax temperature (lower = harder)

        Returns:
            Tissue maps [B, 3, H, W]
        """
        B, C, *spatial = signal.shape
        device = signal.device

        # Compute dictionary
        dictionary = self.compute_dictionary(TR, TE, alpha).to(device)  # [num_atoms, 1]
        dictionary = dictionary.squeeze(1)  # [num_atoms]

        # Flatten signal
        signal_flat = signal.view(B, -1)  # [B, H*W]

        # L2 distance: [B, H*W, 1] - [1, 1, num_atoms] -> [B, H*W, num_atoms]
        diff = signal_flat.unsqueeze(2) - dictionary.unsqueeze(0).unsqueeze(0)
        neg_dist = -(diff**2) / temperature  # [B, H*W, num_atoms]

        # Softmax weights
        weights = F.softmax(neg_dist, dim=2)  # [B, H*W, num_atoms]

        # Weighted sum of parameters: [B, H*W, num_atoms] @ [num_atoms, 3] -> [B, H*W, 3]
        # Weighted sum of parameters
        params = self.params.to(device)  # [num_atoms, 3]
        tissue_maps = torch.matmul(weights, params)  # [B, Spatial, 3]
        tissue_maps = tissue_maps.view(B, *spatial, 3).permute(0, -1, *range(1, 1 + len(spatial)))

        return tissue_maps


def create_bloch_dictionary(
    sequence_type: str = "SE",
    resolution: str = "medium",
    device: torch.device | None = None,
) -> BlochDictionary:
    """Factory function to create Bloch dictionary with preset resolutions.

    Args:
        sequence_type: "SE" or "GRE"
        resolution: "low" (fast), "medium" (balanced), "high" (accurate)
        device: Compute device

    Returns:
        Configured BlochDictionary
    """
    configs = {
        "low": {"T1_steps": 20, "T2_steps": 20, "PD_steps": 5},
        "medium": {"T1_steps": 50, "T2_steps": 50, "PD_steps": 10},
        "high": {"T1_steps": 100, "T2_steps": 100, "PD_steps": 20},
    }

    if resolution not in configs:
        raise ValueError(f"Resolution must be one of {list(configs.keys())}")

    return BlochDictionary(
        sequence_type=sequence_type,
        device=device,
        **configs[resolution],
    )
