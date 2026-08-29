import torch
import torch.nn as nn


class BlochConstraint(nn.Module):
    """
    Enforces physical constraints based on Bloch equations for MRI.

    Principally ensures:
    1. T1, T2 > 0 (Positivity)
    2. T1 >= T2 (Relaxation ordering)

    Can be applied as a soft loss regularizer or a hard constraint (clamping).
    """

    def __init__(
        self,
        t1_min: float = 1e-3,
        t1_max: float = 5000.0,
        apply_t1_t2_ordering: bool = True,
    ):
        """__init__.

        Args:
            t1_min (float): Description.
            t1_max (float): Description.
            apply_t1_t2_ordering (bool): Description.
        """
        super().__init__()
        self.t1_min = t1_min
        self.t1_max = t1_max
        self.apply_t1_t2_ordering = apply_t1_t2_ordering

    def forward(
        self, t1: torch.Tensor, t2: torch.Tensor, pd: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute violation loss.
        Average violation per voxel/gaussian.

        forward method for BlochConstraint.

        Executes PyTorch tensor operations.

        Args:
            t1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            pd (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Positivity / Range Constraints
        # ReLU(min - val) + ReLU(val - max)
        # Accumulate as TENSORS — the previous ``.item()`` per term detached the
        # graph, so this "soft regularizer" contributed zero gradient, and the
        # trailing ``torch.nan_to_num(<float>)`` then raised TypeError.
        t1_violation = torch.relu(self.t1_min - t1) + torch.relu(t1 - self.t1_max)
        t2_violation = torch.relu(self.t1_min - t2)  # T2 usually follows similar lower bound
        # T2 max is implicitly T1

        loss = torch.mean(t1_violation + t2_violation)

        # Ordering Constraint: T1 >= T2
        if self.apply_t1_t2_ordering:
            # Violation if T2 > T1 => ReLU(T2 - T1)
            ordering_violation = torch.relu(t2 - t1)
            loss = loss + torch.mean(ordering_violation)

        if pd is not None:
            # PD (Proton Density) > 0
            pd_violation = torch.relu(-pd)
            loss = loss + torch.mean(pd_violation)

        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0)
        return loss

    def enforce(self, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Hard enforcement (clamping/correction).
        Returns valid tensors.
        """
        with torch.no_grad():
            t1 = t1.clamp(min=self.t1_min, max=self.t1_max)
            t2 = t2.clamp(min=self.t1_min)

            if self.apply_t1_t2_ordering:
                # Ensure T2 <= T1
                t2 = torch.min(t2, t1)

        return t1, t2
