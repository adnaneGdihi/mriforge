import torch
import torch.nn as nn


class UncertaintyEstimator(nn.Module):
    """
    Wraps a model to provide Epistemic Uncertainty estimation via Monte Carlo Dropout.

    References:
        Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning
        for computer vision? NeurIPS.
    """

    def __init__(self, model: nn.Module, num_samples: int = 16):
        """__init__.

        Args:
            model (nn.Module): Description.
            num_samples (int): Description.
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        """Force enable dropout layers during inference (MC Dropout)."""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def forward(
        self,
        x: torch.Tensor,
        kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with multiple MC samples.

        Args:
            x: Input tensor [B, C, H, W]
            kspace: Measured k-space [B, C, H, W, 2] (optional, for Data Consistency)
            mask: Sampling mask [B, C, H, W] (optional)

        Returns:
            Dictionary containing:
                - "reconstruction": Mean prediction [B, C, H, W]
                - "uncertainty": Variance map [B, C, H, W]
                - "cv_map": Coefficient of Variation map [B, C, H, W]

        forward method for UncertaintyEstimator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        predictions = []

        # Ensure dropout is active
        self.enable_dropout()

        with torch.no_grad():
            for _ in range(self.num_samples):
                # Forward pass
                # Handle flexible call signatures
                if kspace is not None and mask is not None:
                    pred = self.model(x, kspace=kspace, mask=mask, **kwargs)
                else:
                    pred = self.model(x, **kwargs)

                # Unwrap tuple if model returns (pred, hidden)
                if isinstance(pred, tuple):
                    pred = pred[0]

                predictions.append(pred)

        # Stack predictions: [N_samples, B, C, H, W]
        stack = torch.stack(predictions)

        # Compute Statistics
        mean_recon = torch.mean(stack, dim=0)

        # Variance as Uncertainty Map (Epistemic Uncertainty)
        uncertainty_map = torch.var(stack, dim=0)

        # Coefficient of Variation (normalized uncertainty)
        # CV = sigma / mu (adds stability epsilon)
        # Use mean magnitude for denominator stability
        mean_mag = torch.abs(mean_recon) + 1e-6
        std_dev = torch.sqrt(uncertainty_map)
        cv_map = std_dev / mean_mag

        return {
            "reconstruction": mean_recon,
            "uncertainty": uncertainty_map,
            "cv_map": cv_map,
            "std_dev": std_dev,
        }
