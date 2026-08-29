"""SSL Inference Strategy

Inference strategy for Self-Supervised Learning models.
Handles feature extraction and representation learning.
"""

from typing import Any

import torch
from torch import nn

from mriforge.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy


class SSLInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for SSL models.

    Handles feature extraction for models trained with self-supervised
    objectives (BYOL, MAE, SimCLR, etc.).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize SSL inference strategy.

        Args:
            model: The SSL model (encoder backbone)
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)
        self.ssl_config = self.config.get("ssl", {})

        # SSL-specific parameters
        self.method = self.ssl_config.get("method", "byol")  # byol, mae, simclr, etc.
        self.feature_dim = self.ssl_config.get("feature_dim", 512)
        self.use_projection_head = self.ssl_config.get("use_projection_head", False)

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.ssl

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor for SSL inference.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor ready for SSL feature extraction
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        input_tensor = input_tensor.detach()

        # Apply SSL-specific preprocessing (e.g., augmentations during training)
        # During inference, we typically use minimal preprocessing
        if hasattr(self.model, "preprocess_input"):
            input_tensor = self.model.preprocess_input(input_tensor)

        return input_tensor

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run SSL feature extraction.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters

        Returns:
            Feature representations, optionally with metadata
        """
        with torch.no_grad():
            # Extract features using the encoder
            if hasattr(self.model, "encode"):
                features = self.model.encode(input_tensor)
            elif hasattr(self.model, "backbone"):
                features = self.model.backbone(input_tensor)
            else:
                # Direct forward pass
                features = self.model(input_tensor)

            # Handle different model output formats
            if isinstance(features, dict):
                if "features" in features:
                    features = features["features"]
                elif "representation" in features:
                    features = features["representation"]
                elif "embedding" in features:
                    features = features["embedding"]
                else:
                    # Take the last tensor output
                    features = list(features.values())[-1]

            # Apply projection head if requested
            if self.use_projection_head and hasattr(self.model, "projection_head"):
                projected_features = self.model.projection_head(features)
                feature_info = {
                    "features": features.detach().cpu(),
                    "projected_features": projected_features.detach().cpu(),
                }
                return projected_features, feature_info
            else:
                feature_info = {
                    "features": features.detach().cpu(),
                }
                return features, feature_info

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess SSL features.

        Args:
            output_tensor: Raw feature output
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed features with metadata
        """
        if isinstance(output_tensor, tuple):
            features, feature_info = output_tensor
        else:
            features = output_tensor
            feature_info = {"features": features.detach().cpu()}

        # Ensure features are detached
        features = features.detach().cpu()

        # Apply L2 normalization if specified
        if self.ssl_config.get("normalize_features", True):
            features = torch.nn.functional.normalize(features, p=2, dim=-1)

        # Update feature info
        feature_info["features"] = features
        feature_info["feature_dim"] = features.shape[-1]
        feature_info["batch_size"] = features.shape[0]

        return features, feature_info

    def extract_features(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Extract features from input tensor.

        Args:
            input_tensor: Input tensor for feature extraction
            **kwargs: Additional extraction parameters

        Returns:
            Tuple of (features, metadata)
        """
        return self.infer(input_tensor, **kwargs)

    def get_similarity_matrix(
        self, features1: torch.Tensor, features2: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute similarity matrix between two sets of features.

        Args:
            features1: First set of features (N, D)
            features2: Second set of features (M, D)
            **kwargs: Additional similarity parameters

        Returns:
            Similarity matrix (N, M)
        """
        # Ensure features are normalized
        features1 = torch.nn.functional.normalize(features1, p=2, dim=-1)
        features2 = torch.nn.functional.normalize(features2, p=2, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(features1, features2.t())

        return similarity

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this SSL inference strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                "ssl_config": self.ssl_config,
                "method": self.method,
                "feature_dim": self.feature_dim,
                "use_projection_head": self.use_projection_head,
            }
        )
        return info

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run inference on a single input tensor.

        Args:
            input_data: Single input tensor
            **kwargs: Additional parameters

        Returns:
            Output tensor
        """
        return self.infer(input_data, **kwargs)

    @torch.no_grad()
    def infer_batch(
        self, input_data_list: list[torch.Tensor], batch_size: int = 4, **kwargs
    ) -> list[torch.Tensor]:
        """Run inference on a batch of input tensors.

        Args:
            input_data_list: List of input tensors
            batch_size: Batch size for processing
            **kwargs: Additional parameters

        Returns:
            List of output tensors
        """
        results = []
        for input_data in input_data_list:
            result = self.infer_single(input_data, **kwargs)
            results.append(result)
        return results
