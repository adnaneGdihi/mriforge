"""Classifier Interfaces - SOLID Design
===================================

This module defines interfaces for classifier models that work with
diffusion models. These classifiers are guidance-free and can operate
independently of the diffusion process.
"""

from abc import abstractmethod

import torch
from torch import nn

# Import the base IModel interface
from .models import IModel


class IClassifier(IModel):
    """Interface for classifier models that work with diffusion models."""

    @abstractmethod
    def classify(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Classify input images.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            **kwargs: Additional classification parameters

        Returns:
            Classification logits or probabilities

        """

    @abstractmethod
    def get_num_classes(self) -> int:
        """Returns the number of classes this classifier can predict."""

    @abstractmethod
    def get_classification_head(self) -> nn.Module:
        """Returns the classification head of the model."""


class IDiffusionClassifier(IClassifier):
    """Interface for classifiers specifically designed for diffusion
    model outputs.
    """

    @abstractmethod
    def classify_generated(
        self,
        generated_images: torch.Tensor,
        diffusion_steps: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Classify generated images from diffusion models.

        Args:
            generated_images: Generated images from diffusion model
            diffusion_steps: Optional diffusion step information
            **kwargs: Additional parameters

        Returns:
            Dictionary containing classification results and confidence scores

        """

    @abstractmethod
    def evaluate_generation_quality(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate the quality of generated images using classification metrics.

        Args:
            real_images: Real images for comparison
            generated_images: Generated images to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing quality metrics

        """
