"""Phase 7.1: Loss Recommendation Engine
Intelligently suggests loss combinations for different tasks.

.. mermaid::

    classDiagram
        class LossRecommender {
            +recommend(task_type, model_type) RecommendationSet
        }
        class TaskAnalyzer {
            +analyze_task_type(description) TaskType
        }
        class RecommendationSet {
            +task_type: TaskType
            +losses: List~LossRecommendation~
            +suggested_config: LossConfigSchema
            +summary() str
        }
        class LossRecommendation {
            +name: str
            +weight: float
            +confidence: float
        }

        LossRecommender *-- TaskAnalyzer
        LossRecommender ..> RecommendationSet : creates
        RecommendationSet *-- LossRecommendation
"""

import logging
from dataclasses import dataclass
from enum import Enum

from spectramr.config.schemas.loss import LossConfigSchema


class TaskType(Enum):
    """Supported task types for loss recommendation."""

    RECONSTRUCTION = "reconstruction"
    SEGMENTATION = "segmentation"
    SYNTHESIS = "synthesis"
    SUPER_RESOLUTION = "super_resolution"
    DENOISING = "denoising"
    ARTIFACT_REMOVAL = "artifact_removal"
    INPAINTING = "inpainting"
    TRANSLATION = "translation"


@dataclass
class LossRecommendation:
    """Single loss recommendation."""

    name: str
    """Loss name (e.g., 'l1', 'ssim', 'perceptual')"""

    enabled: bool
    """Whether to enable this loss"""

    weight: float
    """Suggested weight"""

    reason: str
    """Reason for recommendation"""

    category: str
    """Loss category (reconstruction, adversarial, etc)"""

    confidence: float
    """Confidence score (0-1)"""


@dataclass
class RecommendationSet:
    """Complete set of loss recommendations."""

    task_type: TaskType
    """Detected task type"""

    losses: list[LossRecommendation]
    """Recommended losses"""

    primary_losses: list[str]
    """Names of primary losses (high confidence)"""

    auxiliary_losses: list[str]
    """Names of auxiliary losses (medium confidence)"""

    optional_losses: list[str]
    """Names of optional losses (low confidence, disabled)"""

    suggested_config: LossConfigSchema
    """Generated loss configuration"""

    notes: list[str]
    """Additional notes and recommendations"""

    @property
    def overall_confidence(self) -> float:
        """Overall confidence in recommendations (0-1)."""
        if not self.losses:
            return 0.0
        return sum(l.confidence for l in self.losses) / len(self.losses)

    def summary(self) -> str:
        """Get summary of recommendations."""
        lines = [
            f"Task: {self.task_type.value}",
            f"Overall Confidence: {self.overall_confidence:.1%}",
            f"Primary Losses: {', '.join(self.primary_losses)}",
            f"Auxiliary Losses: {', '.join(self.auxiliary_losses)}",
        ]

        if self.optional_losses:
            lines.append(f"Optional Losses: {', '.join(self.optional_losses)}")

        if self.notes:
            lines.append("\nNotes:")
            for note in self.notes:
                lines.append(f"  - {note}")

        return "\n".join(lines)


class TaskAnalyzer:
    """Analyzes task characteristics.

    .. mermaid::

        graph TD
            Description[Task Description] --> Analysis{Keywords?}
            Model[Model Type] --> Analysis

            Analysis -->|reconstruction| REC[TaskType.RECONSTRUCTION]
            Analysis -->|segment/mask| SEG[TaskType.SEGMENTATION]
            Analysis -->|synthesis/generate| SYN[TaskType.SYNTHESIS]
            Analysis -->|super resolution| SR[TaskType.SUPER_RESOLUTION]
            Analysis -->|denoise/noise| DEN[TaskType.DENOISING]
            Analysis -->|artifact| ART[TaskType.ARTIFACT_REMOVAL]
            Analysis -->|inpaint| INP[TaskType.INPAINTING]
            Analysis -->|translation/domain| TRA[TaskType.TRANSLATION]

            Analysis -->|None| Default[TaskType.RECONSTRUCTION]
    """

    def __init__(self):
        """Initialize task analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_task_type(
        self,
        task_description: str = None,
        model_type: str = None,
        output_type: str = None,
    ) -> TaskType:
        """Analyze and detect task type.

        Args:
            task_description: Text description of task
            model_type: Type of model ('unet', 'gan', 'diffusion', etc)
            output_type: Output type ('image', 'mask', 'volume', etc)

        Returns:
            Detected TaskType
        """
        # Simple heuristic-based detection
        if task_description:
            desc_lower = task_description.lower()

            if "reconstruction" in desc_lower:
                return TaskType.RECONSTRUCTION
            elif "segment" in desc_lower:
                return TaskType.SEGMENTATION
            elif "synthesis" in desc_lower or "generate" in desc_lower:
                return TaskType.SYNTHESIS
            elif "super" in desc_lower and "resolution" in desc_lower:
                return TaskType.SUPER_RESOLUTION
            elif "denoise" in desc_lower or "noise" in desc_lower:
                return TaskType.DENOISING
            elif "artifact" in desc_lower:
                return TaskType.ARTIFACT_REMOVAL
            elif "inpaint" in desc_lower:
                return TaskType.INPAINTING
            elif "translation" in desc_lower or "domain" in desc_lower:
                return TaskType.TRANSLATION

        # Default based on model type - O(1) registry lookup
        MODEL_TYPE_TO_TASK = {
            "gan": TaskType.SYNTHESIS,
            "diffusion": TaskType.SYNTHESIS,
            "vae": TaskType.SYNTHESIS,
            "flow": TaskType.SYNTHESIS,
        }
        return MODEL_TYPE_TO_TASK.get(model_type, TaskType.RECONSTRUCTION)


class LossRecommender:
    """Recommends loss configurations based on task."""

    def __init__(self):
        """Initialize loss recommender."""
        self.logger = logging.getLogger(__name__)
        self.analyzer = TaskAnalyzer()

        # Loss recommendations per task
        self._task_recommendations = self._init_recommendations()

    def _init_recommendations(self) -> dict[TaskType, list[LossRecommendation]]:
        """Initialize task-specific recommendations."""
        return {
            TaskType.RECONSTRUCTION: [
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=10.0,
                    reason="Primary pixel-level reconstruction loss",
                    category="reconstruction",
                    confidence=0.95,
                ),
                LossRecommendation(
                    name="ssim",
                    enabled=True,
                    weight=0.1,
                    reason="Structural similarity for perceptual quality",
                    category="reconstruction",
                    confidence=0.85,
                ),
                LossRecommendation(
                    name="perceptual",
                    enabled=False,
                    weight=0.1,
                    reason="Optional: VGG features for high-quality results",
                    category="reconstruction",
                    confidence=0.70,
                ),
            ],
            TaskType.SEGMENTATION: [
                LossRecommendation(
                    name="dice",
                    enabled=True,
                    weight=1.0,
                    reason="Primary loss for segmentation tasks",
                    category="reconstruction",
                    confidence=0.90,
                ),
                LossRecommendation(
                    name="cross_entropy",
                    enabled=True,
                    weight=1.0,
                    reason="Class probability loss",
                    category="reconstruction",
                    confidence=0.85,
                ),
            ],
            TaskType.SYNTHESIS: [
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=10.0,
                    reason="Reconstruction loss for content preservation",
                    category="reconstruction",
                    confidence=0.85,
                ),
                LossRecommendation(
                    name="perceptual",
                    enabled=True,
                    weight=0.1,
                    reason="Feature-level perceptual similarity",
                    category="reconstruction",
                    confidence=0.80,
                ),
                LossRecommendation(
                    name="gan",
                    enabled=True,
                    weight=1.0,
                    reason="Adversarial loss for realistic generation",
                    category="adversarial",
                    confidence=0.90,
                ),
            ],
            TaskType.SUPER_RESOLUTION: [
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=10.0,
                    reason="Pixel-level reconstruction",
                    category="reconstruction",
                    confidence=0.90,
                ),
                LossRecommendation(
                    name="perceptual",
                    enabled=True,
                    weight=0.1,
                    reason="Perceptual loss for visual quality",
                    category="reconstruction",
                    confidence=0.85,
                ),
                LossRecommendation(
                    name="gan",
                    enabled=False,
                    weight=1.0,
                    reason="Optional: for photorealistic enhancement",
                    category="adversarial",
                    confidence=0.75,
                ),
            ],
            TaskType.DENOISING: [
                LossRecommendation(
                    name="mse",
                    enabled=True,
                    weight=1.0,
                    reason="Primary denoising loss",
                    category="reconstruction",
                    confidence=0.95,
                ),
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=1.0,
                    reason="Robustness to outliers",
                    category="reconstruction",
                    confidence=0.80,
                ),
            ],
            TaskType.ARTIFACT_REMOVAL: [
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=10.0,
                    reason="Primary artifact removal loss",
                    category="reconstruction",
                    confidence=0.90,
                ),
                LossRecommendation(
                    name="ssim",
                    enabled=True,
                    weight=0.1,
                    reason="Structural similarity preservation",
                    category="reconstruction",
                    confidence=0.80,
                ),
                LossRecommendation(
                    name="perceptual",
                    enabled=True,
                    weight=0.1,
                    reason="Perceptual quality preservation",
                    category="reconstruction",
                    confidence=0.75,
                ),
            ],
            TaskType.INPAINTING: [
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=10.0,
                    reason="Content reconstruction in missing regions",
                    category="reconstruction",
                    confidence=0.90,
                ),
                LossRecommendation(
                    name="perceptual",
                    enabled=True,
                    weight=0.1,
                    reason="Consistency with surrounding regions",
                    category="reconstruction",
                    confidence=0.80,
                ),
                LossRecommendation(
                    name="gan",
                    enabled=True,
                    weight=1.0,
                    reason="Realistic completion",
                    category="adversarial",
                    confidence=0.80,
                ),
            ],
            TaskType.TRANSLATION: [
                LossRecommendation(
                    name="l1",
                    enabled=True,
                    weight=10.0,
                    reason="Content preservation",
                    category="reconstruction",
                    confidence=0.85,
                ),
                LossRecommendation(
                    name="perceptual",
                    enabled=True,
                    weight=0.1,
                    reason="Style transfer similarity",
                    category="reconstruction",
                    confidence=0.80,
                ),
                LossRecommendation(
                    name="gan",
                    enabled=True,
                    weight=1.0,
                    reason="Domain-specific realism",
                    category="adversarial",
                    confidence=0.90,
                ),
            ],
        }

    def recommend(
        self,
        task_type: TaskType = None,
        task_description: str = None,
        model_type: str = None,
        output_type: str = None,
        use_gan: bool = None,
        use_perceptual: bool = None,
    ) -> RecommendationSet:
        """Get loss recommendations for task.

        Args:
            task_type: Explicit task type (if known)
            task_description: Text description of task
            model_type: Type of model
            output_type: Type of output
            use_gan: Whether to use GAN losses (override)
            use_perceptual: Whether to use perceptual losses (override)

        Returns:
            RecommendationSet with recommendations
        """
        # Detect task type if not provided
        if task_type is None:
            task_type = self.analyzer.analyze_task_type(task_description, model_type, output_type)

        self.logger.info(f"Recommending losses for task: {task_type.value}")

        # Get base recommendations
        base_recs = self._task_recommendations.get(task_type, [])

        # Apply overrides
        losses = self._apply_overrides(base_recs, use_gan, use_perceptual)

        # Categorize losses
        primary = [l.name for l in losses if l.enabled and l.confidence >= 0.85]
        auxiliary = [l.name for l in losses if l.enabled and l.confidence < 0.85]
        optional = [l.name for l in losses if not l.enabled]

        # Generate configuration
        config = self._generate_config(losses)

        # Generate notes
        notes = self._generate_notes(task_type, losses)

        return RecommendationSet(
            task_type=task_type,
            losses=losses,
            primary_losses=primary,
            auxiliary_losses=auxiliary,
            optional_losses=optional,
            suggested_config=config,
            notes=notes,
        )

    def _apply_overrides(
        self,
        losses: list[LossRecommendation],
        use_gan: bool | None,
        use_perceptual: bool | None,
    ) -> list[LossRecommendation]:
        """Apply user overrides to recommendations.

        Args:
            losses: Base recommendations
            use_gan: Override for GAN losses
            use_perceptual: Override for perceptual losses

        Returns:
            Modified recommendations
        """
        result = []

        for loss in losses:
            modified = LossRecommendation(
                name=loss.name,
                enabled=loss.enabled,
                weight=loss.weight,
                reason=loss.reason,
                category=loss.category,
                confidence=loss.confidence,
            )

            if use_gan is not None and loss.category == "adversarial":
                modified.enabled = use_gan

            if use_perceptual is not None and "perceptual" in loss.name:
                modified.enabled = use_perceptual

            result.append(modified)

        return result

    def _generate_config(self, losses: list[LossRecommendation]) -> LossConfigSchema:
        """Generate loss configuration from recommendations.

        Args:
            losses: Recommended losses

        Returns:
            LossConfigSchema
        """
        from spectramr.config.schemas.loss import (
            GANLossesConfig,
            MetricsConfig,
            ReconstructionLossesConfig,
        )

        # Extract enabled losses from recommendations
        l1_enabled = any(l.name == "l1" and l.enabled for l in losses)
        l2_enabled = any(l.name == "l2" and l.enabled for l in losses)
        ssim_enabled = any(l.name == "ssim" and l.enabled for l in losses)
        perceptual_enabled = any(l.name == "perceptual" and l.enabled for l in losses)
        lpips_enabled = any(l.name == "lpips" and l.enabled for l in losses)
        gan_enabled = any(l.name == "gan" and l.enabled for l in losses)

        # Create reconstruction config with enabled losses
        reconstruction_config = ReconstructionLossesConfig(
            enable_l1=l1_enabled,
            enable_l2=l2_enabled,
            enable_ssim=ssim_enabled,
            enable_perceptual=perceptual_enabled,
            enable_lpips=lpips_enabled,
        )

        # Create GAN config
        gan_config = GANLossesConfig(
            enable_adversarial=gan_enabled,
        )

        # Create metrics config to track enabled metrics
        metrics_config = MetricsConfig(
            compute_ssim=ssim_enabled,
            compute_psnr=l1_enabled or l2_enabled,
            compute_lpips=lpips_enabled,
        )

        # Create unified config
        return LossConfigSchema(
            reconstruction=reconstruction_config,
            gan=gan_config,
            metrics=metrics_config,
        )

    def _generate_notes(self, task_type: TaskType, losses: list[LossRecommendation]) -> list[str]:
        """Generate informational notes.

        Args:
            task_type: Task type
            losses: Recommended losses

        Returns:
            List of notes
        """
        notes = [
            f"These recommendations are optimized for {task_type.value} tasks.",
            "Adjust loss weights based on validation performance.",
        ]

        # Warn about missing recommendations
        if not any(l.enabled for l in losses):
            notes.append("Warning: No losses enabled. Please enable at least one loss.")

        # Suggest GAN if not present
        if not any(l.name == "gan" and l.enabled for l in losses):
            notes.append("Consider enabling GAN loss if generation quality is not satisfactory.")

        # Suggest curriculum learning
        if len([l for l in losses if l.enabled]) > 2:
            notes.append(
                "With multiple losses, consider curriculum learning: enable complex losses after initial training."
            )

        return notes


# Convenience function


def get_loss_recommendation(
    task_type: TaskType = None,
    task_description: str = None,
    **kwargs,
) -> RecommendationSet:
    """Get loss recommendations for a task.

    Args:
        task_type: Task type (if known)
        task_description: Task description
        **kwargs: Additional parameters

    Returns:
        RecommendationSet
    """
    recommender = LossRecommender()
    return recommender.recommend(task_type=task_type, task_description=task_description, **kwargs)
