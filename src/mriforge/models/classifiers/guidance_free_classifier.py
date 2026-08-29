import torch
import torch.nn.functional as F
from torch import nn

from mriforge.infrastructure.di.di_container import resolve_service
from mriforge.infrastructure.services import ILoggingService
from mriforge.models.interfaces.classifiers import IDiffusionClassifier
from mriforge.shared.utils.safe_io import safe_torch_load

# Remove module-level logger resolution
# logger = resolve_service(ILoggingService)


class GuidanceFreeClassifier(IDiffusionClassifier):
    """A guidance-free classifier for diffusion model outputs.

    This classifier operates independently of the diffusion process and can
    classify both real and generated medical images. It's designed to work
    with MRI and other medical imaging modalities.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        image_size: int = 256,
        backbone_type: str = "resnet",
        pretrained: bool = False,
        device: str | None = None,
    ):
        """Initialize the guidance-free classifier.

        Args:
            num_classes: Number of classes to classify
                (e.g., 2 for binary classification)
            input_channels: Number of input channels
                (1 for grayscale MRI, 3 for RGB)
            image_size: Input image size (assumed square)
            backbone_type: Type of backbone network
                ("resnet", "efficientnet", "vit")
            pretrained: Whether to use pretrained weights
            device: Device to run on

        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
        self.backbone_type = backbone_type
        self.pretrained = pretrained

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Build the model
        self.model = self._build_model()
        self.model.to(self.device)

        # Loss function for training
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.training_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    @property
    def logger(self):
        """Lazy logger resolution to avoid import-time DI issues."""
        return resolve_service(ILoggingService)

    def _build_model(self) -> nn.Module:
        """Build the classifier model based on backbone type."""
        if self.backbone_type == "cnn":
            return self._build_cnn_backbone()
        if self.backbone_type == "resnet":
            return self._build_resnet_backbone()
        if self.backbone_type == "efficientnet":
            return self._build_efficientnet_backbone()
        if self.backbone_type == "vit":
            return self._build_vit_backbone()
        raise ValueError(f"Unknown backbone type: {self.backbone_type}")

    def _build_cnn_backbone(self) -> nn.Module:
        """Build simple CNN-based classifier (for backward compatibility).

        This is a basic CNN architecture matching the simple_guidance_free_classifier
        for backward compatibility with existing code.
        """
        layers = []

        # Convolutional layers
        in_channels = self.input_channels
        for out_channels in [32, 64, 128, 256]:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes),
        )

        return nn.Sequential(*layers, self.classification_head)

    def _build_resnet_backbone(self) -> nn.Module:
        """Build ResNet-based classifier."""
        # Simple ResNet-like architecture for medical images
        layers = []

        # Initial convolution
        layers.append(
            nn.Conv2d(
                self.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
        )
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Residual blocks
        in_channels = 64
        for _i, out_channels in enumerate([64, 128, 256, 512]):
            layers.extend(self._make_resnet_block(in_channels, out_channels))
            in_channels = out_channels

        # Global average pooling and classification head
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes),
        )

        return nn.Sequential(*layers, self.classification_head)

    def _make_resnet_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> list[nn.Module]:
        """Create a simple residual block."""
        layers = []

        # First convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Second convolution
        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        layers.append(nn.BatchNorm2d(out_channels))

        return layers

    def _build_efficientnet_backbone(self) -> nn.Module:
        """Build EfficientNet-based classifier."""
        # Simplified EfficientNet-like architecture # IMPL
        layers = []

        # Stem
        layers.append(
            nn.Conv2d(
                self.input_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))

        # MBConv blocks (simplified) # IMPL
        config = [
            (32, 16, 1, 1),  # (in_channels, out_channels, expand_ratio, stride)
            (16, 24, 6, 2),
            (24, 40, 6, 2),
            (40, 80, 6, 1),
            (80, 112, 6, 2),
            (112, 192, 6, 1),
            (192, 320, 6, 2),
        ]

        for in_c, out_c, exp_r, stride in config:
            layers.extend(self._make_mbconv_block(in_c, out_c, exp_r, stride))

        # Head
        layers.append(nn.Conv2d(320, 1280, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(1280))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, self.num_classes),
        )

        return nn.Sequential(*layers, self.classification_head)

    def _make_mbconv_block(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
    ) -> list[nn.Module]:
        """Create a robust MBConv block with Squeeze-and-Excitation and residuals."""
        layers = []
        expanded_channels = in_channels * expand_ratio

        # Expansion
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            )
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.SiLU(inplace=True))  # EfficientNet uses SiLU

        # Depthwise convolution
        layers.append(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False,
            ),
        )
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.SiLU(inplace=True))

        # Squeeze-and-Excitation
        se_reduced = max(1, in_channels // 4)
        se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_reduced, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_reduced, expanded_channels, 1),
            nn.Sigmoid(),
        )
        # We can't simply append SE block as a layer in Sequential easily without a custom
        # Module that applies it. We'll construct a custom MBConv Module to handle the residual.

        class MBConvBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.expand = nn.Sequential(*layers) if expand_ratio != 1 else nn.Identity()
                # For depthwise, it's the last elements of layers list if expand is used.
                # Actually, let's redefine cleanly.

                self.use_residual = in_channels == out_channels and stride == 1

                self.block = nn.Sequential(
                    # Expand
                    nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
                    if expand_ratio != 1
                    else nn.Identity(),
                    nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else nn.Identity(),
                    nn.SiLU(inplace=True) if expand_ratio != 1 else nn.Identity(),
                    # Depthwise
                    nn.Conv2d(
                        expanded_channels,
                        expanded_channels,
                        3,
                        stride=stride,
                        padding=1,
                        groups=expanded_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(expanded_channels),
                    nn.SiLU(inplace=True),
                )
                self.se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(expanded_channels, max(1, in_channels // 4), 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(max(1, in_channels // 4), expanded_channels, 1),
                    nn.Sigmoid(),
                )
                self.project = nn.Sequential(
                    nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

            def forward(self, x):
                """forward method for MBConvBlock.

                Executes PyTorch tensor operations.

                Args:
                    x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                identity = x
                out = self.block(x)
                out = out * self.se(out)
                out = self.project(out)
                if self.use_residual:
                    out = out + identity
                return out

        return [MBConvBlock()]

    def _build_vit_backbone(self) -> nn.Module:
        """Build Vision Transformer-based classifier."""
        # Full ViT for medical images with class token
        patch_size = 16
        num_patches = (self.image_size // patch_size) ** 2
        embed_dim = 768
        num_heads = 12
        num_layers = 12

        class ViTClassifier(nn.Module):
            """Robust ViTClassifier with CLS token."""

            def __init__(
                self,
                input_channels: int,
                num_classes: int,
                image_size: int,
                patch_size: int,
                embed_dim: int,
                num_heads: int,
                num_layers: int,
            ):
                super().__init__()
                self.patch_embed = nn.Conv2d(
                    input_channels,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )

                # Sequence length is num_patches + 1 (for cls token)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

                self.transformer_blocks = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=embed_dim,
                            nhead=num_heads,
                            dim_feedforward=embed_dim * 4,
                            dropout=0.1,
                            batch_first=True,
                            activation="gelu",
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.classification_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, num_classes),
                )

            def forward(self, x):
                """forward method for ViTClassifier.

                Executes PyTorch tensor operations.

                Args:
                    x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                b = x.shape[0]
                x = self.patch_embed(x)  # (b, embed, h, w)
                x = x.flatten(2).transpose(1, 2)  # (b, num_patches, embed)

                # Append cls token
                cls_tokens = self.cls_token.expand(b, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)  # (b, num_patches + 1, embed)
                x = x + self.pos_embed

                for block in self.transformer_blocks:
                    x = block(x)

                # Extract the heavily contextualized CLS token for classification
                cls_out = x[:, 0]
                return self.classification_head(cls_out)

        return ViTClassifier(
            self.input_channels,
            self.num_classes,
            self.image_size,
            patch_size,
            embed_dim,
            num_heads,
            num_layers,
        )

    @property
    def name(self) -> str:
        """Return the model name."""
        return f"GuidanceFreeClassifier_{self.backbone_type}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier."""
        x = x.to(self.device)
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the classifier."""
        return self.forward(x)

    def classify(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Classify input images.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            **kwargs: Additional classification parameters

        Returns:
            Classification logits

        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(self.device))

        # Apply temperature scaling if specified
        temperature = kwargs.get("temperature", 1.0)
        if temperature != 1.0:
            logits = logits / temperature

        return logits

    def get_num_classes(self) -> int:
        """Returns the number of classes this classifier can predict."""
        return self.num_classes

    def get_classification_head(self) -> nn.Module:
        """Returns the classification head of the model."""
        return self.classification_head

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        return (input_shape[0], self.num_classes)

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
        logits = self.classify(generated_images, **kwargs)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]

        results = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "confidence": confidence,
        }

        # Add diffusion step information if provided
        if diffusion_steps is not None:
            results["diffusion_steps"] = diffusion_steps

        return results

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
        # Classify real and generated images
        real_results = self.classify_generated(real_images)
        gen_results = self.classify_generated(generated_images)

        # Calculate quality metrics
        metrics = {}

        # Classification accuracy on real images (should be high)
        real_accuracy = (real_results["predictions"] == 0).float().mean().item()
        metrics["real_classification_accuracy"] = real_accuracy

        # Distribution of predictions on generated images
        gen_pred_dist = torch.bincount(
            gen_results["predictions"],
            minlength=self.num_classes,
        )
        gen_pred_dist = gen_pred_dist.float() / gen_pred_dist.sum()
        metrics["generated_prediction_distribution"] = gen_pred_dist.tolist()

        # Average confidence on generated images
        metrics["generated_avg_confidence"] = gen_results["confidence"].mean().item()

        # KL divergence between real and generated prediction distributions
        real_pred_dist = torch.bincount(
            real_results["predictions"],
            minlength=self.num_classes,
        )
        real_pred_dist = real_pred_dist.float() / real_pred_dist.sum()

        kl_div = F.kl_div(gen_pred_dist.log(), real_pred_dist, reduction="sum").item()
        metrics["kl_divergence"] = kl_div

        # Quality score (higher is better)
        quality_score = real_accuracy * (1 - kl_div / 10)  # Normalize KL divergence
        metrics["quality_score"] = max(0, quality_score)

        return metrics

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            images: Input images
            labels: Ground truth labels
            optimizer: Optimizer to use

        Returns:
            Dictionary containing loss and accuracy

        """
        self.model.train()
        images, labels = images.to(self.device), labels.to(self.device)

        optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    def validate_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """Perform a validation step.

        Args:
            images: Input images
            labels: Ground truth labels

        Returns:
            Dictionary containing loss and accuracy

        """
        self.model.eval()
        images, labels = images.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "input_channels": self.input_channels,
                "image_size": self.image_size,
                "backbone_type": self.backbone_type,
                "training_metrics": self.training_metrics,
            },
            path,
        )
        self.logger.log_info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = safe_torch_load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Update attributes if they differ
        self.num_classes = checkpoint.get("num_classes", self.num_classes)
        self.input_channels = checkpoint.get("input_channels", self.input_channels)
        self.image_size = checkpoint.get("image_size", self.image_size)
        self.backbone_type = checkpoint.get("backbone_type", self.backbone_type)
        self.training_metrics = checkpoint.get(
            "training_metrics",
            self.training_metrics,
        )

        # Rebuild model if attributes changed
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.logger.log_info(f"Checkpoint loaded from {path}")
