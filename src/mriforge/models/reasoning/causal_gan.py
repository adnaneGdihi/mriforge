"""
Causal-GAN: Counterfactual Generation via Structural Causal Models.

Implements a generator conditioned on a Causal Graph.
SCM: Age -> Pathology -> Image.
Allows 'do(Pathology=1)' interventions.
"""

import torch
import torch.nn as nn

from mriforge.models.registry import register_model


class StructuralCausalModel(nn.Module):
    """
    Simulates the causal graph mechanism (Priors).
    Graph: Age -> Pathology
    """

    def __init__(self):
        """__init__."""
        super().__init__()
        # P(Age) ~ Uniform(0, 1) or Normal
        # P(Pathology | Age) = Sigmoid(a * Age + b)

        self.age_to_pathology = nn.Linear(1, 1)  # simple logistic regression

    def sample(self, batch_size: int) -> dict:
        """Sample from the joint distribution P(Age, Pathology)."""
        device = self.age_to_pathology.weight.device

        # 1. Sample Age
        age = torch.rand(batch_size, 1, device=device)  # Normalized age 0-1

        # 2. Sample Pathology | Age
        logits = self.age_to_pathology(age)
        probs = torch.sigmoid(logits)
        pathology = torch.bernoulli(probs)

        return {"age": age, "pathology": pathology}


class CausalGenerator(nn.Module):
    """
    Generator G(z, c) where c = {Age, Pathology}.
    """

    def __init__(self, latent_dim: int = 64, image_channels: int = 1):
        """__init__.

        Args:
            latent_dim (int): Description.
            image_channels (int): Description.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Condition encoders
        self.age_embed = nn.Linear(1, 16)
        self.pathology_embed = nn.Embedding(2, 16)

        # Generator Backbone (Simple MLP/CNN for prototype)
        # Input: z (64) + age (16) + path (16) = 96
        self.fc = nn.Linear(latent_dim + 32, 128 * 8 * 8)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 3, 1, 1),  # 32x32
            nn.Tanh(),
        )

        self.scm = StructuralCausalModel()

    def forward(self, z: torch.Tensor, age: torch.Tensor, pathology: torch.Tensor) -> torch.Tensor:
        """
        Generate image from latents and causal parents.

        forward method for CausalGenerator.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            age (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            pathology (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_size = z.shape[0]

        # Embed conditions
        age_emb = self.age_embed(age)
        path_emb = self.pathology_embed(pathology.long().squeeze(-1))

        # Concatenate
        x = torch.cat([z, age_emb, path_emb], dim=1)

        # Generate
        x = self.fc(x).view(batch_size, 128, 8, 8)
        img = self.conv_blocks(x)
        return img

    def do_intervention(self, n_samples: int, target_pathology: int = 1) -> torch.Tensor:
        """
        Perform do(Pathology = target).
        We sample Age naturally, but Force Pathology.
        """
        device = self.age_embed.weight.device

        # 1. Sample Noise
        z = torch.randn(n_samples, self.latent_dim, device=device)

        # 2. Sample Parents (Age) naturally from SCM
        parents = self.scm.sample(n_samples)
        age = parents["age"]

        # 3. Intervene on Pathology
        # Ignore SCM's pathology, set to target
        pathology = torch.full((n_samples, 1), float(target_pathology), device=device)

        # 4. Generate
        return self.forward(z, age, pathology)


@register_model(name="causal_gan", training_mode="synthesis")
class CausalGAN(nn.Module):
    """Wrapper for the Causal System."""

    def __init__(self):
        """__init__."""
        super().__init__()
        self.generator = CausalGenerator()

    def forward(self, batch_size: int = 1):
        """Standard sampling from observational distribution.

        forward method for CausalGAN.

        Executes PyTorch tensor operations.

        Args:
            batch_size (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        parents = self.generator.scm.sample(batch_size)
        z = torch.randn(batch_size, self.generator.latent_dim, device=parents["age"].device)
        return self.generator(z, parents["age"], parents["pathology"])


def create_causal_gan(**kwargs):
    """create_causal_gan.

    Returns:
        Any: Description.
    """
    return CausalGAN(**kwargs)
