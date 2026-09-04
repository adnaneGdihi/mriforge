from typing import Protocol

from torch import Tensor

# We use ComplexTensor as a type hint alias for Tensor, typical in PyTorch 2.0+
# where complex tensors are just Tensors with complex dtype.
ComplexTensor = Tensor


class GenerativeModel(Protocol):
    """
    Protocol for Generative Models (GAN, VAE, Diffusion).
    Takes a latent vector or noise and optional condition, outputs a real-space image.

    ### Model Hierarchy
    ```mermaid
    classDiagram
        class GenerativeModel {
            <<protocol>>
            +forward(z, c) Tensor
        }
        class ReconstructionModel {
            <<protocol>>
            +forward(kspace, mask) Tensor
        }
        class IModel {
            <<interface>>
            +name str
            +forward(x)
            +get_parameter_count() int
        }
        class IGenerator {
            <<interface>>
            +generate(x)
        }
        class IDiscriminator {
            <<interface>>
            +discriminate(x)
        }
        IModel <|-- IGenerator
        IModel <|-- IDiscriminator
    ```
    """

    def forward(self, z: Tensor, c: Tensor | None = None) -> Tensor:
        """
        Args:
            z: Latent vector or noise [B, latent_dim] or [B, C, H, W]
            c: Optional conditioning vector/image
        Returns:
            x: Generated image [B, C, H, W]
        """
        ...


class ReconstructionModel(Protocol):
    """
    Protocol for Reconstruction Models (Unrolled Networks, Variational Networks).
    Takes k-space data and a sampling mask, outputs a real-space image.
    """

    def forward(self, kspace: ComplexTensor, mask: Tensor) -> Tensor:
        """
        Args:
            kspace: Input k-space data [B, C, H, W] (complex)
            mask: Sampling mask [B, 1, H, W] or [B, C, H, W]
        Returns:
            x: Reconstructed image [B, C, H, W]
        """
        ...


class IGenerator(Protocol):
    """
    Protocol for Generator models (image-to-image, latent-to-image).

    All generators must implement this interface for consistent I/O contracts.
    """

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Generate output from input.

        Args:
            x: Input tensor [B, C, H, W] (image or k-space)
            **kwargs: Additional conditioning inputs

        Returns:
            Generated tensor [B, C_out, H, W]
        """
        ...

    def get_parameter_count(self) -> int:
        """Return total trainable parameters."""
        ...


class IDiscriminator(Protocol):
    """
    Protocol for Discriminator models (PatchGAN, global discriminator).

    All discriminators must implement this interface.
    """

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Discriminate input tensor.

        Args:
            x: Input tensor [B, C, H, W]
            **kwargs: Conditional inputs (e.g., class labels)

        Returns:
            Discrimination scores [B, 1] or [B, 1, H', W'] for PatchGAN
        """
        ...


class IEncoder(Protocol):
    """
    Protocol for Encoder models (VAE encoder, feature extractor).

    Encodes input to latent representation.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input to latent space.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (mu, log_var) for VAE-style encoders,
            or (latent, None) for deterministic encoders.
        """
        ...

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        ...


class IDecoder(Protocol):
    """
    Protocol for Decoder models (VAE decoder, upsampling networks).

    Decodes latent representation to output space.
    """

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent to output.

        Args:
            z: Latent tensor [B, latent_dim] or [B, C, H, W]

        Returns:
            Decoded output [B, C_out, H_out, W_out]
        """
        ...
