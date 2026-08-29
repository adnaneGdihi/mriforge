"""Physics Synchronization Transform.

[ARCHITECT FIX] Ensures K-Space consistency after geometric transformations.

CRITICAL PHYSICS LOGIC:
1. If the Image was cropped/resized, the original 'kspace' (if present)
   is no longer valid (Dimension/FOV mismatch).
2. This transform DISCARDS the old kspace and REGENERATES it via FFT
   from the current, geometrically transformed Image.
3. This guarantees that kspace corresponds exactly to the anatomy seen by the network.
"""

import torch
import torchio as tio

from mriforge.infrastructure.physics.fft_ops import fft2c


class PhysicsSynchronization(tio.Transform):
    """
    Ensures K-Space consistency after geometric transformations (Crop/Resize).

    After CropOrPad changes the image FOV, the original k-space no longer
    matches the cropped anatomy. This transform regenerates k-space from
    the current image state to maintain physics consistency.

    Args:
        fft_norm: FFT normalization mode ('ortho', 'forward', 'backward')
        image_key: Key for the source image in the Subject
        kspace_key: Key for the k-space output in the Subject
        input_is_image: Declares that this arm's ``input`` holds an IMAGE, which
            is the one fact the ``input``/``kspace`` ambiguity check below needs
            and cannot obtain for itself. Only a caller that knows the arm's
            signal domain may set it; the transform still resolves the source
            key through its normal search order, so this suppresses the refusal
            without changing which key is chosen.
    """

    def __init__(
        self,
        fft_norm: str = "ortho",
        image_key: str | None = None,
        kspace_key: str = "kspace",
        input_is_image: bool = False,
    ):
        """__init__.

        Args:
            fft_norm (str): Description.
            image_key (str | None): Description.
            kspace_key (str): Description.
        """
        super().__init__()
        if fft_norm != "ortho":
            raise ValueError(
                f"PhysicsSynchronization.fft_norm must be 'ortho' — fft2c from "
                f"infrastructure.physics.fft_ops is always ortho-normalised; "
                f"got {fft_norm!r}. (CLAUDE.md pitfall #15)"
            )
        self.fft_norm = fft_norm
        self.image_key = image_key
        self.kspace_key = kspace_key
        self.input_is_image = input_is_image

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Detect image key if not specified
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        image_key = self.image_key
        if image_key is None:
            # `input` is AMBIGUOUS: on an image arm it is the image, on a
            # k-space arm it IS the measured k-space. Taking it blindly meant
            # feeding k-space to `fft2c` — a second forward transform — and
            # overwriting `subject["kspace"]` with the result (A4).
            #
            # A subject that carries a DISTINCT `kspace` image alongside
            # `input` cannot be disambiguated by key name here, so this refuses
            # rather than guessing (#9). The caller knows: the transform
            # builder skips this transform entirely on a k-space `dataset_type`,
            # and any other caller can pass `image_key=` explicitly.
            #
            # `input_is_image` is that same answer from the caller that CANNOT
            # name a key: on an image-primary arm the builder has already
            # proved `input` holds an image, but the source key it should sync
            # FROM may be `hr`/`mri` rather than `input` (a different image on
            # the same subject), so naming `image_key="input"` there would
            # silently change which image k-space is derived from. Declaring
            # the domain answers the ambiguity without touching the choice.
            #
            # Cluster job 8012333: 10 `10_paradigms` arms died here. They are
            # `nifti_paired` (image-primary, so not skipped above) served by
            # `UniversalMRIDataset`, which derives a `kspace` key alongside
            # `input` -- the blind spot between the two branches.
            if not self.input_is_image and "kspace" in subject and "input" in subject:
                raise ValueError(
                    "PhysicsSynchronization cannot infer its image key: the "
                    "subject carries both 'input' and 'kspace', and on a "
                    "k-space arm 'input' IS the k-space — synchronising from "
                    "it applies a second forward FFT and overwrites the "
                    "measured data. Pass image_key= explicitly, or do not "
                    "apply this transform to a k-space-primary dataset."
                )
            for key in ["mri", "image", "hr", "input", "target"]:
                if key in subject:
                    image_key = key
                    break

        if image_key is None or image_key not in subject:
            # No image to sync from, skip
            return subject

        # Get the geometrically transformed image data
        # TorchIO format: (C, W, H, D)
        image_data = subject[image_key].data

        # Handle complex vs magnitude images
        if image_data.is_complex():
            # Data is already complex-valued (e.g., from diffusion strategy)
            complex_img = image_data
            if complex_img.dim() == 4 and complex_img.shape[0] == 1:
                pass  # Already (1, W, H, D)
            elif complex_img.dim() == 3:
                complex_img = complex_img.unsqueeze(0)  # (1, W, H, D)
        elif image_data.shape[0] == 2:
            # Already complex (Real/Imag channels)
            complex_img = torch.complex(image_data[0], image_data[1])
            complex_img = complex_img.unsqueeze(0)  # (1, W, H, D)
        elif image_data.shape[0] == 1:
            # Magnitude image, assume zero phase
            complex_img = image_data[0].to(torch.complex64)
            complex_img = complex_img.unsqueeze(0)  # (1, W, H, D)
        elif image_data.shape[0] % 2 == 0:
            # Multi-coil complex data (e.g., 8 channels = 4 coils * 2 Re/Im)
            num_coils = image_data.shape[0] // 2
            reshaped = image_data.view(num_coils, 2, *image_data.shape[1:])
            complex_img = torch.complex(reshaped[:, 0], reshaped[:, 1])
            # Shape: (num_coils, W, H, D)
        else:
            # Multiple channels, take first as magnitude
            complex_img = image_data[0].to(torch.complex64)
            complex_img = complex_img.unsqueeze(0)  # (1, W, H, D)

        # [PHYSICS] Regenerate K-Space via Forward Model (FFT)
        # complex_img is (C, W, H, D)
        # We want to apply 2D FFT along W, H dimensions.
        # fft2c operates on the last two dimensions.
        # So we transpose (C, W, H, D) -> (C, D, W, H)
        complex_img_transposed = complex_img.permute(0, 3, 1, 2)  # (C, D, W, H)

        # Apply canonical centered FFT
        kspace_transposed = fft2c(complex_img_transposed)  # (C, D, W, H)

        # Transpose back to (C, W, H, D)
        kspace = kspace_transposed.permute(0, 2, 3, 1)  # (C, W, H, D)

        # Convert back to real tensor format: (C*2, W, H, D) for Re/Im
        if image_data.is_complex():
            # Input was already complex - convert kspace to real/imag channels
            kspace_tensor = torch.stack([kspace.real, kspace.imag], dim=0)
            # Flatten first two dims if needed: (2, C, W, H, D) → (2*C, W, H, D)
            if kspace_tensor.dim() == 5:
                kspace_tensor = kspace_tensor.view(-1, *kspace_tensor.shape[2:])
        elif image_data.shape[0] == 2:
            kspace_tensor = torch.stack([kspace[0].real, kspace[0].imag], dim=0)
        elif image_data.shape[0] == 1:
            kspace_tensor = kspace[0].abs().unsqueeze(0)
        elif image_data.shape[0] % 2 == 0:
            # Interleave real and imaginary parts
            kspace_tensor = torch.empty_like(image_data)
            kspace_tensor[0::2] = kspace.real
            kspace_tensor[1::2] = kspace.imag
        else:
            kspace_tensor = kspace[0].abs().unsqueeze(0)

        # Overwrite or create 'kspace' in Subject
        # This replaces wrongly cropped k-space with correctly simulated k-space
        subject[self.kspace_key] = tio.ScalarImage(tensor=kspace_tensor)

        return subject
