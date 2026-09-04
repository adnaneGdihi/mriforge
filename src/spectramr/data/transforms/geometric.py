import numpy as np
import torch
import torchio as tio

# Identity affine for spatial consistency
_IDENTITY_AFFINE = np.eye(4)


class EnsureSpatialConsistency(tio.Transform):
    """
    Forces all images in a Subject to share the same affine matrix.

    This is critical for TorchIO Queue/Sampler which requires spatial consistency.
    Must be applied as the FIRST transform in the pipeline.

    Replacing a subject entry requires re-syncing ``Subject.__dict__`` (#1213):
    :class:`torchio.Subject` is a ``dict`` subclass that *mirrors* its entries into
    ``self.__dict__``, and ``Subject.__setitem__`` is not defined — so a bare
    ``subject[name] = new_image`` updates the mapping while ``__dict__[name]`` stays
    bound to the **original** image. ``tio.Crop.apply_transform`` — the operation
    every ``PatchSampler``/``tio.Queue`` uses — iterates ``subject.__dict__.items()``,
    so it would crop the stale object and silently discard everything this chain did
    downstream. Because this transform runs FIRST and replaces *every* image, one
    missing sync corrupts the whole chain's output.
    """

    def __init__(self, reference_affine: np.ndarray = None):
        """__init__.

        Args:
            reference_affine (np.ndarray): Description.
        """
        super().__init__()
        self.reference_affine = (
            reference_affine if reference_affine is not None else _IDENTITY_AFFINE
        )

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Force all images to use the reference affine."""
        for image_name in subject.get_images_names():
            image = subject[image_name]
            # Preserve image class (ScalarImage vs LabelMap)
            cls = type(image)
            new_image = cls(
                tensor=image.data,
                affine=self.reference_affine.copy(),
            )
            subject[image_name] = new_image
        # Re-bind __dict__ to the images just installed. Without this the patch
        # sampler crops the pre-transform objects (#1213 — see the class docstring).
        # One dict update per subject, not per image; torchio's own ``add_image``
        # does exactly this.
        subject.update_attributes()
        return subject


class SmartGeometricStandardization(tio.Transform):
    """
    Intelligent resizing logic that preserves anatomy:

    1. If image is non-square (FastMRI 640x368): Center Crop to target
    2. If image is square but wrong size: Resize to target (preserve FOV)
    3. If image is already correct size: No-op

    This prevents blindly cropping valid anatomy from pre-processed data.

    Note: Excludes 'kspace' images which have different native resolution.
    """

    def __init__(self, target_shape=(320, 320)):
        """__init__.

        Args:
            target_shape (Any): Description.
        """
        super().__init__()
        self.target_shape = target_shape[:2]  # Ensure 2D (H, W)
        # Images to transform (exclude kspace which has different resolution)
        self.include_keys = [
            "input",
            "target",
            "mri",
            "sensitivity",
            "sensitivity_complex",
        ]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        target_h, target_w = self.target_shape

        # [PERFORMANCE] Pure PyTorch Implementation
        # Avoids creating intermediate tio.Subject/ScalarImage objects

        for image_key in list(subject.get_images_names()):
            if image_key not in self.include_keys:
                continue

            image = subject[image_key]
            data = image.data  # (C, H, W, D)

            # Check dimensions
            _, h, w, d = data.shape
            if h == target_h and w == target_w:
                continue

            # Handle Complex: (C, H, W, D) complex -> (C*2, H, W, D) real
            is_complex = torch.is_complex(data)
            if is_complex:
                # View as real: (C, H, W, D, 2)
                real_data = torch.view_as_real(data)
                # Permute to channels: (C, H, W, D, 2) -> (C, 2, H, W, D) -> (C*2, H, W, D)
                # We need (Batch, Channels, Height, Width) for interpolate?
                # Interpolate expects (N, C, H, W) or (N, C, D, H, W)
                # Here we have 4D spatial data? No, usually 2D slices (C, H, W, 1) or 3D volumes (C, H, W, D)
                # FastMRI is typically (C, H, W, 1) or (1, H, W, D) depending on builder.
                # Let's treat (H, W) as the spatial dims to resize.
                # Permute to put H, W last? Or use interpolate on 4D input?
                # Grid sample / interpolate works on (N, C, Hin, Win).

                # Reshape for processing: (N, H, W) where N = C * D * 2
                # (C, H, W, D, 2) -> (C, D, 2, H, W) -> (C*D*2, 1, H, W) ?
                # Simpler: (C*2, D, H, W) ?
                # Let's assume standard layout (C, H, W, D)

                # Standardize to (N, C, H, W)
                # We want to resize H, W.
                # Treat 'batch' as C * D * real_imag
                # (C, H, W, D, 2) -> permute -> (C, D, 2, H, W) -> reshape -> (C*D*2, 1, H, W)
                c_dim = data.shape[0]
                d_dim = data.shape[3]

                # (C, H, W, D, 2) -> (C, D, 2, H, W)
                working_tensor = (
                    real_data.permute(0, 3, 4, 1, 2).reshape(-1, h, w).unsqueeze(1)
                )  # (N, 1, H, W)

            else:
                # Real data: (C, H, W, D)
                # (C, H, W, D) -> (C, D, H, W) -> (C*D, 1, H, W)
                c_dim = data.shape[0]
                d_dim = data.shape[3]
                working_tensor = data.permute(0, 3, 1, 2).reshape(-1, h, w).unsqueeze(1)

            # Perform Transform
            if h == w:
                # Resize (Preserve FOV): Bilinear for images, Nearest for labels?
                # Assuming all are images for now or use 'linear'
                # SmartGeometricStandardization is mostly for T1/T2/FLAIR/Mask
                mode = "bilinear" if not isinstance(image, tio.LabelMap) else "nearest"

                processed = torch.nn.functional.interpolate(
                    working_tensor,
                    size=(target_h, target_w),
                    mode=mode,
                    align_corners=False if mode != "nearest" else None,
                )
            else:
                # Center Crop / Pad
                # TorchIO CropOrPad logic:
                diff_h = target_h - h
                diff_w = target_w - w

                pad_left = max(0, diff_w // 2)
                pad_right = max(0, diff_w - pad_left)
                pad_top = max(0, diff_h // 2)
                pad_bottom = max(0, diff_h - pad_top)

                if diff_h > 0 or diff_w > 0:
                    # Need padding
                    working_tensor = torch.nn.functional.pad(
                        working_tensor,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0,
                    )

                # Update shape after pad
                curr_h, curr_w = working_tensor.shape[2], working_tensor.shape[3]

                # Crop if needed
                if curr_h > target_h or curr_w > target_w:
                    start_h = (curr_h - target_h) // 2
                    start_w = (curr_w - target_w) // 2
                    processed = working_tensor[
                        :, :, start_h : start_h + target_h, start_w : start_w + target_w
                    ]
                else:
                    processed = working_tensor

            # Restore Dimensions
            if is_complex:
                # (N, 1, H', W') -> (C, D, 2, H', W') -> (C, H', W', D, 2) -> complex
                processed = processed.squeeze(1).view(c_dim, d_dim, 2, target_h, target_w)
                processed = processed.permute(0, 3, 4, 1, 2).contiguous()  # (C, H, W, D, 2)
                final_data = torch.view_as_complex(processed)
            else:
                # (N, 1, H', W') -> (C, D, H', W') -> (C, H', W', D)
                processed = processed.squeeze(1).view(c_dim, d_dim, target_h, target_w)
                final_data = processed.permute(0, 2, 3, 1).contiguous()

            # Update Image in place (keeps affine)
            image.set_data(final_data)

        return subject
