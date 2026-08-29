import logging
from typing import Any

try:
    import ants
except ImportError:
    ants = None

from mriforge.domain.services.services import ICoRegistrationService


class CoRegistrationService(ICoRegistrationService):
    """Co-registration service using ANTsPy."""

    def __init__(self, tool_preference: str = "antsxpy", temp_dir: str | None = None):
        self.tool_preference = tool_preference
        self.temp_dir = temp_dir
        self.logger = logging.getLogger(__name__)

        if self.tool_preference == "antsxpy" and ants is None:
            raise RuntimeError("ANTsPy is required but not installed.")

    def register_images(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        output_path: str,
        transform_type: str = "SyN",
    ) -> dict[str, Any]:
        """Register moving image to fixed image using ANTsPy."""

        if ants is None:
            raise RuntimeError("ANTsPy is required for registration.")

        self.logger.info(
            f"Registering {moving_image_path} to {fixed_image_path} using {transform_type}"
        )

        try:
            fixed_img = ants.image_read(fixed_image_path)
            moving_img = ants.image_read(moving_image_path)

            registration = ants.registration(
                fixed=fixed_img, moving=moving_img, type_of_transform=transform_type
            )

            warped_img = registration["warpedmovout"]
            ants.image_write(warped_img, output_path)

            return {
                "status": "success",
                "output_path": output_path,
                "fwdtransforms": registration["fwdtransforms"],
                "invtransforms": registration["invtransforms"],
                "error": None,
            }
        except Exception as exc:
            self.logger.exception(
                "ANTsPy registration failed for moving image %s against fixed image %s",
                moving_image_path,
                fixed_image_path,
            )
            return {
                "status": "failed",
                "output_path": None,
                "fwdtransforms": None,
                "invtransforms": None,
                "error": str(exc),
            }

    def apply_transform(
        self,
        input_image_path: str,
        transform_path: str,
        output_path: str,
        reference_image_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply a pre-computed ANTs transform to an image.

        Args:
            input_image_path: Path to the image to transform.
            transform_path: Path to the ANTs transform file (.mat or warp).
            output_path: Path to write the transformed image.
            reference_image_path: Optional reference image for output geometry.

        Returns:
            Result dict with status and paths.
        """
        if ants is None:
            raise RuntimeError("ANTsPy is required to apply transforms.")

        self.logger.info("Applying transform %s to %s", transform_path, input_image_path)

        try:
            moving_img = ants.image_read(input_image_path)

            # Use reference geometry if provided, otherwise use moving image
            if reference_image_path is not None:
                ref_img = ants.image_read(reference_image_path)
            else:
                ref_img = moving_img

            warped = ants.apply_transforms(
                fixed=ref_img,
                moving=moving_img,
                transformlist=[transform_path],
            )
            ants.image_write(warped, output_path)

            return {
                "status": "success",
                "output_path": output_path,
                "error": None,
            }
        except Exception as exc:
            self.logger.exception(
                "Failed to apply transform %s to %s",
                transform_path,
                input_image_path,
            )
            return {
                "status": "failed",
                "output_path": None,
                "error": str(exc),
            }

    def multi_modal_registration(
        self,
        images: dict[str, str],
        reference_key: str,
        output_dir: str,
        registration_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register multiple images to a common reference space.

        Iterates over all non-reference images and registers each
        to the reference using :meth:`register_images`.

        Args:
            images: Mapping of modality name → file path.
            reference_key: Key in *images* to use as the fixed reference.
            output_dir: Directory for output files.
            registration_params: Optional override params (e.g. transform_type).

        Returns:
            Result dict keyed by modality name with per-image registration results.
        """
        import os

        if reference_key not in images:
            raise ValueError(
                f"Reference key '{reference_key}' not found in images dict. "
                f"Available keys: {list(images.keys())}"
            )

        params = registration_params or {}
        transform_type = params.get("transform_type", "SyN")
        fixed_path = images[reference_key]
        results: dict[str, Any] = {
            reference_key: {"status": "reference", "output_path": fixed_path}
        }

        os.makedirs(output_dir, exist_ok=True)

        for modality, moving_path in images.items():
            if modality == reference_key:
                continue

            out_path = os.path.join(output_dir, f"{modality}_registered.nii.gz")
            result = self.register_images(
                fixed_image_path=fixed_path,
                moving_image_path=moving_path,
                output_path=out_path,
                transform_type=transform_type,
            )
            results[modality] = result

        return results
