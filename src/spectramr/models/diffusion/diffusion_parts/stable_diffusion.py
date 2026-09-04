import logging

import torch

from spectramr.models.diffusion.base_diffusion import Diffusion

logger = logging.getLogger(__name__)


class StableDiffusion(Diffusion):
    """A diffusion process wrapper that includes stability-enhancing features
    like gradient clipping during the loss calculation.
    """

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        device=None,
        gradient_clip_val=1.0,
        enable_gradient_checkpointing=False,
    ):
        # Set device to CPU if CUDA is not available or None
        """__init__.

        Args:
            timesteps (Any): Description.
            beta_schedule (Any): Description.
            device (Any): Description.
            gradient_clip_val (Any): Description.
            enable_gradient_checkpointing (Any): Description.
        """
        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            device = "cpu"
        self.device = device

        super().__init__(timesteps, beta_schedule, device)
        self.gradient_clip_val = gradient_clip_val
        self.enable_gradient_checkpointing = enable_gradient_checkpointing

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        """Calculates the loss. Gradient clipping should be handled by the
        training loop. This class holds the configuration for stability
        features.
        """
        return super().p_losses(denoise_model, x_start, t, noise, loss_type)

    def apply_gradient_checkpointing(self, model):
        """Applies gradient checkpointing to the model if enabled.
        This is a utility function that should be called after model creation.
        """
        if self.enable_gradient_checkpointing:
            if hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(enable=True)
                logger.debug("Enabled gradient checkpointing via model's method.")
            else:
                logger.debug(
                    "Warning: Generic gradient checkpointing not implemented. "  # LIMITATION: HF-specific checkpointing required
                    "Model must have a 'set_grad_checkpointing' method."
                )
        return model
