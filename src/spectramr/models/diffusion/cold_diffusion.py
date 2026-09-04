import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.config.schemas.enums import SchedulerTypes
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.sampling import KSpaceAccelerator
from spectramr.models.diffusion.schedulers.cold_mri_kspace_scheduler import (
    ColdMRIKSpaceScheduler,
)
from spectramr.models.registry import register_model

from .base_diffusion import Diffusion


@register_model(name="cold_diffusion_process", training_mode="diffusion")
class ColdDiffusion(Diffusion):
    """Implements Cold Diffusion (https://arxiv.org/abs/2208.09392).
    Instead of predicting noise, the model predicts the original image x_0.
    The sampling process then involves a degradation step.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str | None = None,
        degradation_type: str = "noise",
        cold_schedule: str = "linear",
        restoration_steps: int = 1,
        frequency_weighting: bool = False,
        accelerator: KSpaceAccelerator | None = None,
        model: nn.Module | None = None,
        # [ARCHITECTURE FIX] Data Consistency config
        dc_method: str = "hard",  # 'hard' or 'soft'
        dc_weight: float = 1.0,  # Weight for soft DC (lambda)
        sampler: str = "ddpm",
        sampling_steps: int = 1000,
        **kwargs,
    ):
        # Set device intelligently - use CUDA if available, otherwise CPU
        """__init__.

        Args:
            timesteps (int): Description.
            beta_schedule (str): Description.
            device (Optional[str]): Description.
            degradation_type (str): Description.
            cold_schedule (str): Description.
            restoration_steps (int): Description.
            frequency_weighting (bool): Description.
            accelerator (Optional[KSpaceAccelerator]): Description.
            model (Optional[nn.Module]): Description.
            dc_method (str): Description.
            dc_weight (float): Description.
            sampler (str): Description.
            sampling_steps (int): Description.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        super().__init__(timesteps, beta_schedule, device)
        self.degradation_type = degradation_type
        self.restoration_steps = restoration_steps
        self.frequency_weighting = frequency_weighting
        self.accelerator = accelerator
        self.model = model

        # [ARCHITECTURE FIX] Store DC config and Initialize ProximalDCStep
        self.dc_method = dc_method
        self.dc_weight = dc_weight
        self.sampler = sampler
        self.sampling_steps = sampling_steps
        self.scheduler = None

        from spectramr.infrastructure.physics.proximal_dc import ProximalDCStep

        if self.dc_method in ("hard", "soft"):
            self.proximal_dc = ProximalDCStep(mode=self.dc_method, soft_lambda=self.dc_weight)
        else:
            self.proximal_dc = None

        # [PHYSICS] Initialize progressive simulator for image-space corruptions
        self.simulator = None
        if self.degradation_type == "physical":
            config_obj = kwargs.get("config")
            if (
                config_obj is not None
                and hasattr(config_obj, "physics")
                and config_obj.physics is not None
            ):
                if getattr(config_obj.physics.digital_twin, "enabled", False):
                    from spectramr.infrastructure.training.strategies.simulator_builder import (
                        build_simulator_from_config,
                    )

                    self.simulator = build_simulator_from_config(config_obj, device=device)

        if self.sampler == SchedulerTypes.COLD_MRI:
            self.scheduler = ColdMRIKSpaceScheduler(
                num_train_timesteps=timesteps,
                prediction_type="sample",  # Cold diffusion enforces sample
                accelerator=accelerator,
            )

        # [PHYSICS] Optimization: Cache frequency weight
        self.frequency_weight_cache = None

        if cold_schedule == "linear":
            self.cold_schedule = torch.linspace(1.0, 0.0, timesteps, device=device)
        elif cold_schedule == "cosine":
            t = torch.linspace(0, timesteps, timesteps + 1)
            self.cold_schedule = torch.cos(0.5 * torch.pi * t / timesteps)[:-1].to(
                device,
            )
        elif cold_schedule == "inverse_annealing":
            # [PHYSICS] Inverse Annealing (Convex Decay)
            # Starts slow, accelerates degradation
            t = torch.linspace(0, 1, timesteps, device=device)
            self.cold_schedule = (1.0 - t) ** 2
        elif cold_schedule == "power_law":
            # [PHYSICS] Power Law Schedule (Gamma Correction)
            # Matches MRI spectral density decay (1/f^gamma)
            gamma = kwargs.get("schedule_kwargs", {}).get("gamma", 1.6)
            if "gamma" in kwargs:  # Direct kwarg support
                gamma = kwargs["gamma"]

            # Schedule: (1 - t/T)^gamma
            # t goes from 0 to 1
            t = torch.linspace(0, 1, timesteps, device=device)
            self.cold_schedule = (1.0 - t) ** gamma
        else:
            raise ValueError(f"Unknown cold schedule: {cold_schedule}")

    def generate(
        self,
        input_data: torch.Tensor,
        timestep: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the Cold Diffusion process.

        Args:
            input_data: Input tensor (measurement/undersampled data)
            timestep: Ignored for full generation loop
            **kwargs: Additional arguments

        Returns:
            Reconstructed/Generated image/k-space
        """
        if self.model is None:
            raise RuntimeError("ColdDiffusion model not set. Cannot generate.")

        # Determine if we should start from measurement
        # For MRI reconstruction, input_data IS the measurement
        measurement = input_data

        # Run sampling loop
        return self.p_sample_loop(self.model, input_data.shape, measurement=measurement, **kwargs)

    def _degrade(self, x_0_pred, t, mask=None):
        """Degrades the predicted x_0 to x_t using the chosen degradation."""
        if self.degradation_type == "noise":
            # [PHYSICS] Standard gaussian noise is not symmetric.
            # We must generate symmetric noise manually and pass it to q_sample.
            noise = torch.randn_like(x_0_pred)
            noise = self.symmetrize_kspace(noise)
            return self.q_sample(x_start=x_0_pred, t=t, noise=noise)
        elif self.degradation_type == "kspace_mask":
            if self.accelerator is None:
                # Fallback to legacy behavior if no accelerator provided
                return self.apply_progressive_mask(x_0_pred, t, mask=mask)

            # Use the consistent accelerator logic
            # x_0_pred is (B, C, H, W)
            # We need to apply mask for each item in batch based on its t
            B = x_0_pred.shape[0]
            out = torch.zeros_like(x_0_pred)
            for i in range(B):
                timestep = t[i].item()
                mask = self.accelerator.get_acceleration_mask(
                    x_0_pred.shape[1:],
                    timestep,
                    device=x_0_pred.device,  # (C, H, W)
                )
                out[i] = x_0_pred[i] * mask
            return out
        elif self.degradation_type == "blur":
            return self.gaussian_blur(x_0_pred, t)
        elif self.degradation_type == "physical":
            return self.apply_progressive_physics(x_0_pred, t)
        raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def apply_progressive_mask(
        self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply progressive k-space masking.

        At t=0, mask is full (no degradation).
        At t=T, mask is minimal (high degradation).
        """
        # x is assumed to be in k-space or image space depending on model.
        # For k-space cold diffusion, x is k-space.

        # Get schedule value for this timestep
        # self.cold_schedule is [T] (1.0 -> 0.0)
        # We want degradation factor 0.0 -> 1.0
        # If schedule is 1.0 (start), factor is 0.0 (no degradation).
        # If schedule is 0.0 (end), factor is 1.0 (full degradation).

        schedule_val = self.cold_schedule[t.long()]  # [B]

        # Degradation factor: how much to move towards the mask
        degradation_factor = 1.0 - schedule_val

        # Fix: Deterministic Degradation with Fixed Mask
        if mask is not None:
            # If a fixed mask is provided (e.g. from measurement), we interpolate
            # between full k-space (t=0) and the masked k-space (t=T).
            # At t=0, degradation is Identity.
            # At t=T, degradation is Mask.
            # Formula: x_t = x * (1 - factor * (1 - mask))
            # If mask=1, factor irrelevant -> 1.
            # If mask=0, term is (1 - factor).

            # Broadcast factor to [B, 1, 1, 1]
            factor_view = degradation_factor.view(-1, 1, 1, 1)
            while factor_view.ndim < x.ndim:
                factor_view = factor_view.unsqueeze(-1)

            return x * (1.0 - factor_view * (1.0 - mask))

        # Create mask
        B, C, H, W = x.shape
        mask = torch.zeros_like(x)

        # This is a simplified random mask. In practice, one might use
        # specific sampling patterns (e.g. variable density).
        # We use the same mask for all channels if C=2 (real/imag)
        for b in range(B):
            d = degradation_factor[b].item()
            # `d` is the DEGRADATION (0 at t=0, 1 at t=T) while `num_pixels`
            # counts pixels the mask KEEPS, so the two are complements. Using
            # `d` directly made t=0 zero the image and t=T the identity -- the
            # exact inverse of this method's docstring, and of the `mask is not
            # None` branch above, which interpolates in the correct direction.
            num_pixels = int((1.0 - d) * H * W)
            indices = torch.randperm(H * W, device=x.device)[:num_pixels]
            mask_flat = torch.zeros(H * W, device=x.device)
            mask_flat[indices] = 1.0
            mask[b] = mask_flat.view(H, W).unsqueeze(0).expand(C, -1, -1)

        return x * mask

    def gaussian_blur(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply progressive Gaussian blur.

        At t=0, sigma is 0 (no blur).
        At t=T, sigma is max_sigma.
        """
        max_sigma = 5.0
        t_norm = t.float() / self.timesteps
        sigma = max_sigma * t_norm

        # Apply blur per image in batch
        # We use a simple Gaussian kernel convolution
        B, C, H, W = x.shape
        x_blurred = torch.zeros_like(x)

        kernel_size = 21
        padding = kernel_size // 2

        for b in range(B):
            s = sigma[b].item()
            if s <= 1e-3:
                x_blurred[b] = x[b]
                continue

            # Create kernel
            k = torch.arange(kernel_size, device=x.device).float() - padding
            k = torch.exp(-(k**2) / (2 * s**2))
            k = k / k.sum()
            k_2d = k[:, None] * k[None, :]
            k_2d = k_2d.expand(C, 1, kernel_size, kernel_size)

            # Convolve
            x_in = x[b].unsqueeze(0)  # (1, C, H, W)
            x_out = F.conv2d(x_in, k_2d, padding=padding, groups=C)
            x_blurred[b] = x_out.squeeze(0)

        return x_blurred

    def apply_progressive_physics(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply progressive physical artifacts in image space.

        At t=0, corruption_factor is 0.0 (clean).
        At t=T, corruption_factor is 1.0 (fully corrupted).
        """
        if self.simulator is None:
            raise RuntimeError(
                "DigitalTwinSimulator is not instantiated. Check config.physics.digital_twin.enabled"
            )

        # Get dynamic scale (0.0 to 1.0 depending on timestep)
        # self.cold_schedule interpolates 1.0 (start) to 0.0 (end)
        schedule_val = self.cold_schedule[t.long()]
        degradation_factor = 1.0 - schedule_val

        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        for i in range(B):
            cf = degradation_factor[i].item()
            # Simulator requires [1, C, H, W] for unbatched scaling consistency
            clean_slice = x[i : i + 1]
            corrupted, _, _ = self.simulator(clean_anatomy=clean_slice, corruption_factor=cf)
            out[i : i + 1] = corrupted

        return out

    @staticmethod
    def symmetrize_kspace(z: torch.Tensor) -> torch.Tensor:
        """
        Enforces Hermitian symmetry by projecting onto the subspace of Real-valued signals.
        Method: IFFT -> Take Real -> FFT.
        Assumes k-space is centered (DC at N/2).
        Input: (B, C=2, H, W) where C=0 is Real, C=1 is Imag
        """
        # 2. Convert to complex and project to real via canonical FFT ops
        z_c = torch.complex(z[:, 0], z[:, 1])  # (B, H, W)

        # 3. IFFT to Image Domain using canonical centered IFFT
        img = ifft2c(z_c)

        # 4. Project to Real (Enforce Physical Constraint)
        img_real = torch.complex(img.real, torch.zeros_like(img.imag))

        # 5. FFT back to K-space using canonical centered FFT
        k_sym = fft2c(img_real)

        # 7. Convert back to 2-channel Real
        out = torch.stack([k_sym.real, k_sym.imag], dim=1)

        return out

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x,
        t,
        t_index,
        cond=None,
        guidance_scale: float = 1.0,
        measurement: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prev_t_index: int | None = None,
    ):
        """Cold diffusion sampling step with classifier-free guidance.
        1. Predict x_0 from the noisy image x_t, using guidance.
        2. Degrade the predicted x_0 to get an estimate of x_{t-1}.
        """

        if prev_t_index is None:
            prev_t_index = max(0, t_index - 1)

        # Helper to extract prediction
        def get_pred(res):
            """get_pred.

            Args:
                res (Any): Description.
            Returns:
                Any: Description.
            """
            return res[0] if isinstance(res, tuple) else res

        # For Cold Diffusion, the model directly predicts x_0.
        # We apply guidance to this prediction.
        if guidance_scale > 1.0 and cond is not None:
            # Predict with and without the condition
            uncond_res = model(x, t, cond=None)
            cond_res = model(x, t, cond=cond)

            uncond_pred = get_pred(uncond_res)
            cond_pred = get_pred(cond_res)

            # Extrapolate to get the guided prediction
            pred_x_start = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            # Standard conditional or unconditional prediction
            res = model(x, t, cond=cond)
            pred_x_start = get_pred(res)

        # Enforce Data Consistency if measurement and mask are provided
        # [ARCHITECTURE FIX] Respect config for Hard vs Soft DC
        if measurement is not None and mask is not None and self.proximal_dc is not None:
            # ProximalDCStep correctly handles image -> k-space -> image projection
            # ensuring rigorous data consistency without domain mismatches.
            pred_x_start = self.proximal_dc(pred_x_start, measurement, mask)

        if t_index == 0:
            return pred_x_start

        if self.degradation_type == "kspace_mask" and self.accelerator is not None:
            # [PHYSICS] Explicit Reconstruction Logic for k-space masking
            # Generate dynamic masks for current (t) and next step (t-1)
            B = x.shape[0]
            mask_current = torch.zeros_like(x)
            mask_prev = torch.zeros_like(x)

            for i in range(B):
                curr_t_val = t[i].item()
                prev_t_val = prev_t_index

                # Mask at t (Current State)
                m_curr = self.accelerator.get_acceleration_mask(
                    x.shape[1:], curr_t_val, device=x.device
                )
                # Mask at t-1 (Target State)
                m_prev = self.accelerator.get_acceleration_mask(
                    x.shape[1:], prev_t_val, device=x.device
                )
                mask_current[i] = m_curr.float()
                mask_prev[i] = m_prev.float()

            x_prev = self._p_sample_mri_logic(x, pred_x_start, mask_current, mask_prev)
            # [FIX] Define x_deterministic for downstream Langevin dynamics
            x_deterministic = x_prev

        else:
            # Generalized Update (Algorithm 2 from Bansal et al.)
            # x_{t-1} = D(x_0_hat, t-1) + (x_t - D(x_0_hat, t))

            # 1. Calculate degradations
            # What should the k-space look like at step t given our guess x_0_hat?
            d_t_hat = self._degrade(pred_x_start, t, mask=mask)

            # What should it look like at the NEXT step (t-1)?
            prev_t = torch.full_like(t, prev_t_index)
            d_prev_hat = self._degrade(pred_x_start, prev_t, mask=mask)

            # 2. The Repair Step (Residual Injection)
            # This adds back the "error" or "details" that x_0_hat missed.
            x_deterministic = d_prev_hat + (x - d_t_hat)
            x_prev = x_deterministic

        # 3. Langevin Dynamics: Null-Space Noise Injection (The Fix)
        # Inject noise only in the null-space to texture hallucination
        if t_index > 0 and mask is not None:
            # Retrieve beta_t
            if hasattr(self, "betas"):
                beta_t = self.betas[t].view(-1, 1, 1, 1)
            else:
                # Fallback if betas not directly accessible (should not happen in standard Diffusion)
                # Approximating linear schedule
                beta_t = (1.0 - (t_index / self.timesteps)) * 0.02

            # Empirical scaling: sigma = eta * sqrt(beta)
            sigma_t = 0.5 * torch.sqrt(beta_t)

            # Generate noise
            z = torch.randn_like(x)

            # [PHYSICS] Fix: Enforce Hermitian Symmetry on Langevin Noise
            # Random noise in K-space must correspond to Real noise in Image space.
            z = self.symmetrize_kspace(z)

            # Project into Null Space: only add noise where we don't have measurements
            # mask=1 (sampled), mask=0 (unsampled)
            # We want to add noise where mask=0
            x_prev = x_deterministic + sigma_t * z * (1.0 - mask)
        else:
            x_prev = x_deterministic

        # 4. [Integrity Fix] Enforce Data Consistency on the *step outcome* (x_prev)
        # This ensures that even after degradation and noise injection, we respect the measurement.
        if (
            measurement is not None
            and mask is not None
            and self.proximal_dc is not None
            and self.dc_method == "hard"
        ):
            x_prev = self.proximal_dc(x_prev, measurement, mask)

        return x_prev

    def p_sample_mri(
        self, x_t: torch.Tensor, mask_current: torch.Tensor, mask_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Specialized sampling step for MRI Reconstruction (In-painting style).

        Uses the specific 'fill in the gaps' logic for Cold Diffusion.
        x_prev = Known(x_t) + New(Prediction)

        Args:
            x_t: k-space at current acceleration level [B, C, H, W]
            mask_current: Mask used for x_t (current known frequencies)
            mask_prev: Mask for the next step (less accelerated, more frequencies)

        Returns:
            x_prev (less accelerated k-space with more frequencies filled)
        """
        # 1. Predict FULL clean k-space (x_0 estimate) using the denoising model
        # We use t=0 as we want the clean prediction
        batch_size = x_t.shape[0]
        t = torch.zeros(batch_size, device=x_t.device, dtype=torch.long)

        # Get model prediction of clean k-space
        with torch.no_grad():
            x_0_pred = self.model(x_t, t)

        # 2. Compute the new frequencies to add
        # mask_prev has MORE ones than mask_current (less acceleration)
        # new_frequencies = mask_prev - mask_current
        new_mask = mask_prev - mask_current
        new_mask = torch.clamp(new_mask, 0, 1)  # Ensure no negative values

        # 3. Combine: keep original known + add predicted new
        # x_prev = x_t * mask_current + x_0_pred * new_mask
        x_prev = x_t * mask_current + x_0_pred * new_mask

        return x_prev

    def _p_sample_mri_logic(self, x_t, x_0_pred, mask_current, mask_prev):
        """Logic implementation for p_sample_mri.

        [ARCHITECT FIX] Cold Diffusion Explosion (The 24,000x Blowup)
        We must strictly replace only the unknown frequencies.
        Do not add the prediction to the input.
        """
        # 1. Degrade the prediction to the target state (t-1)
        degraded_pred = x_0_pred * mask_prev

        # 2. Subtract the portion we already had at state t to find the "noise" (missing frequencies)
        missing_freqs_recovered = degraded_pred - (x_0_pred * mask_current)

        # 3. Add ONLY the newly recovered frequencies to the current state
        x_prev = x_t + missing_freqs_recovered

        return x_prev

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        cond=None,
        guidance_scale: float = 1.0,
        measurement: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """The full sampling loop for cold diffusion with classifier-free guidance."""
        # Initialize x_T
        if measurement is not None and self.degradation_type == "kspace_mask":
            # Start from the undersampled measurement (Conditional Reconstruction)
            img = measurement.clone()
        else:
            # Start from Low Frequency Seed (Unconditional Hallucination)
            # Instead of pure Gaussian noise, we start with a zero-filled k-space
            # with a small random center to seed the generation.
            img = torch.zeros(shape, device=self.device)
            B, C, H, W = shape
            center_h, center_w = H // 2, W // 2
            # 4x4 center patch
            img[:, :, center_h - 2 : center_h + 2, center_w - 2 : center_w + 2] = torch.randn(
                (B, C, 4, 4), device=self.device
            )
            # [PHYSICS] Enforce symmetry on the seed
            img = self.symmetrize_kspace(img)

        # Custom Scheduler Logic (Strided / Optimized)
        if self.scheduler is not None:
            self.scheduler.set_timesteps(self.sampling_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            for i, t in enumerate(timesteps):
                # Current timestep t
                # We need to pass 't' tensor to model
                t_tensor = torch.full((shape[0],), t.item(), device=self.device, dtype=torch.long)

                if i < len(timesteps) - 1:
                    prev_t = timesteps[i + 1].item()
                else:
                    prev_t = 0

                # Predict x_0
                # p_sample usually does prediction + step.
                # Scheduler separates them: Predict x0 -> step()

                # 1. Prediction (with guidance)
                if guidance_scale > 1.0 and cond is not None:
                    uncond_res = model(img, t_tensor, cond=None)
                    cond_res = model(img, t_tensor, cond=cond)
                    uncond_pred = uncond_res[0] if isinstance(uncond_res, tuple) else uncond_res
                    cond_pred = cond_res[0] if isinstance(cond_res, tuple) else cond_res
                    pred_x0 = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                else:
                    res = model(img, t_tensor, cond=cond)
                    pred_x0 = res[0] if isinstance(res, tuple) else res

                # 2. Scheduler Step (Physics-Aware)
                img = self.scheduler.step(model_output=pred_x0, timestep=t.item(), sample=img)

                # 3. Data Consistency (Hard Enforcement)
                # Even with scheduler, we might want to enforce knowns
                if measurement is not None and mask is not None and self.dc_method == "hard":
                    img = img * (1 - mask) + measurement * mask

            return img

        # Default / Legacy Logic (Full Loop)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)

            if i > 0:
                prev_t = i - 1
            else:
                prev_t = 0

            img = self.p_sample(
                model,
                img,
                t,
                i,
                cond=cond,
                guidance_scale=guidance_scale,
                measurement=measurement,
                mask=mask,
                prev_t_index=prev_t,
            )
        return img

    def p_losses(
        self,
        denoise_model,
        x_start,
        t,
        noise=None,
        loss_type: str = "l1",
        cond=None,
        degradation_mask=None,
        **kwargs,
    ):
        """Loss for cold diffusion. The model predicts x_0."""
        # Ensure all input tensors are on the correct device
        x_start = x_start.to(self.device)
        t = t.to(self.device)
        if noise is not None:
            noise = noise.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)
        if degradation_mask is not None:
            degradation_mask = degradation_mask.to(self.device)

        if noise is None:
            noise = torch.randn_like(x_start)
            # [PHYSICS] symmetric noise for real-valued imaging
            noise = self.symmetrize_kspace(noise)

        # Fix 4: Redundant Degrader Conflict
        # Check if pre-degraded data is available in kwargs (passed from batch)
        # The dataset might have already computed x_t and t
        if "kspace_undersampled" in kwargs and kwargs["kspace_undersampled"] is not None:
            x_noisy = kwargs["kspace_undersampled"].to(self.device)
            # If we use pre-degraded data, we MUST trust the t that came with it
            # But p_losses signature takes 't'. We assume the caller passed the correct 't'
            # corresponding to this x_noisy.
        elif degradation_mask is not None:
            # Use the mask provided by the dataset (which follows the config)
            x_noisy = x_start * degradation_mask
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Check if the denoise_model supports the 'cond' parameter
        import inspect

        sig = inspect.signature(denoise_model.forward)
        supports_cond = "cond" in sig.parameters

        # Pass the condition only if the model supports it
        # Pass the condition only if the model supports it
        if supports_cond and cond is not None:
            model_output = denoise_model(x_noisy, t, cond=cond)
        else:
            model_output = denoise_model(x_noisy, t)

        # Handle tuple output (normalized_pred, scale)
        scale = 1.0
        if isinstance(model_output, tuple):
            predicted_x_start, scale = model_output
        else:
            predicted_x_start = model_output

        # Normalize target if scale is present (Fix: Gradient Explosion)
        # We normalize x_start to match the normalized prediction
        if isinstance(scale, torch.Tensor) or scale != 1.0:
            # Ensure scale is broadcastable
            if isinstance(scale, torch.Tensor) and scale.ndim < x_start.ndim:
                while scale.ndim < x_start.ndim:
                    scale = scale.unsqueeze(-1)
            x_start = x_start / scale

        # Apply frequency weighting if enabled
        weight = 1.0
        if self.frequency_weighting:
            B, C, H, W = x_start.shape

            # [PHYSICS] Optimization: Lazy Cache of Frequency Weight
            if self.frequency_weight_cache is None or self.frequency_weight_cache.shape[-2:] != (
                H,
                W,
            ):
                # Create frequency weight mask based on distance from center
                # Higher frequencies (further from center) get higher weight
                y = torch.arange(H, device=self.device) - H // 2
                x = torch.arange(W, device=self.device) - W // 2
                Y, X = torch.meshgrid(y, x, indexing="ij")
                dist = torch.sqrt(X**2 + Y**2)

                # Normalize distance to [0, 1]
                max_dist = torch.sqrt(
                    torch.tensor((H // 2) ** 2 + (W // 2) ** 2, device=self.device)
                )
                dist_norm = dist / max_dist

                # Weight function: 1 + alpha * dist^beta
                # We want to emphasize high frequencies
                alpha = 5.0
                beta = 2.0
                weight_map = 1.0 + alpha * (dist_norm**beta)
                self.frequency_weight_cache = weight_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            weight = self.frequency_weight_cache

        if loss_type == "l1_smooth":
            # L1 with smoothing for stability. Kept inline: this is
            # Charbonnier(eps=1e-4) minus the 1e-4 offset — the registered
            # "charbonnier" does not subtract the offset, so routing through
            # the registry would shift the loss value (gradients identical).
            diff = torch.abs((x_start - predicted_x_start) * weight)
            return torch.mean(torch.sqrt(diff**2 + 1e-8) - 1e-4)
        if loss_type == "log_spectral":
            # [PHYSICS] Log-Spectral Phase Loss (Eco-11)
            # Delegates to the registry to ensure the rigorous implementation
            # from complex_losses.py is used (with 0.1 phase weight as per
            # physics requirements). Lazy import: losses/__init__ imports
            # models.diffusion modules — a module-scope import would cycle.
            from spectramr.models.losses.elementary import shared_loss

            loss_fn = shared_loss("log_spectral_phase", phase_weight=0.1)
            return loss_fn(predicted_x_start * weight, x_start * weight)

        from spectramr.models.losses.elementary import resolve_elementary_loss

        try:
            loss_fn = resolve_elementary_loss(loss_type)
        except ValueError as err:
            raise NotImplementedError(
                f"Unsupported loss type: {loss_type}. "
                "Supported: l1, l2, huber, l1_smooth, log_spectral"
            ) from err
        return loss_fn(predicted_x_start * weight, x_start * weight)

    def generate_with_uncertainty(self, x, t, num_samples=8, **kwargs):
        """
        Runs N parallel diffusion chains to estimate epistemic uncertainty.
        Returns: Mean Reconstruction, Variance Map (Uncertainty)
        """
        # Repeat input batch N times: [B, C, H, W] -> [B*N, C, H, W]
        x_repeated = x.repeat_interleave(num_samples, dim=0)

        measurement = kwargs.get("measurement")
        mask = kwargs.get("mask")

        if measurement is not None:
            measurement = measurement.repeat_interleave(num_samples, dim=0)
        if mask is not None:
            mask = mask.repeat_interleave(num_samples, dim=0)

        # Run sampling loop
        samples = self.generate(x_repeated, measurement=measurement, mask=mask)

        # Reshape to [B, N, C, H, W]
        B = x.shape[0]
        samples = samples.view(B, num_samples, *samples.shape[1:])

        # Calculate Statistics
        mean_recon = samples.mean(dim=1)
        uncertainty_map = samples.var(dim=1).sqrt()  # Standard Deviation

        return mean_recon, uncertainty_map


class MetalDegradation(nn.Module):
    """Simulates metal artifact degradation for cold diffusion.

    At t=0, it is the identity transform.
    As t increases, it creates a centered signal void (simulated metal).
    """

    def __init__(self, shape: tuple[int, int], total_steps: int = 1000):
        """__init__.

        Args:
            shape (tuple[int, int]): Description.
            total_steps (int): Description.
        """
        super().__init__()
        self.shape = shape
        self.total_steps = total_steps

    def forward(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Applies degradation.

        Args:
            x: Input tensor [B, C, H, W]
            t: Current timestep

        Returns:
            Degraded tensor

        forward method for MetalDegradation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if isinstance(t, torch.Tensor):
            t_val = t.flatten()[0].item()
        else:
            t_val = t

        if t_val == 0:
            return x

        # Degradation factor: 0.0 at t=0, 1.0 at t=total_steps
        factor = min(1.0, float(t_val) / self.total_steps)

        # Create a centered void mask
        B, C, H, W = x.shape
        mask = torch.ones_like(x)

        # Define void region (center)
        cy, cx = H // 2, W // 2
        void_h = max(1, H // 8)
        void_w = max(1, W // 8)

        # Apply void: At max factor, center is exactly 0
        mask[
            :,
            :,
            cy - void_h : cy + void_h,
            cx - void_w : cx + void_w,
        ] = 1.0 - factor

        return x * mask
