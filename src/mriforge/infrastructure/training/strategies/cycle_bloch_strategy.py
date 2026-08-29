from typing import Any

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.biophysics.bloch import BlochSimulator
from mriforge.infrastructure.training.contexts import TrainingEnvironment
from mriforge.infrastructure.training.step_io import accepts_step_io
from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.models.losses.registry import create_loss


class ParameterEstimator(nn.Module):
    """
    Lightweight network to estimate quantitative tissue maps (PD, T1, T2)
    from a generated High-Field image.
    """

    def __init__(self, in_channels=1):
        """__init__.

        Args:
            in_channels (Any): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1),  # Output: PD, T1, T2
        )
        self.act = nn.Softplus()  # Enforce positivity for physical parameters

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ParameterEstimator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.act(self.net(x))


class CycleBlochStrategy(BaseTrainingStrategy):
    """Cycle-consistent physics training for realistic anatomically-faithful synthesis.

    Ensures generated high-field MRI images are not only visually plausible but also
    physically capable of producing the observed ultra-low-field signal via forward
    Bloch simulation. Combines cycle consistency with biophysical constraints.

    ## Paradigm Overview

    **Problem**: Standard adversarial training on high-field (HF) images can produce
    visually realistic synthetics that violate MRI physics. Generated tissue parameters
    (T1, T2, PD) may be implausible when simulated via Bloch equations.

    **Solution**: Enforce cycle consistency through Bloch simulation:
    - HF image → estimate tissue params → Bloch simulate ULF → compare with measured ULF
    - Ensures generated HF image would produce measured ULF signal
    - Physically-grounded training without explicit parameter labels

    ## Architecture

    **Components**:
    1. **Generator(ULF) → HF**: Main generator, transforms ULF → High-field estimate
    2. **ParameterEstimator**: Lightweight CNN to estimate (PD, T1, T2) from HF image
    3. **BlochSimulator**: Differentiable MRI physics forward model (ULF sequence)
    4. **Discriminator(HF)**: Adversarial network on HF image

    ## Training Loop

    **Standard GAN Step**:
    1. Generate: HF_fake = Generator(ULF_real)
    2. Discriminate: D(HF_fake) vs D(HF_real)
    3. Update: Generator minimizes discriminator loss

    **Cycle-Physics Step** (novel):
    1. Generate: HF_fake = Generator(ULF_real)
    2. **Estimate Parameters**: (PD, T1, T2) = ParameterEstimator(HF_fake)
    3. **Simulate**: ULF_sim = BlochSimulator(PD, T1, T2, ULF_sequence_params)
    4. **Cycle Loss**: L1(ULF_sim, ULF_real) - simulated must match measured!
    5. Total Loss = L_adversarial + λ·L_cycle_physics

    ## Biophysics Modules

    **ParameterEstimator**:
    - 3-layer CNN: 1 channel input → 3-channel output (PD, T1, T2)
    - Softplus activation: Enforces positive tissue parameters
    - Trained jointly with generator (end-to-end)

    **BlochSimulator**:
    - Simulates ULF MRI sequence on estimated tissue map
    - Inputs: (PD, T1, T2), (TR, TE, FA) sequence parameters
    - Output: Synthetic ULF image via Bloch equation evolution
    - Fully differentiable (PyTorch custom ops)

    ## Typical MRI Parameters

    **Ultra-Low-Field (ULF) Sequence** (0.05T - 0.2T):
    - TR (Repetition Time): 500ms (long, slow)
    - TE (Echo Time): 14ms (short, minimize dephasing)
    - FA (Flip Angle): Variable (e.g., 30°)

    **High-Field (HF) Sequence** (3T):
    - TR: Shorter (faster, clinical speedup)
    - TE: Shorter (better SNR)
    - FA: Optimized per tissue (e.g., 90° for T2)

    ## Configuration

    - `training.training_mode`: 'cycle_bloch' or legacy mode
    - `physics.bloch.enabled`: True for simulator
    - `training.cycle_bloch.cycle_weight`: λ_cycle (0.1-1.0)
    - `physics.bloch.ulf_tr`: Repetition time (ms)
    - `physics.bloch.ulf_te`: Echo time (ms)
    - `physics.bloch.ulf_fa`: Flip angle (degrees)

    ## Loss Components

    1. **Adversarial Loss**: Fool discriminator on HF image
       - BCE or hinge loss on D(HF_fake)

    2. **Cycle Physics Loss**: Generated HF → simulate ULF → match measurement
       - L1(ULF_simulated, ULF_measured)
       - Critical: Enforces physical realism

    3. **Parameter Realism Loss** (optional):
       - Regularize (PD, T1, T2) to physiological ranges
       - T1 ∈ [100, 4000]ms, T2 ∈ [10, 200]ms

    4. **Reconstruction Loss** (optional):
       - L1(HF_fake, HF_real) if available

    ## Key Features

    ✅ **Physics-Grounded**: Cycle loss via Bloch simulation
    ✅ **End-to-End**: Parameters estimated and simulated differentiably
    ✅ **Interpretable**: Estimated (PD, T1, T2) can be visualized
    ✅ **Constraint-Free**: No need for ground truth parameter maps
    ✅ **Cross-Field**: Enables ULF↔HF synthesis with physics fidelity

    ## Inference

    1. Input: Ultra-low-field MRI
    2. Generate: HF_estimate = Generator(ULF)
    3. Estimate: (PD, T1, T2) = ParameterEstimator(HF_estimate)
    4. Outputs:
       - Synthesized high-field image (HF_estimate)
       - Tissue parameter maps (PD, T1, T2)
       - Optional: Simulated ULF for validation

    ## Registration

    Registered in ``TrainingStrategyFactory.STRATEGY_CLASS_PATHS`` as
    ``"cycle_bloch"`` and inherits from ``BaseTrainingStrategy``.
    Fully integrated with the framework registry.

    Attributes:
        state: TrainingState with generator and discriminator
        estimator: ParameterEstimator (PD, T1, T2 prediction)
        bloch_simulator: BlochSimulator for forward physics
        device: Computation device (CUDA/CPU)
        tr_ulf, te_ulf: Sequence parameters (ms)
        num_simulations: Technical parameter for Bloch equation steps

    References:
        - Bloch, F. (1946): Nuclear Induction
        - Practical Bloch equations implementation in MRI physics
    """

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
        """
        super().__init__(env=env, **kwargs)

        # Physics Components
        # Estimator is lazily created on first forward pass to match actual generator output channels,
        # since ChannelAdapter may modify the channel count at runtime.
        self._estimator_channels: int | None = None
        self.estimator: ParameterEstimator | None = None

        # Hyperparameters (Approximations for ULF/HF sequences)
        self.TR_ULF = 500.0  # ms
        self.TE_ULF = 14.0  # ms

        # Loss weights — SSOT: read from config.losses.gan (GANLossesConfig)
        # Safe fallback: config.losses.gan is None when experiments don't configure
        # a losses.gan: YAML section.  Use GANLossesConfig defaults in that case.
        _gan_cfg = self.config.losses.gan
        self.lambda_cycle = _gan_cfg.lambda_cycle_bloch if _gan_cfg else 10.0
        self.lambda_adv = _gan_cfg.lambda_cycle_adv if _gan_cfg else 1.0

        # SSOT: Use registry-based loss creation instead of hardcoded nn modules
        self.criterion_cycle = create_loss("l1")
        self.criterion_gan = create_loss("mse")  # MSE on disc outputs = LSGAN

    def _setup_strategy_specific_components(self) -> None:
        """_setup_strategy_specific_components."""
        pass

    @staticmethod
    def _to_magnitude_image(x: torch.Tensor) -> torch.Tensor:
        # Delegate to the registered ``rss_coils_to_magnitude`` adapter
        # (src/data/adapters/channels.py) so YAMLs can opt in via
        # `adapters.pre_loss_target: [{name: rss_coils_to_magnitude}]`
        # and the audit knows the bridge is in play. Phase 4 will wire
        # the strategy to honor the YAML chain when present and only
        # fall back to this hardcoded call otherwise.
        from mriforge.data.adapters import get_adapter

        return get_adapter("rss_coils_to_magnitude")()(x)

    def _get_estimator(self, x: torch.Tensor) -> ParameterEstimator:
        """Lazily create ParameterEstimator matching actual generator output channels.

        Args:
            x: Generator output tensor [B, C, H, W]

        Returns:
            Initialized ParameterEstimator module
        """
        actual_channels = x.shape[1]
        if self.estimator is None or self._estimator_channels != actual_channels:
            self._estimator_channels = actual_channels
            self.estimator = ParameterEstimator(in_channels=actual_channels).to(
                x.device, non_blocking=True
            )
            # Register the estimator's params with opt_g so the HF->(PD,T1,T2)
            # physics-inversion head actually trains (it was never optimized).
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                existing = {p for g in opt.param_groups for p in g["params"]}
                new_params = [p for p in self.estimator.parameters() if p not in existing]
                if new_params:
                    opt.add_param_group({"params": new_params})
        return self.estimator

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute Cycle-Bloch losses.

        Args:
            input_batch: Ultra-low-field input [B, C, H, W]
            target_batch: High-field target (optional) [B, C, H, W]
            epoch: Current epoch number
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss components:
                - 'total': Total generator loss
                - 'gan': Adversarial loss
                - 'cycle': Cycle-physics loss
        """
        real_ulf = input_batch

        generator = self.env.generator
        discriminator = self.env.discriminator

        if discriminator is None:
            raise ValueError(
                "CycleBlochStrategy requires a discriminator. Either "
                "(a) add a ``discriminator:`` section to the experiment "
                "YAML under ``model:`` (e.g. ``model.discriminator: "
                "{discriminator_type: patch_gan_2d, ...}``), or "
                "(b) switch to a non-adversarial cycle strategy such as "
                "``CycleConsistencyStrategy`` that does not depend on a "
                "discriminator."
            )

        # --- 1. Forward: ULF -> HF (Hallucination) ---
        fake_hf = generator(real_ulf)

        # --- 2. Physics Inversion: HF -> Tissue Maps ---
        # Estimate T1/T2/PD maps from the generated HF image
        estimator = self._get_estimator(fake_hf)
        tissue_maps = estimator(fake_hf)

        # --- 3. Backward: Tissue Maps -> ULF (Simulation) ---
        # Unpack tissue parameters
        pd = tissue_maps[:, 0:1]
        t1 = tissue_maps[:, 1:2]
        t2 = tissue_maps[:, 2:3]

        # Use correct Bloch Simulator targeting ULF params (Spin Echo Assumption)
        rec_ulf = BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=self.TE_ULF, tr=self.TR_ULF)

        # --- GAN Loss ---
        pred_fake = discriminator(fake_hf)
        loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))

        # --- Cycle-Bloch Loss ---
        loss_cycle = self.criterion_cycle(rec_ulf, real_ulf)

        # --- Total Generator Loss ---
        total_loss_g = (self.lambda_adv * loss_gan) + (self.lambda_cycle * loss_cycle)

        # Canonical key is ``g_total_loss``; the orchestrator at
        # base.py:845 accepts ``total`` as a synonym but the on-disk
        # convention should be the canonical handle. See
        # TODO/audit/07_strategies_v6_2_v6_3_exotic.md F18.
        return {
            "g_total_loss": total_loss_g,
            "gan": loss_gan,
            "cycle": loss_cycle,
        }

    @accepts_step_io
    def train_step(
        self,
        batch: Any,
        epoch: int,
        iteration: int = 0,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Orchestrate Cycle-Bloch training step via Trainer-compatible closures.

        Args:
            batch: Batch dictionary.
            epoch: Current epoch.
            iteration: Current iteration.
            input_batch: Pre-unpacked input (optional).
            target_batch: Pre-unpacked target (optional).

        Returns:
            List of optimization configurations for Trainer.
        """
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        real_ulf = input_batch
        if real_ulf is None:
            raise ValueError("Batch must contain 'image' or 'input' key.")

        real_ulf = real_ulf.to(self.device, non_blocking=True)
        if target_batch is not None:
            target_batch = target_batch.to(self.device, non_blocking=True)

        discriminator = self.env.discriminator
        if discriminator is None:
            raise ValueError(
                "CycleBlochStrategy requires a discriminator. Either "
                "(a) add a ``discriminator:`` section to the experiment "
                "YAML under ``model:`` (e.g. ``model.discriminator: "
                "{discriminator_type: patch_gan_2d, ...}``), or "
                "(b) switch to a non-adversarial cycle strategy such as "
                "``CycleConsistencyStrategy`` that does not depend on a "
                "discriminator."
            )

        # Prepare losses dict for shared state tracking
        losses_dict: dict[str, torch.Tensor] = {}

        # 1. Generator step (Hallucinate HF + Physics Consistency)
        g_closure = self._train_generator_step(
            real_ulf, target_batch, discriminator, epoch, iteration, losses_dict
        )

        # 2. Discriminator step (Adversarial Real/Fake)
        d_closure = self._train_discriminator_step(
            real_ulf, target_batch, discriminator, epoch, iteration, losses_dict
        )

        return [
            {
                "optimizer": self.env.opt_g,
                "closure": g_closure,
                "model": self.env.generator,
                "name": "generator",
            },
            {
                "optimizer": self.env.opt_d,
                "closure": d_closure,
                "model": discriminator,
                "name": "discriminator",
            },
        ]

    def _train_generator_step(
        self,
        real_ulf: torch.Tensor,
        target_hf: torch.Tensor | None,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, torch.Tensor],
    ) -> Any:
        """_train_generator_step.

        Args:
            real_ulf (torch.Tensor): Description.
            target_hf (Optional[torch.Tensor]): Description.
            discriminator (nn.Module): Description.
            epoch (int): Description.
            iteration (int): Description.
            losses_dict (dict[str, torch.Tensor]): Description.
        Returns:
            Any: Description.
        """

        def closure():
            """closure.

            Returns:
                Any: Description.
            """
            fake_hf = self.env.generator(real_ulf)

            # Physics Inversion: HF -> Tissue Maps
            estimator = self._get_estimator(fake_hf)
            tissue_maps = estimator(fake_hf)
            pd, t1, t2 = tissue_maps[:, 0:1], tissue_maps[:, 1:2], tissue_maps[:, 2:3]

            # Physics Simulation: Tissue Maps -> ULF
            rec_ulf = BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=self.TE_ULF, tr=self.TR_ULF)

            # Adversarial Loss
            pred_fake = discriminator(fake_hf)
            loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))

            # Cycle-Physics Loss.
            # rec_ulf is 1-ch magnitude (from BlochSimulator); real_ulf may
            # be multi-channel (virtual coils, real/imag interleaved). When
            # the YAML declares `adapters.pre_loss_target: [{name:
            # rss_coils_to_magnitude}]`, apply_adapters runs it (visible to
            # the audit). Otherwise fall back to the inline helper for
            # legacy YAMLs (which delegates to the same registered
            # adapter — see _to_magnitude_image).
            if self.adapter_chains.get("pre_loss_target"):
                rec_ulf_target = self.apply_adapters("pre_loss_target", rec_ulf)
                real_ulf_target = self.apply_adapters("pre_loss_target", real_ulf)
            else:
                rec_ulf_target = self._to_magnitude_image(rec_ulf)
                real_ulf_target = self._to_magnitude_image(real_ulf)
            loss_cycle = self.criterion_cycle(rec_ulf_target, real_ulf_target)

            total_loss_g = (self.lambda_adv * loss_gan) + (self.lambda_cycle * loss_cycle)

            losses_dict.update(
                {
                    "loss_g": total_loss_g,
                    "loss_cycle": loss_cycle,
                }
            )
            self._last_step_metrics = {k: v.detach() for k, v in losses_dict.items()}

            return total_loss_g

        return closure

    def _train_discriminator_step(
        self,
        real_ulf: torch.Tensor,
        target_hf: torch.Tensor | None,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, torch.Tensor],
    ) -> Any:
        """_train_discriminator_step.

        Args:
            real_ulf (torch.Tensor): Description.
            target_hf (Optional[torch.Tensor]): Description.
            discriminator (nn.Module): Description.
            epoch (int): Description.
            iteration (int): Description.
            losses_dict (dict[str, torch.Tensor]): Description.
        Returns:
            Any: Description.
        """

        def closure():
            """closure.

            Returns:
                Any: Description.
            """
            with torch.no_grad():
                fake_hf = self.env.generator(real_ulf).detach()

            loss_d_real = torch.tensor(0.0, device=self.device)
            if target_hf is not None:
                pred_real = discriminator(target_hf)
                loss_d_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))

            pred_fake_d = discriminator(fake_hf)
            loss_d_fake = self.criterion_gan(pred_fake_d, torch.zeros_like(pred_fake_d))

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            losses_dict["loss_d"] = loss_d
            self._last_step_metrics = {k: v.detach() for k, v in losses_dict.items()}
            return loss_d

        return closure
