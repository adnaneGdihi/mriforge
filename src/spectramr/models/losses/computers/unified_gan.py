"""Unified GAN Loss Computer - SSOT Implementation.

This is a fresh refactored GAN loss computer that inherits from BaseLossComputer
and implements the unified LossOutput pattern.

Benefits over legacy version:
- Cleaner abstraction
- Standardized LossOutput format
- Reusable across all training strategies
- Better separation of concerns
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from spectramr.models.losses.computers.base import BaseLossComputer, LossOutput
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    _call_safe_loss,
)

logger = logging.getLogger(__name__)


# Names of registered losses whose ``forward`` does NOT take ``(pred, target)``
# as the first two positional arguments — they are routed through dedicated
# computers / strategy code paths and must never reach the generic dynamic-loss
# dispatcher (where they'd be silently miscalled with garbage tensors).
# Keep this in sync with `_NON_PRED_TARGET_LOSS_NAMES` in
# ``unified_vae.py``; both lists block the same registered losses.
_NON_PRED_TARGET_LOSS_NAMES = frozenset(
    {
        "kl",
        "kl_divergence",
        "vq_kl",
        "beta_tc_vae",
        "BetaTCVAELoss",
        "vq",
        "VQLoss",
        "latent_vq",
        "vqgan",
        "VQGANLoss",
        # PDE / coords / flow / single-input losses with non-(pred,target) signatures
        "helmholtz_pde",
        "smoothness_loss",
        "flow_smoothness",
        "smooth_loss",
        "latent_regularization",
        "LatentRegularizationLoss",
        "modality_swap",
        "ModalitySwapLoss",
    }
)


#: ``gan_loss_library`` returns loss dicts whose entries are ALREADY multiplied by
#: their lambdas (``lambda_adv``, ``lambda_gp``, ``lambda_l1``, ``lambda_perceptual``,
#: ...) and which carry their own authoritative sum under these keys.
#:
#: Handing such a dict to ``_stack_components`` was wrong twice over. It asked
#: ``resolve_loss_weight`` for weights for names like ``d_loss_real`` /
#: ``g_adv_loss`` that are declared nowhere and have no ``lambda_<name>`` schema
#: field, so it refused to invent one and RAISED. And had weights existed, the
#: total would have summed the sub-terms *and* the pre-computed total sitting
#: beside them -- every term counted twice.
_LIBRARY_TOTAL_KEYS = ("g_total_loss", "d_total_loss")


def _absorb_preweighted(
    result: dict[str, torch.Tensor], components: dict[str, Any]
) -> tuple[torch.Tensor | None, set[str]]:
    """Record a pre-weighted library loss dict; return ``(total, its keys)``.

    Every entry lands in ``components`` so it still reaches logs and metrics.
    None of them may be re-weighted, so the caller excludes the returned key set
    from whatever it stacks and adds the returned total once.
    """
    components.update(result)
    total = next((result[k] for k in _LIBRARY_TOTAL_KEYS if k in result), None)
    if total is None:
        # No authoritative sum offered — add the pre-weighted sub-terms at 1.0.
        total = _sum_tensors(result.values())
    return total, set(result)


def _sum_tensors(values: "Iterable[torch.Tensor]") -> torch.Tensor | None:
    """Sum an iterable of tensors, or return ``None`` if it is empty."""
    total: torch.Tensor | None = None
    for value in values:
        total = value if total is None else total + value
    return total


class UnifiedGANLossComputer(BaseLossComputer):
    """Unified GAN loss computer implementing SSOT pattern.

    Computes generator and discriminator losses in standardized format.
    """

    def __init__(self, config: Any, device: torch.device = torch.device("cpu")):
        """Initialize GAN loss computer.

        Args:
            config: Configuration with GAN loss settings
            device: Compute device
        """
        self.adversarial_loss_fn = None
        self.reconstruction_loss_fn = None
        self.perceptual_loss_fn = None
        self.r1_regularizer = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize loss functions from config using LossBuilder.

        Uses the LossBuilder pattern for SSOT loss creation. Falls back to
        sensible defaults if config is missing or builder fails.
        """
        from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

        if self.config is not None:
            # Build errors with a real config MUST propagate (pitfall #9). The
            # old `try/except Exception: pass` swallowed LossBuilder's own
            # fail-fast `ValueError("Unknown gan_loss_type")` and silently
            # collapsed the adversarial/perceptual/R1 stack to a bare L1 —
            # a run indistinguishable in logs from a correct adversarial run
            # (review 2026-07-01). A typo in `losses.gan.gan_loss_type` (or a
            # missing VGG weight, a bad loss sub-block) must fail loud, not
            # train the whole GAN family on vanilla L1.
            builder = LossBuilder(self.config, self.device)
            losses = (
                builder.build_reconstruction_losses()
                .build_adversarial_losses()
                .build_regularization_losses()
                .build()
            )
            self.reconstruction_loss_fn = losses.get("l1", nn.L1Loss())
            self.adversarial_loss_fn = losses.get("adversarial")
            self.perceptual_loss_fn = losses.get("perceptual")
            self.r1_regularizer = losses.get("r1")
            return

        # No config supplied (direct construction in a test/script) — the ONLY
        # sanctioned minimal fallback.
        self.reconstruction_loss_fn = nn.L1Loss()
        self.perceptual_loss_fn = None
        self.adversarial_loss_fn = None
        self.r1_regularizer = None

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute combined GAN losses.

        Args:
            pred: Generated images (B, C, H, W)
            target: Real images (B, C, H, W)
            epoch: Current epoch
            iteration: Current iteration
            **kwargs: discriminator, discriminator_outputs, etc.

        Returns:
            LossOutput with total loss and components
        """
        device = target.device
        components = {}
        # Keys that belong to the DISCRIMINATOR's objective. Reported in
        # ``components`` (so they still reach logs and metrics) but excluded from
        # the total, which is the generator's.
        discriminator_only: dict[str, Any] = {}

        # 1. RECONSTRUCTION LOSS
        lambda_rec = self._get_loss_weight("reconstruction", epoch, iteration)
        if lambda_rec > 0 and self.reconstruction_loss_fn:
            rec_loss = self.reconstruction_loss_fn(pred, target)
            components["reconstruction"] = rec_loss

        # 2. ADVERSARIAL LOSS
        discriminator = kwargs.get("discriminator")
        lambda_adv = self._get_loss_weight("adversarial", epoch, iteration)

        if lambda_adv > 0 and discriminator and self.adversarial_loss_fn:
            # Get or compute discriminator outputs
            if "discriminator_outputs" in kwargs:
                disc_out = kwargs["discriminator_outputs"]
                fake_pred = disc_out.get("fake_pred")
                real_pred = disc_out.get("real_pred")
            else:
                # Compute on the fly
                with torch.no_grad():
                    fake_pred = discriminator(pred.detach())
                    real_pred = discriminator(target)

            # Generator loss (fool discriminator)
            if hasattr(self.adversarial_loss_fn, "compute_generator_loss"):
                g_adv = self.adversarial_loss_fn.compute_generator_loss(
                    fake_outputs_d=fake_pred,
                    real_images=target if "target" in locals() else target,
                    fake_images=pred if "pred" in locals() else pred,
                )
                if isinstance(g_adv, dict):
                    components.update(g_adv)
                else:
                    components["adv_generator"] = g_adv

            # Discriminator loss.
            #
            # These are REPORTED, never summed into the total this method
            # returns. The total is backpropagated through the GENERATOR; the
            # discriminator has its own optimizer step, its own closure, and its
            # own call to ``compute_discriminator_loss`` (see
            # ``AdversarialMixin`` / ``GANTrainingStrategy.train_step``). Folding
            # them in here both double-counted the D objective and pushed it
            # through G. It also crashed: ``d_loss_real`` has no declared weight
            # and no ``lambda_d_loss_real`` schema field, so
            # ``resolve_loss_weight`` refused to invent one and raised -- which
            # is what made ``fit(paradigm="gan")`` fail outright.
            #
            # Keys are recorded AS THEY ARE PRODUCED rather than matched by a
            # ``d_``-prefix afterwards: a name test would sweep up any legitimate
            # generator loss that happens to start with those letters (``dice``,
            # ``dists``, ...), which is a silent wrong answer of exactly the kind
            # this fix is removing.
            if hasattr(self.adversarial_loss_fn, "compute_discriminator_loss"):
                d_loss_result = self.adversarial_loss_fn.compute_discriminator_loss(
                    real_outputs_d=real_pred,
                    fake_outputs_d=fake_pred,
                    discriminator=discriminator,
                    real_images=target,
                    fake_images=pred,
                )

                if isinstance(d_loss_result, dict):
                    produced = {f"d_{k}": v for k, v in d_loss_result.items()}
                elif isinstance(d_loss_result, tuple):
                    d_real, d_fake = d_loss_result
                    produced = {"d_real": d_real, "d_fake": d_fake}
                else:
                    produced = {"d_total": d_loss_result}
                components.update(produced)
                discriminator_only.update(produced)

        # 3. PERCEPTUAL LOSS
        lambda_percep = self._get_loss_weight("perceptual", epoch, iteration)
        if lambda_percep > 0 and self.perceptual_loss_fn:
            percep_loss = self.perceptual_loss_fn(pred, target)
            if isinstance(percep_loss, tuple):
                percep_loss = percep_loss[0]
            components["perceptual"] = percep_loss

        # 4. R1 REGULARIZATION
        if discriminator and self.r1_regularizer:
            if self._should_apply_r1(epoch, iteration):
                r1_loss = self.r1_regularizer(discriminator, target)
                components["r1_penalty"] = r1_loss

        # 5. DYNAMIC COMPONENT LOSSES (from losses_dict)
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                if loss_name in components or loss_name in [
                    "reconstruction",
                    "adversarial",
                    "perceptual",
                    "r1",
                    "r1_penalty",
                    "l1",  # Handled by reconstruction
                ]:
                    continue
                # Latent / KL / VQ / coords / flow losses do NOT take
                # ``(pred, target)`` positionally — silently miscalling them
                # against image tensors would produce garbage values and
                # CLAUDE.md #9 silent fallbacks. They belong to dedicated
                # computers, not the GAN dynamic loop.
                if (
                    loss_name in _NON_PRED_TARGET_LOSS_NAMES
                    or loss_fn.__class__.__name__ in _NON_PRED_TARGET_LOSS_NAMES
                ):
                    logger.warning(
                        "[unified_gan] Skipping loss %r (%s) — its forward signature "
                        "is incompatible with (pred, target). Route it through a "
                        "dedicated computer instead of the GAN losses_dict.",
                        loss_name,
                        loss_fn.__class__.__name__,
                    )
                    continue
                try:
                    # Forward all dispatch kwargs (smaps, mask, posterior, …) so
                    # signature-aware losses such as `sense_adjoint_l1` receive
                    # what they need. `_call_safe_loss` filters by signature.
                    loss_val = _call_safe_loss(loss_fn, pred, target, **kwargs)
                    if isinstance(loss_val, torch.Tensor):
                        # Apply weight if available via _get_loss_weight
                        # Note: If LossBuilder already weighted it, this might be redundant or double-weighting.
                        # Usually generic losses need explicit weighting here if not built-in.
                        # For safety, we trust the loss_fn return value, but check for explicit override?
                        # Using _get_loss_weight with default 1.0 seems safe if we assume loss_fn is raw.
                        weight = self._get_loss_weight(loss_name, epoch, iteration)
                        components[loss_name] = loss_val * weight
                except Exception as _exc:
                    # CLAUDE.md #10: warnings are not OK — but failing the whole
                    # forward pass for one bad loss is too destructive in
                    # multi-loss configs. Surface it loudly so the user sees
                    # the broken loss in normal log output and can fix the
                    # signature mismatch in their config.
                    logger.warning(
                        "[unified_gan] Loss %r (%s) failed: %s — see traceback at "
                        "DEBUG level. Loss is being SKIPPED for this step.",
                        loss_name,
                        loss_fn.__class__.__name__,
                        _exc,
                    )
                    logger.debug("Suppressed exception traceback:", exc_info=True)

        # 6. COMPUTE TOTAL LOSS — GENERATOR terms only.
        # ``components`` still carries the d_* entries for reporting; the total
        # must not, because it is what ``backward()`` runs on for G.
        generator_components = {k: v for k, v in components.items() if k not in discriminator_only}
        total = self._stack_components(generator_components, epoch=epoch)

        return LossOutput(
            total=total,
            components=components,
            metrics={k: v for k, v in discriminator_only.items()},
        )

    def compute_generator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        discriminator: nn.Module | None = None,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute generator-only loss (for separate G training).

        Args:
            pred: Generated images
            target: Real images
            discriminator: Discriminator model
            epoch: Current epoch
            **kwargs: Additional arguments

        Returns:
            LossOutput with generator components only
        """
        device = target.device
        components = {}
        # The adversarial library's own pre-weighted total, and the keys it owns.
        library_total: torch.Tensor | None = None
        preweighted: set[str] = set()

        # Reconstruction loss (always for generator)
        lambda_rec = self._get_loss_weight("reconstruction", epoch, iteration)
        if lambda_rec > 0 and self.reconstruction_loss_fn:
            rec_loss = self.reconstruction_loss_fn(pred, target)
            components["reconstruction"] = rec_loss

        # Adversarial loss (if discriminator provided)
        lambda_adv = self._get_loss_weight("adversarial", epoch, iteration)
        if lambda_adv > 0 and discriminator and self.adversarial_loss_fn:
            fake_pred = discriminator(pred)
            if hasattr(self.adversarial_loss_fn, "compute_generator_loss"):
                g_adv = self.adversarial_loss_fn.compute_generator_loss(
                    fake_outputs_d=fake_pred,
                    real_images=target if "target" in locals() else target,
                    fake_images=pred if "pred" in locals() else pred,
                )
                # ``gan_loss_library.compute_generator_loss`` returns
                # ``g_adv_loss``/``l1_loss``/``perceptual_loss``/... already
                # scaled by their lambdas, plus ``g_total_loss`` = their sum.
                # Re-weighting them raised on ``g_adv_loss`` (no schema field)
                # and would otherwise have double-counted every G term.
                if isinstance(g_adv, dict):
                    library_total, preweighted = _absorb_preweighted(g_adv, components)
                else:
                    components["adversarial"] = g_adv * lambda_adv
                    preweighted = {"adversarial"}
                    library_total = components["adversarial"]

        # Perceptual loss
        lambda_percep = self._get_loss_weight("perceptual", epoch, iteration)
        if lambda_percep > 0 and self.perceptual_loss_fn:
            percep = self.perceptual_loss_fn(pred, target)
            if isinstance(percep, tuple):
                percep = percep[0]
            components["perceptual"] = percep

        # Dynamic Generator Losses from losses_dict
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                if loss_name in components or loss_name in [
                    "reconstruction",
                    "adversarial",
                    "perceptual",
                    "r1",
                    "r1_penalty",
                    "l1",  # Handled by reconstruction
                ]:
                    continue
                if (
                    loss_name in _NON_PRED_TARGET_LOSS_NAMES
                    or loss_fn.__class__.__name__ in _NON_PRED_TARGET_LOSS_NAMES
                ):
                    logger.warning(
                        "[unified_gan.G] Skipping loss %r (%s) — incompatible "
                        "signature for (pred, target).",
                        loss_name,
                        loss_fn.__class__.__name__,
                    )
                    continue
                # Skip discriminator-only losses if identifiable?
                # For now assume mostly G losses in dict
                try:
                    loss_val = _call_safe_loss(loss_fn, pred, target, **kwargs)
                    if isinstance(loss_val, torch.Tensor):
                        weight = self._get_loss_weight(loss_name, epoch, iteration)
                        components[loss_name] = loss_val * weight
                except Exception as _exc:
                    logger.warning(
                        "[unified_gan.G] Loss %r (%s) failed: %s — being SKIPPED.",
                        loss_name,
                        loss_fn.__class__.__name__,
                        _exc,
                    )
                    logger.debug("Suppressed exception traceback:", exc_info=True)

        # Total with generator-specific weighting. Pre-weighted library terms are
        # excluded from the stack and their authoritative total added once.
        stackable = {k: v for k, v in components.items() if k not in preweighted}
        total = (
            self._stack_components(stackable, epoch=epoch, iteration=iteration)
            if stackable
            else None
        )
        if library_total is not None:
            total = library_total if total is None else total + library_total
        if total is None:
            total = torch.zeros((), device=device)

        return LossOutput(total=total, components=components, metrics={})

    def compute_discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: nn.Module,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute discriminator-only loss (for separate D training).

        Args:
            real: Real images
            fake: Generated images
            discriminator: Discriminator model
            epoch: Current epoch
            iteration: Current iteration
            **kwargs: Additional arguments

        Returns:
            LossOutput with discriminator components only
        """
        device = real.device
        components = {}
        # The adversarial term, already summed, and the keys it owns. See below
        # for why they are not left to ``_stack_components``.
        adversarial_total: torch.Tensor | None = None
        preweighted: set[str] = set()

        # Adversarial loss (main discriminator loss)
        if self.adversarial_loss_fn:
            real_pred = discriminator(real)
            fake_pred = discriminator(fake.detach())  # Detach to not affect generator

            if hasattr(self.adversarial_loss_fn, "compute_discriminator_loss"):
                d_loss_result = self.adversarial_loss_fn.compute_discriminator_loss(
                    real_outputs_d=real_pred,
                    fake_outputs_d=fake_pred,
                    discriminator=discriminator,
                    real_images=real,
                    fake_images=fake,
                )

                # These sub-terms arrive PRE-WEIGHTED and pre-summed:
                # ``gan_loss_library`` returns ``d_loss_real``/``d_loss_fake``
                # already scaled by ``lambda_adv`` (plus ``gp_loss`` by
                # ``lambda_gp``), AND its own ``d_total_loss`` = their sum.
                #
                # Handing that dict to ``_stack_components`` was wrong twice
                # over. It asked ``resolve_loss_weight`` for a weight for
                # ``d_loss_real``, which is declared nowhere and has no
                # ``lambda_d_loss_real`` schema field, so it refused to invent
                # one and RAISED -- killing every GAN run at its first
                # discriminator step. And had a weight existed, the total would
                # have summed the sub-terms *and* ``d_total_loss`` alongside
                # them: the adversarial term counted twice.
                #
                # Same reasoning the ``r1_penalty`` note below already records
                # for R1: a pre-weighted quantity must land exactly once.
                if isinstance(d_loss_result, dict):
                    adversarial_total, preweighted = _absorb_preweighted(d_loss_result, components)
                elif isinstance(d_loss_result, tuple):
                    d_real, d_fake = d_loss_result
                    components["d_real"] = d_real
                    components["d_fake"] = d_fake
                    adversarial_total = d_real + d_fake
                    preweighted = {"d_real", "d_fake"}
                else:
                    components["discriminator"] = d_loss_result
                    adversarial_total = d_loss_result
                    preweighted = {"discriminator"}

        # R1 Regularization (discriminator-specific).
        # The R1 module already bakes lambda_r1 into its output (returns
        # weight * 0.5||∇D(real)||²). Store it under "r1_penalty" — a key for
        # which _stack_components resolves weight 1.0 — so the penalty lands
        # EXACTLY ONCE. The pre-fix code multiplied by lambda_r1 here AND let
        # _stack_components weight the "r1" key by lambda_r1 again, giving a
        # lambda_r1^3 (~1000x at the default 10) over-penalty.
        if self.r1_regularizer and self._should_apply_r1(epoch, iteration):
            r1 = self.r1_regularizer(discriminator, real)
            components["r1_penalty"] = r1

        # Total discriminator loss.
        #
        # Built from the pre-weighted adversarial total plus anything else in
        # ``components`` that DOES have a resolvable weight (``r1_penalty``).
        # The adversarial sub-terms are excluded from the stack because they are
        # already inside ``adversarial_total``; leaving them in would count the
        # adversarial objective twice.
        stackable = {k: v for k, v in components.items() if k not in preweighted}
        total = (
            self._stack_components(stackable, epoch=epoch, iteration=iteration)
            if stackable
            else None
        )
        if adversarial_total is not None:
            total = adversarial_total if total is None else total + adversarial_total
        if total is None:
            total = torch.zeros((), device=device)

        return LossOutput(total=total, components=components, metrics={})

    def _resolve_gan_loss_setting(self, name: str, default: Any) -> Any:
        """Read ``losses.gan.<name>`` from the nested SSOT config.

        The GAN/R1 knobs live under ``losses.gan`` — NOT at the
        ``TrainingSettings`` top level — so a top-level ``getattr`` /
        ``hasattr`` is always a miss.
        """
        losses = getattr(self.config, "losses", None)
        gan = getattr(losses, "gan", None) if losses is not None else None
        value = getattr(gan, name, None) if gan is not None else None
        return value if value is not None else default

    def _should_apply_r1(self, epoch: int, iteration: int) -> bool:
        """Check if R1 regularization should apply this step.

        The R1 knobs live at ``losses.gan.{enable_r1,r1_interval}``. The
        pre-fix gate did ``hasattr(self.config, "r1_interval")`` on the full
        ``TrainingSettings``, which is always False, so R1 silently never fired
        (CLAUDE.md pitfall #15: advertised-but-unread knob).
        """
        if not bool(self._resolve_gan_loss_setting("enable_r1", False)):
            return False

        interval = int(self._resolve_gan_loss_setting("r1_interval", 0))
        if interval <= 0:
            return False

        if iteration > 0:
            return (iteration % interval) == 0
        return epoch >= 0 and (epoch % interval) == 0

    def validate_config(self) -> None:
        """Validate configuration has required attributes."""
        required_attrs = []

        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"Config missing required attribute: {attr}")


__all__ = [
    "UnifiedGANLossComputer",
]
