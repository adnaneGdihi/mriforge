"""Loss Builder

Creates loss functions based on objectives configuration.
Supports reconstruction, adversarial, physics, and regularization losses.
SSOT for loss instantiation.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from mriforge.config.settings import TrainingSettings
from mriforge.core.component_signature import signature_contract
from mriforge.domain.exceptions import ConfigurationError
from mriforge.models.losses.registry import LossRegistry, create_loss

from .base import Builder

logger = logging.getLogger(__name__)

#: Losses that ``LossBuilder`` must NOT instantiate as standalone modules
#: because a training strategy computes them inline (they need gradients,
#: encoder features, an injected ``context`` dict, or are bundled inside a
#: composite GAN loss). This is the SSOT for the skip set — the loss-coverage
#: auditor (``infrastructure/loss_audit.py``) imports it so the two cannot
#: drift (the F41 2026-05-23 incident, re-broken when this set grew but the
#: auditor's private copy did not — re-derived 2026-06-11).
STRATEGY_MANAGED_LOSSES: frozenset[str] = frozenset(
    {
        # Strategy-managed (built by the training strategy)
        "commitment",
        "codebook",
        "r1",
        "patch_nce",
        "reconstruction",
        "marker",
        "prior",
        "distill",
        "sim",
        "smooth",
        "padnet_l2",
        "padnet_dc",
        "padnet_reg",
        "bloch",
        "anat",
        "style",
        "content",
        "recon",
        "bloch_residual",
        "physics_constraint",
        "parallel_imaging_kspace",
        "snr_preserving",
        "biophysical_flow",
        "physics_informed",
        # Composite-bundled GAN sub-losses (built by ``_build_composite_gan``;
        # live inside ``losses.gan`` as ``enable_gradient_penalty`` /
        # ``lambda_gp`` / ``lambda_feat_match`` rather than declarative entries).
        "gradient_penalty",
        "feature_matching",
    }
)


class LossBuilder(Builder):
    """Builds loss functions dynamically from objectives config.

    Creates all loss functions needed for training by automatically parsing
    the enabled losses from the LossConfigSchema via `get_enabled_losses()`.
    Config-specific parameters are injected automatically based on loss type.
    """

    def __init__(self, config: TrainingSettings, device: Any):
        """__init__.

        Args:
            config (TrainingSettings): Description.
            device (Any): Description.
        """
        self._config = config
        self._device = device
        self._losses: dict[str, nn.Module] = {}
        self._already_built_dynamic = False

    def get_enabled_losses(self) -> dict[str, float]:
        """get_enabled_losses.

        Returns:
            dict[str, float]: Description.
        """
        if hasattr(self._config.losses, "get_enabled_losses"):
            return self._config.losses.get_enabled_losses()
        return {}

    def get_loss_weights(self) -> dict[str, float]:
        """get_loss_weights.

        Returns:
            dict[str, float]: Description.
        """
        return self.get_enabled_losses()

    def validate_loss_configuration(self) -> bool:
        """validate_loss_configuration.

        Returns:
            bool: Description.
        """
        return True

    def _build_all_dynamic(self):
        """Build all enabled losses dynamically exactly once."""
        if self._already_built_dynamic:
            return
        self._already_built_dynamic = True

        enabled_losses = self.get_enabled_losses()
        if not enabled_losses:
            logger.debug("No enabled losses found in configuration.")
            return

        recon_config = self._config.losses.reconstruction
        gan_config = self._config.losses.gan
        # ``physics``/``evidential`` are read by _schema_loss_kwargs (the SSOT this
        # path now consumes), so they are no longer locals here.

        # ==================== LIST-BASED DOMAIN-AWARE PATH ====================
        # When kspace_losses / image_losses / complex_losses are populated,
        # auto-bridge based on the declared output_domain.
        if self._config.losses.uses_list_based_losses:
            self._build_list_based_losses()
            # After building list-based losses, continue to paradigm-specific
            # losses (GAN, diffusion, latent, etc.) that use the old path.
            # Only build paradigm losses that aren't already in the list-based set.
            # Derived from LOSS_LIST_DOMAINS, not a hand-written tuple. A list
            # missing here is not seen as already-built, so the paradigm
            # fallback below flags its entries as "unmigrated" and refuses the
            # run -- for losses that were in fact built moments earlier.
            from mriforge.config.schemas.loss import LOSS_LIST_DOMAINS

            list_loss_names = {
                c.name
                for list_name in LOSS_LIST_DOMAINS
                for c in getattr(self._config.losses, list_name)
            }
            # Build GAN composite if enabled and not in list-based
            if "adversarial" not in list_loss_names and gan_config:
                if enabled_losses.get("adversarial", 0) > 0:
                    self._build_composite_gan(gan_config, recon_config)

            # Deep Supervision special case
            ds_weight = self._config.losses.lambda_deep_supervision
            if ds_weight > 0:
                try:
                    self._losses["deep_supervision"] = create_loss(
                        "deep_supervision", weight=ds_weight
                    ).to(self._device, non_blocking=True)
                    logger.info("Created Deep Supervision loss")
                except Exception as e:
                    raise ConfigurationError(f"Failed to create Deep Supervision loss: {e}") from e

            # Catch any remaining paradigm-specific losses (diffusion, latent, SSL, etc.)
            # that are enabled but were not processed by the declarative lists or GAN bypass.
            #
            # Strategy-managed losses are computed inline by training strategies
            # (e.g., R1 needs discriminator gradients, PatchNCE needs encoder
            # features) or bundled into a composite GAN loss — see the
            # module-level SSOT constant.
            strategy_managed = STRATEGY_MANAGED_LOSSES

            # Registry name translation (same map as legacy flag-based path)
            _fallback_registry_map = {
                "l2": "mse",
                "r1": "r1_regularization",
                "hist": "histogram_consistency",
                "ffl": "focal_frequency",
                "edge": "sobel_edge",
                "dc": "data_consistency",
                "frequency_domain": "frequency_domain_consistency",
                "ms_ssim": "ms_ssim",
                "pde": "helmholtz_pde",
                "pinn_dc": "data_consistency",
                "bloch": "bloch_residual",
            }

            # Losses computed by the reconstruction loss computer resolve their
            # weight from ``reconstruction.lambda_*`` (resolve_static_loss_weight)
            # and are intentionally kept OUT of the declarative lists — they are
            # NOT silently skipped, so the guard must not flag them (the
            # direct_ulf_to_hf_sr / pma_02 pattern). Same idea as strategy_managed.
            recon_managed = self._config.losses.reconstruction_managed_losses()

            unmigrated: list[tuple[str, float]] = []
            for loss_name, weight in enabled_losses.items():
                if loss_name in strategy_managed:
                    logger.debug(
                        "Skipping '%s' (weight=%s) — strategy-managed loss",
                        loss_name,
                        weight,
                    )
                    continue
                # Translate legacy flag names (``l2``/``edge``/``ffl``/``dc`` …)
                # to their registry twins before the membership test — otherwise
                # a migrated config that uses the legacy alias while its
                # registry-name twin sits in the declarative lists is falsely
                # rejected as "unmigrated". (The map was previously dead.)
                effective = _fallback_registry_map.get(loss_name, loss_name)
                known = (
                    loss_name in self._losses
                    or effective in self._losses
                    or loss_name in list_loss_names
                    or effective in list_loss_names
                    or loss_name in recon_managed
                    or effective in recon_managed
                )
                if not known and weight > 0:
                    unmigrated.append((loss_name, weight))

            if unmigrated:
                # Per CLAUDE.md #9 (silent fallbacks are forbidden) and #10
                # (warnings are not OK): a non-zero loss weight for a key
                # that isn't in the declarative kspace/image/complex lists
                # is an unambiguous config bug — it would silently train
                # without that loss. Refuse instead of warn-and-skip.
                lines = [
                    f"  • '{n}' (weight={w}) → migrate into objectives.image_losses, "
                    f"objectives.kspace_losses, or objectives.complex_losses"
                    for n, w in unmigrated
                ]
                raise ConfigurationError(
                    "Loss configuration references key(s) that are not in the v6.0 "
                    "declarative list-based losses (kspace_losses / image_losses / "
                    "complex_losses). Silently skipping them would train without the "
                    "advertised supervision — refusing per CLAUDE.md #9. Unmigrated "
                    "key(s):\n" + "\n".join(lines)
                )

            return

        # ==================== LEGACY FLAG-BASED PATH ====================
        use_universal_bridge = (
            recon_config.spatial_losses_use_fourier_bridge if recon_config else False
        )

        class FourierBridgeLossWrapper(nn.Module):
            """Wraps any spatial loss with a Fourier Bridge for k-space compatibility."""

            def __init__(self, inner_loss: nn.Module):
                """__init__.

                Args:
                    inner_loss (nn.Module): Description.
                """
                super().__init__()
                from mriforge.models.losses.physics_losses import (
                    DifferentiableFourierBridge,
                )

                self.bridge = DifferentiableFourierBridge(
                    spatial_loss_fn=inner_loss, return_complex=False
                )

            def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
                """forward.

                        Args:
                            pred (torch.Tensor): Description.
                            target (torch.Tensor): Description.
                        Returns:
                            torch.Tensor: Description.

                forward method for FourierBridgeLossWrapper.

                Executes PyTorch tensor operations.

                Args:
                    pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
                    target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
                """
                return self.bridge(pred, target, **kwargs)

        def wrap_spatial_loss(loss_module: nn.Module) -> nn.Module:
            """wrap_spatial_loss.

            Args:
                loss_module (nn.Module): Description.
            Returns:
                nn.Module: Description.
            """
            if use_universal_bridge:
                return FourierBridgeLossWrapper(loss_module)
            return loss_module

        spatial_losses = {
            "l1",
            "smooth_l1",
            "mse",
            "perceptual",
            "ssim",
            "ms_ssim",
            "dists",
            "lpips",
            "sobel_edge",
            "explicit_gradient",
            "graph_consistency",
            "spectral_graph",
        }

        registry_map = {
            "l2": "mse",
            "hist": "histogram_consistency",
            "ffl": "focal_frequency",
            "edge": "sobel_edge",
            "explicit_gradient": "explicit_gradient",
            "graph_consistency": "graph_consistency",
            "spectral_graph": "spectral_graph",
            "dc": "data_consistency",
            "r1": "r1_regularization",
            "frequency_domain": "frequency_domain_consistency",
            "ms_ssim": "ms_ssim",
            "pde": "helmholtz_pde",
            "pinn_dc": "data_consistency",
        }

        if recon_config and hasattr(recon_config, "loss_type"):
            registry_map["reconstruction"] = recon_config.loss_type
            registry_map["recon"] = recon_config.loss_type
        else:
            registry_map["reconstruction"] = "l1"
            registry_map["recon"] = "l1"

        # Kwargs Map Extraction (SSOT shared with the list-based path;
        # see _schema_loss_kwargs -- issue #467 follow-up).
        kwargs_map = self._schema_loss_kwargs()

        # Main Instantiation Loop
        for cfg_name, weight in enabled_losses.items():
            if weight <= 0 and cfg_name not in ("adversarial",):
                continue

            # Special cases for GAN/Adversarial
            if cfg_name == "adversarial" and gan_config:
                self._build_composite_gan(gan_config, recon_config)
                continue

            # Strategy-managed losses: commitment and codebook are computed
            # internally by VQ-VAE/VQ-GAN strategies, not standalone modules.
            # Skip them here; the strategy injects them at training time.
            strategy_managed = STRATEGY_MANAGED_LOSSES
            if cfg_name in strategy_managed:
                logger.debug(
                    "Skipping '%s' loss (weight=%s) — managed by training strategy",
                    cfg_name,
                    weight,
                )
                continue

            # Truly unimplemented losses — fail fast
            if cfg_name == "tissue_bounds":
                raise ConfigurationError(
                    f"Loss '{cfg_name}' is not implemented. "
                    "Remove it from the loss config or provide a registered implementation."
                )

            # Resolve mapped names and load parameters
            reg_name = registry_map.get(cfg_name, cfg_name)
            kwargs = kwargs_map.get(reg_name, {})

            try:
                loss_fn = create_loss(reg_name, **kwargs).to(self._device, non_blocking=True)
                if use_universal_bridge and reg_name in spatial_losses:
                    loss_fn = wrap_spatial_loss(loss_fn)
                self._losses[cfg_name] = loss_fn
                logger.info(f"Created {cfg_name} loss (weight={weight})")
            except Exception as e:
                raise ConfigurationError(f"Failed to create {cfg_name} loss: {e}") from e

        # Deep Supervision special case
        ds_weight = self._config.losses.lambda_deep_supervision
        if ds_weight > 0:
            try:
                self._losses["deep_supervision"] = create_loss(
                    "deep_supervision", weight=ds_weight
                ).to(self._device, non_blocking=True)
                logger.info("Created Deep Supervision loss")
            except Exception as e:
                raise ConfigurationError(f"Failed to create Deep Supervision loss: {e}") from e

    def _schema_loss_kwargs(self) -> dict[str, Any]:
        """Per-loss constructor kwargs declared on the loss SCHEMA (not per-entry).

        SSOT for both build paths. ``_build_all_dynamic`` always honoured these; the
        list-based path did not, so a knob like
        ``losses.reconstruction.log_spectral_skip_fft: true`` was read by the schema,
        logged as configured, and then silently dropped -- ``log_spectral`` ran with
        ``skip_fft=False`` and applied a forward ``fft2c`` to an already-k-space
        tensor, turning a log-SPECTRAL penalty into an image-domain log-magnitude
        one. 17 arms carried exactly that (issue #467 follow-up).

        A per-entry ``kwargs:`` on the list item still wins -- this is the default,
        not an override.
        """
        recon_config = self._config.losses.reconstruction
        physics_config = self._config.losses.physics
        gan_config = self._config.losses.gan
        # ``evidential`` is an optional sub-schema (default None on the v6.0
        # LossConfigSchema). Use getattr so unit-test mocks that don't spec
        # it (and any older configs that predate the field) don't raise.
        evidential_config = getattr(self._config.losses, "evidential", None)

        kwargs_map: dict[str, Any] = {}
        if recon_config:
            kwargs_map["ms_ssim"] = {"data_range": 1.0}
            kwargs_map["ssim"] = {"data_range": 1.0}
            kwargs_map["perceptual"] = {}
            kwargs_map["mind_ssc"] = {}
            kwargs_map["explicit_gradient"] = {}
            kwargs_map["log_spectral"] = {"skip_fft": recon_config.log_spectral_skip_fft}
            kwargs_map["focal_frequency"] = {"alpha": recon_config.ffl_alpha}
            kwargs_map["histogram_consistency"] = {"bins": recon_config.histogram_bins}
            kwargs_map["background_suppression"] = {
                "threshold_ratio": recon_config.background_suppression_threshold_ratio,
                "use_fourier_bridge": recon_config.background_suppression_use_fourier_bridge,
            }
            kwargs_map["rician_consistency"] = {
                "sigma": recon_config.rician_noise_sigma,
                "use_fourier_bridge": recon_config.rician_use_fourier_bridge,
            }
            kwargs_map["frequency_weighted_l1_kspace"] = {
                "alpha": recon_config.frequency_weighted_l1_kspace_alpha
            }
            kwargs_map["weighted_kspace_l1"] = {"exponent": recon_config.weighted_kspace_exponent}
            kwargs_map["sobolev_kspace"] = {}  # Parameterless: weight injected at compute time
            kwargs_map["sense_adjoint_l1"] = {}  # smaps injected dynamically via _call_safe_loss
            kwargs_map["spectral_graph"] = {
                "k": recon_config.spectral_graph_k,
                "patch_size": recon_config.spectral_graph_patch_size,
            }
            try:
                patch_size = self._config.data.sampling.patch_size
                kwargs_map["frequency_weighted_l1_kspace"]["height"] = patch_size[0]
                kwargs_map["frequency_weighted_l1_kspace"]["width"] = patch_size[1]
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        if physics_config:
            kwargs_map["data_consistency"] = {"lambda_dc": physics_config.lambda_physics_constraint}
            kwargs_map["complex_spatial_gradient"] = {
                "use_fourier_bridge": physics_config.complex_spatial_gradient_use_fourier_bridge
            }

        if gan_config:
            kwargs_map["r1_regularization"] = {"weight": gan_config.lambda_r1}

        if evidential_config:
            kwargs_map["evidential"] = {"coeff": evidential_config.lambda_evidential}

        return kwargs_map

    def _build_list_based_losses(self):
        """Build losses from domain-aware kspace/image/complex loss lists.

        Automatically wraps losses with DifferentiableFourierBridge when the
        loss's expected domain differs from the model's output_domain.

        Bridge matrix (output_domain → loss list → action), matching the code below:
          output_domain=kspace:
            kspace_losses  → no bridge (native k-space)
            image_losses   → iFFT bridge (magnitude)
            complex_losses → iFFT bridge (return_complex=True)
          output_domain=image:
            kspace_losses  → no bridge (self-FFT losses only; see the note below)
            image_losses   → no bridge (native)
            complex_losses → no bridge (cast to complex if needed)
          output_domain=complex_image:
            kspace_losses  → no bridge
            image_losses   → no bridge (loss extracts magnitude itself)
            complex_losses → no bridge (native)

        A loss that bridges *internally* (``use_fourier_bridge=True``, e.g.
        ``complex_spatial_gradient`` / ``sense_adjoint_l1`` / ``rician_consistency``)
        must be declared under the list whose bridge mode is ``none``, otherwise the
        tensor is inverse-transformed TWICE. That is silent — both transforms are
        individually valid and the composite is finite — so it is rejected here
        rather than left to produce a meaningless objective (issue #467).
        """
        from mriforge.models.losses.physics_losses import DifferentiableFourierBridge

        output_domain = self._config.losses.policy.output_domain

        class _BridgedLoss(nn.Module):
            """Wraps a spatial loss with DifferentiableFourierBridge for auto domain conversion."""

            def __init__(self, inner_loss: nn.Module, return_complex: bool = False):
                super().__init__()
                self.bridge = DifferentiableFourierBridge(
                    spatial_loss_fn=inner_loss, return_complex=return_complex
                )

            def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
                return self.bridge(pred, target, **kwargs)

        def _create_and_register(name: str, weight: float, kwargs: dict, bridge_mode: str):
            """Create a loss, optionally wrap it, and register.

            Args:
                name: Registry name for the loss.
                weight: Loss weight (for logging only; actual weighting done by the computer).
                kwargs: Extra kwargs for the loss constructor.
                bridge_mode: One of 'none', 'ifft_magnitude', 'ifft_complex'.
            """
            try:
                loss_fn = create_loss(name, **kwargs).to(self._device, non_blocking=True)

                # Reject the double bridge (issue #467). A loss that carries its
                # own DifferentiableFourierBridge would be inverse-transformed a
                # second time by the wrapper below: the outer bridge emits an
                # image (magnitude or complex), the inner one re-reads it as
                # k-space, halves the coil count by re-pairing channels as
                # (real, imag), and iFFTs again. Nothing raises and the value is
                # finite, so the run stays green while the term measures nothing
                # it advertises. Fail loud instead (pitfalls #9 / #16).
                if bridge_mode != "none" and getattr(loss_fn, "use_fourier_bridge", False):
                    raise ConfigurationError(
                        f"Loss '{name}' bridges from k-space internally "
                        f"(use_fourier_bridge=True) but is declared under a loss "
                        f"list that adds a '{bridge_mode}' bridge for "
                        f"output_domain='{output_domain}' — the tensor would be "
                        f"inverse-transformed twice. Declare '{name}' under "
                        f"losses.kspace_losses (bridge 'none', so its own bridge "
                        f"does the single iFFT), or keep the current list and set "
                        f"kwargs: {{use_fourier_bridge: false}} on the entry so "
                        f"the outer bridge is the only one. Note it is `kwargs:`, "
                        f"not `config:` — LossComponentConfig is extra='ignore', "
                        f"so a `config:` block is silently dropped (issue #468)."
                    )

                if bridge_mode == "ifft_magnitude":
                    loss_fn = _BridgedLoss(loss_fn, return_complex=False)
                    logger.info(f"Created {name} loss (weight={weight}) with iFFT→magnitude bridge")
                elif bridge_mode == "ifft_complex":
                    loss_fn = _BridgedLoss(loss_fn, return_complex=True)
                    logger.info(f"Created {name} loss (weight={weight}) with iFFT→complex bridge")
                else:
                    logger.info(f"Created {name} loss (weight={weight}) [native domain]")

                self._losses[name] = loss_fn
            except ConfigurationError:
                # Already a precise, actionable message (e.g. the double-bridge
                # rejection above) — do not bury it under "Failed to create".
                raise
            except Exception as e:
                raise ConfigurationError(f"Failed to create {name} loss: {e}") from e

        # Determine bridge modes based on output_domain
        if output_domain == "kspace":
            kspace_bridge = "none"
            image_bridge = "ifft_magnitude"
            complex_bridge = "ifft_complex"
        elif output_domain == "image":
            # Image-output model: kspace losses receive image-domain inputs
            # as-is. Some kspace losses self-FFT (e.g. focal_frequency,
            # log_spectral), others (complex_l1) do NOT — those will
            # silently produce wrong values. The right declarative fix is
            # to declare an explicit `adapters.pre_loss_pred:
            # [fft_image_to_kspace]` chain when the model truly outputs
            # image and kspace losses are wanted.
            kspace_bridge = "none"
            image_bridge = "none"
            complex_bridge = "none"
        elif output_domain == "complex_image":
            kspace_bridge = "none"
            image_bridge = "none"  # Loss must handle magnitude extraction itself
            complex_bridge = "none"
        elif output_domain == "latent":
            # The model emits a latent. No Fourier bridge is meaningful here --
            # a latent has no k-space -- so every list is native and the schema
            # forbids the combinations that would need one.
            kspace_bridge = "none"
            image_bridge = "none"
            complex_bridge = "none"
        else:
            # The message used to omit 'latent', which the branch directly
            # above handles -- so a correct latent arm that reached here by some
            # other route was told to use a domain it was already not using.
            raise ValueError(
                f"Invalid output_domain='{output_domain}'. "
                "Must be 'kspace', 'image', 'complex_image' or 'latent'. "
                "(The schema now refuses the other SignalDomain members at load, "
                "so reaching this branch means the config bypassed validation.)"
            )

        # Schema-declared per-loss kwargs (SSOT with the legacy path). A knob like
        # ``losses.reconstruction.log_spectral_skip_fft`` used to be honoured ONLY by
        # ``_build_all_dynamic``, so a list-based arm read it, logged it, and then built
        # the loss with its constructor default -- ``log_spectral`` applied a forward
        # ``fft2c`` to an already-k-space tensor, i.e. a log-magnitude penalty on the
        # IMAGE where the YAML asked for one on the spectrum. A per-entry ``kwargs:``
        # still wins; these are defaults, not overrides.
        schema_kwargs = self._schema_loss_kwargs()

        def _merged(component) -> dict:
            merged = {
                **schema_kwargs.get(component.name, {}),
                **(component.kwargs or {}),
            }
            if not merged:
                return merged

            # A kwarg that never reaches the constructor changes the OBJECTIVE
            # silently. `sobolev_order: 1` was declared by 56 arms, swallowed by
            # `extra="ignore"`, and only found by a manual audit weeks later
            # (#560, #615) -- `SobolevKSpaceLoss` had no `order` parameter at
            # all. Raising here rather than letting `create` fail with a bare
            # TypeError is what makes the message actionable.
            #
            # Posture is RAISE, and that is a measured choice: across the
            # loadable corpus all 33 declared `kwargs:` keys and all 19
            # schema-derived `kwargs_map` entries already reach their ctor, so
            # this rejects nothing that exists today. Same rung as
            # `scheduler_resolution`, which raises on an unroutable knob.
            loss_cls = LossRegistry.get_loss_class(component.name)
            if loss_cls is None:
                return merged  # an unknown loss name is `create`'s error to report
            contract = signature_contract(loss_cls)
            # A **kwargs ctor accepts everything (it may read keys via
            # `kwargs.get`), and a class with no owned `__init__` declares no
            # contract. Neither can be checked here.
            if contract.accepts_var_kwargs or not contract.owner:
                return merged
            unroutable = sorted(k for k in merged if k not in contract.accepted)
            if unroutable:
                raise ConfigurationError(
                    f"Loss {component.name!r} cannot consume {unroutable}. "
                    f"{contract.owner}.__init__ accepts "
                    f"{sorted(contract.accepted)}. A kwarg that never reaches "
                    f"the constructor changes the objective silently."
                )
            return merged

        # Build each list
        for component in self._config.losses.kspace_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(
                    component.name, component.weight, _merged(component), kspace_bridge
                )

        for component in self._config.losses.image_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(
                    component.name, component.weight, _merged(component), image_bridge
                )

        for component in self._config.losses.complex_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(
                    component.name, component.weight, _merged(component), complex_bridge
                )

        # Latent losses are native by construction: the schema only admits them
        # with output_domain='latent', so there is never a bridge to insert.
        for component in self._config.losses.latent_losses:
            if component.enabled and component.weight > 0:
                _create_and_register(component.name, component.weight, _merged(component), "none")

    def _build_composite_gan(self, gan_config, recon_config):
        """Helper to build CompositeGANLoss."""
        try:
            gan_loss_type = gan_config.gan_loss_type
            adv_registry_map = {
                "vanilla": "gan_standard",
                "standard": "gan_standard",
                "bce": "gan_standard",
                "lsgan": "gan_lsgan",
                "ralsgan": "gan_ralsgan",
                "wgan": "gan_wgan",
                "wgan-gp": "gan_wgan",
                "hinge": "gan_hinge",
            }
            adv_name = adv_registry_map.get(gan_loss_type.lower())
            if adv_name is None:
                # NN#3: an unknown registered-option value must raise, never
                # silently degrade to the standard GAN loss.
                raise ConfigurationError(
                    f"Unknown gan_loss_type: {gan_loss_type!r}. "
                    f"Valid values: {sorted(adv_registry_map)}"
                )
            adv_strategy = create_loss(adv_name, label_smoothing=gan_config.label_smoothing)

            # Re-use generated perceptual if available
            perceptual = self._losses.get("perceptual", None)
            lambda_perceptual = recon_config.lambda_perceptual if recon_config else 0.0
            if lambda_perceptual > 0 and perceptual is None:
                try:
                    perceptual = create_loss("perceptual").to(self._device, non_blocking=True)
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

            gan_loss = create_loss(
                "gan_composite",
                adv_strategy=adv_strategy,
                perceptual_loss=perceptual,
                lambda_l1=(recon_config.lambda_l1 if recon_config else 10.0),
                lambda_perceptual=lambda_perceptual,
                lambda_adv=gan_config.lambda_adv,
                lambda_feat_match=gan_config.feature_matching,
                lambda_gp=gan_config.lambda_gp,
                lambda_ssim=(recon_config.lambda_ssim if recon_config else 0.0),
                lambda_ms_ssim=(recon_config.lambda_ms_ssim if recon_config else 0.0),
                lambda_lpips=(recon_config.lambda_lpips if recon_config else 0.0),
            ).to(self._device, non_blocking=True)

            self._losses["adversarial"] = gan_loss
            logger.info(f"Created adversarial loss (type={gan_loss_type})")
        except ConfigurationError:
            # Already a descriptive config error (e.g. unknown gan_loss_type) —
            # propagate unchanged rather than re-wrapping it.
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to create adversarial loss: {e}") from e

    # Aliases to keep API compatibility with pipelines calling build_X_losses()
    def build_reconstruction_losses(self) -> "LossBuilder":
        """build_reconstruction_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_adversarial_losses(self) -> "LossBuilder":
        """build_adversarial_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_physics_losses(self) -> "LossBuilder":
        """build_physics_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_regularization_losses(self) -> "LossBuilder":
        """build_regularization_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_structural_losses(self) -> "LossBuilder":
        """build_structural_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_diffusion_losses(self) -> "LossBuilder":
        """build_diffusion_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_latent_losses(self) -> "LossBuilder":
        """build_latent_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def build_ssl_losses(self) -> "LossBuilder":
        """build_ssl_losses.

        Returns:
            'LossBuilder': Description.
        """
        self._build_all_dynamic()
        return self

    def validate(self) -> "LossBuilder":
        """validate.

        Returns:
            'LossBuilder': Description.
        """
        if not self._losses:
            raise ConfigurationError(
                "No losses were built by LossBuilder. Training cannot proceed. "
                "Ensure that the configuration has an 'objectives' section with valid enabled losses."
            )
        logger.info(f"Loss validation passed ({len(self._losses)} losses created)")
        return self

    def build(self) -> dict[str, nn.Module]:
        """build.

        Returns:
            dict[str, nn.Module]: Description.
        """
        return dict(self._losses)
