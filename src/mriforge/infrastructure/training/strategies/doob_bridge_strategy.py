"""7T-marginal-score-guided SDEdit translation ("Doob bridge", MICCAI MRIxFields2026, B-1.10).

This realises the B-1.10 spec ("start a diffusion from the source and condition it to hit the
7T density ``h`` at ``t=T`` via Doob's h-transform"). An UNCONDITIONAL eps-net
(:class:`~mriforge.models.generators.doob_marginal_score_unet.DoobMarginalScoreUNet`) is
trained by denoising score matching on the 7T TARGET alone, so its prediction is the 7T
marginal score ``grad log p_t^{7T} = -eps/sqrt(1-alphabar)`` = the spec's ``grad log h``.
Sampling starts from a partially-noised source (SDEdit init, the bridge initial condition)
and runs the score-based reverse pass; the learned 7T-marginal score in the reverse drift is
what carries samples to the 7T density, while the noised-source init preserves subject
identity.

HONEST SCOPE (read before interpreting): for a TRANSLATION task the terminal 7T image is the
unknown we predict, so a genuine point-wise Doob bridge with an analytic pin to a *known*
endpoint (as in B-3.3's Brownian bridge) is impossible — the only realisable "pin" is to the
7T MARGINAL via the learned score. But adding ``grad log p_t^{7T}`` to the reverse drift is
exactly Anderson's reverse SDE, i.e. standard score-based reverse sampling from a noised
source (SDEdit). So this is NOT a separately-ablatable terminal-conditioning correction on
top of a surviving base drift; the marginal score IS the whole reverse dynamics. We keep the
"Doob/bridge" label because it traces to the spec's formulation, but the operational method is
marginal-score-guided SDEdit. The shipped arms run the deterministic limit (``eta=0``, the
probability-flow / DDIM reverse); the ``g dW`` term is active only for ``eta>0``.

Still genuinely distinct from the siblings (NOT a facade-clone #16): B-3.3 field_bridge is
Brownian-bridge VELOCITY MATCHING (no learned marginal score); B-3.5 field_guided_diffusion
conditions the score on the SOURCE+FIELD and steers with post-hoc CFG. Here the score is
UNCONDITIONAL (the pure 7T marginal) and the source enters only as the SDEdit init.

ABLATIONS: the meaningful one-knob is ``strength`` (the SDEdit init level): the primary arm
anchors to the source (``strength<1``, preserving identity) vs the no-anchor control
(``strength=1``, ~pure noise init -> unconditional 7T generation, subject identity lost) —
this isolates the value of the source-anchored bridge. ``h_scale`` scales the score; ``h_scale=0``
zeroes the eps-net entirely, so the reverse loses its only denoiser and degenerates to a
``~1.7x`` rescale of the noised init (a saturated noise floor, NOT an un-translated source) —
it is a denoiser-OFF SANITY FLOOR, not the headline control.

Validation init noise is drawn from a FIXED seed (reproducible best-checkpoint monitoring);
training noise stays global/stochastic. Fixed-target (->7T) specialisation.
"""

from __future__ import annotations

from typing import Any

import torch

from .field_guided_diffusion_strategy import make_alphas_cumprod, q_sample
from .reconstruction import ReconstructionTrainingStrategy


def compute_doob_loss(
    model: Any, batch: dict[str, Any], *, alphas_cumprod: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Denoising score matching on the 7T target marginal (learns ``grad log h``)."""
    x0 = batch["target"]  # 7T target; the marginal whose score we learn (no source/field)
    bsz = x0.shape[0]
    timesteps = alphas_cumprod.shape[0]
    t = torch.randint(0, timesteps, (bsz,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    eps = model(x_t, timesteps=t)
    loss = torch.mean((eps - noise) ** 2)
    return {"loss_total": loss, "loss_eps": loss}


def compute_doob_residual_loss(
    model: Any, batch: dict[str, Any], *, alphas_cumprod: torch.Tensor
) -> dict[str, torch.Tensor]:
    """DSM on the source->7T RESIDUAL, conditioned on the source (learns the residual score).

    ``r0 = target - source`` is well-defined because the mrixfields source/target volumes are
    paired, registered, and normalized on the shared grid; it is small and mostly high-frequency,
    so the source-conditioned net concentrates capacity on the detail the unconditional bridge
    misses. The sampler composes ``out = source + r`` (see :func:`doob_residual_sample`).
    """
    src = batch["input"]  # source-field image; the always-on translation condition
    x0 = batch["target"] - src  # residual r0 = 7T target - source
    bsz = x0.shape[0]
    timesteps = alphas_cumprod.shape[0]
    t = torch.randint(0, timesteps, (bsz,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    eps = model(x_t, timesteps=t, cond_image=src)
    loss = torch.mean((eps - noise) ** 2)
    return {"loss_total": loss, "loss_eps": loss}


def _seeded_randn(shape: tuple[int, ...], ref: torch.Tensor, seed: int | None) -> torch.Tensor:
    """``randn`` like ``ref`` (device/dtype), from a fixed generator when ``seed`` is set."""
    if seed is None:
        return torch.randn(shape, device=ref.device, dtype=ref.dtype)
    g = torch.Generator(device=ref.device).manual_seed(int(seed))
    return torch.randn(shape, generator=g, device=ref.device, dtype=ref.dtype)


def _ilvr_lowpass(z: torch.Tensor, scale: int) -> torch.Tensor:
    """Low-pass band of ``z``: downsample by ``scale`` then upsample back (ILVR phi_N)."""
    h, w = z.shape[-2:]
    down = torch.nn.functional.interpolate(
        z,
        size=(max(1, h // scale), max(1, w // scale)),
        mode="bilinear",
        align_corners=False,
    )
    return torch.nn.functional.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def _ilvr_anchor(
    x: torch.Tensor,
    source: torch.Tensor,
    a_level: torch.Tensor,
    scale: int,
    noise_seed: int | None,
) -> torch.Tensor:
    """Swap ``x``'s low-frequency band for the (level-matched) noised source's (Choi et al. 2021).

    ``x_anchored = x - phi(x) + phi(y_t)`` with ``y_t = sqrt(a_level)*source +
    sqrt(1-a_level)*noise`` the source noised to ``x``'s current level. Pins coarse anatomy to
    the subject; the high-frequency band (contrast/detail) is left to the learned 7T score.
    """
    noise_ref = _seeded_randn(source.shape, source, noise_seed)
    y_t = a_level.sqrt() * source + (1.0 - a_level).clamp_min(0.0).sqrt() * noise_ref
    return x - _ilvr_lowpass(x, scale) + _ilvr_lowpass(y_t, scale)


def doob_h_transform_sample(
    model: Any,
    source: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    n_steps: int = 50,
    strength: float = 0.6,
    h_scale: float = 1.0,
    eta: float = 0.0,
    anchor_scale: int = 0,
    seed: int | None = None,
) -> torch.Tensor:
    """Marginal-score-guided SDEdit reverse to the 7T marginal (the B-1.10 "Doob bridge").

    SDEdit init: noise the source to time ``t0 = strength*(T-1)`` (``strength<1`` preserves
    coarse subject structure; ``strength->1`` ~ pure noise -> unconditional 7T generation).
    Reverse from ``t0`` to 0 with DDIM steps. ``h_scale`` scales the 7T-marginal score
    (``eps``): ``h_scale=1`` = the full score-based reverse (translation); ``h_scale=0`` zeroes
    the eps-net, removing the only denoiser, so the loop degenerates to a ``~sqrt(a_0/a_t0)``
    rescale of the noised init (a saturated noise FLOOR, not an un-translated source). ``eta``
    adds DDIM stochasticity (the ``g dW`` term; 0 = deterministic). ``anchor_scale>=2`` turns on
    the ILVR source structure anchor: each reverse step swaps the iterate's low-frequency band
    for the (level-matched) source's, pinning subject anatomy while the score translates the
    high-frequency band (0/1 = off, the bare bridge). When ``seed`` is set the init (and
    per-step) noise is drawn from a fixed generator -> reproducible validation. Output is clamped
    to ``[0,1]`` (the post-step iterate can leave it).
    """
    timesteps = alphas_cumprod.shape[0]
    t0 = max(1, min(timesteps - 1, int(strength * (timesteps - 1))))
    n = max(1, min(int(n_steps), t0 + 1))
    a_t0 = alphas_cumprod[t0]
    x = a_t0.sqrt() * source + (1.0 - a_t0).sqrt() * _seeded_randn(source.shape, source, seed)
    use_anchor = int(anchor_scale) >= 2  # 0/1 = off (no low-pass / trivial whole-image swap)
    idx = torch.linspace(t0, 0, n, device=source.device).round().long()
    for i in range(n):
        t = int(idx[i])
        a_t = alphas_cumprod[t]
        t_b = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = h_scale * model(x, timesteps=t_b)  # h_scale scales the 7T-marginal score
        x0_hat = ((x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-3)).clamp(-1.0, 2.0)
        if i < n - 1:
            a_next = alphas_cumprod[int(idx[i + 1])]
            sigma = (
                eta
                * ((1.0 - a_next) / (1.0 - a_t)).clamp_min(0.0).sqrt()
                * (1.0 - a_t / a_next.clamp_min(1e-8)).clamp_min(0.0).sqrt()
            )
            dir_t = (1.0 - a_next - sigma**2).clamp_min(0.0).sqrt() * eps
            step_seed = None if seed is None else int(seed) + i + 1
            x = a_next.sqrt() * x0_hat + dir_t + sigma * _seeded_randn(x.shape, x, step_seed)
            if use_anchor:  # pin x's low-freq band to the source at x's current noise level
                x = _ilvr_anchor(x, source, a_next, int(anchor_scale), step_seed)
        else:
            x = x0_hat
            if use_anchor:  # final iterate (~level 0): anchors output low-freq to the source
                a0_seed = None if seed is None else int(seed)
                x = _ilvr_anchor(x, source, alphas_cumprod[0], int(anchor_scale), a0_seed)
    return x.clamp(0.0, 1.0)


def doob_residual_sample(
    model: Any,
    source: torch.Tensor,
    *,
    alphas_cumprod: torch.Tensor,
    n_steps: int = 50,
    strength: float = 0.6,
    h_scale: float = 1.0,
    eta: float = 0.0,
    anchor_scale: int = 0,
    seed: int | None = None,
) -> torch.Tensor:
    """Source-conditioned residual reverse: sample ``r ~ p(r|source)``, output ``= source + r``.

    The reverse runs in RESIDUAL space. Init is a zero-residual SDEdit init at level
    ``t0 = strength*(T-1)`` (the residual prior mean is 0): ``x = sqrt(1-a_t0)*noise``. Each step
    the SOURCE-CONDITIONED score predicts the residual's noise; the composed output ``source +
    r0_hat`` is clamped to ``[0,1]``. ``h_scale`` scales the score (0 = denoiser-off floor).
    ``anchor_scale>=2`` applies the ILVR anchor in residual space -- the residual's low-frequency
    band is zeroed (``r0_hat -= phi_N(r0_hat)``) so the output low-frequency equals the source
    exactly while the score supplies only high-frequency detail. When ``seed`` is set the init
    (and per-step) noise is drawn from a fixed generator -> reproducible validation.
    """
    timesteps = alphas_cumprod.shape[0]
    t0 = max(1, min(timesteps - 1, int(strength * (timesteps - 1))))
    n = max(1, min(int(n_steps), t0 + 1))
    a_t0 = alphas_cumprod[t0]
    # zero-residual SDEdit init: r_t0 = sqrt(a_t0)*0 + sqrt(1-a_t0)*noise
    x = (1.0 - a_t0).sqrt() * _seeded_randn(source.shape, source, seed)
    use_anchor = int(anchor_scale) >= 2  # 0/1 = off (no residual low-freq zeroing)
    idx = torch.linspace(t0, 0, n, device=source.device).round().long()
    for i in range(n):
        t = int(idx[i])
        a_t = alphas_cumprod[t]
        t_b = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = h_scale * model(x, timesteps=t_b, cond_image=source)  # source-conditioned score
        r0_hat = ((x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-3)).clamp(-1.0, 1.0)
        if use_anchor:  # residual-space ILVR: zero the residual low-freq -> output low-freq==source
            r0_hat = r0_hat - _ilvr_lowpass(r0_hat, int(anchor_scale))
        if i < n - 1:
            a_next = alphas_cumprod[int(idx[i + 1])]
            sigma = (
                eta
                * ((1.0 - a_next) / (1.0 - a_t)).clamp_min(0.0).sqrt()
                * (1.0 - a_t / a_next.clamp_min(1e-8)).clamp_min(0.0).sqrt()
            )
            dir_t = (1.0 - a_next - sigma**2).clamp_min(0.0).sqrt() * eps
            step_seed = None if seed is None else int(seed) + i + 1
            x = a_next.sqrt() * r0_hat + dir_t + sigma * _seeded_randn(x.shape, x, step_seed)
        else:
            x = r0_hat
    return (source + x).clamp(0.0, 1.0)


class DoobBridgeStrategy(ReconstructionTrainingStrategy):
    """Train the 7T marginal score; sample the Doob h-transform bridge (B-1.10)."""

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("doob_bridge", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "doob_bridge", None)
        self._db_timesteps = int(getattr(cfg, "timesteps", 1000)) if cfg else 1000
        self._db_schedule = str(getattr(cfg, "beta_schedule", "cosine")) if cfg else "cosine"
        self._db_sampling_steps = int(getattr(cfg, "sampling_steps", 50)) if cfg else 50
        self._db_strength = float(getattr(cfg, "strength", 0.6)) if cfg else 0.6
        self._db_h_scale = float(getattr(cfg, "h_scale", 1.0)) if cfg else 1.0
        self._db_eta = float(getattr(cfg, "eta", 0.0)) if cfg else 0.0
        self._db_val_seed = int(getattr(cfg, "val_seed", 0)) if cfg else 0
        self._db_anchor_scale = int(getattr(cfg, "anchor_scale", 0)) if cfg else 0
        # Residual mode: predict the source->7T residual, source-conditioned (needs the
        # doob_residual_score_unet, which consumes cond_image); False = unconditional bridge.
        self._db_residual = bool(getattr(cfg, "residual", False)) if cfg else False
        self._acp_cache: dict[tuple, torch.Tensor] = {}

    def _alphas_cumprod(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (self._db_timesteps, self._db_schedule, str(device), str(dtype))
        acp = self._acp_cache.get(key)
        if acp is None:
            acp = make_alphas_cumprod(
                self._db_timesteps, self._db_schedule, device=device, dtype=dtype
            )
            self._acp_cache[key] = acp
        return acp

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "DoobBridgeStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        acp = self._alphas_cumprod(batch["target"].device, batch["target"].dtype)
        if self._db_residual:
            return compute_doob_residual_loss(self.env.generator, batch, alphas_cumprod=acp)
        return compute_doob_loss(self.env.generator, batch, alphas_cumprod=acp)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = sample the Doob bridge from the source (->7T marginal).

        Residual mode samples ``source + r`` via :func:`doob_residual_sample` (source-conditioned);
        the unconditional bridge uses :func:`doob_h_transform_sample`.
        """
        source = batch_context.get("input", input_batch)
        acp = self._alphas_cumprod(source.device, source.dtype)
        gen = self.env.generator
        sample_fn = doob_residual_sample if self._db_residual else doob_h_transform_sample
        kw = {
            "alphas_cumprod": acp,
            "n_steps": getattr(self, "_db_sampling_steps", 50),
            "strength": getattr(self, "_db_strength", 0.6),
            "h_scale": getattr(self, "_db_h_scale", 1.0),
            "eta": getattr(self, "_db_eta", 0.0),
            "anchor_scale": getattr(self, "_db_anchor_scale", 0),
            # FIXED seed for the SDEdit init so validation is reproducible across epochs
            # (the init noise is the only stochasticity at eta=0; an unseeded draw made the
            # best-checkpoint monitor noisy — same class as the reconstruction.py eval fix).
            "seed": getattr(self, "_db_val_seed", 0),
        }
        # The reverse loop holds no autograd graph (no_grad validation), but it is still many
        # forward passes; micro-batch via val_chunk_size to bound peak memory at full size.
        val_cfg = getattr(self.config, "validation", None)
        chunk = int((val_cfg.loader.chunk_size if val_cfg else 0) or 0)
        if chunk > 0 and source.shape[0] > chunk:
            outs = [
                sample_fn(gen, source[i : i + chunk], **kw)
                for i in range(0, source.shape[0], chunk)
            ]
            return torch.cat(outs, dim=0)
        return sample_fn(gen, source, **kw)
