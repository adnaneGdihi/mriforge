r"""Classifier guidance for diffusion sampling (Dhariwal & Nichol 2021).

Classifier guidance shifts the unconditional score
:math:`\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)` by the
gradient of a *noisy* classifier
:math:`\nabla_{\mathbf{x}_t}\log p_\phi(\mathbf{c}\mid\mathbf{x}_t)`:

.. math::
    \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid\mathbf{c})
    = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)
    + s\,\nabla_{\mathbf{x}_t}\log p_\phi(\mathbf{c}\mid\mathbf{x}_t).

In DDIM / DDPM noise-prediction parameterisation the equivalent
correction on the predicted noise is

.. math::
    \tilde{\boldsymbol{\epsilon}}(\mathbf{x}_t,t)
    = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)
    - \sqrt{1-\bar\alpha_t}\,s\,\nabla_{\mathbf{x}_t}\log p_\phi(\mathbf{c}\mid\mathbf{x}_t).

This module is composable: it consumes a noise prediction and emits
a guided noise prediction. Strategies wire it into their reverse-step
loop alongside (or in place of) classifier-free guidance and the
existing physics-guided / DPS samplers.

Reference:
    [58] P. Dhariwal, A. Nichol, "Diffusion models beat GANs on image
        synthesis," *Proc. NeurIPS*, 2021.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierGuidanceSampler(nn.Module):
    r"""DDIM/DDPM classifier-guidance corrector.

    Designed to compose with the framework's existing diffusion
    sampler infrastructure (`DDIMSampler`, `PhysicsGuidedReverseSampler`,
    `MCGSampler`, `DPSMRISampler`). It does **not** itself drive the
    reverse-step loop — it consumes the network's noise prediction
    and an optional class label and returns a corrected noise
    prediction.

    Args:
        classifier: A frozen classifier callable
            ``classifier(x_t, t) -> logits`` returning ``(B, n_classes)``.
            Strategies are responsible for keeping it in eval mode
            and on the right device.
        guidance_scale: Strength :math:`s`. ``0`` disables guidance,
            ``1`` is the original Dhariwal-Nichol setting; larger
            values trade off sample quality for label fidelity.
        loss_fn: How to convert classifier logits + target into a scalar
            log-likelihood. Defaults to negative cross-entropy
            (i.e. log-softmax of the target class).
        clip_grad_norm: If positive, clip the per-batch gradient norm
            before adding it to the noise prediction — stabilises
            sampling at high guidance scales (Dhariwal-Nichol §C.2).
    """

    def __init__(
        self,
        classifier: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        guidance_scale: float = 1.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        clip_grad_norm: float = 0.0,
    ) -> None:
        super().__init__()
        if guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")
        if not callable(classifier):
            raise TypeError("classifier must be callable.")
        if clip_grad_norm < 0:
            raise ValueError(f"clip_grad_norm must be >= 0, got {clip_grad_norm}")
        self.classifier = classifier
        self.guidance_scale = guidance_scale
        self.loss_fn = loss_fn if loss_fn is not None else self._default_log_likelihood
        self.clip_grad_norm = clip_grad_norm

    @staticmethod
    def _default_log_likelihood(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative cross-entropy = log-softmax at the target class.

        ``target`` may be either ``(B,)`` long indices or ``(B, n_classes)``
        soft probabilities.
        """
        if target.dim() == logits.dim() and target.dtype.is_floating_point:
            log_probs = F.log_softmax(logits, dim=-1)
            return (target * log_probs).sum(dim=-1).sum()
        return -F.cross_entropy(logits, target.long(), reduction="sum")

    def classifier_score(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`\\nabla_{\\mathbf{x}_t}\\log p_\\phi(\\mathbf{c}\\mid\\mathbf{x}_t)`.

        Returns a tensor with the same shape as ``x_t``.
        """
        # We need autograd through the (frozen-weight) classifier. Make
        # x_t a leaf with grad enabled, but do NOT propagate gradients
        # to the classifier's parameters.
        x_t = x_t.detach().requires_grad_(True)
        was_training = getattr(self.classifier, "training", False)
        if hasattr(self.classifier, "eval"):
            self.classifier.eval()  # type: ignore[union-attr]
        with torch.enable_grad():
            logits = self.classifier(x_t, t)
            log_p = self.loss_fn(logits, target)
            grad = torch.autograd.grad(log_p, x_t)[0]
        if hasattr(self.classifier, "train") and was_training:
            self.classifier.train()  # type: ignore[union-attr]
        if self.clip_grad_norm > 0:
            flat = grad.reshape(grad.shape[0], -1)
            norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = (norm.clamp_max(self.clip_grad_norm) / norm).view(-1, *([1] * (grad.dim() - 1)))
            grad = grad * scale
        return grad

    def correct_noise(
        self,
        noise_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        r"""Apply the classifier-guidance correction to a noise
        prediction.

        Args:
            noise_pred: :math:`\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)`
                of shape matching ``x_t``.
            x_t: Current diffusion state.
            t: Time / step index — passed through to the classifier.
            target: Conditioning class label(s).
            sqrt_one_minus_alpha_bar: Per-sample noise scale
                :math:`\sqrt{1-\bar\alpha_t}` of shape ``(B,)`` or
                broadcastable to ``noise_pred``.
        """
        if noise_pred.shape != x_t.shape:
            raise ValueError(
                f"noise_pred shape {tuple(noise_pred.shape)} != x_t shape {tuple(x_t.shape)}"
            )
        if self.guidance_scale == 0:
            return noise_pred
        grad = self.classifier_score(x_t, t, target)
        scale = sqrt_one_minus_alpha_bar
        if scale.dim() == 1:
            scale = scale.view(-1, *([1] * (noise_pred.dim() - 1)))
        return noise_pred - scale * self.guidance_scale * grad

    def correct_score(
        self,
        score_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Apply the classifier-guidance correction directly to a score
        (gradient of log-density), for SDE-style samplers.

        :math:`\tilde{s}(\mathbf{x}_t,t) = s_\theta(\mathbf{x}_t,t)
        + s\,\nabla_{\mathbf{x}_t}\log p_\phi(\mathbf{c}\mid\mathbf{x}_t)`.
        """
        if score_pred.shape != x_t.shape:
            raise ValueError(
                f"score_pred shape {tuple(score_pred.shape)} != x_t shape {tuple(x_t.shape)}"
            )
        if self.guidance_scale == 0:
            return score_pred
        grad = self.classifier_score(x_t, t, target)
        return score_pred + self.guidance_scale * grad

    def forward(
        self,
        noise_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """Default forward pathway = ``correct_noise``.

        SDE samplers should call ``correct_score`` directly instead.
        """
        return self.correct_noise(
            noise_pred=noise_pred,
            x_t=x_t,
            t=t,
            target=target,
            sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar,
        )
