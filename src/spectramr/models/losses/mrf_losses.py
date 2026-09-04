r"""Magnetic-resonance-fingerprinting losses (regime: ``mri_fingerprinting``).

MRF's defining operation is **dictionary matching** (Ma et al., *Nature*
495:187-192, 2013): a transient-state acquisition gives every voxel a signal
*fingerprint*, and the tissue parameters are recovered by finding the dictionary
entry — a Bloch trajectory simulated for a known ``(T1, T2, ...)`` — whose
normalised fingerprint has the largest inner product with the measured one.

That operation is what this module puts into the objective, and it is why this is
the loss that backs ``mri_fingerprinting`` in the maturity ledger. The regime had
**no honest loss at all** before it. Several MRF-*named* losses exist and none of
them contains any fingerprinting physics:

* ``tropical_semiring_consistency`` (aliases ``tropical_mrf_consistency``,
  ``tropical_quantitative_consistency``) builds its reference tropical fan with
  ``torch.randn(..., generator=seeded)`` — a *random* reference. It is a generic
  max-plus penalty on any ``[B, C, H, W]``.
* ``conformality_jacobian`` ("MRF §3") is differential geometry on an arbitrary
  Jacobian — no Bloch, no dictionary, no fingerprint.

Tagging either would have made the ledger machine-verify a claim with no physics
behind it, which is exactly the failure the tagging seam exists to prevent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.losses.registry import register_loss


def soft_dictionary_match(
    fingerprints: torch.Tensor,
    dictionary: torch.Tensor,
    dictionary_params: torch.Tensor,
    *,
    temperature: float = 0.01,
) -> torch.Tensor:
    r"""Differentiably match ``fingerprints`` against ``dictionary``.

    The MRF match is ``argmax_d |<f, D_d>| / (||f|| ||D_d||)``, which is not
    differentiable. This is the standard softmax relaxation: similarities are
    turned into weights with a temperature and used to take a convex combination
    of the dictionary's parameters. As ``temperature -> 0`` the weights approach
    the one-hot hard match, recovering the textbook operation; larger values
    blur across neighbouring entries and make the gradient informative.

    The inner product is taken as a **magnitude** so the match is insensitive to
    the fingerprint's global phase — the convention in MRF, where the receive
    phase is arbitrary and carries no tissue information.

    Args:
        fingerprints: ``[V, N]`` measured fingerprints (real or complex).
        dictionary: ``[D, N]`` simulated Bloch trajectories, same dtype domain.
        dictionary_params: ``[D, P]`` parameters paired to ``dictionary`` rows.
        temperature: softmax temperature over the similarities. Must be > 0.

    Returns:
        ``[V, P]`` soft-matched parameters.

    Raises:
        ValueError: on a non-2-D input, a fingerprint-length mismatch between
            query and dictionary, a dictionary/parameter row-count mismatch, or a
            non-positive temperature.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0; got {temperature}. Zero would be the hard "
            "argmax, which has no gradient — use a small positive value."
        )
    for label, t in (
        ("fingerprints", fingerprints),
        ("dictionary", dictionary),
        ("dictionary_params", dictionary_params),
    ):
        if t.ndim != 2:
            raise ValueError(f"{label} must be 2-D; got {tuple(t.shape)}.")
    if fingerprints.shape[1] != dictionary.shape[1]:
        raise ValueError(
            f"fingerprint length {fingerprints.shape[1]} does not match the "
            f"dictionary's {dictionary.shape[1]}. The inner product would be "
            "meaningless even where it is computable."
        )
    if dictionary.shape[0] != dictionary_params.shape[0]:
        raise ValueError(
            f"dictionary has {dictionary.shape[0]} entries but "
            f"dictionary_params has {dictionary_params.shape[0]} rows — the "
            "matched parameters would come from the wrong tissue."
        )

    # Normalise so the match is on trajectory SHAPE, not amplitude: proton
    # density scales a fingerprint and is not what the dictionary indexes.
    f = fingerprints / fingerprints.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = dictionary / dictionary.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if f.is_complex() or d.is_complex():
        # Promote to the widest of the two rather than hardcoding complex64
        # (#346): a complex128 dictionary used to be silently downcast, so the
        # complex path lost precision exactly where the real path below keeps
        # it (float64 stays float64). MRF fingerprints are genuinely complex --
        # mrf_dictionary.py encodes off-resonance as a per-echo phase -- and a
        # float64 dictionary is what someone reaches for when chasing a
        # matching discrepancy, i.e. precisely when the downcast would mislead.
        dtype = torch.promote_types(f.dtype, d.dtype)
        similarity = (f.to(dtype) @ d.to(dtype).conj().T).abs()
    else:
        similarity = (f @ d.T).abs()

    weights = torch.softmax(similarity / temperature, dim=1)  # [V, D]
    return weights @ dictionary_params.to(weights.dtype)  # [V, P]


@register_loss(
    name="mrf_dictionary_match",
    aliases=["MRFDictionaryMatchLoss", "dictionary_match_consistency"],
    domain="image",
    workflows=frozenset({Regime.FINGERPRINTING}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class MRFDictionaryMatchLoss(nn.Module):
    r"""Penalise predicted MRF parameters that disagree with the dictionary match.

    The network predicts ``(T1, T2, ...)`` directly from a fingerprint; this loss
    pushes that prediction toward what dictionary matching says the same
    fingerprint means. It is **self-supervised** in the same sense as
    ``tofts_residual``: the supervision comes from the physics (the dictionary),
    not from ground-truth parameter maps, which real MRF acquisitions do not have.

    Args:
        weight: overall multiplier, ``>= 0``.
        temperature: softmax temperature for :func:`soft_dictionary_match`.

    ``forward(pred_params, fingerprints, dictionary, dictionary_params)``:
        - ``pred_params``: ``[V, P]`` predicted parameters.
        - ``fingerprints``: ``[V, N]`` measured fingerprints.
        - ``dictionary``: ``[D, N]`` simulated Bloch trajectories.
        - ``dictionary_params``: ``[D, P]`` parameters for each entry.

    Every argument is required and every degenerate path raises. There is no
    "no dictionary supplied, fall back to L1" branch by design: a fingerprinting
    loss without a dictionary is not a degraded fingerprinting loss, it is a
    different loss wearing the name (pitfall #16).

    **Scope, stated plainly.** This grades the prediction against the dictionary,
    so it is exactly as good as the dictionary it is handed. ``BlochDictionary``
    in ``infrastructure/physics/mrf_dictionary.py`` is built on a *steady-state*
    (Ernst) simulator and documents itself as not a transient MRF simulator — so
    with that dictionary this is a steady-state approximation of MRF matching.
    Passing a transient dictionary (e.g. from ``epg.simulate_differentiable_epg_fse``)
    makes it the full operation. The loss does not and cannot check which it got.
    """

    def __init__(self, weight: float = 1.0, temperature: float = 0.01) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be >= 0; got {weight}")
        self.weight = float(weight)
        self.temperature = float(temperature)

    def forward(
        self,
        pred_params: torch.Tensor,
        fingerprints: torch.Tensor,
        dictionary: torch.Tensor,
        dictionary_params: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        matched = soft_dictionary_match(
            fingerprints,
            dictionary,
            dictionary_params,
            temperature=self.temperature,
        )
        if pred_params.shape != matched.shape:
            raise ValueError(
                f"pred_params {tuple(pred_params.shape)} does not match the "
                f"dictionary-matched parameters {tuple(matched.shape)}. Left to "
                "broadcast, this would compare every voxel's prediction against "
                "every other voxel's match and still return an ordinary number."
            )
        return self.weight * ((pred_params - matched) ** 2).mean()


__all__ = ["MRFDictionaryMatchLoss", "soft_dictionary_match"]
