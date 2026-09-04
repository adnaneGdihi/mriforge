"""Batch-key transforms that populate the metadata the SFC/conformal
+ fMRI/MRF strategies look for.

These are pure-function transforms invoked on the collated batch dict
(post-dataloader), mirroring the convention of
:mod:`spectramr.data.transforms.acquisition_params_injector`. Each transform
no-ops only when there is no reference tensor to size from (no
``image`` / ``input`` / ``target``). When a reference IS present the key is
always populated, because the consuming strategies are fail-loud: e.g.
:class:`ConformalDiffusionReconStrategy` **raises** without
``conformal_jacobian`` rather than silently degrading to plain MSE
(pitfall #16). Wire the populators into the pipeline for any arm that sets
the corresponding ``data.expose_*`` flag.

Provides:

* :func:`attach_conformal_jacobian` — fills ``batch["conformal_jacobian"]``
  with ``|Φ'(z)|`` for the disk-canonical sampling region.
* :func:`attach_cortex_flatten_grid` — fills
  ``batch["cortex_flatten_grid"]`` with a precomputed grid mapping
  3-D cortical voxels to the disk.
* :func:`attach_glm_design_matrix` — fills ``batch["design_matrix"]``
  for HRF-coupled losses.
* :func:`attach_scanner_id` — fills ``batch["scanner_id"]`` for
  cross-vendor harmonisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import torch

_SCANNER_ID_MODULUS = 1000


def _stable_id(label: str) -> int:
    """Map a vendor / site string to a stable integer in ``[0, 1000)``.

    ``hash()`` is salted per interpreter (``PYTHONHASHSEED``), so under the
    cluster's ``forkserver`` start method every DataLoader worker is a fresh
    process with a different salt: the *same* ``"Siemens"`` string would map to
    a *different* integer in each worker and each run, silently corrupting
    cross-vendor / multi-site conditioning. A cryptographic digest is stable
    across processes and runs, which is the invariant these harmonisation keys
    require.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SCANNER_ID_MODULUS


def _resolve_batch_size(batch: Mapping[str, Any]) -> int:
    for k in ("image", "input", "target", "kspace"):
        v = batch.get(k)
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return 1


def _resolve_ref_tensor(batch: Mapping[str, Any]) -> torch.Tensor | None:
    """Return the first present reference tensor to size populated keys from.

    Prefers ``image`` (the fMRI / torchio-builder convention), then falls back
    to ``input`` / ``target`` — the canonical keys every reconstruction dataset
    (m4raw / bart / SliceDataset) actually emits. Gating only on ``image``
    silently left ``conformal_jacobian`` / ``cortex_flatten_grid`` /
    ``design_matrix`` unattached on recon batches, so the conformal-diffusion
    arm degraded to plain MSE while smoke-PASSing (pitfall #16).
    """
    for k in ("image", "input", "target"):
        v = batch.get(k)
        if isinstance(v, torch.Tensor):
            return v
    return None


def _cached_or_build(
    cache: dict | None,
    key: tuple,
    builder: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Return ``cache[key]``, building it via ``builder()`` on a miss.

    The tensors these populators produce (identity Jacobian, identity flatten
    grid, constant GLM regressor) are **sample-invariant** — they depend only on
    shape/config, never on the pixel data — and are only ever READ downstream
    (the collate step copies them via ``torch.stack``). So caching one instance
    per shape and sharing it across samples is safe and avoids rebuilding the
    linspace/meshgrid/ones every call.
    See backlog_wasted_compute_audit_2026_05_29 SFC-1/2/3.
    """
    if cache is None:
        return builder()
    cached = cache.get(key)
    if cached is None:
        cached = builder()
        cache[key] = cached
    return cached


def attach_conformal_jacobian(
    batch: dict[str, Any],
    *,
    height: int | None = None,
    width: int | None = None,
    radial: bool = True,
    cache: dict | None = None,
) -> dict[str, Any]:
    """Populate ``batch['conformal_jacobian']`` for ConformalDataConsistency.

    Requires a real Jacobian under ``batch['_conformal_jacobian_override']``.
    Raises otherwise (audit C13).

    This used to fabricate ``|Φ'(z)| ≡ 1`` — an identity Jacobian, which its own
    docstring described as "equivalent to the strategy's fallback". That is
    worse than it sounds, because the consumer does not HAVE a silent fallback:
    ``ConformalDiffusionReconStrategy`` deliberately raises when
    ``conformal_jacobian`` is missing, with a comment reading "Fail loud, do NOT
    warn-and-run-plain-MSE ... the 'conformal' paradigm is inert (pitfall #16 —
    the run smoke-PASSes while measuring nothing)".

    An all-ones tensor SATISFIES that ``isinstance`` check. So the data layer
    silently defeated a guard the strategy author built on purpose: the
    projection ran, weighted by 1, numerically identical to the plain path the
    raise exists to prevent — and the wrapper logged that the mechanism fired.

    Raising here restores the author's intent. The documented override is the
    way to supply a real Jacobian; it currently has no producer anywhere, which
    is the honest state of the mechanism.

    Raises:
        ValueError: no ``_conformal_jacobian_override`` was supplied.
    """
    # No-op when the upstream dataset already supplied a real Jacobian. The
    # wrapper's docstring has always promised this ("Each transform no-ops when
    # the upstream dataset already supplies the key"), and no code implemented
    # it — the old body went straight to fabricating an identity, overwriting
    # whatever was there.
    existing = batch.get("conformal_jacobian")
    if isinstance(existing, torch.Tensor):
        return batch

    override = batch.get("_conformal_jacobian_override")
    if isinstance(override, torch.Tensor):
        batch["conformal_jacobian"] = override
        return batch

    raise ValueError(
        "attach_conformal_jacobian has no real Jacobian to attach. It used to "
        "fabricate an identity (|phi'(z)| = 1), which weights the conformal "
        "data-consistency projection by 1 and is therefore numerically the "
        "plain path — while satisfying the isinstance check that "
        "ConformalDiffusionReconStrategy raises on precisely to stop that "
        "(pitfall #16). Supply a precomputed Jacobian for the arm's trajectory "
        "under batch['_conformal_jacobian_override'] (no producer exists yet), "
        "or set data.expose.conformal_jacobian: false so the strategy's own "
        "fail-loud guard reports the gap instead."
    )


def attach_cortex_flatten_grid(
    batch: dict[str, Any],
    *,
    grid_height: int = 64,
    grid_width: int = 64,
    cache: dict | None = None,
) -> dict[str, Any]:
    """Populate ``batch['cortex_flatten_grid']`` for the cortical
    conformal strategy. Default is an identity grid in normalised
    coordinates — caller's precomputed Gu-Yau flattening overrides via
    ``batch['_cortex_flatten_grid_override']``.
    """
    override = batch.get("_cortex_flatten_grid_override")
    if isinstance(override, torch.Tensor):
        batch["cortex_flatten_grid"] = override
        return batch
    ref = _resolve_ref_tensor(batch)
    if ref is None:
        return batch

    def _build_grid() -> torch.Tensor:
        ys = torch.linspace(-1, 1, grid_height, device=ref.device, dtype=ref.dtype)
        xs = torch.linspace(-1, 1, grid_width, device=ref.device, dtype=ref.dtype)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([X, Y], dim=-1)
        return grid.unsqueeze(0).expand(ref.shape[0], -1, -1, -1).clone()

    batch["cortex_flatten_grid"] = _cached_or_build(
        cache,
        ("grid", ref.shape[0], grid_height, grid_width, ref.device, ref.dtype),
        _build_grid,
    )
    return batch


def attach_glm_design_matrix(
    batch: dict[str, Any],
    *,
    n_timepoints: int = 32,
    stimulus_onsets: list[int] | None = None,
    cache: dict | None = None,
) -> dict[str, Any]:
    """Populate ``batch['design_matrix']`` for HRF-coupled training.

    Default builds a simple square-wave stimulus regressor (every
    eighth TR active) per batch sample. Caller overrides via
    ``batch['_design_matrix_override']``.
    """
    override = batch.get("_design_matrix_override")
    if isinstance(override, torch.Tensor):
        batch["design_matrix"] = override
        return batch
    ref = _resolve_ref_tensor(batch)
    if ref is None:
        return batch
    B = ref.shape[0]
    onsets = stimulus_onsets or list(range(0, n_timepoints, 8))

    def _build_design() -> torch.Tensor:
        stim = torch.zeros(n_timepoints, device=ref.device, dtype=ref.dtype)
        for o in onsets:
            if 0 <= o < n_timepoints:
                stim[o] = 1.0
        return stim.unsqueeze(0).expand(B, n_timepoints).clone()

    batch["design_matrix"] = _cached_or_build(
        cache,
        ("design", B, n_timepoints, tuple(onsets), ref.device, ref.dtype),
        _build_design,
    )
    return batch


def attach_scanner_id(
    batch: dict[str, Any],
    *,
    default_id: int = 0,
) -> dict[str, Any]:
    """Populate ``batch['scanner_id']`` for cross-vendor harmonisation.

    When the dataloader already supplies a ``scanner`` or ``vendor``
    string per sample, it is mapped to a stable integer hash; otherwise
    every sample is tagged with ``default_id``.
    """
    if isinstance(batch.get("scanner_id"), torch.Tensor):
        return batch
    B = _resolve_batch_size(batch)
    vendors = batch.get("scanner") or batch.get("vendor")
    if isinstance(vendors, (list, tuple)) and len(vendors) == B:
        ids = torch.tensor([_stable_id(str(v)) for v in vendors], dtype=torch.long)
    else:
        ids = torch.full((B,), default_id, dtype=torch.long)
    batch["scanner_id"] = ids
    return batch


def attach_site_id(
    batch: dict[str, Any],
    *,
    default_id: int = 0,
) -> dict[str, Any]:
    """Populate ``batch['site_id']`` for multi-site / federated conditioning.

    When the dataloader supplies a ``site`` (or ``site_id``) label per sample it
    is mapped to a stable integer hash; otherwise every sample is tagged with
    ``default_id``. Mirrors :func:`attach_scanner_id`. Paired with
    ``data.expose_site_id``; consumed by ``ConditioningMixin`` and site-aware
    models (e.g. ``siren_universal``).
    """
    if isinstance(batch.get("site_id"), torch.Tensor):
        return batch
    B = _resolve_batch_size(batch)
    sites = batch.get("site") or batch.get("site_id")
    if isinstance(sites, (list, tuple)) and len(sites) == B:
        ids = torch.tensor([_stable_id(str(s)) for s in sites], dtype=torch.long)
    else:
        ids = torch.full((B,), default_id, dtype=torch.long)
    batch["site_id"] = ids
    return batch


def attach_field_strength(
    batch: dict[str, Any],
    *,
    default: float | None = None,
) -> dict[str, Any]:
    """Populate ``batch['field_strength']`` (Tesla) for B0 conditioning.

    Uses a per-sample ``field_strength`` from the dataloader when present
    (multi-field datasets); otherwise fills every sample with ``default``
    (typically ``physics.field_strength``). No-op when neither is available, so
    a model that needs it surfaces the absence rather than silently training on a
    fabricated value. Paired with ``data.expose_field_strength``.
    """
    if isinstance(batch.get("field_strength"), torch.Tensor):
        return batch
    B = _resolve_batch_size(batch)
    fs = batch.get("field_strength")
    if isinstance(fs, (list, tuple)) and len(fs) == B:
        vals = torch.tensor([float(v) for v in fs], dtype=torch.float32)
    elif isinstance(fs, (int, float)):
        vals = torch.full((B,), float(fs), dtype=torch.float32)
    elif default is not None:
        vals = torch.full((B,), float(default), dtype=torch.float32)
    else:
        return batch
    batch["field_strength"] = vals
    return batch


__all__ = [
    "attach_conformal_jacobian",
    "attach_cortex_flatten_grid",
    "attach_field_strength",
    "attach_glm_design_matrix",
    "attach_scanner_id",
    "attach_site_id",
]
