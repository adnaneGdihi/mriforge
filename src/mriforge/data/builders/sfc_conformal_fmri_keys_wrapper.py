"""Dataset wrapper that fires the SFC/fMRI/MRF batch-key transforms.

Slots into :meth:`mriforge.infrastructure.builders.directors.data_pipeline_director
.DataPipelineDirector._apply_optional_wrappers` whenever any of the
``data.expose_*`` flags is set. The wrapper itself is a thin
:class:`torch.utils.data.Dataset` delegate that calls
:func:`attach_conformal_jacobian` / :func:`attach_cortex_flatten_grid` /
:func:`attach_glm_design_matrix` / :func:`attach_scanner_id` after
the inner ``__getitem__``.

Each transform no-ops when the upstream dataset already supplies the
key, so the wrapper is safe to stack on top of any base dataset.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from mriforge.data.transforms.sfc_conformal_fmri_keys import (
    attach_conformal_jacobian,
    attach_cortex_flatten_grid,
    attach_field_strength,
    attach_glm_design_matrix,
    attach_scanner_id,
    attach_site_id,
)


class SFCConformalFMRIKeysWrapper(Dataset):
    """Wrap a base dataset to fire the SFC/conformal/fMRI/MRF transforms."""

    def __init__(
        self,
        inner: Dataset,
        *,
        expose_conformal_jacobian: bool = False,
        expose_cortex_flatten_grid: bool = False,
        expose_glm_design_matrix: bool = False,
        expose_scanner_id: bool = False,
        expose_site_id: bool = False,
        expose_field_strength: bool = False,
        field_strength_default: float | None = None,
        glm_design_n_timepoints: int = 32,
    ) -> None:
        super().__init__()
        self.inner = inner
        self._expose_jac = expose_conformal_jacobian
        self._expose_grid = expose_cortex_flatten_grid
        self._expose_design = expose_glm_design_matrix
        self._expose_scanner = expose_scanner_id
        self._expose_site = expose_site_id
        self._expose_field_strength = expose_field_strength
        self._field_strength_default = field_strength_default
        self._glm_n = glm_design_n_timepoints
        # Per-instance cache for the sample-INVARIANT populated tensors (identity
        # Jacobian / flatten grid / GLM regressor). These depend only on
        # shape/config and are read-only downstream, so caching them by shape and
        # sharing across samples avoids rebuilding them every __getitem__.
        # backlog_wasted_compute_audit_2026_05_29 SFC-1/2/3.
        self._key_cache: dict = {}

    def __len__(self) -> int:
        return len(self.inner)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.inner[idx]
        if not isinstance(sample, dict):
            # Fall back: skip wrapping non-dict samples (some legacy
            # datasets return tuples). The audit warns when the strategy
            # needs the key — there's nothing for the wrapper to do
            # here without a dict payload.
            return sample
        if self._expose_jac:
            attach_conformal_jacobian(sample, cache=self._key_cache)
        if self._expose_grid:
            attach_cortex_flatten_grid(sample, cache=self._key_cache)
        if self._expose_design:
            attach_glm_design_matrix(sample, n_timepoints=self._glm_n, cache=self._key_cache)
        if self._expose_scanner:
            attach_scanner_id(sample)
        if self._expose_site:
            attach_site_id(sample)
        if self._expose_field_strength:
            attach_field_strength(sample, default=self._field_strength_default)
        return sample


__all__ = ["SFCConformalFMRIKeysWrapper"]
