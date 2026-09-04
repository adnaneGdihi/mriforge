r"""Unit tests for the Acquisition-Conditioned identifiable VAE (AC-iVAE).

Targets ``spectramr.models.generators.ac_ivae``.

AC-iVAE uses the continuous acquisition vector :math:`\mathbf u=\boldsymbol
\varphi` as the iVAE auxiliary variable, with a conditionally-factorial
exponential-family prior :math:`p_{\boldsymbol\lambda}(\mathbf z\mid\mathbf u)`.
By the iVAE theorem the latent is identifiable when the sufficient-statistic
differences across :math:`n_T+1` distinct values of :math:`\mathbf u` have full
column rank -- a condition the continuous acquisition manifold meets once at
least :math:`2m+1` distinct settings appear in training. The decisive
consistency check: a *single* (constant) acquisition makes the rank condition
fail, collapsing identifiability to the ordinary (unidentifiable) VAE that
CALAMITI operates in -- which is exactly why multi-acquisition data is necessary.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_variability_rank_full_under_distinct_acquisitions() -> None:
    from spectramr.models.generators.ac_ivae import (
        ACIVAEPrior,
        acquisition_variability_rank,
    )

    torch.manual_seed(0)
    m = 3
    prior = ACIVAEPrior(u_dim=5, latent_dim=m)  # n_natural = 2m = 6
    # 2m + 1 = 7 distinct acquisition settings.
    u_values = torch.randn(7, 5)
    rank = acquisition_variability_rank(prior, u_values)
    assert rank == 2 * m, f"distinct acquisitions must give full rank 2m={2 * m}, got {rank}"


def test_variability_rank_collapses_under_single_acquisition() -> None:
    """Constant u (single field/protocol) -> rank deficient -> unidentifiable (CALAMITI)."""
    from spectramr.models.generators.ac_ivae import (
        ACIVAEPrior,
        acquisition_variability_rank,
    )

    torch.manual_seed(1)
    prior = ACIVAEPrior(u_dim=5, latent_dim=3)
    u_const = torch.tensor([2.0, 1.0, 0.0, 90.0, 3.0]).expand(7, 5)
    rank = acquisition_variability_rank(prior, u_const)
    assert rank == 0, f"a single acquisition must be rank-deficient, got {rank}"


def test_forward_reconstructs_and_splits_latent() -> None:
    from spectramr.models.generators.ac_ivae import ACIVAE

    torch.manual_seed(2)
    model = ACIVAE(in_channels=1, anatomy_dim=4, contrast_dim=2, u_dim=5)
    x = torch.randn(2, 1, 16, 16)
    u = torch.randn(2, 5)
    out = model(x, u)
    assert out["reconstruction"].shape == x.shape
    assert out["z_anatomy"].shape[1] == 4
    assert out["z_contrast"].shape[1] == 2


def test_model_is_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "ac_ivae" in MODEL_REGISTRY
