r"""Unit tests for the Field-Pushforward Score Prior (FPSP).

Targets ``spectramr.models.generators.fpsp_score``.

FPSP trains a single score network :math:`s_\theta(\boldsymbol\xi,t)` on
field-independent parameter maps; for any field/protocol the image prior is the
pushforward through the deterministic signal map :math:`I=\mathcal
A(\boldsymbol\xi;\boldsymbol\varphi,B_0)`,

.. math::

   p_{\boldsymbol\varphi,B_0}(I)
     = \bigl(\mathcal A_{\boldsymbol\varphi,B_0}\bigr)_{\#}\,p(\boldsymbol\xi).

The field dependence of the image prior is carried entirely by the known
physics, not by the learned score -- so the score is field-invariant by
construction (its forward takes no field/acquisition argument) and one trained
prior serves every regime.
"""

from __future__ import annotations

import inspect

import pytest
import torch

pytestmark = pytest.mark.unit


def _params(h: int = 8, w: int = 8) -> torch.Tensor:
    rho = torch.full((2, 1, h, w), 0.8)
    t1 = torch.full((2, 1, h, w), 900.0)
    t2 = torch.full((2, 1, h, w), 90.0)
    return torch.cat([rho, t1, t2], dim=1)


def test_score_net_operates_in_parameter_space() -> None:
    from spectramr.models.generators.fpsp_score import FieldPushforwardScorePrior

    torch.manual_seed(0)
    model = FieldPushforwardScorePrior(param_channels=3, width=16)
    params = _params()
    t = torch.tensor([0.3, 0.7])
    score = model(params, t)
    assert score.shape == params.shape


def test_score_is_field_invariant_by_construction() -> None:
    """The score's forward must NOT take a field / acquisition argument."""
    from spectramr.models.generators.fpsp_score import FieldPushforwardScorePrior

    sig = inspect.signature(FieldPushforwardScorePrior.forward)
    names = set(sig.parameters)
    assert "acquisition" not in names and "b0" not in names and "field" not in names


def test_pushforward_field_dependence_is_carried_by_physics() -> None:
    from spectramr.models.generators.fpsp_score import field_pushforward

    params = _params()
    img_t2 = field_pushforward(params, {"TR": 3000.0, "TE": 90.0, "TI": 0.0}, "spin_echo")
    img_flair = field_pushforward(
        params, {"TR": 9000.0, "TE": 90.0, "TI": 2500.0}, "flair"
    )
    assert img_t2.shape == (2, 1, 8, 8)
    assert not torch.allclose(img_t2, img_flair), "same params, different protocol"


def test_pushforward_is_deterministic() -> None:
    from spectramr.models.generators.fpsp_score import field_pushforward

    params = _params()
    acq = {"TR": 3000.0, "TE": 90.0, "TI": 0.0}
    a = field_pushforward(params, acq, "spin_echo")
    b = field_pushforward(params, acq, "spin_echo")
    assert torch.allclose(a, b)


def test_model_is_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "fpsp_score" in MODEL_REGISTRY
