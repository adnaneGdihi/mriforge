"""End-to-end smoke verification for ``mriforge.pipelines.fit.fit`` paradigms.

This is the **non-facade verifier**: it runs each paradigm's ``fit(...)`` for a
couple of real iterations on a tiny synthetic ``(input, target)`` loader with a
paradigm-appropriate model, and asserts the run completes AND the paradigm's
*distinctive* loss key actually appears in the result (a present-but-absent
distinctive loss is the silent-collapse facade the codebase polices — pitfalls
#10/#16). Adding a tailored ``fit()`` default for a new paradigm should come with
an entry here.

.. warning::

   These run a real training loop (DI container + model + loop) and are **heavy**
   — run them on the cluster, NOT on a dev box (a single fit() loop can OOM-kill
   an interactive machine). They are marked ``smoke`` so they stay out of the
   fast unit suite; invoke explicitly with ``pytest tests/smoke/test_fit_paradigms_smoke.py``.

The harness is paradigm-agnostic: each case supplies ``(paradigm, model_factory,
distinctive_loss_substr, extra_fit_kwargs)``. Reconstruction is the baseline that
proves the harness itself; the rest extend it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


class _SynthPairs(Dataset):
    """Tiny synthetic dataset yielding the canonical ``(input, target)`` mapping."""

    def __init__(self, n: int = 4, c: int = 1, h: int = 16, w: int = 16) -> None:
        self.x = torch.randn(n, c, h, w)
        self.y = torch.randn(n, c, h, w)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"input": self.x[i], "target": self.y[i]}


def _loader(batch_size: int = 2) -> DataLoader:
    return DataLoader(_SynthPairs(), batch_size=batch_size)


def _run_fit_smoke(paradigm: str, model, *, distinctive=None, **fit_kwargs) -> dict:
    """Run a tiny real fit() and return the result dict (heavy — cluster only)."""
    from mriforge.pipelines.fit import fit

    result = fit(
        model,
        _loader(),
        paradigm=paradigm,
        device="cpu",
        max_iterations=2,
        **fit_kwargs,
    )
    assert isinstance(result, dict), f"{paradigm}: fit returned {type(result)}"
    assert result.get("success") is True, f"{paradigm}: fit did not succeed — {result.get('error')}"
    if distinctive is not None:
        blob = (
            " ".join(str(k) for k in result)
            + " "
            + str(result.get("best_metrics") or {})
            + " "
            + str(result.get("final_loss") or "")
        )
        assert distinctive in blob, (
            f"{paradigm}: distinctive loss/metric {distinctive!r} not found in "
            f"result — possible facade (strategy ran but its mechanism did not "
            f"fire). result keys={sorted(result)}"
        )
    return result


# ---------------------------------------------------------------------------
# Paradigm cases. Reconstruction is the baseline (a plain conv suffices); add a
# case per tailored paradigm as its compatible model factory is confirmed.
# ---------------------------------------------------------------------------


class _TimestepConv(nn.Module):
    """A tiny model that accepts a diffusion timestep (positional or kwarg)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


def test_reconstruction_smoke():
    """Baseline: fit(reconstruction) runs a real loop on synthetic pairs."""
    _run_fit_smoke("reconstruction", nn.Conv2d(1, 1, 3, padding=1))


@pytest.mark.parametrize(
    "paradigm",
    [
        "masked",
        "cartoon_texture_safe",
        "recoverability_vib",
        "generative",
        "universal_reconstruction",
    ],
)
def test_recon_compatible_paradigms_smoke(paradigm):
    """Paradigms verified (static) to run with a standard image model on
    ``(input, target)`` pairs. The cluster run is the live non-facade check."""
    _run_fit_smoke(paradigm, nn.Conv2d(1, 1, 3, padding=1))


@pytest.mark.parametrize(
    "paradigm",
    ["diffusion", "edm", "x_diffusion", "cross_modal_diffusion", "flow_matching_pfode"],
)
def test_diffusion_family_smoke(paradigm):
    """Diffusion-family paradigms — the model must accept a timestep. The cluster
    run verifies the distinctive (velocity / score / EDM) loss actually fires."""
    _run_fit_smoke(paradigm, _TimestepConv())


def test_gan_smoke_adversarial_term_actually_fires():
    """fit(gan) must train ADVERSARIALLY, not collapse to a vanilla L1 denoiser.

    Wires the previously-dead ``distinctive=`` anti-facade hook for the most
    facade-prone paradigm: the B2 regression (``enable_adversarial`` missing from
    the fit gan default) built a discriminator but trained only L1, and this
    case is what would have caught it pre-merge — it asserts an adversarial term
    appears in the result. The substring is validated on the cluster; if the GAN
    strategy names its adversarial metric differently, adjust ``distinctive`` to
    that key (the unit guard is ``test_fit.py``'s enable_adversarial assertion).
    """
    _run_fit_smoke(
        "gan",
        nn.Conv2d(1, 1, 3, padding=1),
        discriminator=nn.Conv2d(1, 1, 3, padding=1),
        distinctive="adv",
    )
