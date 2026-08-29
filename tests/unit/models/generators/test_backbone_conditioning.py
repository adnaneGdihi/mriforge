"""Conditioning contract for the alternative kspace_filling backbones.

`swin_diff_rec`, `diff_varnet` and `nafnet` all advertised timestep and contrast
conditioning that did not reach the network:

* both VarNet-family backbones encoded ``t`` with ``nn.Linear(1, ·)`` on the raw
  scalar — a rank-1 map, so "which t" was not representable;
* none of the three read ``contrast_emb``, so every arm declaring
  ``num_contrasts`` trained without contrast conditioning;
* ``nafnet`` additionally dropped the embedding entirely on the complex path,
  because its local ``TimeAwareSequential`` gated delivery on
  ``isinstance(module, NAFBlock)`` and ``ComplexNAFBlock`` is not a subclass.

These tests assert the mechanism FIRES, which is the part that was missing —
`test_kspace_cold_diffusion_generator.py` already covers construction.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.diff_varnet import DiffVarNet
from mriforge.models.generators.nafnet_generator import NAFNetGenerator

EMB = 256


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / max(a.norm().item(), 1e-12))


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 8, 32, 32) * 0.1


class TestDiffVarNetConditioning:
    def _model(self) -> DiffVarNet:
        torch.manual_seed(0)
        return DiffVarNet(
            in_channels=8, out_channels=8, image_size=32, base_channels=32
        ).eval()

    def test_timestep_changes_output(self, x):
        m = self._model()
        with torch.no_grad():
            y0 = m(x, torch.zeros(1), max_timesteps=28.0)
            y1 = m(x, torch.full((1,), 20.0), max_timesteps=28.0)
        assert _rel(y0, y1) > 1e-4, "timestep conditioning does not reach the output"

    def test_contrast_changes_output(self, x):
        m = self._model()
        with torch.no_grad():
            a = m(x, torch.zeros(1), max_timesteps=28.0, contrast_emb=torch.zeros(1, EMB))
            b = m(
                x,
                torch.zeros(1),
                max_timesteps=28.0,
                contrast_emb=torch.randn(1, EMB) * 0.5,
            )
        assert _rel(a, b) > 1e-4, "contrast_emb is being swallowed"

    def test_contrast_width_mismatch_raises(self, x):
        """A silent broadcast on mismatched widths would be the next facade."""
        m = self._model()
        with pytest.raises(ValueError, match="contrast_emb width"):
            m(x, torch.zeros(1), max_timesteps=28.0, contrast_emb=torch.zeros(1, 64))

    def test_max_timesteps_is_read(self, x):
        """The generator forwards max_timesteps; it used to be swallowed here."""
        m = self._model()
        with torch.no_grad():
            a = m(x, torch.full((1,), 14.0), max_timesteps=28.0)
            b = m(x, torch.full((1,), 14.0), max_timesteps=1000.0)
        assert _rel(a, b) > 1e-5, "max_timesteps has no effect on the encoding"


class TestNAFNetConditioning:
    """The complex path is what `experiment_11e_nafnet` builds."""

    def _model(self, perturb: bool) -> NAFNetGenerator:
        torch.manual_seed(0)
        m = NAFNetGenerator(
            in_channels=8, out_channels=8, width=64, use_complex_conv=True
        ).eval()
        if perturb:
            # beta/gamma are ZERO-initialised (standard NAFNet residual init) and
            # emb enters only through the branch they scale, so nothing moves at
            # init by design. Perturb to observe the pathway itself.
            for mod in m.modules():
                if hasattr(mod, "beta") and hasattr(mod, "gamma"):
                    with torch.no_grad():
                        mod.beta.fill_(0.1)
                        mod.gamma.fill_(0.1)
        return m

    def test_embedding_reaches_the_complex_block(self, x):
        """Regression for the isinstance gate: ComplexNAFBlock must RECEIVE emb.
        Asserted at the block boundary because the zero-init residual scalers
        make an output-level assertion vacuous at init."""
        m = self._model(perturb=False)
        blk = m.middle_blks[0]
        seen: dict[str, bool] = {}
        original = type(blk).forward

        def spy(self, inp, emb=None):
            seen["got_emb"] = emb is not None
            return original(self, inp, emb)

        type(blk).forward = spy
        try:
            with torch.no_grad():
                m(x, torch.zeros(1), contrast_emb=torch.randn(1, EMB))
        finally:
            type(blk).forward = original
        assert seen.get("got_emb"), "ComplexNAFBlock received emb=None"

    def test_timestep_changes_output_once_residuals_are_nonzero(self, x):
        m = self._model(perturb=True)
        with torch.no_grad():
            a = m(x, torch.zeros(1))
            b = m(x, torch.full((1,), 20.0))
        assert _rel(a, b) > 1e-5

    def test_contrast_changes_output_once_residuals_are_nonzero(self, x):
        m = self._model(perturb=True)
        with torch.no_grad():
            a = m(x, torch.zeros(1), contrast_emb=torch.zeros(1, EMB))
            b = m(x, torch.zeros(1), contrast_emb=torch.randn(1, EMB) * 0.5)
        assert _rel(a, b) > 1e-5

    def test_zero_init_means_no_effect_at_init(self, x):
        """Pins the documented behaviour so a future reader does not 'fix' it:
        at init the residual scalers are zero, so conditioning is inert BY
        DESIGN. This is not the isinstance bug above."""
        m = self._model(perturb=False)
        with torch.no_grad():
            a = m(x, torch.zeros(1), contrast_emb=torch.zeros(1, EMB))
            b = m(x, torch.zeros(1), contrast_emb=torch.randn(1, EMB) * 0.5)
        assert _rel(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_contrast_width_mismatch_raises(self, x):
        m = self._model(perturb=False)
        with pytest.raises(ValueError, match="contrast_emb width"):
            m(x, torch.zeros(1), contrast_emb=torch.zeros(1, 64))


class TestContrastWidthProjection:
    """NAFNet conditions at ``width * 4``; the generator builds
    ``contrast_embedding`` at ``time_embedding_dim``. Those genuinely disagree on
    ``experiment_11e_nafnet`` (48*4 = 192 vs 256), so the backbone must PROJECT
    rather than raise — a raise here made the arm unbuildable at forward time,
    which unit tests missed because they construct NAFNet directly with matching
    widths. This is an end-to-end contract, so it is asserted through the real
    generator.
    """

    def test_arm_with_mismatched_widths_runs_contrast_conditioned_forward(self):
        import yaml

        from mriforge.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        path = "experiments/inprogress/kspace_filling/experiment_11e_nafnet.yaml"
        model_cfg = yaml.safe_load(open(path))["model"]
        kwargs = model_cfg["model_kwargs"]
        # The mismatch this guards: NAFNet width*4 vs contrast_embedding width.
        assert kwargs["base_channels"] * 4 != kwargs["time_embedding_dim"]

        torch.manual_seed(0)
        g = KSpaceColdDiffusionGenerator(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            **kwargs,
        ).eval()

        c = model_cfg["in_channels"]
        x = torch.randn(1, c, 64, 64) * 0.1
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 28:36, :] = 1
        zero = torch.zeros(1, dtype=torch.long)
        extra = {"kspace_measured": torch.randn(1, c, 64, 64) * 0.1, "mask": mask}

        with torch.no_grad():
            a = g(x, zero, contrast_idx=zero, **extra)
            b = g(x, zero, contrast_idx=torch.full((1,), 2, dtype=torch.long), **extra)

        assert torch.isfinite(a).all()
        assert _rel(a, b) > 1e-4, "contrast conditioning inert end-to-end"
