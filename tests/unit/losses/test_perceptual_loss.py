"""Unit tests for perceptual_loss.py.

Covers: PerceptualLoss (SSOT), DISTS, VGGPerceptualLoss.
VGG-backbone forward passes are marked @pytest.mark.slow — canary tests
are intentionally lightweight (no VGG weight download required in fast path).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.registry import create_loss

B, C, H, W = 2, 1, 64, 64


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


# ---------------------------------------------------------------------------
# Helper: attempt VGG construction and skip if network unavailable
# ---------------------------------------------------------------------------

def _try_vgg_import():
    """Return True if VGG19 can be imported and instantiated (no network ok)."""
    try:
        from torchvision import models as _m
        _m.vgg19(weights=None)
        return True
    except Exception:
        return False


_VGG_AVAILABLE = _try_vgg_import()
_vgg_skip = pytest.mark.skipif(
    not _VGG_AVAILABLE,
    reason="torchvision VGG19 not importable in this environment",
)


# ===========================================================================
# PerceptualLoss (SSOT)
# ===========================================================================


class TestPerceptualLoss:
    """Canary, property, edge, raises for PerceptualLoss."""

    # ── Canary ──────────────────────────────────────────────────────────────

    @pytest.mark.slow
    @_vgg_skip
    def test_canary_builds_and_forward_finite(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss()
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.slow
    @_vgg_skip
    @pytest.mark.parametrize("criterion", ["l1", "mse"])
    def test_parametric_criteria(self, criterion: str):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss(criterion=criterion)
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.item() >= 0.0

    # ── Property ────────────────────────────────────────────────────────────

    @pytest.mark.slow
    @_vgg_skip
    def test_property_identity_near_zero(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss()
        x = _img()
        # Perceptual loss(x,x) should be 0 or very near 0
        out = loss_fn(x, x)
        assert out.item() < 1e-4

    @pytest.mark.slow
    @_vgg_skip
    def test_property_nonneg(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss()
        x = _img()
        y = _img(seed=2)
        assert loss_fn(x, y).item() >= 0.0

    @pytest.mark.slow
    @_vgg_skip
    def test_property_gradient_reaches_input(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss()
        x = _img().requires_grad_(True)
        y = _img(seed=2)
        loss_fn(x, y).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    # ── Edge ────────────────────────────────────────────────────────────────

    @pytest.mark.slow
    @_vgg_skip
    def test_edge_multichannel_grayscale(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss()
        x = _img(c=1)
        y = _img(c=1, seed=3)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    @pytest.mark.slow
    @_vgg_skip
    def test_edge_nan_input_returns_zero(self):
        """NaN inputs should return 0 (nan-guard in PerceptualLoss.forward)."""
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = PerceptualLoss(use_input_norm=True)
        x = torch.full((B, 1, H, W), float("nan"))
        y = _img()
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_unknown_criterion(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        # The PerceptualLoss constructor raises on unknown criterion.
        # We skip VGG instantiation by checking before the class init reaches VGG load.
        # NOTE: This will succeed or fail depending on whether VGG is available;
        # the ValueError is raised before VGG instantiation.
        if not _VGG_AVAILABLE:
            pytest.skip("VGG not available; ImportError would mask ValueError")
        with pytest.raises(ValueError, match="criterion"):
            from spectramr.models.losses.perceptual_loss import PerceptualLoss
            PerceptualLoss(criterion="nonexistent_criterion_xyz")

    def test_raises_actionable_message_when_torchvision_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """torchvision is a core dependency imported lazily inside the constructor
        (#941/#968) -- a broken install must raise with an actionable message
        pointing at the pinned index, not a bare ModuleNotFoundError and not a
        silent fallback (non-negotiable #3)."""
        monkeypatch.setitem(sys.modules, "torchvision", None)
        from spectramr.models.losses.perceptual_loss import PerceptualLoss

        with pytest.raises(ImportError, match="pytorch-cu126"):
            PerceptualLoss()

    def test_raises_actionable_message_when_torchvision_abi_mismatched(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A torch/torchvision ABI mismatch surfaces as RuntimeError (e.g.
        `operator torchvision::nms does not exist`), not ImportError -- the
        deferred-import guard must catch and re-raise that case too (#968's
        pyproject.toml comment documents the failure mode)."""

        class _BrokenTorchvision(types.ModuleType):
            def __getattr__(self, name: str):
                raise RuntimeError("operator torchvision::nms does not exist")

        monkeypatch.setitem(
            sys.modules, "torchvision", _BrokenTorchvision("torchvision")
        )
        from spectramr.models.losses.perceptual_loss import PerceptualLoss

        with pytest.raises(ImportError, match="pytorch-cu126"):
            PerceptualLoss()

    # ── Registry ────────────────────────────────────────────────────────────

    @pytest.mark.slow
    @_vgg_skip
    def test_registry_lookup(self):
        from spectramr.models.losses.perceptual_loss import PerceptualLoss
        loss_fn = create_loss("perceptual")
        assert isinstance(loss_fn, PerceptualLoss)


# ===========================================================================
# DISTS
# ===========================================================================


class TestDISTS:

    @pytest.mark.slow
    @_vgg_skip
    def test_canary_builds_and_forward_finite(self):
        from spectramr.models.losses.perceptual_loss import DISTS
        loss_fn = DISTS()
        x = _img(c=1)
        y = _img(c=1, seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.slow
    @_vgg_skip
    @pytest.mark.parametrize("backbone", ["vgg19", "vgg16"])
    def test_parametric_backbones(self, backbone: str):
        from spectramr.models.losses.perceptual_loss import DISTS
        loss_fn = DISTS(backbone=backbone)
        x = _img(c=1)
        y = _img(c=1, seed=2)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.item() >= 0.0

    @pytest.mark.slow
    @_vgg_skip
    def test_property_nonneg(self):
        from spectramr.models.losses.perceptual_loss import DISTS
        loss_fn = DISTS()
        x = _img(c=1)
        y = _img(c=1, seed=3)
        assert loss_fn(x, y).item() >= 0.0

    @pytest.mark.slow
    @_vgg_skip
    def test_property_identity_near_zero(self):
        from spectramr.models.losses.perceptual_loss import DISTS
        loss_fn = DISTS()
        x = _img(c=1)
        out = loss_fn(x, x)
        assert out.item() < 1e-4

    @pytest.mark.slow
    @_vgg_skip
    def test_property_return_components(self):
        from spectramr.models.losses.perceptual_loss import DISTS
        loss_fn = DISTS()
        x = _img(c=1)
        y = _img(c=1, seed=4)
        loss, loss_s, loss_t = loss_fn(x, y, return_components=True)
        assert all(torch.isfinite(t) for t in (loss, loss_s, loss_t))

    def test_raises_unsupported_backbone(self):
        if not _VGG_AVAILABLE:
            pytest.skip("VGG not available")
        from spectramr.models.losses.perceptual_loss import DISTS
        with pytest.raises(ValueError, match="backbone"):
            DISTS(backbone="resnet50_xyz_unknown")

    @pytest.mark.slow
    @_vgg_skip
    def test_registry_lookup(self):
        from spectramr.models.losses.perceptual_loss import DISTS
        loss_fn = create_loss("dists")
        assert isinstance(loss_fn, DISTS)


class TestTorchvisionWeightsAPI:
    """No losses module may pass the deprecated ``pretrained=`` kwarg (#961).

    Every ``TestDISTS`` case above is ``@pytest.mark.slow`` and gated on a
    weights download, so the fast lane never constructs the class. That is
    why ``DISTS.__init__`` kept ``pretrained=True`` at two call sites while
    ``PerceptualLoss`` -- 250 lines above it in the same file -- used the
    modern ``weights=`` API and carried a comment explaining why. An AST
    gate needs no network, so it runs where the defect actually was.

    Scope note, measured on torchvision 0.28.0 rather than assumed: the
    kwarg is **deprecated, not removed**. ``models.vgg19(pretrained=True)``
    still returns a model and emits two ``UserWarning``s. So this gate
    protects against warning noise and a future removal -- #961's stronger
    claim, that ``DISTS`` is unconstructable, does not reproduce.
    """

    #: torchvision constructors these modules call for pretrained backbones.
    _BACKBONE_CALLS = frozenset({"vgg16", "vgg19", "resnet50", "alexnet", "squeezenet1_1"})

    @staticmethod
    def _pretrained_kwarg_sites(path) -> list[str]:
        """``file:line`` of every ``<backbone>(..., pretrained=...)`` call."""
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        sites: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name not in TestTorchvisionWeightsAPI._BACKBONE_CALLS:
                continue
            if any(kw.arg == "pretrained" for kw in node.keywords):
                sites.append(f"{path.name}:{node.lineno}")
        return sites

    def test_no_losses_module_passes_pretrained(self):
        from pathlib import Path

        import spectramr.models.losses as losses_pkg

        root = Path(losses_pkg.__file__).parent
        offenders: list[str] = []
        for py in sorted(root.rglob("*.py")):
            offenders.extend(self._pretrained_kwarg_sites(py))

        assert not offenders, (
            f"{offenders} pass the deprecated `pretrained=` kwarg. torchvision "
            "deprecated it in 0.13 in favour of `weights=<Model>_Weights.…`; it "
            "still resolves on 0.28.0 but emits two UserWarnings per construction "
            "and is slated for removal. Use the weights enum, as PerceptualLoss "
            "and VGGFeatureExtractor already do."
        )

    def test_the_gate_is_not_vacuous(self, tmp_path):
        """A module that DOES pass the kwarg must be detected."""
        offender = tmp_path / "offender.py"
        offender.write_text("m.vgg19(pretrained=True)\n", encoding="utf-8")
        assert self._pretrained_kwarg_sites(offender) == ["offender.py:1"]
