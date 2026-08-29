"""Regression tests for ``domain_inference``'s SSOT registry consultation.

The 2026-05-10 mosaic-audit pass surfaced a class of bugs where models
declared ``output_domain`` via ``@register_model`` but ``domain_inference``
ignored the decorator metadata and used a hand-maintained
``KNOWN_KSPACE_OUTPUT_MODELS`` set instead. The two registries drifted out
of sync (CLAUDE.md #9 silent fallback), causing visualizations to apply
IFFT to image-domain outputs (and vice-versa) and producing the bad
PNGs that motivated this audit.

The decorator metadata is now Priority 2 in the cascade, the hardcoded
sets fall to Priority 3, and YAML overrides (P1) still win. These tests
pin that cascade so a future refactor can't re-introduce the silent
fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mriforge.infrastructure.training.utils.domain_inference import (
    KNOWN_IMAGE_OUTPUT_MODELS,
    KNOWN_KSPACE_OUTPUT_MODELS,
    _registry_output_domain,
    infer_output_domain,
    looks_like_kspace,
    needs_ifft_for_visualization,
)
from tests.utils.data_config_stub import DataConfigStub


class TestLooksLikeKspace:
    """Tensor-level k-space DC-signature veto (shared by the debug snapshot and
    the validation-image visualiser). Guards against IFFT'ing an already-image
    target into a DC blob when the config flags the dataset as k-space
    (smoke 2026-06-15: exp_p7 / exp_p3 real-image DC blobs)."""

    @staticmethod
    def _disk(h=64):
        import torch

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, h), indexing="ij"
        )
        return (((x**2 + y**2) < 0.5).float() * 10.0).unsqueeze(0).unsqueeze(0)

    def test_image_magnitude_is_not_kspace(self) -> None:
        assert looks_like_kspace(self._disk().abs()) is False

    def test_kspace_magnitude_detected(self) -> None:
        import torch

        from mriforge.infrastructure.physics.fft_ops import fft2c

        ksp = fft2c(self._disk().to(torch.complex64))
        assert looks_like_kspace(ksp.abs()) is True


# ── Fixtures ────────────────────────────────────────────────────────────────


def _cfg(
    model_type=None,
    target_domain=None,
    dataset_type="image",
    normalize_kspace=False,
    output_domain=None,
    enable_kspace_recon=False,
    coil_processing_mode=None,
):
    """Config whose ``data`` block carries the real, decomposed sub-block shape.

    A flat ``SimpleNamespace`` here is what let the P6 coil-mode override read as
    dead for months: it agreed with the stale flat reads in ``domain_inference``
    and the pair stayed green while no live config could reach the branch.
    """
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type, target_domain=target_domain),
        data=DataConfigStub(
            dataset_type=dataset_type,
            normalize_kspace=normalize_kspace,
            output_domain=output_domain,
            coil_processing_mode=coil_processing_mode,
        ),
        physics=SimpleNamespace(
            kspace=SimpleNamespace(enable_kspace_recon=enable_kspace_recon)
        ),
    )


# ── Priority 1: explicit YAML target_domain ──────────────────────────────────


def test_p1_yaml_target_domain_overrides_registry():
    """YAML ``model.target_domain`` must beat decorator metadata.

    The user has the final word — a registry annotation is a default,
    not a lock. If the user explicitly sets ``target_domain: image`` the
    visualization layer must trust them.
    """
    # ``varnet`` is in KNOWN_KSPACE_OUTPUT_MODELS but YAML says image
    cfg = _cfg(model_type="varnet", target_domain="image")
    assert infer_output_domain(cfg) == "image"


def test_p1_yaml_target_domain_kspace_overrides_image_registry():
    cfg = _cfg(model_type="standard_unet", target_domain="kspace")
    assert infer_output_domain(cfg) == "kspace"


# ── Priority 2: ``@register_model`` capability metadata ─────────────────────


def test_p2_registry_capability_consulted():
    """A model annotated with ``output_domain`` via the decorator must be
    classified per the decorator without requiring a hardcoded-set entry."""
    # ``unrolled_reconstruction`` registers output_domain='image'. Even with
    # normalize_kspace=True (which would otherwise trip the P6 heuristic),
    # the explicit decorator declaration wins.
    cfg = _cfg(
        model_type="unrolled_reconstruction",
        normalize_kspace=True,
        dataset_type="image",
    )
    assert infer_output_domain(cfg) == "image"


def test_p2_registry_lookup_helper_returns_none_for_unknown():
    assert _registry_output_domain("nonexistent_model_xyz") is None


def test_p2_registry_lookup_handles_missing_capabilities():
    """A registered model with no ``output_domain`` capability annotation
    returns ``None`` from the helper so the cascade falls through to P3+."""
    # Pick the first MODEL_REGISTRY entry that has no output_domain set
    from mriforge.models.registry import MODEL_REGISTRY

    for name, entry in MODEL_REGISTRY.items():
        caps = entry.get("capabilities")
        if caps is None or getattr(caps, "output_domain", None) is None:
            assert _registry_output_domain(name) is None
            break
    else:
        pytest.skip("All registered models annotate output_domain — nothing to test")


# ── Priority 3: hardcoded legacy sets (still functional) ────────────────────


def test_p3_legacy_set_still_works_for_unannotated_models():
    """Models in KNOWN_KSPACE_OUTPUT_MODELS that lack decorator annotation
    still get classified correctly via the legacy set."""
    # Pick a known k-space model that is NOT annotated via decorator
    from mriforge.models.registry import MODEL_REGISTRY

    candidates = [
        m
        for m in KNOWN_KSPACE_OUTPUT_MODELS
        if m not in MODEL_REGISTRY
        or getattr(MODEL_REGISTRY[m].get("capabilities"), "output_domain", None) is None
    ]
    if not candidates:
        pytest.skip("All KNOWN_KSPACE_OUTPUT_MODELS are now decorator-annotated")
    cfg = _cfg(model_type=candidates[0])
    assert infer_output_domain(cfg) == "kspace"


# ── VF field backbones reconstruct in image domain (smoke 2026-06-16) ────────
#
# These VF / multi-acquisition field models are run by ConcreteVirtualFiducialStrategy
# (and motion-meta / distillation), which IFFTs k-space → image BEFORE the backbone.
# Each forward in ``vf_field_generators.py`` is a residual on that image input
# (``... + x``) → image output (internal FFTs in siamese_epi_tracker / bssfp_peak_finder
# are intermediate, not the output). They were mis-listed in KNOWN_KSPACE_OUTPUT_MODELS,
# forcing a spurious IFFT on the already-image prediction → DC-blob fakes (E-VIZ2) — the
# same bug the 2026-06-03 fix corrected for cross_attention_oracle_unet / hyper_mamba_unet.
_VF_IMAGE_BACKBONES = (
    "resnet_unwrap",
    "bloch_siegert_algebraic",
    "afi_ratio_cnn",
    "dam_trigfit",
    "siamese_epi_tracker",
    "phase_tracking_lstm",
    "reciprocity_divisor",
    "mrf_dict_matcher",
    "bssfp_peak_finder",
    "graph_cut_unwrap",
)

# Second batch (all in active VF use): vf_reconstruction_generators.py residual
# image reconstructors + vf_kspace_operators.py marker operators whose forward is
# a residual / global-phase transform of the strategy's image input → image output
# (internal fft2c is marker/phase/filter estimation only, not the output path).
_VF_IMAGE_RECON_OPS = (
    "kalman_pll",
    "wiener_unet",
    "neural_complex_sum",
    "dirichlet_espirit",
    "mtf_deapodization",
    "phase_navigator_lock",
)

_VF_IMAGE_MODELS = _VF_IMAGE_BACKBONES + _VF_IMAGE_RECON_OPS


@pytest.mark.parametrize("model_type", _VF_IMAGE_MODELS)
def test_vf_field_backbone_classifies_as_image(model_type):
    """Moved from the k-space legacy set to the image one (P3)."""
    assert model_type in KNOWN_IMAGE_OUTPUT_MODELS
    assert model_type not in KNOWN_KSPACE_OUTPUT_MODELS
    # On a real VF arm (k-space dataset + normalize_kspace) the P3 image entry
    # must beat the P6 kspace-by-dataset heuristic.
    cfg = _cfg(model_type=model_type, dataset_type="kspace", normalize_kspace=True)
    assert infer_output_domain(cfg) == "image"


@pytest.mark.parametrize("model_type", _VF_IMAGE_MODELS)
def test_vf_field_backbone_no_spurious_ifft_on_predictions(model_type):
    """image-output → the visualizer must NOT IFFT the prediction (the E-VIZ2
    blob); k-space targets from the dataloader are still IFFT'd."""
    cfg = _cfg(model_type=model_type, dataset_type="kspace", normalize_kspace=True)
    needs_pred, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_pred is False
    assert needs_tgt is True


# ── Priority 6: data-pipeline heuristic ─────────────────────────────────────


def test_p6_normalize_kspace_triggers_kspace_when_no_higher_signal():
    """Unannotated model + normalize_kspace=True falls through to P6."""
    cfg = _cfg(model_type="some_unknown_model", normalize_kspace=True)
    assert infer_output_domain(cfg) == "kspace"


def test_p7_default_image_when_nothing_declared():
    cfg = _cfg(model_type="some_unknown_model")
    assert infer_output_domain(cfg) == "image"


# ── needs_ifft_for_visualization decoupling ─────────────────────────────────


def test_needs_ifft_decouples_preds_and_targets():
    """An image-domain model trained on k-space targets returns
    (False, True): no IFFT for predictions, IFFT for the targets coming
    from the dataloader."""
    # ``standard_unet`` is in KNOWN_IMAGE_OUTPUT_MODELS but dataset is k-space
    cfg = _cfg(model_type="standard_unet", dataset_type="kspace")
    needs_pred, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_pred is False
    assert needs_tgt is True


# ── Regression: diffusion strategy must use the SSOT, not model.input_type ──


def test_diffusion_strategy_uses_domain_inference_not_input_type():
    """Pin: the diffusion validation logging path must consult
    ``needs_ifft_for_visualization`` and NOT derive ``is_preds_image``
    from ``model.input_type``.

    Why: ``model.input_type`` is the *generator input* contract; it does
    NOT determine the output domain. Pre-2026-05-13, the diffusion
    strategy at line ~2987 had
    ``is_preds_image = _input_type not in ("kspace",)`` which silently
    misclassified any generator whose input was image-domain but whose
    output was k-space (or vice-versa). The mosaic audit traced this to
    several validation-PNG outliers; the SSOT fix wires the strategy
    into ``domain_inference`` like the GAN mixin already does.

    This test is intentionally a *grep-pin*: a future Claude / refactor
    can't reintroduce the ``input_type``-based heuristic without
    deleting this test.
    """
    import pathlib

    diffusion_src = pathlib.Path(
        "src/mriforge/infrastructure/training/strategies/diffusion.py"
    ).read_text(encoding="utf-8")

    # The SSOT import must be present in the validation logging path.
    assert "needs_ifft_for_visualization" in diffusion_src, (
        "diffusion.py must import needs_ifft_for_visualization from "
        "mriforge.infrastructure.training.utils.domain_inference for validation "
        "image logging — see test rationale for the failure mode."
    )

    # The historical bypass — deriving is_preds_image from input_type as
    # the PRIMARY signal — must not return. We allow it only inside an
    # ``except`` block (defensive fallback when the SSOT import itself
    # fails) and only when paired with a comment noting the SSOT primary.
    primary_bypass = (
        '_input_type = (\n            getattr(self.config.model, "input_type"'
    )
    # The bypass survives ONLY inside the defensive except branch (one
    # occurrence at most). Two or more means a regression.
    occurrences = diffusion_src.count(primary_bypass)
    assert occurrences <= 1, (
        f"diffusion.py contains {occurrences} input_type-based is_preds_image "
        "derivations — the SSOT (needs_ifft_for_visualization) must be the "
        "primary signal, with at most one defensive fallback in an except branch."
    )


def test_p1_image_input_kspace_output_returns_kspace():
    """The exact scenario the diffusion strategy bypass used to miscall:
    input_type=image, but the model declares output_domain=kspace.

    The SSOT must return ``kspace`` so the visualization applies IFFT;
    the old ``input_type``-based heuristic would have returned ``image``
    and skipped the IFFT, producing a DC-blob render.
    """
    cfg = _cfg(model_type="kspace_cold_diffusion")  # output_domain=kspace
    # input_type isn't even a field domain_inference looks at — that's the
    # whole point. Confirm we hit kspace by the registry P2 path.
    assert infer_output_domain(cfg) == "kspace"
    needs_pred, _ = needs_ifft_for_visualization(cfg)
    assert needs_pred is True


# ── Audit-2026-05-14 §1 round-6 — coil_processing_mode overrides ────────────


def test_coil_mode_rss_image_overrides_kspace_dataset_type():
    """``coil_processing_mode: rss_image`` forces image domain on the
    ``targets_are_kspace`` heuristic.

    Pre-2026-05-15 ``dataset_type: kspace`` + ``coil_processing_mode:
    rss_image`` returned ``needs_ifft_targets=True``, even though the
    TorchIO transform pipeline had already IFFT'd the data. The
    validation writer then IFFT'd an image-domain tensor → spectral
    noise pattern (audit §1 — experiment_113 / experiment_114 / etc.).
    """
    cfg = _cfg(
        model_type="d2_mamba",
        dataset_type="kspace",
        coil_processing_mode="rss_image",
    )
    needs_pred, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_tgt is False, (
        "rss_image coil mode must override dataset_type=kspace at the "
        "target-side visualization decision (audit-2026-05-14 §1 round-6)"
    )


def test_coil_mode_magnitude_also_overrides_kspace():
    """``coil_processing_mode: magnitude`` shares the rss_image semantics."""
    cfg = _cfg(
        dataset_type="kspace",
        normalize_kspace=True,
        coil_processing_mode="magnitude",
    )
    _preds, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_tgt is False


def test_coil_mode_svd_keeps_kspace_semantics():
    """``coil_processing_mode: svd`` (virtual-coil compression) preserves
    k-space domain — IFFT for visualization is correct here."""
    cfg = _cfg(
        model_type="kspace_cold_diffusion",
        dataset_type="kspace",
        normalize_kspace=True,
        coil_processing_mode="svd",
    )
    _preds, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_tgt is True


def test_coil_mode_none_keeps_kspace_semantics():
    """``coil_processing_mode: none`` keeps the kspace-by-dataset heuristic
    intact (no image-domain transform was applied to the data)."""
    cfg = _cfg(
        model_type="kspace_cold_diffusion",
        dataset_type="kspace",
        coil_processing_mode="none",
    )
    _preds, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_tgt is True


def test_infer_output_domain_p6_honours_coil_mode_rss_image():
    """``infer_output_domain``'s P6 heuristic respects ``coil_processing_mode``.

    Models without explicit ``target_domain`` (e.g. legacy graph models)
    used to fall through P1-P5 and land at P6 → "kspace" purely because
    ``dataset_type: kspace`` was declared. With ``rss_image`` the
    dataset emits image-domain output, so the inferred output domain
    is image too.
    """
    # Use a model name that is unlikely to be in the kspace registry sets
    # so we exercise the P6 heuristic path.
    cfg = _cfg(
        model_type="some_unregistered_model_xyz",
        dataset_type="kspace",
        coil_processing_mode="rss_image",
    )
    assert infer_output_domain(cfg) == "image"


# ── Smoke audit 2026-06-03 — VF backbones output image, not k-space ─────────
#
# The VF / motion-meta / distillation strategies IFFT k-space → image BEFORE
# the backbone runs, so ``cross_attention_oracle_unet`` and
# ``hyper_mamba_unet`` reconstruct in image domain. They were mis-listed in
# KNOWN_KSPACE_OUTPUT_MODELS, which forced a spurious IFFT on the
# already-image prediction → DC-blob fakes across 10 of 12 "PASS" VF arms in
# run 20260603_182243. They now live in KNOWN_IMAGE_OUTPUT_MODELS.


@pytest.mark.parametrize(
    "model_type", ["cross_attention_oracle_unet", "hyper_mamba_unet"]
)
def test_vf_backbones_are_image_domain_not_kspace(model_type):
    """Regression: the two VF marker backbones must be classified image."""
    assert model_type in KNOWN_IMAGE_OUTPUT_MODELS
    assert model_type not in KNOWN_KSPACE_OUTPUT_MODELS
    # rss_image arm (eval_c2/exp_c6 shape): magnitude image in, image out.
    cfg = _cfg(model_type=model_type, coil_processing_mode="rss_image")
    assert infer_output_domain(cfg) == "image"


@pytest.mark.parametrize(
    "model_type", ["cross_attention_oracle_unet", "hyper_mamba_unet"]
)
def test_vf_backbone_predictions_not_ifft_on_kspace_dataset(model_type):
    """The actual blob fix: predictions from these backbones must NOT be
    IFFT'd, even on a k-space dataset (the method_a / method_c shape: svd
    coil-compression, ``dataset_type: kspace``). Targets still IFFT because
    the on-disk references are k-space — that path was already correct (the
    real images rendered fine in the smoke run)."""
    cfg = _cfg(
        model_type=model_type,
        dataset_type="kspace",
        normalize_kspace=True,
        coil_processing_mode="svd",
    )
    needs_pred, needs_tgt = needs_ifft_for_visualization(cfg)
    assert needs_pred is False, (
        f"{model_type} output is image domain — IFFT'ing the prediction is "
        "the DC-blob bug (smoke audit 2026-06-03)"
    )
    assert needs_tgt is True


@pytest.mark.parametrize(
    "model_type", ["cross_attention_oracle_unet", "hyper_mamba_unet"]
)
def test_vf_backbone_kspace_override_still_wins(model_type):
    """An arm that genuinely emits k-space can still force IFFT via the
    explicit ``model.target_domain: kspace`` (P1 beats the P3 image set)."""
    cfg = _cfg(model_type=model_type, target_domain="kspace")
    assert infer_output_domain(cfg) == "kspace"
