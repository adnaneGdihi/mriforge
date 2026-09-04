"""Tests for the spec-card synthesizer's physics.coil_processing surfacing.

The audit spec card must reflect the unified ``physics.coil_processing`` block
(compression / estimation / combine), not just the legacy
``data.coil_processing_mode``.
"""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.infrastructure.validation.spec_card import (
    _derive_data_form,
    synthesize_spec_card,
)


def _cfg(coil_processing=None):
    physics = (
        SimpleNamespace(coil_processing=coil_processing)
        if coil_processing is not None
        else SimpleNamespace()
    )
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_type="fastmri_kspace", coil_processing_mode="none"
        ),
        model=SimpleNamespace(model_type="unet", in_channels=8, out_channels=2),
        physics=physics,
        losses=None,
    )


def test_coil_processing_block_surfaced_in_data_form():
    cp = SimpleNamespace(
        compression=SimpleNamespace(method="svd"),
        estimation=SimpleNamespace(method="espirit"),
        combine=SimpleNamespace(method="sense"),
        output=SimpleNamespace(domain="kspace", channels="real_interleaved"),
    )
    form = _derive_data_form(_cfg(cp))
    block = form["coil_processing_block"]
    assert block == {
        "compression": "svd",
        "estimation": "espirit",
        "combine": "sense",
        "output": ("kspace", "real_interleaved"),
    }


def test_coil_processing_block_none_when_absent():
    form = _derive_data_form(_cfg(None))
    assert form["coil_processing_block"] is None


def test_spec_card_renders_coil_pipeline_line():
    cp = SimpleNamespace(
        compression=SimpleNamespace(method="svd"),
        estimation=SimpleNamespace(method="espirit"),
        combine=SimpleNamespace(method="sense"),
    )
    card = synthesize_spec_card(_cfg(cp))
    assert "coil_pipeline:" in card
    assert "compress=svd" in card
    assert "estimate=espirit" in card
    assert "combine=sense" in card


def test_spec_card_includes_resolved_ir():
    # WS-X "reuse the IR": the card appends the resolved-context view the
    # compatibility matrix evaluates (additive — does not replace the rich
    # _format_card sections).
    card = synthesize_spec_card(_cfg())
    assert "[RESOLVED CONTEXT" in card
    assert "ResolvedExperimentContext" in card
    assert "model     : unet" in card
    assert "model.io" in card


def test_spec_card_reads_the_decomposed_data_blocks():
    """The card reported ``None`` for the five facts it exists to report.

    ``_safe_get`` walks STRINGS, so after phases 9-10 moved ``patch_size``,
    ``coil_processing_mode``, ``normalization_type``, ``target_channels`` and
    ``losses.output_domain`` into sub-blocks, every one of them resolved to the
    default instead of raising. A real settings object is the only stand-in that
    can catch that -- a ``SimpleNamespace`` has whatever attribute you give it.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.spec_card import (
        _derive_data_form,
        _derive_loss_form,
    )

    settings = TrainingSettings.settings_from_dict(
        {
            "data": {
                "train_path": "/tmp/t",
                "val_path": "/tmp/v",
                "patch_size": [256, 256, 1],
                "coil_processing_mode": "rss",
                "normalization_type": "none",
                "target_channels": 2,
            },
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
            "losses": {"output_domain": "image"},
        }
    )
    data_form = _derive_data_form(settings)
    assert data_form["patch_size"] == (256, 256, 1)
    assert data_form["spatial_dims"] == 2
    assert data_form["coil_processing_mode"] == "rss"
    assert data_form["normalization_type"] == "none"
    assert data_form["target_channels"] == 2
    assert _derive_loss_form(settings)["output_domain"] == "image"


def test_safe_get_walks_dotted_paths_and_falls_back():
    from types import SimpleNamespace

    from spectramr.infrastructure.validation.spec_card import _safe_get

    nested = SimpleNamespace(sampling=SimpleNamespace(patch_size=(64, 64, 1)))
    assert _safe_get(nested, "sampling.patch_size", "patch_size") == (64, 64, 1)
    # Legacy flat spelling still resolves while the rename is `fold` posture.
    flat = SimpleNamespace(patch_size=(32, 32, 1))
    assert _safe_get(flat, "sampling.patch_size", "patch_size") == (32, 32, 1)
    # A path that exists nowhere yields the default rather than raising.
    assert _safe_get(flat, "nope.nothing", default="d") == "d"
