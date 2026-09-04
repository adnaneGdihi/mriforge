"""Regression tests for F16 (smoke_audit_20260521 round 8) — YAML fixes
for the Generator Build Failure cohort.

Eight experiments crashed at training-pipeline-build with explicit
fix hints from ``model_creator``:

1. ``features: list`` vs the generator's ``features: int`` declaration:
   - ``experiment_19_neural_ode``
   - ``experiment_52_dynamic_latent_odes_cine_mri``
   - ``experiment_18_equivariant_networks``
   - ``experiment_35_deep_compressed_sensing_unrolled``
   - ``experiment_38_parallel_imaging_deep_sense``
   - ``experiment_44_hard_constraint_data_consistency``

   Fix: collapse the 5-element list to a single int (the first
   element, which matches silent-coercion behavior).

2. FNO out_channels architectural contradiction:
   - ``experiment_54_neural_operator_reconstruction_fno`` —
     ``out_channels: 8`` → ``out_channels: 2`` (FNO architecturally
     produces 2 channels from its inverse-FFT round-trip).

3. Invalid attention_type:
   - ``exp_kspace_cross_attention`` —
     ``attention_type: multi_contrast_kspace`` (not in registry) →
     ``attention_type: none``, which on 2026-05-21 was the only complex-safe
     option per F14. SUPERSEDED (20fbfeee4, 2026-06-06): ``cross_contrast_olmpa``
     (CC-OLMPA) was registered as a complex-safe cross-contrast attention the
     ComplexUNet blocks build — it is in ``COMPLEX_UNET_BLOCK_ATTENTION`` and
     maps to both feature domains — and the arm was deliberately re-wired to it.
     Pinning ``none`` now would strip the arm's headline mechanism (pitfall #16
     facade), so the test asserts the complex-safe cross-attention instead.

Per CLAUDE.md #9 the model_creator refuses silent coercion — these
YAMLs MUST be fixed at the config layer. These tests pin each YAML so
a bulk migration can't silently strip the F16 fixes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]


def _load(relpath: str) -> dict:
    return yaml.safe_load((REPO / relpath).read_text())


@pytest.mark.parametrize(
    "relpath",
    [
        "experiments/inprogress/ode_temporal/experiment_19_neural_ode.yaml",
        "experiments/inprogress/ode_temporal/experiment_52_dynamic_latent_odes_cine_mri.yaml",
        "experiments/inprogress/physics_driven/experiment_18_equivariant_networks.yaml",
        "experiments/inprogress/physics_driven/experiment_35_deep_compressed_sensing_unrolled.yaml",
        "experiments/inprogress/physics_driven/experiment_38_parallel_imaging_deep_sense.yaml",
        "experiments/inprogress/physics_driven/experiment_44_hard_constraint_data_consistency.yaml",
    ],
)
def test_features_is_int_not_list(relpath: str) -> None:
    """Every affected YAML's ``model.model_kwargs.features`` must be a
    single int (not a list). The model_creator refuses silent
    coercion per CLAUDE.md #9 — the YAML must be the authoritative
    value."""
    cfg = _load(relpath)
    features = cfg["model"]["model_kwargs"].get("features")
    assert isinstance(features, int), (
        f"{relpath}: model.model_kwargs.features must be int "
        f"(generator's __init__ declares ``features: int``); "
        f"got {type(features).__name__}={features!r}. "
        "See TODO/audit/smoke_audit_20260521.md §F16."
    )


def test_fno_out_channels_is_two() -> None:
    """``experiment_54_neural_operator_reconstruction_fno``: FNO emits
    2 channels architecturally; YAML must match. The model_creator
    refuses ``out_channels=8`` per CLAUDE.md #9 silent-fallback rule."""
    cfg = _load(
        "experiments/inprogress/physics_driven/experiment_54_neural_operator_reconstruction_fno.yaml"
    )
    assert cfg["model"]["out_channels"] == 2, (
        "FNOGenerator produces 2 channels architecturally; YAML "
        "model.out_channels must match. See §F16."
    )


def test_kspace_cross_attention_uses_valid_attention_type() -> None:
    """``exp_kspace_cross_attention`` had ``attention_type: multi_contrast_kspace``
    (unregistered). The F16 fix set it to ``none`` because on 2026-05-21 that was
    the only complex-safe option per F14. That is now SUPERSEDED: CC-OLMPA
    (``cross_contrast_olmpa``) was registered as a complex-safe cross-contrast
    attention the ComplexUNet blocks build, and the arm was deliberately re-wired
    to it (20fbfeee4). The arm's whole hypothesis is cross-contrast k-space
    attention, so pinning ``none`` would strip its headline mechanism (pitfall
    #16). The test asserts a real complex-safe cross-attention fires instead."""
    from spectramr.models.blocks.attention_domains import COMPLEX_UNET_BLOCK_ATTENTION

    cfg = _load("experiments/inprogress/multi_contrast/exp_kspace_cross_attention.yaml")
    mk = cfg["model"]["model_kwargs"]
    attn = mk["attention_type"]
    # The named mechanism must fire, not collapse to the stale-F14 facade.
    assert attn != "none", (
        "exp_kspace_cross_attention must keep its cross-contrast attention; "
        "'none' makes the arm's headline mechanism inert (pitfall #16). See §F16."
    )
    # Complex-safe: it must be one the ComplexUNet down/up blocks can build.
    assert attn in COMPLEX_UNET_BLOCK_ATTENTION, (
        f"attention_type={attn!r} is not in the complex_unet block dispatch "
        f"under use_complex_conv; choose one of {sorted(COMPLEX_UNET_BLOCK_ATTENTION)}."
    )
    # The arm's declared headline mechanism is CC-OLMPA specifically.
    assert (
        attn == "cross_contrast_olmpa"
    ), f"exp_kspace_cross_attention's headline mechanism is CC-OLMPA; got {attn!r}."
    # F14 background: block attention is restricted under complex conv, but
    # CC-OLMPA is the complex-safe exception this arm exists to exercise.
    assert mk["use_complex_conv"] is True
    # CC-OLMPA is cross-CONTRAST: it degenerates without >=2 contrasts to attend across.
    assert int(mk.get("num_contrasts", 0)) >= 2, (
        "cross-contrast attention needs multiple contrasts; a single-contrast "
        "config would make CC-OLMPA a facade of a different kind."
    )
