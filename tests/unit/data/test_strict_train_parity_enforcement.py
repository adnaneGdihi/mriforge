"""Enforce ``data.modes.infer.strict_train_parity`` at inference load.

When the flag is true and the checkpoint's recorded transform signature
diverges from the chain the current YAML resolves to, the load must be
refused rather than silently producing wrong-norm / wrong-shape tensors
(CLAUDE.md #9).

These tests previously drove ``DataPipelineDirector.build_inference_handle``,
which had zero production callers -- so the suite was green over a dead copy
while the live enforcement in ``pipelines/infer.py`` had no coverage at all.
Both now route through the single implementation,
:func:`mriforge.data.transforms.signature.enforce_train_infer_parity`.
"""

from __future__ import annotations

import pytest


def _make_minimal_settings(strict: bool = True):
    """A TrainingSettings-shaped stand-in with strict_train_parity set."""
    from mriforge.config.schemas.data import (
        DataConfigSchema,
        DataModesSchema,
        ModeConfigSchema,
        ModeSamplerSchema,
    )

    data = DataConfigSchema(
        modes=DataModesSchema(
            infer=ModeConfigSchema(
                sampler=ModeSamplerSchema(type="full"),
                strict_train_parity=strict,
            )
        )
    )

    class _Settings:
        def __init__(self, data_cfg):
            self.data = data_cfg
            # Phase 11 spelling. The proxy inside compute_infer_signature
            # reads `undersampling`; naming it `acceleration` here is how an
            # earlier version of this suite silently tested nothing.
            self.undersampling = None

    return _Settings(data)


def test_strict_parity_raises_when_signature_diverges() -> None:
    """A bogus checkpoint signature must refuse the load."""
    from mriforge.data.transforms.signature import enforce_train_infer_parity

    settings = _make_minimal_settings(strict=True)
    with pytest.raises(RuntimeError, match="strict_train_parity"):
        enforce_train_infer_parity(settings, "0" * 64)


def test_strict_parity_succeeds_when_signatures_match() -> None:
    """The signature the checker computes is the one it accepts."""
    from mriforge.data.transforms.signature import (
        compute_infer_signature,
        enforce_train_infer_parity,
    )

    settings = _make_minimal_settings(strict=True)
    expected = compute_infer_signature(settings)

    returned = enforce_train_infer_parity(settings, expected)
    assert returned == expected


def test_strict_parity_off_skips_check_even_with_bad_signature() -> None:
    """strict_train_parity=false ⇒ the checkpoint signature is not enforced."""
    from mriforge.data.transforms.signature import enforce_train_infer_parity

    settings = _make_minimal_settings(strict=False)
    # Must not raise despite a deliberately wrong signature.
    assert enforce_train_infer_parity(settings, "0" * 64)


def test_strict_parity_with_none_checkpoint_signature_raises() -> None:
    """A pre-Phase-2 checkpoint records no signature, so it cannot satisfy
    parity -- the user must opt out explicitly rather than be waved through."""
    from mriforge.data.transforms.signature import enforce_train_infer_parity

    settings = _make_minimal_settings(strict=True)
    with pytest.raises(RuntimeError, match="missing"):
        enforce_train_infer_parity(settings, None)


def test_infer_pipeline_calls_the_shared_enforcement() -> None:
    """The live pipeline must route through the shared checker.

    Asserting the seam, not the unit: the defect this suite exists to prevent
    is a second, divergent copy of the parity check -- which is exactly what
    the deleted director method was.
    """
    import inspect

    from mriforge.pipelines import infer as infer_mod

    src = inspect.getsource(infer_mod.run_inference_pipeline)
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "enforce_train_infer_parity(" in code, (
        "run_inference_pipeline no longer calls the shared parity checker"
    )
    assert "diff_signatures(" not in code, (
        "run_inference_pipeline re-inlined the parity comparison; it must "
        "delegate to enforce_train_infer_parity so there is one implementation"
    )
