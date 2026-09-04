"""#1387 — input-concat exemption must be resolved per arm, not per model_type.

``ConfigHealthChecker`` exempted every ``kspace_cold_diffusion`` arm from four
channel checks because the model type is in ``_INPUT_CONCAT_MODELS``. But the
generator resolves S-map concatenation as a CONJUNCTION::

    condition_with_smaps AND backbone_type not in _INTERNAL_DC_BACKBONES

so the six ``diff_varnet`` / ``diff_varnet_kan`` arms are built at ``1x`` and
concatenate nothing -- and were exempted anyway. A declared/actual channel
mismatch on those arms passed the audit and crashed at the first training batch.

The fix routes all four sites through
:meth:`ConfigHealthChecker._expects_input_concat`, which delegates to the
generator's own :func:`config_expects_smaps_concat`. These tests pin BOTH
halves: the truth table, and -- more importantly -- that the checker does not
own a second copy of the rule (CLAUDE.md #17). ``test_checker_follows_the_
generators_set`` mutates the generator's frozenset and asserts the checker's
answer moves with it; a re-listed copy in the checker would keep the old
answer and fail.

Config-level only: SimpleNamespace stand-ins, no model construction, no
forward pass.
"""

from __future__ import annotations

import inspect
import types
from typing import Any

import pytest

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
    config_expects_smaps_concat,
)


def _model(model_type: str, model_kwargs: dict[str, Any] | None = None, **extra: Any):
    ns = types.SimpleNamespace(model_type=model_type, in_channels=8, **extra)
    if model_kwargs is not None:
        ns.model_kwargs = model_kwargs
    return ns


# ── the resolver's truth table ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model_kwargs", "expected", "why"),
    [
        (None, True, "no model_kwargs -> generator defaults (unet, True)"),
        ({}, True, "empty model_kwargs -> generator defaults"),
        ({"backbone_type": "diff_varnet"}, False, "internal-DC backbone, built 1x"),
        ({"backbone_type": "diff_varnet_kan"}, False, "internal-DC backbone, built 1x"),
        ({"backbone_type": "swin_diff_rec"}, True, "swin family DOES concat"),
        ({"backbone_type": "swin_diff_rec_kan"}, True, "swin family DOES concat"),
        ({"backbone_type": "complex_unet"}, True, "not internal-DC"),
        ({"condition_with_smaps": False}, False, "arm declares no conditioning"),
        (
            {"backbone_type": "diff_varnet", "condition_with_smaps": True},
            False,
            "conjunction: an internal-DC backbone overrides the declaration",
        ),
        (
            {"backbone_type": "swin_diff_rec", "condition_with_smaps": False},
            False,
            "conjunction: the declaration overrides a concat-capable backbone",
        ),
    ],
)
def test_config_resolver_truth_table(
    model_kwargs: dict[str, Any] | None, expected: bool, why: str
) -> None:
    assert config_expects_smaps_concat(model_kwargs) is expected, why


def test_condition_with_smaps_false_is_not_reducible_to_the_backbone() -> None:
    """The half a backbone-only predicate would get wrong.

    Both arms below have a concat-capable backbone; only the declaration
    differs. A predicate that consulted ``_INTERNAL_DC_BACKBONES`` alone would
    answer True for both.
    """
    assert config_expects_smaps_concat({"backbone_type": "swin_diff_rec"}) is True
    assert (
        config_expects_smaps_concat(
            {"backbone_type": "swin_diff_rec", "condition_with_smaps": False}
        )
        is False
    )


# ── the checker's predicate ──────────────────────────────────────────────────


def test_checker_exempts_a_concat_arm() -> None:
    checker = ConfigHealthChecker()
    assert checker._expects_input_concat(_model("kspace_cold_diffusion")) is True


def test_checker_does_not_exempt_internal_dc_arms() -> None:
    """The #1387 defect itself: these six arms were exempted and should not be."""
    checker = ConfigHealthChecker()
    for backbone in ("diff_varnet", "diff_varnet_kan"):
        model = _model("kspace_cold_diffusion", {"backbone_type": backbone})
        assert checker._expects_input_concat(model) is False, backbone


def test_checker_exempts_disentangled_mri_unconditionally() -> None:
    """guided_sr_strategy concats HF_reference with no backbone nuance."""
    checker = ConfigHealthChecker()
    assert checker._expects_input_concat(_model("disentangled_mri")) is True
    assert (
        checker._expects_input_concat(
            _model("disentangled_mri", {"backbone_type": "diff_varnet"})
        )
        is True
    )


def test_checker_rejects_unrelated_model_types() -> None:
    checker = ConfigHealthChecker()
    assert checker._expects_input_concat(_model("configurable_unet")) is False
    assert checker._expects_input_concat(None) is False


# ── the one-owner gate (CLAUDE.md #17) ───────────────────────────────────────


def test_checker_follows_the_generators_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checker must READ the generator's set, not carry a copy of it.

    Move ``swin_diff_rec`` into ``_INTERNAL_DC_BACKBONES`` and the checker's
    verdict for a swin arm must flip. If the checker ever re-lists the backbone
    names locally, this stays True and the test fails -- which is the point:
    a second resolver does not announce itself, it just answers differently.
    """
    checker = ConfigHealthChecker()
    swin = _model("kspace_cold_diffusion", {"backbone_type": "swin_diff_rec"})
    assert checker._expects_input_concat(swin) is True

    monkeypatch.setattr(
        KSpaceColdDiffusionGenerator,
        "_INTERNAL_DC_BACKBONES",
        frozenset({"diff_varnet", "diff_varnet_kan", "swin_diff_rec"}),
    )
    assert checker._expects_input_concat(swin) is False, (
        "ConfigHealthChecker did not follow KSpaceColdDiffusionGenerator."
        "_INTERNAL_DC_BACKBONES — it is carrying a second copy of the rule (#17)."
    )


def test_generator_defaults_have_one_owner() -> None:
    """``__init__`` and the config resolver must not default differently."""
    import inspect

    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        DEFAULT_BACKBONE_TYPE,
        DEFAULT_CONDITION_WITH_SMAPS,
    )

    src = inspect.getsource(KSpaceColdDiffusionGenerator.__init__)
    assert 'kwargs.pop("backbone_type", DEFAULT_BACKBONE_TYPE)' in src
    assert 'kwargs.pop("condition_with_smaps", DEFAULT_CONDITION_WITH_SMAPS)' in src
    # And the resolver agrees with those defaults for an arm that omits both.
    assert config_expects_smaps_concat(None) is bool(
        DEFAULT_CONDITION_WITH_SMAPS
        and DEFAULT_BACKBONE_TYPE not in KSpaceColdDiffusionGenerator._INTERNAL_DC_BACKBONES
    )


# ── the consumer sites actually change behaviour ─────────────────────────────


def _data(dataset_type: str = "kspace", single_contrast: bool = False) -> Any:
    """The data block as the SCHEMA declares it.

    ``dataset_type`` and ``pairing.single_contrast`` are declared fields, so a
    real ``TrainingSettings`` always carries them; a fixture that omits them
    does not model the object the check runs on, and would let a defaulting
    read pass unnoticed. Default ``dataset_type="kspace"`` matches the four
    #1387 arms that are not on the M4Raw loader at all.
    """
    return types.SimpleNamespace(
        dataset_type=dataset_type,
        pairing=types.SimpleNamespace(single_contrast=single_contrast),
        coils=types.SimpleNamespace(processing_mode="rss_image"),
        domain=types.SimpleNamespace(target_channels=None),
    )


def _channel_audit_cfg(
    model_type: str,
    model_kwargs: dict[str, Any] | None,
    *,
    dataset_type: str = "kspace",
    single_contrast: bool = False,
) -> Any:
    return types.SimpleNamespace(
        model=_model(model_type, model_kwargs),
        data=_data(dataset_type, single_contrast),
        adapters=types.SimpleNamespace(pre_model=[]),
    )


def test_channel_audit_still_bypasses_a_real_concat_arm() -> None:
    results = ConfigHealthChecker().check_channel_audit_assumptions(
        _channel_audit_cfg("kspace_cold_diffusion", {"backbone_type": "swin_diff_rec"})
    )
    assert any("_INPUT_CONCAT_MODELS" in r.message for r in results)


def test_channel_audit_no_longer_bypasses_an_internal_dc_arm() -> None:
    """Site 3 of 4: the bypass notice must be gone for diff_varnet."""
    results = ConfigHealthChecker().check_channel_audit_assumptions(
        _channel_audit_cfg("kspace_cold_diffusion", {"backbone_type": "diff_varnet"})
    )
    assert not any("_INPUT_CONCAT_MODELS" in r.message for r in results), (
        "diff_varnet concatenates nothing, so the input-concat bypass must not fire"
    )


def test_adapter_chain_no_longer_skips_an_internal_dc_arm() -> None:
    """Site 4 of 4: the static pre_model channel walk must not be skipped."""
    checker = ConfigHealthChecker()
    # ``magnitude`` is IN the F7-Hoist known-effect set, so removing the concat
    # exemption makes the walk genuinely RUN. With an unknown adapter the check
    # defers for an unrelated reason and the test would pass without proving it.
    step = types.SimpleNamespace(name="magnitude")

    def cfg(
        model_kwargs: dict[str, Any],
        *,
        dataset_type: str = "kspace",
        single_contrast: bool = False,
    ) -> Any:
        return types.SimpleNamespace(
            model=_model("kspace_cold_diffusion", model_kwargs),
            data=_data(dataset_type, single_contrast),
            adapters=types.SimpleNamespace(pre_model=[step]),
        )

    skip_msg = "static pre_model channel resolution skipped"
    concat = checker.check_adapter_chain_channel_resolution(
        cfg({"backbone_type": "swin_diff_rec"})
    )
    assert skip_msg in concat.message, "a real concat arm must still be skipped"

    internal_dc = checker.check_adapter_chain_channel_resolution(
        cfg({"backbone_type": "diff_varnet"})
    )
    assert skip_msg not in internal_dc.message, (
        "diff_varnet concatenates nothing, so the chain walk must run"
    )


# ── _PAIRED_CONTRAST_MODELS: the SECOND set that shadowed the same six arms ──
#
# ``_expects_input_concat`` alone left #1387 inert at two of its four sites,
# because ``_PAIRED_CONTRAST_MODELS`` also holds ``kspace_cold_diffusion`` and
# was tested by bare membership. The set's own comment scopes that entry to
# ``dataset_type: m4raw`` + ``single_contrast: false``; ``_is_paired_contrast``
# is that comment made executable.


@pytest.mark.parametrize(
    ("dataset_type", "single_contrast", "expected"),
    [
        # The one shape the set was written for: the M4Raw cross-contrast
        # loader really does cat [T1 || target] along the coil axis.
        ("m4raw", False, True),
        # Same loader, pairing switched OFF — two kspace_filling arms.
        ("m4raw", True, False),
        # Not the M4Raw loader at all: ``single_contrast`` reaches exactly one
        # dataset, so nothing pairs here. Four #1387 arms.
        ("kspace", False, False),
        ("kspace", True, False),
        ("image", False, False),
    ],
)
def test_paired_contrast_truth_table(
    dataset_type: str, single_contrast: bool, expected: bool
) -> None:
    checker = ConfigHealthChecker()
    cfg = types.SimpleNamespace(data=_data(dataset_type, single_contrast))
    assert checker._is_paired_contrast(cfg, "kspace_cold_diffusion") is expected


def test_paired_contrast_is_not_reducible_to_the_dataset_type() -> None:
    """m4raw alone is the wrong predicate — ``single_contrast`` flips it.

    A reader who checks only ``dataset_type == 'm4raw'`` gets
    ``experiment_11_kspace_cold_diffusion_varnet`` and
    ``experiment_11b_diff_varnet`` wrong: both are m4raw AND
    ``single_contrast: true``.
    """
    checker = ConfigHealthChecker()
    paired = types.SimpleNamespace(data=_data("m4raw", False))
    unpaired = types.SimpleNamespace(data=_data("m4raw", True))
    assert checker._is_paired_contrast(paired, "kspace_cold_diffusion")
    assert not checker._is_paired_contrast(unpaired, "kspace_cold_diffusion")


def test_unrelated_model_types_are_never_paired() -> None:
    checker = ConfigHealthChecker()
    cfg = types.SimpleNamespace(data=_data("m4raw", False))
    for model_type in ("unet", "swin_unetr", None, "disentangled_mri"):
        assert not checker._is_paired_contrast(cfg, model_type)


def test_universal_multitask_dual_stays_bare_membership() -> None:
    """Boundary: #1387 narrows the ``kspace_cold_diffusion`` entry ONLY.

    ``universal_multitask_dual`` has zero arms under ``experiments/inprogress/``,
    so nothing here measured its Pattern B fusion. Its verdict must not depend
    on the data block — if a later change makes it config-resolved, that is a
    separate issue with its own blast-radius measurement.
    """
    checker = ConfigHealthChecker()
    for dataset_type in ("m4raw", "kspace", "image"):
        for single_contrast in (True, False):
            cfg = types.SimpleNamespace(data=_data(dataset_type, single_contrast))
            assert checker._is_paired_contrast(cfg, "universal_multitask_dual")


def test_paired_contrast_set_has_exactly_one_reader() -> None:
    """NN17: elect one owner and delete the loser's enforcement.

    Three call sites each tested ``_PAIRED_CONTRAST_MODELS`` directly, at three
    different strengths (one carried ``single_contrast``, two carried nothing).
    That divergence is what let a *narrowing* fix land and stay inert. The set
    is now read only inside ``_is_paired_contrast``; a second reader would
    re-open the same drift.
    """
    src = inspect.getsource(ConfigHealthChecker)
    readers = [
        stripped
        for line in src.splitlines()
        if (stripped := line.strip())
        and "self._PAIRED_CONTRAST_MODELS" in stripped
        and not stripped.startswith("#")
    ]
    assert len(readers) == 1, (
        "expected exactly one reader (inside _is_paired_contrast), found:\n"
        + "\n".join(readers)
    )
    assert "model_type not in self._PAIRED_CONTRAST_MODELS" in readers[0]


def test_adapter_chain_skips_a_genuinely_paired_arm() -> None:
    """Polarity guard: the narrowing must not delete the exemption outright."""
    checker = ConfigHealthChecker()
    step = types.SimpleNamespace(name="magnitude")
    cfg = types.SimpleNamespace(
        model=_model("kspace_cold_diffusion", {"backbone_type": "diff_varnet"}),
        data=_data("m4raw", False),
        adapters=types.SimpleNamespace(pre_model=[step]),
    )
    result = checker.check_adapter_chain_channel_resolution(cfg)
    assert "static pre_model channel resolution skipped" in result.message
