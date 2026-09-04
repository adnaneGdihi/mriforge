"""Schema-load + scientific-metadata gate for MICCAI MRIxFields2026 arms.

Each arm must (a) load under the canonical schema and (b) declare the
anti-facade scientific metadata (hypothesis / baseline / primary_metric;
pitfalls #17-#19).

``baselines/`` is a category apart and is exempted per-policy rather than
wholesale. CUT / CycleGAN / StarGAN-v2 are external reference implementations,
not variants of anything in this cohort: they declare a purely adversarial
``losses.gan`` objective, no ``losses.reconstruction`` block, and no
``metadata.baseline`` -- they ARE the baseline. Applying treatment-arm policy to
them cost job 8000966 ten failures. The exemptions below are each paired with a
positive assertion so they cannot become a loophole, and they follow the one
this module already had (``test_arm_optimizer_momentum_and_regularization``
skips baselines for their canonical GAN betas).
"""

from __future__ import annotations

import pathlib

import pytest

from spectramr.config.settings import TrainingSettings
from spectramr.config.training_budget import NO_TRAINING_MODES
from tests.utils.corpus import tracked_yamls

# Resolved from this file, not the CWD: a relative path silently yields an empty
# ARMS list when pytest is invoked from anywhere but the repo root, and an empty
# parametrisation is a vacuous pass, not an error.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ARM_DIR = _REPO_ROOT / "experiments" / "inprogress" / "mrixfields2026"
ARMS = tracked_yamls(_ARM_DIR)


def _is_baseline(arm: pathlib.Path) -> bool:
    return arm.stem.startswith("baseline_")


def test_arms_exist() -> None:
    assert ARMS, f"no MRIxFields2026 arms found under {_ARM_DIR}"


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_arm_loads_and_declares_metadata(arm: pathlib.Path) -> None:
    cfg = TrainingSettings.from_yaml(str(arm))
    meta = cfg.metadata
    get = meta.get if isinstance(meta, dict) else lambda k: getattr(meta, k, None)
    required = ["hypothesis", "primary_metric"]
    # A baseline has no baseline -- it IS the reference. Pitfall #17 asks a
    # VARIANT to name what it is measured against; demanding it of CUT /
    # CycleGAN / StarGAN-v2 is a category error, and the only way to satisfy it
    # would be to invent a comparison that does not exist.
    if not _is_baseline(arm):
        required.append("baseline")
    for key in required:
        assert get(key), f"{arm.name} missing metadata.{key}"


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_arm_disables_spatial_loss_warmup(arm: pathlib.Path) -> None:
    """Every arm is image-domain, where the k-space-stabilisation spatial-loss
    warmup (default 1000) masks the only configured loss at iteration 0 and trips
    the Tier-2 dead-loss probe. Production-readiness pass (2026-06-28) set
    ``losses.reconstruction.warmup_iterations: 0`` on all 56 arms; this guards it
    against regression / a newly-added arm forgetting it.

    The premise is "the only configured loss". The five ``baselines/`` arms
    configure no reconstruction term at all -- ``losses.gan`` only, and CUT's own
    hypothesis says "no pixel-L1 to target" -- so their schema-default warmup
    masks nothing. They are held to the positive form instead: an arm without a
    reconstruction loss must actually declare an adversarial one, so "no
    reconstruction block" can never quietly mean "no objective".
    """
    cfg = TrainingSettings.from_yaml(str(arm))
    recon = getattr(getattr(cfg, "losses", None), "reconstruction", None)

    # Gate on the CATEGORY, not on "is it adversarial". `_ADVERSARIAL_TASK_ARMS`
    # holds two adversarial *treatment* arms (field_cocycle_*): keying the
    # exemption off `enable_adversarial` would silently exempt them the day one
    # stopped declaring a reconstruction loss, which is the loophole this
    # exemption is supposed to not be.
    if _is_baseline(arm):
        assert recon is not None and not recon.model_fields_set, (
            f"{arm.name}: now declares losses.reconstruction "
            f"({sorted(recon.model_fields_set)}) — the baseline exemption below "
            f"assumes it has none. Hold it to warmup_iterations: 0 instead."
        )
        gan = getattr(getattr(cfg, "losses", None), "gan", None)
        assert gan is not None and gan.enable_adversarial, (
            f"{arm.name}: declares neither a reconstruction loss nor an "
            f"adversarial one — it would train on the schema defaults alone"
        )
        return

    warmup = getattr(recon, "warmup_iterations", None)
    assert warmup == 0, (
        f"{arm.name}: losses.reconstruction.warmup_iterations must be 0 "
        f"(image-domain arm; nonzero warmup masks the only loss at iter 0 → "
        f"dead-loss probe FP), got {warmup!r}"
    )


def _image_loss_names(cfg) -> list[str]:
    losses = getattr(cfg, "losses", None)
    image = getattr(losses, "image_losses", None) or []
    out = []
    for item in image:
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        if name:
            out.append(name)
    return out


# Optimization direction per headline metric (mirrors the wiring rule in
# scripts/data wiring + core/metrics/metric_directions). ssim/dice => maximize;
# gaussian_nll (a loss) and synthseg_dice_risk (a risk) => minimize.
_METRIC_MODE = {
    "ssim": "max",
    "synthseg_dice": "max",
    "synthseg_dice_risk": "min",
    "gaussian_nll": "min",
}


def _emitted_metrics(cfg) -> set[str]:
    """Metric names the run actually computes at validation time.

    Reads `validation.scoring.compute` DIRECTLY. The nested-getattr spelling
    this replaces named `validation.metrics`, which the block decomposition
    moved -- and its `None` default turned the miss into an empty set, so every
    caller asserted `'<metric>' in set()` and blamed the arm for a metric it
    declares. A sentinel here is worse than an AttributeError: it makes the
    helper answer confidently and wrongly.
    """
    vm = cfg.validation.scoring.compute
    if isinstance(vm, dict):
        return {k for k, v in vm.items() if v}
    return set(vm or [])


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_arm_has_metric_driven_best_checkpoint(arm: pathlib.Path) -> None:
    # Regression for the cohort-wide 2026-06-24 fix. checkpoint_best.pt is written
    # ONLY inside the early-stopping branch of pipelines/training_loop.py (~L1061),
    # keyed to early_stopping.metric — NOT metrics.best_metric_name (inert outside
    # that path) and NOT validation.primary_metric (report-cases only). Without an
    # early_stopping block these arms had NO metric-selected best checkpoint at all.
    # Each arm (treatment AND its _ablate_ sibling) must now track its OWN headline
    # metric so the comparison stays one-knob (anti-confound, pitfall #17).
    cfg = TrainingSettings.from_yaml(str(arm))
    mode = str(
        getattr(cfg.training, "training_mode", "")
        or getattr(cfg.training, "strategy_class", "")
        or ""
    )
    if mode in NO_TRAINING_MODES:
        pytest.skip(f"{arm.name}: {mode} optimises nothing, so no checkpoint is selected")
    es = cfg.early_stopping
    assert es is not None and es.enabled, f"{arm.name}: early_stopping must be enabled"
    # Early stopping must be a REAL saturation guard: it should FIRE on a genuine
    # ~20k-iter val plateau (clearing the ~20k-iter loss-curriculum window) yet not
    # clip a run mid-curriculum. Bound the WALL-TIME window (patience x eval_interval),
    # which also absorbs the baseline eval_interval=2000 asymmetry so one number means
    # one wall-time everywhere. Supersedes the old patience>=20000 shim that disabled
    # stopping entirely — that contradicted the intent that ES halt saturation
    # (2026-07-07). checkpoint_best.pt is still written on every improvement regardless
    # of patience, so metric-driven best-checkpoint selection is unaffected.
    # `validation.eval_interval` moved to `validation.schedule.interval_steps`
    # (RENAMES, 2026-07-31, posture `fold`). Read flat through a defaulting
    # getattr it resolved to 1000 on EVERY arm -- `hasattr(cfg.validation,
    # "eval_interval")` is False -- so this guard has been measuring
    # `patience x 1000`, i.e. patience, not the wall-time window it documents.
    # Measured across the cohort: under the stale read 5 arms scored 10000 and
    # 72 scored 20000; under the canonical read all 77 score exactly 20000. The
    # five "failures" were the baselines spelling the same window as
    # patience 10 x 2000 instead of 20 x 1000 -- precisely the asymmetry the
    # comment above says this bound exists to absorb.
    eval_interval = cfg.validation.schedule.interval_steps
    window = es.patience * eval_interval
    assert 15_000 <= window <= 45_000, (
        f"{arm.name}: early-stop window {window} iters (patience {es.patience} x "
        f"eval_interval {eval_interval}) — want ~20k to fire on saturation, not disabled"
    )
    assert es.restore_best_weights, f"{arm.name}: must restore best weights at end"

    meta = cfg.metadata
    get = meta.get if isinstance(meta, dict) else lambda k: getattr(meta, k, None)
    primary = get("primary_metric")
    assert es.metric == f"val_{primary}", (
        f"{arm.name}: monitored metric {es.metric!r} must track the headline "
        f"primary_metric {primary!r} (val_-prefixed)"
    )
    # The monitored metric must actually be emitted, else the F-EARLYSTOP resolver
    # silently misses and no best checkpoint is ever saved (a NEW facade).
    assert primary in _emitted_metrics(cfg), (
        f"{arm.name}: monitored {primary!r} not in validation.metrics — would never resolve"
    )
    mode = str(es.mode).split(".")[-1].lower()
    assert mode == _METRIC_MODE[primary], (
        f"{arm.name}: mode {mode!r} wrong for {primary!r} (expected {_METRIC_MODE[primary]})"
    )


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_arm_wires_val_manifest_and_metric_tracking(arm: pathlib.Path) -> None:
    """Every arm consumes the ordinal-paired held-out validation manifest, treats the
    volumes as pre-normalised, and tracks the core full-reference battery in BOTH
    training and validation. Anchor: the mrixfields2026_val.json wiring, 2026-07-06.
    Guards against a new arm regressing to an internal random split, a double
    normalisation, or validation-only metrics."""
    cfg = TrainingSettings.from_yaml(str(arm))
    data = cfg.data
    assert data.source.validation_index_path == ("data/manifests/mrixfields2026_val.json"), (
        f"{arm.name}: must validate on the held-out val manifest, not an internal split"
    )
    # Pre-normalised volumes: re-normalising would shift the intensity distribution.
    assert data.processing.normalization_type == "none", (
        f"{arm.name}: pre-normalised volumes must use normalization_type=none, "
        f"got {data.processing.normalization_type!r}"
    )
    # Training-time metric tracking (top-level metrics block drives _compute_training_metrics).
    m = cfg.metrics
    assert m is not None and m.enable_tracking, f"{arm.name}: metrics.enable_tracking off"
    for flag in ("compute_ssim", "compute_psnr", "compute_nrmse", "compute_lpips"):
        assert getattr(m, flag) is True, f"{arm.name}: metrics.{flag} must be true"
    # Same core battery emitted at validation (train/val symmetry the user asked for).
    emitted = _emitted_metrics(cfg)
    for core in ("ssim", "psnr", "nrmse", "lpips"):
        assert core in emitted, (
            f"{arm.name}: validation.metrics missing {core!r} (got {sorted(emitted)})"
        )


# Adam beta1 policy (2026-07-07 knob sweep): non-adversarial arms use 0.9 (matches the
# v6.x reference template); the two field_cocycle adversarial arms keep the low-momentum
# GAN recipe 0.5. Baselines carry their own canonical GAN betas (cut/cyclegan 0.5,
# stargan 0.0) and are exempt. b17_dice_risk_calibration takes no gradient steps.
_ADVERSARIAL_TASK_ARMS = {"field_cocycle_anyfield", "field_cocycle_ablate_cocycle"}
# weight_decay 0.05 over-regularizes the parameter-efficient / spectral families; these
# 8 arms drop to 0.01 (whole families, so each ablation matches its treatment — #17).
_LOW_WD_ARMS = {
    "b16_field_fno",
    "b16_ablate_spectral",
    "b28_field_inr",
    "b28_field_inr_augmented",
    "b28_ablate_field",
    "b36_lora_modulation",
    "b36_ablate_rank0",
    "b36_ablate_param_matched",
}


@pytest.mark.parametrize(
    "arm",
    [a for a in ARMS if a.parent.name == "task2"],
    ids=lambda p: p.name,
)
def test_task2_source_is_ulf(arm: pathlib.Path) -> None:
    """MRIxFields2026 Task 2 is ULF restoration: the deliverable SOURCE is 0.1T and the
    higher fields are the target. Every task2 arm must therefore pair under a 0.1T-source
    policy — ``ulf_source`` (source pinned to 0.1T) or ``prior`` (0.1T→HF on the paired
    val manifest). An arm whose mechanism needs a VARYING source (the B-2.7 field_wiener
    family) is Task 3, not Task 2, and lives under task3/ (reclassified 2026-07-07)."""
    cfg = TrainingSettings.from_yaml(str(arm))
    policy = cfg.data.mrixfields.pairing_policy
    assert policy in {"ulf_source", "prior"}, (
        f"{arm.name}: task2 source must be 0.1T, but pairing is {policy!r} (not "
        f"ulf_source/prior). An all_pairs (any-to-any) arm is Task 3 — move it to task3/."
    )


@pytest.mark.parametrize(
    "arm",
    [a for a in ARMS if a.parent.name == "task3"],
    ids=lambda p: p.name,
)
def test_task3_is_any_to_any(arm: pathlib.Path) -> None:
    """MRIxFields2026 Task 3 is any-to-any cross-field translation: every task3 arm pairs
    under ``all_pairs`` (the source field varies). Guards the b27 field_wiener
    reclassification (2026-07-07) and any future task-misfiling — a 0.1T-pinned
    ``ulf_source`` arm belongs in task2/, not here."""
    cfg = TrainingSettings.from_yaml(str(arm))
    policy = cfg.data.mrixfields.pairing_policy
    assert policy == "all_pairs", (
        f"{arm.name}: task3 is any-to-any, so pairing must be all_pairs, got {policy!r}. "
        f"A 0.1T-source (ulf_source/prior) arm is Task 2 — move it to task2/."
    )


# Strategies VERIFIED to thread the per-sample contrast_id from the batch all the way
# to the model forward (both training and validation). The SSOT of "contrast-wired"
# now lives in src (``config.validation_constants.CONTRAST_THREADED_STRATEGIES``) so the
# cluster-side audit guard and this repo-side per-arm guard agree by construction. An arm
# that turns the knob on (model_kwargs.use_contrast_conditioning) while its strategy is
# NOT in the set raises at step 0 on the cluster ("batch carries no 'contrast_id'") — the
# exact regression that shipped when 26 arms were enabled but only 8 strategies threaded
# (2026-07-07). Adding a new contrast-enabled strategy requires threading it AND listing
# it in the src SSOT.
from spectramr.config.validation_constants import (  # noqa: E402
    CONTRAST_THREADED_STRATEGIES as _CONTRAST_THREADED_STRATEGIES,
)


def _use_contrast_conditioning(cfg) -> bool:
    mk = getattr(getattr(cfg, "model", None), "model_kwargs", None)
    if mk is None:
        return False
    get = mk.get if isinstance(mk, dict) else lambda k, d=None: getattr(mk, k, d)
    return bool(get("use_contrast_conditioning", False))


def test_contrast_rollout_is_non_trivial() -> None:
    # Guard the parametrized check below against vacuously passing: the broad rollout
    # (2026-07-07) enabled contrast on ~26 arms. If this collapses to near-zero, the
    # per-arm guard is inert and a threading regression could slip through.
    enabled = [
        a.name for a in ARMS if _use_contrast_conditioning(TrainingSettings.from_yaml(str(a)))
    ]
    assert len(enabled) >= 20, (
        f"expected the broad contrast rollout (~26 arms), found {len(enabled)}: {enabled}"
    )


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_contrast_enabled_arm_uses_a_threaded_strategy(arm: pathlib.Path) -> None:
    """A contrast-enabled arm's strategy MUST forward contrast_id to the model.

    The mrixfields dataset always emits ``contrast_id``; a widened field-FiLM model with
    ``use_contrast_conditioning: true`` RAISES if the strategy drops it (#15, no silent
    mode-averaging). This pairs the YAML knob with the strategy that honours it, so a new
    contrast-enabled arm on an unthreaded strategy fails here (a unit test) instead of at
    step 0 on the cluster."""
    cfg = TrainingSettings.from_yaml(str(arm))
    if not _use_contrast_conditioning(cfg):
        pytest.skip("contrast conditioning off for this arm")
    strat = getattr(cfg.training, "strategy_class", None)
    assert strat in _CONTRAST_THREADED_STRATEGIES, (
        f"{arm.name}: use_contrast_conditioning=true but strategy {strat!r} does not "
        f"thread contrast_id to the model forward. Either thread it (mirror "
        f"field_flow_strategy) and add {strat!r} to _CONTRAST_THREADED_STRATEGIES, or "
        f"disable the knob for this arm."
    )


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_arm_optimizer_momentum_and_regularization(arm: pathlib.Path) -> None:
    """Adam beta1 + weight_decay policy from the 2026-07-07 knob review. Guards against
    a new arm silently inheriting the schema's GAN-specific beta1=0.5 default on
    non-adversarial training, or shipping the over-regularizing weight_decay=0.05 on a
    parameter-efficient (INR / LoRA / FNO) arm."""
    stem = arm.stem
    if stem.startswith("baseline_"):
        pytest.skip("baselines carry their own canonical GAN betas / weight decay")
    if stem == "b17_dice_risk_calibration":
        pytest.skip("zero-gradient conformal certificate pass; optimizer knobs inert")
    cfg = TrainingSettings.from_yaml(str(arm))
    opt = cfg.optimization
    expected_beta1 = 0.5 if stem in _ADVERSARIAL_TASK_ARMS else 0.9
    assert opt.optimizer.beta1 == pytest.approx(expected_beta1), (
        f"{arm.name}: beta1 {opt.optimizer.beta1} != {expected_beta1} "
        f"({'adversarial GAN recipe' if expected_beta1 == 0.5 else 'non-GAN momentum'})"
    )
    if stem in _LOW_WD_ARMS:
        assert opt.optimizer.weight_decay == pytest.approx(0.01), (
            f"{arm.name}: parameter-efficient arm must use weight_decay 0.01 "
            f"(0.05 over-regularizes), got {opt.optimizer.weight_decay}"
        )
