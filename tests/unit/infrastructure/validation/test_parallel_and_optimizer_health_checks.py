"""Tier-0/1 checks for the parallel, compile and optimizer blocks.

These exist so a misconfigured arm fails in ~100 ms at audit time rather than
after the whole training environment has been built on a cluster node -- which
is where every one of these used to surface, if it surfaced at all.

Severity is chosen per the documented precedent (``check_workflow_declared`` is
advisory *because* erroring would redden the whole corpus overnight): a
missing dependency or an arithmetically-impossible precision is an ``error``, a
defensible-but-probably-wrong choice is a ``warning``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mriforge.config.schemas.base import ParallelismConfigSchema
from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker
from tests.utils.config_block_stub import block_stub


def _checker() -> ConfigHealthChecker:
    """Construct without running the heavy __init__ (house style in this dir)."""
    return ConfigHealthChecker.__new__(ConfigHealthChecker)


def _parallel(strategy: str = "none", **ds):
    if strategy == "deepspeed":
        return ParallelismConfigSchema(
            strategy="deepspeed", deepspeed={"enabled": True, **ds}
        )
    if strategy == "fsdp":
        return ParallelismConfigSchema(strategy="fsdp", fsdp={"enabled": True})
    return ParallelismConfigSchema(strategy=strategy)


def _config(
    *,
    strategy="deepspeed",
    use_amp=True,
    amp_dtype=None,
    target_domain="image",
    kspace_recon=False,
    optimizer_type="adamw",
    learning_rate=1e-4,
    compile_model=False,
    **ds,
):  # noqa: D103
    return SimpleNamespace(
        parallel=_parallel(strategy, **ds),
        # `optimization:` is decomposed (optimizer/gradient/precision/compile);
        # these flat kwargs are routed to their canonical homes by the shared
        # stub so the reader walks the same path a real config produces.
        optimization=block_stub(
            "optimization",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            compile_model=compile_model,
        ),
        model=SimpleNamespace(target_domain=target_domain),
        physics=SimpleNamespace(
            kspace=SimpleNamespace(enable_kspace_recon=kspace_recon)
        ),
    )


class TestDeepSpeedExtraInstalled:
    def test_not_deepspeed_is_not_applicable(self) -> None:
        result = _checker().check_deepspeed_extra_installed(_config(strategy="ddp"))
        assert result.passed and result.severity == "info"

    def test_verdict_tracks_the_real_environment(self) -> None:
        """Asserted against ``find_spec``, not a hardcoded expectation, so this
        is meaningful whether or not the extra happens to be installed."""
        import importlib.util

        result = _checker().check_deepspeed_extra_installed(_config())
        assert result.passed is (importlib.util.find_spec("deepspeed") is not None)

    def test_absent_extra_is_an_error_with_the_install_command(
        self, monkeypatch
    ) -> None:
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _n: None)
        result = _checker().check_deepspeed_extra_installed(_config())
        assert not result.passed and result.severity == "error"
        assert "pip install" in (result.fix_hint or "")


class TestDeepSpeedPrecisionCoherent:
    """fp16 + complex/k-space is arithmetically impossible, not merely unwise.

    ``get_autocast_context`` disables autocast for complex+fp16 because there is
    no ``complex16``. DeepSpeed casts module weights to half from INSIDE the
    engine, where that guard cannot see it -- so the one arm class the guard
    exists to protect is exactly the one it stops protecting.
    """

    def test_fp16_on_a_kspace_arm_is_an_error(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(amp_dtype=None, target_domain="kspace")
        )
        assert not result.passed and result.severity == "error"

    def test_explicit_float16_is_caught_too(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(amp_dtype="float16", target_domain="kspace")
        )
        assert not result.passed

    def test_kspace_recon_flag_also_triggers_it(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(amp_dtype=None, target_domain="image", kspace_recon=True)
        )
        assert not result.passed

    def test_bfloat16_is_fine_on_the_same_arm(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(amp_dtype="bfloat16", target_domain="kspace")
        )
        assert result.passed

    def test_fp16_on_an_image_arm_is_fine(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(amp_dtype=None, target_domain="image")
        )
        assert result.passed

    def test_amp_off_is_fine(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(use_amp=False, target_domain="kspace")
        )
        assert result.passed

    def test_not_deepspeed_is_not_applicable(self) -> None:
        result = _checker().check_deepspeed_precision_coherent(
            _config(strategy="ddp", target_domain="kspace")
        )
        assert result.passed and result.severity == "info"


class TestDeepSpeedConsolidatedBestCheckpoint:
    def test_disabling_it_warns_that_the_run_becomes_resume_only(self) -> None:
        """DeepSpeed writes a sharded tag DIRECTORY. Without the consolidated
        copy, discover_best_checkpoint / campaign eval / `mriforge infer` find
        nothing -- at the END of the run."""
        result = _checker().check_deepspeed_consolidated_best_checkpoint(
            _config(save_consolidated_best=False)
        )
        assert not result.passed and result.severity == "warning"

    def test_the_default_passes(self) -> None:
        assert _checker().check_deepspeed_consolidated_best_checkpoint(_config()).passed


class TestLionLearningRateScale:
    """Lion's update is a SIGN, so every parameter moves by exactly ``lr``.

    An AdamW-scale rate does not train worse -- it diverges.
    """

    def test_adamw_scale_lr_warns(self) -> None:
        result = _checker().check_lion_learning_rate_scale(
            _config(optimizer_type="lion", learning_rate=1e-3)
        )
        assert not result.passed and result.severity == "warning"

    def test_a_lion_scale_lr_passes(self) -> None:
        assert (
            _checker()
            .check_lion_learning_rate_scale(
                _config(optimizer_type="lion", learning_rate=1e-4)
            )
            .passed
        )

    def test_other_optimizers_are_not_applicable(self) -> None:
        """The same LR is perfectly normal for AdamW; only Lion is at risk."""
        result = _checker().check_lion_learning_rate_scale(
            _config(optimizer_type="adamw", learning_rate=1e-3)
        )
        assert result.passed and result.severity == "info"

    def test_it_is_a_warning_not_an_error(self) -> None:
        """The useful range is a rule of thumb, and an arm may be probing it."""
        result = _checker().check_lion_learning_rate_scale(
            _config(optimizer_type="lion", learning_rate=1.0)
        )
        assert result.severity == "warning"


class TestCompileWithShardedStrategy:
    @pytest.mark.parametrize("strategy", ["fsdp", "deepspeed"])
    def test_compile_plus_sharding_warns(self, strategy: str) -> None:
        """ModelBuilder.compile() runs BEFORE the sharding wrap, giving
        FSDP(torch.compile(m)) -- the reverse of the recommended order."""
        result = _checker().check_compile_with_sharded_strategy(
            _config(strategy=strategy, compile_model=True)
        )
        assert not result.passed and result.severity == "warning"

    def test_compile_without_sharding_is_fine(self) -> None:
        assert (
            _checker()
            .check_compile_with_sharded_strategy(
                _config(strategy="ddp", compile_model=True)
            )
            .passed
        )

    def test_sharding_without_compile_is_fine(self) -> None:
        assert (
            _checker()
            .check_compile_with_sharded_strategy(
                _config(strategy="fsdp", compile_model=False)
            )
            .passed
        )


class TestDeepSpeedTopologyCoherent:
    """Offloading state a stage never partitions is inert, not merely odd.

    The arm advertises CPU offload as its memory story, pays none of the
    transfer cost, and gets none of the saving -- a facade in the pitfall-#16
    sense, at the topology layer.
    """

    def test_optimizer_offload_at_stage_zero_warns(self) -> None:
        result = _checker().check_deepspeed_topology_coherent(
            _config(zero_stage=0, offload_optimizer="cpu")
        )
        assert not result.passed and result.severity == "warning"

    def test_param_offload_below_stage_three_warns(self) -> None:
        """``offload_param`` only means anything once params are partitioned."""
        result = _checker().check_deepspeed_topology_coherent(
            _config(zero_stage=2, offload_param="cpu")
        )
        assert not result.passed and result.severity == "warning"

    def test_param_offload_at_stage_three_is_fine(self) -> None:
        assert (
            _checker()
            .check_deepspeed_topology_coherent(
                _config(zero_stage=3, offload_param="cpu", offload_optimizer="cpu")
            )
            .passed
        )

    def test_optimizer_offload_at_stage_two_is_fine(self) -> None:
        assert (
            _checker()
            .check_deepspeed_topology_coherent(
                _config(zero_stage=2, offload_optimizer="cpu")
            )
            .passed
        )

    def test_no_offload_is_always_coherent(self) -> None:
        assert (
            _checker().check_deepspeed_topology_coherent(_config(zero_stage=0)).passed
        )

    def test_not_deepspeed_is_not_applicable(self) -> None:
        result = _checker().check_deepspeed_topology_coherent(_config(strategy="ddp"))
        assert result.passed and result.severity == "info"


class TestOptimizerRegistered:
    """The check exists for the MISDIAGNOSIS, not the typo.

    A ``@register_optimizer`` in a module nothing imports is dead, and the only
    symptom is the name vanishing from ``list_available()`` -- so the runtime
    says "unknown optimizer", which reads as a YAML typo and sends the user to
    edit a config that was correct.
    """

    def test_a_registered_name_passes(self) -> None:
        assert (
            _checker()
            .check_optimizer_registered(_config(optimizer_type="adamw"))
            .passed
        )

    def test_an_in_repo_optimizer_is_registered(self) -> None:
        """lars/lamb/lion were enum members with no implementation anywhere."""
        for name in ("lars", "lamb", "lion"):
            assert (
                _checker()
                .check_optimizer_registered(_config(optimizer_type=name))
                .passed
            ), name

    def test_an_unregistered_name_is_an_error(self) -> None:
        result = _checker().check_optimizer_registered(
            _config(optimizer_type="definitely_not_an_optimizer")
        )
        assert not result.passed and result.severity == "error"

    def test_the_fix_hint_points_at_the_import_not_the_yaml(self) -> None:
        """The whole point: send the reader to __init__.py, not to their config."""
        result = _checker().check_optimizer_registered(
            _config(optimizer_type="definitely_not_an_optimizer")
        )
        assert "__init__.py" in (result.fix_hint or "")

    def test_case_is_normalised(self) -> None:
        assert (
            _checker()
            .check_optimizer_registered(_config(optimizer_type="AdamW"))
            .passed
        )


class TestEveryCheckIsActuallyWired:
    """A ``check_*`` that ``run_all_checks`` never calls protects nothing.

    ``meta.health_checker_no_orphan_checks`` enforces this repo-wide; this states
    it locally for the seven added here, so the failure names them directly.
    """

    ADDED = (
        "check_deepspeed_extra_installed",
        "check_deepspeed_precision_coherent",
        "check_deepspeed_topology_coherent",
        "check_deepspeed_consolidated_best_checkpoint",
        "check_optimizer_registered",
        "check_lion_learning_rate_scale",
        "check_compile_with_sharded_strategy",
    )

    @pytest.mark.parametrize("name", ADDED)
    def test_is_invoked_by_run_all_checks(self, name: str) -> None:
        from mriforge.infrastructure.validation.witness.checks.meta_orphan_checks import (
            invoked_check_methods,
        )

        assert name in invoked_check_methods()

    @pytest.mark.parametrize("name", ADDED)
    def test_check_name_drops_the_check_prefix(self, name: str) -> None:
        """The --json payload and the corpus fixtures key off that string."""
        result = getattr(_checker(), name)(_config(strategy="none"))
        assert result.check_name == name.removeprefix("check_")


def _ds_config(
    *,
    zero_stage=3,
    offload_optimizer="cpu",
    accum=4,
    compile_model=False,
    zenflow=None,
    compile_block=None,
):
    """A deepspeed-strategy config with the feature blocks wired in."""
    ds = {
        "enabled": True,
        "zero_stage": zero_stage,
        "offload_optimizer": offload_optimizer,
    }
    if zenflow is not None:
        ds["zenflow"] = zenflow
    if compile_block is not None:
        ds["compile"] = compile_block
    return SimpleNamespace(
        parallel=ParallelismConfigSchema(strategy="deepspeed", deepspeed=ds),
        optimization=block_stub(
            "optimization",
            use_amp=True,
            amp_dtype="bfloat16",
            optimizer_type="adamw",
            learning_rate=1e-4,
            compile_model=compile_model,
            gradient_accumulation_steps=accum,
        ),
        model=SimpleNamespace(target_domain="image"),
        physics=SimpleNamespace(kspace=SimpleNamespace(enable_kspace_recon=False)),
    )


class TestZenFlowAccumulationConflict:
    """ZenFlow REPLACES gradient_accumulation_steps and nothing downstream knows.

    ``configure_zenflow`` ends with
    ``engine._config.gradient_accumulation_steps = engine.update_interval``, so
    an arm declaring 4 with update_interval 16 trains at a 4x larger effective
    batch while provenance, the run banner and effective_batch_size all report 4.
    """

    _ZF = {"enabled": True, "select_strategy": "step", "select_interval": 32}

    def test_a_disagreement_is_an_error(self) -> None:
        result = _checker().check_zenflow_accumulation_conflict(
            _ds_config(accum=4, zenflow={**self._ZF, "update_interval": 16})
        )
        assert not result.passed and result.severity == "error"

    def test_agreement_passes(self) -> None:
        assert (
            _checker()
            .check_zenflow_accumulation_conflict(
                _ds_config(accum=16, zenflow={**self._ZF, "update_interval": 16})
            )
            .passed
        )

    def test_auto_update_interval_with_a_declared_accumulation_errors(self) -> None:
        """'auto' makes the engine choose, discarding the declared value."""
        result = _checker().check_zenflow_accumulation_conflict(
            _ds_config(accum=8, zenflow={"enabled": True})
        )
        assert not result.passed and result.severity == "error"

    def test_auto_with_accumulation_one_is_fine(self) -> None:
        assert (
            _checker()
            .check_zenflow_accumulation_conflict(
                _ds_config(accum=1, zenflow={"enabled": True})
            )
            .passed
        )

    def test_the_fix_hint_names_the_number_to_use(self) -> None:
        result = _checker().check_zenflow_accumulation_conflict(
            _ds_config(accum=4, zenflow={**self._ZF, "update_interval": 16})
        )
        assert "16" in (result.fix_hint or "")

    def test_zenflow_off_is_not_applicable(self) -> None:
        result = _checker().check_zenflow_accumulation_conflict(_ds_config(accum=4))
        assert result.passed and result.severity == "info"

    def test_not_deepspeed_is_not_applicable(self) -> None:
        result = _checker().check_zenflow_accumulation_conflict(_config(strategy="ddp"))
        assert result.passed and result.severity == "info"


class TestDeepCompileSupported:
    def test_stacking_with_torch_compile_is_an_error(self) -> None:
        """ModelBuilder.compile() runs BEFORE deepspeed.initialize, so
        DeepCompile would be handed an already-compiled module and could not
        rewrite the ZeRO collectives -- the only thing it exists for."""
        result = _checker().check_deepcompile_supported(
            _ds_config(compile_model=True, compile_block={"enabled": True})
        )
        assert not result.passed and result.severity == "error"
        # The hint names the CANONICAL key. It used to say `compile_model`,
        # which is the retired spelling -- a reader who copied it would set
        # a key the schema folds, not the one this check reads.
        assert "optimization.compile.enabled" in (result.fix_hint or "")

    def test_deepcompile_alone_tracks_the_real_environment(self) -> None:
        """Asserted against is_deepcompile_supported(), not a hardcoded verdict,
        so this stays meaningful on a CPU CI box and on a CUDA node alike."""
        pytest.importorskip("deepspeed")
        from deepspeed.compile.util import is_deepcompile_supported

        result = _checker().check_deepcompile_supported(
            _ds_config(compile_block={"enabled": True})
        )
        assert result.passed is bool(is_deepcompile_supported())

    def test_an_unsupported_stack_is_an_error(self, monkeypatch) -> None:
        pytest.importorskip("deepspeed")
        import deepspeed.compile.util as util

        monkeypatch.setattr(util, "is_deepcompile_supported", lambda: False)
        result = _checker().check_deepcompile_supported(
            _ds_config(compile_block={"enabled": True})
        )
        assert not result.passed and result.severity == "error"

    def test_deepcompile_off_is_not_applicable(self) -> None:
        result = _checker().check_deepcompile_supported(_ds_config(compile_model=True))
        assert result.passed and result.severity == "info"

    def test_not_deepspeed_is_not_applicable(self) -> None:
        result = _checker().check_deepcompile_supported(_config(strategy="ddp"))
        assert result.passed and result.severity == "info"


def _diffusion_config(training_mode: str, *, use_amp=True, amp_dtype=None):
    """A config whose RESOLVED strategy is whatever ``training_mode`` dispatches to.

    `training` is a real attribute here (unlike ``_config`` above) because the
    check resolves the strategy class rather than matching a name.
    """
    cfg = _config(strategy="none", use_amp=use_amp, amp_dtype=amp_dtype)
    cfg.training = SimpleNamespace(
        training_mode=training_mode, strategy_class=None, diffusion=None
    )
    return cfg


class TestDiffusionPrecisionPolicy:
    """Diffusion arms train in fp32.

    This is not a hypothetical ratchet: 12 arms under ``experiments/inprogress/``
    resolved AMP ON while training a diffusion objective when it landed, and were
    invisible to a grep because the legacy ``optimization.use_amp`` folds onto
    ``optimization.precision.enabled``.
    """

    def test_fp16_on_a_diffusion_arm_is_an_error(self) -> None:
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("diffusion", amp_dtype="float16")
        )
        assert not result.passed and result.severity == "error"
        assert "float32" in (result.fix_hint or "")

    def test_bf16_is_refused_too(self) -> None:
        """The deliberate strictness. bf16 is the *safer* half-precision and is
        the recommended fix in ``check_deepspeed_precision_coherent`` -- but the
        policy here is fp32, and allowing one half-precision path would leave the
        constraint half-enforced."""
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("diffusion", amp_dtype="bfloat16")
        )
        assert not result.passed and result.severity == "error"

    def test_default_dtype_means_fp16_and_is_caught(self) -> None:
        """``amp_dtype=None`` resolves to fp16, so a bare ``use_amp: true`` is
        the most common way in -- and how most of the 12 real arms declared it."""
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("diffusion", amp_dtype=None)
        )
        assert not result.passed

    def test_explicit_float32_passes(self) -> None:
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("diffusion", use_amp=True, amp_dtype="float32")
        )
        assert result.passed and result.severity == "info"

    def test_amp_disabled_passes(self) -> None:
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("diffusion", use_amp=False)
        )
        assert result.passed

    def test_a_non_diffusion_arm_keeps_its_amp(self) -> None:
        """Scoped, not global: reconstruction arms are entitled to AMP and 34 in
        the corpus use it."""
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("reconstruction", amp_dtype="float16")
        )
        assert result.passed and result.severity == "info"

    def test_an_unresolvable_strategy_is_not_applicable(self) -> None:
        """An arm whose strategy cannot be resolved is a DIFFERENT check's
        finding; this one must not convert it into a precision error."""
        result = _checker().check_diffusion_precision_policy(
            _diffusion_config("no_such_training_mode_at_all", amp_dtype="float16")
        )
        assert result.passed and result.severity == "info"


class TestDiffusionPredicateBoundaries:
    """Pins WHY the predicate unions two signals.

    Each disjunct alone has a measured blind spot. These are the exact boundary
    cases; a refactor that drops one silently shrinks coverage from 170 arms to
    162 (or to 139) and these fail loudly instead.
    """

    @staticmethod
    def _is_diffusion(mode: str) -> bool:
        from mriforge.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker as C,
        )

        return C._resolves_to_diffusion_strategy(_diffusion_config(mode))

    @pytest.mark.parametrize(
        "mode",
        [
            # Caught ONLY by issubclass(DiffusionTrainingStrategy) -- no
            # "diffusion" anywhere in the class or module name.
            "edm",  # Elucidated Diffusion Models (Karras et al. 2022)
            "i2sb",
            "stochastic_interpolants",
            "flow_matching_pfode",
            "twin_dps",
            "bloch_schrodinger_bridge",
            # Caught ONLY by the qualname substring -- these inherit straight
            # from BaseTrainingStrategy, not from DiffusionTrainingStrategy.
            "cold_diffusion",
            "kspace_cold_diffusion",
            "x_diffusion",
            "riemannian_mrf_diffusion",
        ],
    )
    def test_known_diffusion_modes_are_caught(self, mode: str) -> None:
        assert self._is_diffusion(mode) is True

    @pytest.mark.parametrize("mode", ["reconstruction", "pnp", "gan", "vae"])
    def test_non_diffusion_modes_are_not(self, mode: str) -> None:
        """``pnp`` matters most: Plug-and-Play is diffusion-ADJACENT (it uses a
        denoiser prior) but subclasses ``ReconstructionTrainingStrategy`` and
        does not train a noise-prediction objective."""
        assert self._is_diffusion(mode) is False

    def test_neither_disjunct_alone_would_do(self) -> None:
        """The union is strictly larger than either half. If this ever stops
        holding, one disjunct has become redundant and should be deleted rather
        than left as decoration."""
        import importlib

        from mriforge.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )
        from mriforge.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
        )

        by_subclass, by_name = set(), set()
        for key, path in TrainingStrategyFactory.STRATEGY_CLASS_PATHS.items():
            module, _, cls_name = path.rpartition(".")
            try:
                cls = getattr(importlib.import_module(module), cls_name)
            except Exception:  # pragma: no cover - unimportable strategy
                continue
            if isinstance(cls, type) and issubclass(cls, DiffusionTrainingStrategy):
                by_subclass.add(key)
            if "diffusion" in f"{cls.__module__}.{cls.__name__}".lower():
                by_name.add(key)

        assert by_subclass - by_name, "issubclass adds nothing; drop it"
        assert by_name - by_subclass, "the substring adds nothing; drop it"


def _compile_config(*, compile_on=True, target_domain="image", kspace_recon=False):
    """``compile_model`` is routed to ``compile.enabled`` by the shared stub.

    Set through ``block_stub`` rather than assigned afterwards:
    ``CompileConfigSchema`` is frozen, so a post-hoc assignment raises.
    """
    cfg = _config(
        strategy="none",
        use_amp=False,
        target_domain=target_domain,
        kspace_recon=kspace_recon,
        compile_model=compile_on,
    )
    cfg.model = SimpleNamespace(model_type="unet", target_domain=target_domain)
    return cfg


class TestCompileWithComplexModel:
    """Inductor cannot codegen complex operators -- and does not say so loudly.

    Measured on torch 2.11 against this repo's own ``fft2c``/``ifft2c`` at
    complex64: compilation SUCCEEDS under ``fullgraph=True`` and is numerically
    correct (max abs error 3.2e-07). Inductor emits one UserWarning --
    "Torchinductor does not support code generation for complex operators" --
    and runs those operators eagerly.

    So the failure mode is not a crash, it is a FALSE THROUGHPUT CLAIM: the arm
    declares compiled, provenance stamps compiled, and the complex regions run
    eager, possibly slower than not compiling at all.
    """

    def test_kspace_target_domain_with_compile_is_an_error(self) -> None:
        result = _checker().check_compile_with_complex_model(
            _compile_config(target_domain="kspace")
        )
        assert not result.passed and result.severity == "error"
        assert "compile.enabled: false" in (result.fix_hint or "")

    def test_kspace_recon_flag_also_triggers_it(self) -> None:
        result = _checker().check_compile_with_complex_model(
            _compile_config(target_domain="image", kspace_recon=True)
        )
        assert not result.passed and result.severity == "error"

    def test_a_real_valued_arm_may_compile(self) -> None:
        """Scoped, not global. Compilation is a real win on real-valued models
        and the check must not become a blanket ban."""
        result = _checker().check_compile_with_complex_model(_compile_config())
        assert result.passed and result.severity == "info"

    def test_compile_off_is_not_applicable(self) -> None:
        result = _checker().check_compile_with_complex_model(
            _compile_config(compile_on=False, target_domain="kspace")
        )
        assert result.passed and result.severity == "info"

    def test_the_message_names_which_signal_fired(self) -> None:
        """Four signals can trigger this and they are not equally obvious; a bare
        'this arm is complex' would send the reader hunting."""
        result = _checker().check_compile_with_complex_model(
            _compile_config(target_domain="kspace", kspace_recon=True)
        )
        assert "model.target_domain" in result.message
        assert "physics.kspace.enable_kspace_recon" in result.message


class TestComplexIsAboutDtypeNotArithmetic:
    """``ComplexConv2d`` is NOT a signal, and that is a measured decision.

    It stores real and imaginary parts as separate REAL tensors and performs one
    fused real ``F.conv2d`` against a block weight matrix, returning float32. It
    compiles cleanly under ``fullgraph=True``. Treating "uses complex
    arithmetic" as "carries complex dtype" would have blocked compilation on
    arms Inductor handles perfectly.
    """

    def test_complex_conv2d_returns_a_real_dtype(self) -> None:
        from mriforge.models.layers.complex_conv import ComplexConv2d

        layer = ComplexConv2d(2, 4, kernel_size=3, padding=1)
        # Interleaved real/imag on the channel axis, so 2 * in_channels.
        out = layer(torch.randn(1, 4, 8, 8))
        assert not out.is_complex(), (
            "ComplexConv2d now returns a complex dtype; it is no longer safe to "
            "exclude it from the complex-arm signals in "
            "_complex_arm_signals -- Inductor cannot codegen complex operators."
        )

    def test_fft2c_does_carry_complex_dtype(self) -> None:
        """The counterpart: this is where the real complex64 lives, which is why
        the signals key on k-space rather than on the conv layer."""
        from mriforge.infrastructure.physics.fft_ops import fft2c

        assert fft2c(torch.randn(1, 1, 8, 8, dtype=torch.complex64)).is_complex()
