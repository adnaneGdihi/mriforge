.. _audit_ladder_user_guide:

==============================================================================
Audit Ladder — User Guide
==============================================================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Why this exists
===============

The most recent full smoke run across the maintainers' arm corpus
took 7 hours on a V100 and surfaced 31 failures across 90 experiments.
**24 of those 31 failures** were statically determinable at config-load
time — channel mismatches, AMP / GradScaler double-unscale, output-vs-
target shape mismatches, loss-domain misalignment. Discovering them in
runtime training is whack-a-mole; each fix takes minutes-to-hours of
GPU time to confirm.

The audit ladder catches them in seconds, before any training runs.

The 4 tiers
===========

Each tier is strictly faster than the next. The smoke wrapper sequences
them and stops at the first failure.

.. list-table::
   :header-rows: 1
   :widths: 8 22 14 56

   * - Tier
     - What it does
     - Cost / arm
     - Catches
   * - **0**
     - Pure Pydantic load.
     - ~ 1 ms
     - Typos, missing fields, illegal enum values, wrong types.
   * - **1**
     - Static cross-validation via
       :class:`~spectramr.infrastructure.validation.config_health_checker.ConfigHealthChecker`.
     - ~ 10–100 ms
     - Channel mismatch (model expects ≠ data provides), advertised-
       options violations (silent fallbacks), loss-domain misalignment,
       AMP + gradient-clipping interaction, registry lookup failures.
   * - **2**
     - Synthetic forward probe via
       :func:`~spectramr.infrastructure.validation.forward_probe.synthetic_forward_probe`.
     - ~ 10–30 s (CPU)
     - Shape mismatches inside the model (linear-layer mat1×mat2),
       AMP / GradScaler double-unscale (real backward pass triggers it),
       output-vs-target shape contracts on the loss path, OOM at the
       configured batch / patch size (with ``--device cuda``).
   * - **3**
     - Existing 10-iter smoke training on real data.
     - minutes–hours
     - Data-loader failures, HDF5 corruption, augmentation explosions,
       NaN gradients on real data, real OOM at full batch.

.. note::

   **Tier 2 builds the model training builds.** The probe resolves its
   constructor arguments through the same code the training path uses
   (:func:`~spectramr.infrastructure.builders.generator_kwargs.resolve_full_generator_kwargs`),
   so contract-gated SSOT injections reach the probed model: the
   ``undersampling`` block as ``acceleration_config``,
   ``data.processing.enable_log_scaling`` as ``kspace_log_scaled``, and the
   ``physics.data_consistency`` block as ``use_dc`` / ``dc_method`` /
   ``dc_weight``. Gradient checkpointing is applied too, so the probe's
   backward pass exercises the same recompute path a run will.

   Before this, the probe assembled its own constructor arguments and skipped
   those injections, so an arm could pass Tier 2 and still build a *different*
   model in training — the probe's verdict did not transfer. If an arm now
   fails Tier 2 with a config-conflict message it previously passed, the
   conflict was always there; only the detection is new.

Quick start
===========

Audit a single experiment:

.. code-block:: bash

   source .venv/bin/activate

   # Tier 0+1 only (~100 ms): structural + static cross-checks
   python -m spectramr.cli audit experiments/templates/comprehensive_config_template.yaml

   # Add Tier-2 synthetic forward probe (~30 s on CPU)
   python -m spectramr.cli audit experiments/templates/comprehensive_config_template.yaml --probe

   # Same, but on GPU (catches OOM at the configured batch size)
   python -m spectramr.cli audit experiments/templates/comprehensive_config_template.yaml --probe --device cuda

   # Structured JSON output for the campaign aggregator
   python -m spectramr.cli audit experiments/templates/comprehensive_config_template.yaml --json

Exit codes:

* ``0`` — all checks passed.
* ``1`` — warnings only (e.g. AMP + gradient-clipping interaction
  flagged, but no error).
* ``2`` — at least one error. The CLI prints / serialises every
  failure with a one-line ``fix_hint``.

Anatomy of a structured failure record
======================================

Every Tier 0–2 failure is a JSON-serialisable record with the same
schema, so the campaign aggregator and the post-hoc grouped-error
report can parse them uniformly:

.. code-block:: json

   {
     "passed": false,
     "check_name": "advertised_options",
     "message": "model.model_kwargs.attention_type='dual_domain' is not in the advertised set for 'spade_cold_diffusion': ['spatial', 'cross']. Silent fallbacks are forbidden.",
     "severity": "error",
     "category": "advertised_options",
     "yaml_keys": ["model.model_kwargs.attention_type"],
     "fix_hint": "Set model.model_kwargs.attention_type to one of ['cross', 'spatial'] OR add 'dual_domain' to spade_cold_diffusion.OPTION_SCHEMA['attention_type']."
   }

The ``category`` field is the grouping key used by the leaderboard to
recreate tables like the one in
:file:`tests_experiments/smoke_test/ERRORS_GROUPED_SUMMARY.md`
*automatically*, instead of hand-curated grep.

.. note::

   **2026-07-14 fix (no unfounded greens).** ``check_advertised_options`` used to
   append an unconditional ``passed=True`` result — *"All <model> model_kwargs are
   in the advertised set."* — whenever it found no violations. It did so even for a
   model with **no** ``OPTION_SCHEMA``, i.e. one whose advertised set it had never
   read, because there was none: the message asserted a fact the check had not
   established.

   That is the pitfall-#16 shape the checker exists to catch, turned inward. Since
   ``audit`` is ``--strict`` by default, a confident-but-baseless ``info`` is not
   harmless padding — it is the reassuring noise that hides the real finding. The
   check now stays **silent** unless it actually compared a declared kwarg against
   an advertised set, and when it does pass it names what it checked
   (``"All checked <model> model_kwargs are in the advertised set: ['attention_type']"``)
   so the claim is auditable rather than a blanket all-clear.

Categories
==========

The category set is split into two groups by intent: *errors* surface
configurations that cannot succeed, *warnings* surface configurations
that may *appear* to succeed while silently breaking metrics, gradients,
or features. Under ``--strict`` every warning is promoted to an error.

.. note::

   **2026-08-22: ``--strict`` is the parser default, and ``--no-strict`` is the
   opt-out.** CLAUDE.md non-negotiable 4, this page, ``execution_ledger.py``'s
   docstring and ``scripts/ci/cluster_verify.sh``'s comment had all said
   "``audit`` is ``--strict`` by default" for months; only argparse disagreed
   (``action="store_true"``, i.e. default ``False``), so an interactive
   ``spectramr audit <arm>`` exited **1** on warnings and read as a soft pass.

   Blast radius, measured before the flip over all 647 arms under
   ``experiments/inprogress`` (Tier 0+1, no probe): **507 pass / 3 warn /
   137 error**. The flip moves exactly the three
   ``vf/exp_vf_01_subvoxel_superres*_v2.yaml`` arms from exit 1 to exit 2; the
   137 already exit 2 and are untouched. Per-arm opt-out stays in the config
   (``synthetic_forward_probe_skip``), which is the reviewable place for it —
   ``--no-strict`` exists for interactive triage, not for gates.

Errors (always block)
---------------------

.. list-table::
   :header-rows: 1
   :widths: 26 6 68

   * - Category
     - Tier
     - When it fires
   * - ``schema``
     - 0
     - Pydantic ValidationError on load.
   * - ``required_section``
     - 1
     - One of ``data`` / ``model`` / ``training`` / ``optimization`` /
       ``logging`` is missing.
   * - ``model_registry`` / ``strategy_registry``
     - 1
     - ``model.model_type`` / ``training.strategy_class`` is not registered.
   * - ``domain_alignment``
     - 1
     - Derived data channels don't match ``model.in_channels``.
   * - ``advertised_options``
     - 1
     - YAML chooses an enum value the model does not list as supported.
       Replaces silent fallbacks (``attention_type='dual_domain' →
       'spatial'`` etc.). **Skips** — emits no result at all — for a model with
       no ``OPTION_SCHEMA``, or whose schema covers none of the declared
       ``model_kwargs``. See the note below.
   * - ``loss_domain_consistency``
     - 1
     - ``losses.image_losses`` is non-empty but
       ``losses.output_domain == "kspace"`` (or symmetric variants).
   * - ``metric_channel_compatibility``
     - 1
     - ``metrics.compute_fid=true`` with non-3-channel output, or
       ``compute_lpips=true`` with non-{1,3}-channel output. Catches
       the silent FID-on-multi-coil failure pattern.
   * - ``declared_losses_registered``
     - 1
     - A loss declared in ``losses.{image,kspace,complex}_losses`` is
       not in the LossRegistry. The loader silently drops it; the loss
       never fires.
   * - ``paradigm_required_fields``
     - 1
     - Strategy looks like VAE / diffusion / cold-diffusion but the
       paradigm-required field (``training.vae.kl_beta_end``,
       ``training.diffusion.num_timesteps``, ...) is unset. The loader
       falls back to a default that may silently break training.
   * - ``marker_subspace_conditioning``
     - 1
     - (VF plan I-5) A declared marker basis is ill-conditioned
       (:math:`\kappa(M) > 10^4`), which destabilises the VF-residual
       projection and inflates conformal intervals. Info-skips when no
       marker basis is declared or present locally.
   * - ``checkpoint_existence``
     - 1
     - (VF plan CW-2) A declared ``checkpoint.resume_from`` /
       ``model.checkpoint_path`` / ``*_basis_path`` / ``*_template_path``
       does not exist on disk. Warning severity (``--strict`` escalates),
       so an arm whose upstream artefact was never built is gated before
       dispatch rather than crashing at load. **Campaign-dependency
       deferral (Option A, 2026-06-16):** a downstream eval / calibration /
       DPS arm may set ``checkpoint.produced_by_arm: <upstream-arm>`` to
       declare that an upstream campaign arm builds its checkpoint. When
       set, a still-absent *campaign-artefact-rooted* checkpoint (under
       ``experiments/active/`` or ``experiments/results/``) is treated as a
       deferred dependency and **info-passes** the standalone ``--strict``
       pre-flight instead of hard-failing — the campaign runner builds the
       producer first, and the real existence gate still fires at actual
       checkpoint-load. The deferral is deliberately narrow (CLAUDE.md #15):
       it does **not** wave through a non-artefact path (typo'd local path →
       error) or a precomputed data artefact (marker basis / template tensor,
       which no training arm produces → still warns).

       **The deferral only buys correctness if the named producer actually
       exists and emits a state_dict-compatible checkpoint.** ``produced_by_arm``
       is a *promise*, not a guarantee — the static audit cannot diff the
       producer's conv-weight shapes against the consumer's model block, so a
       deferred dependency can info-pass while the producer is either missing or
       architecturally incompatible (a latent load-time crash, the checkpoint
       analogue of pitfall #16). Two obligations when wiring a
       ``produced_by_arm``: (a) a runnable arm of that name exists (e.g. under
       ``experiments/inprogress/<paradigm>/``); (b) its ``model`` block
       (``model_type`` + channels + ``model_kwargs``) **mirrors the consumer's
       exactly**. The 2026-06-22 ``m4raw_ddpm.ckpt`` case is the cautionary
       example: ``exp_vf_twin_dps`` (``enhanced_deep_unet`` in/out=8 complex) and
       ``exp_p2_b2_bloch_manifold_dps`` (in/out=1 RSS magnitude) both deferred to
       the *same* ``baseline_m4raw_ddpm`` / ``m4raw_ddpm.ckpt`` — but no such arm
       existed, and a 1-channel state_dict cannot load into an 8-channel U-Net.
       The fix was **two** producer arms (``baseline_m4raw_ddpm`` 8ch and
       ``baseline_m4raw_ddpm_rss`` 1ch), each mirroring its consumer's model
       block and trained as an epsilon-prediction DDPM, with each consumer
       repointed at its own producer's ``checkpoints/best.pt``.
   * - ``forward_pass_shape``
     - 2
     - Model forward returns wrong shape, or its forward / backward
       raises a runtime error that isn't OOM.
   * - ``amp_double_unscale``
     - 2
     - Backward pass raises an ``unscale_``-related RuntimeError.
   * - ``nan_in_forward`` / ``nan_in_loss`` / ``nan_in_gradient``
     - 2
     - The probe produced a NaN/Inf at the corresponding stage. Three
       distinct categories so the aggregator can group them.
   * - ``oom``
     - 2
     - CUDA out-of-memory at the configured batch / patch size.
   * - ``instantiation``
     - 2
     - Model registry returns a class that fails to construct with the
       supplied ``model_kwargs``.

Warnings (block under ``--strict``)
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 6 68

   * - Category
     - Tier
     - When it fires
   * - ``amp_grad_clip_interaction``
     - 1
     - AMP + ``enable_gradient_clipping=true`` + missing / non-positive
       ``gradient_clip_value``. Tier 2 confirms.
   * - ``early_stopping_metric``
     - 1
     - ``early_stopping.metric=val_psnr`` but cascading_super_resolution
       is on so the validation pipeline emits ``val_psnr_2x`` etc. —
       early stopping silently never triggers.
   * - ``identity_collapse``
     - 2
     - Forward output is within 1e-3 relative L1 of input on a random
       tensor. Suppressible per-model via
       ``synthetic_forward_probe_skip = {'identity_collapse'}`` for
       intentional residual / consistency blocks.
   * - ``gradient_explosion``
     - 2
     - Total gradient norm > 1e6 on the first synthetic backward. Real
       training will explode on iter 1 unless gradient clipping is on.
   * - ``dead_loss``
     - 2
     - The *configured* image-domain objective aggregates to a
       zero-gradient total at iteration 0 (warmup gate masks the only
       loss, or output == phantom target). Suppressible per-model via
       ``synthetic_forward_probe_skip = {'dead_loss'}`` for
       SELF-COMPUTING strategies whose real objective is computed by the
       strategy rather than the configured ``image_losses`` placeholder
       (e.g. Fisher-Rao geodesic, McCann ICNN — identity-at-init, so the
       placeholder l1 is ~0 even though the arm trains on the cluster).

Advisory (reported, never blocks)
---------------------------------

``declared_keys_are_not_discarded`` (Tier 1)
    Lists every key the YAML declares that an ``extra="ignore"`` block dropped
    before the model existed. The run never sees them; the file still shows
    them. Nothing else in the ladder can see this class, because the resolved
    config is *correct* — the value simply is not in it.

    Measured over ``experiments/inprogress/`` on 2026-08-02: **430 of 636 arms**
    carry at least one, **2,139 declarations** across **80 distinct paths**.
    Some look load-bearing — ``checkpoint.save_dir`` (134 arms),
    ``data.volume_format`` (214), ``physics.concomitant.field_strength_T`` (68),
    ``loss_logging.enabled`` (285).

    It is **advisory on purpose.** At that corpus size a warning would exit 2
    under the strict smoke wrapper and fail hundreds of arms at once for
    something that changes no behaviour. Same polarity as
    ``check_workflow_declared``: report now, ratchet to error once the corpus is
    drained. Closes issues #675 and #681.

    It reads ``ExecutionLedger.current()``, which ``spectramr audit`` arms before
    resolving the config. **If no ledger is armed the check says "NOT
    measured", never "none found"** — a check that cannot tell those apart is
    the silence it exists to break. If you resolve a config yourself and want
    this answer, call ``ExecutionLedger.begin_run()`` first.

    Advisory does not mean hidden — it appears on both audit surfaces::

        # an arm with discarded keys
        ✅ [health:declared_keys_are_not_discarded] 9 declared key(s) were
           discarded by an extra='ignore' block and the run never sees them:
           checkpoint.save_best, data.data_type, data.input_hr_dir, …

        # a clean arm — a distinct message, not silence
        ✅ [health:declared_keys_are_not_discarded] every declared key reached
           the resolved config.

    and under ``--json`` as a ``tier_0_1.results`` entry carrying the full list
    in ``yaml_keys``. Being ``passed=True`` at ``info`` keeps it out of the
    ``n_errors`` / ``n_warnings`` tallies that drive the exit code, so it
    reports without blocking.

``declared_model_kwargs_are_read`` (Tier 1)
    Lists every ``model.model_kwargs`` entry the arm declares that the resolved
    model class **provably never reads**. Distinct from
    ``component_kwargs_reach_constructor`` above, which asks whether a kwarg
    *arrived*: a parameter named in ``__init__``'s signature, documented in its
    docstring and then never referenced in the body arrives perfectly — the
    ledger records a clean delivery — and changes nothing. Nothing else in the
    ladder can see this class, because delivery genuinely succeeded.

    Verified on ``KSpaceColdDiffusionGenerator``: flipping ``activation`` or
    ``use_complex_conv`` leaves the module tree, the parameter count *and* the
    forward output bit-identical, and an invalid value does not raise, because
    the validating resolver is never called with it.

    **Not a duplicate of** ``scripts/ci/check_model_kwargs_are_read.py``. That
    gate asks whether a key exists *anywhere* in ``src/spectramr`` — as a dict key
    or as any function's named parameter — and targets a name that exists
    NOWHERE (issue #1075). Its permissiveness is exactly what lets this class
    through: ``activation`` and ``use_complex_conv`` *are* named parameters of
    ``KSpaceUNetBlock.__init__``, so the package-wide vocabulary counts them as
    read, and neither appears in that gate's 763-entry baseline. This check asks
    the narrower question — does the class *this arm resolves to* read the key?
    The CI gate catches ``process_type`` (which exists nowhere); this catches
    ``activation`` (which exists, but not on this arm's path).

    **Arm-scoped on purpose.** A dead parameter nobody sets is untidiness; a
    dead parameter an *experiment declares* is a false controlled variable — it
    reads as a knob the author chose, it is stamped into provenance, and in an
    ablation cohort it looks like a held-fixed axis. Only the intersection of
    "unread by this model" and "written by this arm" is reported.

    Measured over ``experiments/inprogress/`` on 2026-08-20 (647 arms with a
    resolvable ``model_type`` and a ``model_kwargs`` block): **90 arms (13.9 %)**
    declare at least one — ``activation`` (82), ``use_complex_conv`` (82),
    ``img_size`` (5), ``features`` (1), ``bottleneck_only`` (1).

    **85 of those, across 9 cohorts, are what the check actually reports**: the
    5 ``img_size`` arms are suppressed by ``DELIBERATELY_UNREAD``. The two
    numbers answer different questions — 90 is the raw corpus, 85 is the
    reported surface, and it is 85 that has to reach zero before the ratchet.

    It is **advisory on purpose, and unlike its sibling the corpus IS counted.**
    ``component_kwargs_reach_constructor`` stays advisory because nobody had
    measured its corpus; this one is advisory because the measurement came back
    90, and a warning exits 2 under the strict smoke wrapper — failing 90 arms
    for a defect none of them introduced. Ratchet to ``warning`` once the corpus
    is drained. The fix is to give the constructor a real parameter (forward it
    to the component that owns the behaviour) or delete the key — never to
    silence the check.

    **Scope, and what a pass does not mean.** The detector is static and
    conservative: it answers only for parameters named in the signature, and
    declines to answer at all for a constructor reaching for
    ``locals()`` / ``vars()`` / ``eval`` (a false positive sends an author to
    delete a live knob, which is worse than a miss). It therefore cannot see a
    knob that is inert *by method* — ``dc_weight`` under ``dc_method: hard``,
    which replaces observed lines rather than blending toward them — or inert
    *by branch* — ``reflect_padding_bottleneck_layers`` under a
    ``backbone_type`` whose construction path never reads it. Both are
    properties of a *configuration* rather than of a class and belong to the
    Tier-2 probe. A pass here means "no provably-unread declared knob", never
    "every declared knob matters".

    A parameter a constructor accepts and knowingly ignores belongs in
    ``inert_knobs.DELIBERATELY_UNREAD`` with a one-line reason — an entry there
    is a declaration of intent, not a suppression, so the next reader does not
    rediscover it. ``ComplexUNet.img_size`` ("unused, kept for factory
    compatibility") is the seed entry.

    It appears on both audit surfaces::

        # an arm declaring unread knobs
        ✅ [health:declared_model_kwargs_are_read] 2 declared model_kwarg(s) are
           never read by KSpaceColdDiffusionGenerator, so this arm's value cannot
           affect the run while still being stamped into provenance:
           activation='complex', use_complex_conv=True

        # a clean arm — a distinct message, not silence
        ✅ [health:declared_model_kwargs_are_read] all 5 declared model_kwargs
           are read by FNOGenerator (or allowlisted as deliberate).

    and under ``--json`` as a ``tier_0_1.results`` entry carrying the full list
    in ``yaml_keys``.

``physics_config`` (Tier 1)
    A k-space reconstruction/diffusion arm has ``physics`` present (the
    schema always constructs one) but INERT:
    ``physics.data_consistency.enabled`` is false AND no ``undersampling:``
    block is declared. Re-aimed 2026-08-12 (#933) from an unsatisfiable
    ``physics is None`` gate that could never fire.

    Measured over ``experiments/inprogress/`` on 2026-08-12: 120 arms are
    recon/diffusion-on-k-space (applicable), and **2** are physics-inert by
    this predicate —
    ``experiments/inprogress/workflow_baselines/b0_structural_denoise_m4raw.yaml``
    and
    ``experiments/inprogress/quality_matching/exp_qm_02b_restore_on_real.yaml``.

    It is **advisory on purpose, not by corpus size** (unlike
    ``declared_keys_are_not_discarded`` above, whose count is what makes it
    advisory). At least one of the two flagged arms is a *documented,
    deliberate* no-physics control — ``b0_structural_denoise_m4raw.yaml``'s
    own header says "the only task on this cluster whose degradation is
    PHYSICALLY REAL... No acceleration, no physics block, no DC, no coil
    maps, no adapters" and it declares ``workflow.task: denoising``. Both
    flagged arms already fail ``spectramr audit --strict`` today independent
    of this check, via the pre-existing ``acceleration_present`` error (and,
    for ``exp_qm_02b_restore_on_real.yaml``, three further pre-existing
    errors — ``domain_alignment`` ×2, ``data_model_compatibility``,
    ``workflow_dataset_signal_domain``) — so this check adds zero net-new
    corpus breakage even at this severity.

    A blanket ``workflow.task == "denoising"`` exemption would not be
    correct either: ``exp_qm_02b_restore_on_real.yaml`` shares the same
    no-undersampling posture but declares ``workflow.task: reconstruction``,
    so a task-keyed allowlist would need to characterise more than one
    spelling. Report now, design a real no-physics-task vocabulary (or an
    explicit per-arm opt-out, mirroring
    ``synthetic_forward_probe_skip``) and ratchet to error later, rather
    than landing a repointed check that reproduces the exact failure mode
    ``check_legacy_schema_mixing`` was deleted for (firing on legitimate,
    documented, intentional use).

"Reported" means both surfaces (#1275)
--------------------------------------

The heading above says *reported*, and until 2026-08-20 that was only true of
``spectramr audit``. The two surfaces render a report differently, and the
difference fell exactly on this tier:

``spectramr audit``
    ``cli/app.py`` prints **every** result, and ``HealthCheckResult.__rich__``
    gives a passing one a green icon. Passing results also reach ``--json``
    (``report.to_dict``) and the ledger artifact.

``spectramr train`` / ``train-distributed``
    ``HealthCheckReport.log_summary`` rendered a result **only when it did not
    pass**. An advisory returns ``passed=True`` by design — that is the polarity
    that keeps it out of ``report.passed`` and ``report.warnings`` — so its text
    could not reach a training log at all. It was counted in the
    ``Config Health: n/m checks passed`` denominator and discarded.

Advisory and invisible are different things, and the code could not express the
first without the second. ``HealthCheckResult.always_report`` now does:
``log_summary`` emits a **passing** result at ``info`` when it is set, reading
nothing that decides pass/fail, so a check's exit-code behaviour is unchanged.

Set it on the **branch that states a finding**, never on the check as a whole.
Measured on ``experiments/inprogress/kspace_filling/attention_shootout/
experiment_11_attention_none.yaml``: 140 of 141 results are passing ``info``
results and 16 of those carry a ``category`` — but nearly all 16 read
"not applicable", "not configured" or "check skipped". Emitting on
``severity == "info"``, or on ``category``, would have traded one invisible
finding for sixteen visible non-findings.

The casualty that prompted this was
``check_deepspeed_zero_stage_has_ranks_to_shard``. On the run that held four
GPUs and trained on one, it produced its diagnosis verbatim
("``zero_stage=2 at a DECLARED world size of 1 ... partitions nothing``") and
the operator saw ``Config Health: 141/141 checks passed`` instead.

.. note::

   Only ``pipelines/train.py``'s call site passes ``log_summary=True``. The
   witness pass in ``bootstrap.py`` filters passing verdicts the same way, and
   is deliberately left alone: a train run reaches the checker twice, and
   ``validate_config_health``'s ``log_summary`` flag exists precisely to stop
   the same checks narrating themselves twice into one job log.

Runtime input-channel concatenation
===================================

Some models build their first conv expecting the *post-concat* input
shape that the training strategy produces, not the raw
``model.in_channels`` declared in YAML. The canonical example is
``kspace_cold_diffusion``: the diffusion strategy concatenates S-maps
onto the input at runtime (see
``diffusion.py::_prepare_diffusion_inputs``), so the model's backbone
is built for ``in_channels * 2`` channels. Without the strategy, the
probe's ``[B, in_channels, H, W]`` tensor would crash the first conv
("expected 16 channels, got 8").

The probe inspects the constructed model for two opt-in attributes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Model attribute
     - Probe behaviour
   * - ``condition_with_smaps = True``
     - Input tensor built with ``2 × in_channels`` (mirrors
       kspace_cold_diffusion's S-map concat).
   * - ``synthetic_forward_probe_input_channels = N``
     - Input tensor built with exactly ``N`` channels (escape hatch
       for non-2× expansions; takes precedence over
       ``condition_with_smaps``).
   * - *(neither set)*
     - Input tensor uses raw ``in_channels`` (the common case).

These attributes live on the model class, not in the YAML — they
describe a structural property of the model's forward path, and they
must stay consistent with whatever the training strategy actually
does at runtime. If you flip ``condition_with_smaps`` off without
also disabling the strategy's concat, real training will crash with
the same shape mismatch the probe used to hide.

Probe input: Shepp-Logan, not white noise
=========================================

The Tier-2 forward probe builds its dummy input from the existing
:func:`spectramr.infrastructure.physics.signal_models.tse_b0_model.shepp_logan_2d`
phantom rather than ``torch.randn``. Two reasons:

1. **Saved probe images are inspectable.** When the smoke wrapper
   passes ``--audit-probe`` it also passes ``--save-probe-images
   <dir>`` automatically, dumping ``<arm>_input.png``,
   ``<arm>_output.png`` and ``<arm>_target.png`` to
   ``experiments/results/smoke_test/probe_images_<TS>/``. With a
   recognisable phantom you can immediately see whether the model
   preserved anatomy, scrambled channels, or mishandled the
   k-space ↔ image-domain conversion. With pure noise you only see
   noise.

2. **Structure-sensitive bugs surface earlier.** Identity collapse,
   channel-tile bugs, and wrong-domain output (showing FFT-of-image
   when you expected image-of-FFT) are visually obvious on a
   phantom. They're invisible on noise.

The phantom is wrapped by
:func:`spectramr.infrastructure.validation.phantom_builder.synthetic_phantom`,
which handles 3-D ``(B, C, L)``, 4-D ``(B, C, H, W)`` and 5-D
``(B, C, D, H, W)`` shapes plus an optional ``add_phase=True`` for complex
outputs (so real / imag tiled into the channel axis sees a non-trivial phase).

Rank-3 ``(B, C, L)`` is a supported layout, not an error: 1-D models (HRF time
courses, tangent-score networks, MRF fingerprints) probe with it, and before
support landed they fell back to ``torch.randn`` and lost the phantom's
deterministic spectral content (``F-PHANTOM-RANK3``).

.. note::

   **2026-07-14 fix.** ``add_phase`` was silently ignored on the rank-3 path —
   the 1-D branch returned before the phase ramp, so a complex 1-D phantom came
   back with a **zero imaginary part**: a real signal wearing a complex dtype,
   which is the one thing this helper exists to avoid. Any complex 1-D model
   (MRF fingerprints are complex) was therefore probed with a purely real input,
   so the probe could not exercise the complex path at all — an unwired knob
   (pitfall #15) hiding inside the tool meant to catch them. All ranks now share
   one ``_apply_dtype_and_phase`` tail, so a future rank cannot silently skip a
   step.

Opt out with ``--noise``:

.. code-block:: bash

   # Phantom (default — recommended)
   python -m spectramr.cli audit foo.yaml --probe

   # Pure white-noise input (legacy)
   python -m spectramr.cli audit foo.yaml --probe --noise

   # Save the probe input/output/target as inspectable PNG mosaics
   python -m spectramr.cli audit foo.yaml --probe --save-probe-images /tmp/probe_pngs

The image-save path tiles channels horizontally and shows the
centre slice for 5-D volumes. Complex tensors are split into
magnitude / phase rows. The save is best-effort: missing matplotlib,
weird shapes, etc. are silently swallowed so the probe never fails
because of an inspection artefact.

``--strict`` mode and the no-warnings rule
==========================================

The user discourages silent fallbacks (CLAUDE.md pitfall #9). The
``--strict`` flag turns that rule into an exit-code:

============= ============ ==============================================
Audit outcome Default exit Under ``--strict`` exit
============= ============ ==============================================
Clean         0            0
Warnings only 1            **2** (treated as failure)
Errors        2            2
============= ============ ==============================================

The smoke wrapper enables ``--strict`` by default. Re-run with
``--allow-warnings`` for the legacy "warnings are OK" behaviour
(useful only when triaging an existing arm you can't fix yet).

Why this matters: 14 of last run's "passed" arms had OOM during
validation, 11 had silently-omitted losses, 9 had early-stopping
metrics that never matched. They look green in the summary table.
Under ``--strict`` they go red, where they belong.

Audit before you train
======================

The ladder earns its keep as a gate rather than a report: audit first, and let
the training run start only if it passed.

.. code-block:: bash

   # Default: audit (Tier 0+1, --strict) → train only if the audit passed.
   spectramr audit <arm>.yaml && \
     spectramr train --config <arm>.yaml -O training.max_iterations=10 \
                    -O logging.log_interval=1

   # Audit only — skip train. ~100 ms per arm. The fast triage.
   spectramr audit <arm>.yaml

   # Audit + Tier-2 probe, then train. Catches more before real data is loaded.
   spectramr audit <arm>.yaml --probe && \
     spectramr train --config <arm>.yaml -O training.max_iterations=10

   # Tolerate audit warnings rather than promoting them to errors.
   spectramr audit <arm>.yaml --no-strict

   # Skip a cohort when sweeping a directory of arms.
   spectramr audit <arm>.yaml --exclude 'vf/*'

.. note::

   The maintainers drive this from a SLURM batch wrapper over their full arm
   corpus. Neither that wrapper nor that corpus is part of this distribution,
   and neither needs to be: looping the command above over your own arms is the
   whole of what it does.

The audit's structured JSON is written to
``experiments/results/smoke_test/audit_<timestamp>_<arm>.json`` next
to the existing log file. Failures cause the wrapper to skip ``train``
for that arm but continue to the next one (the audit is the gate, not
a circuit-breaker for the whole batch).

The existing ``--retry-log`` flag still works: a failed audit produces
a ``❌ FAIL: <name>`` line in the log just like a failed train, so
``--retry-log path/to/previous.log`` re-runs only the failed arms.

Cluster-mount path exemption (F5c)
==================================

Tier 1 includes a ``hardcoded_cluster_paths`` check that rejects YAML
fields (``data.data_root``, ``data.index_path``, ``data.validation_index_path``,
``data.test_index_path``) starting with a hardcoded cluster prefix
(``/project/<allocation>/``, ``/project/<other-allocation>/``, ``/scratch/<allocation>/``).

The check exempts paths under the user's own cluster mount. Three
mechanisms (any one is enough):

1. **Explicit env-var** — set ``SPECTRAMR_DATA_ROOT``, ``PROJECT_ROOT``,
   or ``SPECTRAMR_ROOT`` in your cluster ``.env``. Any path *under* one
   of these is exempt:

   .. code-block:: bash

      # In your cluster .env:
      export SPECTRAMR_DATA_ROOT=/project/<allocation>/<your-account>/spectramr

2. **Auto-detect via cwd + $USER** (F5c, 2026-05-20) — when no env-var
   is set, the check inspects ``$USER`` and the current working
   directory. If ``cwd`` lives under a forbidden cluster prefix AND
   the path-segment immediately after the prefix equals ``$USER``,
   that subtree is treated as a configured root.

   Concretely: ``USER=<your-username>`` running from
   ``/project/<allocation>/<user>/spectramr`` auto-exempts paths under
   ``/project/<allocation>/<user>/`` without any ``.env`` edits. A
   colleague's leak (``/project/<allocation>/<other-account>/...``) still
   trips the check because the segment doesn't match ``$USER``.

3. **Relative paths** — the recommended canonical form. YAMLs that
   say ``data_root: databases/m4raw/data`` rely on ``PathResolver``
   to join against the configured project root at runtime. These
   never trip the check at any stage.

Audit anchor: ``tests/unit/infrastructure/validation/test_health_checker_json_and_new_checks.py``
(``test_hardcoded_path_check_auto_detects_cluster_owner_from_cwd_and_user``,
``test_hardcoded_path_check_cwd_auto_detect_does_not_exempt_other_users``).

Channel-audit blind-spot taxonomy (F6 / F7)
===========================================

The Tier-1 ``check_channel_audit_assumptions`` flags cases where the
static channel-derivation in ``check_domain_alignment`` cannot
statically prove the model's ``in_channels`` matches the data shape.
Two outcomes:

* **info** — the case is by-design and another check covers it.
* **warning** — a real blind spot that ``--strict`` mode rejects.

The three categories the check inspects:

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Case
     - Severity
     - Why
   * - ``model_type`` in ``_INPUT_CONCAT_MODELS``
       (e.g. ``kspace_cold_diffusion``, ``disentangled_mri``)
     - **info**
     - The strategy concatenates conditioning channels at runtime by
       design. The runtime DomainMismatch check at
       ``strategies/base.py:660`` covers it; static verification adds
       nothing. (F6, 2026-05-20.)
   * - ``adapters.pre_model`` declared and every step is in
       ``_ADAPTER_CHANNEL_EFFECTS``
     - **info**
     - ``check_adapter_chain_channel_resolution`` already resolves
       the post-adapter width statically. No blind spot remains.
       (F6b, 2026-05-20.)
   * - ``adapters.pre_model`` declared with ≥1 unknown-effect adapter
     - warning
     - The static derivation cannot fold an unknown channel transform.
       ``--probe`` or a YAML annotation is required.
   * - ``data.coil_processing_mode='none'`` and either
       ``data.target_channels`` unset OR ``≠ model.in_channels``
     - warning
     - Channel count deferred to the per-file h5 header with no
       user-declared expectation. Set ``coil_processing_mode`` or
       ``data.target_channels=<expected>`` to resolve.
   * - ``data.coil_processing_mode='none'`` and
       ``data.target_channels == model.in_channels``
     - **info**
     - The user has explicitly declared the expected channel count.
       The runtime DomainMismatch check is the source of truth for
       the actual header value. (F6c, 2026-05-20.)

Audit anchor: ``tests/unit/infrastructure/validation/test_health_checker_json_and_new_checks.py``
(``test_channel_audit_assumptions_*``).

Capability-contract & channel refinements (2026-06-05)
======================================================

Three precision fixes landed while clearing the ``experiments/inprogress``
structural-error backlog (config-fix branch):

* **Agnostic losses pass any block.** ``check_loss_domain_block_match``
  now treats a loss registered ``@register_loss(domain="agnostic")``
  (e.g. ``nll_bits_per_dim`` on a normalizing flow) as valid under
  ``image_losses`` / ``kspace_losses`` / ``complex_losses`` — an
  agnostic loss is domain-independent by construction, so it imposes no
  block constraint. A genuine cross-domain mismatch (a ``kspace`` loss
  under ``image_losses``) still errors.

* **FNO is out-channels-independent.** ``fno`` joined
  ``_OUT_CHANNELS_INDEPENDENT_MODELS``: it reconstructs a (1- or 2-ch)
  image/k-space estimate whose channel count is independent of the input
  coil count, so an ``svd(num_virtual_coils=4)`` → 8-ch *input* maps to a
  2-ch reconstruction *output* without ``domain_alignment`` firing on
  ``out_channels``.

* **Runtime-concat models skip the adapter-chain channel check.**
  ``check_adapter_chain_channel_resolution`` now returns *info* (not
  *error*) for ``model_type`` in ``_INPUT_CONCAT_MODELS`` /
  ``_PAIRED_CONTRAST_MODELS`` (e.g. ``kspace_cold_diffusion``): these
  concatenate an S-map / reference / paired-contrast tensor at runtime,
  so a declared ``in_channels`` larger than the static pre-model chain
  output is expected — mirroring the existing ``domain_alignment``
  exemption.

The k-space→image reconstruction cohort (``hermitian_fno``,
``cross_attention_oracle_unet``, ``enhanced_deep_unet``) bridges k-space
data into its declared image/complex-image input via an explicit
``adapters.pre_model: [ifft_kspace_to_image]`` chain (plus
``magnitude_from_complex`` when the model is magnitude-image, with a
symmetric ``pre_loss_target`` so the loss compares like-for-like) — never
a silent strategy-internal transform.

Audit anchor:
``tests/unit/infrastructure/validation/test_health_checker_json_and_new_checks.py``
(``test_loss_domain_block_match_*``) and
``tests/unit/models/test_domain_contracts.py``.

Forbidden silent fallbacks
==========================

The "discourage fallbacks" rule from CLAUDE.md is enforced by the
``advertised_options`` check. To opt your model in, declare an
``OPTION_SCHEMA`` class attribute mapping kwarg names to allowed
values:

.. code-block:: python

   from spectramr.models.registry import register_model

   @register_model(name="my_model", training_mode="reconstruction")
   class MyModel(nn.Module):
       OPTION_SCHEMA = {
           "attention_type": ["spatial", "cross", "dual_domain"],
           "norm": ["batch", "group", "layer"],
       }
       def __init__(self, in_channels: int = 1, attention_type: str = "spatial",
                    norm: str = "batch", **kwargs):
           # Use the value as-is. The audit guarantees it is in the
           # advertised set, so no runtime fallback is needed.
           ...

Models without an ``OPTION_SCHEMA`` are skipped — the check is opt-in
to preserve backward compatibility, but every new model should declare
one.

Cookbook
========

"30-second sanity check before pushing a new YAML"
--------------------------------------------------

.. code-block:: bash

   python -m spectramr.cli audit experiments/inprogress/<paradigm>/<arm>.yaml

"Audit a whole directory of arms in under a minute"
---------------------------------------------------

.. code-block:: bash

   find experiments/ -name '*.yaml' -print0 |
     xargs -0 -n1 spectramr audit

"Catch AMP / shape bugs without launching real data"
----------------------------------------------------

.. code-block:: bash

   find experiments/ -name '*.yaml' -print0 |
     xargs -0 -n1 -I{} spectramr audit {} --probe

"Aggregate audit JSON across the smoke run"
-------------------------------------------

.. code-block:: bash

   jq -s '
     map(.tier_0_1.results[]? | select(.passed == false)) |
     group_by(.category) |
     map({category: .[0].category, count: length, examples: [.[0:3][].message]})
   ' experiments/results/smoke_test/audit_*.json

The output is the same shape as the manual
``ERRORS_GROUPED_SUMMARY.md`` — but generated.

Regression corpus (``tests/audit/``)
=====================================

A YAML-based regression corpus lives under ``tests/audit/corpus/`` and is
driven by ``tests/audit/test_audit_regression.py``.  It provides a
reproducible record of which audit checks fire on which YAML shapes —
useful for validating checker changes without re-running the full smoke
suite.

**Corpus layout**::

    tests/audit/corpus/
    ├── passing/          # one canonical YAML per paradigm family (reconstruction,
    │                     #   gan, diffusion).  Each must produce zero error-severity
    │                     #   failures from ConfigHealthChecker.run_all_checks.
    └── failing/          # one YAML per violated check.  Filename convention:
                          #   <check_name>__<mode>.yaml
                          #   where <check_name> must match HealthCheckResult.check_name
                          #   or a ValidatorRegistry rule name.

Every fixture must declare ``config_version: '1.0'`` — a ratchet enforced by
``test_no_failing_fixture_declares_a_refused_config_version``, not a migration
you are expected to perform.

**Running**::

    source .venv/bin/activate
    pytest tests/audit/ -v --tb=short          # loads real TrainingSettings; no GPU

.. _audit-corpus-tier-markers:

Declaring where a fixture is caught
-----------------------------------

A failing fixture must be **rejected by the audit ladder**, with the rejection
attributable to the check its filename names.  There is more than one legitimate
place for that to happen, so the fixture declares which one in its header.  The
driver holds no table of its own.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - header marker
     - contract
   * - *(none)*
     - ``TrainingSettings.from_yaml`` must succeed and ``ConfigHealthChecker``
       must emit a non-passing result named ``<check>``.  This is the default.
   * - ``# expect: tier0 <leaf>``
     - ``from_yaml`` must **raise**, and the failure must be located at
       ``<leaf>``.  Pydantic ``loc`` tuples are matched exactly, so ``data``
       means *the* failing field, not a word in the message.  A Tier-0
       rejection is a *stronger* result than the Tier-1 check firing — the
       config never reaches the health checker at all.
   * - ``# expect: registry``
     - The dict-level ``ValidatorRegistry`` rule ``<check>`` must fire on the
       raw document.  Use where no ``ConfigHealthChecker`` check carries the
       name.  The fixture leaves the health-checker parametrisation by
       *collection*, not by a runtime skip, and
       ``test_registry_expectations_name_a_registered_rule`` refuses a marker
       that names no rule — otherwise the marker would be an escape hatch that
       leaves the fixture tested by nothing.
   * - ``# known-dead: <reason>``
     - ``xfail(strict=True)``.  The named check is known to be structurally
       incapable of firing.  Repairing the check turns the test **red**, which
       is the point: the marker cannot rot into a silent pass.

**Only mark ``tier0`` when the schema rejection *is* the named violation.**
Three fixtures were formerly rejected at Tier 0 for reasons unrelated to their
check — a retired top-level ``objectives:`` block, ``optimization.epochs``, and
``validation.val_frequency``, none of which are fields any more.  Annotating
those as tier-0 would have turned them green while testing nothing; they were
repaired instead.

**Adding a new check to the corpus:**

1. Identify ``HealthCheckResult.check_name`` from the checker source, or the
   ``ValidatorRegistry`` rule name.
2. Create ``tests/audit/corpus/failing/<check_name>__<description>.yaml`` at
   ``config_version: '1.0'`` with exactly one deliberate violation.
3. Run ``pytest tests/audit/ -v``.  If the named check does not fire, that is a
   result, not a reason to soften the test — either the fixture does not express
   the violation, or the check is dead.  Add a marker only for the case you have
   actually established.
4. Keep the violated block non-empty.  A section carrying only a comment parses
   as ``null``, and several checks early-return ``passed=True`` on a null
   section, which silently disarms the fixture.

.. warning::

   **A checker crash is not evidence that the check fired.**  Until 2026-08-09
   this driver built a ``SimpleNamespace`` from the raw YAML and wrapped
   ``run_all_checks`` in ``except Exception: return``, commented *"A crash in
   the checker itself counts as the check firing."*  It does not: the crash came
   from the stub lacking ``data.source``, inside ``check_train_val_split_leakage``
   — a check unrelated to any fixture's name, which a perfectly valid config
   would trigger identically.  Combined with all 30 fixtures still declaring the
   refused ``config_version: '6.0'``, every entry aborted before any check ran
   and every abort was absorbed: 25 reported ``skipped`` and 5 reported
   ``passed``.  The corpus verified nothing while looking green (issue #922).

   The driver now loads real ``TrainingSettings`` and lets any unanticipated
   exception propagate.  Two checks were consequently marked ``known-dead``
   (issue #933): ``legacy_schema_mixing``, whose ``_LEGACY_CONFLICTS`` legacy
   leaves were all removed by the block decomposition and whose distinction the
   fold erases anyway, and ``physics_config``, whose ``physics is None`` gate
   was never true because the schema always constructs a
   ``PhysicsConfigSchema``. Both asked a **raw-document** question of a
   resolved object; that question belongs to the execution ledger, which still
   sees the pre-fold document.

   **Repaired 2026-08-12 (#933).** ``legacy_schema_mixing`` was deleted
   outright — its own premise was obsolete (``enable_image_normalization``
   and ``normalization_type`` are not rivals) and both ``data.*`` legacy
   leaves are ``fold`` posture, so a repointed version could not distinguish
   "user set both" from "user set one" and would have fired on legitimate
   canonical configs. Its corpus fixture
   (``legacy_schema_mixing__normalize_images_and_normalization_type.yaml``)
   was removed with it. ``physics_config`` was re-aimed at the answerable
   question — data consistency disabled AND no ``undersampling:`` block —
   and its corpus fixture's ``known-dead`` marker was removed since the
   check now genuinely fires on it. The re-aimed predicate was measured
   locally against all 647 ``experiments/inprogress`` arms before landing:
   120 are recon/diffusion-on-k-space, and 2 are physics-inert. A first
   landing at ``severity="error"`` was reverted after review: at least one
   of the two (``workflow_baselines/b0_structural_denoise_m4raw.yaml``) is a
   documented, deliberate no-physics control, not a bug, so "small blast
   radius" was not sufficient on its own — the brief's error-severity branch
   requires each flagged arm to *genuinely* have inert physics, which does
   not hold here. It lands ``severity="info"`` (advisory-first); see
   "Advisory (reported, never blocks)" above for the corpus detail and both
   arm paths. (Local CPU measurement — the cluster audit is still the
   authority on whether any given arm is clean.)

target_domain vs registered output_domain (E-VIZ2, 2026-06-16)
==============================================================

``ConfigHealthChecker.check_target_domain_matches_registered_output_domain``
closes a gap the sibling ``check_model_loss_output_domain`` could not see.
``model.target_domain`` is **Priority-1** in
:func:`spectramr.infrastructure.training.utils.domain_inference.infer_output_domain`
— it *overrides* the model's registered ``output_domain`` capability and drives
``needs_ifft_for_visualization``. A YAML that declares ``target_domain: kspace``
on a model registered ``output_domain="image"`` therefore makes the validation
writer IFFT an already-image prediction → **k-space rendered as an image**
(DC-blob + concentric rings), the E-VIZ2 smoke finding (exp_p7,
``hamiltonian_trajectory_generator``).

The check reads the *registered* domain (not ``target_domain``) and so:

* **errors** when an annotated model's ``target_domain`` contradicts its
  registered ``output_domain`` (the exp_p7 case — fixed by setting
  ``target_domain: image``); and
* **abstains** (info/skip) when the model is unannotated — e.g.
  ``reciprocity_divisor`` (exp_vf_27), which relies on the legacy
  ``KNOWN_KSPACE_OUTPUT_MODELS`` set. Those rely on ``target_domain`` (P1) as the
  SSOT override; the holistic VF-family domain re-classification (annotate the
  ~15 sibling VF decorators + prune the legacy set) is the tracked follow-up so
  this check can eventually bite them too.

``check_attention_domain_compatibility`` — attention vs backbone vs domain
==========================================================================

The k-space cold-diffusion backbone consumes k-space feature maps when
``model_kwargs.force_pure_kspace: true`` and image features otherwise, and
several attention blocks carry an internal domain assumption. The dispatch
derives ``feature_domain`` from ``force_pure_kspace`` and threads it through so
the blocks orient their FFTs correctly. This
Tier-1 check enforces, at ~100 ms, the contracts that would otherwise surface as
a build-time crash or a silently-mislabeled arm. It is scoped to the
``kspace_cold_diffusion`` model family (and its ``kspace_cold_diffusion_generator``
alias); every other ``model_type`` skips with an informational pass.

.. list-table::
   :header-rows: 1
   :widths: 8 20 72

   * - Rule
     - Severity
     - Fires when
   * - R0
     - error
     - ``model_kwargs.attention_type`` is not an advertised option (not in
       :data:`spectramr.models.blocks.attention_domains.ATTENTION_DOMAIN_SUPPORT`).
   * - R1
     - error
     - ``force_pure_kspace: true`` + ``backbone_type: unet`` (builds
       ``PureKSpaceUNet``, which has no attention seam) with any
       ``attention_type`` other than ``none`` — the request would be silently
       dropped (pitfall #16 facade). The constructor default is ``self``, so an
       arm that omits the key also fails. **Fix:** set ``attention_type: none``,
       or ``backbone_type: complex_unet`` to keep block-level attention.
   * - R2
     - error
     - ``backbone_type: complex_unet`` requesting an attention the block
       dispatch cannot build (e.g. ``spatial``, which exists only in the
       up-block dispatch and crashes the down block at build time).
   * - R3
     - error
     - ``backbone_type: complex_unet`` requesting an attention that does not
       support the derived ``feature_domain``. Vacuously green today (every
       attention supports both domains after the 2026-07-03 change); this is the
       ratchet a future single-domain attention block registers against.

The single source of truth for the advertised set, the per-type domain support,
and the ``complex_unet`` block-dispatch coverage is
:mod:`spectramr.models.blocks.attention_domains`, which both this check and the
runtime dispatch consult. ``KSpaceColdDiffusionGenerator.__init__`` mirrors R1
and R3 as build-time ``ValueError`` raises, so a non-YAML caller cannot slip
past. Regression coverage:
``tests/unit/infrastructure/validation/test_attention_domain_compatibility.py``.

Witnesses added by the 2026-09 cohort review
============================================

Seven witnesses landed with the corpus-wide review of ``experiments/inprogress/``
(2026-09-02). Each ships with a planted violation in its test module under
``tests/unit/infrastructure/validation/witness/checks/``; the ratchet notes say
which ones are still advisory and what promotes them.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Witness
     - Severity
     - What it refuses
   * - ``nex_reference_route``
     - error
     - ``data.use_repetitions: true`` on a route other than ``m4raw``: only that
       route builds a repetition-averaged target, so the arm advertised a
       reference it never received (79 arms were drained).
   * - ``held_out_test_split_disjoint``
     - error
     - a declared ``data.source.test_index_path`` that shares a subject or file
       with the train or validation manifest (validation selects the checkpoint,
       so it is part of the training pool).
   * - ``held_out_test_declared``
     - info (advisory)
     - a baseline / headline / reference arm with no held-out test set. Promoted
       to a warning once the Tier-1 cohorts adopt ``test_index_path``.
   * - ``warmup_shorter_than_budget``
     - error
     - ``optimization.scheduler.warmup_steps >= training.max_iterations``: the
       learning rate never leaves warm-up.
   * - ``training_budget_is_positive``
     - error
     - ``training.epochs: 0`` with no positive ``max_iterations``.
   * - ``undersampling_block_is_applied``
     - error / info
     - an ``undersampling:`` block that reaches the data through no route
       (``data.trajectory``, ``data.image_undersampling``, a dynamic mask, the
       digital twin, or a strategy declaring ``applies_undersampling``). Error on
       image-domain datasets; on k-space datasets whose strategy has not declared
       the flag it is reported as UNVERIFIED until the declaration census is done.
       ``base_acceleration: 1.0`` with no range and no dynamic mask is the
       explicit fully-sampled declaration and passes: there is nothing to apply.
   * - ``strategy_class_matches_training_mode``
     - error
     - an explicit ``training.strategy_class`` that is a generic base
       (reconstruction, physics-driven, diffusion, GAN, VAE) while
       ``training.training_mode`` maps to a specific mechanism class: the arm
       runs without the mechanism its mode names. A specialisation of the mapped
       class and a sibling mechanism declared by name both pass;
       ``metadata.tags.strategy_override_reason`` is the documented escape.
   * - ``ood_acceleration_range_is_read``
     - error
     - ``physics.digital_twin.ood_acceleration_range`` declared where no validation
       pass reads it: the twin does not undersample (``enable_undersampling`` false)
       or the strategy is not one of the twin-driven readers (``virtual_fiducial``,
       ``vf_admm`` and subclasses, which re-score every rung as
       ``val_ood_{R}x_<metric>``). Its predecessor
       ``undersampling.out_of_distribution_range`` was declared on 58 arms and
       read by nothing (VF review 2026-09-03).
   * - ``image_losses_reach_the_objective``
     - error / info
     - a ``losses.image_losses`` entry the strategy neither computes inline
       (``inline_losses``) nor folds (``folds_image_losses`` with a registered
       name) is dropped at runtime while the loss census reads it as the
       objective; 26 mrixfields arms declared an ``l1`` on score-matching and
       velocity strategies (2026-09-03). A strategy that has not declared its
       ownership is reported UNVERIFIED.
   * - ``no_dead_precision_flag``
     - error
     - ``training.enable_mixed_precision`` declared: no code path reads it and the
       run uses ``optimization.precision.enabled`` either way (#887). The
       ``inprogress/`` corpus was drained on 2026-09-03; the witness reports the
       spelling in the other trees until they drain and a ``raise`` rename record
       can retire it corpus-wide.
   * - ``validation_metric_names_resolve``
     - error / info
     - a name in ``validation.scoring.compute``, ``metrics.best_metric_name`` or
       ``early_stopping.metric`` that is neither registered nor declared by the
       strategy in ``capabilities.emitted_metrics``. Error when the strategy
       declares its emitted set, an UNVERIFIED line when it does not.

Two existing detectors changed in the same review: a schedule-certification
witness that does not apply now returns an INFO skip instead of a warning that
failed ``--strict`` on every non-cold arm, and ``acceleration_present`` fires only
where a measurement is reconstructed: on the two reconstruction bases matched
exactly, on every cold-diffusion subclass, and on a strategy declaring
``applies_undersampling``. Twenty arms use ``ReconstructionTrainingStrategy`` as a
generic base (conformal calibration, BALD, PILOT, QSM, spin-SDE, federated DP)
without reconstructing anything, so a subclass match would demand a block those
arms cannot honour. A declared non-reconstruction ``workflow.task`` (denoising,
synthesis, super-resolution, ...) passes without a block. The check and
``undersampling_block_is_applied`` share the ``applies_undersampling`` declaration,
so the two cannot demand opposite things of one arm.

The bulk audit (``spectramr audit <directory>``) and the single-arm audit now build
one report: the Tier-0/1 witness ladder, bridged to health-check results. The
bulk loop used to call the health checker directly, so no witness that is not a
health check (``schedule.*``, the budget, undersampling, validation-metric and
strategy-dispatch witnesses) ever ran over a directory, and it counted results by
severity, so a passed advisory (the concomitant correction at 3 T) became an
``ERROR(strict)`` the single-arm run could not reproduce. Both surfaces now read
``HealthCheckReport.errors`` / ``warnings`` (a *failed* result of that severity),
and that advisory is an INFO that is always reported. A directory run therefore
reports what the per-arm runs report, at about a second per arm.

Four detectors changed with the first directory-wide witness run (2026-09-03):

- The training-budget rule has one owner, ``spectramr.config.training_budget``,
  read by both the validator registry's ``epochs_valid`` (the train-time gate)
  and the ``training_budget_is_positive`` witness; a calibration mode that
  optimises nothing (``calibration``, ``phys_residual_conformal``) is added
  there, not in either caller.
- ``domain_alignment`` doubles the expected *input* width when
  ``training.diffusion.condition_on_input`` is true on a non-cold, non-latent
  diffusion arm, because the strategy concatenates the measurement onto the
  noised target; the output keeps the loaded width.
- The compatibility-matrix resolver's data domain is what the loader emits
  (dataset family, then a coil-processing mode of ``rss_image`` or
  ``magnitude``), one derivation shared with the spec card; it used to be the
  model's declared target domain.
- The matrix's ``domain_chain`` rule is deleted: ``data_model_compatibility``
  owns the data-to-model leg, folds the declared adapter chain and knows which
  strategies adjoint k-space to the image domain internally.

``scientific_metadata_expected_outcome`` (a health check, not a witness) reads
``metadata.expected_outcome``: ``floor`` and ``ceiling`` describe a position relative
to ``metadata.baseline`` and warn when no baseline is declared; ``comparable``
stands on its own.

The launch gate is not a witness but belongs beside them: ``spectramr train``
refuses an arm whose ``metadata.status`` is ``needs_implementation``, ``inert`` or
``blocked`` unless ``--allow-status`` names that status (see
:doc:`config_schema_reference`).

Out of scope
============

* **OOM prediction without running the probe.** Static estimation of
  GPU memory from layer shapes is plausible but the variance from
  AMP / activation checkpointing / gradient accumulation is too large
  for a clean check. Use ``--audit-probe --device cuda`` instead.
* **HDF5 file integrity.** Belongs in a data-pipeline pre-flight, not
  the config audit.
* **Auto-fix mode.** The user explicitly forbids silent corrections;
  the audit reports, the human fixes.
