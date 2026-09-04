.. _config_key_reference:

=====================================
Retired Configuration Keys
=====================================

.. warning::

   **This page is generated.** Edit
   ``spectramr/config/schemas/renames.py`` and run
   the page generator in the maintainers' tree; do not edit the tables below.

Every key that has moved, where it went, and why. The rename table is the single
source of truth for four things at once -- the schema shim that accepts or
rejects the old spelling, the fixer that rewrites YAML, the corpus gate that
counts what is left, and the ``--override`` path translation -- so a key listed
here behaves identically in all four.

Two postures
============

**fold** -- staged. The old spelling still LOADS: a ``mode="before"`` validator
moves the value to its canonical path before validation. It is gone from Python,
so there is one read path and, temporarily, two accepted spellings in YAML. Fold
records are what let a rename land before the corpus migration.

**raise** -- retired. The old spelling is an error naming its replacement. A
``raise`` record only lands together with the migration that drives its corpus
usage to zero.

Fix any of these automatically::

    python scripts/ci/migrate_config_keys.py <your-arm>.yaml --apply


What the fixer cannot reach
===========================

The fixer edits *lines*, not a parsed document -- a ruamel round-trip reflows the
whole file and buries a rename under a diff nobody reviews. The cost is that it
cannot enter a flow mapping::

    validation: {metrics: [psnr], val_batch_size: 4}

A key in that shape used to be reported as **absent**, which is the same answer
the tool gives for a file that never declared it. That silence mattered because
the promotion rule reads a count: drive a fold record to zero, flip its posture to
``raise``, delete the shim -- and every arm hiding the key in a flow mapping then
fails at load.

Both migrators now run a parser-based **detector** alongside the line-scanning
**rewriter**, and refuse when they disagree:

``UNSUPPORTED``
    Declared, and unreachable. Reported **separately from the STAGED countdown**
    and non-zero exit. A record may not be promoted while any remain.

Reflow the offending file to block style, then re-run. For the live count, run
either migrator over the corpus -- this page is generated from ``RENAMES`` alone
and does not scan ``experiments/``, so a number written here would only drift.

Retired outright (92)
=====================

These raise on load. Write the replacement.

.. list-table::
   :header-rows: 1
   :widths: 26 26 10 38

   * - Retired key
     - Replacement
     - Since
     - Why
   * - ``data.data_layout``
     - ``data.source.layout``
     - 2026-07-31
     - WHERE the bytes come from. Six answers to one question, previously scattered across ~300 lines of `data:`. `dataset_type` is deliberately NOT folded: it is a DISPATCH key rather than a location -- dataset_instantiator, config_health_checker and axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan binds it to `workflow.regime`, and 636 arms declare it as the first line of their `data:` block where nothing about it is ambiguous. `datasets` and `manifest_roles` stay for a different reason: both are already nested blocks, and this phase groups scalars rather than re-parenting blocks. `data_root`/`data_layout` drop the `data_` prefix the outer block supplies; the `_path` leaves keep their full names, since under the naming rule `<thing>_path` is a file and index/validation_index/paired_manifest are the things.
   * - ``data.enable_graph_encoding``
     - ``data.domain.enable_graph_encoding``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.expose_acquisition_params``
     - ``data.expose.acquisition_params``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_conformal_jacobian``
     - ``data.expose.conformal_jacobian``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_cortex_flatten_grid``
     - ``data.expose.cortex_flatten_grid``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_field_strength``
     - ``data.expose.field_strength``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_field_strength_target``
     - ``data.expose.field_strength_target``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_glm_design_matrix``
     - ``data.expose.glm_design_matrix``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_scanner_id``
     - ``data.expose.scanner_id``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.expose_site_id``
     - ``data.expose.site_id``
     - 2026-07-31
     - Eight flags whose shared `expose_` prefix was already naming the block. `expose.scanner_id: true` reads as the sentence it is.
   * - ``data.graph_config``
     - ``data.domain.graph_config``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.graph_type``
     - ``data.domain.graph_type``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.hf_resolution``
     - ``data.pairing.hf_resolution``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.holdout_subject``
     - ``data.split.holdout_subject``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.input_artifact``
     - ``data.domain.input_artifact``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.kspace_scale_domain``
     - ``data.processing.kspace_scale_domain``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.loso_fold``
     - ``data.split.loso_fold``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.max_train_subjects``
     - ``data.split.max_train_subjects``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.max_val_subjects``
     - ``data.split.max_val_subjects``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.mrixfields_max_resident_volumes``
     - ``data.mrixfields.max_resident_volumes``
     - 2026-07-31
     - Cohort-specific options for the paired ULF/HF source. Six fields sharing one prefix are a sub-block spelled the long way.
   * - ``data.mrixfields_output_contrast``
     - ``data.mrixfields.output_contrast``
     - 2026-07-31
     - Cohort-specific options for the paired ULF/HF source. Six fields sharing one prefix are a sub-block spelled the long way.
   * - ``data.mrixfields_pairing_policy``
     - ``data.mrixfields.pairing_policy``
     - 2026-07-31
     - Cohort-specific options for the paired ULF/HF source. Six fields sharing one prefix are a sub-block spelled the long way.
   * - ``data.mrixfields_rescale_per_image``
     - ``data.mrixfields.rescale_per_image``
     - 2026-07-31
     - Cohort-specific options for the paired ULF/HF source. Six fields sharing one prefix are a sub-block spelled the long way.
   * - ``data.mrixfields_slice_mode``
     - ``data.mrixfields.slice_mode``
     - 2026-07-31
     - Cohort-specific options for the paired ULF/HF source. Six fields sharing one prefix are a sub-block spelled the long way.
   * - ``data.mrixfields_target_field``
     - ``data.mrixfields.target_field``
     - 2026-07-31
     - Cohort-specific options for the paired ULF/HF source. Six fields sharing one prefix are a sub-block spelled the long way.
   * - ``data.num_synthetic_samples``
     - ``data.sampling.num_synthetic_samples``
     - 2026-07-31
     - How samples are DRAWN from a volume, before anything is done to them. The TorchIO queue's three knobs finally sit together: `patch_size` is what a sample IS, `samples_per_volume` how many are taken per subject per epoch, `queue_length` the RAM buffer holding them -- tuning one blind to the others is how a queue build comes to dominate a smoke run. `patch_size` is also the most overloaded name in the repo (~134 model attributes mean the ViT/MAE patch embedding, not this), and `sampling:` disambiguates it. `phase_encode_axis` stays FLAT: nothing reads it from config -- FMRIVolumeDataset takes it as a constructor default and no dataset_type selects that class.
   * - ``data.output_domain``
     - ``data.domain.output``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.preprocessing_dir``
     - ``data.source.preprocessing_dir``
     - 2026-07-31
     - WHERE the bytes come from. Six answers to one question, previously scattered across ~300 lines of `data:`. `dataset_type` is deliberately NOT folded: it is a DISPATCH key rather than a location -- dataset_instantiator, config_health_checker and axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan binds it to `workflow.regime`, and 636 arms declare it as the first line of their `data:` block where nothing about it is ambiguous. `datasets` and `manifest_roles` stay for a different reason: both are already nested blocks, and this phase groups scalars rather than re-parenting blocks. `data_root`/`data_layout` drop the `data_` prefix the outer block supplies; the `_path` leaves keep their full names, since under the naming rule `<thing>_path` is a file and index/validation_index/paired_manifest are the things.
   * - ``data.rescale_images``
     - ``data.processing.enable_image_rescale``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.rescale_percentiles``
     - ``data.processing.rescale_percentiles``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.sessions``
     - ``data.pairing.sessions``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.slice_2d``
     - ``data.sampling.enable_slice_2d``
     - 2026-07-31
     - How samples are DRAWN from a volume, before anything is done to them. The TorchIO queue's three knobs finally sit together: `patch_size` is what a sample IS, `samples_per_volume` how many are taken per subject per epoch, `queue_length` the RAM buffer holding them -- tuning one blind to the others is how a queue build comes to dominate a smoke run. `patch_size` is also the most overloaded name in the repo (~134 model attributes mean the ViT/MAE patch embedding, not this), and `sampling:` disambiguates it. `phase_encode_axis` stays FLAT: nothing reads it from config -- FMRIVolumeDataset takes it as a constructor default and no dataset_type selects that class.
   * - ``data.svd_calibration_lines``
     - ``data.coils.svd_calibration_lines``
     - 2026-07-31
     - How multi-coil data is reduced before the model sees it. Note `num_virtual_coils` keeps its full name inside `coils:` -- the naming rule is `num_<thing>` and the thing is virtual coils.
   * - ``data.target_artifact``
     - ``data.domain.target_artifact``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.target_contrasts``
     - ``data.pairing.target_contrasts``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.target_sessions``
     - ``data.pairing.target_sessions``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``deep_supervision_weight *(root)*``
     - ``losses.lambda_deep_supervision``
     - 2026-07-31
     - A loss weight sitting at the config root. Every other lambda lives under `losses.` and resolves through the loss-weight SSOT (pitfall #13b).
   * - ``device *(root)*``
     - ``run.device``
     - 2026-07-31
     - A bare scalar at the document root, where a reader looks first and learns least. Resolution is unchanged: still only through `spectramr.core.compute_device` (non-negotiable 9b).
   * - ``logging.anomaly_check_interval``
     - ``logging.intervals.anomaly_check``
     - 2026-07-31
     - Every-N cadences. The block supplies the word `interval`, so the leaves are the things being timed rather than four repetitions of the suffix.
   * - ``logging.debug_log_steps``
     - ``logging.snapshots.log_steps``
     - 2026-07-31
     - The per-step tensor/JSON debug dumps. The block is `snapshots`, NOT the plan's `debug_snapshots`: that is one of the retired scalars' own names, and a destination block may not share a legacy leaf name -- a migrated arm writing `debug_snapshots: {enabled: true}` would have the whole block folded into itself. The `debug_` prefix is redundant besides: a snapshot is a debug artifact.
   * - ``logging.debug_snapshot_interval_steps``
     - ``logging.snapshots.interval_steps``
     - 2026-07-31
     - The per-step tensor/JSON debug dumps. The block is `snapshots`, NOT the plan's `debug_snapshots`: that is one of the retired scalars' own names, and a destination block may not share a legacy leaf name -- a migrated arm writing `debug_snapshots: {enabled: true}` would have the whole block folded into itself. The `debug_` prefix is redundant besides: a snapshot is a debug artifact.
   * - ``logging.debug_snapshot_max_calls``
     - ``logging.snapshots.max_calls``
     - 2026-07-31
     - The per-step tensor/JSON debug dumps. The block is `snapshots`, NOT the plan's `debug_snapshots`: that is one of the retired scalars' own names, and a destination block may not share a legacy leaf name -- a migrated arm writing `debug_snapshots: {enabled: true}` would have the whole block folded into itself. The `debug_` prefix is redundant besides: a snapshot is a debug artifact.
   * - ``logging.debug_snapshot_save_images``
     - ``logging.snapshots.save_images``
     - 2026-07-31
     - The per-step tensor/JSON debug dumps. The block is `snapshots`, NOT the plan's `debug_snapshots`: that is one of the retired scalars' own names, and a destination block may not share a legacy leaf name -- a migrated arm writing `debug_snapshots: {enabled: true}` would have the whole block folded into itself. The `debug_` prefix is redundant besides: a snapshot is a debug artifact.
   * - ``logging.debug_snapshot_save_json``
     - ``logging.snapshots.save_json``
     - 2026-07-31
     - The per-step tensor/JSON debug dumps. The block is `snapshots`, NOT the plan's `debug_snapshots`: that is one of the retired scalars' own names, and a destination block may not share a legacy leaf name -- a migrated arm writing `debug_snapshots: {enabled: true}` would have the whole block folded into itself. The `debug_` prefix is redundant besides: a snapshot is a debug artifact.
   * - ``logging.debug_snapshots``
     - ``logging.snapshots.enabled``
     - 2026-07-31
     - The per-step tensor/JSON debug dumps. The block is `snapshots`, NOT the plan's `debug_snapshots`: that is one of the retired scalars' own names, and a destination block may not share a legacy leaf name -- a migrated arm writing `debug_snapshots: {enabled: true}` would have the whole block folded into itself. The `debug_` prefix is redundant besides: a snapshot is a debug artifact.
   * - ``logging.report_cases_subdir``
     - ``logging.report_cases.subdir``
     - 2026-07-31
     - Per-case artifacts kept for the end-of-training report.
   * - ``logging.save_report_cases``
     - ``logging.report_cases.enabled``
     - 2026-07-31
     - Per-case artifacts kept for the end-of-training report.
   * - ``logging.tensorboard_dir``
     - ``logging.tracking.tensorboard_dir``
     - 2026-07-31
     - The experiment-tracking backend. `enable_experiment_tracking` becomes the block's bare `enabled` gate. The two `wandb_*` fields are NOT folded in: W&B is deferred, so a non-null value now RAISES rather than being silently accepted and ignored (issue #675, `LoggingConfigSchema._refuse_deferred_wandb`).
   * - ``losses.disable_default_losses``
     - ``losses.policy.exclude_defaults``
     - 2026-07-31
     - HOW the objective is assembled, as opposed to WHICH terms are in it. `losses:` is otherwise sixteen loss-family blocks, so a loose scalar beside them reads as a seventeenth; neither of these is a family. `disable_default_losses` becomes `exclude_defaults` and does NOT use the `negate` transform: the naming rule forbids a negated boolean, but this field is a `list[str]` of loss NAMES, so inverting it is meaningless -- `enable_default_losses: ['mse']` would mean 'enable ONLY mse', the opposite of a filter. `normalize_losses`, `clip_loss_value` and `loss_scaling` are NOT folded in: all three have zero readers and zero corpus declarations (issue #676), and `lambda_deep_supervision` is already correctly placed as a loss WEIGHT under the naming rule.
   * - ``losses.reconstruction.enable_bloch_residual``
     - ``losses.physics.enable_bloch_residual``
     - 2026-08-23
     - Gate for the retired `lambda_bloch_residual` above. Zero readers, zero corpus declarations (#421).
   * - ``losses.reconstruction.enable_content``
     - ``losses.reconstruction.enable_content_consistency``
     - 2026-08-23
     - Gate renamed alongside `lambda_content` -> `lambda_content_consistency`, so the pair states which of the two losses it gates. Zero readers, zero corpus declarations (#421).
   * - ``losses.reconstruction.enable_physics_constraint``
     - ``losses.physics.enable_physics_constraint``
     - 2026-08-23
     - Gate for the retired `lambda_physics_constraint` above. Zero readers, zero corpus declarations (#421).
   * - ``losses.reconstruction.enable_snr_preserving``
     - ``losses.physics.enable_snr_preserving``
     - 2026-08-23
     - Gate for the retired `lambda_snr_preserving` above. Zero readers, zero corpus declarations (#421).
   * - ``losses.reconstruction.lambda_bloch_residual``
     - ``losses.physics.lambda_bloch_residual``
     - 2026-08-23
     - Duplicated `losses.physics.lambda_bloch_residual`, which is the spelling `physics_driven_strategy.py:362` actually reads. Both canonicalise to `bloch_residual`, at defaults 0.0 and 1.0, so a materialised config raised a false conflict (#421). Zero corpus arms declared this spelling.
   * - ``losses.reconstruction.lambda_content``
     - ``losses.reconstruction.lambda_perceptual``
     - 2026-08-23
     - Overloaded key with TWO successors -- pick by meaning. For the VGG content/perceptual weight use `losses.reconstruction.lambda_perceptual` (the registry aliases `content` -> `perceptual`, and this is what 100% of measured corpus usage meant). For the DISENTANGLEMENT content-consistency term use `losses.reconstruction.lambda_content_consistency`, which canonicalises to itself and so no longer collides. The one corpus arm that declared this (`kspace_filling/experiment_11_kspace_cold_diffusion_perceptual.yaml`) already carried `lambda_perceptual: 0.1` at the identical value, so its migration was a pure line deletion with a provably unchanged weight table (#421).
   * - ``losses.reconstruction.lambda_physics_constraint``
     - ``losses.physics.lambda_physics_constraint``
     - 2026-08-23
     - Duplicated `losses.physics.lambda_physics_constraint`, which is the spelling `loss_builder.py:473` reads as `lambda_dc`. Both canonicalise to `physics_constraint`, at defaults 0.0 and 0.1, so a materialised config raised a false conflict (#421). Zero corpus arms declared this spelling.
   * - ``losses.reconstruction.lambda_snr_preserving``
     - ``losses.physics.lambda_snr_preserving``
     - 2026-08-23
     - Duplicated `losses.physics.lambda_snr_preserving`; both canonicalise to `snr_preserving`, which made a materialised config raise a false weight conflict (#421). Zero corpus arms declared this spelling. NOTE that the elected winner is itself unread -- it stays in the `test_schema_key_consumption.py` allowlist as pre-existing non-negotiable 8 debt, tracked separately; this record only elects the owner.
   * - ``model_domain *(root)*``
     - ``model.model_domain``
     - 2026-07-31
     - The root field was a write-only alias: NOTHING read it. A `mode='before'` validator copied it into `model.model_domain` / `model.target_domain`, which are the fields consumers actually read. Write the nested key directly.
   * - ``optimization.accumulate_grad_steps``
     - ``optimization.gradient.accumulation_steps``
     - 2026-07-31
     - Two spellings of one number. Only the canonical one is read by the training loop; the legacy name was folded in by a hand-written validator, so which one won depended on declaration order rather than intent.
   * - ``optimization.amp_dtype``
     - ``optimization.precision.dtype``
     - 2026-07-31
     - AMP is one decision with two parts. Split across a `use_*` boolean and a dtype 40 lines apart, the third state — dtype 'float32' disables AMP even when the flag is true — was unreadable.
   * - ``optimization.amsgrad``
     - ``optimization.optimizer.amsgrad``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.detect_anomalies``
     - ``optimization.gradient.detect_anomalies``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.gradient_checkpointing``
     - ``optimization.gradient.enable_checkpointing``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.lookahead``
     - ``optimization.optimizer.lookahead``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.nesterov``
     - ``optimization.optimizer.nesterov``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.param_group_overrides``
     - ``optimization.optimizer.param_groups``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``reporting.per_case_metrics``
     - ``reporting.per_call_metrics``
     - 2026-08-05
     - The name promised a distribution the artifact never carried. The sink writes one row per validation CALL holding that call's BATCH-AGGREGATE metrics -- there is no per-sample value to record, because the run computes metrics batch-wise. A plotter believed the name and rendered eight copies of one scalar as a box-and-whisker (#503). Zero corpus arms declared the key, so this retires at `raise` rather than folding.
   * - ``seed *(root)*``
     - ``run.seed``
     - 2026-07-31
     - Two spellings of one seed. `training.seed` WON at runtime (pipelines/train.py prefers it), so 96 arms setting the root key were writing the loser. Both now name `run.seed`.
   * - ``training.bloch_field.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the bloch_field strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.brenier_synthesis.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the brenier_synthesis strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.cartoon_texture_safe.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the cartoon_texture_safe strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.field_conditioned_inr.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the field_conditioned_inr strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.field_fno.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the field_fno strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.field_wiener.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the field_wiener strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.fisher_rao_geodesic.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the fisher_rao_geodesic strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.koopman_field.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the koopman_field strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.lora_modulation.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the lora_modulation strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.mccann_field_path.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the mccann_field_path strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.monotone_field.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the monotone_field strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.recoverability_vib.lambda_recon``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the recoverability_vib strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.scattering_besov.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the scattering_besov strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.seed``
     - ``run.seed``
     - 2026-07-31
     - The seed is a fact about the RUN, not about the training paradigm — it also drives data shuffling and augmentation, neither of which lives under `training:`.
   * - ``training.steerable_synthesis.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the steerable_synthesis strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``training.ulf_redegrad_tta.lambda_l1``
     - ``losses.image_losses``
     - 2026-09-03
     - The inline L1 weight of the ulf_redegrad_tta strategy was read from its own training sub-block while losses.image_losses documented another value (16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses (an author-written losses.reconstruction.lambda_l1 still wins there). The 43 arms that declared this key were drained, so it retires at raise.
   * - ``validation.hallucination_test``
     - ``validation.gates.hallucination_test``
     - 2026-07-31
     - The checks that can FAIL a run, as opposed to `metrics`, which only measures. Each exists to catch a way a run can score well while meaning nothing: a measurement-independent DC blob (#20), an in-distribution-only severity range, hallucinated structure. Grouping them puts the anti-facade gates where a reader looks for them instead of scattered among cadence and batch-size knobs.
   * - ``validation.held_out_severity_eval``
     - ``validation.gates.held_out_severity_eval``
     - 2026-07-31
     - The checks that can FAIL a run, as opposed to `metrics`, which only measures. Each exists to catch a way a run can score well while meaning nothing: a measurement-independent DC blob (#20), an in-distribution-only severity range, hallucinated structure. Grouping them puts the anti-facade gates where a reader looks for them instead of scattered among cadence and batch-size knobs.
   * - ``validation.input_dependence_tol``
     - ``validation.gates.input_dependence_tol``
     - 2026-07-31
     - The checks that can FAIL a run, as opposed to `metrics`, which only measures. Each exists to catch a way a run can score well while meaning nothing: a measurement-independent DC blob (#20), an in-distribution-only severity range, hallucinated structure. Grouping them puts the anti-facade gates where a reader looks for them instead of scattered among cadence and batch-size knobs.
   * - ``validation.multistep_cold_sampling``
     - ``validation.sampling.enable_multistep_cold``
     - 2026-07-31
     - Reverse-diffusion sampling AT VALIDATION TIME, where it differs from training. Both leaves shed the `sampl*` stem the block now supplies.
   * - ``validation.sampler_steps``
     - ``validation.sampling.steps``
     - 2026-07-31
     - Reverse-diffusion sampling AT VALIDATION TIME, where it differs from training. Both leaves shed the `sampl*` stem the block now supplies.
   * - ``validation.shuffle_validation``
     - ``validation.loader.shuffle``
     - 2026-07-31
     - HOW validation batches are drawn. The two batch-size spellings were not interchangeable: `effective_val_batch_size` and data_builder.py preferred the short `val_batch_size`, while training_pipeline_director.py read only the long `validation_batch_size` -- and 74 arms declare the two with different values, so the batch size a run used depended on which builder path it took. One field ends that. The `val_`/`validation_` prefixes go: the block already says which pass this is.
   * - ``validation.val_batch_size``
     - ``validation.loader.batch_size``
     - 2026-07-31
     - HOW validation batches are drawn. The two batch-size spellings were not interchangeable: `effective_val_batch_size` and data_builder.py preferred the short `val_batch_size`, while training_pipeline_director.py read only the long `validation_batch_size` -- and 74 arms declare the two with different values, so the batch size a run used depended on which builder path it took. One field ends that. The `val_`/`validation_` prefixes go: the block already says which pass this is.
   * - ``workflow.name``
     - ``workflow.regime``
     - 2026-07-31
     - `name` is the most overloaded key in the corpus — every loss-list entry and every `metadata:` block has one — so it read as a label for the block rather than the load-bearing physical claim it is. `regime` says what the value means.

Staged -- the old spelling still loads (101)
============================================

Accepted for now and folded into place. Migrate at your convenience;
``scripts/ci/check_no_legacy_config_keys.py`` prints how many remain.

.. list-table::
   :header-rows: 1
   :widths: 26 26 10 38

   * - Retired key
     - Replacement
     - Since
     - Why
   * - ``acceleration *(root)*``
     - ``undersampling``
     - 2026-08-02
     - The block name meant two unrelated things: the MRI k-space ACCELERATION FACTOR (base_acceleration, center_fraction, mask types, the schedule ladder -- 26 fields, what the block is actually for) and COMPUTE acceleration (mixed_precision, use_compile, use_distributed, use_gradient_checkpointing, gradient_accumulation_steps). A reader could not tell which sense a given key belonged to from the block name. The five compute knobs are all inert -- zero readers on an acceleration receiver -- and four of them duplicate a live `optimization.*` field; they stay flat inside the renamed block rather than being folded, so their inertness stays visible (issue #680).
   * - ``data.allow_unpaired``
     - ``data.pairing.allow_unpaired``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.batch_size``
     - ``data.loader.batch_size``
     - 2026-07-31
     - How samples are batched and moved to the device -- mechanical knobs that change throughput, never what the model sees. `max_prefetch` was a deprecated alias folded in by a hand-written validator; it lands on the same field here, so shim, fixer and gate finally read one table. No corpus arm declares both.
   * - ``data.bidirectional_mode``
     - ``data.pairing.bidirectional_mode``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.coil_processing_mode``
     - ``data.coils.processing_mode``
     - 2026-07-31
     - How multi-coil data is reduced before the model sees it. Note `num_virtual_coils` keeps its full name inside `coils:` -- the naming rule is `num_<thing>` and the thing is virtual coils.
   * - ``data.contrasts``
     - ``data.pairing.contrasts``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.data_range``
     - ``data.processing.data_range``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.data_root``
     - ``data.source.root``
     - 2026-07-31
     - WHERE the bytes come from. Six answers to one question, previously scattered across ~300 lines of `data:`. `dataset_type` is deliberately NOT folded: it is a DISPATCH key rather than a location -- dataset_instantiator, config_health_checker and axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan binds it to `workflow.regime`, and 636 arms declare it as the first line of their `data:` block where nothing about it is ambiguous. `datasets` and `manifest_roles` stay for a different reason: both are already nested blocks, and this phase groups scalars rather than re-parenting blocks. `data_root`/`data_layout` drop the `data_` prefix the outer block supplies; the `_path` leaves keep their full names, since under the naming rule `<thing>_path` is a file and index/validation_index/paired_manifest are the things.
   * - ``data.enable_slab_mode``
     - ``data.sampling.enable_slab_mode``
     - 2026-07-31
     - How samples are DRAWN from a volume, before anything is done to them. The TorchIO queue's three knobs finally sit together: `patch_size` is what a sample IS, `samples_per_volume` how many are taken per subject per epoch, `queue_length` the RAM buffer holding them -- tuning one blind to the others is how a queue build comes to dominate a smoke run. `patch_size` is also the most overloaded name in the repo (~134 model attributes mean the ViT/MAE patch embedding, not this), and `sampling:` disambiguates it. `phase_encode_axis` stays FLAT: nothing reads it from config -- FMRIVolumeDataset takes it as a constructor default and no dataset_type selects that class.
   * - ``data.holdout_site``
     - ``data.split.holdout_site``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.index_path``
     - ``data.source.index_path``
     - 2026-07-31
     - WHERE the bytes come from. Six answers to one question, previously scattered across ~300 lines of `data:`. `dataset_type` is deliberately NOT folded: it is a DISPATCH key rather than a location -- dataset_instantiator, config_health_checker and axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan binds it to `workflow.regime`, and 636 arms declare it as the first line of their `data:` block where nothing about it is ambiguous. `datasets` and `manifest_roles` stay for a different reason: both are already nested blocks, and this phase groups scalars rather than re-parenting blocks. `data_root`/`data_layout` drop the `data_` prefix the outer block supplies; the `_path` leaves keep their full names, since under the naming rule `<thing>_path` is a file and index/validation_index/paired_manifest are the things.
   * - ``data.kspace_percentile``
     - ``data.processing.kspace_percentile``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.log_scaling``
     - ``data.processing.enable_log_scaling``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.log_scaling_center_fraction``
     - ``data.processing.log_scaling_center_fraction``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.max_prefetch``
     - ``data.loader.prefetch_factor``
     - 2026-07-31
     - How samples are batched and moved to the device -- mechanical knobs that change throughput, never what the model sees. `max_prefetch` was a deprecated alias folded in by a hand-written validator; it lands on the same field here, so shim, fixer and gate finally read one table. No corpus arm declares both.
   * - ``data.normalization_kwargs``
     - ``data.processing.normalization_kwargs``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.normalization_type``
     - ``data.processing.normalization_type``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.normalize_images``
     - ``data.processing.enable_image_normalization``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.normalize_kspace``
     - ``data.processing.enable_kspace_normalization``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.num_virtual_coils``
     - ``data.coils.num_virtual_coils``
     - 2026-07-31
     - How multi-coil data is reduced before the model sees it. Note `num_virtual_coils` keeps its full name inside `coils:` -- the naming rule is `num_<thing>` and the thing is virtual coils.
   * - ``data.num_workers``
     - ``data.loader.num_workers``
     - 2026-07-31
     - How samples are batched and moved to the device -- mechanical knobs that change throughput, never what the model sees. `max_prefetch` was a deprecated alias folded in by a hand-written validator; it lands on the same field here, so shim, fixer and gate finally read one table. No corpus arm declares both.
   * - ``data.paired_manifest_path``
     - ``data.source.paired_manifest_path``
     - 2026-07-31
     - WHERE the bytes come from. Six answers to one question, previously scattered across ~300 lines of `data:`. `dataset_type` is deliberately NOT folded: it is a DISPATCH key rather than a location -- dataset_instantiator, config_health_checker and axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan binds it to `workflow.regime`, and 636 arms declare it as the first line of their `data:` block where nothing about it is ambiguous. `datasets` and `manifest_roles` stay for a different reason: both are already nested blocks, and this phase groups scalars rather than re-parenting blocks. `data_root`/`data_layout` drop the `data_` prefix the outer block supplies; the `_path` leaves keep their full names, since under the naming rule `<thing>_path` is a file and index/validation_index/paired_manifest are the things.
   * - ``data.patch_size``
     - ``data.sampling.patch_size``
     - 2026-07-31
     - How samples are DRAWN from a volume, before anything is done to them. The TorchIO queue's three knobs finally sit together: `patch_size` is what a sample IS, `samples_per_volume` how many are taken per subject per epoch, `queue_length` the RAM buffer holding them -- tuning one blind to the others is how a queue build comes to dominate a smoke run. `patch_size` is also the most overloaded name in the repo (~134 model attributes mean the ViT/MAE patch embedding, not this), and `sampling:` disambiguates it. `phase_encode_axis` stays FLAT: nothing reads it from config -- FMRIVolumeDataset takes it as a constructor default and no dataset_type selects that class.
   * - ``data.persistent_workers``
     - ``data.loader.persistent_workers``
     - 2026-07-31
     - How samples are batched and moved to the device -- mechanical knobs that change throughput, never what the model sees. `max_prefetch` was a deprecated alias folded in by a hand-written validator; it lands on the same field here, so shim, fixer and gate finally read one table. No corpus arm declares both.
   * - ``data.pin_memory``
     - ``data.loader.pin_memory``
     - 2026-07-31
     - How samples are batched and moved to the device -- mechanical knobs that change throughput, never what the model sees. `max_prefetch` was a deprecated alias folded in by a hand-written validator; it lands on the same field here, so shim, fixer and gate finally read one table. No corpus arm declares both.
   * - ``data.prefetch_factor``
     - ``data.loader.prefetch_factor``
     - 2026-07-31
     - How samples are batched and moved to the device -- mechanical knobs that change throughput, never what the model sees. `max_prefetch` was a deprecated alias folded in by a hand-written validator; it lands on the same field here, so shim, fixer and gate finally read one table. No corpus arm declares both.
   * - ``data.queue_length``
     - ``data.sampling.queue_length``
     - 2026-07-31
     - How samples are DRAWN from a volume, before anything is done to them. The TorchIO queue's three knobs finally sit together: `patch_size` is what a sample IS, `samples_per_volume` how many are taken per subject per epoch, `queue_length` the RAM buffer holding them -- tuning one blind to the others is how a queue build comes to dominate a smoke run. `patch_size` is also the most overloaded name in the repo (~134 model attributes mean the ViT/MAE patch embedding, not this), and `sampling:` disambiguates it. `phase_encode_axis` stays FLAT: nothing reads it from config -- FMRIVolumeDataset takes it as a constructor default and no dataset_type selects that class.
   * - ``data.rescale_range``
     - ``data.processing.rescale_range``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.samples_per_volume``
     - ``data.sampling.samples_per_volume``
     - 2026-07-31
     - How samples are DRAWN from a volume, before anything is done to them. The TorchIO queue's three knobs finally sit together: `patch_size` is what a sample IS, `samples_per_volume` how many are taken per subject per epoch, `queue_length` the RAM buffer holding them -- tuning one blind to the others is how a queue build comes to dominate a smoke run. `patch_size` is also the most overloaded name in the repo (~134 model attributes mean the ViT/MAE patch embedding, not this), and `sampling:` disambiguates it. `phase_encode_axis` stays FLAT: nothing reads it from config -- FMRIVolumeDataset takes it as a constructor default and no dataset_type selects that class.
   * - ``data.single_contrast``
     - ``data.pairing.single_contrast``
     - 2026-07-31
     - What counts as the INPUT and what counts as the TARGET. The plan called this block `contrast:`, which is wrong twice over: three of the eight fields are ULF/HF field-strength pairing with no contrast content, and `data.contrast` would sit beside the pre-existing `data.input_contrast`/`data.target_contrast` -- which are ContrastConfigSchema NORMALISATION specs (percentile, out_range, clamp), not selection filters. Leaf names are carried across unchanged for the same reason: `contrasts` is NOT renamed to `input_contrasts`, because singular-vs-plural is the worst available disambiguator against `data.input_contrast` and the block prefix already does that work. `bidirectional_mode` keeps its `_mode` spelling rather than becoming `direction`: it is a genuine mode (the `hf_to_hf` value DROPS the opposite arm), and 132 arms declare it.
   * - ``data.split_strategy``
     - ``data.split.type``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.target_channels``
     - ``data.domain.target_channels``
     - 2026-07-31
     - What representation the loader hands the model -- image or k-space, how many target channels, which artifact variants, and whether the sample arrives as a graph. `output_domain` drops its suffix inside `domain:` (the block supplies the noun, as with `expose:`), which also disambiguates it from `losses.output_domain`. `return_image_domain` stays FLAT: no reader in src/spectramr, already carried in KNOWN_UNCONSUMED.
   * - ``data.train_sites``
     - ``data.split.train_sites``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``data.transforms``
     - ``data.processing.transforms``
     - 2026-07-31
     - What is done to the VALUES between disk and the model, in the order it happens: k-space scaling, then image-domain normalisation and rescale, then config-driven transforms. The k-space group sat ~200 lines from the image group, so nothing showed they compose. The four booleans take the ratified `enable_<thing>` spelling; each now sits directly above the nouns it gates. `resample`/`crop_or_pad` stay put -- they were already nested blocks, and this phase groups scalars rather than re-parenting blocks.
   * - ``data.validation_index_path``
     - ``data.source.validation_index_path``
     - 2026-07-31
     - WHERE the bytes come from. Six answers to one question, previously scattered across ~300 lines of `data:`. `dataset_type` is deliberately NOT folded: it is a DISPATCH key rather than a location -- dataset_instantiator, config_health_checker and axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan binds it to `workflow.regime`, and 636 arms declare it as the first line of their `data:` block where nothing about it is ambiguous. `datasets` and `manifest_roles` stay for a different reason: both are already nested blocks, and this phase groups scalars rather than re-parenting blocks. `data_root`/`data_layout` drop the `data_` prefix the outer block supplies; the `_path` leaves keep their full names, since under the naming rule `<thing>_path` is a file and index/validation_index/paired_manifest are the things.
   * - ``data.validation_split``
     - ``data.split.validation_fraction``
     - 2026-07-31
     - What decides which records train and which validate. The strategy now sits next to the companions that make it valid -- `loso` needs `holdout_site`, `loso_subject` needs exactly one of `holdout_subject`/`loso_fold` -- which is precisely what `validate_split_strategy` checks. `validation_split` becomes `validation_fraction` because it is a fraction in [0, 1], not a split. `test_split` stays FLAT: nothing reads it (#665), and a tidy home for an inert knob implies it works.
   * - ``logging.enable_experiment_tracking``
     - ``logging.tracking.enabled``
     - 2026-07-31
     - The experiment-tracking backend. `enable_experiment_tracking` becomes the block's bare `enabled` gate. The two `wandb_*` fields are NOT folded in: W&B is deferred, so a non-null value now RAISES rather than being silently accepted and ignored (issue #675, `LoggingConfigSchema._refuse_deferred_wandb`).
   * - ``logging.enable_tensorboard``
     - ``logging.tracking.enable_tensorboard``
     - 2026-07-31
     - The experiment-tracking backend. `enable_experiment_tracking` becomes the block's bare `enabled` gate. The two `wandb_*` fields are NOT folded in: W&B is deferred, so a non-null value now RAISES rather than being silently accepted and ignored (issue #675, `LoggingConfigSchema._refuse_deferred_wandb`).
   * - ``logging.experiment_name``
     - ``logging.identity.experiment``
     - 2026-07-31
     - WHAT this run is called, for a human and for the tracking backend. The `_name` suffixes go: inside `identity:` there is nothing else `experiment` or `run` could denote.
   * - ``logging.level``
     - ``logging.sinks.level``
     - 2026-07-31
     - WHERE log lines go and how much of them. `level` and `silent` are not destinations but they decide what reaches one, so they read better here than three blocks away. The `log_` prefix goes -- the outer block already supplies it, which is why 41 arms wrote `log_level` and had it silently discarded (issue #675).
   * - ``logging.log_difference_images``
     - ``logging.images.log_difference``
     - 2026-07-31
     - Which image panels are written, and how many. The log/save distinction is kept on the leaf because the two are genuinely different sinks (TensorBoard vs disk) and bootstrap.py ORs them. `save_images_per_epoch` is NOT folded in: nothing reads it and 882 arms set it, so a tidy home would imply it works.
   * - ``logging.log_dir``
     - ``logging.sinks.dir``
     - 2026-07-31
     - WHERE log lines go and how much of them. `level` and `silent` are not destinations but they decide what reaches one, so they read better here than three blocks away. The `log_` prefix goes -- the outer block already supplies it, which is why 41 arms wrote `log_level` and had it silently discarded (issue #675).
   * - ``logging.log_input_images``
     - ``logging.images.log_input``
     - 2026-07-31
     - Which image panels are written, and how many. The log/save distinction is kept on the leaf because the two are genuinely different sinks (TensorBoard vs disk) and bootstrap.py ORs them. `save_images_per_epoch` is NOT folded in: nothing reads it and 882 arms set it, so a tidy home would imply it works.
   * - ``logging.log_interval``
     - ``logging.intervals.log``
     - 2026-07-31
     - Every-N cadences. The block supplies the word `interval`, so the leaves are the things being timed rather than four repetitions of the suffix.
   * - ``logging.log_to_console``
     - ``logging.sinks.to_console``
     - 2026-07-31
     - WHERE log lines go and how much of them. `level` and `silent` are not destinations but they decide what reaches one, so they read better here than three blocks away. The `log_` prefix goes -- the outer block already supplies it, which is why 41 arms wrote `log_level` and had it silently discarded (issue #675).
   * - ``logging.log_to_file``
     - ``logging.sinks.to_file``
     - 2026-07-31
     - WHERE log lines go and how much of them. `level` and `silent` are not destinations but they decide what reaches one, so they read better here than three blocks away. The `log_` prefix goes -- the outer block already supplies it, which is why 41 arms wrote `log_level` and had it silently discarded (issue #675).
   * - ``logging.log_validation_images``
     - ``logging.images.log_validation``
     - 2026-07-31
     - Which image panels are written, and how many. The log/save distinction is kept on the leaf because the two are genuinely different sinks (TensorBoard vs disk) and bootstrap.py ORs them. `save_images_per_epoch` is NOT folded in: nothing reads it and 882 arms set it, so a tidy home would imply it works.
   * - ``logging.max_images_per_batch``
     - ``logging.images.max_per_batch``
     - 2026-07-31
     - Which image panels are written, and how many. The log/save distinction is kept on the leaf because the two are genuinely different sinks (TensorBoard vs disk) and bootstrap.py ORs them. `save_images_per_epoch` is NOT folded in: nothing reads it and 882 arms set it, so a tidy home would imply it works.
   * - ``logging.notes``
     - ``logging.identity.notes``
     - 2026-07-31
     - WHAT this run is called, for a human and for the tracking backend. The `_name` suffixes go: inside `identity:` there is nothing else `experiment` or `run` could denote.
   * - ``logging.run_name``
     - ``logging.identity.run``
     - 2026-07-31
     - WHAT this run is called, for a human and for the tracking backend. The `_name` suffixes go: inside `identity:` there is nothing else `experiment` or `run` could denote.
   * - ``logging.save_interval``
     - ``logging.intervals.save``
     - 2026-07-31
     - Every-N cadences. The block supplies the word `interval`, so the leaves are the things being timed rather than four repetitions of the suffix.
   * - ``logging.save_validation_images``
     - ``logging.images.save_validation``
     - 2026-07-31
     - Which image panels are written, and how many. The log/save distinction is kept on the leaf because the two are genuinely different sinks (TensorBoard vs disk) and bootstrap.py ORs them. `save_images_per_epoch` is NOT folded in: nothing reads it and 882 arms set it, so a tidy home would imply it works.
   * - ``logging.silent``
     - ``logging.sinks.silent``
     - 2026-07-31
     - WHERE log lines go and how much of them. `level` and `silent` are not destinations but they decide what reaches one, so they read better here than three blocks away. The `log_` prefix goes -- the outer block already supplies it, which is why 41 arms wrote `log_level` and had it silently discarded (issue #675).
   * - ``logging.tags``
     - ``logging.identity.tags``
     - 2026-07-31
     - WHAT this run is called, for a human and for the tracking backend. The `_name` suffixes go: inside `identity:` there is nothing else `experiment` or `run` could denote.
   * - ``logging.tracking_service``
     - ``logging.tracking.service``
     - 2026-07-31
     - The experiment-tracking backend. `enable_experiment_tracking` becomes the block's bare `enabled` gate. The two `wandb_*` fields are NOT folded in: W&B is deferred, so a non-null value now RAISES rather than being silently accepted and ignored (issue #675, `LoggingConfigSchema._refuse_deferred_wandb`).
   * - ``logging.validation_image_interval``
     - ``logging.intervals.validation_images``
     - 2026-07-31
     - Every-N cadences. The block supplies the word `interval`, so the leaves are the things being timed rather than four repetitions of the suffix.
   * - ``losses.output_domain``
     - ``losses.policy.output_domain``
     - 2026-07-31
     - HOW the objective is assembled, as opposed to WHICH terms are in it. `losses:` is otherwise sixteen loss-family blocks, so a loose scalar beside them reads as a seventeenth; neither of these is a family. `disable_default_losses` becomes `exclude_defaults` and does NOT use the `negate` transform: the naming rule forbids a negated boolean, but this field is a `list[str]` of loss NAMES, so inverting it is meaningless -- `enable_default_losses: ['mse']` would mean 'enable ONLY mse', the opposite of a filter. `normalize_losses`, `clip_loss_value` and `loss_scaling` are NOT folded in: all three have zero readers and zero corpus declarations (issue #676), and `lambda_deep_supervision` is already correctly placed as a loss WEIGHT under the naming rule.
   * - ``optimization.beta1``
     - ``optimization.optimizer.beta1``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.beta2``
     - ``optimization.optimizer.beta2``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.betas``
     - ``optimization.optimizer.betas``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.compile_backend``
     - ``optimization.compile.backend``
     - 2026-07-31
     - A `compile_` prefix repeated five times is a sub-block spelled the long way.
   * - ``optimization.compile_dynamic``
     - ``optimization.compile.dynamic``
     - 2026-07-31
     - A `compile_` prefix repeated five times is a sub-block spelled the long way.
   * - ``optimization.compile_fullgraph``
     - ``optimization.compile.fullgraph``
     - 2026-07-31
     - A `compile_` prefix repeated five times is a sub-block spelled the long way.
   * - ``optimization.compile_mode``
     - ``optimization.compile.mode``
     - 2026-07-31
     - A `compile_` prefix repeated five times is a sub-block spelled the long way.
   * - ``optimization.compile_model``
     - ``optimization.compile.enabled``
     - 2026-07-31
     - A `compile_` prefix repeated five times is a sub-block spelled the long way.
   * - ``optimization.discriminator_learning_rate``
     - ``optimization.optimizer.discriminator_learning_rate``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.enable_gradient_clipping``
     - ``optimization.gradient.clip.enabled``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.enable_memory_fragmentation_mitigation``
     - ``optimization.memory.enable_fragmentation_mitigation``
     - 2026-07-31
     - Diagnostics, not a training decision — and the `memory_` prefix repeated six times was already naming the block.
   * - ``optimization.enable_memory_monitoring``
     - ``optimization.memory.enable_monitoring``
     - 2026-07-31
     - Diagnostics, not a training decision — and the `memory_` prefix repeated six times was already naming the block.
   * - ``optimization.eps``
     - ``optimization.optimizer.eps``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.generator_learning_rate``
     - ``optimization.optimizer.generator_learning_rate``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.gradient_accumulation_steps``
     - ``optimization.gradient.accumulation_steps``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.gradient_clip_method``
     - ``optimization.gradient.clip.method``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.gradient_clip_value``
     - ``optimization.gradient.clip.value``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.learning_rate``
     - ``optimization.optimizer.learning_rate``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.memory_cleanup_interval``
     - ``optimization.memory.cleanup_interval``
     - 2026-07-31
     - Diagnostics, not a training decision — and the `memory_` prefix repeated six times was already naming the block.
   * - ``optimization.memory_monitoring_interval``
     - ``optimization.memory.monitoring_interval``
     - 2026-07-31
     - Diagnostics, not a training decision — and the `memory_` prefix repeated six times was already naming the block.
   * - ``optimization.memory_safety_margin``
     - ``optimization.memory.safety_margin``
     - 2026-07-31
     - Diagnostics, not a training decision — and the `memory_` prefix repeated six times was already naming the block.
   * - ``optimization.momentum``
     - ``optimization.optimizer.momentum``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.optimize_batch_size_for_memory``
     - ``optimization.memory.enable_batch_size_optimization``
     - 2026-07-31
     - Diagnostics, not a training decision — and the `memory_` prefix repeated six times was already naming the block.
   * - ``optimization.optimizer_kwargs``
     - ``optimization.optimizer.kwargs``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.optimizer_type``
     - ``optimization.optimizer.type``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``optimization.use_amp``
     - ``optimization.precision.enabled``
     - 2026-07-31
     - AMP is one decision with two parts. Split across a `use_*` boolean and a dtype 40 lines apart, the third state — dtype 'float32' disables AMP even when the flag is true — was unreadable.
   * - ``optimization.use_gradient_checkpointing``
     - ``optimization.gradient.enable_checkpointing``
     - 2026-07-31
     - Everything between `loss.backward()` and `optimizer.step()`. Note the two checkpointing spellings collapse to ONE field here — `gradient_checkpointing` was an alias declared only so it would validate under extra=forbid, which is one knob too many.
   * - ``optimization.weight_decay``
     - ``optimization.optimizer.weight_decay``
     - 2026-07-31
     - Which optimizer, at what rate, with which hyper-parameters is one decision; it was spread across 15 keys interleaved with memory and compile knobs. `eps` declared next to `memory_safety_margin` is how #624 (eps twice, momentum on adamw) stayed invisible.
   * - ``training.diffusion.num_timesteps``
     - ``training.diffusion.timesteps``
     - 2026-08-12
     - Two spellings of the timestep count, and the corpus was told to use the one nothing reads. `DiffusionTrainingConfigSchema` declares `timesteps`; `num_timesteps` was absorbed by that class's `extra='allow'` into an untyped attribute, while THREE places advertised it as canonical -- `validation_constants.py`, `base.py`'s own path list, and a `loss.py` field description saying 'DEPRECATED: use training.diffusion.num_timesteps'. The reader takes the other spelling (`cold_diffusion_inference_strategy.py` reads `diffusion_config.get('timesteps', 1000)`) and one site pops `num_timesteps` outright. 28 arms declare it, and 3 of them were RUNNING WRONG: they asked for 100 timesteps and got 1000. The other 25 asked for 1000, which is also the default, so they were unaffected by luck rather than by design (issue #980).
   * - ``validation.compute_image_metrics``
     - ``validation.scoring.enable_image_metrics``
     - 2026-07-31
     - WHAT validation measures. The block is `scoring` rather than the obvious `metrics` because `validation.metrics` is one of the keys being retired here, and the fold matches on the key ALONE: a block named `metrics` would make that key mean both the old scalar and its own destination, so a legacy arm would break on declaration order and a MIGRATED arm writing `metrics: {compute: [...]}` would have the whole block folded into itself. The list leaf is `compute` for a plainer reason -- it matches the standing `metrics.compute_*` -> `metrics.compute: [psnr, ssim]` migration, so the two surfaces read the same way -- and `compute_image_metrics` becomes `enable_image_metrics` because it is a switch, which `compute_` beside a `compute` list would disguise. `validation_metric` is deliberately NOT folded onto `primary` despite naming the same idea: the defaults differ ('loss' vs 'psnr'), so folding would silently repoint the 53 arms that set it.
   * - ``validation.domain``
     - ``validation.scoring.domain``
     - 2026-07-31
     - WHAT validation measures. The block is `scoring` rather than the obvious `metrics` because `validation.metrics` is one of the keys being retired here, and the fold matches on the key ALONE: a block named `metrics` would make that key mean both the old scalar and its own destination, so a legacy arm would break on declaration order and a MIGRATED arm writing `metrics: {compute: [...]}` would have the whole block folded into itself. The list leaf is `compute` for a plainer reason -- it matches the standing `metrics.compute_*` -> `metrics.compute: [psnr, ssim]` migration, so the two surfaces read the same way -- and `compute_image_metrics` becomes `enable_image_metrics` because it is a switch, which `compute_` beside a `compute` list would disguise. `validation_metric` is deliberately NOT folded onto `primary` despite naming the same idea: the defaults differ ('loss' vs 'psnr'), so folding would silently repoint the 53 arms that set it.
   * - ``validation.enable_visualization``
     - ``validation.visualization.enabled``
     - 2026-07-31
     - Validation image dumps: the gate and its cadence. `enable_visualization` becomes the block's bare `enabled` -- the naming rule reserves that spelling for exactly this. `num_visualizations` and `visualization_dir` are NOT folded in: nothing reads either (both sit in KNOWN_UNCONSUMED), and giving an inert knob a tidy home implies it works.
   * - ``validation.eval_interval``
     - ``validation.schedule.interval_steps``
     - 2026-07-31
     - HOW OFTEN validation runs. Four spellings of one cadence, two of them the same number: `eval_interval` and `frequency_steps` share a default of 1000, 58 arms declare both, and not one disagrees -- so merging them is a measurement, not a guess. Only `eval_interval` was ever read (pipelines/training_loop.py), which means the merge also carries 58 arms' previously-dead `frequency_steps` into the loop. `on_epoch` and `interval_epochs` are NOT duplicates and both survive: the first selects epoch-based mode, the second is its N.
   * - ``validation.eval_on_epoch``
     - ``validation.schedule.on_epoch``
     - 2026-07-31
     - HOW OFTEN validation runs. Four spellings of one cadence, two of them the same number: `eval_interval` and `frequency_steps` share a default of 1000, 58 arms declare both, and not one disagrees -- so merging them is a measurement, not a guess. Only `eval_interval` was ever read (pipelines/training_loop.py), which means the merge also carries 58 arms' previously-dead `frequency_steps` into the loop. `on_epoch` and `interval_epochs` are NOT duplicates and both survive: the first selects epoch-based mode, the second is its N.
   * - ``validation.frequency_epochs``
     - ``validation.schedule.interval_epochs``
     - 2026-07-31
     - HOW OFTEN validation runs. Four spellings of one cadence, two of them the same number: `eval_interval` and `frequency_steps` share a default of 1000, 58 arms declare both, and not one disagrees -- so merging them is a measurement, not a guess. Only `eval_interval` was ever read (pipelines/training_loop.py), which means the merge also carries 58 arms' previously-dead `frequency_steps` into the loop. `on_epoch` and `interval_epochs` are NOT duplicates and both survive: the first selects epoch-based mode, the second is its N.
   * - ``validation.frequency_steps``
     - ``validation.schedule.interval_steps``
     - 2026-07-31
     - HOW OFTEN validation runs. Four spellings of one cadence, two of them the same number: `eval_interval` and `frequency_steps` share a default of 1000, 58 arms declare both, and not one disagrees -- so merging them is a measurement, not a guess. Only `eval_interval` was ever read (pipelines/training_loop.py), which means the merge also carries 58 arms' previously-dead `frequency_steps` into the loop. `on_epoch` and `interval_epochs` are NOT duplicates and both survive: the first selects epoch-based mode, the second is its N.
   * - ``validation.metrics``
     - ``validation.scoring.compute``
     - 2026-07-31
     - WHAT validation measures. The block is `scoring` rather than the obvious `metrics` because `validation.metrics` is one of the keys being retired here, and the fold matches on the key ALONE: a block named `metrics` would make that key mean both the old scalar and its own destination, so a legacy arm would break on declaration order and a MIGRATED arm writing `metrics: {compute: [...]}` would have the whole block folded into itself. The list leaf is `compute` for a plainer reason -- it matches the standing `metrics.compute_*` -> `metrics.compute: [psnr, ssim]` migration, so the two surfaces read the same way -- and `compute_image_metrics` becomes `enable_image_metrics` because it is a switch, which `compute_` beside a `compute` list would disguise. `validation_metric` is deliberately NOT folded onto `primary` despite naming the same idea: the defaults differ ('loss' vs 'psnr'), so folding would silently repoint the 53 arms that set it.
   * - ``validation.num_samples``
     - ``validation.loader.num_samples``
     - 2026-07-31
     - HOW validation batches are drawn. The two batch-size spellings were not interchangeable: `effective_val_batch_size` and data_builder.py preferred the short `val_batch_size`, while training_pipeline_director.py read only the long `validation_batch_size` -- and 74 arms declare the two with different values, so the batch size a run used depended on which builder path it took. One field ends that. The `val_`/`validation_` prefixes go: the block already says which pass this is.
   * - ``validation.num_validation_batches``
     - ``validation.loader.num_batches``
     - 2026-07-31
     - HOW validation batches are drawn. The two batch-size spellings were not interchangeable: `effective_val_batch_size` and data_builder.py preferred the short `val_batch_size`, while training_pipeline_director.py read only the long `validation_batch_size` -- and 74 arms declare the two with different values, so the batch size a run used depended on which builder path it took. One field ends that. The `val_`/`validation_` prefixes go: the block already says which pass this is.
   * - ``validation.output_transform``
     - ``validation.scoring.output_transform``
     - 2026-07-31
     - WHAT validation measures. The block is `scoring` rather than the obvious `metrics` because `validation.metrics` is one of the keys being retired here, and the fold matches on the key ALONE: a block named `metrics` would make that key mean both the old scalar and its own destination, so a legacy arm would break on declaration order and a MIGRATED arm writing `metrics: {compute: [...]}` would have the whole block folded into itself. The list leaf is `compute` for a plainer reason -- it matches the standing `metrics.compute_*` -> `metrics.compute: [psnr, ssim]` migration, so the two surfaces read the same way -- and `compute_image_metrics` becomes `enable_image_metrics` because it is a switch, which `compute_` beside a `compute` list would disguise. `validation_metric` is deliberately NOT folded onto `primary` despite naming the same idea: the defaults differ ('loss' vs 'psnr'), so folding would silently repoint the 53 arms that set it.
   * - ``validation.primary_metric``
     - ``validation.scoring.primary``
     - 2026-07-31
     - WHAT validation measures. The block is `scoring` rather than the obvious `metrics` because `validation.metrics` is one of the keys being retired here, and the fold matches on the key ALONE: a block named `metrics` would make that key mean both the old scalar and its own destination, so a legacy arm would break on declaration order and a MIGRATED arm writing `metrics: {compute: [...]}` would have the whole block folded into itself. The list leaf is `compute` for a plainer reason -- it matches the standing `metrics.compute_*` -> `metrics.compute: [psnr, ssim]` migration, so the two surfaces read the same way -- and `compute_image_metrics` becomes `enable_image_metrics` because it is a switch, which `compute_` beside a `compute` list would disguise. `validation_metric` is deliberately NOT folded onto `primary` despite naming the same idea: the defaults differ ('loss' vs 'psnr'), so folding would silently repoint the 53 arms that set it.
   * - ``validation.val_chunk_size``
     - ``validation.loader.chunk_size``
     - 2026-07-31
     - HOW validation batches are drawn. The two batch-size spellings were not interchangeable: `effective_val_batch_size` and data_builder.py preferred the short `val_batch_size`, while training_pipeline_director.py read only the long `validation_batch_size` -- and 74 arms declare the two with different values, so the batch size a run used depended on which builder path it took. One field ends that. The `val_`/`validation_` prefixes go: the block already says which pass this is.
   * - ``validation.validation_batch_size``
     - ``validation.loader.batch_size``
     - 2026-07-31
     - The long spelling of the validation batch size. It LOSES to `val_batch_size`, which both field descriptions and the retired `effective_val_batch_size` property already named as the winner -- and which `ValidationConfigSchema._resolve_batch_size_duplicate` already enforces at parse time. The migrator reads the same `superseded_by` field, so the fixer and the validator cannot disagree about which spelling wins.
   * - ``validation.visualization_interval``
     - ``validation.visualization.interval``
     - 2026-07-31
     - Validation image dumps: the gate and its cadence. `enable_visualization` becomes the block's bare `enabled` -- the naming rule reserves that spelling for exactly this. `num_visualizations` and `visualization_dir` are NOT folded in: nothing reads either (both sit in KNOWN_UNCONSUMED), and giving an inert knob a tidy home implies it works.
