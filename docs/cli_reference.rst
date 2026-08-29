CLI Reference & Entry-Point Scheme
==================================

MRIForge exposes a single installed entry point, the ``mriforge`` console
script, which maps to :func:`mriforge.cli.app.main` (``pyproject.toml``
``[project.scripts]``). Every operation is a subcommand of that one parser.

.. code-block:: bash

   mriforge <command> [options]
   # equivalently:
   python -m mriforge.cli <command> [options]

Dispatch architecture
---------------------

There is **one** argparse parser (in :mod:`mriforge.cli.app`). Each subcommand
sets a handler via ``set_defaults(func=...)`` and ``main()`` calls
``args.func(args)``.

The training commands (``train`` / ``sanity_check``) call
:func:`mriforge.main.train_command` / :func:`mriforge.main.sanity_check_command`
**directly** with the parsed args. They do *not* rebuild ``sys.argv`` and
re-invoke a second parser.

.. note::

   **History (2026-05-29 cleanup).** Previously ``train``/``sanity_check``
   reconstructed a fake ``sys.argv`` string list and called
   ``mriforge.main:main()``, which re-parsed everything through a *second,
   divergent* argparse. That dual-parser seam silently dropped flags —
   ``--device`` and ``--seed`` never reached training, and ``--debug`` raised
   *unrecognized arguments* because ``main.py``'s train parser didn't define
   it. The handlers now dispatch directly, and the ``app.py`` train/sanity
   subparsers carry the full flag set (``--config``, ``--device``, ``--seed``,
   ``--dry-run``/``--dry_run``, ``--override``, ``--resume``, ``--debug``).

   ``mriforge.cli.app`` is now the **single** parser. The four commands that
   used to live only in ``main.py`` — ``infer``, ``infer-dataset``,
   ``experiment``, and ``train-distributed`` — were ported into it (each
   delegates to the existing ``main.py`` handler), so ``mriforge infer-dataset``
   and friends now work through the console script. ``mriforge.main:main`` is a
   deprecation shim: ``python -m mriforge.main <cmd> ...`` still works (it warns
   and delegates to ``mriforge.cli.app:main``), but new scripts should call
   ``mriforge`` directly. The per-command functions in ``main.py``
   (``train_command``, ``infer_command``, ``experiment_command``, …) remain
   importable.

.. note::

   **Override traversal safety (2026-07-01).** ``--override DOTTED.PATH=VALUE``
   may build a *new* nested path (absent/``None`` intermediate nodes are
   created), but a path that traverses an **existing non-dict** node — e.g.
   ``optimization.learning_rate.typo=1`` — now raises a ``ValueError``
   naming the offending node instead of silently replacing the subtree with
   ``{}`` (pitfall #9; on permissive ``dict[str, Any]`` fields the Pydantic
   re-validation could not catch the loss).

   **Determinism/seed wiring (2026-07-01).** ``infer``, ``infer-dataset``, and
   ``experiment`` now resolve ``training.seed`` and ``training.deterministic``
   from the ``--config`` YAML (``experiment`` after applying overrides),
   mirroring ``train``. Previously they hardcoded
   ``initialize_accelerator(device, 42)`` — both knobs were silent no-ops on
   those verbs (pitfall #15). An absent knob still resolves to
   ``deterministic=True``.

Distributed (multi-GPU) training launches the same console parser under
``torchrun`` (this is what the SLURM backend emits)::

   torchrun --nproc_per_node=N -m mriforge.cli train-distributed --config <arm>.yaml

Common commands
---------------

.. code-block:: bash

   # Train (config is the SSOT; loaded once into a frozen TrainingSettings)
   mriforge train --config experiments/inprogress/<paradigm>/<arm>.yaml
   mriforge train -c <arm>.yaml --device cpu --seed 7 --dry-run
   mriforge train -c <arm>.yaml -O optimization.learning_rate=1e-4

   # Overfit a single batch (collapse-vs-bug diagnostic)
   mriforge sanity_check -c <arm>.yaml

   # Pre-flight a config (Tier 0+1 ~100 ms; --probe adds Tier 2 forward pass)
   mriforge audit <arm>.yaml [--probe]

   # Bulk-audit a whole cohort (positional is a directory → recurse all YAMLs)
   mriforge audit experiments/inprogress/<cohort>

   # ...skipping ablation arms to focus the audit on training arms.
   # --exclude PATTERN is fnmatch glob over the path relative to the directory
   # (a bare substring is wrapped as *PATTERN*); repeatable. Use -path-style
   # '*ablation*' so it catches BOTH ablations/ subdirs and *ablation* filenames.
   mriforge audit experiments/inprogress/<cohort> --exclude '*ablation*'
   mriforge audit experiments/inprogress/<cohort> --exclude '*ablation*' --exclude '*baseline*'

Generating reports (``report``)
-------------------------------

``mriforge report`` builds the figures + tables (and the quality-control QC
report) from an already-downloaded run directory — the same
:func:`mriforge.infrastructure.reporting.generate_report` orchestrator the
end-of-training hook calls, so the output is identical whether training triggers
it or you invoke it by hand. **Every registered figure is attempted** (data-less
ones soft-skip), so the report includes all figures the reporting module
supports; set ``reporting.figures`` in a ``--config`` YAML to restrict. It reads ``logs/training_metrics.csv`` /
``final_metrics.json`` (via the aggregator) plus recorded ``report_cases`` images
— or, when those npz cases are absent, the downloaded ``real_images`` /
``fake_images`` PNG pairs.

.. code-block:: bash

   # Report from a downloaded run (figures + tables + qc_report.html)
   mriforge report --exp-dir experiments/results/<run>/

   # Reuse a config's reporting: block verbatim (parity with the training hook)
   mriforge report -e experiments/results/<run>/ -c experiments/inprogress/<paradigm>/<arm>.yaml

   # Toggle the QC figures / HTML wrapper / interactive layer (override the config)
   mriforge report -e <run>/ --no-html            # figures only, skip the HTML
   mriforge report -e <run>/ --qc --task reconstruction
   mriforge report -e <run>/ --no-interactive     # static-only HTML (no plotly layer)

   # Batch: report EVERY run under a cohort root + a linking report_index.html
   mriforge report --exp-dir experiments/results/ --recursive
   mriforge report -e experiments/results/vf/ --all -c <arm>.yaml   # config parity per run

Outputs land under ``<exp-dir>/<out-subdir>/`` (default ``report/``): the vector
+ raster figures, ``report_summary.md``, ``report_manifest.json``, and the
self-contained ``qc_report.html``. When :mod:`plotly` is installed
(``pip install mriforge[viz]``) and ``--interactive`` is on (the default) the HTML
carries an interactive layer — hoverable group IQMs, interactive learning curves /
metric distributions / Bland-Altman, and **2-D + 3-D MRIQC-style slice viewers**
(scrub subjects/slices, flick Prediction/Target/\|Error\|). plotly.js is inlined
once so the report works **offline** (no CDN); ``--no-interactive`` or a missing
plotly falls back to static PNGs. The 3-D viewer appears only when the run
recorded volumes (``reporting.record_volumes``); see :doc:`reporting_pipeline`.
Both ``--interactive`` and the config's ``reporting.interactive`` are forwarded
(CLI wins). With ``--recursive`` (alias ``--all``/``-r``) ``--exp-dir`` is treated
as a *cohort root*: every run beneath it (any dir carrying
``logs/training_metrics.csv`` / ``final_metrics.json`` / ...) is reported and a
top-level ``report_index.html`` links them all; a failing run is logged and
skipped rather than aborting the batch. To fire the single-run report
automatically at the end of training, add a ``reporting:`` block
(``enabled: true``) to the run's YAML; see :doc:`reporting_pipeline`.

Cluster diagnostics & global flags
----------------------------------

``mriforge doctor`` prints an environment snapshot — no job launched — so you can
confirm a node is set up before submitting:

.. code-block:: bash

   mriforge doctor                       # version, torch/CUDA, devices+memory,
                                        # cudnn flags, cache/data roots, MRIFORGE_* knobs, SLURM
   mriforge doctor --json                # machine-readable (for SLURM prologue checks)
   mriforge doctor -c <arm>.yaml         # also validate the config loads against the schema
   mriforge doctor --require-cuda        # exit non-zero if no GPU is visible (pre-flight GATE)

``doctor`` is import-safe: torch is probed defensively, so it still runs (and
reports the failure) on a broken / CPU-only environment — exactly when you need
it. ``--require-cuda`` / ``-c`` make it a gate you can chain before ``train`` in
a SLURM script (``mriforge doctor --require-cuda -c arm.yaml && mriforge train -c arm.yaml``).

Global flags (before the subcommand):

* ``--version`` — print the installed version and exit.
* ``-v`` / ``--verbose`` — print full tracebacks on error (otherwise a concise
  one-line error is logged; the env var ``MRIFORGE_DEBUG=1`` does the same).

**Error boundary.** ``main()`` wraps the dispatch: ``Ctrl-C`` exits ``130`` with
a clean message (no raw ``KeyboardInterrupt`` traceback in the SLURM log), a
handler's own ``sys.exit`` code is preserved, and any other exception becomes a
concise error + exit ``1`` (full traceback only under ``-v`` / ``MRIFORGE_DEBUG``).

Ablation studies
----------------

``mriforge ablation`` trains the baseline config plus one variant per
``--vary`` override, then writes ``ablation_results.json`` with the
baseline→variant delta and percent change per validation metric.

.. code-block:: bash

   mriforge ablation \
       --config experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml \
       --vary model.model_kwargs.force_pure_kspace=false \
       --output-dir experiments/results/exp11_fpk_ablation \
       --device cuda

* ``--vary DOTTED.PATH=VALUE`` (repeatable) — each defines one variant. The
  value is type-coerced exactly like ``--override`` (``false`` → ``bool``,
  ``1e-4`` → ``float``, ``none`` → ``None``). A spec without ``=`` raises at
  parse time (no silent fallback).
* ``--max-iterations N`` — caps iterations per arm for quick local sweeps
  (folded into every variant *and* the baseline for a fair comparison).
* ``--output-dir`` — defaults to ``<config_stem>_ablation/``.

Under the hood the command calls
:func:`mriforge.pipelines.ablation.run_ablation_study` with the default
training-backed evaluator
:func:`mriforge.pipelines.ablation.train_and_score`, which trains each config
via :func:`mriforge.pipelines.train.run_training_pipeline` and reads the best
validation metrics back from ``validation_metrics.csv``.

When to use ``ablation`` vs ``campaign`` vs ``hpo``
---------------------------------------------------

All three run more than one configuration; they differ in scale and venue:

.. list-table::
   :header-rows: 1
   :widths: 14 30 30 26

   * - Command
     - What it does
     - Where / how
     - Use when
   * - ``ablation``
     - Baseline + N single-knob ``--vary`` variants; automated delta report
     - **Local, sequential**, one process
     - A few variants, interactive, single GPU
   * - ``campaign``
     - Many declared arms (``CampaignConfigSchema``); leaderboard + reports
     - **Cluster (SLURM)**, parallel
     - Large studies, many arms
   * - ``hpo``
     - Optuna search over a base YAML; each trial a subprocess
     - Local or cluster, sampler/pruner-driven
     - Tuning continuous/categorical hyperparameters

.. warning::

   ``ablation`` trains the baseline and every variant **sequentially in one
   process**. For many arms or long runs, prefer ``campaign`` so arms run in
   parallel on the cluster.

Command wiring status (2026-06-11 audit)
----------------------------------------

Every subcommand was traced from the parser to its pipeline/use-case. Current
state:

* **Working:** ``train``, ``sanity_check``, ``ablation``, ``infer``,
  ``train_distributed``, ``benchmark``, ``export``, ``audit``,
  ``campaign {submit,status,evaluate,cancel,watch}``, ``hpo``, ``report``,
  ``meta_evaluate``, ``doctor``, and the late-wired ``audit-ksd`` /
  ``infer-protocol`` / ``simulate-acquisition`` / ``design-mrf-sequence`` /
  ``regulatory`` subcommands.
* **``experiment`` — fixed (2026-06-11).** It was dead-on-arrival: it built an
  ``ExperimentDirector`` whose ``validate()`` raised unconditionally (it
  required generator/loss config the CLI never supplied), exiting 1 before any
  training ran. It now drops the director and translates ``--experiment`` /
  ``--max-epochs`` / ``--checkpoint-interval`` into config overrides
  (``training.output_dir`` / ``training.epochs`` / ``checkpoint.save_interval``),
  then runs the canonical ``run_training_pipeline`` — i.e. it's a thin
  experiment-named wrapper over ``train``.
* **Deprecated:** ``predict`` (uses the deprecated
  ``application.pipelines.inference_pipeline.InferencePipeline`` and calls
  ``torch.load``/``torch.save`` directly — prefer ``infer``); ``infer_dataset``
  (an explicit alias for ``infer``, kept for cluster-job compatibility).
* **Orphan pipelines — removed (2026-06-11).** ``KoopmanAdvectionPipeline`` and
  ``QuantumImplicitNeRFPipeline`` (``pipelines/pipeline_{a,b}_*.py``) were
  redundant model+loss composites (``forward(clean, deformed, …) → loss dict``),
  reachable from no subcommand. They are **not** distinct models: their cores are
  already first-class registered models — ``neural_advection`` and
  ``dynamic_mr_nerf`` (both ``@register_model(training_mode="virtual_fiducial")``)
  — their losses are registered (``koopman_linearity``, ``hyperelastic_jacobian``)
  and the Koopman filter is a block (``models/temporal/koopman_operator.py``). So
  they ran via the standard ``run_training_pipeline`` + virtual-fiducial strategy
  with no special command; registering the composites would merely *duplicate*
  those models. The wrappers were deleted; express the methods through
  ``model.model_type`` + ``objectives`` instead. (``experiment_pipeline_b`` already
  uses ``model_type: dynamic_mr_nerf``; ``experiment_pipeline_a`` still names a
  placeholder ``hyper_mamba_unet`` — switch it to ``neural_advection`` +
  the koopman/jacobian losses to actually run the named method.)

Startup performance & the first-import wait (2026-06-19)
--------------------------------------------------------

The first heavy verb (``train`` / ``audit`` / ``infer`` / …) in a fresh process
must import **PyTorch + the model registry** (transitively monai / torchio). Cold,
that is tens of seconds — during which the terminal previously looked frozen
("after the import line it waits for a whole minute"). Two changes make the
unified entry point well-behaved:

* **The light paths stay light.** ``import mriforge`` (PEP 562 lazy ``__getattr__``)
  and ``mriforge --help`` / ``build_parser()`` never import torch. Two things that
  *used* to leak the heavy graph into a light path are fixed:

  - :mod:`mriforge.main` no longer imports ``run_training_pipeline`` at module
    top-level — it pulled the whole pipeline → registry → monai/torchio graph the
    instant *anything* imported the module (e.g. ``from mriforge.main import
    _parse_value`` for the ``ablation`` verb). It is now imported **lazily** inside
    ``__common_train_setup`` / ``experiment_command``, so ``import mriforge.main``
    is cheap and a malformed config fails *before* the heavy import on the train
    path. Pinned by ``tests/unit/test_main_lazy_imports.py``.

* **The unavoidable wait is now legible, not silent.** Before dispatching a heavy
  verb, ``main()`` prints one concise line to **stderr** (never stdout, so
  ``audit --json | jq`` stays parseable) via ``_emit_startup_notice``::

     ⏳ mriforge audit: importing PyTorch + model registry (first call in a fresh process is slow, ~30–60 s)…

  It fires only for the heavy verbs (``train``, ``sanity_check``, ``ablation``,
  ``infer``, ``infer-dataset``, ``experiment``, ``train-distributed``, ``predict``,
  ``benchmark``, ``export``, ``list-features``, ``audit``, ``hpo``, ``report``,
  ``meta-evaluate``) — the lightweight verbs (``doctor``, ``campaign``,
  ``regulatory``, ``launch``) stay quiet. Suppress it in batch jobs with
  ``MRIFORGE_QUIET=1`` **or** the existing ``MRIFORGE_SUPPRESS_CLINICAL_WARNING=1``
  (the same switch that silences the clinical banner also silences the notice).
  Pinned by ``tests/unit/cli/test_app_startup_notice.py``.

.. note::

   This addresses *perceived* latency and the leakage of the heavy graph into the
   light paths. The wall-clock of an actual ``train`` / single-file ``audit`` is
   still dominated by the one-time registry import (which genuinely needs every
   model's ``__init__`` signature for the ``advertised_options`` check); that
   import is not removed, only made visible and confined to the paths that need it.
