Running Pipelines: Modes & Lifecycle
====================================

MRIForge runs every paradigm (GAN, diffusion, VAE/VQ-VAE, SSL/MAE,
reconstruction-only, domain-adaptation, physics-driven, virtual-fiducial, …)
through **one config and one entry point**. You do not pick the paradigm on the
command line — it is selected by ``training.training_mode`` in the YAML and
resolved to a strategy by ``TrainingStrategyFactory``. What you *do* pick on the
command line is the **run mode**: validate, smoke-test, train, resume, ablate,
sweep, infer, or report.

Three orthogonal axes
---------------------

Keep these separate in your head — they compose freely:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Axis
     - Decided by
     - Covered in
   * - **WHAT** (paradigm)
     - ``training.training_mode`` in the config
     - :doc:`config_schema_reference`
   * - **WHICH** (run mode)
     - the CLI verb + flags
     - **this page**
   * - **WHERE / HOW-MANY**
     - the launcher backend + single/campaign
     - :doc:`execution_modes`, :doc:`campaigns_user_guide`

The same config flows, frozen and loaded once, through whichever mode you run —
so a config that trains is the same object that gets audited, dry-run, smoke-
tested, swept, and served.

Two equivalent entry points
---------------------------

Every example below works with either spelling:

.. code-block:: bash

   mriforge <verb> [options]            # the installed console script
   python -m mriforge.cli <verb> [options]   # module form (identical parser)

``mriforge --help`` lists every verb; ``mriforge <verb> --help`` shows that verb's
flags. (Heavy verbs print a one-line ``⏳ importing PyTorch + model registry…``
notice on first use so the unavoidable cold import is not a silent wait — see
:doc:`cli_reference`.)

The recommended lifecycle
-------------------------

A config travels from *idea* to *result* through a sequence of progressively
heavier modes. Each step is cheap insurance against the next: catch a schema typo
in 100 ms (``audit``) before you discover it 40 minutes into a cluster job.

.. code-block:: text

   doctor → audit → train --dry-run → sanity_check → train → report → infer
   (env)    (config)  (wiring)         (does it learn?) (real)  (figures) (serve)

**0. Is my environment sane?** ``doctor`` — torch/CUDA, visible devices, cache
and data roots, env knobs. The cluster pre-flight gate.

.. code-block:: bash

   mriforge doctor --require-cuda            # exit non-zero if no GPU is visible
   mriforge doctor --config exp.yaml --json  # also confirm the YAML loads

**1. Is my config valid?** ``audit`` — the audit ladder. Tier 0 (Pydantic v6
schema) + Tier 1 (static cross-validation) in ~100 ms; add ``--probe`` for the
Tier-2 synthetic forward pass (~30 s, instantiates the model, catches AMP / shape
/ OOM). Note the config is a **positional** argument here, not ``--config``.

.. code-block:: bash

   mriforge audit experiments/inprogress/<paradigm>/<arm>.yaml          # Tier 0+1
   mriforge audit experiments/inprogress/<paradigm>/<arm>.yaml --probe  # + Tier 2
   mriforge audit experiments/inprogress/<paradigm>/ --strict           # bulk; warnings → errors

``--strict`` promotes every warning to an error (exit 2) — the smoke-wrapper
default. A directory argument audits every YAML beneath it and prints an aggregate
summary. See :doc:`audit_ladder_user_guide`.

**2. Does the whole thing wire up?** ``train --dry-run`` — loads the config,
builds the full DI container (model + losses + data + strategy), then stops
*before* the training loop. Proves every component resolves and the dataloader
constructs, without spending a single gradient step.

.. code-block:: bash

   mriforge train --config exp.yaml --dry-run     # --dry_run also accepted

**3. Does the model actually learn?** ``sanity_check`` — overfits a **single
batch**. It injects a fixed set of overrides (``data.batch_size=1``, EMA off,
``learning_rate=1e-4``, no warmup, constant LR) so the loss *must* collapse toward
zero if the model, losses, and gradients are wired correctly. A sanity check that
*cannot* overfit one batch is a bug you want to find before a full run.

.. code-block:: bash

   mriforge sanity_check --config exp.yaml --device cuda

**4. The real run.** ``train`` — the canonical training pipeline.

.. code-block:: bash

   mriforge train --config exp.yaml --device cuda --seed 42

   # tweak config values inline without editing the YAML (repeatable, nested keys):
   mriforge train -c exp.yaml -O optimization.learning_rate=1e-4 \
                             -O validation.val_interval=100

   # resume from a checkpoint (explicit path, or 'auto' for the latest in output_dir):
   mriforge train -c exp.yaml --resume experiments/results/exp/checkpoints/best.pt
   mriforge train -c exp.yaml --resume auto

``--override / -O`` uses **dotted nested paths** because the config is nested
(``config.optimization.learning_rate``, never ``config.lr``); each ``-O`` is one
key. Overrides are re-validated against the schema, so an illegal value still
fails loudly.

**5. Figures and tables.** ``report`` — runs the same reporting pipeline the
end-of-training hook uses, against an existing output directory. Idempotent: the
output is identical whether training triggered it or you invoke it by hand.

.. code-block:: bash

   mriforge report --exp-dir experiments/results/exp --task reconstruction \
                  --method "my-method"

**6. Serve a trained checkpoint.** ``infer`` — the SSOT inference pipeline. The
**training** YAML is required (it is the single source of truth for how to
reconstruct), plus the checkpoint and input.

.. code-block:: bash

   mriforge infer --config exp.yaml --checkpoint experiments/results/exp/checkpoints/best.pt \
                 --input data/test/ --output predictions/ --device cuda

.. note::

   ``predict`` (``--model``) and ``infer-dataset`` are **deprecated**. ``predict``
   bypasses the data SSOT (calls ``torch.load`` / ``torch.save`` directly) and
   ``infer-dataset`` is a back-compat alias for ``infer``. Prefer ``infer``.

Variant modes — same config, different question
-----------------------------------------------

These reuse the training pipeline but ask a comparative or search question.

**Ablation** — train the baseline plus one variant per ``--vary`` override, then
emit ``ablation_results.json`` with the per-metric baseline→variant delta. Runs
the arms **sequentially in-process**; for many arms or long runs use ``campaign``.

.. code-block:: bash

   mriforge ablation -c base.yaml \
       --vary model.model_kwargs.force_pure_kspace=false \
       --vary objectives.reconstruction.lambda_l1=0.5 \
       --max-iterations 2000 --output-dir experiments/results/exp_ablation --device cuda

**Hyperparameter optimization** — Optuna-backed search over a base config. Each
trial spawns a **separate trainer subprocess** so one trial crash (NaN / OOM)
can't poison the study. Define what may vary with a built-in preset or a search-
space YAML; with neither, every trial runs the base config unchanged.

.. code-block:: bash

   mriforge hpo --list-presets                       # discover built-in search spaces
   mriforge hpo --list-schema-paths                  # every tunable dotted-path key
   mriforge hpo --print-template > my_space.yaml     # starter search-space YAML

   mriforge hpo -c base.yaml -m unet --n-trials 50 \
       --search-preset lr_wd_curriculum --sampler tpe --pruner hyperband \
       --objective-metric val_loss --max-iter 30000 --device cuda

**Distributed (DDP)** — launched through ``torchrun``; the verb itself just wires
the DDP pipeline. The import-time env setup (cache root, thread isolation,
``PYTORCH_CUDA_ALLOC_CONF``) is preserved.

.. code-block:: bash

   torchrun --nproc_per_node=4 -m mriforge.cli train-distributed \
       --config exp.yaml --backend nccl --resume auto

Utility modes
-------------

.. code-block:: bash

   mriforge benchmark --suite all          # quality / throughput / memory micro-benchmarks
   mriforge export --model best.pt --config exp.yaml --format onnx   # ONNX / TorchScript
   mriforge list-features --module models --format markdown          # what's registered
   mriforge meta-evaluate ...              # rank a metric set (see meta-evaluation docs)

Scaling out & choosing where to run
-----------------------------------

The verbs above answer *which mode*. **Where** they run (this process, Docker,
Apptainer, or a SLURM job) and **how many** (one run or a whole campaign) are the
other two axes, handled by the unified launcher and campaign manifests:

.. code-block:: bash

   mriforge launch exp.yaml --where slurm --gpus 2          # train as a SLURM job
   mriforge launch exp.yaml --pipeline infer --where local -- --checkpoint best.pt --input d/
   mriforge launch campaign.yaml --fanout campaign --where slurm   # a whole sweep

``launch`` is an additive front door over the same machinery — every dedicated
verb still works on its own. ``--dry-run`` on ``launch`` prints the exact command
/ sbatch script that *would* run. See :doc:`execution_modes` for the full
WHAT × WHERE × HOW-MANY cube and :doc:`campaigns_user_guide` for sweeps.

Going config-free: the imperative API
-------------------------------------

When you'd rather write Python than a YAML — notebooks, quick experiments,
embedding training in a larger script — the imperative surface removes the config
axis entirely:

.. code-block:: python

   from mriforge import fit, make_model, make_dataloader, make_optimizer

   model = make_model("unet", in_channels=1, out_channels=1)
   ...

See :doc:`scripting_api` for ``fit`` / ``Trainer`` and the ``make_*`` builders.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Mode
     - Verb
     - Use it to…
   * - Environment
     - ``doctor``
     - confirm torch/CUDA, devices, roots
   * - Validate (static)
     - ``audit`` *(+ ``--probe``)*
     - catch schema / wiring / shape bugs fast
   * - Validate (wiring)
     - ``train --dry-run``
     - build the full container without training
   * - Smoke
     - ``sanity_check``
     - overfit one batch — does it learn?
   * - Train
     - ``train``
     - the real run (``-O``, ``--resume``)
   * - Resume
     - ``train --resume <path|auto>``
     - continue from a checkpoint
   * - Ablate
     - ``ablation --vary k=v``
     - baseline→variant deltas (sequential)
   * - Sweep
     - ``hpo``
     - Optuna search (subprocess-isolated)
   * - Distributed
     - ``torchrun … train-distributed``
     - multi-GPU DDP
   * - Infer
     - ``infer``
     - apply a checkpoint to new data
   * - Report
     - ``report``
     - figures + tables from an output dir
   * - Scale out
     - ``launch`` / ``campaign``
     - choose where / run many
   * - Config-free
     - ``mriforge.fit`` / ``Trainer``
     - imperative Python API

.. seealso::

   * :doc:`cli_reference` — full per-command flag reference and dispatch internals.
   * :doc:`execution_modes` — WHAT × WHERE × HOW-MANY launch cube (local / Docker / Apptainer / SLURM).
   * :doc:`campaigns_user_guide` — campaign manifests and comparative evaluation.
   * :doc:`audit_ladder_user_guide` — the Tier 0/1/2 audit ladder in depth.
   * :doc:`scripting_api` — the imperative ``fit`` / ``Trainer`` surface.
   * :doc:`config_schema_reference` — the v6.x config schema (``training.training_mode`` and friends).
