Execution Modes
===============

spectraMR separates *what* you run from *where* and *how many* — so the launch
surface is a small set of orthogonal axes rather than a pile of separate
"modes". Pick a value on each axis:

============== ====================================== ==============================================
Axis           Question                               Values
============== ====================================== ==============================================
WHAT           which config                           a training YAML (or a campaign manifest)
WHICH-PIPELINE which verb (must accept ``--config``)   ``train`` · ``sanity_check`` · ``infer`` · ``hpo`` · ``ablation`` · ``experiment``
WHERE          where it executes                      ``local`` · ``docker`` · ``apptainer`` · ``slurm``
HOW-MANY       one run or a sweep                     ``single`` · ``campaign``
============== ====================================== ==============================================

Plus a fourth, *imperative* surface that removes the WHAT axis entirely — you
write Python instead of a config: see :doc:`scripting_api`.

The unified launcher: ``spectramr launch``
----------------------------------------

``launch`` addresses the cube directly. The dedicated commands
(``spectramr train``, ``spectramr campaign submit``, ``sbatch …``) still work —
``launch`` is an additive front door over the same machinery.

.. code-block:: bash

   # WHAT × WHICH-PIPELINE × WHERE × HOW-MANY
   spectramr launch X.yaml                                   # train, local, single (defaults)
   spectramr launch X.yaml --where slurm --gpus 2            # train on SLURM (2 GPUs)
   spectramr launch X.yaml --where docker                    # train in a Docker container
   spectramr launch X.yaml --pipeline infer --where local \
       -- --checkpoint best.pt --input d/ --output o/      # a different verb; '--' passes verb args
   spectramr launch C.yaml --fanout campaign --where slurm   # a whole campaign

``--dry-run`` prints the exact command / sbatch script that *would* run without
executing it — useful for inspecting the resolved resources:

.. code-block:: bash

   $ spectramr launch X.yaml --where slurm --gpus 2 --mem 64G --dry-run
   #!/bin/bash
   #SBATCH --job-name=spectramr-train
   #SBATCH --account=<your-slurm-account>
   ...
   #SBATCH --gpus=2
   python -m spectramr.cli train --config X.yaml

Resource flags — ``--account`` · ``--partition`` · ``--mem`` · ``--gpus`` ·
``--time`` · ``--nodes`` — populate a single ``ResourceSpec``. ``account`` and
``partition`` fall back to ``$SPECTRAMR_SLURM_ACCOUNT`` / ``$SPECTRAMR_SLURM_PARTITION``,
so a non-default user need not edit anything to submit.

Job arrays
~~~~~~~~~~

``ResourceSpec.array`` (a SLURM array spec like ``"0-9"``, ``"0-19%4"``, or
``"1,3,5"``) adds a ``#SBATCH --array=…`` line directly after ``--gpus`` — the
single source of truth for fanning one script out over a manifest of arms,
replacing the hand-written array ``.sbatch`` files. It is ``None`` by default, so
every non-array caller (and the campaign golden) renders a byte-identical
directive block; an empty/whitespace spec is rejected rather than emitting an
invalid ``--array=`` (pitfall #15).

The per-task body — resolve the config for THIS array index from a manifest,
the ``TRAIN_ITERS`` smoke cap, the Tier-0/1 audit pre-flight, then ``train`` —
is the torch-free SSOT :mod:`spectramr.cli.manifest_dispatch` (WS-4 PR-B), so
``scripts/training/dispatch_experiments.sbatch`` is now a thin shell::

   python -m spectramr.cli.manifest_dispatch --manifest M.txt --index $SLURM_ARRAY_TASK_ID \
       [--train-iters N] [--resume] [--no-audit] [--dry-run]

The ``TRAIN_ITERS`` cap is a ``yaml.safe_load`` parse there, not a
``grep | sed | … || true`` pipeline: ``safe_load`` strips inline comments
natively and a missing ``eval_interval`` is just ``None``, eliminating the
``set -e`` foot-gun that aborted 56 array tasks in production. The shell keeps
only the REPO_ROOT spool anchor and the fail-loud venv/torch probe (their
regression tests run torch-free and must not reach Python before the venv); the
``--dry-run`` path is torch-free, so it runs the unit tests without a GPU.

Header SSOT (no hand-maintained directives)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The committed ``#SBATCH`` blocks of the array scripts are **generated from**
:meth:`~spectramr.infrastructure.execution.backends.SlurmBackend.render_directives`
(WS-4 PR-C), not hand-maintained, so the directive set, line order, and values
of the launcher and the committed headers can never drift:

* ``submit_exp11_fpk_ablation.sbatch`` is a **fixed-range** array (exactly two
  arms) → its header carries a committed ``#SBATCH --array=0-1`` line, rendered
  from ``ResourceSpec(array="0-1", …)``.
* ``dispatch_experiments.sbatch`` is a **dynamic-range** array (the wrapper
  ``submit_experiment_array.sh`` passes ``--array`` on the ``sbatch`` CLI to
  match the manifest length) → its committed header carries **no** ``--array``
  line (``array=None``). The wrapper also forwards optional per-launch resource
  overrides on the same ``sbatch`` CLI — ``SLURM_TIME`` → ``--time``,
  ``SLURM_MEM`` → ``--mem``, ``SLURM_GPUS`` → ``--gpus`` (sbatch's
  ``[type:]count`` syntax — a bare count like ``2`` lets the chosen
  account/partition/cluster decide the hardware, while ``<type>:count`` like
  ``ada:2`` / ``volta:4`` pins a GPU model; and ``SLURM_ACCOUNT`` /
  ``SLURM_PARTITION``). Each is emitted only
  when set and **wins over** the committed header default (120:00:00 / 128GB /
  gpus=1),
  so the header-parity test is unaffected (the committed directives don't
  change). ``SLURM_GPUS`` must be a positive count or ``<type>:count`` or the
  wrapper aborts.

A header-parity test for each script (``tests/unit/scripts/test_*.py``) asserts
the committed ``#SBATCH`` lines equal ``render_directives(...)`` for the
script's ``ResourceSpec``, so a hand-reorder or a one-off value tweak fails CI.
To change a directive, change the ``ResourceSpec`` and regenerate — do not edit
the ``.sbatch`` header in place.

Provenance
----------

When a run is started through ``spectramr launch``, the resolved backend +
``ResourceSpec`` are handed to the child via ``SPECTRAMR_LAUNCH_*`` environment
variables (inherited in-process, by SLURM through ``--export``, and forwarded
into containers with ``--env``). The run's ``run_summary.json`` then records
them under a ``"launch"`` block — so a result is traceable to *where* and *with
what resources* it ran, not just *what* config. A plain ``spectramr train`` (not
launched) leaves the block ``null``.

Choosing a WHERE
----------------

- **local** — runs in *this* process (no subprocess), the fastest path for
  development and the default.
- **docker** — ``docker run --gpus <N> -v $PWD:/workspace <image> <verb> …``;
  image overridable via ``$SPECTRAMR_DOCKER_IMAGE``.
- **apptainer** — ``apptainer run --nv --env CUDA_VISIBLE_DEVICES=0,…,N-1
  --bind $PWD:/workspace <sif> <verb> …`` for HPC; sif overridable via
  ``$SPECTRAMR_APPTAINER_SIF``.
- **slurm** — generates and submits an ``#SBATCH`` script from the
  ``ResourceSpec``.

.. note::

   The ``--gpus`` *count* is honoured by **every** backend, default ``1``:
   **slurm** emits a generic ``#SBATCH --gpus=N`` (no GPU type pinned, so the
   chosen account/partition/cluster decides the hardware); **docker** emits
   ``--gpus N`` (the container is
   re-numbered to see exactly N devices); **apptainer** emits ``--nv`` plus
   ``CUDA_VISIBLE_DEVICES=0,…,N-1``. The sentinel ``--gpus 0`` means **all
   visible GPUs** (``--gpus all`` / unrestricted ``--nv``). A negative count is
   rejected at startup (pitfall #9).

Choosing a HOW-MANY
-------------------

- **single** — one run of one config.
- **campaign** — a manifest of many arms with comparative evaluation; see
  :doc:`campaigns_user_guide`. ``launch --fanout campaign --where slurm`` (default)
  submits each arm as an sbatch job; ``--where docker`` / ``apptainer`` run each
  arm in a container (synchronously, parallel-mode campaigns only). ``--where
  local`` is not supported for campaigns.

.. seealso::

   * :doc:`scripting_api` — the imperative ``fit`` / ``Trainer`` surface.
   * :doc:`plugins` — custom components defined outside the source tree.
   * :doc:`cli_reference` — the full per-command reference.
   * :doc:`campaigns_user_guide` — campaign manifests and evaluation.
