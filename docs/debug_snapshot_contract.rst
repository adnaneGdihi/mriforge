Debug snapshot contract
=======================

A debug snapshot exists to answer one question: **is the run doing what the
config asked for?** It can only answer it if the tensor names mean fixed things
and the artifact records what was done to the data. This page is the normative
statement of both.

.. contents::
   :local:
   :depth: 2

Why this contract exists
------------------------

An ``experiment_11`` cold-diffusion run was reported as "the input images look
fully sampled with phase-aligned averaging — they should look like a single NEX
that was zero-filled." The physics was correct: ``degradation_source: "input"``
was honoured and ``q_sample`` really is ``x_0 * mask``. The *snapshot* was
wrong, twice, and neither defect was visible from the artifact:

* ``first_steps/input_prepared`` is captured **before** the forward pass, and
  ``_prepare_model_input`` is a pure k-space ↔ image converter — identity for an
  arm whose loader already yields k-space. So the key named "prepared" held the
  clean, fully-sampled tensor, and nothing in the snapshot said so.
* The one snapshot that did hold the degraded tensor was gated on
  ``logging.intervals.log * 5``. With ``intervals.log: 5000`` that is step
  25 000, so it had never been written (fixed in PR #1177).

The reader had no way to falsify either reading, because the snapshot never
recorded what the config declared. That is the gap this contract closes.

The three canonical keys
------------------------

Every strategy's ``first_steps`` snapshot carries these three, and they mean
exactly this:

``input_raw``
   The **original data point**, as the dataloader delivered it, before any
   strategy-side preparation. This is what the ``data:`` block produced.

``input_prepared``
   The data **ready to be injected into the model** — domain-converted,
   normalized, and otherwise final. See the carve-out below when a strategy
   degrades further inside the step.

``target``
   The **ultimate ground truth** the loss grades against. It is **rendered in
   image space**: when the config declares that targets are k-space,
   ``target`` joins ``authoritative_kspace_keys`` and the previewer applies
   ``ifft2c`` unconditionally, bypassing the spectrum heuristic that
   false-negates on normalized multicoil M4Raw data.

All three must reflect **what the user specified in the configuration**. If the
arm declares ``acceleration.R: 8``, ``input_prepared`` shows 8× undersampling.
If it declares ``data.target_mode: phase_aligned_mean``, ``target`` is the
coherent NEX average and ``input_raw`` is one noisier repetition — the two
*should* look different, and the provenance block is what tells the reader that
this difference was ordered rather than accidental.

The in-step-degradation carve-out
---------------------------------

``first_steps`` is written before the forward pass, so a strategy whose forward
process transforms the prepared input *inside* the step cannot make
``input_prepared`` be the model input. Diffusion is the whole family: ``q_sample``
adds noise, or (cold) zero-fills with ``x_0 * mask``.

Such a strategy must do **both** of:

1. Set ``snapshot_prepared_is_model_input = False``. The base class stamps
   ``prepared_equals_model_input`` into ``first_steps``' ``extra``, so the
   artifact states plainly that this key is *not* the model input.
2. Emit its own snapshot carrying the real model input, and name its tag in
   ``snapshot_model_input_tag`` so the reader knows where to look.
   ``DiffusionTrainingStrategy`` sets ``"diffusion_step"``, whose
   ``noisy_kspace`` key is the degraded tensor actually fed to the network.

Both are class attributes, not config knobs: they state what the code does, so
they are not the user's to set and non-negotiable 8 does not apply.

Enforcement: the tag must name a snapshot that exists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 2 is a **pointer**, and for a long time nothing checked its target. The
carve-out is declared on ``DiffusionTrainingStrategy``, but ``diffusion_step``
was emitted only inside ``_prepare_diffusion_inputs`` — reachable solely through
``DiffusionTrainingStrategy._compute_losses_impl``, the hook subclasses override.
Seven subclasses therefore inherited the declaration and never emitted its
target: every artifact told a reader "the visible input is not the model input,
see ``diffusion_step``" and no such snapshot existed. That is a facade
(pitfall #16) in the sharpest form — the machine-readable claim contradicted the
artifact beside it.

Emission is therefore driven from ``BaseTrainingStrategy._compute_losses`` — the
**wrapper**, not the overridable ``_impl`` hook. This is the same wrapper/hook
boundary that caused the bug, used the other way round: an overriding subclass
cannot silently drop it.

The wrapper cannot capture the tensor itself. Flow matching's interpolant, EDM's
preconditioned ``c_in * noised``, ambient's ``q_sample`` output and the
cross-contrast bridge are different tensors produced by different math, each
local to its own ``_impl``. So the protocol is **declare-then-emit**::

    # inside _compute_losses_impl, right where the real model input is formed
    self._declare_model_input(
        {"model_input": x_t, "target": x_1},
        in_kspace_keys=set(),          # explicit; see the naming trap below
        extra={"model_input_key": "model_input", "note": "..."},
    )

The wrapper then emits the stash under ``snapshot_model_input_tag`` and
**raises** when the carve-out is declared and nothing was stashed. A future
subclass that declares the carve-out and forgets to emit now fails loud instead
of inheriting a dangling pointer (non-negotiable 3, no silent fallback).

A strategy that already emits its own snapshot under the declared tag satisfies
the contract without declaring anything — ``VirtualFiducialStrategy`` writes
``vf_twin`` directly. The satisfaction flag is set on the **attempt**, not on a
successful write, because ``save_debug_snapshot`` caps writes per
``(run_dir, tag)``: a write-based flag would start raising at step
``max_calls + 1`` of a perfectly healthy run.

The two halves of the check have deliberately different scope:

* **Missing tag** (carve-out declared, ``snapshot_model_input_tag`` unset) is a
  *static* defect — it depends only on class attributes, so it raises on every
  run, snapshots on or off.
* **Missing declaration** is an *artifact* defect. The contract binds what an
  artifact claims, so it is enforced only on runs that write artifacts, gated on
  ``snapshots_are_enabled``.

``snapshots_are_enabled`` is deliberately **not** ``snapshot_step_is_due``. The
latter folds in ``interval_steps`` and moves with the step counter; gating on it
would resurrect #706's shape, where the wrapper and the call site count steps
from different places and disagree the moment a caller omits ``iteration`` or a
run resumes mid-interval. With ``interval_steps: 100`` the site would declare at
step 0 and the wrapper would raise at step 1 — a violation manufactured by the
gate. The declaration is likewise unconditional at the call site, for the same
reason.

Getting it wrong in the other direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The carve-out is inherited, so a subclass can also carry it *undeservedly*.
``PaDNetTrainingStrategy`` subclasses ``DiffusionTrainingStrategy`` but runs no
forward process — it hands ``input_batch`` to ``predict_q_maps`` verbatim and the
Bloch physics happen downstream on the predicted q-maps. It therefore restates
``snapshot_prepared_is_model_input = True``: its artifacts had been
*under*-claiming an honest snapshot and pointing readers at a ``diffusion_step``
that neither exists nor should.

Conversely, a strategy that degrades in-step but does **not** subclass the
diffusion family inherits the default ``True`` and silently mislabels the
pre-degradation tensor as the model input. Both ``XDiffusionTrainingStrategy``
and ``ConcreteVirtualFiducialStrategy`` extend ``BaseTrainingStrategy``
directly and did exactly that.

**Check the direction, not just the presence, of the declaration.** The question
is whether *this* strategy transforms its input inside the step, never what its
parent does.

The naming trap
~~~~~~~~~~~~~~~

``save_debug_snapshot`` holds a fixed ``authoritative_kspace_keys`` set
(``input``, ``input_raw``, ``input_prepared``, ``target``, ``model_input``,
``model_output``, ``noisy_kspace``, ``model_output_pre_dc``,
``model_output_post_dc``) which is IFFT'd **unconditionally** for an arm whose
config declares k-space data, bypassing the spectrum veto. Naming a declared
tensor ``model_input`` is therefore safe **only when its domain follows the arm's
declared data domain** — true when the tensor is a linear function of the arm's
own ``target``, as it is for the noise/interpolant strategies.

It is *not* true for ``AmbientDiffusionStrategy``, whose input is image-domain
regardless (``ifft2c(...).real``). Naming it ``model_input`` would have the
previewer IFFT an already-image tensor, reproducing VIS-1; it uses
``ambient_x_t`` instead and points ``extra["model_input_key"]`` at that.

Pass ``in_kspace_keys`` **explicitly**, including the empty set. ``None`` falls
back to a ``"kspace"`` substring match over key names, which misses ``target``
entirely.

Tensors that superpose domains on the channel axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key is not always one treatment. Cold diffusion feeds the network
``torch.cat([noisy_images, smaps_k], dim=1)``. Both halves are even-channel
real-stacked ``float32`` in the same ``(R0,I0,R1,I1,...)`` interleaving, so
nothing about the tensor distinguishes them, and ``model_input`` is in the
authoritative set above — whatever the previewer decides, it decides for all
sixteen channels at once.

**Two facts, not one.** A tensor's *domain* (does it need ``ifft2c``?) and its
*compression* (does it need ``expm1``?) are separate questions. They travelled
together for as long as the maps half was image-domain — not k-space, therefore
not compressed, therefore neither transform. #1327 split them apart, by routing
the maps through ``prepare_smaps_for_kspace_conditioning``. A segment therefore
declares them separately, and the corpus has one live example of each failure.

*Before #1327 — the crosshair.* The maps were image-domain and the previewer
applied ``expm1`` + ``ifft2c`` to them anyway. The IFFT of a smooth,
DC-dominated image-domain field is a separable sinc-like kernel: a bright centre
pixel plus a horizontal and a vertical ridge. After the RSS combine that cross
owns the frame's energy, and the closing per-sample min-max divides everything
by it. On ``experiment_11_attention_none`` this rendered ``model_input.png`` as
a black frame with a white dot and a crosshair — reading as "the model input is
worse than a zero-filled image" while the k-space handed to the network was
fine.

*Since #1327 — the flattened spectrum.* The maps now reach the concat through
``fft2c``, level-matched and amplitude-capped, so that half IS k-space and DOES
need the inverse. What it still never had is the arm's ``log1p``: it goes into
the concat straight from the transform, while ``noisy_images`` descends from the
``apply_kspace_normalization``-compressed target. Deriving compression from
domain therefore applies ``expm1`` to an uncompressed spectrum, which clamps
every bin above ``DECOMPRESS_MAGNITUDE_CEILING`` to a single value — magnitude
flattened, phase intact — and renders the washed-out ringing "brain" that reads
as broken data. Same defect class as the crosshair, opposite direction.

**The renderer cannot catch either on its own.** ``decompress_for_view``'s
magnitude guard compares one scalar ``|x|.max()`` against
``DECOMPRESS_MAGNITUDE_CEILING``; on a mixed tensor the k-space half sets that
max well under the ceiling, so the guard passes and ``expm1`` still reaches the
maps. A single scalar cannot describe two treatments. The declaration has to
come from the emitting strategy, the only layer that knows what it concatenated.

Declare the decomposition with ``channel_segments``. A segment is
``(label, width, is_kspace)`` or, when its compression does not follow from its
domain, ``(label, width, is_kspace, log_compressed)``::

    self.save_debug_snapshot(
        snap,
        step=step,
        tag="model_output_dc",
        in_kspace_keys={"model_input", "smaps", ...},
        channel_segments={
            "model_input": [
                ("kspace", noisy_images.shape[1], True, True),
                # fft2c'd maps: k-space, but never log1p'd.
                ("smaps", smaps_part.shape[1], True, False),
            ]
        },
        # The standalone `smaps` key is the SAME tensor; only `log_scaled_keys`
        # can correct that one, and omitting it leaves the two renders of one
        # tensor disagreeing.
        log_scaled_keys={"noisy_kspace", "target", "model_input"},
    )

Each segment is rendered as its own ``<key>__<label>.png`` under its own domain
and compression. The stats table still reports the **undivided** tensor, so the
artifact continues to record the real model input — the split is a rendering
concern only, and the faithful record and the readable picture stop being in
tension.

The fourth field is optional and additive: a three-tuple segment inherits the
parent tensor's compression answer, which is exactly what every declaration
written before #1327 meant. Omitting it is right whenever domain and compression
still agree; reach for it only when a tensor genuinely disagrees with itself.

Three rules attach to it:

* **Never "fix" a bad render by narrowing** ``log_scaled_keys``. That suppresses
  a decompression the k-space half still requires and re-introduces #682 later;
  ``utils/kspace_view`` says so at length. Declaring that a *segment* was never
  compressed is a different statement — ``expm1`` was never correct for it.
* **Declare** ``log_scaled_keys`` **as soon as one recorded tensor is not
  post-normalization.** ``None`` means "every declared-k-space key is
  decompressed", which is right only when the snapshot captures exclusively
  compressed tensors. A conditioning half stashed under its own key is the case
  that breaks it.
* **Never slice the recorded tensor** down to its readable half. The artifact
  must show what the model was fed. Splitting the *picture* is the fix;
  splitting the *record* is a contract violation.

A declaration whose widths do not sum to the tensor's channel count is refused
with a warning and the tensor renders whole. A mis-declared split must be
visible, not silently approximated — otherwise a confident ``__kspace``
filename sits on a picture built from the wrong channels.

The provenance block
--------------------

``snapshot.json`` carries a top-level ``provenance`` key (and ``snapshot.txt`` a
rendered copy) built by
:func:`mriforge.infrastructure.training.snapshot_provenance.build_snapshot_provenance`.
It records two halves side by side, and **a divergence between them is the
finding**::

    "provenance": {
      "source": "train",
      "declared": {
        "target_mode": "phase_aligned_mean",
        "nex_target_exclude_input": false,
        "normalization": {
          "enable_kspace_normalization": true,
          "kspace_percentile": 0.99,
          "kspace_scale_domain": "image",
          "enable_log_scaling": false,
          "normalization_type": "percentile"
        },
        "augmentation_enabled": true
      },
      "applied": {
        "dataset_chain": ["SFCConformalFMRIKeysWrapper", "M4RawDataset"],
        "transforms": [
          {"name": "KSpaceNormalizationTransform", "params": {...}},
          {"name": "RandomAffine", "params": {"scales": [0.9, 1.1]}}
        ]
      },
      "model_input_linearization": [
        {"module": "HilbertOrder", "mode": "hilbert", "shape": [256, 256]}
      ],
      "incomplete": []
    }

``declared``
   Read from the config SSOT — the NEX target construction
   (``target_mode``, ``nex_target_exclude_input``, ``return_image_domain``) and
   the full ``data.processing`` normalization block.

``applied``
   Read from the objects the pipeline actually built: the dataset wrapper chain
   (unwrapped through ``inner`` / ``dataset`` / ``base_dataset`` /
   ``subjects_dataset``) and the real ``tio.Compose``, flattened in execution
   order with each transform's ``args_names`` parameters. The wrapper chain
   matters on its own — ``SFCConformalFMRIKeysWrapper`` attaches batch keys
   *outside* the transform, so a record built from the ``Compose`` alone would
   omit it.

   The transform is read under both ``.transform`` and ``._transform``, which
   are the same declared fact under two storage names: TorchIO 1.2 privatized
   it (``SubjectsDataset`` keeps ``self._transform`` and publishes only
   ``set_transform()``), while this repo's own datasets still assign a public
   ``self.transform``. The private spelling is accepted only if it is callable —
   ``set_transform``'s own contract — so an unrelated ``_transform`` cannot
   write a confident wrong entry into the artifact.

   .. note::

      **Artifacts written before 2026-08-18 report less than this.** Every
      patch-sampled arm hands the builder a ``tio.Queue``, whose only delegate
      attribute is ``subjects_dataset``; before it was in the list the walk
      stopped dead and such records read
      ``"dataset_chain": ["Queue"]`` with ``⚠ INCOMPLETE: Queue exposes no
      .transform``. That is an honest gap, not a claim that no transforms ran —
      and it is why the ``experiment_11_attention_none`` normalization
      divergence (a ``data.processing`` block declared and then silently
      defaulted away) had nothing in the applied half to contradict it. A blank
      half cannot falsify a declaration.

``model_input_linearization``
   Space-filling-curve reordering the **model** applies to its own input.
   ``HilbertOrder`` and ``ImageTopologyLinearizer`` permute the spatial axes
   into curve order before the sequence backbone sees them, so on a Mamba/SSM
   arm the tensor the network consumes is not the one the snapshot renders. The
   mode is read off the *constructed* module, which covers every curve
   (hilbert / morton / snake / zigzag / raster) without a hardcoded list and
   reports what was built rather than what config asked for.

``incomplete``
   Everything that could not be resolved, named explicitly. A silently partial
   record would recreate the very facade this exists to expose (pitfall #16), so
   the builder never returns a half-record without saying so. ``snapshot.txt``
   renders these as ``⚠ INCOMPLETE`` lines.

Reading a snapshot
------------------

* Check ``provenance.declared`` first. An ``input_prepared`` that looks fully
  sampled is a **defect** if the arm declared acceleration and **correct** if it
  declared none. The picture alone cannot tell you which.
* Check ``extra.prepared_equals_model_input``. If it is ``false``, the tensor
  you want is under the tag named in ``extra.model_input_snapshot_tag``.
* On a cold-diffusion arm, ``mask``'s mean in ``snapshot.json`` is the sampling
  fraction ``1/R`` for the rung that step drew. The ladder is drawn **randomly
  per step**, so a low-``t`` snapshot is legitimately near-fully-sampled — that
  is the forward process working, not a regression.
* Read ``incomplete`` before concluding anything from an absence. A transform
  missing from ``applied.transforms`` means "not recorded" only if
  ``incomplete`` is empty.

Cost
----

The provenance record is built **once per strategy** and cached
(``BaseTrainingStrategy._snapshot_provenance``). Nothing it reads changes after
the environment is built, and non-negotiable 9 forbids paying a dataset walk per
step. Every read is best-effort and type-guarded: a diagnostic must never be the
reason a training run dies, and a ``MagicMock`` test environment must never
stringify into the artifact (#693).

See also
--------

* :doc:`run_provenance_and_logging` — the run-level provenance record.
