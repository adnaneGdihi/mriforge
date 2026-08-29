Config-declarable transform registry
====================================

``data.processing.transforms`` is the YAML seam for "run this transform on every
subject". This page describes what it does now, and what it silently did not do
before 2026-08-04.

The defect
----------

The field was typed ``list[dict[str, Any]]`` — anything validated — and its only
consumer scanned the list for the single literal string ``"graph_encoding"`` and
``break``\ ed:

.. code-block:: python

   for t_config in transforms_config:
       t_name = t_config.get("name")
       if t_name == "graph_encoding":
           enable_graph_encoding = True
           ...
           break

Every other entry was accepted at config load and then discarded without a word.
There was no dotted-path resolver anywhere in the data path, so the four arms
declaring ``name: mriforge.data.transforms.slice_profile.SliceProfileTransform``
never ran that transform; the arm spelling the key ``type: scout_acquisition``
was not even read; and an entry after a ``graph_encoding`` entry was dropped by
the ``break``. Each of those arms is *named* for the mechanism it was not
running, smoke-passed, and reported success — pitfall #16 (inert mechanism)
sitting behind pitfall #15 (an advertised knob nothing reads).

Three chains were dead at link 0
--------------------------------

The consequence was not confined to the transforms. Three of them are the sole
documented producer of a key a live consumer reads, and because the transform
could not be constructed from any config, the consumer silently did nothing:

.. list-table::
   :header-rows: 1

   * - Transform
     - Key it produces
     - Consumer that was silently inert
   * - ``PhaseResidualTransform``
     - ``phase_residual``
     - ``inverse_bloch_phase_strategy`` — the Tikhonov phase-smoothness prior,
       its defining term. 17 arms set ``lambda_phase_smooth``; the weight was
       validated, read and INFO-logged over a term that could never fire.
   * - ``ScoutAcquisitionTransform``
     - ``scout``
     - ``scas_strategy`` — the hypernet was never built, never joined
       ``opt_g``, and neither ``scas_mask`` nor ``scas_logits`` was written, so
       the density penalty also no-opped. The arm was plain LOUPE under the
       SCAS name.
   * - ``ForegroundMaskExtractor``
     - ``foreground_mask``
     - ``core.metrics.context`` — eight no-reference metrics fell back to a
       cruder ``0.05 × max`` threshold instead of the 99th-percentile-of-nonzero
       mask, which the module's own comment explains is the wrong statistic
       because background dominates the histogram.

The registry
------------

:mod:`mriforge.data.transforms.registry` makes registry membership the validator,
exactly as ``MetricsRegistry`` + ``check_metric_names_are_registered`` does for
``metrics.compute``: a name that is not registered **raises**, and every
registered transform is reachable from YAML.

.. code-block:: python

   @register_transform("phase_residual", produces=("phase_residual",))
   class PhaseResidualTransform(tio.Transform): ...

.. code-block:: yaml

   data:
     processing:
       transforms:
         - name: phase_residual
           kwargs: {kernel_size: 9}
         - name: scout_acquisition
           kwargs: {scout_resolution: [32, 32]}

Kwargs may be nested under ``kwargs:`` (preferred) or written flat beside
``name`` — the flat spelling is what the committed ``graph_encoding`` arms use,
so it keeps working. An unknown kwarg is *not* swallowed: it reaches the
transform constructor and surfaces as a ``TypeError`` naming the transform.

``produces`` is the anti-facade payload. It lets an audit answer "is there any
registered producer for the key this strategy reads?" without importing every
strategy — see :func:`~mriforge.data.transforms.registry.transforms_producing`.
``tests/unit/data/transforms/test_registry.py`` pins that invariant for the
three keys above, so a future refactor that unregisters one of them fails a test
instead of quietly reinstating the dead chain.

Where the transforms are applied
--------------------------------

:meth:`~mriforge.data.builders.torchio_transform_builder.TorchIOTransformBuilder._append_registry_transforms`
appends them to **both** the train and the val chain, last but one (before the
image→k-space bridge). Applying a declared transform to train only would
reproduce the normalization split this audit already found elsewhere: the model
would be graded on data the transform never touched.

Three layers of enforcement
---------------------------

#. **Schema** — ``TransformSpecSchema`` requires ``name``, which alone rejects
   the ``type:`` spelling at config load. Registry membership is deliberately
   *not* checked here: ``config/`` may not import ``data/`` (non-negotiable #5).
#. **Builder** — ``TorchIOTransformConfig.from_training_config`` resolves every
   name through the registry and raises ``KeyError`` listing the valid names,
   with an explicit hint when the name looks like a dotted import path.
#. **Audit** — ``check_transform_names_are_registered`` reports the same problem
   at Tier 1 (~100 ms), before anything loads data.

Adding a transform
------------------

#. Put the class in ``src/mriforge/data/transforms/`` (non-negotiable #12 — one
   canonical home).
#. Decorate it with ``@register_transform("<name>", produces=(...))``.
#. Import the module from ``src/mriforge/data/transforms/__init__.py`` — the
   decorator only runs when the module is imported, and a transform nobody
   imports is registered nowhere. This is the same maintenance contract as
   ``data/adapters/__init__.py``.
#. Add a test asserting it is registered and constructs with defaults.

Currently registered: ``foreground_mask``, ``graph_encoding``, ``phase_residual``,
``scout_acquisition``. The remaining caller-free transforms under
``data/transforms/`` are a separate decision — several are duplicates that
*disagree numerically* with a live implementation and should be deleted rather
than registered.
