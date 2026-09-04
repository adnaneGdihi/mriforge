=================
Known limitations
=================

This page lists behaviour that is real, reproducible, and *not* what a reader
would reasonably assume from the surrounding documentation. It exists because a
documented limitation is honest and a documented feature that is inert is not.

Everything here was measured on the shipped package, not inferred from a plan or
an audit note. Where a number appears, the command that produced it appears with
it, because these counts drift.

.. note::

   This page is **hand-maintained**, not generated. Nothing recomputes it, so a
   fixed limitation will linger here until someone deletes the entry. Each item
   names a tracking issue; the issue, not this page, is the source of truth for
   whether it is still open.

Environment
===========

Most documented environment variables are read by nothing in this package
--------------------------------------------------------------------------

:doc:`environment_variables` documents **71** variables in the ``SPECTRAMR_*`` /
``SIM2RANK_*`` families. **48 of those names appear nowhere in the shipped
package at all** -- 36 ``SIM2RANK_*`` belonging to the meta-evaluation batch
pipeline, and 12 ``SPECTRAMR_*`` test- and CI-harness knobs. Nothing here can read
them, so setting one has no effect.

This is absence rather than inertness, and the distinction matters: the 48 are
not knobs this package reads and ignores, they are knobs whose *reader* is a
script that is not distributed. The remaining 23 names do appear in the package;
that a name appears is not by itself proof it is read on any particular path, so
treat the 23 as "possible" rather than "verified wired".

The same is true one level up. The page's "Script-scoped variables" table has 110
rows citing 28 distinct scripts, and 27 of those 28 are absent here. The rows are
kept deliberately -- they record how the published results were produced -- but
they are a description of the maintainers' cluster workflow, not an interface.

To re-measure, from the root of a checkout::

    grep -rhoE '\b(SPECTRAMR|SIM2RANK)_[A-Z0-9_]{2,}' docs | sort -u > /tmp/documented
    grep -rhoE '\b(SPECTRAMR|SIM2RANK)_[A-Z0-9_]{2,}' src  | sort -u > /tmp/present
    comm -23 /tmp/documented /tmp/present | wc -l      # -> 48

Documentation
=============

Commands in the documentation are checked against the tree, prose references are not
-------------------------------------------------------------------------------------

``scripts/ci/check_docs_paths_exist.py`` fails on any *pasteable* command in
``docs/`` that names a repository path the tree does not contain, and the
published tree passes it with zero findings. Prose references are reported by the
same gate but do not fail it, and **342 of them remain**: sentences that cite a
file for provenance -- "this knob is read at ``scripts/sim2rank/style.py``" --
where the file is maintainer-side and not published.

Those citations are accurate about where the code lives; they are simply not
paths you can open in your own checkout. They were left rather than stripped
because removing them would lose the provenance without gaining anything a reader
can act on. To list them::

    python scripts/ci/check_docs_paths_exist.py

Configuration
=============

Unknown keys inside an ``extra="ignore"`` block are dropped in silence
----------------------------------------------------------------------

The configuration schema is mostly strict: of the 339 schema classes reachable
from a config, **252 are** ``extra="forbid"`` and reject an unknown
key outright. But **66 are** ``extra="ignore"``, and inside one of those a key
the schema does not declare is discarded with no error and no warning. The YAML
still shows it; the run never sees it; the arm trains on the default.

This is the failure mode to know about when a knob you set appears to do
nothing. It is not detectable by reading the config back, because the key is
gone by then.

The private tree carries a census script that reports every discarded key across
a directory of configs, but it reads the internal experiment-corpus layout and is
**not published with this release**. This page therefore names the failure mode
rather than handing you a command that is not in your checkout. To check one
config by hand, compare the keys your YAML sets in a block against the fields the
schema for that block declares: a key in the first and not the second is being
dropped.

To re-measure the class counts above:

.. code-block:: python

   import typing
   from pydantic import BaseModel
   from spectramr.config.schemas import training as training_schemas
   from spectramr.config.settings import TrainingSettings

   def models_in(annotation):
       """Unwrap typing constructs to a fixed point.

       A fixed depth is not enough. A discriminated union is annotated
       ``Optional[Annotated[A | B | ..., FieldInfo(discriminator=...)]]``, so its
       members first appear three ``get_args`` levels down -- and those are the
       polymorphic blocks most likely to carry loosely-validated keys.
       """
       found, stack = set(), [annotation]
       while stack:
           a = stack.pop()
           if isinstance(a, type) and issubclass(a, BaseModel):
               found.add(a)
           stack.extend(typing.get_args(a))
       return found

   seen, counts = set(), {}
   def walk(m):
       if m in seen:
           return
       seen.add(m)
       policy = m.model_config.get("extra") or "ignore"
       counts[policy] = counts.get(policy, 0) + 1
       for f in m.model_fields.values():
           for sub in models_in(f.annotation):
               walk(sub)

   # TWO roots. Field-reachability from TrainingSettings finds everything held
   # AS a field; it cannot reach a top-level paradigm schema, which a YAML
   # selects by `training_mode`. Seeding only the first root reports 8 open
   # classes where 20 are reachable.
   walk(TrainingSettings)
   for name in sorted(set(training_schemas._MODE_DISPATCH.values())):
       walk(getattr(training_schemas, name))

   print(sum(counts.values()), counts)
   print(sorted(m.__name__ for m in seen if m.model_config.get("extra") == "allow"))

Twenty-one schema classes accept *any* key
------------------------------------------

``extra="allow"`` is the opposite hazard: the key is not dropped, it is kept —
as an untyped extra field that nothing validates and nothing reads. A typo
becomes a carried value rather than an error.

One of them is open by design: ``TrainingSettings.metadata`` accepts any key
because the corpus carries roughly 60 free-form ones (``note``, ``group``,
``marker_source``, ...) that are prose for a human reader, and rejecting them
would fail the config-load gate on most arms for no gain. Its ``status`` field
is closed, which is the half that matters: a free-text status is how ~190 arms
came to admit an unimplemented mechanism in words nothing read.

Twenty-one classes reachable from a config are ``extra="allow"``:

* ``AdapterStepSchema``
* ``ColdParams`` [#du]_
* ``ExperimentMetadataSchema``
* ``KSDAuditConfigSchema``
* ``LatentDiffusionParams`` [#du]_
* ``LatentTrainingConfigSchema``
* ``TrainingConfigAcqHypernetwork`` [#md]_
* ``TrainingConfigBlochManifoldDPS`` [#md]_
* ``TrainingConfigCorruptionCalibration`` [#md]_
* ``TrainingConfigDataEfficiencyHarness`` [#md]_
* ``TrainingConfigDispersionBlochAE`` [#md]_
* ``TrainingConfigEquivarianceConformal`` [#md]_
* ``TrainingConfigIBVF`` [#md]_
* ``TrainingConfigPairedSynthesis`` [#md]_
* ``TrainingConfigPhysResidualConformal`` [#md]_
* ``TrainingConfigSE3Navigator`` [#md]_
* ``TrainingConfigTissueDiffusionPretrain`` [#md]_
* ``TrainingConfigTwinDPS`` [#md]_
* ``TrainingStrategyConfigSchema``
* ``TransformSpecSchema``
* ``UnspecifiedParams`` [#du]_

.. [#du] Reached only through the discriminated union at
   ``TrainingStrategyConfigSchema.diffusion``. A schema walk that unwraps a fixed
   number of ``get_args`` levels does not see these, which is why an earlier
   version of this page listed five.

.. [#md] A *top-level* schema, selected by ``training_mode`` rather than held as
   a field of anything. A walk seeded only from ``TrainingSettings`` never
   arrives at one, which is why a later version of this page listed eight. The
   dispatch table is ``_MODE_DISPATCH`` in
   ``spectramr/config/schemas/training/__init__.py``; a mode with no entry there
   loads through the default schema, which is already counted above.

These are deliberately open — they carry strategy- and transform-specific
payloads whose keys vary by component — so this is a documented trade-off rather
than a defect. Spell keys inside them carefully; nothing will tell you if you do
not. The three diffusion parameter blocks matter most in practice: an
``extra="allow"`` class is where a misspelled cold-diffusion or latent-diffusion
knob is carried silently instead of rejected.

``config/presets/`` cannot load its own presets
-----------------------------------------------

The ``spectramr.config.presets`` subsystem is importable and documented, and
resolves nothing. ``discover_and_register_presets()`` returns ``0`` and
``get_baseline_presets()`` returns ``[]``, because discovery defaults to an
experiments directory and globs ``experiment_*.yaml`` while the three shipped
presets are named ``baseline_*.yaml``. The example in
``spectramr/config/presets/cli.py``'s own docstring raises ``KeyError``.

Do not build on this module. Load configuration from a YAML file instead, which
is the path everything else uses.

Tracked as issue #1563.

Continuous integration does not run
-----------------------------------

``.github/workflows/`` ships and **does** execute here. This section previously said
the opposite -- "none of it executes, GitHub Actions is disabled for this project" --
which was true when this repository was created and is no longer: Actions is enabled,
and ``pr-required.yml`` (8 jobs, 13 guard scripts, the architecture fitness functions
and the YAML-audit tier) fires on real pull requests.

Read the correction rather than only the conclusion, because the retired sentence was
load-bearing in one dangerous place: it described ``release.yml`` as "tag-triggered and
equally inert, so a first publish needs ``python -m build && twine upload`` by hand".
``release.yml`` triggers on ``push:`` of a ``v*`` tag and publishes to PyPI through
Trusted Publishing. With Actions enabled, **pushing a version tag publishes** -- it is
not a local, reversible act, and PyPI never re-issues a filename it has already seen.
Push the commit first, confirm the lane is green, and tag only when you intend to
release.

Note which repository each statement is about. Actions is disabled on the private
research repository, which is why ``make gate`` exists there: it *derives* the local
lane by parsing ``pr-required.yml`` rather than restating it. That is a property of
that tree, not of this one, and a claim carried across the two is how this section
came to be wrong. ``tests/unit/ci/test_workflow_triggers.py`` takes the workflows as
its subject in both.

This is stated here rather than left to be inferred from a badge. Run the lane
yourself:

.. code-block:: console

   $ make gate                                    # the whole lane
   $ make gate GATE_ARGS="--list"                 # what it derived, without running
   $ make gate GATE_ARGS="--jobs guards --quiet"  # one job

It reports three states and never folds the third into the first: ``PASS``,
``FAIL`` (exit 1) and ``UNRUNNABLE`` (exit 2, a required tool is absent). An
unrunnable step is not a pass -- which is the same distinction the next entry on
this page makes about a green ``audit``.

Auditing
========

A green ``audit`` does not mean every check ran
-----------------------------------------------

``spectramr audit`` runs about 150 config checks. On any given arm a number of them
do not run -- 17 of 144 on the arm sampled below. A check that declined to run is
reported with ``passed: true`` at ``severity: info``, does not affect the exit
code, and **is printed with the same green tick as a check that passed**:

.. code-block:: text

   ✅ [health:latent_decode_resolution] Not a latent model; skipping.
   ✅ [health:train_val_split_leakage] split-leakage check skipped (explicit_manifest):
      train and/or validation manifest not present locally.

Most of those are correct -- a latent-decode check has nothing to say about a
non-latent arm, and reporting it as a failure would be noise. Two groups are
structural rather than arm-shaped, and are worth knowing before you read a green
audit as coverage.

**Split-leakage is not checked in a clone.** ``train_val_split_leakage`` compares
the train and validation manifests. Those manifests are generated, not committed
-- ``data/manifests`` is in ``.gitignore`` -- so on an arm that declares
``index_path`` / ``validation_index_path`` the check has nothing to open and
skips. It was skipped on all five arms sampled across five cohorts: four for the
missing manifest, and one whose arm resolves its splits by directory crawl
instead, which the check reports separately as folder-disjoint. Generate the
manifests before you rely on this check.

**The five ``workflow_*`` checks need a ``workflow:`` block.** When an arm does
not declare one they skip, one for one -- confirmed against a sample where the
single arm carrying a ``workflow:`` block reported its regime and the other four
skipped. This is deliberate: an absent declaration is advisory and a wrong one is
a hard error, so the checks decline rather than guess. It does mean an arm with
no ``workflow:`` block is not being checked against the regime contract.

To see which checks declined on your own arm:

.. code-block:: console

   $ spectramr audit <arm>.yaml | grep -i skip

A tick beside the word *skipping* is a check that did not run. That is not the
same as a check that passed.

Optional dependencies
=====================

Mamba/SSM models require a separate, compiled install
-----------------------------------------------------

The ``.[mamba]`` extra is deliberately excluded from ``.[all]`` and ``.[dev]``:
it compiles a CUDA selective-scan kernel and needs ``nvcc`` plus a torch that is
already installed. Install it as a second step:

.. code-block:: console

   $ pip install -e '.[mamba]' --no-build-isolation

Without it, Mamba models **fail loudly** rather than degrading. The Gated-Conv +
GRU fallback exists only for CPU/CI wiring tests and is opt-in through
``SPECTRAMR_ALLOW_MAMBA_FALLBACK``; it is not an approximation of the Mamba path
and must not be used for results.

Some optional features have no ``pip install`` route
----------------------------------------------------

spectraMR declares seventeen optional-dependency extras, and the pattern is
consistent: a feature that needs a package outside the core install gets an
extra, and its runtime guard raises a clean ``ImportError`` naming that package.
``pip install "spectramr[iqa]"`` enables the closed-form perceptual metrics,
``[topology]`` enables the persistent-homology losses, and so on.

**Twenty optional packages are guarded in the source but appear in no extra.**
The guard half works — you get an accurate error naming the package — but there
is no ``spectramr[...]`` group that installs it, ``[all]`` included. You have to
install it yourself.

Four registered component names are affected. Each is present in its registry, so
a configuration may name it, and each raises at construction until you install
the package by hand:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25

   * - Kind
     - Registered name
     - Install first
   * - model
     - ``medical_dino_encoder``
     - ``pip install timm``
   * - loss
     - ``dino_perceptual``
     - ``pip install timm``
   * - metric
     - ``frd``
     - ``pip install pyradiomics SimpleITK``
   * - metric
     - ``rfs``
     - ``pip install pyradiomics SimpleITK``

Three more features degrade rather than fail, and say so in the log:

* ``latent_gan_generator`` raises only when ``text_conditioning_enabled`` is set
  (needs ``transformers``).
* ``symbolic_regression_wrapper`` trains normally but skips equation discovery
  without ``pysr``.
* ``neural_ode`` integrates with its built-in fixed-step RK4 without
  ``torchdiffeq``. Both solvers walk the same grid, so this changes the
  implementation and not the discretisation — but torchdiffeq's adjoint and
  adaptive solvers are unreachable.

The remaining undeclared packages guard non-registry code paths: ``beartype``,
``pydicom``, ``safetensors``, ``sigpy``, ``GPUtil``, ``ants``, ``fastrad``,
``flash_attn``, ``keras``, ``nvtx``, ``pypapi``, ``torch_fidelity``,
``torch_geometric`` and ``xformers``.

This is tracked upstream. Declaring the extras is the intended repair; it needs a
resolvable version bound checked against a package index for each one, and a
decision per package, because two of them (``flash_attn``, ``xformers``) compile
CUDA kernels and could not sit inside ``[all]`` any more than ``[mamba]`` can.


The published tree
==================

What the shipped package does not carry, and what that costs a reader who
expects it. Merged from the release-sanitization branch, which maintained this
page independently.

The experiment corpus is not published
--------------------------------------


The private research tree carries 647 experiment configurations under
``experiments/inprogress/``. Those are not part of this release: they encode one
site's data layout, cluster allocation and in-flight research arms, and several
reference datasets whose licences do not permit redistribution of derived data.

What ships instead is a single file: ``experiments/templates/comprehensive_config_template.yaml``. **No exemplar arm ships at all** -- the
template is the only worked configuration in the published tree, and it is a
template rather than an arm that was run. Writing a new configuration against it
is possible; reading a known-good arm that produced a published result is not.

The two remaining files under ``experiments/templates/`` are withheld on purpose
rather than overlooked: a template is *copied*, so one that does not parse is
worse than an absent one -- the reader's first act inherits the defect. Both were
measured with ``spectramr audit``, not assumed.

**Consequence:** a handful of test gates in the private tree exist to check that
corpus, and cannot be meaningful without it. They are removed here rather than
shipped, per the rule that a gate which can no longer see its subject is deleted,
not left in to report success. The removed gates are:

.. code-block:: text

    tests/unit/config/test_dead_legacy_key_spellings.py
    tests/unit/utils/test_config_load_baseline.py
    tests/smoke/test_config_validation.py
    tests/smoke/test_deep_config_integrity.py
    tests/smoke/test_vf_smoke.py
    tests/audit/test_experiment_yaml_syntax.py

Leaving them in was measured, not assumed. Against an empty corpus they produce
6 failures, 16 passes and 4 skips, and the empty input reaches them in three
different shapes -- only one of which is silent:

* an explicit anti-vacuity guard fires, e.g. *"cohort directory is empty -- the
  guard would be vacuous"*. Loud, and the correct design.
* ``@pytest.mark.parametrize`` over an empty list, which pytest reports as
  ``got empty parameter set``. Skipped, and visible.
* a ``for config_path in ALL_CONFIGS:`` loop **inside the test body**. The loop
  never executes, no assertion runs, and the test passes. One file reports seven
  green tests having validated nothing.

The third shape is the reason these are removed rather than left to sort
themselves out: it is indistinguishable from a real pass in any CI summary.

Cluster and scheduler integration is unconfigured
-------------------------------------------------


The SLURM submission helpers generate batch scripts but ship with **no default
account, partition or mail address**. This is deliberate. An earlier revision
carried one site's allocation name as the default; a placeholder such as
``--account=your-account`` would be worse than the leak, because SLURM rejects an
unknown account outright, whereas an omitted ``#SBATCH --account`` directive lets
the scheduler apply the submitter's own default.

Supply your own values through ``ResourceSpec`` or the environment; see
``.env.example`` for the full set of variables and which component reads each one.

Some paradigm strategies read batch keys that no dataset produces
-----------------------------------------------------------------


The framework carries strategies for a wide range of paradigms, and several of the
more specialised ones read quantities from the training batch that no shipped data
pipeline emits -- ``T1_map``, ``brain_mask``, ``gradient_waveform``,
``resonance_params``, ``deformation_field`` and others.

Measured with an internal batch-key census, not published with this release: **59**
such keys, of which 55 have a read site in ``src/``. What happens when the key is
absent is not uniform, and the difference is the whole point:

* **23** raise. The absence is loud, the run stops, and the message names the key.
  This is the correct shape and needs no warning here.
* **32** are guarded, and the term they feed simply does not contribute. Training
  completes and reports success with one component of the objective silently
  inactive.

The second group is the reason this section exists. Such a strategy is not broken,
but it is **not exercised by any configuration that ships**: it needs a dataset that
emits those keys, which is a data-layer integration this release does not include.
Treat those paradigms as reference implementations to build against, not as paths
that run end to end on the shipped example arms.

At least one case is worse than an inactive term, and is filed rather than fixed:
the QSM pipeline substitutes an all-ones brain mask when none is supplied and runs
three physics operators on it, which produces a susceptibility map that is undefined
rather than approximate.

**The counts above are a static AST census and should be read as such.** The tool
cannot distinguish a training batch from any other local named ``batch``, which is
why it is a report and not a CI gate; and its own worst category -- a mechanism
running on a fabricated value -- reads ``0`` while at least one instance exists,
because it inspects only the default argument of ``.get()`` and not a fabrication in
the following statement. Both the QSM case and the census blind spot have tracking
issues.

Issue references in history and docstrings do not resolve
---------------------------------------------------------


This repository begins at a single initial commit. Rationale comments throughout
the source cite issue and pull-request numbers from the private tree in which the
work was done -- roughly 386 distinct numbers across 585 files. Those numbers do
not correspond to issues here, and GitHub will autolink ``#N`` to whatever issue
or PR happens to hold that number in this repository.

They are kept rather than stripped because the surrounding sentence is usually the
only record of *why* a non-obvious piece of code is shaped the way it is, and a
rationale with a dangling reference is still a rationale. Read them as provenance,
not as links.

Running the test suite requires a git checkout
-----------------------------------------------


The suite enumerates the experiment corpus from ``git ls-files`` rather than a
filesystem glob, through the single owner ``tests/utils/corpus.py``. That choice
is deliberate and is documented in the module itself: a glob has a different
subject on every machine, and one cluster job once spent 24 of its failures on
two arms that exist in no git history at all.

The consequence is that **a tree with no** ``.git`` **cannot enumerate the
corpus**. That is not an error -- an sdist, a release tarball and a downloaded
ZIP are all legitimate ways to obtain this code, and none of them carry a
``.git``. In such a tree the corpus-dependent tests **skip visibly**, naming the
reason, and the rest of the suite runs normally:

.. code-block:: text

   SKIPPED [2] tests/utils/corpus.py:140: tests.utils.corpus enumerates
   git-tracked files, and <dir> sits in a tree with no .git -- an sdist,
   tarball or ZIP rather than a checkout.

To run those tests, obtain the code with ``git clone`` rather than as an archive.

A tree that *does* have a ``.git`` but whose git is unusable -- git missing from
``PATH``, a corrupt repository, a ``safe.directory`` refusal -- still raises,
loudly, and deliberately. Those two absences look identical from the outside and
call for opposite responses: the first has nothing to fix, while the second would
silently disable a large group of tests in a working tree if it were softened
into a skip.

Note that the published tree does not carry the experiment corpus at all (see
the first entry on this page), so most of these tests have no subject here even
in a clone.
