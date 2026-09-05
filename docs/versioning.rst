Versioning and release branches
===============================

Every ``spectraMR`` release is numbered ``MAJOR.MINOR.BUILD`` -- three integers,
nothing else. The scheme is `PEP 440 <https://peps.python.org/pep-0440/>`_
compatible, which is what allows it onto PyPI at all, and it agrees with
`Semantic Versioning <https://semver.org/>`_ about what the three positions mean.

What the three positions mean
-----------------------------

``MAJOR``
   An incompatible change to something published as stable: the
   :doc:`scripting_api` surface, the meaning of a config key, or the definition
   of a metric. Numbers produced either side of a major bump are not required to
   be comparable, and what moved is recorded rather than left to be discovered.

``MINOR``
   New capability, added compatibly -- a paradigm, a model, a loss, a metric, or
   a config key whose default preserves existing behaviour.

``BUILD``
   Everything else: a fix, a documentation correction, a packaging change. This
   is the position that advances for a rebuild of the same feature set, and it
   is the one the ``nightly`` branch moves between releases.

The three branches
------------------

The published repository carries three long-lived branches. The version a branch
reports is what tells you which one you are looking at.

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Branch
     - Version it carries
     - What it is
   * - ``main``
     - ``X.Y.B``
     - **Stable builds only.** The tag ``vX.Y.B`` is cut from here, and that tag
       is what publishes to PyPI. A commit on ``main`` is a state that was
       released or is about to be.
   * - ``nightly``
     - ``X.Y.B.devN``
     - **The latest build.** Replaced wholesale by each new export snapshot, so
       its history is not stable -- do not base work on it, and do not open pull
       requests against it. It exists to be installed and tried.
   * - ``dev``
     - ``X.Y.B.devN``
     - **Integration**, cut from ``main``. This is the branch pull requests
       target. It is where a change waits for the release that carries it.

``X.Y.B`` on ``nightly`` and ``dev`` is the version being worked *towards*, not
the last one released: while ``main`` sits at ``0.1.0``, those two carry
``0.1.1.devN``.

Why ``.devN`` and not ``+build``
--------------------------------

A pre-release must sort **before** the release it precedes, and it must be
publishable. PEP 440 gives ``0.1.1.dev4 < 0.1.1``, so an installer resolving
``spectramr`` never prefers a nightly over the stable release of the same
number, and ``pip install --pre`` is the explicit opt-in.

The obvious-looking alternative, a local version such as ``0.1.1+build4``, is
not usable here: PyPI **rejects local versions outright**, so a number that
reads naturally would make the artefact unpublishable -- and it would fail at
upload time, after the tag exists. The constraint is the index's, not a
preference.

Where the number lives
----------------------

``src/spectramr/__init__.py`` holds ``__version__``, and ``[tool.hatch.version]``
reads it -- so the package metadata and the wheel filename are derived, never
restated. Three other files state the version independently
(``CHANGELOG.md``, ``CITATION.cff``, and the git tag), because each is read by
something that cannot import the package.

``scripts/release/build_dist.py`` is the **sole comparator** of that set
(non-negotiable 17): it reads all of them, and fails the build on any
disagreement, with ``--expect-version`` adding the tag as a fifth. Nothing else
compares them, and nothing else should.

Because those files are written by hand and compared only at release time,
moving the number is done with one command rather than three edits:

.. code-block:: bash

   python scripts/release/bump_version.py show        # what every file says now
   python scripts/release/bump_version.py minor       # dry run: print the plan
   python scripts/release/bump_version.py minor --apply

   python scripts/release/bump_version.py nightly --apply   # 0.1.0 -> 0.1.1.dev1
   python scripts/release/bump_version.py release --apply   # 0.1.1.dev4 -> 0.1.1

``release`` is the only mode that touches ``CHANGELOG.md``: it cuts the
accumulated ``[Unreleased]`` section into a dated heading for the version being
released and opens a fresh empty one. A ``nightly`` bump deliberately leaves the
changelog alone -- a dev build is not a release, and giving it a changelog
heading would make the file's release history unreadable.

Publishing order
----------------

The order below is forced by what each step can see, not by preference:

#. **Public tree** -- the export snapshot reaches the published repository.
#. **Zenodo** -- the deposit is dispatched. Zenodo cannot read a private
   repository, and the deposition is irreversible once published.
#. **Tag** -- ``vX.Y.B`` is pushed last, because it triggers the PyPI upload and
   the release notes have to quote a DOI that already exists.

Tagging as part of the publish push inverts that order and cannot be undone:
PyPI never re-issues a filename it has seen, so a wrong wheel costs a version
number rather than an amendment.
