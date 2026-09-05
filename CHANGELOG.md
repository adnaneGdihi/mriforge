# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-09-04

### Added
- `scripts/release/zenodo_deposit.py` can deposit a **new version** of the published
  Zenodo record (`actions/newversion`) instead of only minting a new record, and
  discards the files Zenodo inherits from the previous version before uploading.
- README: a "Bring your own code" guide covering the three plugin-discovery layers
  (entry points, `SPECTRAMR_PLUGINS`, `plugins.paths`) and the collision rules.
- `.readthedocs.yaml` pre-installs the CPU torch wheel, so a docs build does not
  resolve the 5.1 GB CUDA stack it never uses.
- The **version policy** is written down and mechanised: `docs/versioning.rst` states the
  MAJOR.MINOR.BUILD scheme and the branch-to-version mapping (`main` carries stable
  releases, `nightly` the latest build, `dev` integration), and
  `scripts/release/bump_version.py` is the sole **writer** of the four independent
  version statements that `build_dist.py` compares -- a writer with its own idea of
  where the version lives is the second owner non-negotiable 17 forbids.
- The published repository's settings are a **declared, diffable model**
  (`scripts/release/public_repo_settings.yaml`) with `public_settings_diff.py` to
  compare it against the live repository and `public_settings_apply.py` to converge it,
  rather than console state nobody can review.
- The published repository runs a **two-lane CI**: a blocking `pr-required` lane and an
  advisory lane, shipped through the export overlay, plus a `manual-full-suite` workflow
  for the tiers too long to gate a PR on.
- The release lane -- build, test suite, GitHub release -- runs on the self-hosted
  `thor` runner.

### Changed
- The Zenodo concept-DOI check is now a gate that raises **before** `actions/publish`
  rather than a note printed after it; the weaker copy in `report_badge` is gone.
- README badges: the licence badge moves to the dynamic form now that the repository
  is public, and ~90 lines of commented-out badge archaeology were removed.
- Every GitHub Actions reference is **SHA-pinned**, and zizmor's advisory rules are
  absolute rather than best-effort.
- `gudhi` and `scalene` are marked off `linux/aarch64`, so `pip install -e ".[dev]"`
  resolves on ARM64. The two are excluded for *different* reasons: gudhi publishes
  neither an aarch64 wheel nor an sdist, so there is nothing to install; scalene ships
  an sdist but no linux-aarch64 wheel, so it would compile four native objects as a
  side effect of installing the test suite. macOS arm64 keeps its scalene wheel.
- `CubicalPHWassersteinLoss`'s `ImportError` now names **which** package is missing
  rather than the union of both, and on `linux/aarch64` says that the `[topology]`
  extra cannot supply gudhi there -- pointing at an extra that cannot help sends the
  reader round a loop that ends back at the same error.

### Removed
- README: the "What pip actually resolves" section. The load-bearing part -- cu126 is
  the last wheel lane shipping `sm_70`, so a V100 needs it -- is now under
  Installation as "Pinning the CUDA build".

### Fixed
- The public export ships `tests/unit/release/conftest.py`. A dropped `conftest.py` is a
  different failure shape from a dropped test subject: pytest resolves the bare name
  against the **root** `conftest.py`, fails to find the fixture, and aborts collection
  for the whole directory -- so the published tree lost far more than the one file, and
  only in the export. `test_export_public_tree.py` now gates that shape.

## [0.1.0] - 2026-09-04

<!-- Cut from `[Unreleased]` in preparation for the `v0.1.0` tag, which is the
     action that actually publishes. The heading is not cosmetic: it is one of the
     four version declarations `scripts/release/build_dist.py` reconciles, and
     `changelog_version()` deliberately SKIPS `[Unreleased]`, so while this section
     carried that heading the reconciler reported `CHANGELOG.md: unreadable` and
     `release.yml` would have failed at the build step -- before reaching PyPI --
     for every tag pushed. Verified by calling the reconciler directly rather than
     by reading it.

     The date must equal the day the tag is pushed. If that slips, change it here
     first; the tag is what makes the claim public, and until it is pushed this
     heading claims nothing that a reader can see. The earlier version of this file
     dated a released 0.1.0 at 2026-08-01 while no tag, no GitHub release and no
     PyPI distribution existed, which is the failure this note exists to prevent. -->

### Added
- Initial public release of spectraMR.
- Multi-paradigm training framework: 206 `training_mode` keys resolving to 153
  training-strategy classes.
- 586 registered models, reachable from a cold import after
  `populate_model_registry()`.
- 217 registered losses and 213 registered metrics, across image, k-space,
  complex, physics, adversarial, latent, distillation and virtual-fiducial
  domains.

<!-- Every count above is a REGISTRATION count, produced by calling
     populate_model_registry() and reading len(MODEL_REGISTRY) -- not a
     decorator-site grep. The two
     are different questions and nothing in CI compares them: a grep counts every
     textual occurrence, including the ones in comments, docstrings and tests, while
     the registry is what a config can actually reach. They also drift, in both
     directions -- the metric count above was 211 here and measured 213, and the
     model count of 590 above is an earlier reading than the rest. Re-run the snippet
     before changing a number here; do not carry one forward, and do not let a
     verified number vouch for its neighbour. -->
- Three-tier audit ladder (Tier 0 schema, Tier 1 static cross-validation,
  Tier 2 synthetic forward probe).
- MRI physics single source of truth under `src/spectramr/infrastructure/physics/`.
- Sphinx documentation and a Read the Docs build configuration
  (`.readthedocs.yaml`). The hosted site is not live as of v0.1.0.
- Apache-2.0 licence + clinical-use disclaimer.
- PyPI Trusted Publishing pipeline (`release.yml`) producing wheel + sdist on tag
  push. `spectramr` 0.1.0 is published, carrying both artefacts
  (`spectramr-0.1.0-py3-none-any.whl` and `spectramr-0.1.0.tar.gz`).
- GitHub Actions CI on pull requests: a blocking `pr-required` lane (changed-line
  lint, repository guards, architecture fitness functions, unit-test collection,
  physics tests, dependency and secret scanning) aggregated behind a single
  `required` check; an advisory `pr-advisory` lane (full lint, mypy, pre-commit,
  docs build, dependency review, zizmor); and CodeQL. The blocking lane runs the
  physics and architecture suites and **collects** the unit suite — it does not
  execute the full unit suite, which is too long for per-PR CI.
- Issue templates (bug / feature / paradigm-proposal) and PR template with DCO
  sign-off checklist.

<!-- Both links resolve as of 2026-09-04: `v0.1.0` is pushed and the GitHub Release
     is published. Verified by status, and the probe was checked against a tag that
     does not exist first -- `releases/tag/v9.9.9` and `compare/v9.9.9...HEAD` both
     return 404, so the 200s here mean something. The `[0.1.0]` heading date above
     and `date-released` in CITATION.cff are two further declarations of the same
     day, and nothing reconciles the three; change them together. -->
[Unreleased]: https://github.com/adnaneGdihi/spectramr/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.1
[0.1.0]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.0
