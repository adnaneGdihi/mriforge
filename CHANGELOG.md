# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-31

### Added
- Initial public release of MRIForge.
- Multi-paradigm training framework: 153 training strategies, selectable through
  206 `training_mode` spellings.
- 586 registered model architectures and 217 registered losses, spanning image,
  k-space, complex, physics, adversarial, latent, distillation and
  virtual-fiducial domains. These are **registration** counts measured on this
  package at the v0.1.0 freeze — not decorator-site greps, which count 599 and
  244 respectively and include sites no config can reach. See `README.md` for
  what a registration count does and does not claim.
- Three-tier audit ladder (Tier 0 schema, Tier 1 static cross-validation,
  Tier 2 synthetic forward probe).
- MRI physics single source of truth under `src/mriforge/infrastructure/physics/`.
- Sphinx documentation and a Read the Docs build configuration
  (`.readthedocs.yaml`). The hosted site is not live as of v0.1.0.
- Apache-2.0 licence + clinical-use disclaimer.
- PyPI Trusted Publishing pipeline (`release.yml`) producing wheel + sdist on tag
  push. Nothing has been published to PyPI as of v0.1.0.
- GitHub Actions CI on pull requests: a blocking `pr-required` lane (changed-line
  lint, repository guards, architecture fitness functions, unit-test collection,
  physics tests, dependency and secret scanning) aggregated behind a single
  `required` check; an advisory `pr-advisory` lane (full lint, mypy, pre-commit,
  docs build, dependency review, zizmor); and CodeQL. The blocking lane runs the
  physics and architecture suites and **collects** the unit suite — it does not
  execute the full unit suite, which is too long for per-PR CI.
- Issue templates (bug / feature / paradigm-proposal) and PR template with DCO
  sign-off checklist.

[Unreleased]: https://github.com/adnaneGdihi/mriforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/adnaneGdihi/mriforge/releases/tag/v0.1.0
