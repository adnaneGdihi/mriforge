# Governance

MRIForge is currently maintained by a single maintainer. Decisions are made by
the maintainer; substantive proposals are accepted as GitHub Issues and
resolved by lazy consensus (no objection within 14 days = accepted). This
document will be revised when additional maintainers join.

## Decision categories

- **Patch-level changes** (bug fixes, docs, dependency bumps) — merged by the
  maintainer at their discretion once CI passes.
- **Minor-level changes** (new paradigms, new public API) — opened as an issue
  using the `paradigm_proposal` or `feature_request` template; merged after the
  14-day lazy-consensus window.
- **Major-level changes** (breaking API, layout migrations) — require an
  explicit `accept` label from the maintainer; the 14-day window does not
  apply because lazy consensus alone is not sufficient for breaking changes.

## Adding maintainers

A contributor becomes eligible for maintainer status after sustained
contribution (≥10 merged PRs in 12 months) and demonstrating familiarity with
the registry-dispatcher pattern and physics SSOT discipline. Maintainership is
offered, not requested.
