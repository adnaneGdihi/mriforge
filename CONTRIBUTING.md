# Contributing to spectraMR

Thank you for considering a contribution. spectraMR is research software, so
correctness matters more than speed. The rules below exist so the framework
stays auditable and reproducible across releases.

## Development environment

```bash
git clone https://github.com/adnaneGdihi/spectramr.git
cd spectramr
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The `[dev]` extra pulls in `[mri]`, `[diffusion]`, `[viz]`, `[docs]`, `[test]`
and the lint / type / pre-commit tooling.

## Branching and pull requests

- Feature branches off `main`. Never push directly to `main`.
- One PR per logical change. Squash-merge on the GitHub UI.
- PR titles must use [Conventional Commits](https://www.conventionalcommits.org)
  prefixes:
  - `feat:` new feature (minor version bump)
  - `fix:` bug fix (patch version bump)
  - `docs:` documentation only
  - `test:` test-only changes
  - `refactor:` no behaviour change
  - `chore:` build / tooling / dependency updates
  - `ci:` CI configuration changes
  - `perf:` performance improvements
  - `build:` build system changes

  Examples:
  - `feat(losses): add Wasserstein-GP loss with @register_loss alias`
  - `fix(physics): correct fft2c normalisation for AMP path`
  - `docs(api): regenerate autoapi after registry rename`

## Before you open a PR: `make gate`

**Actions runs the lanes below on your pull request here. On the private research
repository it is disabled, so the same lanes
execute.** Run the blocking lane yourself:

```bash
make gate                                        # the whole lane
make gate GATE_ARGS="--list"                     # what it derived, without running it
make gate GATE_ARGS="--jobs guards --quiet"      # one job
```

`scripts/ci/run_required_locally.py` **parses** `.github/workflows/pr-required.yml`
rather than restating it, so a job added there runs here with no edit. It reports
three states, and never folds the third into the first:

| State | Meaning | Exit |
|---|---|---|
| `PASS` | the step ran and succeeded | — |
| `FAIL` | the step ran and failed | 1 |
| `UNRUNNABLE` | the step could not run; a tool is absent | 2 |

`UNRUNNABLE` is deliberately not a pass. Install the missing tool, or pass
`--allow-unrunnable` to state that you accept the gap — silence would make an
unrun check indistinguishable from a passing one, which is exactly the failure
`docs/known_limitations.rst` records for the audit ladder's green ticks.

## CI policy (advisory vs. blocking)

> **This table describes the lane's *design*, not its current behaviour.**
> On the private research repository Actions is disabled and nothing below runs on
> a PR; here it does. Read it as the
> specification `make gate` executes locally.

The codebase is large and pre-publish hardening is in progress. To keep CI
informative without blocking merges, most quality checks currently run in
**advisory mode** — failures are visible on the PR but don't fail the run.

| Check | Mode | What blocks merge |
|---|---|---|
| `lint-diff` (ruff on changed Python files) | **Blocking** | New lint violations in added/modified files |
| `physics-tests` (`pytest tests/physics/`) | **Blocking** | Physics-correctness regression |
| `security` (pip-audit, gitleaks) | **Blocking** | New CVEs, leaked secrets |
| `lint` (ruff + mypy, full repo) | Advisory | — |
| `pre-commit` (all hooks, all files) | Advisory | — |
| `test` (`pytest` non-physics) | Advisory | — |
| `docs` (Sphinx build) | Advisory (sphinx-autoapi cost on 4500 files) | — |
| `codeql` (Python static analysis) | Advisory (private-repo limitation) | — |
| `dependency-review` | Advisory (private-repo limitation) | — |
| `zizmor` (Actions workflow static analysis) | Advisory (unpinned-uses backlog) | — |

Advisory checks still run on every PR; the maintainer watches them shrink.
The plan for promoting advisory checks back to blocking lives with the gates
themselves, in `scripts/ci/` and their baselines: a check graduates when its
baseline reaches zero and a planted violation has been watched to turn it red.

## Test policy

Every PR runs `pytest -m "not gpu and not e2e and not integration"` in CI.
Coverage must not regress more than 1 percentage point. GPU tests are run
locally:

```bash
pytest -m gpu                              # requires CUDA
pytest tests/physics/ -v                   # MRI physics suite
pytest -m "integration"                    # opt-in, multi-component flows
```

If your change touches anything under `src/spectramr/infrastructure/physics/`,
you must also run `pytest tests/physics/` locally.

If your change adds a YAML key, declare it on the owning Pydantic schema in the
same change — a class with `extra="ignore"` drops an undeclared key without a word
and the run then trains on the default, so a clean parse is not evidence the key was
read. Confirm it by auditing a config that actually sets it:

```bash
spectramr audit <your-config>.yaml
python scripts/ci/report_discarded_config_keys.py <dir-holding-it>   # is the key READ?
```

### Maintainer release checklist (GPU validation)

GitHub Actions runs only CPU jobs — fork-PR security and free-tier budget
preclude self-hosted GPU runners. The maintainer therefore validates GPU
behaviour manually before every tag.

We use [Lightning AI Studios](https://lightning.ai) for this. Lightning's free
tier provides a persistent CPU studio plus a monthly GPU-minutes allowance
sufficient for a single pre-release validation pass.

**Pre-release procedure (run before tagging `vX.Y.Z`):**

1. **Spin up a CUDA studio** on Lightning AI and clone the release branch.

   ```bash
   git clone --branch release/vX.Y.Z https://github.com/adnaneGdihi/spectramr.git
   cd spectramr
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -e ".[dev]"
   ```

2. **Run the GPU-only suite** with strict mode enabled.

   ```bash
   pytest -m gpu -v --tb=short
   pytest tests/physics/ -v --tb=short
   pytest tests/integration/test_determinism_sentinel.py -v
   ```

3. **Run the Tier-2 forward-probe audit** on every configuration the tree ships.
   (`experiments/` holds the full arm corpus in the research repository and the
   templates plus exemplar arms in the published one; the command is the same.)

   ```bash
   spectramr audit experiments/ --probe --strict
   ```

4. **Record the outcome** in `CHANGELOG.md` under the release being prepared:
   hardware (e.g. `Lightning Studio L4 24GB`),
   PyTorch version, CUDA version, pass/fail per suite. Future contributors
   can then compare future GPU runs against this baseline.

5. **Shut the studio down.** Free-tier GPU minutes are not refundable and a
   forgotten warm studio burns the budget overnight.

**Why not GPU CI?** A self-hosted runner connected to a public-repo
`pull_request` workflow can execute arbitrary code from forked PRs, with the
attendant secret-leakage and crypto-mining risk. GitHub's own documented
guidance is to never connect self-hosted runners to public-repo PRs from
forks. Until the project has either (a) a paid CI budget for hosted GPU
runners or (b) an institutional cluster with strict PR gating, GPU validation
stays manual and pre-release-only.

### Determinism sentinel

`tests/integration/test_determinism_sentinel.py` guards a load-bearing
invariant: that `initialize_accelerator(...)` produces bit-identical RNG
streams given the same seed and rank. Several training paradigms in this repo
(diffusion, cold diffusion, cycle-Bloch) rely on this for reproducible
sampling. If you change `src/spectramr/accelerator.py`, run this test before
opening the PR — it executes in under a second on CPU and catches the kind of
silent drift that no other test surfaces.

## Where new components live

spectraMR uses a registry-dispatcher pattern. Adding a component means writing
the class, decorating it, and mounting it on the package `__init__.py`.

| Component | Directory | Decorator |
|---|---|---|
| Training paradigm | `src/spectramr/infrastructure/training/strategies/` | subclass of `BaseTrainingStrategy`; register in `STRATEGY_CLASS_PATHS` |
| Model architecture | `src/spectramr/models/generators/` (or `discriminators/`) | `@register_model(name=..., training_mode=...)` |
| Loss | `src/spectramr/models/losses/` | `@register_loss(name=..., domain=...)` |
| Metric | `src/spectramr/core/metrics/` | `@register_metric(name=...)` |
| Dataset | `src/spectramr/data/datasets/` | `@register_dataset(name=...)` |
| Physics primitive | `src/spectramr/infrastructure/physics/` | direct import; no registry |

### Adding a new training paradigm (four-step recipe)

1. **Write the strategy.** Subclass `BaseTrainingStrategy` in a new file under
   `src/spectramr/infrastructure/training/strategies/`. Implement `train_step`,
   `validate_step`, and any overrides for `setup_optimizer`,
   `before_epoch`, `after_epoch`.
2. **Register the dispatch key.** Add an entry to
   `STRATEGY_CLASS_PATHS` in
   `src/spectramr/infrastructure/training/strategy_factory.py`, plus matching
   entries in `VALID_TRAINING_MODES` and `TRAINING_MODE_CONSTRAINTS` in
   `src/spectramr/config/validation_constants.py`.
3. **Write a YAML.** Place a reference config under
   `experiments/inprogress/<paradigm>/<arm>.yaml` with
   `training.strategy_class = "spectramr.infrastructure.training.strategies.<YourClass>"`
   and `training.training_mode = "<your_alias>"`.
4. **Land tests + docs.** Add a unit test under `tests/unit/infrastructure/training/`
   and a docs page under `docs/explanation/` or `docs/how_to/`.

Run `python -m spectramr.cli audit experiments/inprogress/<paradigm>/<arm>.yaml`
and confirm it returns zero blocking errors before opening the PR.

## Figure and data provenance

**No fastMRI-derived pixels are published from this repository.** The fastMRI
data-use agreement permits research use but not redistribution of the images,
and a rendered figure of a reconstruction *is* a redistribution of those
pixels. [M4Raw](https://doi.org/10.1038/s41597-023-02181-4) is CC-BY-4.0 and
may ship, provided the caption names the dataset and its licence.

This covers everything a reader can see — README images, `docs/` figures,
notebook outputs, and any test fixture built from a real acquisition.
Synthetic data (a seeded `torch.randn`, a phantom, simulator output) is
unrestricted, but state that in the file or the test that produces it:
"looks like noise" is not provenance, and the next person cannot re-derive it.

Today the question is moot rather than merely answered — the public export
ships **no image or PDF at all**, and its single binary fixture,
`tests/unit/data/builders/_coil_processing_parity_golden.pt`, is generated by
`torch.randn` under `manual_seed(1234)` / `manual_seed(4321)` in
`test_coil_processing_parity.py`. That zero is a *reading* taken against one
allowlist at one commit, not a standing property of the tree, so the release
procedure re-reads it from the export manifest at freeze. Adding the first
figure is therefore a deliberate act: provenance-check it individually, and
record the result here.

## DCO sign-off

Every commit must carry a `Signed-off-by:` line. Use `git commit -s` to add it
automatically. By signing off you certify the Developer Certificate of Origin
([DCO 1.1](https://developercertificate.org)) — that you wrote the code or
have the right to contribute it under the project's licence.

## PR checklist

Before requesting review, confirm:

- [ ] `pytest -m "not gpu"` passes locally
- [ ] Coverage did not regress
- [ ] Audit smoke test passes for any modified YAML configs
- [ ] Docs updated if public API changed
- [ ] Any new figure or binary fixture carries a provenance note (see *Figure and data provenance*)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR title uses a Conventional Commits prefix
- [ ] Every commit includes `Signed-off-by:`

## Code review

A solo maintainer reviews every PR. Expect a turnaround of 1–7 days. Review
focus is correctness, registry hygiene, and physics SSOT compliance.
Architectural feedback may request module relocation if a component lands
outside its canonical directory (see the table above).
