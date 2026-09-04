# spectraMR

> **NOT FOR CLINICAL USE.** Research software only. See [DISCLAIMER.md](DISCLAIMER.md).

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22291316.svg)](https://doi.org/10.5281/zenodo.22291316)

<!-- The licence badge is the STATIC form, deliberately. The dynamic endpoint
     `img.shields.io/github/license/<owner>/<repo>.svg` reads the GitHub API as an
     anonymous client, which cannot see a private repository: it rendered
     `<title>license: repo not found</title>` here while returning HTTP 200, so it
     was displaying a broken badge in the one slot a reader most expects to be
     right. Measured 2026-09-03, by reading the SVG's <title> rather than its
     status code -- the same trap the conda badge note below records. Swap back to
     the dynamic form on the day this repository becomes public, and not before;
     until then the static string and `LICENSE` are kept in step by
     `pyproject.toml`'s `license = "Apache-2.0"`, which is what actually ships. -->


<!-- Restore each badge below on the day the thing it measures exists. Every one of
     them resolved to a 404 or an unfilled placeholder: nothing is on PyPI, the
     project is not imported on Read the Docs, codecov has no report, and the CI
     badge named `test.yml`, a workflow that does not exist in this repository.
     That last clause is now stale in one respect and kept for the record: the
     badge below was retargeted onto `pr-required.yml`, which does exist and is
     the single required aggregator.
     The note that stood here added "and Actions is disabled for the project
     anyway, so no lane runs at all" -- that was true when the repo was created
     and is not true now: Actions is ENABLED on adnaneGdihi/spectramr, and the
     merged dependabot PRs there ran real checks. So the CI badge below is the
     one that could be restored today; the rest still measure things that do not
     exist. A badge is a claim a reader cannot check without clicking; a broken
     one is worse than an absent one.

     The last two are new. Each is left deliberately UNFILLED, and each names the
     one number that fills it -- because for both of them, the obvious way to
     check a badge does not work. Measured 2026-09-03, not assumed:

       * DOI -- FILLED 2026-09-04, and promoted out of this block to the live badge
       row at the top. It carries the CONCEPT DOI (10.5281/zenodo.22291316),
       NOT the per-deposit version DOI that Zenodo's page shows you first -- whose
       number is deliberately not written out here, because the test that keeps this
       badge honest greps the whole file for it and a worked example in a comment
       would be indistinguishable from the mistake. It is recorded once, in
       `zenodo_deposit.VERSION_DOI`. Zenodo mints both: the concept DOI is the parent that
       always redirects to the newest version (verified -- it resolves to record
       22291317 today), while the version DOI is frozen on one deposit and would
       silently stop tracking the project at v0.2.0. `zenodo_deposit.report_badge`
       prefers `conceptdoi` for exactly this reason and owns the line's shape;
       `scripts/release/zenodo_deposit.py:CONCEPT_DOI` owns the number, and
       `tests/unit/release/test_zenodo_deposit.py` pins README, the BibTeX block
       below and `CITATION.cff` to it so the four cannot drift apart.

       Kept for the record, because it is the trap that made the form non-obvious:
       Zenodo publishes TWO badge endpoints and only one applies here. The repo-id
       form -- `/badge/<github repo id>.svg` linking to `/badge/latestdoi/<id>` --
       belongs to Zenodo's GitHub INTEGRATION, which only sees public repositories
       and only fires on a published GitHub release. `zenodo_deposit.py` deposits
       over the REST API instead, deliberately, because that works while the
       repository is private -- so it never registers the repo-id mapping. Probed
       against this repo's real numeric id (1347566284 -- a numeric id survives a
       rename, which is why the badge is keyed on it and not on the name):
       `/badge/1347566284.svg` -> 404 and `/badge/latestdoi/1347566284` -> 404. Do
       NOT "restore" the repo-id form; it is the shape that looks filled and is dead.

       And a correction to the advice in the conda bullet below, which does NOT
       generalise to this endpoint: `/badge/DOI/<doi>.svg` emits no `<title>` at
       all, so grepping for one reports every Zenodo badge as broken. Worse, the
       endpoint does not validate its argument -- it renders whatever string you
       hand it. Measured 2026-09-04: a real DOI, a nonexistent zenodo id and the
       literal `not-a-doi-at-all` all returned HTTP 200 with a well-formed SVG
       (1217 / 1223 / 1200 bytes) reading back the string it was given. Neither the
       status code NOR the rendered content can tell a live DOI from a typo. The
       only probe that discriminates is resolving the DOI itself:
       `curl -sI https://doi.org/<doi>` -> 302 for a real one, 404 for a fake.
       * Conda. Keyed on the channel `adnanegdihi`, which is the GitHub handle
         lower-cased and NOT a verified anaconda.org account -- if the account is
         named differently, `ANACONDA_CHANNEL` (the repository variable the
         workflow reads) and this URL must be changed together; the badge cannot
         read the variable. Its status code proves nothing in either direction:
         shields serves HTTP 200 for a channel/package that does not exist, with
         the words "conda: not found" rendered INTO the SVG, and anaconda.org
         serves HTTP 200 for a nonsense path too (a nonexistent channel returned
         23,643 bytes with a 404 marker in the body). So verify this badge by
         LOOKING at it, or by grepping the SVG's `<title>` -- never by curl'ing
         for a 2xx.

[![PyPI version](https://img.shields.io/pypi/v/spectramr.svg)](https://pypi.org/project/spectramr/)
[![Python](https://img.shields.io/pypi/pyversions/spectramr.svg)](https://pypi.org/project/spectramr/)
[![CI](https://github.com/adnaneGdihi/spectramr/actions/workflows/pr-required.yml/badge.svg)](https://github.com/adnaneGdihi/spectramr/actions/workflows/pr-required.yml)
[![codecov](https://codecov.io/gh/adnaneGdihi/spectramr/branch/main/graph/badge.svg)](https://codecov.io/gh/adnaneGdihi/spectramr)
[![Documentation Status](https://readthedocs.org/projects/spectramr/badge/?version=latest)](https://spectramr.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/spectramr/month)](https://pepy.tech/project/spectramr)
[![Conda](https://img.shields.io/conda/vn/adnanegdihi/spectramr.svg)](https://anaconda.org/adnanegdihi/spectramr)
-->

spectraMR is a research framework for MRI reconstruction, super-resolution,
quantitative mapping, and generative modelling. It registers 153 training
strategies (selectable through 206 `training_mode` spellings), 586 model
architectures, 217 losses, and a single-source-of-truth MRI physics layer
(centered FFT, Cartesian / VD / radial / NUFFT sampling masks, ESPIRiT and
SIREN-PINN coil maps, hard / soft data consistency, Bloch / motion / B0 / B1⁻
simulation).

These are **registration** counts, measured on the shipped package rather than
estimated, under `pip install spectramr[mri]` -- the install the quick start
prescribes. They are a property of the *installed extras*, not of the
distribution: a bare `pip install spectramr` registers **175** models rather than
586, because a model whose module fails to import is not registered at all
(loudly -- discovery names the module and the missing package). See
[Installation](#installation).

Every registered model, loss, metric and transform is reachable from
a cold import — 586 / 217 / 211 / 10, cold-probe equal to walk, verified by
`scripts/maintenance/prove_reachable.py --audit`; the 206 strategy paths are a
static dict and all 206 resolve. That is a reachability claim, not a validation
one: it says a config can select the component, not that the component is
benchmarked. Per-regime maturity is graded LIVE / PARTIAL / EVAL_ONLY / STUB by
the `Maturity` ledger, and `docs/known_limitations.rst` records what is known
not to work. Re-measure rather than quoting these; they move week to week.

## Quick start

```bash
pip install spectramr[mri]
```

A single forward pass through a small U-Net via the registry:

```python
import torch
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

# Required. The registry is EMPTY on a plain import -- a model is registered
# only once the module holding its decorator has been imported, and this call
# is what curates those imports. Without it MODEL_REGISTRY.get() returns None
# and the next line raises TypeError.
populate_model_registry()

entry = MODEL_REGISTRY.get("toeplitz_attention_unet")
model = entry["class"](in_channels=2, out_channels=2)
y = model(torch.randn(1, 2, 64, 64))    # -> torch.Size([1, 2, 64, 64])
```

Counting for yourself:

```python
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY
from spectramr.models.losses.registry import LossRegistry
from spectramr.core.metrics.registry import MetricsRegistry

populate_model_registry()
paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS   # a CLASS attribute
len(MODEL_REGISTRY), len(LossRegistry.list_available()), \
    len(MetricsRegistry.list_available()), len(set(paths.values())), len(paths)
```

The same model from a YAML:

```yaml
# an excerpt of experiments/templates/comprehensive_config_template.yaml
config_version: '1.0'
model:
  model_type: toeplitz_attention_unet
  in_channels: 2
  out_channels: 2
training:
  training_mode: reconstruction
  strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
```

`config_version: '1.0'` is the only accepted value; anything else is refused at
load, with the accepted set named in the error.

The excerpt above is a fragment, not a runnable config. The complete template
ships and passes the audit as-is:

```bash
spectramr audit experiments/templates/comprehensive_config_template.yaml
spectramr train --config experiments/templates/comprehensive_config_template.yaml
```

`audit` runs Tier 0 (schema) and Tier 1 (health checks); add `--probe` for a
Tier 2 synthetic forward pass. It exits 0 on a pass, 1 on warnings, 2 on errors
-- and it is `--strict` by default, so a warning is not a pass.

## Installation

Optional dependencies come in two kinds. **Feature** groups gate a capability
(absent, it raises at construction — never a silent fallback); **role** groups
gate a workflow and are imported by nothing under `src/`.

```bash
pip install spectramr                 # core only
pip install spectramr[mri]            # TorchIO, MONAI, nibabel, torchkbnufft, pydicom
pip install spectramr[diffusion]      # diffusers — pretrained SD-VAE backbone
pip install spectramr[viz]            # matplotlib, tensorboard, seaborn, plotly
pip install spectramr[hpo]            # Optuna
pip install spectramr[all]            # EVERYTHING that installs in one shot
pip install spectramr[dev]            # all + the config-migration toolchain
```

The role groups are installable on their own, which is what CI lanes do:
`[test]` (pytest + plugins), `[qa]` (ruff, mypy, pre-commit, pip-audit,
codespell, detect-secrets), `[docs]` (Sphinx) and `[profile]` (Scalene, GPUtil,
nvtx). `[all]` contains all four, so it is a superset of whatever any lane
installs, and `[dev]` is `[all]` plus tooling nothing else needs.

Three groups are **not** in `[all]`, each because it physically cannot install
in a single resolve — not as a matter of curation, and each verified by an
actual build rather than assumed: `mamba` compiles the CUDA selective-scan
kernel; `attention` fails because flash-attn omits torch from its build
requirements; `radiomics` has no cp312 wheel and its C extension fails to
compile. (`bnb` and `deepspeed` were long excluded on the assumption that they
need a CUDA toolchain — both build clean under isolation, so they are in.) On a
node with `nvcc`:

```bash
pip install -e '.[all]'
pip install -e '.[mamba]' --no-build-isolation
```

`[mri]` is the practical floor, not a convenience. The core install is a
genuine subset and it is a small one: it registers **175** of the 586 models,
and `spectramr.infrastructure.training` cannot be imported at all, because the
dataset layer imports TorchIO unconditionally. `spectramr --help`,
`spectramr --version` and the loss registry (217, unaffected) still work. Install
`[mri]` unless you are deliberately vendoring a subset.

Three extras sit deliberately **outside** `[all]`, because each pulls a heavy or
platform-specific build:

```bash
pip install spectramr[bnb]          # bitsandbytes
pip install spectramr[deepspeed]    # deepspeed
pip install -e '.[mamba]' --no-build-isolation   # mamba-ssm, causal-conv1d
```

`[mamba]` compiles a CUDA selective-scan kernel, needs `nvcc`, and must be
installed **after** torch is present. Without it, Mamba/SSM models fail loudly
rather than degrading silently.

A few registered components need a package that no extra installs -- see
[docs/known_limitations.rst](docs/known_limitations.rst).

### What pip actually resolves

The `pytorch-cu126` index pin in `pyproject.toml` lives under `[tool.uv.sources]`
and `[[tool.uv.index]]`. **Neither reaches wheel metadata** -- the published
requirement is a bare `torch>=2.11` -- so installing from PyPI does not give you
the pinned build. Measured on a fresh venv from the published wheel:

| | Resolved |
|---|---|
| torch | `2.13.0+cu130` (CUDA **13.0**, not the pinned 12.6) |
| CUDA runtime | the full `nvidia-*-cu13` stack, `cuda-toolkit`, `triton` -- pulled automatically |
| pandas | `3.0.5` -- `pandas>=2.3` admits the 3.x major |
| Total | **5.1 GB** |

So CUDA-enabled PyTorch is *not* installed separately: you get it by default, at
a CUDA version this project does not pin. Pin it yourself if that matters, by
installing torch **first** from the index you want -- afterwards costs a
multi-gigabyte reinstall:

```bash
# CUDA 12.6 -- what this project pins, and the only wheel lane that still ships
# sm_70 for Volta / V100 (compute capability 7.0) GPUs. `uv.lock` resolves
# torch 2.13.0+cu126 / torchvision 0.28.0+cu126 from this index.
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

For a CPU-only or air-gapped machine, substitute the `cpu` index. The CI runs
against the CPU wheel; downstream GPU work is your responsibility.

## What's in the box

| Layer | Where | Highlights |
|---|---|---|
| Training paradigms | `spectramr.infrastructure.training.strategies` | GAN, diffusion (cold / score / Lévy / resetting), VAE/VQ-VAE, MAE/SSL, reconstruction, domain adaptation, physics-driven (PINN), disentangled, sensitivity-estimation, cycle-Bloch |
| Model registry | `spectramr.models` | U-Nets, complex U-Nets, attention-bottleneck nets, geometric-prior nets (hyperbolic, Heisenberg, tropical, sheaf, …), state-space (S4D / Hyena), Toeplitz / Bloch-LRS / Lanczos / MPS attention |
| Loss registry | `spectramr.models.losses` | image, k-space, complex, physics-residual, adversarial, latent, distillation, virtual-fiducial, intertwining (spectral-triple) |
| Physics SSOT | `spectramr.infrastructure.physics` | `fft2c`/`ifft2c`, mask generators (Cartesian, VD, radial, NUFFT, SLE-κ), ESPIRiT, SENSE, PINN, Bloch, motion, B0, B1⁻ |
| Configuration | `spectramr.config` | `config_version: '1.0'` frozen Pydantic v2 schema, paradigm-specific sub-schemas, three-tier audit ladder |
| CLI | `spectramr.cli` | 24 verbs; `spectramr --help` lists them. The common ones are `audit`, `train`, `predict`, `infer`, `benchmark`, `hpo`, `report`, `doctor` |

## Maturity by regime

A **regime** is the physical acquisition setting an experiment declares
(`workflow.regime`). Each is graded against the live registries by
`spectramr.config.schemas.enums.Maturity`, and the grades are enforced by
`tests/unit/domain/workflows/test_maturity_ledger.py` -- they are read off the
code, not maintained by hand.

| Maturity | Regimes |
|---|---|
| **LIVE** -- registered forward model, regime-tagged strategy and metrics | `mri_structural`, `mri_quantitative`, `mri_diffusion_weighted`, `mri_dynamic`, `mri_functional`, `mri_perfusion`, `mri_flow`, `mri_spectroscopy`, `mri_fingerprinting` |
| **STUB** -- nothing exists; every pipeline raises `WorkflowNotImplementedError` | `ct`, `xray`, `ultrasound`, `optical`, `nmr_spectroscopy` |

The five STUB regimes are declared so the vocabulary is closed and a typo raises
instead of silently meaning nothing. They are not implemented, and spectraMR does
not claim to be a CT, X-ray, ultrasound or optical framework.

The ledger grades **regimes**, not individual models. No per-model guarantee is
made or implied.

## Citing

If you use spectraMR in your research, please cite the software:

```bibtex
@software{gdihi2026spectramr,
  author    = {Gdihi, Adnane},
  title     = {spectraMR: A Multi-Paradigm Research Framework for MRI Reconstruction, Super-Resolution, and Generative Modelling},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22291316},
  url       = {https://github.com/adnaneGdihi/spectramr},
  version   = {0.1.0}
}
```

GitHub renders a "Cite this repository" button from [CITATION.cff](CITATION.cff)
that produces this BibTeX automatically. Citation tools that consume CFF 1.2.0
will pick the version up directly.

## Documentation

Full documentation is hosted at https://spectramr.readthedocs.io and follows the
[Diátaxis](https://diataxis.fr) quadrants:

- **Tutorials** — guided walk-throughs from `pip install` to a first
  reconstruction.
- **How-to guides** — add a paradigm, add a model, add a loss, write an
  experiment YAML.
- **Reference** — auto-generated API documentation plus the YAML-schema
  reference and registry catalogues.
- **Explanation** — clean-architecture layering, the audit ladder, the physics
  SSOT discipline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Fork, branch off `main`, implement.
2. `pre-commit install` and let it run on every commit.
3. `pytest -m "not gpu"` must pass. If you touched a YAML config, run
   `spectramr audit <path>` on it -- the audit is `--strict` by default, so a
   warning is a failure.
4. PR title uses a Conventional Commits prefix; commits use `git commit -s`
   (DCO sign-off).
5. CI runs a single required aggregator over: changed-line lint, repository
   guards, architecture fitness functions, a collection pass over the unit
   suite, a physics check, a config-schema audit, and a security scan. The
   collection pass **imports** every unit-test module without executing the
   tests, so a green lane is not "the unit suite passed" -- run `pytest`
   locally.

By participating you agree to the
[Contributor Covenant 2.1](CODE_OF_CONDUCT.md).

## Licence

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution of
upstream dependencies.

## Disclaimer

spectraMR is **NOT FOR CLINICAL USE**. It is research software and has not been
evaluated by any regulatory authority. See [DISCLAIMER.md](DISCLAIMER.md).
