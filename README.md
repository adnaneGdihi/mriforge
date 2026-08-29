# MRIForge

> **NOT FOR CLINICAL USE.** Research software only. See [DISCLAIMER.md](DISCLAIMER.md).

[![PyPI version](https://img.shields.io/pypi/v/mriforge.svg)](https://pypi.org/project/mriforge/)
[![Python](https://img.shields.io/pypi/pyversions/mriforge.svg)](https://pypi.org/project/mriforge/)
[![License](https://img.shields.io/github/license/adnaneGdihi/mriforge.svg)](LICENSE)
[![CI](https://github.com/adnaneGdihi/mriforge/actions/workflows/pr-required.yml/badge.svg)](https://github.com/adnaneGdihi/mriforge/actions/workflows/pr-required.yml)
[![codecov](https://codecov.io/gh/adnaneGdihi/mriforge/branch/main/graph/badge.svg)](https://codecov.io/gh/adnaneGdihi/mriforge)
[![Documentation Status](https://readthedocs.org/projects/mriforge/badge/?version=latest)](https://mriforge.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/mriforge/month)](https://pepy.tech/project/mriforge)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/<TO_BE_FILLED_AFTER_FIRST_RELEASE>.svg)](https://doi.org/<TO_BE_FILLED>)

MRIForge is a research framework for MRI reconstruction, super-resolution,
quantitative mapping, and generative modelling. It registers 153 training
strategies (selectable through 206 `training_mode` spellings), 586 model
architectures, 217 losses, and a single-source-of-truth MRI physics layer
(centered FFT, Cartesian / VD / radial / NUFFT sampling masks, ESPIRiT and
SIREN-PINN coil maps, hard / soft data consistency, Bloch / motion / B0 / B1⁻
simulation).

These are **registration** counts, measured on the shipped package rather than
estimated. Every registered model, loss, metric and transform is reachable from
a cold import — 586 / 217 / 211 / 10, cold-probe equal to walk, verified by
`scripts/maintenance/prove_reachable.py --audit`; the 206 strategy paths are a
static dict and all 206 resolve. That is a reachability claim, not a validation
one: it says a config can select the component, not that the component is
benchmarked. Per-regime maturity is graded LIVE / PARTIAL / EVAL_ONLY / STUB by
the `Maturity` ledger, and `docs/known_limitations.rst` records what is known
not to work. Re-measure rather than quoting these; they move week to week.

## Quick start

```bash
pip install mriforge[mri]
```

A single forward pass through a small U-Net via the registry:

```python
import torch
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY

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
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY
from mriforge.models.losses.registry import LossRegistry
from mriforge.core.metrics.registry import MetricsRegistry

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
  strategy_class: mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
```

`config_version: '1.0'` is the only accepted value; anything else is refused at
load, with the accepted set named in the error.

The excerpt above is a fragment, not a runnable config. The complete template
ships and passes the audit as-is:

```bash
mriforge audit experiments/templates/comprehensive_config_template.yaml
mriforge train --config experiments/templates/comprehensive_config_template.yaml
```

`audit` runs Tier 0 (schema) and Tier 1 (health checks); add `--probe` for a
Tier 2 synthetic forward pass. It exits 0 on a pass, 1 on warnings, 2 on errors
-- and it is `--strict` by default, so a warning is not a pass.

## Installation

```bash
pip install mriforge               # core only
pip install mriforge[mri]          # TorchIO, MONAI, nibabel, torchkbnufft
pip install mriforge[diffusion]    # diffusers
pip install mriforge[viz]          # matplotlib, tensorboard, seaborn, scikit-learn, plotly
pip install mriforge[hpo]          # Optuna
pip install mriforge[iqa]          # piq
pip install mriforge[eval]         # lpips
pip install mriforge[quality]      # PyWavelets, scikit-learn, statsmodels
pip install mriforge[topology]     # gudhi, POT
pip install mriforge[schedulefree]  # schedule-free optimizers
pip install mriforge[profile]      # scalene
pip install mriforge[test]         # pytest and plugins
pip install mriforge[docs]         # Sphinx and theme
pip install mriforge[all]          # all of the above
pip install mriforge[dev]          # [all] + pre-commit, ruff, mypy, ruamel.yaml
```

Three extras sit deliberately **outside** `[all]`, because each pulls a heavy or
platform-specific build:

```bash
pip install mriforge[bnb]          # bitsandbytes
pip install mriforge[deepspeed]    # deepspeed
pip install -e '.[mamba]' --no-build-isolation   # mamba-ssm, causal-conv1d
```

`[mamba]` compiles a CUDA selective-scan kernel, needs `nvcc`, and must be
installed **after** torch is present. Without it, Mamba/SSM models fail loudly
rather than degrading silently.

A few registered components need a package that no extra installs -- see
[docs/known_limitations.rst](docs/known_limitations.rst).

CUDA-enabled PyTorch is installed separately. Install the wheel matching your
CUDA version from `https://download.pytorch.org/whl/`:

```bash
# Example: CUDA 12.9
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

The CI runs against the CPU wheel; downstream GPU work is your responsibility.

## What's in the box

| Layer | Where | Highlights |
|---|---|---|
| Training paradigms | `mriforge.infrastructure.training.strategies` | GAN, diffusion (cold / score / Lévy / resetting), VAE/VQ-VAE, MAE/SSL, reconstruction, domain adaptation, physics-driven (PINN), disentangled, sensitivity-estimation, cycle-Bloch |
| Model registry | `mriforge.models` | U-Nets, complex U-Nets, attention-bottleneck nets, geometric-prior nets (hyperbolic, Heisenberg, tropical, sheaf, …), state-space (S4D / Hyena), Toeplitz / Bloch-LRS / Lanczos / MPS attention |
| Loss registry | `mriforge.models.losses` | image, k-space, complex, physics-residual, adversarial, latent, distillation, virtual-fiducial, intertwining (spectral-triple) |
| Physics SSOT | `mriforge.infrastructure.physics` | `fft2c`/`ifft2c`, mask generators (Cartesian, VD, radial, NUFFT, SLE-κ), ESPIRiT, SENSE, PINN, Bloch, motion, B0, B1⁻ |
| Configuration | `mriforge.config` | `config_version: '1.0'` frozen Pydantic v2 schema, paradigm-specific sub-schemas, three-tier audit ladder |
| CLI | `mriforge.cli` | 24 verbs; `mriforge --help` lists them. The common ones are `audit`, `train`, `predict`, `infer`, `benchmark`, `hpo`, `report`, `doctor` |

## Maturity by regime

A **regime** is the physical acquisition setting an experiment declares
(`workflow.regime`). Each is graded against the live registries by
`mriforge.config.schemas.enums.Maturity`, and the grades are enforced by
`tests/unit/domain/workflows/test_maturity_ledger.py` -- they are read off the
code, not maintained by hand.

| Maturity | Regimes |
|---|---|
| **LIVE** -- registered forward model, regime-tagged strategy and metrics | `mri_structural`, `mri_quantitative`, `mri_diffusion_weighted`, `mri_dynamic`, `mri_functional`, `mri_perfusion`, `mri_flow`, `mri_spectroscopy`, `mri_fingerprinting` |
| **STUB** -- nothing exists; every pipeline raises `WorkflowNotImplementedError` | `ct`, `xray`, `ultrasound`, `optical`, `nmr_spectroscopy` |

The five STUB regimes are declared so the vocabulary is closed and a typo raises
instead of silently meaning nothing. They are not implemented, and MRIForge does
not claim to be a CT, X-ray, ultrasound or optical framework.

The ledger grades **regimes**, not individual models. No per-model guarantee is
made or implied.

## Citing

If you use MRIForge in your research, please cite the software:

```bibtex
@software{gdihi2026mriforge,
  author    = {Gdihi, Adnane},
  title     = {MRIForge: A Multi-Paradigm Research Framework for MRI Reconstruction, Super-Resolution, and Generative Modelling},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {<TO_BE_FILLED_AFTER_FIRST_RELEASE>},
  url       = {https://github.com/adnaneGdihi/mriforge},
  version   = {0.1.0}
}
```

GitHub renders a "Cite this repository" button from [CITATION.cff](CITATION.cff)
that produces this BibTeX automatically. Citation tools that consume CFF 1.2.0
will pick the version up directly.

## Documentation

Full documentation is hosted at https://mriforge.readthedocs.io and follows the
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
   `mriforge audit <path>` on it -- the audit is `--strict` by default, so a
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

MRIForge is **NOT FOR CLINICAL USE**. It is research software and has not been
evaluated by any regulatory authority. See [DISCLAIMER.md](DISCLAIMER.md).
