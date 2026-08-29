# Registries

MRIForge uses a registry-dispatcher pattern across **eight surfaces**.
Every registry maps a string name to a class or descriptor; YAML configs
reference the name; the framework looks it up at load time. This page is
the catalogue.

The pattern's invariant — and a load-bearing one — is that a registration
only fires when its module is **imported**. The corresponding package
`__init__.py` MUST explicitly import every file that contains a
decorator. A `@register_*` whose module is never imported is silently
dead. See the [add_model](../how_to/add_model.md) guide's "Mounting" section
for the failure mode and the cold-subprocess regression test that
catches it.

## The eight surfaces

| Registry | Module | Decorator | YAML key |
|---|---|---|---|
| Models | `mriforge.models.registry` | `@register_model` | `model.model_type` |
| Losses | `mriforge.models.losses.registry` | `@register_loss` | `losses.*_losses[*].name` |
| Metrics | `mriforge.core.metrics` | `@register_metric` | `metrics.<name>` / `validation.metrics[*]` |
| Datasets | `mriforge.data.datasets` | `@register_dataset` | `data.dataset_type` |
| Strategies | `mriforge.infrastructure.training.strategy_factory.STRATEGY_CLASS_PATHS` | dict entry | `training.strategy_class` |
| Training modes | `mriforge.config.validation_constants.VALID_TRAINING_MODES` + `TRAINING_MODE_CONSTRAINTS` | dict entry | `training.training_mode` |
| Mask types | `mriforge.infrastructure.physics.sampling.MaskType` | enum value | `data.kspace_sampling_mask` |
| Adapters | `mriforge.data.adapters.registry` | `@register_adapter` | `adapters.{pre_model, post_model, pre_loss_pred, ...}[*].name` |

## Live catalogue

To enumerate the entire live registry on your machine:

```bash
python -m mriforge.cli list-features --module all --format markdown
```

This prints every registered name in every surface, grouped by category.
Output is rendered as a tree:

```
Models (~305 entries):
  reconstruction:
    enhanced_deep_unet
    toeplitz_attention_unet
    bloch_lrs_attention_unet
    lanczos_attention_unet
    ...
  diffusion:
    ddpm_unet
    score_based_unet
    ...
  ...

Losses (~159 entries):
  image:  l1, l2, ssim, hfen, ...
  kspace: nmse_kspace, magnitude_kspace, phase_kspace, ...
  ...
```

## Strategies — the dispatch table

`STRATEGY_CLASS_PATHS` is a plain `dict[str, str]` mapping the
`training.training_mode` alias to the fully-qualified strategy class
path. The factory imports the class at runtime via `importlib`.

```python
# Excerpt: src/mriforge/infrastructure/training/strategy_factory.py
STRATEGY_CLASS_PATHS = {
    "gan": "mriforge.infrastructure.training.strategies.gan.GANTrainingStrategy",
    "diffusion": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
    "reconstruction": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
    "vae": "mriforge.infrastructure.training.strategies.vae.VAETrainingStrategy",
    "vqvae": "mriforge.infrastructure.training.strategies.vae.VQVAETrainingStrategy",
    ...
}
```

Adding a new mode requires a coordinated change in three places
(`STRATEGY_CLASS_PATHS`, `VALID_TRAINING_MODES`, `TRAINING_MODE_CONSTRAINTS`)
— see [add_paradigm § step 2](../how_to/add_paradigm.md#2-register-the-dispatch-key).

## Mask types — the operator-dispatch enum

```python
# src/mriforge/infrastructure/physics/sampling.py
class MaskType(str, Enum):
    CARTESIAN = "cartesian"
    VARIABLE_DENSITY = "variable_density"
    RADIAL = "radial"
    NUFFT = "nufft"
    SLE_KAPPA = "sle_kappa"     # 2026 SLE-κ Loewner-equation trajectories
```

`KSpaceMaskGenerator(mask_type=MaskType.RADIAL, ...)` dispatches to the
corresponding `_generate_radial` method. Adding a new mask type requires:

1. Add the enum value.
2. Add a `_generate_<name>` method.
3. Add the dispatch entry to `_DISPATCH`.
4. Add a test in `tests/unit/infrastructure/physics/test_sampling.py`.

## Adapters — bridging domain mismatches

When a model emits k-space but the loss wants image, an adapter chain
declared under `adapters.pre_loss_pred:` bridges the gap. Adapters are
themselves registered:

```python
# src/mriforge/data/adapters/fourier.py
@register_adapter(
    name="ifft_kspace_to_image",
    bridges_from={"domain": "kspace"},
    bridges_to={"domain": "complex_image"},
    invertible=True,
    insertion_points=("pre_model", "post_model", "pre_loss_pred",
                      "pre_loss_target", "pre_metric"),
)
class IFFTKspaceToImage(nn.Module):
    ...
```

Currently registered adapters (run `list-features --module adapters` for
the live list):

| Name | From → To | Use |
|---|---|---|
| `fft_image_to_kspace` | image → kspace | k-space loss on image-output model |
| `ifft_kspace_to_image` | kspace → complex_image | image loss on k-space-output model |
| `magnitude_from_complex` | complex_image → image | drop phase before SSIM / PSNR |
| `rss_coils_to_magnitude` | multi-coil image → image | RSS combine before scalar metrics |
| `real_imag_interleave_to_complex` | image (2C) → complex_image | model expects complex tensor |
| `complex_to_real_imag_interleave` | complex → image (2C) | model expects real-valued tensor |
| `slice_3d_to_2d` | 3D → 2D | 2D model on volumetric data |
| `gather_2d_to_3d` | 2D → 3D | reassemble slices |

No adapter bridges `spatial_dims=2 → 1`. Fingerprint-style models that
declare `spatial_dims=(1,)` (e.g. `mrf_tangent_score`) can't be used
with 2D image datasets directly — the audit catches this. See
[scripts/release/fix_audit_issues.py](https://github.com/adnaneGdihi/mriforge/blob/main/scripts/release/fix_audit_issues.py)
for the flagged-not-auto-fixed reasoning.

## Programmatic access

```python
import mriforge  # fires the clinical-use warning once

from mriforge.models.registry import MODEL_REGISTRY
from mriforge.models.losses.registry import LossRegistry
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory
from mriforge.data.adapters.registry import list_adapters
from mriforge.infrastructure.physics.sampling import MaskType

# Count what's available:
print(f"{len(MODEL_REGISTRY)} models")
print(f"{len(LossRegistry._registry)} losses")
print(f"{len(TrainingStrategyFactory.STRATEGY_CLASS_PATHS)} strategies")
print(f"{len(list_adapters())} adapters")
print(f"{len(list(MaskType))} mask types")
```

On a clean checkout this prints something like:

```
305 models
159 losses
80+ strategies
8 adapters
5 mask types
```

## Why a registry, not a giant `if/elif`?

- **O(1) dispatch** — no chain of comparisons.
- **No circular imports** — the registry holds class references, not the
  classes' dependencies.
- **YAML-driven configuration** — adding a new arm doesn't touch
  framework code beyond the new file.
- **Audit + discoverability** — every registered name shows up in the
  `list-features` catalogue and in the audit's allowed-values lists.

The cost is the "silent failure" trap mentioned at the top — decorators
need their modules imported. We pay that price with explicit
`__init__.py` re-exports and the cold-subprocess registration test.

## Related

- [Add a paradigm](../how_to/add_paradigm.md)
- [Add a model](../how_to/add_model.md)
- [Add a loss](../how_to/add_loss.md)
