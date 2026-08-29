# YAML schema reference

The authoritative schema is the Pydantic v2 model tree under
`src/mriforge/config/schemas/`. This page is the human-readable summary
of the v6.0 layout — every top-level block, every required field, every
common enum.

For the short tutorial-style intro see
[write_experiment_yaml](../how_to/write_experiment_yaml.md). For the
audit ladder that enforces this schema see
[audit_ladder](../explanation/audit_ladder.md).

## Top-level: `TrainingSettings`

A v6.0 YAML deserialises into a frozen `TrainingSettings` Pydantic
model. The settings object is **passed by reference** through every
layer — no module re-parses the YAML, no layer mutates it.

The 12 required top-level blocks:

```
config_version           # literal '6.0'
metadata                 # name, description, tags, version
acceleration             # base_acceleration, center_fraction
artifacts                # persistent_root
checkpoint               # enabled + retention policy
data                     # data_root, dataset_type, patch_size, ...
losses                   # output_domain + image/kspace/complex/latent lists
logging                  # experiment_name, level
loss_logging             # CSV/buffer config
metrics                  # best_metric_name, domain
model                    # model_type + model_kwargs
optimization             # learning_rate, optimizer_type, weight_decay
training                 # training_mode + strategy_class + epochs
validation               # enabled + metrics
physics                  # required block, may be empty {}
```

## `metadata`

```yaml
metadata:
  name: my_arm                       # required, snake_case
  description: |
    Free-form text. Multi-line is fine.
  tags:
    paradigm: diffusion              # high-level paradigm family
    type: baseline                   # baseline | ablation | breakthrough | …
    novelty: my_arm                  # short slug identifying the novelty
  version: '6.0'                     # must match config_version
```

`tags` is a free-form dict; the audit doesn't enforce specific keys but
the convention above keeps the smoke-mosaic plotter happy.

## `data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `dataset_type` | enum | yes | `kspace`, `nifti`, `nifti_paired`, `npy_slice`, `m4raw`, `preprocessed`, `dicom`, `pde_synthetic`, … |
| `data_root` | path | yes | Use `${MRIFORGE_DATA_ROOT}/...` |
| `patch_size` | `list[int, int, int]` | yes | (H, W, D); D=1 for 2D |
| `batch_size` | int | yes | per-GPU |
| `coil_processing_mode` | enum | yes | `rss`, `sense`, `as_is` |
| `num_workers` | int | no | dataloader workers (default 4) |
| `validation_split` | float | no | 0.0–1.0 |
| `manifest_path` | path | no | explicit pkl/json manifest |
| `index_path` | path | no | pre-computed dataset index for fast startup |
| `phase_encode_axis` | int | no | required for EPI distortion strategies |
| `expose_*` | bool | no | extra-batch-key flags (see [write_experiment_yaml § gotcha 6](../how_to/write_experiment_yaml.md#6-strategies-that-read-extra-batch-keys-need-dataexpose_-flags)) |
| `augmentation` | object | no | `{enabled, probability, ...}` |

## `model`

| Field | Type | Required | Notes |
|---|---|---|---|
| `model_type` | str | yes | A registered name; see `list-features --module models` |
| `in_channels` | int | yes | Match `dataset_type` × `coil_processing_mode` |
| `out_channels` | int | yes | Usually equals `in_channels` |
| `input_type` | enum | yes | `image`, `kspace`, `complex_image` |
| `model_kwargs` | dict | yes | Forwarded to the model constructor (e.g. `features: [32, 64, 128, 256]`). May be empty `{}` |

## `losses`

| Field | Type | Required | Notes |
|---|---|---|---|
| `output_domain` | enum | yes | `image`, `kspace`, `complex`, `latent` |
| `image_losses` | list | yes (may be empty) | Each entry: `{name, weight, enabled, ...kwargs}` |
| `kspace_losses` | list | yes (may be empty) | |
| `complex_losses` | list | yes (may be empty) | |
| `latent_losses` | list | no | Only when `output_domain == "latent"` |

Each loss entry:

```yaml
- name: l1                # registered loss name
  weight: 1.0
  enabled: true
  # any additional kwargs forwarded to the loss constructor
```

The audit's `loss_domain_block_match` check enforces that every enabled
loss's registered `domain` matches the block it appears in (with the
escape hatches documented in
[add_loss § domain-spillover](../how_to/add_loss.md#domain-spillover-when-the-audit-fights-you)).

## `optimization`

| Field | Type | Required | Notes |
|---|---|---|---|
| `optimizer_type` | enum | yes | `adamw`, `adam`, `sgd`, `rmsprop` |
| `learning_rate` | float | yes | |
| `weight_decay` | float | no | 0 by default |
| `enable_gradient_clipping` | bool | no | |
| `gradient_clip_value` | float | no | |
| `use_amp` | bool | no | Mixed precision. Most strategies are AMP-safe via `fft_ops`'s FP32 wrappers |
| `lr_schedule` | object | no | Cosine, step, plateau, … |

## `training`

| Field | Type | Required | Notes |
|---|---|---|---|
| `training_mode` | enum | yes | One of `VALID_TRAINING_MODES` |
| `strategy_class` | dotted path | yes | The corresponding entry from `STRATEGY_CLASS_PATHS` |
| `epochs` | int | yes | |
| `max_iterations` | int | no | Cap (mutually exclusive with `epochs` for some paradigms) |
| `device` | str | yes | `cuda`, `cpu`, `cuda:0` |
| `seed` | int | yes | Reproducibility |
| `output_dir` | path | yes | Conventional: `experiments/results/<name>` |
| `diffusion` | object | conditional | Required for `*Diffusion*` strategies — see [gotcha #1](../how_to/write_experiment_yaml.md#1-diffusion-paradigms-need-trainingdiffusion--num_timesteps) |
| `num_timesteps` | int | conditional | Required alongside `diffusion` |

## `validation`

```yaml
validation:
  enabled: true
  metrics: [psnr, ssim]              # registered metric names
  split: 0.1                          # fraction of data if no explicit split
  eval_interval: 5000                 # iterations between val passes
  val_batch_size: 2                   # may differ from training batch_size
```

## `metrics`

```yaml
metrics:
  best_metric_name: val_psnr          # what early-stopping watches
  best_metric_mode: max               # max | min
  compute_psnr: true
  compute_ssim: true
  compute_hfen: true
  domain: image                       # which metric_compute_* hooks fire
  metric_interval: 500
```

## `acceleration`

```yaml
acceleration:
  base_acceleration: 4                # 1, 2, 4, 8
  center_fraction: 0.08               # 0.04 .. 0.5
  max_acceleration: 8                 # optional cap for curriculum schedules
```

## `physics`

May be `{}` for image-domain experiments. For k-space / SENSE
experiments, declares the operator chain:

```yaml
physics:
  fft: {type: centered, norm: ortho}
  mask_generator:
    pattern: variable_density
    center_fraction: 0.08
  sensitivity:
    method: espirit                  # or 'siren_pinn'
  data_consistency: {mode: soft, weight: 0.01}
```

## Conditional blocks

Some paradigms require additional top-level blocks. The audit's
`paradigm_required_fields` check enforces them.

| Paradigm family | Required block | Fields |
|---|---|---|
| Any `*Diffusion*` | `training.diffusion` + `training.num_timesteps` | `type`, `degradation`, `num_timesteps` |
| MRF | `mrf` | `spiral_rotation_schedule`, `n_timepoints`, `sar_max_w_per_kg`, `gradient_slew_rate_t_per_m_per_s`, `phantom_calibration_path` |
| EPI distortion | `data.phase_encode_axis` | -2 or -1 |
| Federated | `federated` | `n_sites`, `aggregation_method`, `dp_noise_sigma` |
| Continual learning | `cl` | `memory_size`, `replay_strategy` |
| HPO | `hpo` | `search_space`, `n_trials`, `study_name` |

See `src/mriforge/config/schemas/training/` for the per-paradigm sub-schemas.

## Adapters (optional)

When the model's output domain disagrees with the loss's input domain,
declare the bridge:

```yaml
adapters:
  pre_loss_pred:
    - {name: ifft_kspace_to_image}
    - {name: magnitude_from_complex}
  pre_loss_target:
    - {name: ifft_kspace_to_image}
    - {name: magnitude_from_complex}
```

See [registries § adapters](registries.md#adapters--bridging-domain-mismatches)
for the list of available adapter names.

## Reporting (optional)

To auto-generate end-of-training figures:

```yaml
reporting:
  enabled: true
  task: reconstruction               # or 'super_resolution', 'synthesis'
  method_name: my_arm
  out_subdir: report
  figures:
    - fig_1_2_learning_curves
    - fig_1_3_loss_decomposition
    # paradigm-specific:
    - fig_c1_beltrami_field          # SFC / Beltrami arms
    - fig_c2_spd_geodesic            # SPD-manifold arms
    - fig_c4_fingerprint_embedding   # MRF arms
```

## Generating the JSON schema

```bash
python -m mriforge.cli audit --json-schema > schema.json
```

Useful for IDE auto-completion (VS Code, JetBrains) — point the YAML
schema extension at the generated file.

## Reference template

The canonical empty-but-complete template lives at
`src/mriforge/config/schemas/templates/v1.0_reference.yaml`. Copy it
when starting a new experiment from scratch.

## Related

- [Write an experiment YAML](../how_to/write_experiment_yaml.md) — tutorial-style intro with the gotcha checklist.
- [Audit ladder](../explanation/audit_ladder.md) — what each tier enforces against this schema.
- [Registries reference](registries.md) — every registered name available in YAML.
