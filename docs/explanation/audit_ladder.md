# The audit ladder

Every YAML config can be validated at three tiers, each progressively
more expensive and more thorough. The whole apparatus is the **audit
ladder** — a load-bearing piece of the framework that catches
mis-configurations *before* training starts, when fixing them is cheap.

## Why a three-tier ladder?

The cost of a config bug grows with how far the bug propagates:

| Where caught | Cost |
|---|---|
| Tier 0 (schema) | ~50 ms — fix the YAML, retry |
| Tier 1 (cross-validation) | ~50–100 ms — fix the YAML, retry |
| Tier 2 (synthetic probe) | ~30 s on GPU — fix the YAML or the model, retry |
| Training start-up | minutes — wasted GPU allocation, lost slot in SLURM queue |
| Mid-training | hours-to-days — checkpoint partially trained, restart |
| Post-training | weeks — paper deadline impact |

Each tier is designed to fail fast on the bugs the cheaper tier
*cannot* see, and to skip the bugs the cheaper tier *can*.

## Tier 0 — Schema (~50 ms)

What it does: **Pydantic v2 validation of every key, type, and enum
value in the YAML.** Pure structural check; no semantics, no imports.

Examples of what it catches:

- Missing required block (`config_version`, `data`, `model`, `losses`, …)
- Wrong type (`epochs: "100"` should be `int`)
- Unknown key (`extra="forbid"` rejects anything not in the schema)
- Enum violation (`coil_processing_mode: foo` rejected — must be `rss` / `sense` / `as_is`)
- Out-of-range float (`learning_rate: -0.1`)

Implemented by: `TrainingSettings.from_yaml(path)` in
`src/mriforge/config/settings.py`. The Pydantic engine does the work.

Run it implicitly: any time a YAML is loaded. There's no explicit Tier 0
CLI — if you can `from_yaml` it, it passed Tier 0.

## Tier 1 — Static cross-validation (~50–100 ms)

What it does: **whole-config consistency checks that span multiple
fields and registries.** Operates on the already-parsed
`TrainingSettings` object.

The check list lives in
[`src/mriforge/infrastructure/validation/config_health_checker.py`](https://github.com/adnaneGdihi/mriforge/blob/main/src/mriforge/infrastructure/validation/config_health_checker.py)
— currently 56 checks. A few of the most-fired:

| Check | What it enforces |
|---|---|
| `model_registry` | `model.model_type` is in `MODEL_REGISTRY` |
| `strategy_registry` | `training.strategy_class` is importable |
| `paradigm_required_fields` | Diffusion-paradigm YAMLs declare `training.diffusion` + `num_timesteps`; MRF YAMLs declare the `mrf:` block; etc. |
| `loss_domain_block_match` | Every enabled loss's `@register_loss(domain=...)` matches the block (`image_losses`, `kspace_losses`, …) it appears in |
| `domain_alignment` | RSS-collapsed datasets pair with single-channel models; multi-coil datasets pair with multi-channel models |
| `data_model_compatibility` | `data.patch_size` dimensionality matches `model.spatial_dims` from the registry |
| `hardcoded_cluster_paths` | `data.data_root` doesn't leak another team's cluster prefix (with the [user-mount exemption](https://github.com/adnaneGdihi/mriforge/blob/main/docs/CLUSTER_DATA_LAYOUT.md#how-the-audit-treats-your-cluster-mount)) |
| `early_stopping_metric_compatibility` | `early_stopping.metric` corresponds to a metric that the validation loop actually emits |
| `complex_unet_even_channels` | Backbones with complex-channel layouts get an even `in_channels` |
| `output_dir_convention` | `training.output_dir` follows `experiments/results/<name>` |

Run it explicitly:

```bash
mriforge audit experiments/inprogress/<arm>.yaml
```

Output:

```
✅ [model_registry] model_type='enhanced_deep_unet' registered
✅ [strategy_registry] strategy='mriforge...DiffusionTrainingStrategy' → import-check passed
❌ [hardcoded_cluster_paths] data.data_root='/project/<someone-else>/...' starts with hardcoded cluster prefix '/project/'. ...
   fix: Replace the hardcoded path with a relative path (e.g. 'data/manifests/foo.json')...

[summary] 55/56 checks passed | 1 error | 0 warnings
```

Add `--strict` (the default in the smoke wrapper) to treat warnings as
errors too.

Add `--json` to emit a machine-readable verdict that the smoke wrapper
parses. The JSON includes every check, its severity, and a fix hint —
see `tests_experiments/smoke_test/audit_<timestamp>_*.json` for examples.

## Tier 2 — Synthetic forward probe (~30 s, GPU optional)

What it does: **builds the actual model and runs one synthetic forward
+ backward pass.** Catches everything Tier 1 can't see:

- Shape mismatches between sub-components
- AMP / dtype issues (`fft_ops` autocast handling)
- OOM at the declared `batch_size`
- Non-finite outputs (NaN / Inf in the forward)
- Gradient flow problems (no gradient reaches a parameter)
- Custom adapter chain producing wrong shapes

Run it explicitly:

```bash
mriforge audit experiments/inprogress/<arm>.yaml --probe
mriforge audit experiments/inprogress/<arm>.yaml --probe --save-probe-images out/
```

The probe generates a synthetic batch shaped like the YAML expects
(without touching the real dataset), runs it through the model, and
reports per-component checks. The `--save-probe-images` flag dumps PNG
visualisations of the synthetic input + model output, useful for spot-
checking that the model isn't trivially collapsing.

Tier 2 is **opt-in** for the audit CLI because of the cost. The smoke
wrapper (`scripts/ci/smoke_test_vf_configs.sh`) runs it on every
YAML it discovers, but CI workflows on free-tier runners stop at
Tier 1 — there's no GPU, and the probe is too slow for the budget.

## When each tier runs in CI

| Trigger | Tier | Why |
|---|---|---|
| `pull_request` → `audit-smoke.yml` | Tier 0+1 | Fast (~100ms/yaml), no GPU needed |
| `pull_request` → `test.yml` | Tier 0 (implicit via YAML loading) | Most tests use synthetic configs |
| Manual: `mriforge audit <arm> --probe` | Tier 0+1+2 | One arm, all three tiers |
| Manual: pre-release on Lightning Studio | Tier 2 + GPU tests | See [GPU validation in CONTRIBUTING](https://github.com/adnaneGdihi/mriforge/blob/main/CONTRIBUTING.md#maintainer-release-checklist-gpu-validation) |

## What a failure looks like

The most-common failure classes in production, ranked by surprise factor:

### 1. `hardcoded_cluster_paths`

`data.data_root='/project/<allocation>/<user>/mriforge/databases/...'`

**Cause:** YAML committed with an absolute cluster path. Local
machines don't have the path; cluster machines have it but only after
sync. The cluster's `PROJECT_ROOT` env var triggers an [exemption](../CLUSTER_DATA_LAYOUT.md#how-the-audit-treats-your-cluster-mount)
so this check doesn't false-positive on legitimate cluster runs.

**Fix:** Use `${MRIFORGE_DATA_ROOT}/databases/...` in the YAML and set
`MRIFORGE_DATA_ROOT` per-host. See
[fix_audit_issues.py](https://github.com/adnaneGdihi/mriforge/blob/main/scripts/release/fix_audit_issues.py)
for the bulk-rewrite script.

### 2. `loss_domain_block_match`

`model_type='enhanced_deep_unet' declares output_domain='image' but
losses.output_domain='latent'.`

**Cause:** Two ways for output domain and loss domain to disagree:
either you put a `domain="latent"` loss in `image_losses[]`, or your
model emits image-space but the `losses.output_domain` is `latent`.

**Fix:** Three escape hatches documented in
[add_loss § domain-spillover](../how_to/add_loss.md#domain-spillover-when-the-audit-fights-you).

### 3. `data_model_compatibility`

`Data form {'spatial_dims': 2, 'domain': 'image'} does not satisfy
model capabilities {'spatial_dims': (1,), 'domain': None}`

**Cause:** Image dataset feeding a 1-D fingerprint model (e.g. an
MRF tangent-score net). No registered adapter bridges 2D → 1D, so
the audit refuses to plumb dimensions silently.

**Fix:** Either swap the `dataset_type` to a fingerprint-emitting
dataset, register a new adapter, or change the model's
`@register_model(spatial_dims=...)` decorator if it actually accepts
multiple dimensionalities.

### 4. `paradigm_required_fields`

`training.training_mode='cold_diffusion' requires training.diffusion
to be set`

**Cause:** The diffusion-family paradigms (cold, score, Lévy,
resetting) all need a `training.diffusion` sub-block declaring the
noise schedule and degradation. The audit matches the strategy class
name pattern (anything containing `Diffusion`).

**Fix:** Add the block — see
[write_experiment_yaml § gotcha 1](../how_to/write_experiment_yaml.md#1-diffusion-paradigms-need-trainingdiffusion--num_timesteps).

## Recommendation banners

The audit can also emit non-blocking **recommendations** prefixed with
`💡 AUDIT-R*`. These don't fail the audit but flag issues like:

- `AUDIT-R1: max_iterations and epochs both set` — only one is honoured.
- `AUDIT-R2: model_kwargs.features list doesn't reach patch_size in downsampling depth` — your bottleneck is probably bigger than intended.
- `AUDIT-R3: validation_split=0.1 but train_records < 100` — your val set will have <10 samples.

Strict mode (`--strict`) does not promote recommendations to errors —
they're advisory. The smoke wrapper does run with `--strict`, but only
treats real warnings as failure (not recommendations).

## Implementation notes for contributors

Adding a new audit check:

1. Add a method to `ConfigHealthChecker` in `config_health_checker.py`
   following the existing pattern: returns a `HealthCheckResult` with
   `check_name`, `passed`, `severity`, `message`, `category`,
   `yaml_keys`, `fix_hint`.
2. Wire it into `run_all_checks(self, config)` near the other entries.
3. Add unit tests in
   `tests/unit/infrastructure/validation/test_health_checker_json_and_new_checks.py`
   following the existing pattern.
4. Add a row to the Tier 1 table above.

Audit checks should be **fast** (single-digit milliseconds), **pure**
(no filesystem, no network, no GPU), and **specific** (each check
flags one named problem, with a concrete fix hint).

## Related

- [Write an experiment YAML](../how_to/write_experiment_yaml.md) — the gotcha checklist mirrors the most-fired audit checks.
