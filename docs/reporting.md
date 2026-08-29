# End-of-experiment Reporting Pipeline

Implements the design in `TODO/report_step` — a
defensible, automation-oriented figure / table generator that runs as
the last step of every training experiment.

## What you get out of the box

For every training run with `reporting.enabled: true` in its YAML:

```
<experiment_dir>/report/
├── figures/
│   ├── fig_1_2_learning_curves.pdf            ← Phase-1 Fig 1.2
│   ├── fig_1_3_loss_decomposition.pdf         ← Phase-1 Fig 1.3
│   ├── fig_1_15_computational_profile.pdf     ← Phase-1 Fig 1.15
│   ├── mri_a11_cohort_table.pdf               ← Part-A Fig A.11
│   └── *.meta.json                            ← per-figure provenance
├── tables/
│   ├── tab_2_1_main_results.{md,tex}          ← Phase-2 Tab 2.1
│   └── tab_2_4_dataset_descriptor.{md,tex}    ← Phase-2 Tab 2.4
└── report_summary.md                          ← human-readable index
```

All figures stamp `git=<sha>  ·  seed=<n>  ·  data=<version>` in the
bottom-right corner and write a sidecar `*.meta.json` so that any
reviewer can answer "how was this number computed?" in one lookup.

## Enabling reporting in an experiment

Add a `reporting:` block to any v6.0 YAML — shape mirrors
[`src/config/schemas/reporting.py`](../src/mriforge/config/schemas/reporting.py):

```yaml
reporting:
  enabled: true
  task: reconstruction        # default | reconstruction | synthesis | super_resolution
                              # | gan | diffusion | vae | calibration
  method_name: my_baseline    # label used in tables / figures (defaults to YAML stem)
  out_subdir: report          # subdir of <experiment_dir> to write into

  # Override the task preset (omit / null → use defensible default):
  figures:
    - fig_1_2_learning_curves
    - fig_1_3_loss_decomposition
    - fig_1_5_predicted_vs_true
    - fig_1_4_residual_diagnostics
    - fig_1_11_stratified_performance
    - fig_1_12_failure_gallery
    - fig_1_15_computational_profile
    - mri_a7_kspace_recon
    - mri_a11_cohort_table
  tables:
    - tab_2_1_main_results
    - tab_2_4_dataset_descriptor

  # Metrics included in the main results table (tab_2_1).
  # Names resolve to either the existing mriforge.core.metrics registry OR the
  # Tier-1 IQMs in mriforge.infrastructure.reporting.metrics (vif, fsim, haarpsi, gmsd).
  metrics:
    - psnr
    - ssim
    - ms_ssim
    - lpips
    - vif
    - fsim
    - haarpsi
    - gmsd
    - rmse
    - nmse
    - hfen

  # Optional cohort metadata for the dataset descriptor + cohort plot:
  cohort:
    n_total: 100
    train_n: 60
    val_n: 20
    test_n: 20
    age_mean: 42.5
    age_std: 11.2
    sex_split: 52F/48M
    scanners: Siemens 3T
    field_strength: 3T
    pathology: healthy
    split_rule: subject-level (no slice leakage)

  # Optional hyperparameter table rows:
  hyperparameters:
    - {name: learning_rate, distribution: log-uniform, range: 1e-5..1e-3, final: 1e-4, criterion: best val_psnr}

  # Optional baseline runs to fold into the same tables:
  extra_runs:
    - experiments/outputs/baseline_v1

  fail_on_error: false        # if true, a plotter exception aborts training wrap-up
```

### Schema field reference

| Field | Type | Used by | Notes |
|---|---|---|---|
| `enabled` | bool | pipeline hook | Master switch — `false` makes the hook a no-op |
| `task` | str | preset selector | One of: `default`, `reconstruction`, `synthesis`, `super_resolution`, `gan`, `diffusion`, `vae`, `calibration` |
| `method_name` | str | every plotter / table label | Defaults to experiment dir name |
| `out_subdir` | str | orchestrator | Subdir of `<experiment_dir>` to write into |
| `figures` | list[str] | plotter dispatcher | `null` → use the task preset's defensible defaults |
| `tables` | list[str] | table dispatcher | `null` → use the task preset's defensible defaults |
| `metrics` | list[str] | `tab_2_1_main_results` | `null` → table auto-detects from eval JSON |
| `cohort` | dict | `tab_2_4_dataset_descriptor` + `mri_a11_cohort_table` | Free-form keys (see example) |
| `hyperparameters` | list[dict] | `tab_2_3_hyperparameters` | Each dict: `{name, distribution, range, final, criterion}` |
| `extra_runs` | list[str] | aggregator | Other experiment dirs to include in the same tables (e.g. baselines) |
| `fail_on_error` | bool | hook wrapper | `false` (default) makes plotter failures soft-fail |

The `task` preset selects a defensible default figure / table set per
[`src/infrastructure/reporting/pipeline.py:TASK_PRESETS`](../src/mriforge/infrastructure/reporting/pipeline.py).

## Manually generating a report

```bash
source .venv/bin/activate
python -m mriforge.cli report \
  --exp-dir experiments/outputs/my_baseline \
  --task reconstruction \
  --method my_baseline \
  --cohort-json /tmp/my_cohort.json \
  --seed 42 \
  --dataset-version m4raw_v1
```

Output is identical to the in-pipeline version — same orchestrator.

## Architecture

| Layer | Module | Role |
|---|---|---|
| Style | [`style.py`](../src/mriforge/infrastructure/reporting/style.py), [`styles/ieee.mplstyle`](../src/mriforge/infrastructure/reporting/styles/ieee.mplstyle) | Single source of truth for fonts, colours, sizes (Phase 0) |
| Metadata | [`metadata.py`](../src/mriforge/infrastructure/reporting/metadata.py) | Git hash, seed, dataset stamping; sidecar JSON writer |
| Aggregator | [`aggregator.py`](../src/mriforge/infrastructure/reporting/aggregator.py) | Tidy long-format DataFrame from `training_metrics.csv` + `validation_metrics.csv` + `final_eval.json` (Phase 3) |
| Plotters | `plotters/` | One module per figure type; all share signature `make(df, out_path, **kw)` |
| Tables | `tables/` | Builders that emit Markdown + LaTeX (IEEE) |
| Metrics | `metrics/` | Tier-1 radiologist-correlate IQMs (VIF, FSIM, HaarPSI, GMSD) + radiomics CCC + hallucination index |
| Pipeline | [`pipeline.py`](../src/mriforge/infrastructure/reporting/pipeline.py) | Top-level `generate_report()` orchestrator + task presets |
| CLI | [`cli.py:report`](../src/mriforge/cli/app.py) | `python -m mriforge.cli report ...` |
| Schema | [`schemas/reporting.py`](../src/mriforge/config/schemas/reporting.py) | `ReportingSettings` Pydantic model |
| Hook | [`pipelines/train.py:_maybe_run_reporting`](../src/mriforge/pipelines/train.py) | Called by `run_training_pipeline` after the training loop returns |

## Canonical figure inventory

Implements `TODO/report_step` Phase 1 entries. Plotter IDs (used in
`reporting.figures` YAML) and the questions they answer:

| ID | Figure | Question |
|---|---|---|
| `fig_1_1_headline_pareto` | Headline Pareto | What's the central trade-off? |
| `fig_1_2_learning_curves` | Learning curves with seed bands | Did the model converge? |
| `fig_1_3_loss_decomposition` | Loss-component stacked area | Which term dominates? |
| `fig_1_4_residual_diagnostics` | Residuals 4-panel (vs fitted, Q-Q, hist, ACF) | Is the regression well-specified? |
| `fig_1_5_predicted_vs_true` | Joint scatter + marginals + R² / RMSE | Bias / variance / dynamic range? |
| `fig_1_9_ablation_strip` | ΔMetric horizontal bars w/ 95 % CI | Is each component necessary? |
| `fig_1_11_stratified_performance` | Per-subgroup metric bars (with n) | Where does the model fail? |
| `fig_1_12_failure_gallery` | 3×3 worst-case panels | Qualitative complement to aggregate metrics |
| `fig_1_15_computational_profile` | Runtime + memory bars | Cost profile |

Generative-paradigm-specific (Phase 1 supplements):

| ID | Figure | Question |
|---|---|---|
| `gen_diffusion_diagnostics` | Loss vs t, sample-quality vs NFE, CFG sweep | Does the score converge? |
| `gen_gan_diagnostics` | D-acc → 0.5 trace, G/D loss ratio, ‖∇D‖ | Is the GAN at equilibrium? |
| `gen_vae_diagnostics` | Recon vs KL, per-latent KL collapse | Is the latent posterior healthy? |

MRI-task-specific (Part A):

| ID | Task | Figure |
|---|---|---|
| `mri_a1_sr_triptych` | Super-resolution | LR / pred / HR / residual ×N + radial spectrum |
| `mri_a2_synthesis_c2c` | Contrast-to-contrast | input / synth / target + 2-D joint hist + MI |
| `mri_a7_kspace_recon` | k-space → image | ZF / CG-SENSE / proposed / FS-ref + radial k-error |
| `mri_a11_cohort_table` | All | Split-size bars + cohort summary text |
| `mri_a12_fiducial_check` | Virtual-fiducial paradigm | pred / target / residual ×N with the canonical fiducial ROI outlined in cyan; full-image PSNR vs ROI-PSNR per case |

`mri_a12_fiducial_check` regenerates the canonical
[`VirtualFiducial`](../src/mriforge/infrastructure/physics/virtual_fiducial.py)
grid at `pred.shape[-2:]` and uses it as the ROI mask, so the pipeline
does **not** need to plumb an extra fiducial-mask field through
`cases` — the same `{pred, target, …}` contract used by
`mri_a1_sr_triptych` works. Make sure the YAML's
`physics.virtual_fiducial` (when present) uses the same
`grid_spacing` / `sigma` as the figure's defaults
(`grid_spacing=16`, `sigma=2.0`), otherwise pass overrides via the
plotter `**kwargs` in the dispatch call. Wired into every
`experiments/inprogress/vf/*_v2.yaml` `reporting.figures`
list as of 2026-05-11.

The full Part-A list (A.3 I2I, A.4 N2N, A.5 2D→3D, A.6 LF→HF, A.8
k-space filling, A.9 motion, A.10 deformation) follow the same
plotter contract — extend
`plotters/mri_specific/`
and register under a fresh ID.

## Canonical table inventory

| ID | Table | Output |
|---|---|---|
| `tab_2_1_main_results` | Methods × metrics, mean ± std, best/2nd-best, Holm-corrected p | `.md` + `.tex` |
| `tab_2_2_ablation` | Variant × Δ-metric vs full | `.md` + `.tex` |
| `tab_2_3_hyperparameters` | name / distribution / range / final / criterion | `.md` + `.tex` |
| `tab_2_4_dataset_descriptor` | n / age / sex / scanner / pathology / split rule | `.md` + `.tex` |

## Defensible metrics (Part C of `TODO/report_step`)

The framework's existing `src/core/metrics/` registry
covers PSNR, SSIM, MS-SSIM, NRMSE, NMSE, HFEN, LPIPS, FID, KID, NIQE,
BRISQUE, dice, HD95, IoU, g-factor, k-space error, GMSD, FSIM, VIF, and
more. As of the May-2026 audit it counts **94 registered metrics** —
covering 32+ of the 50 entries in TODO/report_step Part C directly.

### Tier-1 radiologist-correlate IQMs (image-level)

Lives outside the core registry because it operates on numpy/PIL-style
prediction/target arrays at report time:

| Metric | Reference | Source |
|---|---|---|
| **GMSD** | Xue *et al.*, IEEE TIP, 2014 | [`metrics/gmsd.py`](../src/mriforge/infrastructure/reporting/metrics/gmsd.py) |
| **HaarPSI** | Reisenhofer *et al.*, SPIC, 2018 | [`metrics/haarpsi.py`](../src/mriforge/infrastructure/reporting/metrics/haarpsi.py) |
| **VIF** | Sheikh & Bovik, IEEE TIP, 2006 | [`metrics/vif_fsim.py`](../src/mriforge/infrastructure/reporting/metrics/vif_fsim.py) |
| **FSIM** | Zhang *et al.*, IEEE TIP, 2011 | [`metrics/vif_fsim.py`](../src/mriforge/infrastructure/reporting/metrics/vif_fsim.py) |

### Part-C gap fillers (registered)

[`src/core/metrics/report_step_metrics.py`](../src/mriforge/core/metrics/report_step_metrics.py)
adds the 18 metrics from the Part-C table that were previously absent
from the registry. All are decorated with `@register_metric` so they
resolve from any YAML's `reporting.metrics` block:

| TODO # | Registry name (aliases) | Reference |
|---|---|---|
| 12 | `dists` (`DISTS`) | Ding *et al.*, 2020 — DISTS via the existing loss impl |
| 17 | `mutual_information` (`mi`, `MI`) | Pluim *et al.*, IEEE TMI, 2003 |
| 21 | `edge_preservation_index` (`epi`) | Sobel-edge gradient correlation |
| 23 | `radial_k_error` | Mean ‖F·x̂ − F·x‖ binned by radial k |
| 28 | `average_surface_distance` (`asd`) | Symmetric mean surface distance |
| 29 | `target_registration_error` (`tre`) | Mean Euclidean landmark distance |
| 30 | `folding_fraction` | Voxel-wise Pr(det J ≤ 0) |
| 31 | `dvf_mae` | Mean abs error on a DVF |
| 32 | `through_plane_fwhm` | LSF-derived real z-resolution |
| 34 | `bland_altman_bias` (`ba_bias`) | Mean (pred − target) |
| 35 | `limits_of_agreement_upper` / `_lower` | mean ± 1.96 σ |
| 36 | `icc_3_1` (`ICC`) | Shrout & Fleiss, 1979 — ICC(3,1) |
| 37 | `coefficient_of_variation` (`cv`) | σ / |μ| |
| 39 | `detection_sensitivity` / `detection_specificity` | TP/(TP+FN) and TN/(TN+FP) |
| 40 | `auroc` / `auprc` | scikit-learn impls |
| 41 | `expected_calibration_error` (`ece`) | Standard 15-bin ECE |
| 42 | `nll_bits_per_dim` (`bpd`) | NLL/log 2 per element |
| 45 | `wasserstein_1d` (`w1`) | scipy.stats.wasserstein_distance |
| 46 | `sliced_wasserstein` (`sw`) | Random-projection 1-D Wasserstein |
| 47 | `mmd_metric` | Multi-bandwidth Gaussian-kernel MMD² |
| 49 | `residual_whiteness` (`whiteness`) | Spectral flatness of denoising residual |
| 50 | `cohen_kappa` (`kappa`) | Weighted κ for Likert reads |

All 18 verified via forward-pass smoke test (see
`scripts/.../smoke_test_vf_configs.sh -e <exp>` to drive any of them
through a real `tab_2_1_main_results` rendering).

Plus the radiomics-based hallucination audit (Part B):

| Function | Use |
|---|---|
| `concordance_correlation_coefficient(a, b)` | Lin's CCC for per-feature agreement |
| `feature_preservation_profile(real, synth)` | Median CCC, fraction CCC ≥ 0.85, per-IBSI-family medians, Bland-Altman bias per feature |
| `hallucination_index(real, synth, test_retest_std=…)` | Fraction of texture features whose deviation exceeds `k · σ_test-retest` |

Both consume tidy radiomic feature DataFrames (rows = subjects,
columns = IBSI feature names — generate them with the existing
PyRadiomics-backed [`src/core/metrics/radiomic.py`](../src/mriforge/core/metrics/radiomic.py)).

## Adding a new figure type

1. Create `my_figure.py` in `plotters/`
   with signature `make(df, out_path, *, metadata=None, **kwargs) -> Path | None`.
2. Add to the registration block in
   [`plotters/__init__.py`](../src/mriforge/infrastructure/reporting/plotters/__init__.py).
3. Add to the relevant task preset in
   [`pipeline.py:TASK_PRESETS`](../src/mriforge/infrastructure/reporting/pipeline.py)
   (or invoke explicitly via `reporting.figures` in YAML).

## Quality-assurance checklist (TODO/report_step Phase 6)

| Check | Where |
|---|---|
| Reproducibility — `python -m mriforge.cli report -e <dir>` regenerates bit-exact | sidecar `*.meta.json` records git SHA + seed |
| Each figure renders at journal column width (3.6 in, 7.2 in) | enforced via per-plotter `figsize` |
| Color-blind safe palette | Okabe-Ito (8 hues) — see [`style.py:OKABE_ITO`](../src/mriforge/infrastructure/reporting/style.py) |
| Print preview in greyscale | line styles + markers selected per method, not just colour |
| Numerical claims match underlying CSV | tables read directly from `final_eval.json` and aggregator output |
| No figure depends on a deleted log file | aggregator soft-fails on missing artifacts; plotters return `None` |

## Soft-fail philosophy

By default `fail_on_error: false`: a plotter exception logs a warning
and the rest of the report still generates. The end-of-training hook
is also wrapped — a reporting bug **cannot** break a long training
run's wrap-up. Set `fail_on_error: true` (or pass `--fail-on-error`
via CLI in CI) for strict regeneration.

## Bulk-adding the `reporting:` block to many experiments

A helper script in `scripts/add_reporting_blocks.py`
walks `experiments/{inprogress,active,training}/**/*.yaml`, infers the
right task preset, and writes a fully-populated `reporting:` block —
`enabled`, `task`, `method_name` (derived from filename),
`out_subdir`, an explicit `figures` list, `tables` list, `metrics`
list, and commented placeholders for `cohort` / `hyperparameters` /
`extra_runs`. It is **idempotent** — files that already have a
top-level `reporting:` key are skipped untouched.

Add the block by hand to any arm that needs it — it is a single top-level
key, and an arm without one falls back to the defaults described above:

```yaml
reporting:
  enabled: true
  task: reconstruction
  method_name: my_method      # defaults to the config's filename
  out_subdir: report
  figures: [...]
  tables: [...]
  metrics: [...]
```

The maintainers bulk-apply this across their arm corpus with a script that is
not part of this distribution; for a handful of arms the key above is the whole
of what it writes.

Task detection precedence:

1. `metadata.tags.paradigm` (most reliable when present)
2. `model.task`
3. `training.training_mode`
4. `model.model_type` keyword heuristic
5. Filename / path keyword (last resort)
6. fallback → `default`

Override / refine the detector by editing `PARADIGM_TO_TASK` in the
script. After bulk-add, customise high-value experiments by hand:

```yaml
reporting:
  enabled: true
  task: super_resolution     # detector chose this
  method_name: my_baseline   # add explicit label
  cohort:                    # optional — populates dataset descriptor
    train_n: 60
    val_n: 20
    test_n: 20
```

A snapshot of the May 2026 bulk-add across the repo:

| Task preset | Experiments tagged |
|---|---|
| reconstruction | 344 |
| diffusion | 158 |
| gan | 153 |
| super_resolution | 42 |
| vae | 24 |
| synthesis | 12 |
| default | 7 |
| **Total** | **740** |

## Testing

```bash
source .venv/bin/activate
python -c "from mriforge.infrastructure.reporting import generate_report; print('OK')"
python -m mriforge.cli report --help
```

See the smoke-test snippet in this conversation's history for a
self-contained end-to-end example that fabricates a `training_metrics.csv`
and exercises every plotter / table builder.
