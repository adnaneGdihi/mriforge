# Cluster Data Layout Reference

This document describes the data layout on the research cluster for MRI
reconstruction experiments.

## Environment variables — full inventory

Every environment variable the framework reads is listed and documented in
[.env.example](../.env.example) at the repo root. Copy that file to `.env`
(gitignored) and the training scripts will pick it up automatically:

```bash
cp .env.example .env
$EDITOR .env                          # set MRIFORGE_DATA_ROOT, etc.
make env-show                         # inspect what's resolved
```

The Python module [src/mriforge/core/env.py](../src/mriforge/core/env.py) is
the single source of truth — every framework call site that reads an env
var should reference a constant from there, e.g.:

```python
from mriforge.core import env
data_root = env.data_root()                       # Path, falls back to ./databases
if env.suppress_clinical_warning():
    ...
if env.is_distributed():
    ...                                            # torchrun-launched run
```

A regression test (`tests/unit/core/test_env.py::test_env_example_lists_every_constant_name`)
fails CI if `.env.example` ever drifts from the Python module.

Categories (full content in `.env.example`):

| Category | Variables |
|---|---|
| Path resolution | `MRIFORGE_DATA_ROOT`, `PROJECT_ROOT`, `MRIFORGE_CLUSTER_ROOT`, `MRIFORGE_CACHE_ROOT`, `MRIFORGE_LEGACY_ABS_PREFIXES`, `MRIFORGE_LEGACY_CLUSTER_PREFIX`, `FASTMRI_DATASETS_ROOT` |
| Device / determinism | `MRIFORGE_SUPPRESS_CLINICAL_WARNING`, `FORCE_CPU`, `MRIFORGE_DEVICE`, `MRIFORGE_NO_GPU_PROBE`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG` |
| CUDA / PyTorch tuning | `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_CACHE_MAXSIZE`, `CUDA_CACHE_CONFIG`, `TORCH_HOME` |
| Threading | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` |
| Distributed (set by torchrun) | `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` |
| CLI rendering | `FORCE_COLOR`, `NO_COLOR` |
| Filesystem | `TMPDIR`, `XDG_CACHE_HOME` |

### Scripts that auto-source `.env`

- [Makefile](../Makefile) — every target (`make train`, `make predict`, `make env-show`)

The container entrypoint, the smoke-test wrapper, the multi-node launcher and the
SLURM job template do the same, but they belong to the internal deployment tree
and are not published with this release. If you write your own launcher, source
`.env` before invoking any `mriforge` verb -- that is the whole contract.

## Resolving paths via `MRIFORGE_DATA_ROOT`

MRIForge does not ship with data. Every shipped YAML uses the placeholder
`${MRIFORGE_DATA_ROOT}` for paths outside the package source tree; the
Pydantic-v2 loader expands the variable before validating the path.

```bash
# Local development
export MRIFORGE_DATA_ROOT="$HOME/mriforge/databases"

# HPC / cluster
export MRIFORGE_DATA_ROOT="/project/<your-account>/mriforge/databases"
```

If unset, the framework falls back to `./databases` relative to the current
working directory (see `mriforge/bootstrap.py`). The path-resolution layer in
`mriforge.data.metadata.path_resolver` honours `PROJECT_ROOT` and
`MRIFORGE_LEGACY_ABS_PREFIXES` for finer-grained control. The
`tests/unit/test_no_cluster_paths.py` regression test fails CI if a literal
`/project/<user>/...` path ever leaks back into the shipped tree.

### How the audit treats your cluster mount

The `hardcoded_cluster_paths` health check rejects YAMLs that contain
*another* team's cluster prefix (a literal `/project/<allocation>/...` or
`/scratch/<allocation>/...` copied from an old config). It explicitly
**exempts** paths that live under the value of your own
`MRIFORGE_DATA_ROOT` or `PROJECT_ROOT` — that's where your data is supposed
to be, not a leak.

```bash
# This combination passes the audit:
export MRIFORGE_DATA_ROOT=/project/<me>/mriforge
# YAML can contain: data_root: /project/<me>/mriforge/databases/fastmri  ✅

# This still fails the audit:
export MRIFORGE_DATA_ROOT=/project/<me>/mriforge
# YAML containing: data_root: /project/<someone-else>/mriforge/...      ❌
```

The exemption uses a trailing-slash prefix match (so
`/project/<allocation>/me` will not silently exempt
`/project/<allocation>/meadow/...`).

## Data Root

All datasets are located under `databases/` on the cluster. The structure is
catalogued in `d_tree.json` (658K lines).

---

## Dataset Inventory

### 1. FastMRI Datasets (K-Space)

**Use for:** Physics-constrained reconstruction, k-space experiments

| Dataset | Path | Format | Size (files) | Resolution |
|---------|------|--------|--------------|------------|
| **Knee Single-Coil** | `databases/fastmri/datasets/knee_singlecoil_train/singlecoil_train/` | HDF5 | ~973 | 320×320 |
| **Brain Multi-Coil Train** | `databases/brain/fastmri/brain_multicoil_train_batch_{0..7}.tar.xz` | **`.tar.xz` — NOT EXTRACTED** | 8 archives | 320×320 |
| **Brain Multi-Coil Val/Test** | — | **ABSENT** | 0 | — |
| **Prostate Diffusion** | `databases/fastmri/datasets/prostate_diffusion/fastMRI_prostate_DIFF_IDS_001_011/` | HDF5 | ~100+ | 200×150 |
| **Brain DICOM** | `databases/fastmri/datasets/brain_dicom/fastMRI_brain_DICOM/` | DICOM | varies | varies |

> ⚠️ **fastMRI BRAIN IS NOT EXTRACTED, and there is no val/test split.**
>
> This table used to advertise `databases/fastmri/datasets/brain_multicoil_{train,val,test}/`
> with "~5000+" and "~1000+" HDF5 files. **Those paths do not exist.** Per `data_tree.json`
> (a `tree -J` of the cluster `databases/`, 4,649 dirs / 141,557 files), the fastMRI brain
> data is **8 unextracted `.tar.xz` archives, train split only**. The knee data *is*
> extracted, which is probably how the assumption slipped in.
>
> Anyone planning fastMRI brain work off the old table budgeted **zero** time for data prep.
> The real prerequisite chain is:
>
> ```bash
> # 1. extract (train split only -- there is no val/test)
> tar -xJf databases/brain/fastmri/brain_multicoil_train_batch_*.tar.xz -C <brain-h5-dir>
> # 2. fastMRI+ lesion annotations, fetched from the fastMRI+ release and
> #    pinned by checksum, into databases/fastmri_plus/brain.csv
> # 3. manifest + human QC of the y-origin overlays
> ```
>
> `scripts/data/download_fastmri_datasets.py` **cannot re-fetch**: its presigned S3 URLs
> carry `Expires=1775455808` → **2026-04-06**, already past.

**HDF5 Structure:**
```
file.h5
├── kspace       # Complex k-space data [slices, coils, H, W]
├── reconstruction_rss  # RSS reference image [slices, H, W]
└── attrs        # Metadata (acquisition, padding, etc.)
```

---

### 2. M4Raw Datasets (Low-Field K-Space)

**Use for:** Low-field MRI reconstruction, motion correction

| Dataset | Path | Format | Resolution |
|---------|------|--------|------------|
| **Multi-Coil Train** | `databases/m4raw/data/multicoil_train/multicoil_train/` | HDF5 | 256×256 |
| **Multi-Coil Val** | `databases/m4raw/data/multicoil_val/multicoil_val/` | HDF5 | 256×256 |
| **Multi-Coil Test** | `databases/m4raw/data/multicoil_test/multicoil_test/` | HDF5 | 256×256 |

---

### 3. ULF Paired Datasets (Ultra-Low-Field)

**Use for:** Field translation, super-resolution, paired training

| Dataset | Path | Format | Contrasts |
|---------|------|--------|-----------|
| **64mT-3T Paired** | `databases/ulf_paired/ulf_paired_64mt_3t/` | NIfTI | T1w, T2w, FLAIR, ADC, DWI |
| **ULF for LDM** | `databases/processed/ulf_to_hf_ldm/ulf/` | NIfTI | Various |

**Directory Structure:**
```
ulf_paired_64mt_3t/
├── Data/
│   ├── 64mT_data/           # Ultra-low-field source
│   │   └── sub-XXX/
│   │       └── ses-0X/
│   │           └── anat/T1w.nii.gz
│   └── 3T_data/             # High-field target
│       └── sub-XXX/
│           └── anat/T1w.nii.gz
```

---

### 4. BraTS Datasets (Image Space)

**Use for:** Super-resolution, tumor segmentation

| Dataset | Path | Format | Resolution |
|---------|------|--------|------------|
| **BraTS-SR LR** | `databases/brats_sr/A_LRSI/` | PNG | 2D slices |
| **BraTS-SR HR** | `databases/brats_sr/A_HRSI/` | PNG | 2D slices |
| **Lesion Maps** | `databases/brats_sr/Lesion_map/` | PNG | 2D slices |

---

### 5. Consistency Phantom

**Use for:** Stability analysis, reproducibility

| Dataset | Path | Description |
|---------|------|-------------|
| **3T Phantom** | `databases/consistency/3T_data_PROMRI/` | DICOM series |
| **VLF 10-Day** | `databases/consistency/VeryLowField_Phantom_data_10days/` | Multi-day acquisition |

---

## Preprocessing Outputs

After running `scripts/preprocessing/preprocessing.py`, each dataset gets a `_image/` sibling directory:

```
{dataset_name}/
└── {dataset_name}_image/
    ├── nifti_reconstructed/   # K-space → NIfTI (RSS)
    ├── compressed_kspace/     # Coil-compressed k-space
    ├── coil_sensitivity/      # ESPIRiT maps
    ├── gt_images/             # Ground truth images
    ├── registered/            # Co-registered volumes
    ├── isotropic/             # 1mm isotropic resampled
    ├── kspace_generated/      # NIfTI → k-space (synthetic)
    ├── statistics/            # Normalization stats
    └── manifests/             # Parquet metadata
```

---

## Recommended layout — raw vs processed, internal vs external (2026-06-07)

The compilation in `TODO/scientific_validation/DATASETS_compilation.md` adds ~40
candidate datasets. Organize them by **mutability + cost**, not by topic — so the
expensive-to-refetch raw data is safe, the regenerable stuff is disposable, and
configs never break when you re-organize.

```
$MRIFORGE_DATA_ROOT/databases/
├── <family>/                       # curated families (fastmri, m4raw, ulf_paired, …)
│   ├── …/<split>/                  #   RAW k-space/NIfTI (immutable; never auto-deleted)
│   ├── m4raw/gre/                  #   ← the unused GRE set; manifest m4raw_gre.json
│   └── <name>_image/               #   PROCESSED (regenerable _image sibling)
│
├── external/                       # one dir per dataset id (== manifest id == downloader dest)
│   ├── _DOWNLOAD_LEDGER.json        #   internal/external + status + small_sample (downloader)
│   └── <id>/                        #   e.g. oracle_bssfp/, osi2one_b0/, kasper_7t_spiral_fmri/
│       ├── raw/                     #     immutable as-downloaded (h5/mrd/nifti)  ← data_root
│       ├── processed/              #     regenerable artifacts (optional)
│       └── SOURCE.json             #     provenance: DOI, access, field_T, needs, n_samples
│
└── _cache  ->  $MRIFORGE_CACHE_ROOT  # symlink to SCRATCH (disposable intermediates)
```

**The three tiers (the load-bearing idea):**

| Tier | What | Where | Touched by `--clean-artifacts`? |
|------|------|-------|---------------------------------|
| **raw** | as-downloaded k-space / images | `$PROJECT` (quota'd, backed up) | **never** |
| **processed** | `_image/` siblings, coil maps, isotropic, kspace_generated | `$SCRATCH` (or project) | **yes** (regenerable) |
| **cache** | TorchIO queues, intermediate tensors | `$SCRATCH` via `_cache` symlink | **yes** |

So a full disk is recoverable: wipe `processed`/`cache`, keep `raw` + manifests.

**Rules that keep configs stable:**

1. **Manifests are the SSOT for *selection*, not directories.** Field strength,
   role (RAW/MAP), and which arms a dataset serves live in the **manifest**
   (`data/manifests/external/<id>.json` → `source` block), never in the path.
   Re-organizing `databases/` then only needs the manifest `data_root` updated,
   not every YAML. Select a dataset with `data.index_path:
   data/manifests/external/<id>.json`.
2. **One id, one place.** `databases/external/<id>/raw` ↔
   `data/manifests/external/<id>.json` ↔ ledger row `<id>`. The downloader writes
   exactly there.
3. **raw is immutable + provenanced.** Every external `<id>/` carries a
   `SOURCE.json` (DOI, access, license, field, n_samples). Never edit files under
   `raw/` in place — derive into `processed/`.
4. **Disk: big-immutable on `$PROJECT`, regenerable on `$SCRATCH`.** Point
   `MRIFORGE_CACHE_ROOT`/`TMPDIR` at scratch; symlink `databases/_cache` there. The
   ledger's `failed_diskspace` / `manual_required` rows tell you what's still
   **external** (not on disk) and why.
5. **gitignored by design.** Both `databases/` and `data/manifests/` are
   gitignored — they are *generated*. The tracked SSOT is
   `TODO/scientific_validation/datasets.json` + the generator
   (`scripts/data/gen_external_dataset_manifests.py`); regenerate manifests from
   it, never hand-edit the stubs.

**Bring-up order on a fresh cluster account:**

```bash
export MRIFORGE_DATA_ROOT=/project/<you>/mriforge/databases
export MRIFORGE_CACHE_ROOT=/scratch/<you>/mriforge_cache
# Obtain each dataset from its own source, unpack under $MRIFORGE_DATA_ROOT/external/<id>/raw,
# then build the manifests:
python scripts/data/regenerate_cluster_manifests.py --data-base "$MRIFORGE_DATA_ROOT"
# point an experiment:  data.index_path: data/manifests/external/fastmri_brain.json
```

## Experiment Domain Mapping

| Experiment Type | Required Domain | Recommended Dataset |
|-----------------|-----------------|---------------------|
| **K-Space Diffusion** | kspace | FastMRI Knee/Brain |
| **Physics-Constrained** | kspace | FastMRI Knee |
| **Super-Resolution** | image | BraTS-SR, ULF Paired |
| **Field Translation** | image (paired) | ULF Paired |
| **Motion Correction** | kspace | M4Raw |
| **Reconstruction** | kspace | FastMRI, M4Raw |

---

## Config Path Mapping

For experiments, update `data.datasets[].path` to point to cluster paths:

```yaml
# K-space experiments
data:
  dataset_type: kspace
  datasets:
    - name: fastmri_knee
      path: databases/fastmri/datasets/knee_singlecoil_train/singlecoil_train

# Image-space experiments
data:
  dataset_type: paired_slices
  datasets:
    - name: brats_sr
      path: databases/brats_sr
```

---

## Canonical Shapes

| Dataset | K-Space Shape | Image Shape |
|---------|--------------|-------------|
| FastMRI Knee | (slices, 1, 640, 368/372) | (slices, 320, 320) |
| FastMRI Brain | (slices, 16-20, 320, 320) | (slices, 320, 320) |
| Prostate | (50, 32, 200, 150) | (50, 200, 150) |
| M4Raw | (slices, coils, 256, 256) | (slices, 256, 256) |
| ULF | N/A | (256, 256, 256) |

---

## Multi-Contrast Configuration (ULF Paired)

The ULF paired dataset supports **5 contrasts**: T1w, T2w, FLAIR, ADC, DWI.

### Config Structure for Multi-Contrast Experiments

```yaml
data:
  dataset_type: 3d_volumetric
  known_dataset: ulf_paired_brain

  # Specify which contrasts to use
  contrasts:
    - T1w
    - T2w
    - FLAIR

  primary_contrast: T1w  # Main contrast for single-contrast training

  # Source (ULF 64mT) and Target (HF 3T) directories
  input_lr_dir: databases/ulf_paired/ulf_paired_64mt_3t/Data/64mT_data
  input_hr_dir: databases/ulf_paired/ulf_paired_64mt_3t/Data/3T_data

  # Strategy for pairing
  strategy: conditional  # Pairs ULF→HF by matching subject+contrast

  preprocessing:
    registration:
      enabled: true  # Must be true for cross-field pairing
      method: rigid  # rigid for cross-field, SyN for intra-field
```

### Contrast File Templates

ULF and HF volumes follow BIDS-like naming:

```
ULF (64mT):
  sub-XXX/ses-0Y/anat/{subject}_ses-{session}_run-1_{contrast}.nii.gz

HF (3T):
  sub-XXX/anat/{subject}_acq-highres_{contrast}.nii.gz
```

### Model Configuration for Multi-Contrast

```yaml
model:
  model_kwargs:
    use_contrast_guidance: true   # Enable contrast embedding
    num_contrasts: 3              # Number of contrast types
    contrast_embed_dim: 128       # Embedding dimension
```

### Experiment Types by Contrast Mode

| Mode | Use Case | Config |
|------|----------|--------|
| **Single-contrast** | Train on T1w only | `primary_contrast: T1w` |
| **Multi-contrast** | Joint all contrasts | `contrasts: [T1w, T2w, FLAIR]` |
| **Contrast translation** | T2w→T1w synthesis | `source_contrast: T2w, target_contrast: T1w` |

---

## Notes

1. **Local dummy files**: The local `databases/` contains only 51KB placeholder files for testing. Real data is only on cluster.
2. **d_tree.json**: Generated by indexing cluster storage, 658K lines describing full file tree.
3. **Manifests**: Parquet files with per-file metadata (shapes, statistics, splits).
