---
orphan: true
---

# Workflow backlog — registration · wiring · usage status

Living tracker for the `workflow:` contract. It records, per regime and per
component, three distinct states so a "declared LIVE" claim can never hide a
gap:

- **Registered** — the component exists in its registry (decorator fired /
  added to `OperatorRegistry` / `STRATEGY_CLASS_PATHS`).
- **Wired** — reachable through the normal execution path (operator used by a
  strategy; loss selectable via a `losses.*` block; strategy dispatchable via
  `TrainingStrategyFactory`; metric selectable via `metrics`; data block
  declared on the schema).
- **Used** — an actual config/arm or an end-to-end test exercises it.

Legend: ✅ done · 🟡 partial · ⬜ missing · — n/a. "pitfall N" and
"non-negotiable N" cite the numbered rules in the repository's internal
`CLAUDE.md`, which is not distributed; "internal issue N" cites the private
research tracker. Neither is written `#N`, because GitHub would autolink that
to an unrelated pull request of whichever repository you are reading this in.
The maturity-ledger test
(`tests/unit/domain/workflows/test_maturity_ledger.py`) enforces the maturity
column against the live registries — **including PARTIAL**, as of 2026-07-16.

## Maturity status

**All nine MR regimes with real physics are LIVE as of 2026-07-16. `EVAL_ONLY` is now empty.**

| Regime | Current | Backed by (forward model · strategy · metrics · losses) |
|---|---|---|
| `mri_structural` | **LIVE** | `fft2d` · Reconstruction · `cjv`/`wm2max` · *agnostic, deliberately* |
| `mri_flow` | **LIVE** | `phase_contrast` · PhaseContrastFlow · `vnr`/`divergence`/… · 3 flow losses |
| `mri_perfusion` | **LIVE** | **`extended_tofts` (signal model)** · PerfusionKinetic · `ktrans`/`bat`/… · Tofts/box/smoothness |
| `mri_quantitative` | **LIVE** | `fft2d` · MultiEchoB0Fit, OneShotMultiParameter · `b0_field_rmse`/`geodesic_qmap_error`/`cross_scanner_t1t2_concordance` · `bloch_signal_synthesis_consistency`, `contrast_consistency` |
| `mri_fingerprinting` | **LIVE** | `fft2d` · ConformalMRFDictless · `geodesic_mrf_parameter_error` · **`mrf_dictionary_match`** |
| `mri_diffusion_weighted` | **LIVE** | `fft2d` **+ `adc_monoexp`** · QSpaceDiffusion, BeltramiEPI · `adc_mae` · `dwi_adc_monoexp`, `beltrami_epi_residual` |
| `mri_dynamic` | **LIVE** | `fft2d` · LowRankSparse · **`temporal_fidelity`** · `storm` |
| `mri_functional` | **LIVE** *(recon-scoped)* | `fft2d` · BeltramiEPI · **`tsnr`**, **`temporal_fidelity`** · `beltrami_epi_residual` |
| `mri_spectroscopy` | **LIVE** | **`mrs_lorentzian` (signal model)** · MRSQuantification · `spectral_linewidth`/`crlb`/`freq_domain_snr` · `mrs_fid_residual`, `mrs_prior_knowledge` |
| `nmr_spectroscopy` | STUB | — |
| `ct` / `xray` / `ultrasound` / `optical` | STUB | typed seam only |

> **`mri_functional`'s LIVE is scoped by its own `supported_tasks`**, which are
> the RECON family and nothing else. It claims "the framework can reconstruct
> and denoise a functional series" — real EPI readout physics, graded by tSNR and
> temporal fidelity. It is **not** a claim that BOLD modelling exists: there is no
> GLM task, no BOLD task member, and `BeltramiEPIDistortionStrategy` models the
> READOUT, not neural activity. `hrf_likelihood` is real BOLD physics and stays
> deliberately untagged for exactly this reason. This is why `supported_tasks` is
> a load-bearing fact rather than decoration — it is what bounds the claim.

## The LIVE rule (rewritten 2026-07-16)

```
LIVE ⟺ a registered forward model — a linear operator OR a signal model
      ∧ a regime-tagged strategy
      ∧ regime-tagged metrics
```

Together: the regime's physics is **expressible**, **trainable**, and
**gradeable on its own terms**.

**Why the forward-model clause was widened.** It used to read
`forward_operator is not None`, and it failed in both directions at once. Too
weak for most regimes — *every* MR profile declares `fft2d` simply because MR
k-space is FFT-reconstructed, so the clause tested "is this MR?" rather than "is
this regime's physics wired?". And impossible for `mri_perfusion`, which has the
richest regime-specific physics in the repo: the tracer-kinetic map is nonlinear
with no adjoint, so the only way to satisfy the clause was to register a fake
`adjoint` that `DataConsistencyLayer` / `FISTAMBIRSolver` / `NullSpaceProjection`
would consume assuming `⟨Ax,y⟩ = ⟨x,A†y⟩`. **The rule actively rewarded the
facade it existed to prevent** (pitfall 16).

`SignalModelRegistry` (`infrastructure/physics/signal_models/registry.py`) is the
nonlinear half. Its contract has **no adjoint in it at all**, so nothing in it
can be mistaken for a linear operator. A profile picks a field by whether an
adjoint exists, not by taste — DWI honestly declares **both** (`fft2d` for the
readout, `adc_monoexp` for the decay).

**Why LIVE requires metrics but NOT losses.** The asymmetry is real. A regime can
legitimately train on agnostic objectives: `mri_structural` trains on L1/SSIM and
that is *correct*, not a gap. But a regime cannot be *graded* generically once its
output stops being an image — PSNR on a Ktrans map is pitfall 18. Requiring a
regime-tagged loss would force mis-tagging L1 as "structural-only", which is
false, and mass-tagging generic components is precisely how an allow-list rots
(the MRO tag leak, one layer up). Regimes that *do* have distinctive physics are
held to having a tagged loss by a separate test —
`test_regimes_with_distinctive_physics_have_tagged_losses` — with structural as
the single, argued exemption.

**The registry has a production reader, not just the ledger.**
`PerfusionKineticMappingStrategy` resolves `data.perfusion.kinetic_model` through
`get_signal_model` and checks the resolved spec's `parameters` contract before
calling it. That makes the key a live dispatch knob rather than documentation
(pitfall 15), and means the ledger's claim and the runtime dispatch resolve the *same
name from the same registry* rather than agreeing by coincidence.
`SignalModelSpec.parameters` is load-bearing: `gamma_variate` is PERFUSION physics
too, but its signature is `(t_s, amplitude, t0_s, alpha, beta_s)`, so a regime-only
check would bind `ktrans → amplitude` and return a plausible curve.

## Contract infrastructure (Track A)

| Piece | Registered | Wired | Used | Notes |
|---|---|---|---|---|
| `workflow:` schema + field | ✅ | ✅ | ✅ | optional on `TrainingSettings`; fixtures carry it |
| `check_workflow_declared` | ✅ | ✅ | ✅ | Tier-1 audit; missing/STUB/bad-task |
| `check_workflow_required_axes` | ✅ | ✅ | ✅ | pitfall 19; keyed off `dataset_type` |
| `check_workflow_spatial_rank` | ✅ | ✅ | ✅ | rank vs `dataset_type` |
| `check_knob_applicability` | ✅ | ✅ | 🟡 | fires for `mrf`/`data.quantitative`/`data.phase_contrast`/`data.perfusion`; no arm exercises it yet |
| `check_workflow_signal_domain` | ✅ | ✅ | 🟡 | `profile.signal_domains` vs the model's `input_domain`; 0/121 false positives, and its first finding is below |
| runtime maturity gate | ✅ | ✅ | 🟡 | wired into `train.py`/`infer.py`; no STUB/EVAL_ONLY arm in-repo to trip it |
| `ArtifactOrigin` on degradations | ✅ | ✅ | 🟡 | all 28 classified; no caller filters by it yet |
| component tagging seam | ✅ | ✅ | ✅ | all 5 registries; 9 strategies + 13 losses + 21 metrics tagged |
| ledger asserts **PARTIAL** | ✅ | ✅ | ✅ | two tiers; the `if/elif` fall-through is closed |
| `SignalModelRegistry` | ✅ | ✅ | ✅ | LIVE's widened clause; production reader = the perfusion strategy |
| tag read from own `__dict__` | ✅ | ✅ | ✅ | the MRO leak (64 reported / 1 declared) is fixed |

## Per-regime component status

### `mri_flow` (LIVE)

| Component | Registered | Wired | Used | Gap |
|---|---|---|---|---|
| physics `flow_encoding` | ✅ | ✅ | ✅ | analytic round-trip test |
| operator `phase_contrast` | ✅ | ✅ | ✅ | consumed by the strategy's encode/decode adapters |
| loss `phase_contrast_velocity` | ✅ | ✅ | ✅ | `losses.physics.lambda_phase_contrast_velocity` |
| loss `through_plane_flux_conservation` | ✅ | ✅ | ✅ | `voxel_area` threaded from `data.phase_contrast` (the loss has no `__init__`) |
| loss `velocity_unwrap_consistency` | ✅ | ✅ | ✅ | " |
| metrics `vnr`/`divergence`/`mass_conservation` | ✅ | 🟡 | ✅ | tagged FLOW; no `metric_map` entry, but reachable via `validation.metrics: [name]` |
| metrics `velocity_rmse`/`peak_velocity_error`/`net_flow_error` | ✅ | 🟡 | ✅ | tagged FLOW; no `compute_*` flag, same name-list route |
| strategy `PhaseContrastFlowStrategy` | ✅ | ✅ | ✅ | factory `phase_contrast_flow`; unit-tested, recovers a known field |
| `data.phase_contrast` block | ✅ | ✅ | 🟡 | venc/scheme/flux; no arm — 4D-flow data IS on the cluster (`osu_4dflow_cine`, `whole_heart_5d`), but no `dataset_type` exposes `VELOCITY_ENCODING` from it |

### `mri_perfusion` (LIVE — on a signal model, with no forward operator)

| Component | Registered | Wired | Used | Gap |
|---|---|---|---|---|
| physics `perfusion_kinetics` | ✅ | ✅ | ✅ | vectorised 2026-07-16; equivalence-tested vs the old loop |
| loss `tofts_residual` | ✅ | ✅ | ✅ | the primary, **self-supervised** objective |
| loss `perfusion_physiological_box` | ✅ | ✅ | ✅ | **load-bearing** — see the vp note below |
| loss `perfusion_map_smoothness` | ✅ | ✅ | ✅ | `losses.physics.lambda_perfusion_map_smoothness` |
| loss `aif_consistency` | ✅ | ⬜ | ⬜ | **honestly inert**: needs a `generator.predict_aif` head no model implements. The strategy RAISES if it is weighted, and `data.perfusion` offers no `aif_source: learned`, rather than advertising an unread knob (pitfall 15) |
| metrics `ktrans`/`wash_slope`/`bat`/`negative_voxels` | ✅ | 🟡 | ✅ | tagged; no `metric_map` entry, reachable via `validation.metrics: [name]` |
| metrics `cbf_rmse`/`att_mae` | ✅ | 🟡 | ✅ | tagged; no `compute_*` flag, same name-list route |
| strategy `PerfusionKineticMappingStrategy` | ✅ | ✅ | ✅ | factory `perfusion_kinetic`; recovers known kinetics |
| `data.perfusion` block | ✅ | ✅ | 🟡 | time axis/AIF/model; no arm (no DCE data) |
| signal model `extended_tofts` | ✅ | ✅ | ✅ | **the LIVE clause**; resolved by the strategy from `data.perfusion.kinetic_model`, parameter contract checked |
| signal model `gamma_variate` | ✅ | ⬜ | ⬜ | registered (DSC first-pass); not in the `kinetic_model` Literal because no loss consumes it, and its `parameters` differ from Tofts' so the strategy would reject it |
| forward operator | — | — | — | **n/a, and that is the correct answer, not a gap**: nonlinear, no adjoint. Pinned by a test so nobody "completes" it by inventing one |

> **The extended-Tofts inverse problem is ill-posed for `vp`.** `vp` scales the
> AIF directly while `Ktrans/ve` scale its convolution, so they trade off:
> fitting the residual alone drives `vp` **negative** while the residual sits at
> ~1e-4 and looks like a triumph. A constraint fixes it at an *unchanged*
> residual (`Ktrans` 0.227 → 0.211 vs true 0.20) — it resolves an ambiguity the
> data cannot, rather than fighting the data.
>
> **Which** constraint depends on `training.perfusion_kinetic.parameter_activation`,
> and this is easy to get backwards. Under the default `softplus`, `vp > 0` is
> forced *structurally*, so the box loss's `clamp(-vp)` term is identically zero
> with zero gradient — softplus does this job, and the box loss's live
> contribution is only `ve+vp <= 1`. Under `none`, it is the box loss or nothing.
> Do not read a healthy box loss as evidence the non-negativity constraint is
> being learned. Pinned by `test_box_loss_rescues_the_degenerate_vp`.

### The five regimes that reached LIVE on 2026-07-16

All five had a real, dispatchable strategy the whole time. Three things hid it:
the ledger never asserted PARTIAL at all, the capability tag leaked down the MRO
(64 strategies reported `mri_structural`, one declared it), and **no metric was
tagged for any of them**, so nothing could grade them.

| Regime | Forward model | Loss (registered · wired) | Metric | Notes |
|---|---|---|---|---|
| `mri_quantitative` | `fft2d` | `bloch_signal_synthesis_consistency` ✅✅, `contrast_consistency` ✅✅ | `b0_field_rmse`, `geodesic_qmap_error`, `cross_scanner_t1t2_concordance` | both losses drive predicted maps through real relaxometry (SPGR / Bottomley prior) and raise on missing acquisition params |
| `mri_fingerprinting` | `fft2d` | **`mrf_dictionary_match`** ✅✅ (NEW) | `geodesic_mrf_parameter_error` | the regime had NO honest loss; see below |
| `mri_diffusion_weighted` | `fft2d` **+ `adc_monoexp`** | `dwi_adc_monoexp` ✅✅, `beltrami_epi_residual` ✅✅ | `adc_mae` | the only regime declaring both kinds of forward model — FFT readout AND nonlinear decay |
| `mri_dynamic` | `fft2d` | `storm` ✅✅ | **`temporal_fidelity`** (NEW) | `storm`'s Laplacian was unbounded below; see below |
| `mri_functional` | `fft2d` | `beltrami_epi_residual` ✅✅ | **`tsnr`**, **`temporal_fidelity`** (NEW) | recon-scoped — see the note under Maturity status |

`fit_adc_loglinear` moved from `models/losses/dwi_adc_monoexp_loss.py` to
`infrastructure/physics/signal_models/diffusion_models.py` (canonical homes, non-negotiable 6),
alongside the new `adc_monoexp_forward`. The loss re-exports it.

#### Why `mrf_dictionary_match` had to be written

`mri_fingerprinting` had **no honest loss to tag**. Every MRF-*named* loss in the
repo is a themed name over generic maths:

- `tropical_semiring_consistency` (aliases `tropical_mrf_consistency`,
  `tropical_quantitative_consistency`) builds its reference tropical fan with
  `torch.randn(..., generator=seeded)` — a **random** reference. It is a generic
  max-plus penalty on any `[B, C, H, W]`.
- `conformality_jacobian` ("MRF §3") is differential geometry on an arbitrary
  Jacobian — no Bloch, no dictionary, no fingerprint.

Tagging either would have made the ledger machine-*verify* a fingerprinting claim
with no fingerprinting physics behind it. The new loss does MRF's defining
operation (Ma et al., *Nature* 2013): normalised inner-product matching against a
Bloch dictionary, softmax-relaxed so it is differentiable (hard `argmax` is not).
It is **self-supervised** — the dictionary is the supervision, so no ground-truth
`(T1, T2)` maps are needed, exactly as `tofts_residual` does for perfusion. Its
scope is stated in the docstring: it is only as good as the dictionary handed to
it, and `BlochDictionary` is a *steady-state* (Ernst) approximation.

#### Why `temporal_fidelity` had to be written

`mri_dynamic` and `mri_functional` had **zero** metrics — and per-frame PSNR/SSIM
structurally cannot grade them. A reconstruction that emits the temporal **mean**
at every frame scores excellent per-frame PSNR (each frame is a perfectly good
image of the average) while having destroyed every dynamic in the acquisition.
Per-frame metrics reduce over exactly the axis the claim lives on. Under
`temporal_fidelity` that reconstruction correlates with nothing and scores 0.

`tsnr` is the standard fMRI IQM (Triantafyllou 2005; MRIQC) and closes a schema
flag (`compute_tsnr`) that had **no implementation at all**. Read it with its
documented caveat: **tSNR rewards temporal smoothing**. A 5-tap moving average on
fast dynamics drops fidelity from 0.82 to **−0.45** (anticorrelated!) while tSNR
*triples*, 11.6 → 39.8 — pinned by
`test_tsnr_rewards_temporal_smoothing_the_documented_caveat`. It measures noise,
not fidelity, and cannot tell them apart; the pairing with `temporal_fidelity` is
the point.

### `mri_spectroscopy` (LIVE — the last EVAL_ONLY regime, completed)

| Component | Registered | Wired | Used | Gap |
|---|---|---|---|---|
| physics `spectroscopy` (Lorentzian FID) | ✅ | ✅ | ✅ | analytic; absorption/magnitude lineshapes checked against closed forms |
| signal model `mrs_lorentzian` | ✅ | ✅ | ✅ | **the LIVE clause**; resolved by the strategy from `data.spectroscopy.signal_model`, parameter contract checked |
| loss `mrs_fid_residual` | ✅ | ✅ | ✅ | the primary, **self-supervised** objective; recovers known resonances |
| loss `mrs_prior_knowledge` | ✅ | ✅ | ✅ | AMARES-style hinges; **the amplitude term is DEAD under softplus** — see below |
| metrics `spectral_linewidth`/`crlb`/`freq_domain_snr` | ✅ | 🟡 | ✅ | tagged SPECTROSCOPIC; reachable via `validation.metrics: [name]` |
| strategy `MRSQuantificationStrategy` | ✅ | ✅ | ✅ | factory `mrs_quantification`; per-parameter activation |
| `data.spectroscopy` block | ✅ | ✅ | 🟡 | FID length/dwell/resonances; no arm (no MRS data) |
| forward operator | — | — | — | **n/a, and correct**: the Lorentzian sum is nonlinear in frequency, linewidth and phase, with no adjoint. Pinned by a test |
| metabolite-basis (LCModel) model | ⬜ | ⬜ | ⬜ | needs a basis set the repo does not ship. `signal_model` is a closed Literal with one member for that reason (pitfall 15) |

It was never unbackable either. MRS quantification **is** fitting the FID — the
metabolite concentrations are the resonance amplitudes — the fit is analytic and
self-supervised, and none of it needed data the repo lacks. What it needed was for
somebody to write it.

> **The magnitude spectrum is √3 times broader than the absorption spectrum, and
> this is the easiest way to misread an MRS linewidth by 73%.** The FT of
> `exp(-π·Δν·t)` for `t ≥ 0` is `1/(π·Δν + i·2π·f)`. Its **real part** is a true
> Lorentzian with FWHM = `Δν`; its **magnitude** is the *square root* of a
> Lorentzian — not a Lorentzian — with FWHM = `√3·Δν`. Both are verified against
> their closed forms (deviation 3.13e-03 and 3.29e-05).
>
> This bites here because `spectral_linewidth`, the metric backing this regime,
> documents its input as a **magnitude** spectrum. It therefore reports `√3·Δν`
> for a model fitted with linewidth `Δν`. Grading a fitted `Δν` directly against
> that metric is a 73% error dressed up as a disagreement (pitfall 18). Pinned by
> `test_magnitude_mode_fwhm_is_sqrt3_times_broader`; the constant lives in code as
> `MAGNITUDE_FWHM_FACTOR`, not in folklore.

> **The MRS residual TELLS THE TRUTH — which is the opposite of the perfusion
> trap, and changes what you do about a bad fit.** The objective is identifiable
> but non-convex: the frequency capture range is roughly ±30 Hz.
>
> | frequency init (true 120) | residual | recovered? |
> |---|---|---|
> | off by 30 Hz | 1.07e-04 | yes |
> | off by 100 Hz | 1.64e-01 | no |
> | all zeros | 1.66e-01 | no (all 3 resonances collapse onto one line) |
>
> A wrong MRS fit is **three orders of magnitude worse** and announces itself.
> Contrast `tofts_residual`, where a ~1e-4 residual coexists happily with a
> *negative* `vp`: there the data cannot identify the parameter, the residual
> lies, and a **constraint** is the answer. Here the data identifies everything
> and what is missing is a **starting point** — which is exactly why AMARES
> supplies prior knowledge of metabolite positions. Do not "fix" a bad frequency
> fit with a constraint.

> **`mrs_prior_knowledge`'s amplitude term is dead under the default activation** —
> the same trap as the perfusion box loss and `vp`. `parameter_activation="softplus"`
> forces amplitudes and linewidths positive *structurally*, so `clamp(-amplitude)`
> is identically zero with zero gradient; only the linewidth bounds are live. A
> healthy prior-knowledge loss is NOT evidence non-negativity is being learned.
> Pinned by `test_the_amplitude_term_is_dead_under_softplus`, which asserts the
> gradient is exactly zero.

## Strategy tags — and the deliberate non-tags

Tagged (9). Each was read and verified as real, regime-specific physics:

| Strategy | Regime | Why it is honest |
|---|---|---|
| `ReconstructionTrainingStrategy` | structural | the original LIVE baseline |
| `PhaseContrastFlowStrategy` | flow | phase-contrast encoding physics |
| `PerfusionKineticMappingStrategy` | perfusion | extended-Tofts refit |
| `MultiEchoB0FitStrategy` | quantitative | differentiable Hz field fit, graded by `B0FieldRMSE` |
| `OneShotMultiParameterStrategy` | quantitative | Bloch-anchored T1/T2/PD; raises rather than degrade |
| `ConformalMRFDictlessReconStrategy` | fingerprinting | dictionary-free fingerprint matching |
| `QSpaceDiffusionStrategy` | diffusion_weighted | real SH basis from `batch["b_vectors"]` — **now wired (internal issue 350)**: reads the exact key `LoadDWIMetadata` writes and raises if absent. Still not the load-bearing evidence for the DWI tag (that is `adc_monoexp`); honest added coverage |
| `LowRankSparseStrategy` | dynamic | RPCA `X = L + S` |
| `BeltramiEPIDistortionStrategy` | functional + diffusion_weighted | EPI readout physics (see caveat) |

**NOT tagged, deliberately.** A wrong tag is worse than no tag: it makes a false
claim machine-*verified*.

| Strategy | Why not |
|---|---|
| `B0MappingStrategy` | **the sharpest trap** — its own docstring: *"despite the `b0_mapping` name, this strategy does not estimate a B0 off-resonance field map in Hz"*. It is deformable registration. The obvious name is wrong. |
| `SpatiotemporalMRFReconStrategy` | no Bloch, no dictionary; its "4D Beltrami" is `0.05 * x.diff(dim=-1)`. Runs identically on any 5-D tensor. |
| `HRFManifoldDiffusionStrategy` | textbook 1-D DSM, nothing HRF-specific; `input_batch` is never used beyond reading `device`. |
| `CorticalConformalFMRIReconStrategy`, `SpatiotemporalAdaptiveSFCReconStrategy` | degrade to plain L1 + Laplacian without their conformal inputs. |
| `RiemannianDFCDiffusionStrategy` | real SPD geometry, but on `[B,n,n]` connectivity matrices — which the FUNCTIONAL profile's own `signal_domains`/`spatial_ranks` deny. |
| `CRLBMRFPulseDesignStrategy` | self-admitted toy Fisher surrogate; **and** its `ACQUISITION_DESIGN` task is not in fingerprinting's `supported_tasks`. |
| `CrossScannerMRFHarmonisationStrategy` | real MRF content, but **no `Task` member describes harmonisation**. Do not invent one to force a tag. |
| `RiemannianMRFDiffusionStrategy` | `kwargs.get("fingerprint")` returns `None` silently, and the fingerprint conditioning is what would make it FINGERPRINTING rather than QUANTITATIVE. Tag it once that is made loud. |

Losses and metrics carry the same discipline. Deliberately **untagged**:

| Component | Why not |
|---|---|
| `tropical_semiring_consistency` (loss) | aliases itself `tropical_mrf_consistency`, but its reference fan is `torch.randn` — a generic max-plus penalty. The single sharpest lure in the loss registry. |
| `conformality_jacobian` (loss) | "MRF §3" in the label, pure differential geometry in the body. |
| `hrf_likelihood` (loss) | real BOLD GLM physics, but `mri_functional`'s claim is recon-scoped by its `supported_tasks`; tagging it would imply the framework models neural activity. Needs a BOLD/GLM `Task` member first. |
| `efc` / `fber` / `qi1` (metrics) | MRIQC IQMs that apply to **any** MR image — MRIQC runs them in the functional pipeline too. Only `cjv` and `wm2max` presuppose anatomical tissue contrast, so only those are `mri_structural`. |
| `cosine_preservation_score` (metric) | fingerprint-adjacent, but uses `prediction`/`target` off-contract (embedded `[B,d]` vs raw `[B,N]`). `geodesic_mrf_parameter_error` already backs the regime honestly. |
| `st_mad` (metric) | generic video-IQA with a `temporal_weight`; not MRI-dynamic-specific. |
| L1 / SSIM / PSNR (everything agnostic) | `None` = agnostic is the correct answer, not a gap. This is why LIVE does not require a tagged loss. |

> **`mri_functional`'s PARTIAL means "EPI distortion correction exists", NOT
> "BOLD modelling exists".** `BeltramiEPIDistortionStrategy` is real,
> unconditional physics (true gyromagnetic ratio, phase-encode-only
> displacement, physics-in-the-loop, no fallback), and `{functional,
> diffusion_weighted}` is a *precise* claim — EPI is the readout for exactly
> those two regimes. But it models the **readout**, not neural activity. LIVE
> still requires tSNR/GLM losses, a BOLD metric, and a real BOLD recon strategy.
> This is written down rather than smuggled.

## Bugs found while completing the regimes (2026-07-16)

Every one of these was **silent** and returned a **plausible number**. That is the
common shape: they are invisible to value assertions, so only structure catches them.

- **`storm`'s Laplacian was not a Laplacian, and the loss was unbounded below.**
  The loop wrote four entries per iteration with `=` rather than `+=`, so each
  step overwrote the last one's diagonal and the interior diagonal ended at **1**
  instead of the node degree **2**. Row sums were non-zero and the minimum
  eigenvalue **negative** (−0.73 at T=5), so `Tr(X L Xᵀ)` had no lower bound: a
  constant frame sequence scored **−3**, and scaling it drove the "smoothness
  penalty" toward −∞. It did not merely fail to smooth — it **rewarded blowing
  the reconstruction up**, while showing a loss curve going satisfyingly down.
- **`beltrami_epi_residual` fell back to plain L1** without a `delta_b0`, *while
  still reporting under the key `beltrami_epi_residual`*. An arm whose headline
  method is EPI distortion correction trained as a vanilla denoiser and logged a
  term named after physics it was not doing (pitfall 16).
- **`hrf_likelihood` defaulted its reference to `zeros_like(pred)`** — not a
  degraded GLM but a **shrink-the-HRF-to-zero penalty**. It converged beautifully,
  on the opposite of the intended objective (internal issue 338).
- **`conformality_jacobian` returned a hard `0.0`** on a missing Jacobian — a term
  with no gradient, silently, forever. A zero loss is indistinguishable in the
  logs from a *satisfied* constraint.
- **`CJV` returned sentinels `1.0` / `10.0`** on degenerate input. Both are
  perfectly ordinary CJV readings, so the one image that most needs flagging — a
  collapsed prediction with no tissue contrast at all — scored a healthy `1.0`.
  Now `NaN` (skip), matching `b0_field_rmse` / `adc_mae`.

## What the adversarial review caught in the NEW code (2026-07-16)

The verticals were reviewed again after the regimes went LIVE, each finding handed
to a skeptic told to REFUTE it. Two defects survived, **both in code written that
same day**, and both had a green test asserting the very property they broke.

- **`temporal_fidelity` was not scale-free.** The gradeable gate tested
  `ref_norm > 1e-6`; the *scoring* gate tested `pred_norm * ref_norm > 1e-6`
  against the same constant. For a perfect reconstruction the product is
  `ref_norm²`, so scoring 1.0 silently required `ref_norm > 1e-3`. Every voxel in
  between was graded and scored **0.0 despite being bit-perfect** — which is
  exactly the value the metric reserves for *total temporal collapse*. On a
  higher-is-better metric the two opposite verdicts were indistinguishable.
  `test_insensitive_to_per_voxel_scale_and_offset` asserts precisely this property
  and could not fail: `ref * 3.0 + 7.0` scales **up**, away from the threshold.
  Both gates are now on a per-voxel std, which is also T-independent where a norm
  is not (`norm = std·√(T-1)`).
- **`tsnr`'s foreground threshold did not exclude air.** The 25th intensity
  percentile excludes air only when air is under **25% of the FOV** — and a real
  EPI FOV is 60-80% air. The threshold landed *inside* air, most air voxels passed
  with `mean_t/std_t ≈ 0`, and the score tracked FOV/cropping rather than image
  quality: at a realistic 30% brain fraction it read **~41 against a true ~100**.
  It was not a tSNR in the Triantafyllou/MRIQC sense at all. Now an explicit
  `mask` when available (what MRIQC does), else **Otsu**, which adapts to the
  actual air fraction. Otsu is a heuristic and still degrades past the realistic
  range (~83 at 10% brain, where ~2% of air leaks through and air outnumbers brain
  9:1) — pinned as a documented limit rather than papered over.

Also fixed, found in the same pass: **`storm` allocated `[B, P, P]`** (`P = C·H·W`)
via two bmms and read only its diagonal — **4.3 GB at 128², 68.7 GB at 256²**, a
guaranteed OOM on exactly the dynamic MRI the regulariser is for. Contracting `L`
with `X` first gives the identical trace (`max abs diff 0.0`) at `[B, T, P]` —
8.4 MB — and `O(B·T·P)` instead of `O(B·T·P²)`.

Refuted and dropped after reproduction attempts: the ADC forward/inverse round
trip (exact to 2.3e-10), tSNR detrending (verified), `soft_dictionary_match` at
`temperature → 0` (exactly the hard argmax), softmax overflow (`torch.softmax`
subtracts the row max), every broadcast path in the MRF loss (all guards fire),
and the storm Laplacian fix itself (min eig −8.6e-08, row sums 0).

## What an adversarial review caught (2026-07-16)

A three-dimension review of the verticals, each finding handed to a skeptic told
to refute it, surfaced six defects that unit tests had missed. Recorded because
the *shapes* recur:

- **A broadcast that never raises.** `velocity_pred[:, idx]` (dropping the
  channel axis) against a `[B,1,H,W]` mask silently became `[B,B,H,W]` — every
  sample's velocity times every other sample's mask — and returned a perfectly
  ordinary number. Value assertions cannot catch this; assert the *shape the loss
  receives*.
- **An activation applied in one place and not the others**, so the maps that got
  graded were not the maps that got fitted. The unit test passed because it
  called the split function in isolation: it proved the activation existed, never
  that its output reached the graders. **Assert the seam, not the unit.**
- **A test that passed vacuously** — `assert torch.isfinite(loss)` is true whether
  or not the knob under test is ever read.
- **Docstrings that claimed more than their tests proved** ("the strategy's actual
  scientific claim" on a test that never calls the strategy). A facade in prose.

## Cross-cutting deferred items

- ~~The LIVE rule / `signal_model` registry~~ — **done 2026-07-16**, see above.
- ~~`fmri_mrf_losses` silent fallbacks~~ — **fixed 2026-07-16**. All three now
  raise: `beltrami_epi_residual` (fell back to plain L1 with no `delta_b0`, while
  still reporting under its own name), `hrf_likelihood` (defaulted its reference
  to `zeros_like(pred)` — a shrink-the-HRF-to-zero penalty, not a degraded GLM),
  `conformality_jacobian` (returned a hard `0.0`, which reads in the logs exactly
  like a satisfied constraint).
- **The metric-reachability item previously recorded here was WRONG, and the
  truth is worse.** `validation.metrics: [name]` resolves **any** of the 206
  registered metrics by name with no allow-list (`ValidationMetricsConfig.from_dict`
  → `MetricSpec.from_name`), so "no `compute_*` flag" never meant unreachable —
  the flow/perfusion metrics were reachable all along. The real defects:
  - ~~There are **three** pairwise-inconsistent flag→name maps:
    `InfrastructureBuilder.metric_map`, `_extract_metrics_from_config.compute_flags`,
    and `pipelines.training_loop.metric_name_map`. The same flag behaves differently
    depending on which path runs — fatal via one, warn-and-skip via the other.~~ —
    **fixed 2026-07-18.** `core.metrics.flag_map.metric_for_flag` is now the single
    owner of the flag→name relationship (identity + one documented alias). The builder
    consumes the derived `FLAG_TO_METRIC`; the two remaining literal maps are
    **ratchet-locked** to `metric_for_flag` by `tests/unit/core/metrics/test_flag_map.py`,
    so the same flag can no longer map to two names. Each consumer keeps its own
    *coverage* policy (the per-batch builder deliberately excludes offline/report/
    distribution metrics) — coverage is a separate, intentional concern from naming.
  - ~~**`compute_kspace_entropy: true` / `compute_kspace_high_freq_error: true`
    hard-crash the run** with `ConfigurationError`~~ — **fixed 2026-07-18.** No metric
    implemented either, so both flags were removed from the schema, all three flag
    maps, and the two direction tables (plus the two phantom `KSpaceEntropy` /
    `KSpaceHighFreqError` rows in `COMPLETE_FEATURE_REFERENCE.md`). Enabling one now
    fails at load under `extra="forbid"` — a clean rejection, not a build-time crash.
    `KNOWN_MAP_TO_REGISTRY_GAPS` is now empty and stays as a ratchet. Backlog the
    metric + registration if the measurement is wanted.
  - 20 flags are doubly dead (no map entry, no registered metric), including the
    fMRI QA family `compute_dvars`/`compute_gcor`/`compute_sfnr`. `compute_tsnr`
    was one of them until `tsnr` landed 2026-07-16.
  - ~~Near-miss name pairs that silently never fire: `compute_neg_voxels` vs the
    registered `negative_voxels`; `compute_spike_percent` vs `spike_percentage`.~~ —
    **fixed 2026-07-18.** `compute_neg_voxels` now resolves to `negative_voxels` via the
    `metric_for_flag` alias table (was the unregistered identity `neg_voxels`);
    `compute_spike_percent` — a byte-duplicate of `compute_spike_percentage` whose
    identity target was never registered — was removed from the schema and the
    training-loop map.
- ~~`MetricsRegistry.clear()` leaks `_workflow_tags`~~ — **fixed 2026-07-16**.
  Tags outlived their metrics, so after a test called `clear()` the ledger still
  reported coverage backed by nothing.
- ~~`register_metric(workflows=...)` is unvalidated~~ — **fixed 2026-07-16**. The
  ledger does `regime in workflows`; against a bare string that is a SUBSTRING
  search, so a typo'd tag could silently satisfy or miss a maturity claim. Now
  raises, matching the adjacent `direction=`.
- **`CCSNR` violates the `IMetric` contract**: `forward(dwi, cc_mask, ...)` can
  never be invoked by `ValidationMetricsComputer`, which calls `metric(pred, target)`.
- `ToftsModel` (metric) is a **duplicate, non-differentiable second Tofts
  implementation** with different Parker constants (`sigma2=0.134` vs the SSOT's
  `0.132`, plus a `dose` factor) that feeds *frames* to minute-scaled constants.
  Physics-SSOT violation. Filed (internal issue 339).
- `spgr_signal_to_concentration` accepts and discards `t10_s`/`tr_s`/
  `flip_angle_deg` (pitfall 15). Filed (internal issue 341).
- DTI / IVIM signal models + tensor losses (FA/MD metrics). Added DWI coverage,
  **not** a gap in its LIVE claim, which rests on `adc_monoexp`.
- A `Task` member for MRF harmonisation (blocks `CrossScannerMRFHarmonisation`),
  and a BOLD/GLM task member — which is what would let `hrf_likelihood` be tagged
  and widen `mri_functional`'s claim beyond reconstruction.
- ~~`mri_spectroscopy` → PARTIAL~~ — **went straight to LIVE, 2026-07-16.**
  EVAL_ONLY is now empty; `test_no_mr_regime_with_real_physics_is_left_at_eval_only`
  keeps it that way and names any regime that slides back.
- **A metabolite-basis (LCModel) signal model** for spectroscopy — the linear
  combination of *measured* basis spectra, rather than free Lorentzians. Needs a
  basis set the repo does not ship, which is why `data.spectroscopy.signal_model`
  is a closed Literal with one member (pitfall 15). It would also fix the frequency
  capture range by construction, since basis positions are chemistry.
- ~~A `spectrum` `Domain` literal~~ — **added 2026-07-16**, with
  `check_workflow_signal_domain` cross-checking `profile.signal_domains` against
  the resolved model's `input_domain`. See below for its first finding.
- Full component tagging: the bulk of losses/metrics remain untagged
  (`None` = agnostic, intentional).
- ~~Signal-domain audit rule~~ — **landed 2026-07-16** as
  `check_workflow_signal_domain`, and wired into the audit run. It was skipped
  originally "for false-positive risk"; the risk was real and determined the
  design (see below). Swept against the live registry: **0/121** annotated models
  rejected across all eight image/k-space regimes.
- **`mri_spectroscopy` has no registered model that can legally serve it** — the
  new check's first finding. 588 models are registered, 121 declare an
  `input_domain`, and **not one declares `spectrum`**, so every annotated model is
  correctly rejected for this regime. Nothing breaks today (no MRS data, no MRS
  arms, and an unannotated model still skips), and the regime's LIVE claim is
  unaffected — LIVE asserts physics + strategy + metrics, and none of the nine
  regimes' rules mention models. But an MRS arm cannot be built from the registry
  as it stands. Fix: annotate an FID-capable generator with
  `input_domain="spectrum"` (or `("image", "spectrum")` where a conv net genuinely
  serves both, as `data.spectroscopy` packs the FID as `2*T` real channels).
- ~~**`QSpaceDiffusionStrategy`'s q-space mechanism is unconditionally inert
  (pitfall 16).** It gates its SH / Laplace-Beltrami term on `batch["bvecs"]`,
  and nothing writes that key.~~ — **fixed 2026-07-18 (internal issue 350).** `LoadDWIMetadata`
  now attaches `b_vectors` (and `b_values`) as **tensors**; the strategy reads that
  exact key (`bvecs` is gone), reduces the collated `(B,N,3)` to the shared `(N,3)`
  with a homogeneity check, and **raises** if a diffusion batch lacks directions
  rather than silently returning the base recon losses. The seam is pinned by
  `test_qspace_diffusion_bvecs_wiring_2026_07.py` (transform → collate → strategy
  fires; missing / heterogeneous directions raise) and by
  `test_qspace_strategy_reads_the_directions_key_the_data_layer_writes`
  (producer∩consumer ≠ ∅). `mri_diffusion_weighted`'s LIVE claim still rests on
  `adc_monoexp` / `dwi_adc_monoexp` / `adc_mae`; the q-space regulariser is honest
  added coverage, not the load-bearing evidence. Follow-up: heterogeneous
  per-sample direction sets (ragged batches) still raise rather than being handled.
- **Two `WorkflowProfile` fields are still declared-but-unread**, so
  `signal_domains` was *not* "the last declared-but-unchecked profile field"
  (as commit `17d6c990d` claimed):
  - `modality` — set by every profile, read by nothing. Argued deferral: every
    runnable regime is MR and the non-MR regimes are `STUB`, which
    `check_workflow_declared` rejects before a modality gate could fire (see
    `domain/workflows/knobs.py`). The gate lands with the first non-MR regime.
  - `optional_axes` — inert twice over: no reader, and no profile populates it.
    An optional axis has no rule to state, since its absence cannot make an arm
    inadmissible. **Candidate for deletion, not for a reader** (pitfall 15).

  Both are pinned by `test_every_profile_field_is_classified_as_read_or_unread`
  and `test_the_unread_profile_fields_really_have_no_reader`, so a new field
  cannot be added without a reader or a stated reason, and the "last field" claim
  cannot be re-made while any remain.


## What is deliberately NOT here

- **No `experiments/` arms for flow or perfusion** — but *not* for the reason
  this section gave until 2026-07-17. The old premise ("the cluster has M4Raw
  0.3T static single-contrast magnitude + fastMRI knee/brain — no 4D-flow, no
  DCE") was wrong twice over, and the cluster data inventory taken
  2026-06-16 (an internal operations note, not distributed) had already
  retired it:
  - **4D-flow data is present and extracted** (`osu_4dflow_cine`,
    `whole_heart_5d`). The cluster holds the whole 44-dataset external catalogue,
    not M4Raw alone.
  - **M4Raw is multi-contrast** T1/T2/FLAIR (per-contrast repetition counts live
    in `data/datasets/m4raw_dataset.py`), not single-contrast. Contrast is not an
    `Axis` member in any case — it is `Task.CONTRAST_TRANSLATION`.

  DCE remains genuinely absent (no perfusion/tracer dataset in the inventory), so
  a perfusion arm would still be pitfall 19. A flow arm is blocked by **wiring,
  not data**: no `dataset_type` is routed to expose `VELOCITY_ENCODING` from those
  `.dat`/`.mat` acquisitions. The strategies' scientific claims are verified on
  analytic synthetic physics instead, which is stronger than a smoke run.
- **No `DATASET_TYPE_AXES` annotation** for a speculative `4d_flow`/`dce`
  `dataset_type` — and note what that does and does not buy, because the previous
  wording here had the polarity backwards. It claimed that since no
  `dataset_type` exposes `VELOCITY_ENCODING` or `TEMPORAL`,
  `check_workflow_required_axes` "**rejects** every flow/perfusion arm on the
  available data. That is the guard working — do not disarm it."

  It is the opposite. The check rejects an arm only when the `dataset_type` **is**
  annotated in `data/datasets/axis_exposure.py` and its declared axes lack the
  required one. An **unannotated** `dataset_type` returns `None` from
  `exposed_axes_for` and the check **skips** it (`passed=True`, `severity="info"`)
  — an absent annotation is never a rejection. So the missing annotation disarms
  the guard rather than constituting it; what protects a regime is a *present*
  annotation that omits the axis. Adding annotations is how the guard gets teeth.
  (Pinned by `test_required_axes_guard_skips_unannotated_but_rejects_annotated`.)
- **Maturity is a fact about the code, not about a cluster.** If it tracked local
  data, `mri_flow` would be PARTIAL here and LIVE at a 4D-flow site — incoherent
  for a frozen profile. Data availability is a separate, already-wired gate.
