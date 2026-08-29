# Workflows — imaging regime × task

The framework is MRI-shaped, but "MRI" is not one thing. A structural T1 recon,
a quantitative T2 map, a 4D-flow velocity field, and an MRS spectrum are
different *signals* that need different physics, losses, and metrics. The
`workflow:` block makes that difference a first-class, checkable declaration.

```yaml
workflow:
  regime: mri_quantitative   # the physical REGIME — what the signal IS
  task: parameter_mapping    # what you're DOING to it
  signal_domain: kspace      # which of the regime's domains this arm CONSUMES
  spatial_rank: 2            # 2 = slices, 3 = volumes
```

`signal_domain` and `spatial_rank` are optional and **narrow** the regime. A
profile gives a *set* — `mri_quantitative` yields `image`/`kspace`/`complex_image`
at rank 2 or 3 — so the profile alone cannot say which one an arm is in.
Absent leaves the value inferred (advisory); wrong is a hard audit error. An
author who is unsure should omit rather than guess.

`signal_domain` means what the arm **consumes** (matching
`ModelCapabilities.input_domain`), never what it emits. Every parameter-mapping
arm emits off-regime — MRS consumes a `spectrum` and emits resonance maps in the
`image` domain — so an emits-reading would reject the very arms these regimes
exist for.

The `regime` field was called `name` until 2026-07-31; the old spelling raises
and names its replacement.

Two `Task` members were resolved on 2026-07-31, and the two had opposite fixes.
`acquisition_design` was **wired**: `PILOTStrategy` and `BALDAcquisitionStrategy`
really do design acquisitions, but both subclass `ReconstructionTrainingStrategy`
and inherited its `{RECONSTRUCTION}` tag, so a real capability never announced
itself. `segmentation` was **removed**: across 152 strategy classes none tagged
it, no loss carried it, no profile listed it, and no arm declared it — the only
segmentation the framework has is metrics, and a task you can score but not
train is not a task an arm can declare.

`WorkflowProfile.optional_axes` was deleted the same day. It was inert twice
over — nothing read it and no profile populated it — and unlike an
unread-but-populated field it did not want a reader: `required_axes` alone
drives `check_workflow_required_axes`, and an *optional* axis states no rule,
since by construction its absence cannot make an arm inadmissible.

Both `regime` and `task` are closed enums
(`mriforge.config.schemas.enums.Regime` / `Task`), so a typo fails at Tier-0
(Pydantic `ValidationError`) rather than silently mislabelling the arm.

## Regime vs task vs domain

Three orthogonal axes, easy to conflate:

- **Regime** (`Regime`) — the physical nature of the signal: `mri_structural`,
  `mri_quantitative`, `mri_flow`, … `mri_diffusion_weighted` is spelled out in
  full because bare "diffusion" already means DDPM/score models here.
- **Task** (`Task`) — what the arm does: `reconstruction`, `super_resolution`,
  `field_translation`, `parameter_mapping`, …
- **Signal domain** (`mriforge.models.capabilities.Domain`) — where a tensor
  lives: `image` / `kspace` / `complex_image` / `latent`. This is unchanged and
  unrelated to regime.

4D/5D acquisitions are **axis compositions**, not enum members: "4D cine" is 3
spatial ranks + `Axis.TEMPORAL`; "5D 4D-flow" adds `Axis.VELOCITY_ENCODING`.
Encoding dimensionality as `spatial_ranks × required_axes` is what lets the
audit check "regime needs TEMPORAL; this dataset exposes none".

Which means a 5-D arm's axes are frequently a **per-arm** fact, not a per-type
one — the same `dataset_type` serves arms whose acquisitions differ. `Axis` is
also strictly the *non-spatial, acquisition* vocabulary: a parameter-map stack
`(T1, T2, PD)` is **not** an axis. It is a heterogeneous tuple of three
different units that no acquisition varied to produce — the estimand, not an
encoding — so it gets no `Axis` member and a map-serving arm honestly exposes
none.

## The maturity ladder

Every regime carries a `Maturity` that says how far the framework can honestly
run it:

| Maturity    | Meaning                                              | Behaviour                                   |
|-------------|------------------------------------------------------|---------------------------------------------|
| `LIVE`      | a registered forward model (operator **or** signal model), a regime-tagged strategy, **and** regime-tagged metrics | runs |
| `PARTIAL`   | a regime-tagged strategy, **or** both tagged losses and tagged metrics | runs; audit reports the gap |
| `EVAL_ONLY` | metrics exist, zero losses/strategies                | `evaluate`/`predict` run; `train` raises    |
| `STUB`      | nothing exists                                       | every pipeline raises `WorkflowNotImplementedError` |

Maturity is not hand-waved. It is **declared** on the frozen
`WorkflowProfile` and **asserted against the live registries** by
`tests/unit/domain/workflows/test_maturity_ledger.py`. A profile that claims
`LIVE` without a registered, tagged strategy fails that test; an `EVAL_ONLY`
regime that has no tagged metric fails it too. This is the anti-facade
guarantee — the claim cannot lie.

`EVAL_ONLY` currently has **no members** — `mri_spectroscopy` was the last one and
reached LIVE on 2026-07-16. The rung stays in the ladder because it describes a
real state (a regime you can grade but not train), and
`test_no_mr_regime_with_real_physics_is_left_at_eval_only` keeps the list empty by
naming any regime that slides back into it.

### The three LIVE clauses

Together they say the regime's physics is **expressible**, **trainable**, and
**gradeable on its own terms**.

**A forward model is an operator *or* a signal model**, and which one a regime
declares is a fact about the physics, not a style choice. An `OperatorRegistry`
entry *promises* `adjoint()`, and generic callers — `DataConsistencyLayer`,
`FISTAMBIRSolver`, `NullSpaceProjection` — consume that promise assuming
`⟨Ax,y⟩ = ⟨x,A†y⟩`. A nonlinear map has no adjoint, so it goes in the
`SignalModelRegistry`, whose contract has **no adjoint in it at all**.

Until 2026-07-16 the clause read `forward_operator is not None`, and it failed in
both directions at once. It was far too weak for most regimes — *every* MR profile
declares `fft2d`, because MR k-space is FFT-reconstructed, so the clause tested
"is this MR?" rather than "is this regime's physics wired?". And it was impossible
for `mri_perfusion`, whose tracer-kinetic map is nonlinear: the only way to satisfy
it was to register a fake `adjoint`, so **the rule actively rewarded the facade it
existed to prevent**. `mri_diffusion_weighted` honestly declares both — `fft2d` for
the readout, `adc_monoexp` for the decay.

**Metrics are required; losses deliberately are not.** The asymmetry is real. A
regime may honestly train on agnostic objectives — `mri_structural` trains on
L1/SSIM, and that is *correct*, not a gap. But a regime cannot be *graded*
generically once its output stops being an image: PSNR on a Ktrans map is a
metric↔claim mismatch. Requiring a regime-tagged loss would force mis-tagging L1
as "structural-only", which is false, and mass-tagging generic components is
exactly how an allow-list rots — the same failure as the MRO leak below, one layer
up. Regimes that *do* have distinctive physics are separately held to having a
tagged loss by `test_regimes_with_distinctive_physics_have_tagged_losses`, with
structural the single argued exemption.

### Why PARTIAL has two tiers

PARTIAL is satisfied by **either** a tagged strategy (*strategy-backed*) **or**
tagged losses **and** tagged metrics (*physics-backed*). The `and` in the second
tier is load-bearing. The obvious rule — "a strategy *or* a loss *or* a metric" —
is strictly **weaker than the EVAL_ONLY rule**, which already demands metrics
*and* no strategies *and* no losses. One tagged metric would satisfy it, and a
metrics-only regime is *literally the EVAL_ONLY state*. Since PARTIAL allows
`train` while EVAL_ONLY raises on it, the permissive form would let a single
metric tag buy a trainable claim.

This branch did not exist until 2026-07-16: PARTIAL fell through the
`if/elif/elif` with no assertions at all, so five regimes sat there for months
with nothing tagged. Two things hid it — the missing branch, and a leak where
`capabilities` (a `ClassVar`) was read with `getattr`, so all 63 subclasses of
the structural strategy inherited its `mri_structural` tag for free. 64 strategies
reported the tag; one declared it. The ledger now reads each class's own
`__dict__`: inheriting a parent's regime is not opting into it.

In this repository the ledger stands as — **all nine MR regimes with real physics
are LIVE as of 2026-07-16, and EVAL_ONLY is empty**:

- `mri_structural` — **LIVE** (`ReconstructionTrainingStrategy`, `fft2d`, graded
  by the anatomical IQMs `cjv`/`wm2max`). Its losses are agnostic *by design*.
- `mri_flow` — **LIVE**: real phase-contrast physics
  (`infrastructure/physics/signal_models/flow_encoding` + the `phase_contrast`
  operator, whose `pc_adjoint` is a true inverse within `|v| < venc`),
  FLOW-tagged losses and metrics, and the trainable `PhaseContrastFlowStrategy`.
- `mri_perfusion` — **LIVE on a signal model**, with `forward_operator=None`, and
  that is the *correct* answer rather than a gap: the tracer-kinetic map is
  nonlinear and has no adjoint. `PerfusionKineticMappingStrategy` resolves
  `data.perfusion.kinetic_model` through the `SignalModelRegistry`, so the
  ledger's claim and the runtime dispatch resolve the same key from the same
  registry rather than agreeing by coincidence.
- `mri_quantitative` — **LIVE**: `MultiEchoB0FitStrategy` /
  `OneShotMultiParameterStrategy`, relaxometry losses that push predicted maps
  through the SPGR equation and the Bottomley prior, graded in physical units by
  `b0_field_rmse` / `geodesic_qmap_error` / `cross_scanner_t1t2_concordance`.
- `mri_fingerprinting` — **LIVE**: `ConformalMRFDictlessReconStrategy` +
  `mrf_dictionary_match`, which does MRF's defining operation (softmax-relaxed
  inner-product matching against a Bloch dictionary). The regime previously had
  *no honest loss at all* — every MRF-named one is a themed name over generic
  maths, `tropical_mrf_consistency`'s reference fan being literally `torch.randn`.
- `mri_diffusion_weighted` — **LIVE**, and the one regime declaring both kinds of
  forward model: `fft2d` for the readout, `adc_monoexp` for the decay.
- `mri_dynamic` — **LIVE**: `LowRankSparseStrategy`, the SToRM manifold
  regulariser, and `temporal_fidelity`.
- `mri_functional` — **LIVE, and read its `supported_tasks` before reading that
  as more than it is.** They are the RECON family and nothing else, so the claim
  is "the framework can reconstruct and denoise a functional series" — real EPI
  readout physics (`BeltramiEPIDistortionStrategy`), graded by `tsnr` and
  `temporal_fidelity`. It is **not** a claim that BOLD modelling exists: the
  strategy models the readout, not neural activity, and there is no GLM or BOLD
  task member. `hrf_likelihood` is real BOLD physics and stays deliberately
  untagged for that reason. `supported_tasks` is what bounds a LIVE claim, which
  is why it is a fact and not decoration.
- `mri_spectroscopy` — **LIVE**, and the last regime to leave EVAL_ONLY. Like
  perfusion it is LIVE on a signal model with `forward_operator=None`: the
  Lorentzian sum is nonlinear in frequency, linewidth and phase and has no
  adjoint. MRS quantification **is** fitting the FID — the metabolite
  concentrations are the resonance amplitudes — so the objective is
  self-supervised, exactly as tracer kinetics are. It was never unbackable; it
  needed writing. **Two traps are documented rather than smoothed over**: a
  magnitude spectrum's peaks are √3 times broader than the absorption
  spectrum's, and `spectral_linewidth` takes a magnitude spectrum; and the
  frequency fit has a ±30 Hz capture range, though — unlike perfusion's `vp` —
  a bad fit shows a residual three orders of magnitude worse, so the residual
  can be trusted. The per-combination backlog is maintained with the internal
  documentation and is not published with this release.
- `nmr_spectroscopy`, `ct`, `xray`, `ultrasound`, `optical` — **STUB** (typed
  seams the framework does not implement).

Maturity is a fact about the **code**, not about the data on any one cluster. If
it tracked local data, `mri_flow` would be PARTIAL here (no 4D-flow data) and
LIVE at a flow site — incoherent for a frozen profile. Data availability is a
separate, already-wired gate: `check_workflow_required_axes`.

The full table is auto-derived in
[Workflow profiles](../reference/workflow_profiles.md).

## Component tagging (`None` = agnostic)

A component opts into a regime/task by setting `workflows=` / `tasks=` on its
registration:

```python
@register_metric("vnr", workflows=frozenset({Regime.FLOW}))
class VelocityNoiseRatio: ...

class ReconstructionTrainingStrategy(...):
    capabilities = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.RECONSTRUCTION}),
    )
```

The invariant from `mriforge.models.capabilities` holds: **`None` (the default)
means "unannotated / agnostic" and is skipped**. Only genuinely
regime/task-specific components are tagged — agnostic losses/metrics (`l1`,
`ssim`, `psnr`) stay `None`. Tagging everything is how allow-lists rot.

A tag must be **declared on the class itself**. `capabilities` is a `ClassVar`,
so a tag on a base class is visible via `getattr` on every subclass — which is
how allow-lists rot *without anyone tagging anything*. The ledger reads
`cls.__dict__` for exactly this reason.

**A wrong tag is worse than no tag**: it makes a false claim machine-*verified*.
`B0MappingStrategy` is the cautionary case — the name says `b0_mapping`, and its
own docstring says *"despite the `b0_mapping` name, this strategy does not
estimate a B0 off-resonance field map in Hz"* (it is deformable registration).
Tagging it `mri_quantitative` on the strength of its name would have asserted a
claim the code explicitly disclaims. The deliberate non-tags, and the reason for
each, are recorded in the internal workflow backlog, which is not published
with this release.

The registry-walk that reads these tags lives in
`mriforge.infrastructure.validation.workflow_ledger` (in `infrastructure`, not
`domain`, because reading strategy capabilities means importing strategy
classes — a dependency `domain` may not take).

## Enforcement

Two layers enforce the contract.

**Audit (Tier-1)** — `ConfigHealthChecker.check_workflow_declared` **hard-errors** when:

- the regime is a `STUB` the framework cannot run;
- the `task` is not in the regime's `supported_tasks`.

An **absent** `workflow:` block is **advisory** (`info`), not an error — the
"optional now, required later" seam. None of the 1,465 experiment YAMLs on `dev`
predates this feature with a `workflow:` block, so erroring would redden every
arm on the first `mriforge audit` (which is `--strict` by default) rather than
enforce anything. The two checks above still fire on any arm that *did* declare,
so a **wrong** declaration never passes silently. Ratchet the absent case to
`error` once the cohorts are annotated — see issue #283.

`check_workflow_required_axes` (Tier-1) is the machine-readable form of
**pitfall #19** ("hypothesis untestable on this data"): it errors when a
regime's `required_axes` are absent from what the arm exposes — e.g.
`mri_functional` (needs `TEMPORAL`) on `dataset_type: image` (exposes none).

Axes resolve by **two routes, declared first** (both in
`mriforge.data.datasets.axis_exposure`):

1. **declared** — `declared_axes_for(data_cfg)` reads a per-*arm* statement the
   config layer already validated: today `data.bart.bart_dim_map`, which
   `BartConfigSchema` requires non-empty when enabled, rejects unknown roles in,
   and refuses to leave a non-singleton dimension unnamed.
2. **annotated** — `exposed_axes_for(dataset_type)` falls back to a per-*type*
   claim about a whole corpus, hand-written in `DATASET_TYPE_AXES`.

The declaration wins because it is the stronger claim, and because some types
cannot be annotated at all. `bart_kspace` is the worked example: five of its
arms declare an `echo` axis and three declare `flip`, so no single table row is
true for both — and until the declared route existed, every one of those eight
arms **skipped** the one rule whose job is to consume exactly the fact they were
declaring.

`_BART_ROLE_TO_AXIS` is the single adapter between the BART role vocabulary and
`Axis`; a coverage test asserts every BART role is either mapped or excused with
a reason, so a role added later cannot map to nothing in silence. Roles no
regime states a rule about (`flip`, `map`, `repetition`) deliberately get no
`Axis` member — the member's only reader would be a rule that cannot fire, the
argument that deleted `optional_axes`. `repetition` in particular must never
become `TEMPORAL`: NEX averages of a static object are not a time series.

An arm that neither declares nor is annotated is skipped, never guessed. Note
the load-bearing distinction between `None` and `frozenset()`: the first skips,
the second is a positive "this arm carries no non-spatial axis" and **rejects**.

`check_workflow_spatial_rank` (Tier-1) is the spatial companion: it errors
when a regime's `spatial_ranks` cannot be met by the `dataset_type`'s rank —
e.g. the rank-3 `mri_functional` regime on `dataset_type: 2d`. Ranks per
`dataset_type` live in the same SSOT module; ambiguous types (e.g. `image`,
which may be 2-D slices or 3-D volumes) are unannotated and skipped.

`check_knob_applicability` (Tier-1) catches the **inert-knob trap**: a config
block enabled outside the regime it is meaningful for — an `mrf:` trajectory
block under `mri_structural`, or `data.quantitative` maps when the arm is not
mapping parameters. Scope per block is declared in
`mriforge.domain.workflows.knobs` (`KNOB_APPLICABILITY`). Only regime gating is
modelled today; modality gating ("coil arrays are meaningless for ultrasound")
lands with the first non-MR regime, so the table never carries a rule that
cannot fire.

`check_workflow_dataset_signal_domain` (Tier-1) is the **dataset-level**
companion to `check_workflow_signal_domain`. The regime-level check compares a
model against everything the *regime* can be — `mri_structural` spans
`{image, kspace, complex_image}` — so a k-space model on magnitude-only
`mrixfields` data passes it, because `kspace` is a legal structural domain in
the abstract. This check compares the model's `input_domain` against what
*this* `dataset_type` actually materialises (`DATASET_TYPE_SIGNAL_DOMAINS`),
which is the mismatch the regime-level check structurally cannot see: the
loader never produces k-space, so the model is dead on arrival.

Its escape hatch is deliberately narrow. A `pre_model` adapter (e.g.
`fft_image_to_kspace`) genuinely can bridge the domain, so an arm declaring one
skips the check — but only `pre_model`, and only when at least one step is
`enabled`. The other four adapter hooks run *after* the model has consumed its
input and so cannot bridge it; an all-disabled chain bridges nothing. Skipping
on the mere presence of an `adapters` block would let `adapters: {}` silently
disarm a hard error.

`check_workflow_component_regime` (Tier-1) catches a **regime-tagged component
declared under the wrong regime** — a `Regime.FLOW` metric such as
`mass_conservation` on an `mri_structural` arm. Only components carrying an
explicit `workflows=` tag are inspected; untagged ones are agnostic by
convention and skipped, so today it reports "nothing to match" on every
annotated arm in the corpus (none of their losses or metrics is tagged). That
is the honest state: a green result here means the declaration is
*well-formed*, not that it is *right*.

Both follow the same polarity as the axis guards — **absent is a skip, never a
rejection**. An unannotated `dataset_type`, an untagged component, or a missing
`workflow:` block all pass with `severity="info"`. Only a *present* annotation
that contradicts itself errors.

### Artifact origin

Every degradation in `DEGRADATION_REGISTRY`
(`infrastructure/physics/digital_twin_extensions`) carries an `ArtifactOrigin`
(`scanner` / `acquisition` / `patient`), so `list_degradations(origin=…)` is a
real query. The classification is completeness-enforced: the registry rebuild
indexes the origin table by name, so a new degradation added without an origin
fails at import.

**Runtime** — `mriforge.domain.workflows.enforce_pipeline_maturity_for_config`
is called at every pipeline entry (`pipelines/train.py`, `pipelines/infer.py`)
and raises `WorkflowNotImplementedError`:

- a `STUB` regime raises from *every* pipeline;
- an `EVAL_ONLY` regime raises from `train` only (`evaluate`/`predict` run).

An arm that declares no `workflow:` block is a runtime no-op — a missing
declaration is the audit's job, not the runtime's.

## Migration posture

`workflow:` is optional on `TrainingSettings` today, and a missing block is an
**advisory** audit finding rather than an error (issue #283) — decoupled from
Pydantic construction, so the field can be flipped to required once every config
declares it. The ratchet is deliberate: annotate the cohorts *first*, then raise
the severity; doing it in the other order just reddens 1,465 arms at once. The dead
`DataType` enum (which annotated no field and carried a constant `"mri"`) was
deleted in the same change.
