:orphan:

.. _sim2rank-reliability-theory:

=====================================================================
sim2rank Reliability Theory --- Proof-Carrying Account of the Rankers
=====================================================================


.. contents:: On this page
   :local:
   :depth: 2

This page is the Sphinx-reachable companion to the formal-proofs paper
(source: ``docs/sim2rank_reliability_theory.tex``, build with
``tectonic sim2rank_reliability_theory.tex``). The paper proves that the
``sim2rank`` meta-evaluation pipeline *reliably sorts image-quality
metrics (IQMs) by quality along each degradation axis*, for **every
ranker generation the codebase ships** --- not just the active ones.

What the paper proves
=====================

The reliability claim is decomposed into three independently-checkable
layers: **(L1) order-preservation** (each score is a monotone functional
of a well-defined population quality), **(L2) estimator consistency**
(the empirical score concentrates on that value, with an explicit
exponential rate where the statistic is a bounded U-statistic), and
**(L3) aggregation soundness** (the cross-axis combiners never overturn a
unanimous per-axis verdict and penalize single-axis blind spots).

Beyond the original three layers, four machine-checked extensions close
the gaps a literature audit (2026-05-29/30) found *beyond* the internal
``TODO/sim2rank_critique.md``. Each is a sorry-free Lean module under
``docs/lean/Sim2Rank/Sim2Rank/`` whose theorems depend only on the
standard axioms (``propext``, ``Classical.choice``, ``Quot.sound``):
**L2⁺** betting / anytime-valid confidence (``Betting.lean``), **L3⁺**
Kemeny--Young Condorcet-consistent aggregation (``Kemeny.lean``), **L4**
sim-to-real ranking transfer (``Transfer.lean``), and **L5** conformal
risk control on the selected metric (``Conformal.lean``).

L2⁺ --- betting / e-value anytime-valid confidence
--------------------------------------------------

L2 is a Hoeffding bound (``2·exp(−mη²/2)``): non-adaptive and not
anytime-valid (the severity sweep cannot stop early without invalidating
the guarantee). The state of the art for bounded means is the
**betting / game-theoretic** confidence sequence (Waudby-Smith & Ramdas,
*JRSS-B* 2024; Ramdas et al., *Statist. Sci.* 2023): a player bets on the
centered concordance increments, the **wealth process**
``K_t = ∏_{s<t} (1 + λ_s (X_s − η))`` is a nonnegative supermartingale
under the null, and **Ville's inequality** turns it into a
variance-adaptive, time-uniform confidence sequence enabling principled
early stopping. ``Betting.lean`` proves sorry-free the wealth-process
algebra (``wealth_zero``/``wealth_succ``/``wealth_nonneg``), a finite
**Markov inequality** (``finite_markov``) and the **e-value test
validity** it yields (``evalue_test_valid``: an e-value with
:math:`\mathbb{E}[K] \le 1` satisfies :math:`\mathbb{P}(K \ge 1/\alpha)
\le \alpha`), and the fixed-time betting test
(``betting_test_valid_fixed``). The **anytime-valid** upgrade
(``betting_cs_anytime_covers``) is proved *from* an explicit typed
hypothesis ``hville`` (Ville's maximal inequality for a nonnegative
supermartingale): Mathlib v4.29.1 ships only Doob's *submartingale*
``maximal_ineq``, so the lone analytic input is pinned to one standard,
citable lemma rather than buried in a ``sorry``.

L3⁺ --- Kemeny-median cross-axis aggregation (Condorcet consistency)
-------------------------------------------------------------------

The L3 aggregators (Minimax--Borda, z-standardised) are only
**Pareto**-consistent: a metric weakly dominant on every axis is ranked
no lower. The self-critique (``TODO/sim2rank_critique.md`` §1) flags this
as an Arrow/IIA weakness and proposes Schulze, which lacks a
maximum-likelihood characterization and is not machine-checked. **L3⁺**
installs the **Kemeny--Young** consensus (uniquely neutral + consistent +
Condorcet by Young--Levenglick 1978, and the Mallows MLE by Young 1988)
and machine-checks its defining property in ``Kemeny.lean``. Representing
a ranking by real positions ``r : C → ℝ`` (lower = better) and the
majority margin by an antisymmetric ``w``, the Kemeny agreement score is
``kemenyScore w r = ∑_{a ≠ b, r a < r b} w a b``. The exchange lemma
``kemeny_moveTop_gt`` shows that if ``c`` is a Condorcet winner and a
ranking fails to place ``c`` first, moving ``c`` below everyone *strictly*
raises the score; hence ``kemeny_optimal_ranks_condorcet_first`` proves
any Kemeny-optimal ranking places a Condorcet winner first. Bridging to
L3, ``margin_pos_of_unanimous`` shows a unanimous per-axis preference
gives a strictly positive margin, so the capstone
``kemeny_optimal_ranks_pareto_winner_first`` certifies a Pareto winner is
ranked **first** --- strictly stronger than ``borda_pareto``.

L4 --- sim-to-real ranking transfer (external validity)
-------------------------------------------------------

L1--L3 certify the ranking against the *simulated* law. **L4** closes the
construct-validity gap the MRI image-quality literature identifies (Mason
*et al.*, *IEEE TMI* 2020; the "reassess FR-IQA for medical images"
line): a metric that tracks the simulator's severity knob need not track
the *real* acquisition law. Working in the pipeline's finite probability
space (content cells :math:`\times` severity grid), a law is a
probability vector ``p`` and the population value of a bounded concordance
kernel ``h`` is the U-statistic functional ``pmean h p``. The lemma
``pmean_diff_le_tv`` proves the **bounded-functional transfer bound**

.. math::

   |\,\mathrm{pmean}\,h\,p - \mathrm{pmean}\,h\,q\,| \;\le\; 4\,C\,\mathrm{TV}(p,q)
   \qquad (|h|\le C),

via the product split :math:`p_i p_j - q_i q_j = (p_i-q_i)p_j + q_i(p_j-q_j)`
and :math:`\sum p = \sum q = 1` --- only ``Finset`` sums, no measure
theory. ``ranking_transfer`` then certifies that if the digital twin is
within total-variation budget :math:`\mathrm{TV}(p,q) < \Delta/8` of the
real law, the **real-law ranking equals the simulated ranking**;
``ranking_transfer_recovers_sim`` is the consistency check (at
:math:`\mathrm{TV}=0` it reduces to ``Concentration.ustat_order_recovery``,
so L4 is a strict generalization). Chaining L2 (empirical-sim
:math:`\to` population-sim) with L4 (population-sim :math:`\to`
population-real) gives an end-to-end guarantee whenever the TV certificate
holds.

L5 --- conformal risk control for the selected metric
-----------------------------------------------------

L1--L4 certify the *ranking*; **L5** bounds the *risk of the metric
actually deployed*. Conformal Risk Control (Angelopoulos, Bates, Fisch,
Lei, Schuster, *ICLR* 2024) chooses a threshold so the expected
diagnostic loss of the selection is controlled at level ``α`` with a
finite-sample ``(n+1)α/n`` bound and no distributional assumptions beyond
exchangeability. ``Conformal.lean`` machine-checks the deterministic
finite-sample core sorry-free: ``crc_full_sample_mean_le`` (the full
calibration-plus-test mean clears ``α``), ``crc_calibration_mean_le`` (the
calibration mean is :math:`\le (n+1)\alpha/n`), and ``crc_threshold_upset``
(well-posedness: the admissible-threshold set is an up-set for a
nonincreasing loss). The headline ``conformal_risk_control`` assembles the
population bound :math:`\mathrm{risk} \le (n+1)\alpha/n \to \alpha` from a
single typed exchangeability hypothesis ``hexch`` --- the lone
probabilistic input, supplied by the symmetry of i.i.d./exchangeable data.
Composed with L4 this yields the regulatorily-meaningful **validation
badge**: a metric whose sim-ranking transfers to real *and* whose deployed
diagnostic risk is distribution-free-controlled.

K-RCPS --- entrywise conformal risk control
-------------------------------------------

The L5 selector above certifies a *single scalar* threshold. **K-RCPS**
(Teneggi *et al.*, ICML 2023) extends this to *entrywise* (per-pixel /
per-region) risk control for image reconstruction:
``spectramr.core.metrics.meta_evaluation.krcps.krcps_calibrate`` calibrates the
smallest scale :math:`\hat\beta` over a supplied per-entry width profile
:math:`w` (e.g. a heteroscedastic uncertainty map, or a K-group-constant
profile) such that the marginal miscoverage loss
:math:`\ell_i(\beta) = \tfrac1D\sum_d \mathbf 1[|r_{i,d}| > \beta w_d]` clears the
same CRC rule :math:`(\sum_i\ell_i + B)/(n+1)\le\alpha`. The calibrated entrywise
interval half-width is :math:`\hat\beta\,w_d`. Because :math:`\ell_i(\beta)` is
nonincreasing in :math:`\beta`, the admissible set is an up-set
(``crc_threshold_upset``) and the calibration *reuses* ``select_conformal_threshold``
verbatim --- so K-RCPS inherits the machine-checked :math:`(n+1)\alpha/n` bound.
**Corner case:** a uniform width profile (:math:`w_d\equiv1`) reduces K-RCPS to
the scalar selector, so it provably generalises scalar RCPS. This is RCPS with an
entrywise width profile plus the CRC marginal guarantee; the convex
interval-*volume* minimisation of Teneggi 2023 (optimising :math:`w` rather than
supplying it) and the validation-pipeline ``ConformalConfig`` wiring are deferred
increments (a config knob with no consumer would be a dead-knob facade, pitfall
#15).

In-domain IQA recalibration (SOTA plan T5, exploratory)
-------------------------------------------------------

Zero-shot foundation IQA (CLIP-IQA / Q-Align) has no radiologist-anchored MRI
validation (the REJECTed P4.4), so
``spectramr.core.metrics.meta_evaluation.iqa_recalibration`` recalibrates an
existing score (a foundation IQA value or one of the ~41 NR-battery metrics) onto
MRIQC- / radiologist-anchored labels via a **monotone** map — ``platt`` (affine)
or ``isotonic`` (pool-adjacent-violators) — and *accepts deployment only if* the
held-out Kendall-τ clears a preregistered threshold (``accept_recalibration``).
**Corner case:** the identity map recovers the failing zero-shot metric, so a
non-trivial fit clearing the τ gate is exactly what justifies the metric. It is
**exploratory**: the acceptance decision needs an anchored label set, which is not
shipped — the utility is a tested calibration tool, not a validated metric.

Degradation model (so the axes are concrete)
--------------------------------------------

The paper first fixes the degradation operators the proofs condition on:

* **Eight single-parameter families** (``simulator.py``) with exact
  severity schedules --- Gaussian/Rician noise
  (``sigma(theta)=max(0.3*theta*sigma_x, 0.05*theta)``), Gaussian blur
  (``4*theta+1e-3`` px), k-space undersampling
  (``keep(theta)=max(0.05, 1-theta)``), motion (PE-line random phase
  ramps over ``0.5*theta`` of lines), bias field, gamma, and contrast ---
  sampled on a deterministic **Halton / van der Corput** severity grid.
* **The Orthogonal Degradation Matrix (ODM)** (``odm.py``): three
  physics-anchored k-space axes --- **A thermal** (SNR 40 -> 2 dB),
  **B motion** (PE-direction phase; the wired default is the *chirped*
  blur+ghost hybrid, stated honestly), **C aliasing** (R = 1 -> 12,
  shrinking ACS) --- reconstructed by the SENSE adjoint. Their
  orthogonality (disjoint action on the noise term, the phase of
  ``F x``, and the sampling mask ``M`` of ``y = M F S x + n``) is a
  **modeling premise**, not a theorem.

Theorems, by generation
-----------------------

* **Per-axis core** --- ADR (monotone discriminability), the Kendall /
  Spearman concentration theorem (rate
  ``P(|tau_hat - tau| >= eta) <= 2 exp(-floor(n/2) eta^2 / 2)``), SCVR
  (two-way ANOVA F-ratio).
* **Active composite** --- CDRS (clean-stability x cross-family
  monotone-consistency, with a proved single-axis non-monotonicity
  hazard) and the **Sim2Rank** min--max fusion (consensus-preserving,
  affine-scale-free, outlier-fragile).
* **Spectral / manifold ensemble** --- LGDR, SFA, ESD, CDSCR, FSPD.
* **Clinical calibrator** --- HiBSL (ordered-probit).
* **Causal / sufficiency generation** --- DCMS (mediation
  proportion-mediated) and ACSS (Shah--Peters Generalised Covariance
  Measure sufficiency test).
* **Legacy five-method suite** --- MS\ :sup:`3`, KSG MI,
  task-defensibility, Sobol pick-freeze indices, Bradley--Terry /
  sign-agreement d'.
* **Aggregators** --- Minimax--Borda and z-standardized weighting.

Running the layers (CLI + dispatcher)
=====================================

The modern entry point is ``spectramr meta-evaluate``. Two run-shape knobs select
the cell of the {2d,3d} x {real,betting} matrix; each is read, validated, and
stamped into ``summary.json`` (CLAUDE.md pitfall #15):

* ``--betting`` (optional ``--betting-alpha 0.05``) --- also emit the **L2⁺**
  variance-adaptive, anytime-valid betting partial order
  (``betting_cs_anytime_covers``) alongside the always-on Hoeffding-powered
  gate. Omitting it is the *real* powered-only implementation. The result lands
  in ``summary.json["betting"]``.
* ``--eval-mode {2d,3d}`` --- evaluation dimensionality. ``2d`` is the per-slice
  path that ships; ``3d`` is validated by ``resolve_eval_mode`` and **raises**
  ``NotImplementedError`` at startup (the volumetric engine is deferred to
  ``TODO/backlog_sim2rank_3d_eval.md``), never silently degrading to 2-D. The
  resolved value is stamped into ``summary.json["eval_mode"]``.

The matrix is dispatched as a SLURM array job by
``scripts/sim2rank/submit_sim2rank_matrix.sbatch`` with **one task per
dimensionality** (task 0 = ``2d``, task 1 = ``3d``). Each task runs *both*
certification variants of its ``eval_mode`` and then **compares them together**
via ``scripts/sim2rank/compare_meta_eval.py`` (the pure core is
``meta_evaluation.compare_summaries``), writing
``<out>/<mode>_comparison/comparison.{json,md}``. Because betting is additive,
the real and betting arms of one ``eval_mode`` must have identical rankings
(Spearman ``== 1``); the comparison asserts that and reports the directed pairs
the L2⁺ betting order certifies beyond the pooled Hoeffding gate. The ``3d`` task
exercises the fail-loud guard: both arms raise, the comparison reports
"both arms missing", and the task exits non-zero. Use ``--array=0`` for the
2-D job only. See ``docs/sim2rank_empirical_and_roadmap.md`` (sections 3.2, 3.5)
for the proof-correspondence of each run.

.. code-block:: bash

   # one dimensionality, both certification variants, compared together:
   spectramr meta-evaluate --input <m4raw_dir> --eval-mode 2d \
       --out results/2d_real/summary.json                  # real (powered only)
   spectramr meta-evaluate --input <m4raw_dir> --eval-mode 2d --betting \
       --out results/2d_betting/summary.json               # + anytime-valid betting
   # 3d is the same command with --eval-mode 3d; the two summary.json files are
   # directly comparable, since both carry per-metric scores under the same keys.
   spectramr meta-evaluate --input <m4raw_dir> --eval-mode 3d \
       --out results/3d_real/summary.json

Comprehensive run, unified degradations, and generations
========================================================

.. note::

   ``spectramr meta-evaluate`` is the entry point this distribution ships. The
   maintainers additionally drive the framework from a SLURM batch pipeline
   configured through ``SIM2RANK_*`` environment variables; that pipeline is not
   distributed, and **the distribution reads only** ``SIM2RANK_MODE`` **of those
   names**. Where this page describes a sweep in terms of those variables, read
   it as a description of the study that was run, not as knobs you can set here.

The ``meta-evaluate`` CLI above is the lightweight modern entry point. The
**comprehensive** sim2rank run is ``scripts/sim2rank/run_full_pipeline.sbatch``,
which sweeps **all four generations** of rankers side-by-side — **Gen 1**
(ADR/SCVR/Borda) and **Gen 2** (MS³/MI/Task/Sobol/BT) are always-on; **Gen 3**
(the six structural rankers) and **Gen 4** (BT-LL/DCMS/HiBSL/ACSS) are opt-in
via ``SIM2RANK_GEN3`` / ``SIM2RANK_GEN4``. (A flag-only search misleadingly makes
Gen 2 look "missing" — it is dispatched unconditionally.)

Two knobs bring the option-B betting layer and the unified degradation set into
that comprehensive run, both consumed by the ``--novel-meta`` pipeline:

* ``SIM2RANK_BETTING=1`` (``--betting``) — emit the L2⁺ betting certification on
  the novel-meta aggregate (option B), the legacy-pipeline counterpart of
  ``meta-evaluate --betting``. Stamped into ``novel_meta_eval_results.json``.
* ``SIM2RANK_UNIFIED_DEGRADATIONS=1`` (default; ``--no-unified-degradations`` to
  opt out) — the novel-meta sweep runs the **unified** degradation set: the
  physics ``DEGRADATION_REGISTRY`` SSOT (28 ops) plus the photometric
  ``gamma`` / ``contrast`` = **30 operators, one implementation per
  degradation**, instead of the core simulator's 8 self-contained families
  (which re-implemented 6 registry degradations under different conventions).
  The merge respects the inward-only dependency rule via the
  ``SimulatorConfig.operator_library`` injection seam (the core layer cannot
  import ``infrastructure/physics`` directly).

.. code-block:: bash

   # The distributed entry point is the CLI above. --betting turns on the
   # anytime-valid betting certification; --core-degradations restricts the
   # sweep to the core axes rather than the full bank.
   spectramr meta-evaluate --input <m4raw_dir> --betting --core-degradations \
       --out results/comprehensive/summary.json

Unified ranking module
======================

The rankers are now **one module, one contract, one registry**
(``spectramr.core.metrics.meta_evaluation.rankers``): every ranker is a
``BaseRanker`` (``rank(metric_set, dataset) -> RankingResult``) registered via
``@register_ranker(name, generation, requires_likert, requires_task_net)``. The
``generation`` tag is lineage metadata, **not** a "legacy vs novel" pipeline
fork — that split is gone. The 17 registered rankers:

* **Gen 1** (statistical): ``adr``, ``scvr``, ``cdrs`` — expose the per-metric
  statistics the ``Sim2Rank`` fusion ranker already computes (faithful reuse).
* **Gen 2** (information / sensitivity): ``ms3``, ``mi`` (KSG), ``sobol``
  (SROCC-entropy surrogate), ``bt`` (sign-agreement d′), ``task`` — ported
  faithfully from ``meta_eval.py`` (each pinned by an ``atol≤1e-9`` parity test
  against the original).
* **Gen 3** (structural): ``cdscr``, ``lgdr``, ``sfa``, ``esd``, ``fspd``,
  ``sim2rank``.
* **Gen 4** (clinical / causal): ``dcms`` (needs a task network), ``acss`` /
  ``hibsl`` (need radiologist Likert). These are **capability-gated** —
  ``available_rankers(have_likert, have_task_net)`` omits them when the external
  ground truth is absent, so on M4Raw (no Likert, no task-net) **14 rankers run
  and the 3 Gen-4 rankers are skipped**, never silently fabricated.

``MetaEvaluationPipeline.from_registry()`` resolves the available rankers — the
wired entry point that replaced the hard-coded list. The novel-meta path on
M4Raw uses it, so the leaderboard is the consensus of all 14 dataset-derivable
rankers.

Simulated → real transfer (L4 badge)
====================================

``SIM2RANK_TRANSFER_BADGE=1`` (``--transfer-badge``; requires
``INPUT_MODE=m4raw`` + ``NOVEL_META=1``) certifies that the **simulated** metric
ranking transfers to the **real** M4Raw acquisition law. M4Raw provides the real
law for free: each (subject, contrast) has multiple real acquisitions, so a
single repetition is a real *degraded* image and the multi-rep pseudo-GT is the
*clean* reference. The metric values on those real pairs estimate the real
score-law; ``transfer_certificates`` issues the per-pair ``TV < Δ/8`` badge
(Lean ``Sim2Rank.Transfer.ranking_transfer``) into ``novel_meta_eval_results.json``.
A transfer badge on ``--synthetic`` input fails loud (sim-vs-sim is trivial).

**Crash→NaN robustness.** ``precompute_metric_values`` (and the real-cohort
builder ``_build_real_metric_values``) record a metric that raises on a sample as
``NaN`` so one broken metric never aborts the whole 27k-call evaluation; the
contract is that *downstream consumers mask non-finite values*. The score-law TV
estimator ``transfer.tv_distance_samples`` therefore drops ``NaN``/``±inf`` before
binning (a raw ``int(NaN)`` raises, and a raw ``min``/``max`` over a list with a
``NaN`` returns an order-dependent bound that poisons the bin width). A metric that
crashed on its *entire* real cohort has no estimable TV and is **skipped** by
``transfer_certificates`` — never allowed to crash the run, nor (since the pooled
``TV`` is a worst-case ``max`` over metrics) to spuriously void every transfer
certificate. A partly-crashed metric still contributes its finite-subsample TV.

The same contract is now enforced across **every** consumer of the per-metric
values, not just the L4 estimator (the audit that followed the ``transfer``
crash hardened the remaining gaps):

* ``RankingResult.standardize`` (the z-scoring chokepoint every ranker flows
  through) standardises over the finite scores only and *omits* a non-finite
  one. Previously a single ``NaN`` score made ``mean``/``std`` ``NaN`` — and
  since ``NaN < 1e-12`` is False — poisoned **every** metric's z-score; an
  omitted metric is simply treated as "absent on this axis" by the aggregator.
* ``BaseRanker._ranks_from_scores`` ranks finite scores descending and places
  non-finite ones **last** (tie-broken by name). The old ``sorted(key=-score)``
  left a ``NaN``-scored (crashed) metric wherever it sat in dict order — so a
  crashed metric could surface as rank #1.
* ``powered.certify_partial_order`` requires a **finite** gap before certifying
  a pair: an ``inf`` score gives ``abs(inf) >= threshold == True`` and would
  otherwise spuriously certify the crashed metric as the better one.
* ``comparison.spearman_rank_correlation`` drops metrics with a non-finite score
  on either side before ranking, so a crashed metric cannot fake a perfect
  ``rho = 1.0`` rank agreement.
* ``conformal`` (``crc_admissible`` / ``conformal_metric_selection`` /
  ``select_conformal_threshold``) calibrates over the finite losses only and
  reports a ``NaN`` ``calib_mean``/``risk_bound`` when none are defined.
* ``sfa._cosine`` returns a neutral ``0.0`` on a non-finite norm or result (a
  metric that produced a ``NaN`` gradient/score vector).

Rankers consuming ``_per_family_average`` (``ms3``, ``bt``, ``sobol``) need no
guard: that helper already maps an all-``NaN`` ``(family, severity)`` cell to
``0.0`` (``denom.clamp(min=1.0)`` + ``where(isfinite, t, 0)``), and the wired
betting increments are signs in ``{-1, 0, +1}`` (finite by construction).

Upstream cause — ``requires_complex`` is now wired
--------------------------------------------------

Tracing *why* metrics went ``NaN`` (re-running each on a representative pair, since
the crash→NaN catch discards the reason) showed the dominant cause was an **input
contract mismatch**, not bad math: M4Raw is complex k-space, reconstructed via
``ifft2c`` + ``coil_combine_sense`` (``infrastructure/physics/m4raw_pseudo_gt.py``),
but the per-metric loop fed every metric the **magnitude** image. The phase/complex
metrics (``complex_hfen``, ``ipen``, ``phase_mse``) declare ``requires_complex`` on
their ``MetricSpec`` — a flag that was **declared but read nowhere** (pitfall #15).
``complex_hfen`` raised ``imag is not implemented`` → ``NaN``; worse, ``ipen`` /
``phase_mse`` returned *plausible but meaningless* values (the phase of a real image
is identically zero).

The flag is now honoured:

* **Real cohort** (``_build_real_metric_values``, the L4 transfer path): the complex
  coil-combined recon — already in hand as ``synthesize_pseudo_gt``'s ``coil_images``
  + ``smaps`` — is recombined via the physics SSOT (``_complex_recon_on_clean_scale``,
  p99-normalised to match the magnitude scale) and fed to ``requires_complex``
  metrics; everything else still gets magnitude. **Reference and degraded share one
  coil operator:** the reference is the NEX merge (complex k-space average, √N SNR)
  of the multi-rep group, ESPIRiT (``estimate_csm_espirit``) is estimated **once**
  from that high-SNR merged k-space, and those maps are **reused** for each single-rep
  degraded acquisition (``synthesize_pseudo_gt(..., smaps=ref_maps)``). So the
  comparison isolates the noise/averaging degradation, not a per-acquisition
  recon-basis shift, and a single noisy rep never drives its own poor map estimate.
* **Simulated cohort** (``precompute_metric_values``): the simulated clean references
  are magnitude pseudo-GT (no phase), so ``requires_complex`` metrics there are
  recorded as ``NaN`` with a **logged** reason (honest skip), and run on complex only
  if the cohort actually carries complex tensors. Genuinely *valuing* phase metrics in
  the simulated sweep would require complex clean references + a complex-domain
  degradation sweep (a change to the simulated law), which is intentionally **not**
  done here.

Model-backbone metrics (``lpips``, ``dists``, ``fid``, ``kid``, ``inception_score``)
are grayscale→RGB adapted (``channel_adapter.adapt_to_rgb`` / a ``repeat(1, 3, …)``
+ ImageNet-norm path) and constructed with ``device=`` so VGG/Inception run on GPU
(~10 ms/call vs ~7 s on CPU); the CPU-only "hangs" seen during tracing (``cw_ssim``,
``dists``, ``med_fid``, ``mmd_metric``) do not occur on the cluster's CUDA run.

.. code-block:: bash

   # the unified rankers + betting, on real M4Raw:
   spectramr meta-evaluate --input databases/m4raw/data/multicoil_train \
     --max-subjects 8 --device cuda --betting \
     --out results/m4raw_unified/summary.json

Reproducing the build
=====================

.. code-block:: bash

   cd docs
   tectonic sim2rank_reliability_theory.tex   # -> sim2rank_reliability_theory.pdf

The Lean extensions build with ``lake build Sim2Rank`` from
``docs/lean/Sim2Rank``; the structural regression tests
``tests/unit/docs/test_sim2rank_{transfer,kemeny,betting,conformal}_layer.py``
pin each layer's theorem inventory and sorry-free status, and
``tests/unit/docs/test_sim2rank_reliability_theory.py`` pins the paper's
own structural invariants.
