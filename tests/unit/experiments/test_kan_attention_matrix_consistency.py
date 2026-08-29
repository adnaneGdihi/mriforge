"""Meta-consistency tests for the KAN-vs-MLP enhancement matrix.

Confirms that the 10 arms of
``experiments/inprogress/kspace_filling/attention_enhancements/`` differ
ONLY in the intended axes (gating backend x architectural enhancement)
and share everything else — loss weights, optimizer settings, schedule,
data pipeline, etc. Without this guard, a future maintainer can
accidentally bump the learning rate in one arm and silently invalidate
the head-to-head comparison the paper depends on.

Compares the RESOLVED documents, not the YAML text. It used to compare
``yaml.safe_load`` output block-by-block, which meant it was pinning the *files*
rather than the *runs*. The 2026-08-02 canonical-key drain moved
``acceleration`` -> ``undersampling``, ``data.batch_size`` ->
``data.loader.batch_size`` and ``optimization.*`` into sub-blocks; every one of
those is a pure text move that leaves the resolved document byte-identical, and
each broke a comparison here for no behavioural reason.

Repointing the paths would have fixed today and re-broken on the next rename.
Reading the resolved settings ends the class: ``TrainingSettings.from_yaml``
normalises whatever spelling the file uses, so this guard now fires only when
two arms genuinely differ — which is the invariant the paper's head-to-head
actually rests on.

No GPU / no data needed; the configs are constructed but nothing is trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.physics.data_consistency import VALID_DC_METHODS
from tests.utils.corpus import tracked_yamls

MATRIX_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "attention_enhancements"
)

EXPECTED_ARMS = {
    "experiment_11_attn_kan_baseline",
    "experiment_11_attn_kan_sparse",
    "experiment_11_attn_kan_wavelet",
    "experiment_11_attn_kan_smap",
    "experiment_11_attn_kan_combined",
    "experiment_11_attn_mlp_baseline",
    "experiment_11_attn_mlp_sparse",
    "experiment_11_attn_mlp_wavelet",
    "experiment_11_attn_mlp_smap",
    "experiment_11_attn_mlp_combined",
}


@pytest.fixture(scope="module")
def matrix_configs() -> dict[str, dict[str, Any]]:
    """Resolve all 10 enhancement-matrix arms into a {name -> resolved dump} dict.

    Module-scoped: the arms are constructed once, not once per assertion.
    """
    return {
        path.stem: TrainingSettings.from_yaml(str(path)).model_dump(mode="json")
        for path in tracked_yamls(MATRIX_DIR, "experiment_11_attn_*.yaml", recursive=False)
    }


class TestMatrixCompleteness:
    """The matrix must have exactly the 10 expected arms."""

    def test_all_10_arms_present(self, matrix_configs):
        actual = set(matrix_configs.keys())
        assert actual == EXPECTED_ARMS, (
            f"missing: {EXPECTED_ARMS - actual}\n"
            f"unexpected: {actual - EXPECTED_ARMS}"
        )


class TestPinnedAcrossArms:
    """All 10 arms must share these settings (the paper's comparability rests on it)."""

    def _arms(self, matrix_configs):
        return list(matrix_configs.values())

    def test_loss_block_identical(self, matrix_configs):
        """All 10 arms use the same 5-term composite loss with the same weights.

        If a maintainer bumps `complex_l1` from 1.0 → 0.8 in one arm to fix
        an unrelated bug, this test fires immediately. Without it, the
        head-to-head numbers in the paper would silently mean different
        things across arms.
        """
        arms = self._arms(matrix_configs)
        ref = arms[0]["losses"]
        for arm_idx, cfg in enumerate(arms[1:], start=1):
            assert cfg["losses"] == ref, f"loss block diverged in arm index {arm_idx}"

    def test_optimizer_block_identical(self, matrix_configs):
        """Optimizer + scheduler settings pinned across arms, EXCEPT
        ``gradient_accumulation_steps`` (and the KAN LR/WD ratios, which live in
        ``model_kwargs`` and differ per gating row by design). The lighter
        ``wavelet_freq`` arms fit ``batch_size: 2`` so they use ``accum: 2``,
        while the dense-attention arms OOM at ``2`` and use ``batch_size: 1`` +
        ``accum: 4``; both reach the SAME effective batch, which is pinned
        separately by :meth:`test_effective_batch_uniform`. Comparing the raw
        accum split would flag a legitimate memory accommodation as drift."""
        arms = self._arms(matrix_configs)

        def _opt(cfg):
            block = dict(cfg["optimization"])
            gradient = dict(block["gradient"])
            gradient.pop("accumulation_steps", None)
            block["gradient"] = gradient
            return block

        ref = _opt(arms[0])
        for cfg in arms[1:]:
            assert _opt(cfg) == ref, "optimizer block diverged (beyond grad-accum)"

    def test_data_block_identical(self, matrix_configs):
        """Data pipeline pinned across arms EXCEPT ``batch_size`` — the
        ``wavelet_freq`` arms fit ``2`` while the dense-attention arms need ``1``
        (see :meth:`test_effective_batch_uniform`)."""
        arms = self._arms(matrix_configs)

        def _data(cfg):
            block = dict(cfg["data"])
            loader = dict(block["loader"])
            loader.pop("batch_size", None)
            block["loader"] = loader
            return block

        ref = _data(arms[0])
        for cfg in arms[1:]:
            assert _data(cfg) == ref, "data pipeline diverged (beyond batch_size)"

    def test_effective_batch_uniform(self, matrix_configs):
        """The scientifically-relevant quantity is the EFFECTIVE batch
        (``batch_size x gradient_accumulation_steps``); it MUST be identical
        across all arms so gradient statistics are comparable. The lighter
        ``wavelet_freq`` arms split it 2x2; the dense-attention arms 1x4 (they
        OOM at ``batch_size: 2``). Both equal 4. This is the real invariant the
        raw ``data``/``optimization`` equality checks were proxying for."""
        for name, cfg in matrix_configs.items():
            eff = (
                cfg["data"]["loader"]["batch_size"]
                * cfg["optimization"]["gradient"]["accumulation_steps"]
            )
            assert eff == 4, (
                f"arm {name}: effective batch {eff} != 4; the memory-driven "
                "batch_size/grad_accum split must preserve the effective batch."
            )

    def test_undersampling_block_identical(self, matrix_configs):
        """Was ``acceleration:``; renamed by the 2026-08-02 canonical-key drain."""
        arms = self._arms(matrix_configs)
        ref = arms[0]["undersampling"]
        for cfg in arms[1:]:
            assert cfg["undersampling"] == ref, "undersampling schedule diverged"

    def test_training_iterations_pinned(self, matrix_configs):
        """All arms run for the same iteration budget (fair wall-clock)."""
        for cfg in self._arms(matrix_configs):
            assert cfg["training"]["max_iterations"] == 30000

    def test_diffusion_block_identical(self, matrix_configs):
        arms = self._arms(matrix_configs)
        ref = arms[0]["training"]["diffusion"]
        for cfg in arms[1:]:
            assert cfg["training"]["diffusion"] == ref, "diffusion block diverged"


class TestGatingBackendAxis:
    """KAN row uses kan_adaptive ADC + KAN-friendly LR ratios; MLP row uses CNN ADC + 1.0 ratios."""

    def _kan_arms(self, matrix_configs):
        return [v for k, v in matrix_configs.items() if "_kan_" in k]

    def _mlp_arms(self, matrix_configs):
        return [v for k, v in matrix_configs.items() if "_mlp_" in k]

    def test_kan_row_dc_not_cnn_head(self, matrix_configs):
        """[2026-05 soft-DC migration] The cohort moved to ``dc_method: soft``
        cohort-wide — the DC-blob fix, since adaptive/kan_adaptive DC re-injects
        the always-sampled ACS centre (the blob enabler; enforced cohort-wide by
        ``test_kspace_filling_cohort_invariants::test_h``). DC is therefore NO
        LONGER the gating-axis differentiator; the gating axis is carried by
        ``kan_lr_ratio`` / ``gate_type`` (see ``test_kan_arms_use_lr_ratio_below_one``).
        The surviving invariant is that the KAN row must NOT smuggle the *plain
        CNN* ``adaptive`` head (the MLP control's head); ``soft`` (the migrated
        default), the KAN head, or null are all acceptable."""
        for cfg in self._kan_arms(matrix_configs):
            mk = cfg["model"]["model_kwargs"]
            assert mk["dc_method"] != "adaptive", (
                f"KAN-row arm {cfg['metadata']['name']}: dc_method='adaptive' is the "
                "MLP control's plain-CNN head; carrying it in the KAN row smuggles "
                "the control's DC across the gating axis."
            )
            assert mk["dc_method"] is None or mk["dc_method"] in VALID_DC_METHODS

    def test_mlp_row_dc_not_kan_head(self, matrix_configs):
        """[2026-05 soft-DC migration] Mirror of :meth:`test_kan_row_dc_not_cnn_head`:
        the MLP row must NOT smuggle the KAN-flavoured ``kan_adaptive`` head (which
        would inject B-spline DC params into the MLP control and break the
        head-to-head). ``soft`` (the migrated cohort default), the CNN ``adaptive``
        head, or null are acceptable."""
        for cfg in self._mlp_arms(matrix_configs):
            mk = cfg["model"]["model_kwargs"]
            assert mk["dc_method"] != "kan_adaptive", (
                f"MLP-row arm {cfg['metadata']['name']}: dc_method='kan_adaptive' "
                "injects B-spline DC params into the MLP control and breaks the "
                "head-to-head."
            )
            assert mk["dc_method"] is None or mk["dc_method"] in VALID_DC_METHODS

    def test_the_rows_actually_declare_a_dc_method(self, matrix_configs):
        """Anti-vacuity for the two guards above.

        They are now stated as *forbidden* values, which is what makes them
        stable — the pair above enumerated the ALLOWED heads instead, and went
        red across the board when the cohort moved to ``hard`` DC (48/58 arms,
        2026-07-29 soft-DC review). ``hard`` smuggles neither row's head, so the
        invariant these tests exist for was never violated; the enumeration had
        simply gone stale against a landed decision.

        The cost of a forbidden-value form is that a missing key would satisfy
        it silently, so assert the key is there and shared.
        """
        methods = {
            cfg["model"]["model_kwargs"]["dc_method"] for cfg in matrix_configs.values()
        }
        assert methods == {"hard"}, (
            f"the matrix no longer shares one DC method ({methods}); the "
            "head-to-head compares arms with different data consistency"
        )

    def test_kan_arms_use_lr_ratio_below_one(self, matrix_configs):
        """KAN B-spline coefficients need a reduced LR per the plan §3.3."""
        for cfg in self._kan_arms(matrix_configs):
            mk = cfg["model"]["model_kwargs"]
            assert mk.get("kan_lr_ratio", 1.0) < 1.0, (
                f"KAN-row arm {cfg['metadata']['name']} should have "
                f"kan_lr_ratio < 1.0"
            )

    def test_mlp_arms_use_unit_lr_ratio(self, matrix_configs):
        """MLP arms have no KAN params — leaving lr_ratio at 1.0 is the
        correct null choice (the param-group split degenerates to a single
        effective LR)."""
        for cfg in self._mlp_arms(matrix_configs):
            mk = cfg["model"]["model_kwargs"]
            assert mk.get("kan_lr_ratio", 1.0) == 1.0


class TestEnhancementAxis:
    """Each enhancement variant carries the right architectural flag."""

    def test_baseline_arms_use_kan_dual_domain_with_default_kspace_score(
        self, matrix_configs
    ):
        for label in (
            "experiment_11_attn_kan_baseline",
            "experiment_11_attn_mlp_baseline",
        ):
            cfg = matrix_configs[label]
            mk = cfg["model"]["model_kwargs"]
            assert mk["attention_type"] == "kan_dual_domain"
            kdd = mk.get("kan_dual_domain_kwargs", {}) or {}
            # baseline = no sparse, no smap
            assert kdd.get("kspace_score_fn", "softmax") == "softmax"
            assert not kdd.get("condition_on_smaps", False)

    def test_sparse_arms_use_topk_with_k8(self, matrix_configs):
        for label in ("experiment_11_attn_kan_sparse", "experiment_11_attn_mlp_sparse"):
            cfg = matrix_configs[label]
            kdd = cfg["model"]["model_kwargs"]["kan_dual_domain_kwargs"]
            assert kdd["kspace_score_fn"] == "topk"
            assert kdd["kspace_topk_k"] == 8

    def test_smap_arms_enable_smap_film(self, matrix_configs):
        for label in ("experiment_11_attn_kan_smap", "experiment_11_attn_mlp_smap"):
            cfg = matrix_configs[label]
            kdd = cfg["model"]["model_kwargs"]["kan_dual_domain_kwargs"]
            assert kdd.get("condition_on_smaps", False) is True

    def test_wavelet_arms_use_wavelet_freq_attention_type(self, matrix_configs):
        for label in (
            "experiment_11_attn_kan_wavelet",
            "experiment_11_attn_mlp_wavelet",
        ):
            cfg = matrix_configs[label]
            mk = cfg["model"]["model_kwargs"]
            assert mk["attention_type"] == "wavelet_freq"

    def test_combined_arms_have_both_sparse_and_smap(self, matrix_configs):
        for label in (
            "experiment_11_attn_kan_combined",
            "experiment_11_attn_mlp_combined",
        ):
            cfg = matrix_configs[label]
            kdd = cfg["model"]["model_kwargs"]["kan_dual_domain_kwargs"]
            assert kdd["kspace_score_fn"] == "topk"
            assert kdd["kspace_topk_k"] == 8
            assert kdd.get("condition_on_smaps", False) is True


class TestKanVsMlpDiffOnlyOnGatingAxis:
    """For each enhancement, the KAN and MLP arms must differ ONLY on the
    gating axis — NOT on enhancement, loss, optimizer, data, schedule.

    This is the key invariant the paper's per-row Δ depends on.
    """

    @pytest.mark.parametrize(
        "enhancement",
        ["baseline", "sparse", "wavelet", "smap", "combined"],
    )
    def test_paired_arms_differ_only_on_gating(self, matrix_configs, enhancement):
        kan_cfg = matrix_configs[f"experiment_11_attn_kan_{enhancement}"]
        mlp_cfg = matrix_configs[f"experiment_11_attn_mlp_{enhancement}"]

        # Same loss, optimizer, data, schedule, undersampling
        assert kan_cfg["losses"] == mlp_cfg["losses"]
        assert kan_cfg["optimization"] == mlp_cfg["optimization"]
        assert kan_cfg["data"] == mlp_cfg["data"]
        assert kan_cfg["undersampling"] == mlp_cfg["undersampling"]
        assert kan_cfg["training"]["diffusion"] == mlp_cfg["training"]["diffusion"]
        assert (
            kan_cfg["training"]["max_iterations"]
            == mlp_cfg["training"]["max_iterations"]
        )

        # Same architectural enhancement (attention_type at minimum)
        assert (
            kan_cfg["model"]["model_kwargs"]["attention_type"]
            == mlp_cfg["model"]["model_kwargs"]["attention_type"]
        ), f"attention_type diverged for enhancement={enhancement}"

        # The gating axis is realised via gating-specific kwargs (kan_lr_ratio
        # vs the unit-ratio MLP control). After the 2026-04-26 smoke audit
        # all arms route DC through the physics block (dc_method: null), so
        # the differentiator is now the LR/WD ratio knobs — not dc_method.
        kan_mk = kan_cfg["model"]["model_kwargs"]
        mlp_mk = mlp_cfg["model"]["model_kwargs"]
        assert kan_mk.get("kan_lr_ratio", 1.0) != mlp_mk.get("kan_lr_ratio", 1.0), (
            f"KAN and MLP arms for enhancement={enhancement} must differ on "
            f"the gating-axis LR ratio (the test would be vacuous otherwise)"
        )
