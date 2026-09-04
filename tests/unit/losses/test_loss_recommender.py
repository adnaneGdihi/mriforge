"""Unit tests for loss_recommender.py.

Covers: TaskAnalyzer, LossRecommender, RecommendationSet,
        LossRecommendation, get_loss_recommendation.
"""

from __future__ import annotations

import pytest

from spectramr.models.losses.loss_recommender import (
    LossRecommendation,
    LossRecommender,
    RecommendationSet,
    TaskAnalyzer,
    TaskType,
    get_loss_recommendation,
)


class TestTaskAnalyzer:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_returns_tasktype(self):
        analyzer = TaskAnalyzer()
        t = analyzer.analyze_task_type("reconstruction task")
        assert isinstance(t, TaskType)

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("desc,expected", [
        ("MRI reconstruction", TaskType.RECONSTRUCTION),
        ("segmentation of brain", TaskType.SEGMENTATION),
        ("image synthesis and generation", TaskType.SYNTHESIS),
        ("super resolution", TaskType.SUPER_RESOLUTION),
        ("denoising pipeline", TaskType.DENOISING),
        ("artifact removal", TaskType.ARTIFACT_REMOVAL),
        ("inpainting", TaskType.INPAINTING),
        ("domain translation", TaskType.TRANSLATION),
    ])
    def test_parametric_description_routing(self, desc: str, expected: TaskType):
        analyzer = TaskAnalyzer()
        t = analyzer.analyze_task_type(task_description=desc)
        assert t == expected

    @pytest.mark.parametrize("model_type,expected", [
        ("gan", TaskType.SYNTHESIS),
        ("diffusion", TaskType.SYNTHESIS),
        ("vae", TaskType.SYNTHESIS),
        ("flow", TaskType.SYNTHESIS),
        ("unet", TaskType.RECONSTRUCTION),  # fallback
    ])
    def test_parametric_model_type_routing(self, model_type: str, expected: TaskType):
        analyzer = TaskAnalyzer()
        t = analyzer.analyze_task_type(model_type=model_type)
        assert t == expected

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_none_inputs_gives_reconstruction(self):
        analyzer = TaskAnalyzer()
        t = analyzer.analyze_task_type()
        assert t == TaskType.RECONSTRUCTION

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_description_takes_priority_over_model(self):
        analyzer = TaskAnalyzer()
        # "reconstruction" in description beats model_type="gan"
        t = analyzer.analyze_task_type(
            task_description="reconstruction task", model_type="gan"
        )
        assert t == TaskType.RECONSTRUCTION


class TestLossRecommender:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_recommend_returns_recommendation_set(self):
        recommender = LossRecommender()
        result = recommender.recommend(task_type=TaskType.RECONSTRUCTION)
        assert isinstance(result, RecommendationSet)

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("task", list(TaskType))
    def test_parametric_all_task_types(self, task: TaskType):
        recommender = LossRecommender()
        result = recommender.recommend(task_type=task)
        assert result.task_type == task
        assert isinstance(result.losses, list)
        assert len(result.losses) > 0

    @pytest.mark.parametrize("use_gan", [True, False])
    def test_parametric_use_gan_override(self, use_gan: bool):
        recommender = LossRecommender()
        result = recommender.recommend(
            task_type=TaskType.SYNTHESIS, use_gan=use_gan
        )
        adversarial = [l for l in result.losses if l.category == "adversarial"]
        if adversarial:
            assert all(l.enabled == use_gan for l in adversarial)

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_config_is_generated(self):
        recommender = LossRecommender()
        result = recommender.recommend(task_type=TaskType.RECONSTRUCTION)
        assert result.suggested_config is not None

    def test_property_primary_losses_high_confidence(self):
        recommender = LossRecommender()
        result = recommender.recommend(task_type=TaskType.RECONSTRUCTION)
        for name in result.primary_losses:
            matching = [l for l in result.losses if l.name == name and l.enabled]
            assert any(l.confidence >= 0.85 for l in matching)

    def test_property_overall_confidence_in_range(self):
        recommender = LossRecommender()
        result = recommender.recommend(task_type=TaskType.SYNTHESIS)
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_property_notes_are_strings(self):
        recommender = LossRecommender()
        result = recommender.recommend(task_type=TaskType.DENOISING)
        assert all(isinstance(n, str) for n in result.notes)

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_use_perceptual_false(self):
        recommender = LossRecommender()
        result = recommender.recommend(
            task_type=TaskType.SUPER_RESOLUTION, use_perceptual=False
        )
        perceptual = [l for l in result.losses if "perceptual" in l.name]
        if perceptual:
            assert all(not l.enabled for l in perceptual)

    def test_edge_empty_primary_losses_if_all_low_confidence(self):
        """Summary method should not raise even with empty primaries."""
        recommender = LossRecommender()
        result = recommender.recommend(task_type=TaskType.DENOISING)
        summary = result.summary()
        assert isinstance(summary, str)

    # ── Raises (no silent fallback for unknown task_type) ────────────────────

    def test_edge_missing_task_type_defaults_via_analyzer(self):
        """No task_type → analyzer derives one; must not raise."""
        recommender = LossRecommender()
        result = recommender.recommend(task_description="reconstruction pipeline")
        assert isinstance(result, RecommendationSet)


class TestGetLossRecommendation:

    def test_canary_functional_interface(self):
        result = get_loss_recommendation(task_type=TaskType.RECONSTRUCTION)
        assert isinstance(result, RecommendationSet)

    def test_parametric_task_description(self):
        result = get_loss_recommendation(task_description="denoising MRI images")
        assert result.task_type == TaskType.DENOISING


class TestRecommendationSet:

    def test_overall_confidence_empty_is_zero(self):
        rs = RecommendationSet(
            task_type=TaskType.RECONSTRUCTION,
            losses=[],
            primary_losses=[],
            auxiliary_losses=[],
            optional_losses=[],
            suggested_config=None,
            notes=[],
        )
        assert rs.overall_confidence == 0.0

    def test_summary_with_notes(self):
        rec = LossRecommendation(
            name="l1", enabled=True, weight=1.0,
            reason="primary", category="reconstruction", confidence=0.9
        )
        rs = RecommendationSet(
            task_type=TaskType.RECONSTRUCTION,
            losses=[rec],
            primary_losses=["l1"],
            auxiliary_losses=[],
            optional_losses=[],
            suggested_config=None,
            notes=["Note A"],
        )
        summary = rs.summary()
        assert "reconstruction" in summary.lower()
        assert "Note A" in summary
