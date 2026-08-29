"""F13 regression — score-SDE / consistency must save CLEAN reverse-diffusion output.

Audit anchor: TODO/audit/smoke_audit_20260516.md §F13.

Pre-F13 (2026-05-16 smoke), ``DiffusionTrainingStrategy._generate_validation_prediction``
fell through to a path that did:

    noisy_input = self.q_sample(model_input, timestep, torch.randn_like(model_input))
    hr_fakes = _forward_chunked(noisy_input, timestep, ...)

For a ``ScoreBasedDiffusionGenerator`` whose ``forward`` returns predicted
noise (``return self._model_forward(x_t, t)``), this saved pure-noise tensors
as ``metrics/fake_images/*.png`` (V2 in the 2026-05-16 mosaic — persisted
after the F12 mtime filter). For a ``ConsistencyModelGenerator``, the
forward call with ``t=high_noise_level`` produced a high-noise approximation
rather than a clean reconstruction.

F13 dispatches to the generator's dedicated sampler before falling back:

- ``multistep_inference`` (consistency models) — 2-step clean denoising.
- ``generate`` (score-based) — runs ``p_sample_loop`` reverse SDE.
- legacy noisy_input + forward — fallback for non-diffusion generators.

These tests pin the source so a future "tolerant" refactor cannot silently
re-introduce the noised-output bug.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Path corrected 2026-05-24: src→mriforge refactor moved the package under
# src/mriforge/; the stale src/infrastructure/ path FileNotFoundError'd and
# silently disabled the F13 dispatch checks. See smoke_audit_20260524.md.
DIFFUSION_PY = REPO_ROOT / "src" / "mriforge" / "infrastructure" / "training" / "strategies" / "diffusion.py"


def _source() -> str:
    return DIFFUSION_PY.read_text()


def test_f13_dispatches_to_multistep_inference_for_consistency_models() -> None:
    """Consistency-model branch must call ``multistep_inference`` BEFORE falling back."""
    src = _source()
    assert "hasattr(gen, \"multistep_inference\")" in src, (
        "F13 wiring missing: the validation-prediction dispatch must check "
        "for ``multistep_inference`` (consistency models) before falling "
        "back to the noisy-forward path. Otherwise consistency-model "
        "validation PNGs are high-noise-level approximations, not clean "
        "reconstructions. See TODO/audit/smoke_audit_20260516.md §F13."
    )
    assert "gen.multistep_inference(" in src, (
        "F13 wiring missing: the consistency-model branch detected "
        "``multistep_inference`` but never calls it. The dispatch must "
        "actually invoke the method to produce a clean denoised image."
    )


def test_f13_dispatches_to_generate_for_score_based_diffusion() -> None:
    """Score-based diffusion branch must call ``generate`` (p_sample_loop)."""
    src = _source()
    assert "hasattr(gen, \"generate\")" in src, (
        "F13 wiring missing: the validation-prediction dispatch must check "
        "for ``generate`` (score-based diffusion) before falling back to "
        "the noisy-forward path. Without this, score-based validation "
        "saves predicted noise as fake_images (V2 in the 2026-05-16 "
        "smoke mosaic). See TODO/audit/smoke_audit_20260516.md §F13."
    )
    assert "gen.generate(model_input)" in src, (
        "F13 wiring missing: the score-based branch detected ``generate`` "
        "but never calls it. The dispatch must run p_sample_loop to "
        "produce a clean reverse-diffusion image."
    )


def test_f13_falls_back_to_noisy_forward_for_legacy_generators() -> None:
    """Non-diffusion generators (no sampler method) keep the legacy path."""
    src = _source()
    # The legacy fallback must still exist — it handles non-diffusion
    # generators that route through DiffusionTrainingStrategy via the
    # ``training_mode: diffusion`` YAML key.
    assert "noisy_input = self.q_sample(" in src, (
        "F13 must preserve the legacy noisy_input fallback for generators "
        "that have neither ``multistep_inference`` nor ``generate``. "
        "Removing the fallback breaks any generator that genuinely expects "
        "a noisy single-step forward."
    )


def test_f13_dispatch_ordering_consistency_before_score_before_fallback() -> None:
    """The dispatch order matters — consistency check first, score-based next,
    fallback last. A consistency model that also exposes ``generate`` (for
    legacy reasons) must hit the multistep_inference branch, not the
    score-based one.
    """
    src = _source()
    idx_consistency = src.find("hasattr(gen, \"multistep_inference\")")
    idx_score = src.find("hasattr(gen, \"generate\")")
    idx_fallback = src.find("noisy_input = self.q_sample(")
    assert idx_consistency > 0, "consistency check missing"
    assert idx_score > 0, "score-based check missing"
    assert idx_fallback > 0, "fallback missing"
    # Multiple q_sample occurrences exist; we want the one INSIDE the F13
    # else-branch, which is the latest in the file (the latent-diffusion
    # branch also has one but earlier).
    # The relative ordering of the three checks WITHIN the new F13 block
    # is what matters; find the last q_sample which should be the legacy
    # fallback in the else-branch.
    last_q_sample = src.rfind("noisy_input = self.q_sample(")
    assert idx_consistency < idx_score < last_q_sample, (
        "F13 dispatch ordering broken. Expected: multistep_inference check "
        "→ generate check → legacy noisy fallback. Reordering risks "
        "consistency models taking the score-based code path "
        "(``generate`` may exist on both via mixin inheritance)."
    )


def test_f13_audit_anchor_documented_in_source() -> None:
    """The F13 patch carries an inline anchor pointing back to the audit doc."""
    src = _source()
    assert "F13 (2026-05-17 round 5)" in src, (
        "F13 fix should anchor itself to the audit doc with a comment "
        "naming the round, so future contributors can trace the rationale "
        "via git-blame → audit doc. See "
        "TODO/audit/smoke_audit_20260516.md §F13."
    )
    assert "predicted noise" in src or "V2" in src, (
        "F13 inline comment should explicitly mention the symptom "
        "(predicted-noise output → V2 pure-noise visualization) so the "
        "rationale is obvious without consulting the audit doc."
    )
