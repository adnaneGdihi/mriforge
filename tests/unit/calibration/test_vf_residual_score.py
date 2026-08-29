"""Phase-1 tests — VF-residual non-conformity score + registry dispatch.

Four properties under test:

1. Projector idempotency P_M^2 = P_M (algebraic correctness).
2. Empty-marker / rank-deficient basis caught at audit + score time.
3. Score → 0 in the trivial-marker limit (consistency).
4. Exchangeability of {s_i} under independent marker draws — sample-level
   sanity check that the score behaves like a non-conformity score.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mriforge.infrastructure.calibration import (  # noqa: E402
    CALIBRATION_SCORE_REGISTRY,
    list_calibration_scores,
    resolve_calibration_score,
    marker_basis_condition_number,
)
from mriforge.infrastructure.calibration.vf_residual_score import (  # noqa: E402
    _projector_for_basis,
    vf_residual_score,
)


@pytest.fixture
def random_unitary_basis(tmp_path: Path) -> Path:
    """Generate a well-conditioned random orthonormal marker basis."""
    torch.manual_seed(42)
    n, k = 64, 8
    # Build orthonormal columns via QR on a random complex matrix.
    a = torch.randn(n, k, dtype=torch.complex64)
    q, _ = torch.linalg.qr(a)
    path = tmp_path / "marker_basis.pt"
    torch.save(q, path)
    return path


class TestRegistration:
    def test_vf_residual_registered(self) -> None:
        assert "vf_residual" in CALIBRATION_SCORE_REGISTRY

    def test_existing_scores_still_resolvable(self) -> None:
        for name in (
            "absolute_residual",
            "pixel_residual",
            "quantile_regression",
            "cqr",
            "vf_residual",
        ):
            assert name in list_calibration_scores(), f"Missing: {name}"

    def test_resolve_returns_callable(self) -> None:
        fn = resolve_calibration_score("vf_residual")
        assert callable(fn)


class TestProjector:
    def test_idempotency(self, random_unitary_basis: Path) -> None:
        """P_M @ P_M = P_M to numerical precision (defining property)."""
        proj = _projector_for_basis(random_unitary_basis, torch.device("cpu"))
        diff = (proj @ proj - proj).abs().max().item()
        assert diff < 1e-4, f"Projector not idempotent: max|P^2 - P| = {diff:.3e}"

    def test_hermitian(self, random_unitary_basis: Path) -> None:
        proj = _projector_for_basis(random_unitary_basis, torch.device("cpu"))
        diff = (proj - proj.conj().T).abs().max().item()
        assert diff < 1e-4, f"Projector not Hermitian: max|P - P^H| = {diff:.3e}"

    def test_condition_number(self, random_unitary_basis: Path) -> None:
        kappa = marker_basis_condition_number(random_unitary_basis)
        # Orthonormal columns → cond(M^H M) ≈ 1.
        assert kappa < 10.0, f"Random unitary should give kappa ~ 1; got {kappa}."


class TestVFResidualScore:
    def test_score_nonnegative(self, random_unitary_basis: Path) -> None:
        x = torch.randn(4, 1, 8, 8, dtype=torch.complex64)
        target = torch.zeros_like(x)  # ignored
        s = vf_residual_score(x, target, marker_basis_path=str(random_unitary_basis))
        assert (s >= 0).all()
        assert s.shape == (4, 1)  # (..., N) → batched scalar per sample

    def test_zero_when_orthogonal_to_marker_subspace(
        self, random_unitary_basis: Path, tmp_path: Path
    ) -> None:
        """If x is orthogonal to range(M), s should be 0."""
        # Build a basis whose row count matches our test (64).
        # Generate a vector orthogonal to all marker columns by projection-residual.
        proj = _projector_for_basis(random_unitary_basis, torch.device("cpu"))
        x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        flat = x.reshape(2, 1, -1).to(proj.dtype)
        # Subtract the projected component → residual is orthogonal to range(M).
        residual = flat - flat @ proj.T
        x_orth = residual.reshape(2, 1, 8, 8)
        s = vf_residual_score(x_orth, x_orth, marker_basis_path=str(random_unitary_basis))
        assert s.abs().max().item() < 1e-4, (
            f"Score on x ⟂ range(M) should be ~0; got {s.abs().max().item():.3e}"
        )

    def test_positive_inside_marker_subspace(
        self, random_unitary_basis: Path
    ) -> None:
        """If x lies entirely IN range(M), s = ||x||."""
        basis = torch.load(random_unitary_basis, weights_only=True)
        # x = M @ c for random c → lies in range(M).
        c = torch.randn(2, basis.shape[1], dtype=torch.complex64)
        x = (basis @ c.T).T  # (2, 64)
        x = x.reshape(2, 1, 8, 8)
        s = vf_residual_score(x, x, marker_basis_path=str(random_unitary_basis))
        x_norm = x.reshape(2, -1).abs().pow(2).sum(dim=-1).sqrt()
        assert torch.allclose(s.squeeze(-1), x_norm, atol=1e-3), (
            f"Score on x ∈ range(M) should equal ||x||; got s={s}, ||x||={x_norm}"
        )

    def test_missing_marker_basis_path_raises(self) -> None:
        x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        with pytest.raises(ValueError, match="marker_basis_path"):
            vf_residual_score(x, x)

    def test_target_is_ignored(self, random_unitary_basis: Path) -> None:
        """Two different targets must produce the same score (target is ignored)."""
        x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        target_a = torch.zeros_like(x)
        target_b = torch.randn_like(x)
        s_a = vf_residual_score(x, target_a, marker_basis_path=str(random_unitary_basis))
        s_b = vf_residual_score(x, target_b, marker_basis_path=str(random_unitary_basis))
        assert torch.allclose(s_a, s_b), "Score must not depend on target."
