"""Unit tests for :mod:`mriforge.infrastructure.training.utils.kspace_masks`.

Focus: ``generate_batch_masks`` serves the fixed-seed cascade from a memoised
on-device table instead of pulling the timestep tensor to the host. The host
copy was a blocking synchronise that a Scalene profile of
``experiment_11_attention_none`` charged 24.24 % of the run (~225 s), against
0.8 s of actual mask construction.

The tests that carry weight here are the ones that would go red if the cache
served a mask the uncached resolver would not have produced, plus the sync
probe -- which is itself checked against a planted violation so a green result
cannot mean "the probe is blind" (non-negotiable 15).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.training.utils.kspace_masks import (
    KSpaceMaskGenerator,
    SamplingPatternRegistry,
)
from mriforge.infrastructure.training.utils.mask_table_cache import MaskTableCache

T, H, W = 12, 32, 32
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast path is only taken off-CPU"
)


def _generator(device: str = "cpu", seed: int = 42) -> KSpaceMaskGenerator:
    return KSpaceMaskGenerator(
        num_timesteps=T,
        device=torch.device(device),
        default_pattern="variable_density",
        accelerator_kwargs={"seed": seed},
    )


def _reference(gen: KSpaceMaskGenerator, timesteps: list[int]) -> torch.Tensor:
    """Masks built one at a time through the uncached resolver."""
    return torch.stack(
        [gen.generate_acceleration_mask(t, (H, W), 4, None) for t in timesteps], dim=0
    )


class TestUncachedResolverIsMemoisable:
    """The premise the cache rests on: the fixed-seed path is a pure function."""

    def test_same_timestep_twice_is_bit_identical(self) -> None:
        gen = _generator()
        assert torch.equal(
            gen.generate_acceleration_mask(5, (H, W)),
            gen.generate_acceleration_mask(5, (H, W)),
        )

    def test_cascade_actually_varies_with_timestep(self) -> None:
        """Guards against a vacuous parity result from a constant cascade."""
        gen = _generator()
        fractions = {
            float(gen.generate_acceleration_mask(t, (H, W)).float().mean())
            for t in (0, T // 2, T - 1)
        }
        assert len(fractions) == 3, f"cascade is not timestep-dependent: {fractions}"


class TestCpuPathIsUntouched:
    """On CPU there is no sync to remove, so the legacy path must still run."""

    def test_cpu_timesteps_do_not_populate_the_cache(self) -> None:
        gen = _generator()
        gen.generate_batch_masks(2, torch.tensor([3, 7]), (H, W))
        assert len(gen._mask_tables) == 0

    def test_cpu_result_matches_the_reference(self) -> None:
        gen = _generator()
        out = gen.generate_batch_masks(2, torch.tensor([3, 7]), (H, W))
        assert torch.equal(out, _reference(gen, [3, 7]))


@requires_cuda
@pytest.mark.gpu
class TestCudaFastPath:
    """The memoised path must be indistinguishable from the uncached one."""

    def test_parity_with_uncached_resolver(self) -> None:
        gen = _generator("cuda")
        steps = [0, 4, 9]
        out = gen.generate_batch_masks(len(steps), torch.tensor(steps, device="cuda"), (H, W))
        assert torch.equal(out, _reference(gen, steps))

    def test_parity_holds_for_every_timestep(self) -> None:
        gen = _generator("cuda")
        steps = list(range(T))
        out = gen.generate_batch_masks(T, torch.tensor(steps, device="cuda"), (H, W))
        assert torch.equal(out, _reference(gen, steps))

    def test_shape_and_dtype_contract_preserved(self) -> None:
        gen = _generator("cuda")
        out = gen.generate_batch_masks(2, torch.tensor([1, 2], device="cuda"), (H, W))
        assert out.shape == (2, 1, H, W)
        assert out.dtype is torch.bool

    def test_repeated_timesteps_in_one_batch(self) -> None:
        gen = _generator("cuda")
        out = gen.generate_batch_masks(3, torch.tensor([5, 5, 5], device="cuda"), (H, W))
        assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])

    def test_table_is_built_once_and_reused(self) -> None:
        gen = _generator("cuda")
        for step in range(T):
            gen.generate_batch_masks(1, torch.tensor([step], device="cuda"), (H, W))
        assert len(gen._mask_tables) == 1

    def test_a_different_seed_is_not_served_a_stale_cascade(self) -> None:
        """The seed lives on the inner accelerator and is part of the key."""
        gen = _generator("cuda")
        steps = torch.tensor([2, 6], device="cuda")
        first = gen.generate_batch_masks(2, steps, (H, W)).clone()
        gen._get_accelerator("variable_density").accelerator.seed = 4242
        second = gen.generate_batch_masks(2, steps, (H, W))
        assert len(gen._mask_tables) == 2
        assert torch.equal(second, _reference(gen, [2, 6]))
        del first


@requires_cuda
@pytest.mark.gpu
class TestNoHostSynchronise:
    """The point of the change, asserted with a probe proven to discriminate."""

    @staticmethod
    def _run_under_sync_debug(fn) -> RuntimeError | None:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fn()
            return None
        except RuntimeError as exc:
            return exc
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_probe_catches_the_removed_host_copy(self) -> None:
        """Planted violation: the exact line this change deleted must go red.

        Without this, a green ``test_warm_path_does_not_synchronise`` could
        equally mean the probe never fires.
        """
        steps = torch.tensor([1, 2], device="cuda")
        caught = self._run_under_sync_debug(
            lambda: steps.detach().to("cpu", non_blocking=False).tolist()
        )
        assert caught is not None, "sync probe is blind; the test below proves nothing"
        assert "synchroniz" in str(caught).lower()

    def test_warm_path_does_not_synchronise(self) -> None:
        gen = _generator("cuda")
        steps = torch.tensor([1, 2], device="cuda")
        gen.generate_batch_masks(2, steps, (H, W))  # build the table first
        assert (
            self._run_under_sync_debug(lambda: gen.generate_batch_masks(2, steps, (H, W))) is None
        )


class TestFastAndSlowPathsAgreeOnBadInput:
    """The two paths gave DIFFERENT answers to the same malformed call (#1509).

    The divergence ran in both directions, and in each direction the silent
    answer was the one that ran by default:

    * ``timesteps.numel() < batch_size`` -- CPU raised ``IndexError`` from
      ``timestep_list[i]``; CUDA truncated with ``[:batch_size]`` and returned a
      SHORT tensor, which broadcast downstream and degraded every sample in the
      batch with sample 0's mask.
    * a timestep outside ``[0, num_timesteps)`` -- CUDA raised a device-side
      index error from ``index_select``; CPU returned a mask, because the
      accelerator clamps its own schedule lookup.

    Each guard has exactly one owner, placed where it costs no host sync:
    the length check is tensor metadata read ahead of the device branch, and the
    bound check sits in ``generate_acceleration_mask`` where the timestep is
    already a host ``int``. The CUDA device-side assert is deliberately NOT
    replaced by a host-side bounds check -- that would reintroduce the very sync
    #1507 removed.
    """

    def test_short_timestep_tensor_raises_on_the_cpu_path(self) -> None:
        gen = _generator()
        with pytest.raises(ValueError, match="each sample needs its own timestep"):
            gen.generate_batch_masks(
                batch_size=4, timesteps=torch.tensor([3]), image_shape=(H, W)
            )

    @requires_cuda
    @pytest.mark.gpu
    def test_short_timestep_tensor_raises_on_the_cuda_path(self) -> None:
        """The shape that used to return ``[1, 1, H, W]`` for a batch of 4.

        Placed ahead of the device branch precisely so this and the CPU case
        cannot drift apart again; running both is what makes that structural
        claim checkable rather than asserted.
        """
        gen = _generator(device="cuda")
        with pytest.raises(ValueError, match="each sample needs its own timestep"):
            gen.generate_batch_masks(
                batch_size=4,
                timesteps=torch.tensor([3], device="cuda"),
                image_shape=(H, W),
            )

    def test_matching_length_still_works(self) -> None:
        """Discrimination check: the guard rejects the malformed call only."""
        gen = _generator()
        masks = gen.generate_batch_masks(
            batch_size=3, timesteps=torch.tensor([1, 2, 3]), image_shape=(H, W)
        )
        assert masks.shape == (3, 1, H, W)

    def test_longer_timestep_tensor_still_truncates(self) -> None:
        """Unchanged behaviour, pinned: both paths already agreed on this."""
        gen = _generator()
        masks = gen.generate_batch_masks(
            batch_size=2, timesteps=torch.tensor([1, 2, 3, 4]), image_shape=(H, W)
        )
        assert masks.shape == (2, 1, H, W)

    def test_timestep_past_the_horizon_raises(self) -> None:
        """Was answered silently with a mask on every CPU entry point."""
        gen = _generator()
        with pytest.raises(ValueError, match=r"outside the schedule"):
            gen.generate_acceleration_mask(T, (H, W))

    def test_negative_timestep_raises(self) -> None:
        gen = _generator()
        with pytest.raises(ValueError, match=r"outside the schedule"):
            gen.generate_acceleration_mask(-1, (H, W))

    def test_the_two_endpoints_of_the_horizon_are_accepted(self) -> None:
        """Fencepost discrimination: ``T - 1`` is valid, ``T`` is not."""
        gen = _generator()
        assert gen.generate_acceleration_mask(0, (H, W)).shape[-2:] == (H, W)
        assert gen.generate_acceleration_mask(T - 1, (H, W)).shape[-2:] == (H, W)

    def test_out_of_range_timestep_reaches_the_guard_through_batch_masks(self) -> None:
        """The slow path funnels through the same owner -- no second check."""
        gen = _generator()
        with pytest.raises(ValueError, match=r"outside the schedule"):
            gen.generate_batch_masks(
                batch_size=1, timesteps=torch.tensor([T + 5]), image_shape=(H, W)
            )


class TestTheCacheKeyIsTheResolvedTypeNotTheSpelling:
    """Two accepted spellings of one accelerator must share one table.

    ``SamplingPatternRegistry`` accepts 31 spellings for 19 canonical types and
    9 of those types have more than one spelling. ``_get_accelerator`` already
    memoises one instance per RESOLVED type, so two spellings provably produce
    identical masks -- but the table cache used to key on the raw string the
    arm wrote, so each spelling built and held its own bit-identical table.

    These tests pin the collapse at the key level (runs anywhere) and end to end
    on an accelerator (where the fast path is actually reachable).
    """

    # Every canonical type that has more than one accepted spelling, so a new
    # alias arriving in the registry is covered without editing this list.
    @staticmethod
    def _multi_spelling_pairs() -> list[tuple[str, str]]:
        from collections import defaultdict

        by_type: dict[str, list[str]] = defaultdict(list)
        for spelling in SamplingPatternRegistry.list_accepted():
            by_type[SamplingPatternRegistry.resolve(spelling)].append(spelling)
        return [(v[0], v[1]) for v in by_type.values() if len(v) > 1]

    def test_the_registry_still_has_multi_spelling_types(self) -> None:
        """If this ever hits zero the tests below become vacuous, not passing."""
        assert self._multi_spelling_pairs()

    def test_two_spellings_of_one_type_build_the_same_key(self) -> None:
        gen = _generator("cpu")
        for first, second in self._multi_spelling_pairs():
            acc = gen._get_accelerator(first)
            key_a = MaskTableCache.build_key(
                gen._resolve_acceleration_type(first),
                (8, 8),
                4,
                torch.device("cpu"),
                acc,
            )
            key_b = MaskTableCache.build_key(
                gen._resolve_acceleration_type(second),
                (8, 8),
                4,
                torch.device("cpu"),
                acc,
            )
            assert key_a == key_b, f"{first!r} and {second!r} must share a key"

    def test_keying_on_the_raw_spelling_would_not_collapse_them(self) -> None:
        """The discrimination: this is what the old key did, and why it was wrong.

        Without this the test above passes for a trivial reason -- if the two
        spellings were equal strings, resolving them would be a no-op.
        """
        gen = _generator("cpu")
        acc = gen._get_accelerator("uniform")
        raw_a = MaskTableCache.build_key("uniform", (8, 8), 4, torch.device("cpu"), acc)
        raw_b = MaskTableCache.build_key(
            "uniform_cartesian", (8, 8), 4, torch.device("cpu"), acc
        )
        assert raw_a != raw_b

    def test_the_two_spellings_share_one_accelerator_instance(self) -> None:
        """Why merging the tables is SAFE, not merely tidy."""
        gen = _generator("cpu")
        for first, second in self._multi_spelling_pairs():
            assert gen._get_accelerator(first) is gen._get_accelerator(second)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="fast path needs a device")
    def test_two_spellings_share_one_table_end_to_end(self) -> None:
        gen = _generator("cuda")
        steps = torch.tensor([0, 1], device="cuda")
        gen.generate_batch_masks(2, steps, (H, W), 4, pattern="uniform")
        after_first = len(gen._mask_tables._tables)
        gen.generate_batch_masks(2, steps, (H, W), 4, pattern="uniform_cartesian")
        assert after_first == 1
        assert len(gen._mask_tables._tables) == 1
