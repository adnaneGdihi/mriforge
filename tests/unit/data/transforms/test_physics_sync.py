"""Tests for :mod:`mriforge.data.transforms.physics_sync`.

Pins the ``fft_norm`` knob contract: ``fft2c`` from
``infrastructure.physics.fft_ops`` is always ortho-normalised, so the
advertised ``fft_norm`` parameter must reject any non-ortho value rather
than silently accept it (CLAUDE.md pitfall #15 — unwired knob).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")

from mriforge.data.transforms.physics_sync import PhysicsSynchronization  # noqa: E402


def test_default_fft_norm_is_ortho() -> None:
    """Construction with defaults succeeds and stores ``fft_norm='ortho'``."""
    t = PhysicsSynchronization()
    assert t.fft_norm == "ortho"


@pytest.mark.parametrize("bad_norm", ["forward", "backward"])
def test_fft_norm_non_ortho_raises(bad_norm: str) -> None:
    """Any non-ortho ``fft_norm`` raises — fft2c SSOT is always ortho (#15)."""
    with pytest.raises(ValueError, match="ortho"):
        PhysicsSynchronization(fft_norm=bad_norm)


def test_ortho_construction_round_trips_kspace() -> None:
    """Smoke: ortho construction applies and regenerates k-space in-place."""
    torch.manual_seed(0)
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=torch.randn(1, 16, 16, 2)),
    )
    t = PhysicsSynchronization()
    out = t(subject)
    assert "kspace" in out
    assert out["kspace"].data.shape[1:] == (16, 16, 2)


class TestItRefusesToTreatKspaceAsAnImage:
    """A4. ``PhysicsSynchronization`` re-derives k-space from the IMAGE, and its
    key search tried ``"input"`` FIRST — but on a k-space arm ``input`` IS the
    measured k-space, so it fed k-space to ``fft2c`` and overwrote
    ``subject["kspace"]`` with a second forward transform.

    Not spare compute: ``strategies/mixins/kspace.py`` returns ``data["kspace"]``
    as the canonical accessor and ``graph_transform`` reads ``subject["kspace"]``
    directly, so the doubly-transformed tensor is what recon strategies consume.
    174 arms enable augmentation; 49 of those are k-space.
    """

    @staticmethod
    def _subject(**images):
        import torchio as tio

        return tio.Subject(
            **{k: tio.ScalarImage(tensor=v) for k, v in images.items()}
        )

    def test_an_ambiguous_subject_raises_instead_of_guessing(self) -> None:
        """`input` + a distinct `kspace` cannot be disambiguated by key name."""
        import torch

        from mriforge.data.transforms.physics_sync import PhysicsSynchronization

        subject = self._subject(
            input=torch.rand(1, 8, 8, 1), kspace=torch.rand(2, 8, 8, 1)
        )
        with pytest.raises(ValueError, match="cannot infer its image key"):
            PhysicsSynchronization()(subject)

    def test_the_message_says_what_goes_wrong_not_just_that_it_is_ambiguous(
        self,
    ) -> None:
        """"Ambiguous" invites a coin flip. Naming the consequence does not."""
        import torch

        from mriforge.data.transforms.physics_sync import PhysicsSynchronization

        with pytest.raises(ValueError) as exc:
            PhysicsSynchronization()(
                self._subject(
                    input=torch.rand(1, 8, 8, 1), kspace=torch.rand(2, 8, 8, 1)
                )
            )
        message = str(exc.value)
        assert "second forward FFT" in message
        assert "image_key=" in message

    def test_an_explicit_image_key_is_honoured(self) -> None:
        """The caller knows what the transform cannot infer."""
        import torch

        from mriforge.data.transforms.physics_sync import PhysicsSynchronization

        subject = self._subject(
            mri=torch.rand(1, 8, 8, 1), kspace=torch.rand(2, 8, 8, 1)
        )
        out = PhysicsSynchronization(image_key="mri")(subject)
        assert "kspace" in out

    def test_an_image_only_subject_still_syncs(self) -> None:
        """The behaviour the transform exists for is unchanged."""
        import torch

        from mriforge.data.transforms.physics_sync import PhysicsSynchronization

        out = PhysicsSynchronization()(self._subject(input=torch.rand(1, 8, 8, 1)))
        assert "kspace" in out

    def test_image_keys_are_now_preferred_over_input(self) -> None:
        """`input` moved to the END of the search order. It is the ambiguous
        one, so it should be the last resort, not the first guess."""
        import inspect

        from mriforge.data.transforms.physics_sync import PhysicsSynchronization

        src = inspect.getsource(PhysicsSynchronization.apply_transform)
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        order = code.split('for key in [', 1)[1].split("]", 1)[0]
        assert order.index('"input"') > order.index('"mri"')


class TestTheBuilderSkipsItOnKspaceArms:
    """The real gate: a k-space arm has no image to re-derive k-space from, so
    the transform is not applicable and applying it destroys measured data."""

    def test_kspace_arm_gets_no_physics_sync(self) -> None:
        import inspect

        from mriforge.data.builders.torchio_transform_builder import (
            TorchIOTransformBuilder,
        )

        src = inspect.getsource(TorchIOTransformBuilder.build_train_transforms)
        assert "is_kspace_dataset_type(config.dataset_type)" in src

    def test_the_gate_uses_the_shared_predicate(self) -> None:
        """Not a local re-derivation — `data/signal_domain.py` is the one home,
        and `spec_card` reads the same fact."""
        from mriforge.data.signal_domain import is_kspace_dataset_type

        assert is_kspace_dataset_type("m4raw")
        assert not is_kspace_dataset_type("nifti_paired")


class TestInputIsImageResolvesTheAmbiguityWithoutWideningIt:
    """`input_is_image` is the caller's answer to the one question the
    `input`/`kspace` guard cannot answer for itself.

    Cluster job 8012333: 10 `10_paradigms` arms died in the DataLoader here.
    They are `nifti_paired` -- image-primary, so the builder does not skip the
    transform -- served by `UniversalMRIDataset`, which derives a `kspace` key
    alongside `input`. That combination is the blind spot between the two
    branches: not a k-space arm, but a subject that looks like one by key name.
    """

    @staticmethod
    def _subject(**arrays):
        return tio.Subject(
            **{k: tio.ScalarImage(tensor=v) for k, v in arrays.items()}
        )

    def test_ambiguous_subject_still_refuses_by_default(self) -> None:
        """Absent a declaration the guard must still refuse (pitfall #9)."""
        subject = self._subject(
            input=torch.rand(1, 8, 8, 1), kspace=torch.rand(2, 8, 8, 1)
        )
        with pytest.raises(ValueError, match="cannot infer its image key"):
            PhysicsSynchronization()(subject)

    def test_declaring_the_domain_lets_an_image_primary_arm_through(self) -> None:
        """The 10-arm fix: k-space is regenerated instead of the pipeline dying."""
        subject = self._subject(
            input=torch.rand(1, 8, 8, 1), kspace=torch.zeros(2, 8, 8, 1)
        )
        out = PhysicsSynchronization(input_is_image=True)(subject)
        assert not torch.allclose(out["kspace"].data, torch.zeros(2, 8, 8, 1)), (
            "kspace was not regenerated from the image"
        )

    def test_it_does_not_change_which_key_is_synced_from(self) -> None:
        """The reason this is a domain flag and not `image_key="input"`.

        On a subject carrying both `hr` and `input` the search order picks
        `hr` -- a DIFFERENT image. Naming `input` at the call site would have
        silently changed which image k-space is derived from, on every
        image-primary arm, with no test able to see it.
        """
        arrays = dict(
            hr=torch.full((1, 8, 8, 1), 0.5),
            input=torch.rand(1, 8, 8, 1),
            kspace=torch.zeros(2, 8, 8, 1),
        )
        declared = PhysicsSynchronization(input_is_image=True)(self._subject(**arrays))
        from_hr = PhysicsSynchronization(image_key="hr")(self._subject(**arrays))
        assert torch.allclose(declared["kspace"].data, from_hr["kspace"].data)

    def test_the_a4_protection_is_not_weakened(self) -> None:
        """A k-space arm reaching the transform undeclared must STILL raise.

        This is the bug the guard exists for: `input` IS the measured k-space,
        so syncing from it applies a second forward FFT and overwrites the
        measurement. Widening the guard to fix the 10 arms must not reopen it.
        """
        kspace = torch.rand(2, 8, 8, 1)
        with pytest.raises(ValueError, match="cannot infer its image key"):
            PhysicsSynchronization()(self._subject(input=kspace, kspace=kspace))
