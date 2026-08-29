"""Unit tests for the extracted constructor-contract module.

Split out of ``generator_kwargs`` in the Wave 0 exit-criterion work (#1400).
These pin the extraction itself -- one owner per symbol, re-exported identically
-- plus the two behaviours the split must not have changed: ``resolve_contract``
never raises, and ``accepts`` honours ``**kwargs``.
"""

from __future__ import annotations

from mriforge.core.component_signature import SignatureContract
from mriforge.infrastructure.builders.generator_contract import (
    SKIP_MODEL_FIELDS,
    accepts,
    resolve_contract,
)


class TestReExportIdentity:
    def test_generator_kwargs_reexports_the_same_objects(self) -> None:
        """One owner per symbol (NN17): the old spellings must not be copies."""
        from mriforge.infrastructure.builders import generator_kwargs as gk

        assert gk._SKIP_MODEL_FIELDS is SKIP_MODEL_FIELDS
        assert gk._accepts is accepts
        assert gk.resolve_contract is resolve_contract

    def test_forward_probe_still_resolves_the_same_function(self) -> None:
        """The one production importer outside builders/ (#1400 split safety)."""
        from mriforge.infrastructure.validation import forward_probe

        assert forward_probe.resolve_contract is resolve_contract


class TestSkipModelFields:
    def test_is_a_non_empty_frozenset_of_str(self) -> None:
        assert isinstance(SKIP_MODEL_FIELDS, frozenset)
        assert SKIP_MODEL_FIELDS
        assert all(isinstance(f, str) for f in SKIP_MODEL_FIELDS)

    def test_carries_the_fields_whose_leak_crashed_a_real_arm(self) -> None:
        """Each was added after a named arm crashed; a silent prune re-breaks it."""
        for field in ("target_domain", "conditioning", "checkpoint_path", "model_kwargs"):
            assert field in SKIP_MODEL_FIELDS


class TestResolveContract:
    def test_unknown_model_type_returns_the_empty_contract(self) -> None:
        contract = resolve_contract(model_type="definitely-not-a-registered-model")
        assert contract.accepted == frozenset()
        assert not contract.accepts_var_kwargs

    def test_no_arguments_returns_the_empty_contract(self) -> None:
        assert resolve_contract().accepted == frozenset()

    def test_reads_an_explicit_signature(self) -> None:
        class _Explicit:
            def __init__(self, alpha: int = 1, beta: str = "b") -> None: ...

        contract = resolve_contract(model_cls=_Explicit)
        assert {"alpha", "beta"} <= contract.accepted

    def test_never_raises_on_an_uninspectable_class(self) -> None:
        """Tolerance is the contract, not an oversight -- the probe depends on it."""

        class _Hostile:
            def __init__(self, *a, **k) -> None: ...

        assert isinstance(resolve_contract(model_cls=_Hostile), SignatureContract)


class TestAccepts:
    def test_named_parameter_is_accepted(self) -> None:
        c = SignatureContract(accepted=frozenset({"alpha"}), accepts_var_kwargs=False, owner="x")
        assert accepts(c, "alpha")
        assert not accepts(c, "beta")

    def test_var_kwargs_accepts_anything(self) -> None:
        """Generators reading DC keys via ``kwargs.get`` bypass SSOT without this."""
        c = SignatureContract(accepted=frozenset(), accepts_var_kwargs=True, owner="x")
        assert accepts(c, "anything_at_all")
