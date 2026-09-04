"""The signature contract must read the MRO-OWNED ``__init__``, not the class's.

``inspect.signature(SomeModule)`` reports whatever ``nn.Module.__init__``
promises, which is why 8 registered metrics were silently dead before PR #718.
Every test here pins the difference between the two readings.
"""

import torch.nn as nn

from spectramr.core.component_signature import (
    SignatureContract,
    owned_init,
    signature_contract,
)


class _OwnsInit(nn.Module):
    def __init__(self, order: int = 2, reduction: str = "mean"):
        super().__init__()


class _InheritsInit(nn.Module):
    pass


class _TakesVarKwargs(nn.Module):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__()


class _Subclass(_OwnsInit):
    """No own ``__init__`` — the contract must resolve to the PARENT's."""


class _PlainObject:
    def __init__(self, a: int, *, b: str = "x"):
        pass


def test_owned_init_finds_the_defining_class():
    assert owned_init(_OwnsInit) is _OwnsInit


def test_owned_init_walks_up_to_the_parent_that_defines_it():
    assert owned_init(_Subclass) is _OwnsInit


def test_owned_init_returns_none_when_only_nn_module_defines_it():
    # This is the case `inspect.signature` gets wrong: it would report
    # nn.Module's permissive signature as though the class accepted it.
    assert owned_init(_InheritsInit) is None


def test_contract_lists_named_parameters_without_self():
    c = signature_contract(_OwnsInit)
    assert c.accepted == frozenset({"order", "reduction"})
    assert c.accepts_var_kwargs is False
    assert c.owner == "_OwnsInit"


def test_contract_flags_var_kwargs_classes():
    c = signature_contract(_TakesVarKwargs)
    assert c.accepted == frozenset({"alpha"})
    assert c.accepts_var_kwargs is True


def test_contract_for_an_inherited_init_accepts_nothing():
    c = signature_contract(_InheritsInit)
    assert c.accepted == frozenset()
    assert c.accepts_var_kwargs is False
    assert c.owner == ""


def test_contract_includes_keyword_only_parameters():
    c = signature_contract(_PlainObject)
    assert c.accepted == frozenset({"a", "b"})


def test_contract_is_frozen():
    c = signature_contract(_OwnsInit)
    assert isinstance(c, SignatureContract)
    try:
        c.accepted = frozenset()  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("SignatureContract must be frozen")


def test_reads_the_owners_own_function_not_the_bound_attribute():
    """The contract reads ``owner.__dict__["__init__"]``, not ``owner.__init__``.

    Attribute access goes through the descriptor protocol, which mypy correctly
    calls unsound — the attribute can resolve to an incompatible subclass's
    ``__init__``. This module exists to report the function a class actually
    defines, so it reads the class dict that ``owned_init`` already proved has
    the key.
    """
    owner = owned_init(_Subclass)
    assert owner is _OwnsInit
    c = signature_contract(_Subclass)
    assert c.accepted == frozenset({"order", "reduction"})
    assert c.owner == "_OwnsInit"


def test_slots_class_is_read_normally():
    class _Slots:
        __slots__ = ("a",)

        def __init__(self, a: int = 1):
            self.a = a

    c = signature_contract(_Slots)
    assert c.accepted == frozenset({"a"})
    assert c.accepts_var_kwargs is False


def test_unreadable_c_extension_signature_falls_back_permissively():
    """``dict`` defines ``__init__`` as a slot wrapper with no readable
    signature. Reporting "accepts nothing" there would make a caller DROP every
    key on a reading it could not make, so the fallback must be permissive.
    """
    c = signature_contract(dict)
    assert c.owner == "dict"
    assert c.accepts_var_kwargs is True
    assert c.accepted == frozenset()
