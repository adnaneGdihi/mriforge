"""Tests for ``ModuleABCMeta``.

Targets ``spectramr.shared.utils.metaclass``. Thin marker class combining
``ABCMeta`` with the ``torch.nn.Module`` metaclass for abstract Module
subclasses.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import pytest

from spectramr.shared.utils.metaclass import ModuleABCMeta


def test_module_abc_meta_inherits_from_abc_meta() -> None:
    """``ModuleABCMeta`` is a subclass of ``ABCMeta``."""
    assert issubclass(ModuleABCMeta, ABCMeta)


def test_class_using_metaclass_can_have_abstract_methods() -> None:
    """A class using ``ModuleABCMeta`` honours ``@abstractmethod``."""

    class AbstractThing(metaclass=ModuleABCMeta):
        @abstractmethod
        def do_thing(self) -> None: ...

    # Cannot instantiate without overriding the abstract method.
    with pytest.raises(TypeError, match="abstract"):
        AbstractThing()


def test_concrete_subclass_instantiates() -> None:
    """A concrete override of the abstract method allows instantiation."""

    class AbstractThing(metaclass=ModuleABCMeta):
        @abstractmethod
        def do_thing(self) -> None: ...

    class Concrete(AbstractThing):
        def do_thing(self) -> None:
            return None

    instance = Concrete()
    assert instance.do_thing() is None
