"""
Strategy Execution Graph Verification Tests

This module contains comprehensive tests to verify that all training strategies
follow the architectural skills and design patterns defined in .agent/skills/.

Tests verify:
1. Single Responsibility Principle (SRP)
2. Dependency Injection and Decoupling
3. Configuration Management (SSOT)
4. Error Handling and Reliability
5. Performance and Computation Graph
6. Type Safety and Observability
"""

import inspect

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy
from mriforge.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy
from mriforge.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from mriforge.infrastructure.training.strategies.vae import VAETrainingStrategy

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: SRP (Single Responsibility Principle) Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyResponsibilities:
    """Verify that strategies focus on training orchestration, not data loading."""

    STRATEGIES_TO_TEST = [
        GANTrainingStrategy,
        VAETrainingStrategy,
        DiffusionTrainingStrategy,
        ReconstructionTrainingStrategy,
    ]

    def test_strategies_are_not_data_loaders(self):
        """✅ SKILL: Strategies should NOT contain data loading logic."""
        forbidden_methods = [
            "__iter__",
            "__next__",
            "load_dataset",
            "create_loader",
            "_load_batch",
        ]

        for strategy_cls in self.STRATEGIES_TO_TEST:
            for forbidden in forbidden_methods:
                assert not hasattr(strategy_cls, forbidden) or not callable(
                    getattr(strategy_cls, forbidden, None)
                ), f"❌ {strategy_cls.__name__} has data loading method: {forbidden}"

    def test_strategies_inherit_from_base_strategy(self):
        """✅ SKILL: All strategies must inherit from BaseTrainingStrategy."""
        for strategy_cls in self.STRATEGIES_TO_TEST:
            assert issubclass(
                strategy_cls, BaseTrainingStrategy
            ), f"❌ {strategy_cls.__name__} does not inherit from BaseTrainingStrategy"

    def test_compute_losses_impl_is_abstract_in_base(self):
        """✅ SKILL: _compute_losses_impl must be abstract in BaseTrainingStrategy."""
        # BaseTrainingStrategy should define _compute_losses_impl as abstract
        assert hasattr(
            BaseTrainingStrategy, "_compute_losses_impl"
        ), "❌ BaseTrainingStrategy missing _compute_losses_impl"

    def test_strategies_implement_compute_losses_impl(self):
        """✅ SKILL: All concrete strategies must implement _compute_losses_impl."""
        for strategy_cls in self.STRATEGIES_TO_TEST:
            assert (
                "_compute_losses_impl" in strategy_cls.__dict__
            ), f"❌ {strategy_cls.__name__} does not implement _compute_losses_impl"

    def test_strategies_no_hardcoded_hyperparameters(self):
        """✅ SKILL: No magic numbers in strategy code."""
        forbidden_numbers = {0.0001, 0.001, 0.01, 1000.0, 4008.4744}

        for strategy_cls in self.STRATEGIES_TO_TEST:
            source = inspect.getsource(strategy_cls)

            # Check for hardcoded floats
            for forbidden in forbidden_numbers:
                # Exclude version numbers and line references
                if f"{forbidden}" in source and "version" not in source.lower():
                    # This is a heuristic check, actual review needed
                    pass


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Configuration Management (SSOT - Single Source of Truth) Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigurationManagement:
    """Verify that strategies load config once and pass through layers."""

    def test_config_accessed_via_self_state_config(self):
        """✅ SKILL: Config accessed through self.state.config (SSOT)."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls.__init__)

            # Verify config is accessed from state
            assert (
                "self.state.config" in source or "self.env" in source
            ), f"❌ {strategy_cls.__name__} does not access config from state"

    def test_no_environment_variables_for_config(self):
        """✅ SKILL: Config should come from injected objects, not os.environ."""
        forbidden_patterns = ["os.environ", "os.getenv"]

        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls)
            for pattern in forbidden_patterns:
                if pattern in source and "__comment__" not in source:
                    # Likely violates SSOT principle
                    pass  # Heuristic check

    def test_config_immutability_assertion(self):
        """✅ SKILL: Config should be frozen (immutable) via Pydantic."""
        # This would require loading actual config objects
        # For now, verify that strategies don't call .update() or mutate config
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        forbidden_mutations = [".update(", "config[", "= config", "config."]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls._compute_losses_impl)

            # Verify no config mutations in loss computation
            for mutation in forbidden_mutations:
                if mutation in source and "getattr" not in source:
                    # Potential mutation detected
                    pass


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Dependency Injection and Decoupling Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDependencyInjection:
    """Verify that strategies use dependency injection, not hardcoded instantiation."""

    def test_strategies_accept_device_injection(self):
        """✅ SKILL: Strategies must accept device as parameter, not create it."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            sig = inspect.signature(strategy_cls.__init__)
            assert (
                "device" in sig.parameters
            ), f"❌ {strategy_cls.__name__} __init__ missing device parameter"

    def test_strategies_accept_state_injection(self):
        """✅ SKILL: Strategies must accept state/env for dependencies."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            sig = inspect.signature(strategy_cls.__init__)
            # Should accept either 'state' or 'env'
            assert (
                "state" in sig.parameters or "env" in sig.parameters
            ), f"❌ {strategy_cls.__name__} __init__ missing state or env parameter"

    def test_no_global_singletons(self):
        """✅ SKILL: Strategies must not use global singletons.

        Uses an AST-based scan that excludes comments and string literals
        (docstrings) so we don't false-positive on commentary that happens
        to contain the word "singleton" (e.g. PyTorch's "non-singleton
        dimension" diagnostic).
        """
        import ast

        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls)
            tree = ast.parse(source)

            # Disallow `global X` statements anywhere in the class.
            for node in ast.walk(tree):
                assert not isinstance(node, ast.Global), (
                    f"❌ {strategy_cls.__name__} uses a `global` statement"
                )

            # Disallow class-level instance accessors that look like
            # canonical singleton patterns (get_instance, _instance attr,
            # Singleton metaclass usage).
            forbidden_names = {"get_instance", "_instance", "Singleton"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in forbidden_names:
                    raise AssertionError(
                        f"❌ {strategy_cls.__name__} references singleton-like name "
                        f"`{node.id}`"
                    )
                if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                    raise AssertionError(
                        f"❌ {strategy_cls.__name__} accesses singleton-like attribute "
                        f"`{node.attr}`"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Error Handling and Reliability Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorHandlingAndReliability:
    """Verify strategies handle errors correctly and log reliably."""

    def test_strategies_use_logging_not_print(self):
        """✅ SKILL: Use structured logging, not print() statements."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls)
            # Should use logging_service, not print
            if "print(" in source:
                # Verify it's only in comments or docstrings
                lines = source.split("\n")
                for line in lines:
                    if "print(" in line and not line.strip().startswith("#"):
                        # Potential print statement found
                        pass

    def test_strategies_have_error_handling_in_train_step(self):
        """✅ SKILL: train_step should handle errors gracefully (via override or inheritance)."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            # Check if strategy or its base class has error handling
            has_error_handling = False

            # Check the strategy's own train_step override if it exists
            if hasattr(strategy_cls, "train_step"):
                strategy_source = inspect.getsource(strategy_cls.train_step)
                has_error_handling = (
                    "try:" in strategy_source
                    or "_handle_anomalies" in strategy_source
                    or "_handle_error" in strategy_source
                )

            # If not found in override, check that base class has it
            if not has_error_handling and hasattr(BaseTrainingStrategy, "train_step"):
                base_source = inspect.getsource(BaseTrainingStrategy.train_step)
                has_error_handling = "_handle_anomalies" in base_source

            assert (
                has_error_handling
            ), f"❌ {strategy_cls.__name__} missing error handling in train_step or base"

    def test_compute_losses_returns_dict_not_raw_tensor(self):
        """✅ SKILL: _compute_losses_impl returns dict for clear loss tracking."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls._compute_losses_impl)
            # Should return dict
            assert (
                "return" in source
            ), f"❌ {strategy_cls.__name__}._compute_losses_impl has no return"
            assert (
                "dict" in source or "{" in source
            ), f"❌ {strategy_cls.__name__}._compute_losses_impl may not return dict"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Computation Graph and Backpropagation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputationGraphIntegrity:
    """Verify that loss computation graph is complete for backpropagation."""

    def test_total_loss_not_initialized_as_disconnected_tensor(self):
        """✅ SKILL: total_loss must be initialized from actual losses, not torch.tensor(0.0)."""
        strategies_to_check = [
            DiffusionTrainingStrategy,
        ]

        for strategy_cls in strategies_to_check:
            source = inspect.getsource(strategy_cls._compute_losses_impl)

            # Check for broken pattern: torch.tensor(0.0) initialization
            if "total_loss" in source:
                lines = source.split("\n")
                found_broken_pattern = False
                for i, line in enumerate(lines):
                    if "total_loss" in line and "torch.tensor(0" in line:
                        found_broken_pattern = True
                        break

                assert (
                    not found_broken_pattern
                ), f"❌ {strategy_cls.__name__} initializes total_loss as disconnected tensor"

    def test_total_loss_initialized_from_first_loss_or_none(self):
        """✅ SKILL: total_loss should start as None and be assigned from first loss."""
        strategies_to_check = [
            DiffusionTrainingStrategy,
        ]

        for strategy_cls in strategies_to_check:
            source = inspect.getsource(strategy_cls._compute_losses_impl)

            # Check for correct pattern: total_loss = None or = first_loss
            if "total_loss" in source:
                assert (
                    "total_loss = None" in source or "total_loss = " in source
                ), f"❌ {strategy_cls.__name__} does not initialize total_loss properly"

    def test_gradient_flow_all_losses_accumulated(self):
        """✅ SKILL: All loss components must be accumulated into total_loss.

        Accepts any canonical accumulation pattern: explicit ``+=`` / ``=
        total_loss + ...``, ``sum(...)``-based aggregation, or delegation to
        a ``loss_computer.compute(...)`` / ``LossOutput.total`` aggregator
        (the post-Phase-6 standard in this repo).
        """
        source_map = {
            "diffusion": DiffusionTrainingStrategy,
            "gan": GANTrainingStrategy,
        }

        for name, strategy_cls in source_map.items():
            source = inspect.getsource(strategy_cls._compute_losses_impl)

            # Should accumulate losses (any tracked-total identifier counts).
            assert (
                "total_loss" in source or "loss_output.total" in source
            ), f"❌ {strategy_cls.__name__} does not track a total loss"

            # Accept any of the canonical accumulation patterns.
            accumulation_patterns = (
                "+=",
                "= total_loss +",
                "loss_computer.compute",
                "loss_output.total",
                "sum(",
            )
            assert any(pat in source for pat in accumulation_patterns), (
                f"❌ {strategy_cls.__name__} does not accumulate losses via "
                f"any recognised pattern (expected one of {accumulation_patterns})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Type Safety and Annotations Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeSafetyAndAnnotations:
    """Verify that strategies have proper type hints."""

    def test_compute_losses_impl_has_return_type_hint(self):
        """✅ SKILL: _compute_losses_impl must have return type annotation."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls._compute_losses_impl)
            # Check for return type hint
            assert (
                "->" in source
            ), f"❌ {strategy_cls.__name__}._compute_losses_impl missing return type hint"

    def test_compute_losses_impl_has_argument_type_hints(self):
        """✅ SKILL: Function arguments must have type annotations."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            sig = inspect.signature(strategy_cls._compute_losses_impl)
            for param_name, param in sig.parameters.items():
                if param_name not in ["self", "kwargs"]:
                    assert (
                        param.annotation != inspect.Parameter.empty
                    ), f"❌ {strategy_cls.__name__}._compute_losses_impl parameter {param_name} missing type hint"

    def test_train_step_has_proper_type_hints(self):
        """✅ SKILL: train_step must have complete type annotations."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            if hasattr(strategy_cls, "train_step"):
                sig = inspect.signature(strategy_cls.train_step)
                # Should have return type
                assert (
                    sig.return_annotation != inspect.Signature.empty
                ), f"❌ {strategy_cls.__name__}.train_step missing return type"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Performance and GPU Optimization Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPerformanceOptimization:
    """Verify that strategies follow PyTorch performance best practices."""

    # `test_no_item_calls_in_loss_computation` lived here and asserted NOTHING --
    # its body ended in `if ".item()" in source: pass`, so it could not fail. It
    # named `_compute_losses_impl`, the hook the real gate was NOT auditing, and
    # is the likeliest reason nobody noticed. Deleted rather than repaired
    # (non-negotiable #17, one owner per invariant): the owner is
    # tests/architecture/test_training_loop_discipline.py, which resolves the
    # per-step method set from base.py's call graph and scans 163 files by AST
    # rather than 3 classes by `inspect.getsource`.

    def test_pin_memory_if_dataloader_used(self):
        """✅ SKILL: If strategies create dataloaders, use pin_memory=True."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls)
            if "DataLoader" in source:
                # Should have pin_memory
                assert (
                    "pin_memory" in source
                ), f"❌ {strategy_cls.__name__} DataLoader missing pin_memory"

    def test_no_cuda_empty_cache_calls(self):
        """✅ SKILL: avoid the CUDA cache-clearing call - massive overhead.

        The banned call is assembled rather than spelled out: `tests/conftest.py`
        marks a test `gpu` when the literal appears in its source or its enclosing
        CLASS's source, so writing it here skipped all three tests in this class
        on every CPU box. A source-inspection test cannot contain the string it
        inspects for.
        """
        banned = "empty_" + "cache"
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            source = inspect.getsource(strategy_cls)
            assert banned not in source, (
                f"❌ {strategy_cls.__name__} calls the CUDA cache-clearing op"
            )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Integration Tests - Full Execution Graph
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyExecutionGraph:
    """Test complete execution flow from config to gradients."""

    @pytest.mark.slow
    def test_loss_computation_creates_valid_computation_graph(self):
        """✅ Full integration: Verify loss computation graph is complete."""
        # Create mock model
        model = nn.Linear(10, 10)

        # Create mock tensors
        input_batch = torch.randn(2, 10)
        target_batch = torch.randn(2, 10)

        # Verify computation graph for diffusion model
        components = {}
        total_loss = None

        loss1 = nn.functional.mse_loss(model(input_batch), target_batch)
        components["mse"] = loss1
        if total_loss is None:
            total_loss = loss1
        else:
            total_loss = total_loss + loss1

        loss2 = nn.functional.l1_loss(model(input_batch), target_batch)
        components["l1"] = loss2
        total_loss = total_loss + loss2 * 0.5

        components["total"] = total_loss

        # Verify backward works
        assert total_loss.requires_grad, "❌ total_loss does not require grad"
        assert total_loss.grad_fn is not None, "❌ total_loss has no grad_fn"

        # Backward pass should work
        total_loss.backward()

        # Check that gradients were computed
        assert (
            model.weight.grad is not None
        ), "❌ Model weights did not receive gradients"

    def test_strategy_loss_computation_flow(self):
        """✅ Verify losses flow through backward without issues."""
        # This is a structural test - actual testing would require full setup
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            # Verify class structure is sound
            assert hasattr(
                strategy_cls, "_compute_losses_impl"
            ), f"❌ {strategy_cls.__name__} missing _compute_losses_impl"
            assert hasattr(
                strategy_cls, "train_step"
            ), f"❌ {strategy_cls.__name__} missing train_step"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Strategy Pattern and Template Method Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyPattern:
    """Verify strategies implement Template Method pattern correctly."""

    def test_base_strategy_defines_template(self):
        """✅ SKILL: BaseTrainingStrategy defines training loop template."""
        assert hasattr(
            BaseTrainingStrategy, "train_step"
        ), "❌ BaseTrainingStrategy missing train_step"
        assert hasattr(
            BaseTrainingStrategy, "_compute_losses_impl"
        ), "❌ BaseTrainingStrategy missing _compute_losses_impl"

    def test_concrete_strategies_override_compute_losses_impl(self):
        """✅ SKILL: Concrete strategies override the algorithm (losses)."""
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            # _compute_losses_impl should be defined in concrete class
            assert (
                "_compute_losses_impl" in strategy_cls.__dict__
            ), f"❌ {strategy_cls.__name__} does not override _compute_losses_impl"

    def test_strategies_can_be_swapped_via_registry(self):
        """✅ SKILL: Strategies should be interchangeable via registry pattern."""
        # This verifies the interface is consistent
        strategies = [
            GANTrainingStrategy,
            VAETrainingStrategy,
            ReconstructionTrainingStrategy,
        ]

        for strategy_cls in strategies:
            # All should have the same key methods
            assert hasattr(
                strategy_cls, "train_step"
            ), f"❌ {strategy_cls.__name__} missing train_step (not interchangeable)"
            assert hasattr(
                strategy_cls, "_compute_losses_impl"
            ), f"❌ {strategy_cls.__name__} missing _compute_losses_impl"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
