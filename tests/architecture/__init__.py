"""
Architecture fitness function tests.

These tests use AST-static analysis to enforce Clean Architecture invariants
across src/mriforge/.  They carry the ``@pytest.mark.architecture`` marker and
run without executing any GAN/PyTorch code, so collection and execution are
fast everywhere.

See CLAUDE.md "Architecture (load-bearing)" and pitfalls #2/#4/#9/#12/#13
for the invariants being checked.
"""
