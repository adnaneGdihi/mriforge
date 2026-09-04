"""Regression test for the audit-15-F11 fix.

In bootstrap.build_container, the schema-only domain-validation
(``verify_startup_loss_domains``) must precede the loss-instantiation
audit (``verify_startup_losses``). The previous order paid the
loss-construction cost before catching a domain mismatch, defeating
the fail-fast principle.
"""

from __future__ import annotations

from pathlib import Path


def test_domain_validation_precedes_loss_audit() -> None:
    src = (
        Path(__file__).resolve().parents[2] / "src" / "spectramr" / "bootstrap.py"
    ).read_text()
    domain_pos = src.find("verify_startup_loss_domains")
    losses_pos = src.find("verify_startup_losses(config.losses)")

    assert domain_pos > 0, "verify_startup_loss_domains call missing"
    assert losses_pos > 0, "verify_startup_losses call missing"
    assert domain_pos < losses_pos, (
        "bootstrap.build_container now calls verify_startup_losses "
        "BEFORE verify_startup_loss_domains. That order pays the loss-"
        "construction cost just to throw the results away when domain "
        "validation fails — see audit 15 F11. Re-order them."
    )
