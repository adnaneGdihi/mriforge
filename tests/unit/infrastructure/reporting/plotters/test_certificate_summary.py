"""Render test for the certificate-summary figure.

Covers the inverted-reference-line fix: the 0.9 coverage guide is now drawn on
the R1 panel inside the ``if p:`` block (a real statement), where there are
coverage bars to read against it — previously a ``... if not p else None``
conditional-expression drew it only when the panel was empty. Both the
data-present and data-absent branches must render.
"""

import json

from spectramr.infrastructure.reporting.plotters import certificate_summary


def test_certificate_summary_renders_with_r1(tmp_path):
    r1 = tmp_path / "r1.json"
    r1.write_text(json.dumps({"empirical_coverage": 0.92, "alpha": 0.1}))
    out = tmp_path / "cert.png"
    res = certificate_summary.render_certificate_summary(out, r1_path=r1)
    assert res.exists()


def test_certificate_summary_renders_without_any_payload(tmp_path):
    out = tmp_path / "cert_empty.png"
    res = certificate_summary.render_certificate_summary(out)
    assert res.exists()
