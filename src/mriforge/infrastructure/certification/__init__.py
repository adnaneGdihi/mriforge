"""Regulatory-grade statistical certification.

Houses the Hoeffding / PAC-Bayes / conformal bound implementations
that the R1–R5 ULF certificates rely on. See
:mod:`mriforge.infrastructure.certification.pathology_recall` for R4 (PRC).
"""

from mriforge.infrastructure.certification.pathology_recall import (
    PathologyRecallCertificate,
    TestSetSizeAudit,
    build_pathology_recall_certificate,
    hoeffding_recall_lower_bound,
    minimum_n_for_target_recall,
    pathology_test_set_size_sufficient,
    write_certificates_json,
)

__all__ = [
    "PathologyRecallCertificate",
    "TestSetSizeAudit",
    "build_pathology_recall_certificate",
    "hoeffding_recall_lower_bound",
    "minimum_n_for_target_recall",
    "pathology_test_set_size_sufficient",
    "write_certificates_json",
]
