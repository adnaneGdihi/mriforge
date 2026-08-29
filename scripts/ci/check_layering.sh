#!/usr/bin/env bash
# scripts/ci/check_layering.sh — layering + SSOT guard (ratcheted)
#
# Phase 6 of TODO/backlog_ssot_and_layering_cleanup.md; repaired in the
# cached-cascade campaign Phase 0 (2026-06).
#
# Runs a set of greps that lock in the architectural rules from CLAUDE.md
# (layer-direction, physics-SSOT, data-SSOT, canonical-loss-home).
#
# RATCHET MODEL
# ------------
# Before the src->mriforge rename this script greped ``src.pipelines`` over
# paths like ``src/infrastructure`` — post-rename neither the import prefix
# nor the path exists, so every check PASSED VACUOUSLY. This rewrite scans
# the REAL tree (``src/mriforge/...`` paths, ``(src.mriforge|mriforge)`` import
# prefixes, ERE-safe physics exemption). Because the real tree carries a
# handful of pre-existing violations (dependency-inversion debt owned by
# WS-3/WS-6/WS-7), the script ratchets against a committed baseline:
#
#   * NEW violations (not in the baseline)      -> FAIL (a regression)
#   * baseline violations no longer present     -> info (tighten the baseline)
#   * everything in the baseline still present  -> PASS (known debt, tracked)
#
# This keeps the gate non-vacuous (it really scans + catches regressions)
# while not forcing 10 risky refactors into the bootstrap phase.
#
# Usage:
#   bash scripts/ci/check_layering.sh                   # ratcheted audit
#   bash scripts/ci/check_layering.sh --quiet           # only print failures
#   bash scripts/ci/check_layering.sh --update-baseline # regenerate baseline
#   bash scripts/ci/check_layering.sh --strict          # ANY violation fails (WS-C end-state)
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

BASELINE="scripts/ci/layering_baseline.txt"
QUIET=false
UPDATE_BASELINE=false
STRICT=false
for arg in "$@"; do
    case "${arg}" in
        --quiet) QUIET=true ;;
        --update-baseline) UPDATE_BASELINE=true ;;
        --strict) STRICT=true ;;
    esac
done

log() {
    if [[ "${QUIET}" == "false" ]]; then
        echo "$@"
    fi
}

# Accumulate every violation (normalized: "path: <import-stmt-trimmed>",
# line number dropped so the baseline survives line moves within a file).
CUR="$(mktemp)"
BASE_SORTED="$(mktemp)"
trap 'rm -f "${CUR}" "${BASE_SORTED}"' EXIT

# Normalize a raw `grep -rEn` hit into a stable baseline key:
#   1. drop the line number, so the key survives a line move within the file;
#   2. collapse the imported-symbol list of a `from X import ...`, so the key
#      survives a reflow between the single-line and parenthesized multi-line
#      forms. `ruff format` reflows imports, and a BLOCKING gate that reports an
#      unchanged accepted-debt import as a NEW violation after `make format` is
#      worse than no gate at all.
# Plain `import x` statements and non-import findings (DataLoader(, yaml.safe_load,
# @register_loss) carry no symbol list and pass through rule 2 untouched.
norm() {
    sed -E 's/^([^:]+):[0-9]+:[[:space:]]*/\1: /' \
        | sed -E 's/^(.*[[:space:]]from[[:space:]]+[A-Za-z0-9_.]+[[:space:]]+import)([[:space:]].*)?$/\1 .../' \
        | sed -E 's/[[:space:]]+$//'
}

# Test seam: `check_layering.sh --norm-filter` reads raw violation lines on stdin and
# writes their normalized baseline keys to stdout. Exercised by
# tests/unit/ci/test_check_layering.py, which pins the reflow-invariance above.
if [[ "${1:-}" == "--norm-filter" ]]; then
    norm
    exit 0
fi

emit() {
    local name="$1"
    local matches="$2"
    if [[ -z "${matches}" ]]; then
        log "  clean    ${name}"
    else
        local n
        n=$(printf '%s\n' "${matches}" | grep -c .)
        log "  found ${n}  ${name}"
        printf '%s\n' "${matches}" | norm >> "${CUR}"
    fi
}

log "scripts/ci/check_layering.sh — architectural-rule audit (ratcheted)"
log "============================================================"

# Import-prefix alternation: both the editable ``src.mriforge.*`` form and
# the installed ``mriforge.*`` form.
PKG='(src\.mriforge|mriforge)'

# 1. Layer-direction rule: lower layers must never import from higher layers.
#    cli/ -> pipelines/ -> application/ -> infrastructure/ -> models/, domain/ -> core/, config/
#    Imports go rightward only.
#
#    ELECTED AWAY (#1398, non-negotiable 17). Five emit blocks that enforced this
#    rule -- infrastructure|application -> pipelines|cli, pipelines -> cli, and
#    models|domain|core -> application|pipelines|cli -- were DELETED here, not
#    kept as defence in depth. Every grep in this file is anchored at ``^``, so a
#    function-local import was structurally invisible to all of them; the two live
#    violations are exactly that shape (col 8 in application/use_cases/
#    hpo_use_case.py). tests/architecture/test_layer_direction.py walks the AST and
#    sees both shapes, so it is now the SOLE owner of import direction, with the
#    plants in tests/unit/architecture/test_layer_direction_detector.py.
#
#    ONE rule stays here, deliberately: models|domain|core -> infrastructure/
#    <non-physics>, below. The two checkers genuinely DISAGREE about it -- this
#    file exempts infrastructure.physics for core/domain and forbids the rest for
#    models/, while the AST table forbids ALL of infrastructure for core/domain
#    and allows it entirely for models/. Electing either one silently changes the
#    policy, so it is an owner decision, not gate work (#1398).

# Models/domain/core may import infrastructure.physics (the SSOT, CLAUDE.md
# pitfall #2) but nothing else from infrastructure/. ERE-safe: grep the
# infrastructure import then filter out the physics exemption with grep -Ev
# (the original used a PCRE ``(?!physics)`` lookahead that grep -E cannot honor).
emit "no models|domain|core -> infrastructure/<non-physics> imports" \
    "$(grep -rEn --include='*.py' "^(from|import)[[:space:]]+${PKG}\.infrastructure\." \
        src/mriforge/models src/mriforge/domain src/mriforge/core \
        | grep -Ev ":(from|import) ${PKG}\.infrastructure\.physics" || true)"

# 2. yaml.safe_load/load forbidden in service/orchestration dirs — build-time
#    YAML routes through mriforge.config.io.
emit "no yaml.safe_load/load in services|coordination|orchestration|application" \
    "$(grep -rEn --include='*.py' "yaml\.(safe_load|load)\(" \
        src/mriforge/application src/mriforge/infrastructure/services \
        src/mriforge/infrastructure/coordination src/mriforge/infrastructure/orchestration || true)"

# 3. Raw torch.fft.fftshift on complex k-space modules (fft2c/ifft2c handle
#    centering). Spectral operators on real feature maps are exempt — scope is
#    the canonical k-space modules only.
emit "no torch.fft.fftshift in models/diffusion or core/metrics" \
    "$(grep -rEn --include='*.py' "torch\.fft\.fftshift\(" \
        src/mriforge/models/diffusion src/mriforge/core/metrics || true)"

# 4. Direct DataLoader instantiation: ELECTED AWAY (#1362, non-negotiable 17).
#    The block here matched the literal name ``DataLoader``, so a real subclass --
#    tio.SubjectsLoader -- was unseen, and with it the one construction site
#    missing worker_init_fn. scripts/ci/check_dataloader_construction_ssot.py now
#    owns this rule: it resolves the binding vocabulary instead of matching a name,
#    and scans all of src/mriforge/ rather than these three directories.

# 5. @register_loss outside src/mriforge/models/losses/ (canonical-home rule).
emit "@register_loss only under models/losses/" \
    "$(grep -rEn --include='*.py' "^@register_loss\(" \
        src/mriforge/infrastructure src/mriforge/models/ot src/mriforge/models/blocks \
        src/mriforge/models/generators src/mriforge/data src/mriforge/application \
        src/mriforge/pipelines src/mriforge/core || true)"

log "============================================================"

# Normalize + dedupe the accumulated violations.
sort -u -o "${CUR}" "${CUR}"
CUR_COUNT=$(grep -c . "${CUR}" || true)

if [[ "${UPDATE_BASELINE}" == "true" ]]; then
    cp "${CUR}" "${BASELINE}"
    log "baseline updated: ${CUR_COUNT} known violation(s) written to ${BASELINE}"
    exit 0
fi

touch "${BASELINE}"
sort -u "${BASELINE}" > "${BASE_SORTED}"

if [[ "${STRICT}" == "true" ]]; then
    if [[ "${CUR_COUNT}" -gt 0 ]]; then
        echo "STRICT: ${CUR_COUNT} layering violation(s) present (no baseline tolerance):"
        sed 's/^/  /' "${CUR}"
        exit 1
    fi
    log "STRICT: zero layering violations."
    exit 0
fi

# Regressions = current violations not in the baseline.
NEW="$(comm -23 "${CUR}" "${BASE_SORTED}" || true)"
# Fixed = baseline entries no longer present (baseline can be tightened).
FIXED="$(comm -13 "${CUR}" "${BASE_SORTED}" || true)"

if [[ -n "${FIXED}" && "${QUIET}" == "false" ]]; then
    log ""
    log "INFO: ${BASELINE} lists violations that are now fixed — run --update-baseline to tighten:"
    printf '%s\n' "${FIXED}" | sed 's/^/  - /'
fi

if [[ -n "${NEW}" ]]; then
    echo ""
    echo "FAIL: new layering violation(s) not in ${BASELINE}:"
    printf '%s\n' "${NEW}" | sed 's/^/  + /'
    echo ""
    echo "Fix the import (invert via a Protocol/DI seam) or, if it is genuinely"
    echo "accepted debt, run: bash scripts/ci/check_layering.sh --update-baseline"
    exit 1
fi

log ""
log "PASS: ${CUR_COUNT} known violation(s) (baseline), 0 new. Layering gate is non-vacuous."
exit 0
