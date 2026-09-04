"""CPU/cgroup probes, after the move from ``infrastructure.logging.provenance``.

These were reporting-only helpers whose ``usable_cores`` docstring already
claimed to be "what dataloader-worker decisions should be judged against" -- and
nothing judged anything against it. They live in ``core`` now so the number
stamped into provenance and the number the worker clamp decides on come from one
implementation, reachable rightward from both ``infrastructure/`` and ``data/``.

The behavioural tests for the cgroup parsers live in the provenance suite (they
moved with the code); this module pins the properties the *decision* path needs.
"""

from __future__ import annotations

from spectramr.core import resources as res


def test_env_int_parses_slurms_heterogeneous_form():
    """``SLURM_CPUS_ON_NODE`` is ``"72(x2)"`` on an uneven allocation."""
    assert res.env_int(("A",), {"A": "72(x2)"}) == 72
    assert res.env_int(("A",), {"A": "8"}) == 8


def test_env_int_walks_the_priority_order_and_skips_unusable_values():
    env = {"FIRST": "", "SECOND": "0", "THIRD": "not-a-number", "FOURTH": "6"}
    assert res.env_int(("FIRST", "SECOND", "THIRD", "FOURTH"), env) == 6
    assert res.env_int(("MISSING",), env) is None


def test_env_int_reads_the_injected_mapping_not_the_process_environment():
    """Injection is what keeps topology resolution a pure function."""
    assert res.env_int(("PATH",), {}) is None


def test_alloc_cpu_env_prefers_per_task_over_on_node():
    """Per-task is THIS process's share; ``*_ON_NODE`` covers every task on it."""
    assert res.ALLOC_CPU_ENV[0] == "SLURM_CPUS_PER_TASK"
    env = {"SLURM_CPUS_PER_TASK": "8", "SLURM_CPUS_ON_NODE": "72"}
    assert res.env_int(res.ALLOC_CPU_ENV, env) == 8


def test_cpu_resources_reports_every_constraint_it_probed():
    info = res.cpu_resources({})
    for key in (
        "model",
        "physical_cores",
        "logical_cores",
        "affinity_cores",
        "allocated_cores",
        "cgroup_quota_cores",
        "usable_cores",
        "max_frequency_mhz",
    ):
        assert key in info


def test_usable_cores_is_the_tightest_constraint_not_the_node_size():
    """A 128-core node granted 2 cores must not read as a 128-core run."""
    info = res.cpu_resources({"SLURM_CPUS_PER_TASK": "2"})
    assert info["allocated_cores"] == 2
    assert info["usable_cores"] <= 2


def test_cpu_resources_is_fail_open():
    """Reporting must never abort a run, even with nothing to read."""
    info = res.cpu_resources({})
    assert info["usable_cores"] is None or info["usable_cores"] > 0


def test_read_first_line_returns_none_for_an_unreadable_path():
    assert res.read_first_line("/nonexistent/definitely/not/here") is None


# --- GPU inventory (#1274) -------------------------------------------------
# `SLURM_GPUS_ON_NODE` used to be read in exactly one place, as
# `provenance.gpu_resources()["allocated_count"]`, and stamped beside the live
# world size where nothing ever subtracted them. These pin the properties the
# DECISION path needs, now that `pipelines/distributed.py` refuses on it.


def test_env_int_source_names_the_variable_it_used():
    """The value alone is not enough: per-node and per-job spell it differently."""
    value, var = res.env_int_source(
        res.ALLOC_GPU_ENV, {"SLURM_GPUS_PER_NODE": "4", "SLURM_GPUS": "8"}
    )
    assert (value, var) == (4, "SLURM_GPUS_PER_NODE")


def test_env_int_source_reports_nothing_found_as_a_pair():
    assert res.env_int_source(res.ALLOC_GPU_ENV, {}) == (None, None)


def test_the_two_gpu_env_tuples_are_disjoint_and_ordered_per_node_first():
    """A per-node var must win, or a 2-node job divides a number already divided."""
    assert not set(res.ALLOC_GPU_ENV_PER_NODE) & set(res.ALLOC_GPU_ENV_JOB)
    assert res.ALLOC_GPU_ENV == res.ALLOC_GPU_ENV_PER_NODE + res.ALLOC_GPU_ENV_JOB


def test_a_per_node_grant_is_used_as_is():
    assert res.allocated_gpus_per_node(2, {"SLURM_GPUS_ON_NODE": "4"}) == (
        4,
        "env:SLURM_GPUS_ON_NODE",
    )


def test_a_job_total_is_divided_by_the_node_count():
    """``--gpus=8`` on 2 nodes is 4 per node.

    Comparing the undivided 8 against one node's rank count is the per-node vs.
    global confusion that makes a CORRECT multi-node run look broken.
    """
    assert res.allocated_gpus_per_node(2, {"SLURM_GPUS": "8"}) == (
        4,
        "derived:SLURM_GPUS/2nodes",
    )


def test_a_job_total_that_does_not_divide_is_left_unknown():
    """Heterogeneous allocation: no per-node number is honest, so return None.

    Rounding would produce a plausible figure that a refusal could then act on —
    a silent substitution (non-negotiable 3) in the one place that must not have
    one, since the consequence is refusing someone's correct job.
    """
    value, source = res.allocated_gpus_per_node(3, {"SLURM_GPUS": "8"})
    assert value is None
    assert "undividable" in source


def test_no_scheduler_grant_says_so_rather_than_returning_zero():
    """``None`` is 'no grant'; ``0`` would read as 'granted nothing' and refuse."""
    value, source = res.allocated_gpus_per_node(1, {})
    assert value is None
    assert source == "unset:no-scheduler-gpu-grant"


def test_visible_gpu_count_is_fail_open():
    """Reporting contract: an answer or ``None``, never an exception.

    ``0`` is a real answer ("torch says no CUDA"); ``None`` means torch could not
    be asked at all. The refusal distinguishes them, so the probe must too.
    """
    count = res.visible_gpu_count()
    assert count is None or (isinstance(count, int) and count >= 0)
