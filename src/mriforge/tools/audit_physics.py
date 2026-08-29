import glob
import os
import re
from collections import defaultdict
from typing import Any

# Define Tiers
TIER_1_GREEN = "🟢 Tier 1 (Mature)"
TIER_2_YELLOW = "🟡 Tier 2 (Partial)"
TIER_3_RED = "🔴 Tier 3 (Model-Only)"


class SimpleConfigParser:
    """Regex-based parser to avoid PyYAML dependency."""

    @staticmethod
    def parse(filepath: str) -> dict[str, Any]:
        """parse.

        Args:
            filepath (str): Description.
        Returns:
            dict[str, Any]: Description.
        """
        with open(filepath) as f:
            content = f.read()

        config = {
            "physics": {
                "data_consistency": {"enabled": False, "weight": 0.0},
                "pinn": {"enabled": False},
                "divergence_free": False,
                "motion_correction": {"enable_motion_correction": False},
            },
            "objectives": {"reconstruction": {"lambda_kspace": 0.0, "lambda_frequency": 0.0}},
            "model": {"model_type": ""},
            "metadata": {"tags": []},
        }

        # Helper to extract value given a key path context
        # 1. Physics -> Data Consistency
        if re.search(r"data_consistency:\s*\n\s+enabled:\s*true", content):
            config["physics"]["data_consistency"]["enabled"] = True

        weight_match = re.search(r"data_consistency:.*?weight:\s*([\d\.]+)", content, re.DOTALL)
        if weight_match:
            config["physics"]["data_consistency"]["weight"] = float(weight_match.group(1))

        # 2. Physics -> PINN
        if re.search(r"pinn:\s*\n\s+enabled:\s*true", content):
            config["physics"]["pinn"]["enabled"] = True

        # 3. Physics -> Motion
        if re.search(r"motion_correction:\s*\n\s+enable_motion_correction:\s*true", content):
            config["physics"]["motion_correction"]["enable_motion_correction"] = True

        # 4. Objectives -> K-Space
        kspace_match = re.search(r"lambda_kspace:\s*([\d\.]+)", content)
        if kspace_match:
            config["objectives"]["reconstruction"]["lambda_kspace"] = float(kspace_match.group(1))

        freq_match = re.search(r"lambda_frequency:\s*([\d\.]+)", content)
        if freq_match:
            config["objectives"]["reconstruction"]["lambda_frequency"] = float(freq_match.group(1))

        # 5. Model Type
        type_match = re.search(r"model_type:\s*(\w+)", content)
        if type_match:
            config["model"]["model_type"] = type_match.group(1)

        return config


def check_physics_compliance(
    config: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    """
    Analyzes a config dict and returns (Tier, Issues, AuditDetails).
    """
    issues = []
    details = {
        "dc_enabled": config["physics"]["data_consistency"]["enabled"],
        "dc_weight": config["physics"]["data_consistency"]["weight"],
        "kspace_loss": max(
            config["objectives"]["reconstruction"]["lambda_kspace"],
            config["objectives"]["reconstruction"]["lambda_frequency"],
        ),
        "pinn_enabled": config["physics"]["pinn"]["enabled"],
        "special_physics": [],
    }

    # 3. Specialized Physics
    if details["pinn_enabled"]:
        details["special_physics"].append("PINN")

    if config["physics"][
        "divergence_free"
    ]:  # Not strictly parsed by regex above but kept for structure
        details["special_physics"].append("DivFree")

    if config["physics"]["motion_correction"]["enable_motion_correction"]:
        details["special_physics"].append("Motion")

    # Classification Logic
    is_dc_active = details[
        "dc_enabled"
    ]  # and details["dc_weight"] > 0 (Implicit in enabled for this parser)
    if details["dc_weight"] == 0.0:
        is_dc_active = False  # Correct for weight 0

    is_kspace_active = details["kspace_loss"] > 0

    if is_dc_active and (
        is_kspace_active or details["pinn_enabled"] or "DivFree" in details["special_physics"]
    ):
        return TIER_1_GREEN, [], details

    elif is_dc_active or is_kspace_active or details["pinn_enabled"]:
        # Has some physics, but not complete suite
        if not is_dc_active:
            issues.append("DC Disabled/Zero")
        if not is_kspace_active:
            issues.append("No K-Space Loss")
        return TIER_2_YELLOW, issues, details

    else:
        issues.append("No Physics Detected")
        return TIER_3_RED, issues, details


import argparse


def main():
    """main.

    Returns:
        Any: Description.
    """
    parser = argparse.ArgumentParser(description="Audit physics compliance.")
    parser.add_argument(
        "--dir", type=str, default="experiments/validated", help="Directory to audit"
    )
    args = parser.parse_args()

    experiments_dir = args.dir
    yaml_files = sorted(glob.glob(os.path.join(experiments_dir, "*.yaml")))

    print(f"Auditing {len(yaml_files)} experiments in {experiments_dir}...\n")

    # Format Headers
    # ID | Tier | DC (W) | K-Loss | Special | Issues
    row_fmt = "{:<55} | {:<20} | {:<10} | {:<8} | {:<15} | {:<20}"
    print(row_fmt.format("Experiment Config", "Tier", "DC (W)", "K-Loss", "Special", "Issues"))
    print("-" * 140)

    tier_counts = defaultdict(int)

    for yaml_path in yaml_files:
        filename = os.path.basename(yaml_path)
        if filename.startswith("dummy"):
            continue

        config = SimpleConfigParser.parse(yaml_path)
        tier, issues, details = check_physics_compliance(config)

        tier_counts[tier] += 1

        # Format columns
        dc_str = f"{'✅' if details['dc_enabled'] else '❌'} ({details['dc_weight']})"
        k_str = f"{details['kspace_loss']}"
        special_str = ", ".join(details["special_physics"]) if details["special_physics"] else "-"
        issue_str = ", ".join(issues)

        print(row_fmt.format(filename, tier, dc_str, k_str, special_str, issue_str))

    print("-" * 140)
    print("\nSummary:")
    for tier, count in sorted(tier_counts.items()):
        print(f"{tier}: {count}")


if __name__ == "__main__":
    main()
