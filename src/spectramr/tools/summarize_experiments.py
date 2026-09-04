import glob
import os
from pathlib import Path

import yaml


def get_physics_components(config: dict) -> str:
    """Extract physics components from config.

    Args:
        config: Configuration dictionary

    Returns:
        Comma-separated string of physics components or "None"
    """
    physics = config.get("physics", {})
    components = []

    if physics.get("data_consistency", {}).get("enabled"):
        components.append("DC")

    if physics.get("kspace", {}).get("enable_kspace_recon"):
        components.append("K-Space")

    if physics.get("pinn", {}).get("enabled"):
        components.append("PINN")

    if physics.get("regularization", {}).get("enable_tv"):
        components.append("TV")

    if not components:
        return "None"
    return ", ".join(components)


def summarize_experiments() -> None:
    """Summarize all experiments in YAML format and print as Markdown table."""
    root_dir = "experiments/validated"
    data = []

    files = sorted(glob.glob(os.path.join(root_dir, "experiment_*.yaml")))

    for f in files:
        try:
            with open(f) as file:
                config = yaml.safe_load(file)

            # Name
            name = Path(f).stem

            # Model
            model_conf = config.get("model", {})
            model_name = model_conf.get("model_type", "Unknown")

            # Input/Output
            # Infer from dataset type or model input_type
            data_conf = config.get("data", {})
            dataset_type = data_conf.get("dataset_type", "image")

            input_type = model_conf.get("input_type", dataset_type)
            if "kspace" in str(dataset_type) or "kspace" in str(input_type):
                input_type = "K-Space"
            else:
                input_type = "Image"

            # Output is usually Image for these tasks, unless specified
            output_type = "Image"

            # Task (Paradigm)
            training_conf = config.get("training", {})
            strategy = training_conf.get("strategy_class")

            if strategy:
                # Extract short name from strategy class path
                # e.g. "spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy" -> "gan"
                task = strategy.split(".")[-2] if "." in strategy else strategy
            else:
                legacy_mode = training_conf.get("training_mode", "reconstruction")
                task = f"{legacy_mode} (legacy)"

            # Physics
            physics = get_physics_components(config)

            data.append(
                {
                    "Experiment File": name,
                    "Model": model_name,
                    "Input": input_type,
                    "Output": output_type,
                    "Task": task,
                    "Physics": physics,
                }
            )

        except Exception as e:
            print(f"Error parsing {f}: {e}")

    # Manual Markdown Table Generation
    header = "| Experiment File | Model | Input | Output | Task | Physics |"
    divider = "| :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(divider)

    for row in data:
        line = f"| {row['Experiment File']} | {row['Model']} | {row['Input']} | {row['Output']} | {row['Task']} | {row['Physics']} |"
        print(line)


if __name__ == "__main__":
    summarize_experiments()
