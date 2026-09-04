import argparse
import logging
import os
from glob import glob

import pandas as pd

logger = logging.getLogger(__name__)


def analyze_gradient_logs(log_dir: str) -> dict[str, object]:
    """Analyze gradient logs and return a structured summary.

    Args:
        log_dir (str): The path to the log directory containing the log files.

    Returns:
        dict[str, object]: Summary metadata with detection flags.

    """
    logger.info(f"Analyzing gradient logs in: {log_dir}")

    # Find log files, supporting both .xlsx and .csv
    issues_files = glob(os.path.join(log_dir, "gradient_issues_*.xlsx")) + glob(
        os.path.join(log_dir, "gradient_issues_*.csv"),
    )
    summary_files = glob(os.path.join(log_dir, "gradient_summary_*.xlsx")) + glob(
        os.path.join(log_dir, "gradient_summary_*.csv"),
    )

    result: dict[str, object] = {
        "log_dir": log_dir,
        "issues_files": issues_files,
        "summary_files": summary_files,
        "exploding_detected": False,
        "vanishing_detected": False,
        "nan_inf_detected": False,
        "summary_rows": 0,
    }

    if not issues_files and not summary_files:
        logger.info("No gradient log files found in the specified directory.")
        return result

    # --- Issues Analysis ---
    exploding_detected = False
    vanishing_detected = False
    nan_inf_detected = False

    # Analyze issues file
    if issues_files:
        logger.info(f"\n--- Analyzing {len(issues_files)} Gradient Issues File(s) ---")
        for f_path in issues_files:
            logger.info(f"Reading issues from: {f_path}")
            try:
                if f_path.endswith(".xlsx"):
                    issues_sheets = pd.read_excel(f_path, sheet_name=None)
                    if (
                        "Vanishing_Gradients" in issues_sheets
                        and not issues_sheets["Vanishing_Gradients"].empty
                    ):
                        vanishing_detected = True
                        logger.info(
                            f"\nFound Vanishing Gradients in {os.path.basename(f_path)}:",
                        )
                        logger.info(issues_sheets["Vanishing_Gradients"].head())
                    if (
                        "Exploding_Gradients" in issues_sheets
                        and not issues_sheets["Exploding_Gradients"].empty
                    ):
                        exploding_detected = True
                        logger.info(
                            f"\nFound Exploding Gradients in {os.path.basename(f_path)}:",
                        )
                        logger.info(issues_sheets["Exploding_Gradients"].head())
                    if "Summary" in issues_sheets and not issues_sheets["Summary"].empty:
                        summary_issue_df = issues_sheets["Summary"]
                        if (
                            "total_nan_inf_events" in summary_issue_df.columns
                            and summary_issue_df["total_nan_inf_events"].sum() > 0
                        ):
                            nan_inf_detected = True
                            logger.info(
                                f"\nFound NaN/Inf Gradients in {os.path.basename(f_path)}:",
                            )
                            logger.info(summary_issue_df[["total_nan_inf_events"]])
                elif f_path.endswith(".csv"):
                    # Handle CSV files, inferring issue type from filename
                    issues_df = pd.read_csv(f_path)
                    filename = os.path.basename(f_path).lower()
                    if "vanishing" in filename:
                        vanishing_detected = True
                        logger.info(f"\nFound Vanishing Gradients in {filename}:")
                        logger.info(issues_df.head())
                    elif "exploding" in filename:
                        exploding_detected = True
                        logger.info(f"\nFound Exploding Gradients in {filename}:")
                        logger.info(issues_df.head())
                    elif "summary" in filename:
                        if (
                            "total_nan_inf_events" in issues_df.columns
                            and issues_df["total_nan_inf_events"].sum() > 0
                        ):
                            nan_inf_detected = True
                            logger.info(f"\nFound NaN/Inf Gradients in {filename}:")
                            logger.info(issues_df[["total_nan_inf_events"]])
                    # Fallback for generic issue files
                    elif "issue_type" in issues_df.columns:
                        if "vanishing" in issues_df["issue_type"].str.lower().unique():
                            vanishing_detected = True
                        if "exploding" in issues_df["issue_type"].str.lower().unique():
                            exploding_detected = True
                        if (
                            "nan" in issues_df["issue_type"].str.lower().unique()
                            or "inf" in issues_df["issue_type"].str.lower().unique()
                        ):
                            nan_inf_detected = True
                        logger.info(f"\nFound issues in generic file {filename}:")
                        logger.info(issues_df.head())
                    else:
                        logger.info(
                            f"Warning: Could not determine issue type for CSV file {filename}.",
                        )

            except Exception as e:
                logger.info(f"Could not read or process {f_path}: {e}")

    # --- Summary Analysis ---
    all_summary_df = pd.DataFrame()
    # Analyze summary file
    if summary_files:
        logger.info(
            f"\n--- Analyzing {len(summary_files)} Overall Gradient Summary File(s) ---",
        )
        for f_path in summary_files:
            logger.info(f"Reading summary from: {f_path}")
            try:
                if f_path.endswith(".xlsx"):
                    summary_df = pd.read_excel(f_path)
                elif f_path.endswith(".csv"):
                    summary_df = pd.read_csv(f_path)

                if not summary_df.empty:
                    all_summary_df = pd.concat(
                        [all_summary_df, summary_df],
                        ignore_index=True,
                    )
            except Exception as e:
                logger.info(f"Could not read summary sheet from {f_path}: {e}")

    if not all_summary_df.empty:
        logger.info("\n--- Overall Gradient Stats (from summary files) ---")
        # Top 5 steps with highest total gradient norm
        if "total_grad_norm" in all_summary_df.columns:
            top_5_high_norm = all_summary_df.nlargest(5, "total_grad_norm")
            logger.info("\nTop 5 Batches with Highest Total Gradient Norm:")
            logger.info(top_5_high_norm[["epoch", "batch", "model_name", "total_grad_norm"]])

        # Top 5 steps with highest zero gradient ratio
        if "zero_grad_ratio" in all_summary_df.columns:
            top_5_zero_ratio = all_summary_df.nlargest(5, "zero_grad_ratio")
            logger.info("\nTop 5 Batches with Highest Zero Gradient Ratio:")
            logger.info(top_5_zero_ratio[["epoch", "batch", "model_name", "zero_grad_ratio"]])

    logger.info("\n--- Recommendations ---")
    if exploding_detected:
        logger.info("- **Exploding Gradients Detected:**")
        logger.info("  - Try reducing the learning rate (`--lr`).")
        logger.info(
            "  - Try decreasing the gradient clipping value (`--gradient_clip_threshold`).",
        )
        logger.info(
            "  - Ensure you are using a stable optimizer like Adam with the `--improved_optimizer_params` flag.",
        )
    if vanishing_detected:
        logger.info("- **Vanishing Gradients Detected:**")
        logger.info("  - Try increasing the learning rate (`--lr`).")
        logger.info(
            "  - Check your model architecture for issues like too many layers without residual connections.",
        )
        logger.info(
            "  - Consider using a different weight initialization method (`--weight_init_method`).",
        )
    if nan_inf_detected:
        logger.info("- **NaN/Inf Gradients Detected:**")
        logger.info("  - This is a critical issue, often caused by numerical instability.")
        logger.info("  - Drastically reduce the learning rate (`--lr`).")
        logger.info(
            "  - Check for division by zero or other unstable operations in your model or loss functions.",
        )
    if (
        not all_summary_df.empty
        and "zero_grad_ratio" in all_summary_df.columns
        and all_summary_df["zero_grad_ratio"].max() > 0.5
    ):
        logger.info("- **High Zero Gradient Ratio (Dead Neurons) Detected:**")
        logger.info("  - This can be caused by the ReLU activation function.")
        logger.info("  - Try using a different activation function, like LeakyReLU.")
        logger.info("  - This can also be a sign of a learning rate that is too high.")

    result["exploding_detected"] = exploding_detected
    result["vanishing_detected"] = vanishing_detected
    result["nan_inf_detected"] = nan_inf_detected
    result["summary_rows"] = int(all_summary_df.shape[0])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze gradient logs from GAN training.",
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to the directory containing the gradient log files.",
    )
    args = parser.parse_args()
    analyze_gradient_logs(args.log_dir)
