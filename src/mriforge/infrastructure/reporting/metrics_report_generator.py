"""Metrics Report Generator - Creates HTML dashboards and plots from training metrics."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class MetricsReportGenerator:
    """Generate HTML reports and visualization plots from training metrics CSV.

    Creates:
    - Matplotlib plots for loss, PSNR, SSIM, LPIPS, and other metrics
    - HTML dashboard combining all plots
    - JSON export of metrics data with aggregations
    - Summary statistics table
    """

    def __init__(
        self,
        log_dir: str | Path,
        output_dir: str | Path | None = None,
        figsize: tuple[int, int] = (12, 6),
        dpi: int = 100,
    ):
        """Initialize report generator.

        Args:
            log_dir: Directory containing validation_metrics.csv
            output_dir: Directory to save plots and HTML (defaults to log_dir/reports)
            figsize: Figure size for plots
            dpi: DPI for saved PNG images
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir) if output_dir else self.log_dir / "reports"
        self.figsize = figsize
        self.dpi = dpi

        # Create output directories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        sns.set_style("darkgrid")
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["font.size"] = 10

        # Data storage
        self.df: pd.DataFrame | None = None
        self.metrics: dict = {}

    def generate(self) -> str:
        """Generate all reports and return path to HTML dashboard.

        Returns:
            Path to generated metrics_report.html file
        """
        try:
            # Load metrics
            self._load_metrics()

            if self.df is None or self.df.empty:
                # F-METRICREPORT-INFO / 2026-05-20 — this fires for any
                # YAML with ``validation.enabled=false`` (e.g.
                # F-LATENTFLOW-VAL'd experiment_87) where no CSV is
                # written. The "missing report" is expected behaviour
                # there, not a failure. Demote to info; the upstream
                # `_load_metrics` log already records the missing CSV.
                logger.info(
                    "No metrics data found, skipping report generation "
                    "(validation may have been disabled in this YAML)."
                )
                return str(self.output_dir)

            # Generate individual plots
            self._plot_losses()
            self._plot_psnr_ssim()
            self._plot_lpips()
            self._plot_discriminator_scores()
            self._plot_metric_summary()

            # Extract statistics
            stats = self._compute_statistics()

            # Generate HTML dashboard
            html_path = self._generate_html_dashboard(stats)

            # Export JSON metrics
            json_path = self._export_json_metrics(stats)

            logger.info("✓ Reports generated successfully")
            logger.info(f"  - Dashboard: {html_path}")
            logger.info(f"  - Plots: {self.plots_dir}")
            logger.info(f"  - JSON export: {json_path}")

            return str(html_path)

        except Exception as e:
            logger.error(f"Failed to generate metrics report: {e}", exc_info=True)
            return ""

    def _load_metrics(self) -> None:
        """Load metrics from CSV file.

        Handles schema expansion where later rows may have more columns
        than the original header (e.g., new metrics added mid-training).
        """
        csv_path = self.log_dir / "validation_metrics.csv"

        if not csv_path.exists():
            # F-METRICREPORT-INFO / 2026-05-20 — see ``generate``;
            # missing CSV is expected for validation-disabled YAMLs.
            logger.info(f"Metrics CSV not found (validation likely disabled): {csv_path}")
            return

        try:
            # First attempt: standard read
            self.df = pd.read_csv(csv_path, on_bad_lines="error")
            logger.info(f"Loaded metrics from {csv_path}")
        except pd.errors.ParserError:
            # Schema expansion detected — rows have more columns than header.
            # Re-read with max-column detection.
            logger.debug(
                f"Schema expansion detected in {csv_path}, re-reading with max-column detection"
            )
            try:
                import csv as _csv

                with open(csv_path, newline="") as fh:
                    reader = _csv.reader(fh)
                    header = next(reader)
                    max_cols = len(header)
                    for row in reader:
                        max_cols = max(max_cols, len(row))

                # Pad header with placeholder names for extra columns
                if max_cols > len(header):
                    header.extend([f"metric_{i}" for i in range(len(header), max_cols)])

                self.df = pd.read_csv(
                    csv_path,
                    names=header,
                    header=0,
                    on_bad_lines="skip",
                )
                logger.info(f"Loaded metrics from {csv_path} (schema expanded: {max_cols} cols)")
            except Exception as e2:
                logger.error(f"Failed to load metrics CSV after retry: {e2}")
                return
        except Exception as e:
            logger.error(f"Failed to load metrics CSV: {e}")
            return

        if self.df is not None:
            logger.debug(f"Metrics shape: {self.df.shape}, cols: {list(self.df.columns)}")

    # ──────────────────────────────────────────────────────────────────
    # All four plot methods below delegate to plot_training_curves so
    # they share EMA smoothing, auto log/linear y-axis, final-value
    # annotations, and a consistent Okabe-Ito palette.
    # ──────────────────────────────────────────────────────────────────

    def _plot_losses(self) -> None:
        if self.df is None:
            return
        from mriforge.infrastructure.reporting._training_curves import plot_training_curves

        loss_cols = [c for c in self.df.columns if "loss" in c.lower()][:6]
        if not loss_cols:
            logger.debug("No loss columns found")
            return
        plot_training_curves(
            self.df,
            loss_cols,
            title="Training losses",
            ylabel="Loss (auto log)",
            save_path=self.plots_dir / "losses.png",
            higher_is_better=False,
            column_legend_names={c: c.replace("val_", "") for c in loss_cols},
        )

    def _plot_psnr_ssim(self) -> None:
        if self.df is None:
            return
        from mriforge.infrastructure.reporting._training_curves import plot_training_curves

        psnr_cols = [c for c in self.df.columns if "psnr" in c.lower()][:4]
        ssim_cols = [c for c in self.df.columns if "ssim" in c.lower()][:4]
        if psnr_cols:
            plot_training_curves(
                self.df,
                psnr_cols,
                title="PSNR trajectory",
                ylabel="PSNR (dB)",
                save_path=self.plots_dir / "psnr.png",
                higher_is_better=True,
                column_legend_names={c: c.replace("val_", "") for c in psnr_cols},
            )
        if ssim_cols:
            plot_training_curves(
                self.df,
                ssim_cols,
                title="SSIM trajectory",
                ylabel="SSIM",
                save_path=self.plots_dir / "ssim.png",
                higher_is_better=True,
                log_y=False,  # SSIM ∈ [0, 1] — never log
                column_legend_names={c: c.replace("val_", "") for c in ssim_cols},
            )
        # Back-compat alias for the HTML dashboard which links psnr_ssim.png
        # by writing a side-by-side composite.
        if psnr_cols and ssim_cols:
            self._plot_psnr_ssim_composite(psnr_cols, ssim_cols)

    def _plot_psnr_ssim_composite(
        self,
        psnr_cols: list[str],
        ssim_cols: list[str],
    ) -> None:
        """Back-compat: composite (PSNR | SSIM) figure used by HTML."""
        from mriforge.infrastructure.reporting._training_curves import (
            ema as _ema,  # noqa: F401
        )
        from mriforge.infrastructure.reporting.style import (
            colour_for,
            use_default_style,
        )

        use_default_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=110)
        for ax, cols, ylabel, title, hib in (
            (ax1, psnr_cols, "PSNR (dB)", "PSNR", True),
            (ax2, ssim_cols, "SSIM", "SSIM", True),
        ):
            for i, col in enumerate(cols):
                vals = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=float)
                steps = self.df.index.to_numpy()
                from mriforge.infrastructure.reporting._training_curves import ema

                smoothed = ema(vals)
                colour = colour_for(col, fallback_index=i + 1)
                ax.plot(steps, vals, color=colour, alpha=0.18, linewidth=0.8)
                ax.plot(steps, smoothed, color=colour, linewidth=1.8, label=col.replace("val_", ""))
            ax.set_xlabel("Step", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(
                f"{title} ({'higher better' if hib else 'lower better'})",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", frameon=False, fontsize=9)
        plt.tight_layout()
        save_path = self.plots_dir / "psnr_ssim.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved composite PSNR/SSIM plot to %s", save_path)

    def _plot_lpips(self) -> None:
        if self.df is None:
            return
        from mriforge.infrastructure.reporting._training_curves import plot_training_curves

        lpips_cols = [c for c in self.df.columns if "lpips" in c.lower()][:4]
        if not lpips_cols:
            logger.debug("No LPIPS columns found")
            return
        plot_training_curves(
            self.df,
            lpips_cols,
            title="Perceptual loss (LPIPS)",
            ylabel="LPIPS",
            save_path=self.plots_dir / "lpips_trend.png",
            higher_is_better=False,
            log_y=False,
            column_legend_names={c: c.replace("val_", "") for c in lpips_cols},
        )

    def _plot_discriminator_scores(self) -> None:
        if self.df is None:
            return
        from mriforge.infrastructure.reporting._training_curves import plot_training_curves

        score_cols = [c for c in self.df.columns if "score" in c.lower()][:6]
        if not score_cols:
            logger.debug("No discriminator score columns found")
            return
        plot_training_curves(
            self.df,
            score_cols,
            title="Discriminator scores",
            ylabel="D-score",
            save_path=self.plots_dir / "discriminator_scores.png",
            higher_is_better=None,
            log_y=False,
            column_legend_names={c: c.replace("val_", "") for c in score_cols},
        )

    def _plot_metric_summary(self) -> None:
        """Plot summary of all key metrics in one view."""
        if self.df is None or self.df.empty:
            return

        # Get best and worst values for key metrics
        key_metrics = {}
        for pattern in ["psnr", "ssim", "lpips", "loss"]:
            matching = [col for col in self.df.columns if pattern in col.lower()]
            if matching:
                col = matching[0]
                # [FIX] Handle type conversions: convert last value to float, handle NaN
                last_val = self.df[col].iloc[-1]
                try:
                    # Coerce to float, handling string representations and NaN
                    numeric_val = float(last_val) if last_val is not None else 0.0
                    # Check if the converted value is actually a number (not NaN)
                    if isinstance(numeric_val, float) and not (
                        numeric_val != numeric_val
                    ):  # NaN check
                        key_metrics[col] = numeric_val
                    else:
                        key_metrics[col] = 0.0
                except (ValueError, TypeError):
                    # If conversion fails (e.g., string like "N/A"), use 0
                    key_metrics[col] = 0.0

        if not key_metrics:
            logger.debug("No key metrics to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(key_metrics.keys())
        values = list(key_metrics.values())

        # Normalize to 0-1 range for visualization
        # [FIX] Filter out zero/invalid values before calling max()
        numeric_values = [v for v in values if isinstance(v, (int, float)) and v == v]
        if numeric_values and max(numeric_values) != 0:
            max_val = max(numeric_values)
            values_norm = [v / max_val if v > 0 else 0 for v in values]
        else:
            values_norm = values

        bars = ax.bar(range(len(names)), values_norm, color=sns.color_palette("husl", len(names)))

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values, strict=False)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([name.replace("val_", "") for name in names], rotation=45, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Final Metrics Summary (Last Iteration)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = self.plots_dir / "metrics_summary.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.debug(f"Saved summary plot to {save_path}")

    def _compute_statistics(self) -> dict:
        """Compute statistics from metrics dataframe."""
        if self.df is None or self.df.empty:
            return {}

        stats = {
            "total_iterations": len(self.df),
            "num_metrics": len([col for col in self.df.columns if "val_" in col]),
            "epoch_range": (
                f"{self.df['epoch'].min():.0f} - {self.df['epoch'].max():.0f}"
                if "epoch" in self.df.columns
                else "N/A"
            ),
        }

        # Best values for key metrics
        for pattern in ["psnr", "ssim", "lpips"]:
            matching = [col for col in self.df.columns if pattern in col.lower()]
            if matching:
                col = matching[0]
                # [FIX] Convert column to numeric, coercing errors to NaN
                col_numeric = pd.to_numeric(self.df[col], errors="coerce")
                # Filter out NaN values
                col_numeric_clean = col_numeric.dropna()

                if col_numeric_clean.empty:
                    stats[f"best_{pattern}"] = "N/A"
                    continue

                if pattern == "lpips":  # Lower is better
                    best_val = col_numeric_clean.min()
                    best_iter = col_numeric_clean.idxmin()
                else:  # Higher is better
                    best_val = col_numeric_clean.max()
                    best_iter = col_numeric_clean.idxmax()

                try:
                    stats[f"best_{pattern}"] = f"{float(best_val):.4f} (iter {int(best_iter)})"
                except (ValueError, TypeError):
                    stats[f"best_{pattern}"] = "N/A"

        return stats

    def _generate_html_dashboard(self, stats: dict) -> Path:
        """Generate HTML dashboard with embedded plots."""
        html_path = self.output_dir / "metrics_report.html"

        # Collect plot files
        plot_files = sorted(self.plots_dir.glob("*.png")) if self.plots_dir.exists() else []

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Metrics Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }}
        .stat-card .label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .stat-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            font-family: 'Courier New', monospace;
        }}
        .content {{
            padding: 30px;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .plot-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 25px rgba(0, 0, 0, 0.15);
        }}
        .plot-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .plot-title {{
            padding: 15px;
            background: #f8f9fa;
            border-top: 1px solid #ddd;
            font-weight: 600;
            color: #333;
            text-align: center;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
        }}
        h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Training Metrics Report</h1>
            <p>Real-time performance visualization and statistics</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="label">Total Iterations</div>
                <div class="value">{stats.get("total_iterations", "N/A")}</div>
            </div>
            <div class="stat-card">
                <div class="label">Epoch Range</div>
                <div class="value">{stats.get("epoch_range", "N/A")}</div>
            </div>
            <div class="stat-card">
                <div class="label">Metrics Tracked</div>
                <div class="value">{stats.get("num_metrics", "N/A")}</div>
            </div>
"""

        # Add best metric cards
        for key in ["psnr", "ssim", "lpips"]:
            if f"best_{key}" in stats:
                html_content += f"""            <div class="stat-card">
                <div class="label">Best {key.upper()}</div>
                <div class="value">{stats[f"best_{key}"]}</div>
            </div>
"""

        html_content += """        </div>

        <div class="content">
            <h2>📊 Metric Visualizations</h2>
            <div class="plots-grid">
"""

        # Add plot images
        for plot_file in plot_files:
            rel_path = plot_file.relative_to(self.output_dir)
            plot_name = plot_file.stem.replace("_", " ").title()
            html_content += f"""                <div class="plot-container">
                    <img src="{rel_path}" alt="{plot_name}">
                    <div class="plot-title">{plot_name}</div>
                </div>
"""

        html_content += """            </div>
        </div>

        <div class="footer">
            <p>Generated automatically during training | Metrics captured in real-time</p>
        </div>
    </div>
</body>
</html>
"""

        html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Generated HTML dashboard at {html_path}")
        return html_path

    def _export_json_metrics(self, stats: dict) -> Path:
        """Export metrics as JSON with aggregations."""
        json_path = self.output_dir / "metrics_export.json"

        export_data = {
            "metadata": {
                "source": str(self.log_dir),
                "generated": pd.Timestamp.now().isoformat(),
            },
            "statistics": stats,
            "raw_data": {},
        }

        if self.df is not None and not self.df.empty:
            # Export raw metrics as records
            export_data["raw_data"] = self.df.to_dict("records")

        json_path.write_text(json.dumps(export_data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Exported JSON metrics to {json_path}")
        return json_path


__all__ = ["MetricsReportGenerator"]
