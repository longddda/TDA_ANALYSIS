# -*- coding: utf-8 -*-
"""
Batch network-summary analysis for bee point-cloud trials.
"""

import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

SINGLE_FIGSIZE = (10.0, 8.0)

GROUP_COLORS = ["#E6A15B", "#4D5FB3", "#D74B45", "#4EA28D", "#8BCB67"]
TIME_SERIES_BLUE = "#4D5FB3"
TIME_SERIES_RED = "#D74B45"


def build_epsilon_graph(points: np.ndarray, epsilon: float) -> nx.Graph:
    """Build an epsilon graph for one frame."""
    n = points.shape[0]
    dmat = squareform(pdist(points))

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    # Connect points whose distance is at most epsilon.
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i, j] <= epsilon:
                graph.add_edge(i, j)
    return graph


def gcc_ratio(graph: nx.Graph) -> float:
    """Return the ratio of nodes in the largest connected component."""
    if graph.number_of_nodes() == 0:
        return 0.0
    comps = list(nx.connected_components(graph))
    if not comps:
        return 0.0
    return max(len(c) for c in comps) / graph.number_of_nodes()


def edge_density(graph: nx.Graph) -> float:
    """Return undirected edge density."""
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    return 2 * graph.number_of_edges() / (n * (n - 1))


def analyze_one_trial(pcs: np.ndarray, epsilon: float, down_t: int = 1) -> pd.DataFrame:
    """Compute frame-wise network metrics for one trial."""
    _, _, dim = pcs.shape
    assert dim == 2, f"Expected points shape (T, N, 2), got {pcs.shape}"

    rows = []

    # Build one epsilon graph per sampled frame.
    for t in range(0, pcs.shape[0], down_t):
        graph = build_epsilon_graph(pcs[t], epsilon)
        rows.append(
            {
                "frame": t,
                "gcc_ratio": gcc_ratio(graph),
                "edge_density": edge_density(graph),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
            }
        )
    return pd.DataFrame(rows)


def safe_peak(values: np.ndarray) -> float:
    """Return a NaN-safe peak value."""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(values))


def high_duration(values: np.ndarray, threshold: float, frame_step: float = 1.0) -> float:
    """Return total time spent above a threshold."""
    if len(values) == 0:
        return np.nan
    return float(np.sum(values >= threshold) * frame_step)


def normalized_auc(values: np.ndarray, frames: np.ndarray) -> float:
    """Return AUC normalized by the total trial duration."""
    if len(values) < 2:
        return np.nan
    total_time = frames[-1] - frames[0]
    if total_time <= 0:
        return float(np.nanmean(values))
    auc = np.trapz(values, frames)
    return float(auc / total_time)


def extract_trial_summary(
    df_trial: pd.DataFrame,
    gcc_high_thr: float = 0.8,
    density_high_thr_mode: str = "relative",
    density_relative_ratio: float = 0.8,
    density_absolute_thr: Optional[float] = None,
    down_t: int = 1,
) -> Dict[str, float]:
    """Extract trial-level summary metrics from one time series."""
    frames = df_trial["frame"].to_numpy(dtype=float)
    gcc_vals = df_trial["gcc_ratio"].to_numpy(dtype=float)
    density_vals = df_trial["edge_density"].to_numpy(dtype=float)

    gcc_peak_val = safe_peak(gcc_vals)
    density_peak_val = safe_peak(density_vals)
    gcc_high_dur = high_duration(gcc_vals, threshold=gcc_high_thr, frame_step=down_t)

    if density_high_thr_mode == "relative":
        if np.isnan(density_peak_val):
            density_thr = np.nan
            density_high_dur = np.nan
        else:
            # Define the density threshold relative to the within-trial peak.
            density_thr = density_relative_ratio * density_peak_val
            density_high_dur = high_duration(density_vals, threshold=density_thr, frame_step=down_t)
    elif density_high_thr_mode == "absolute":
        if density_absolute_thr is None:
            raise ValueError("density_absolute_thr must be provided when density_high_thr_mode='absolute'")
        density_thr = float(density_absolute_thr)
        density_high_dur = high_duration(density_vals, threshold=density_thr, frame_step=down_t)
    else:
        raise ValueError("density_high_thr_mode must be 'relative' or 'absolute'")

    # Normalized AUC acts as an average level over time.
    gcc_auc = normalized_auc(gcc_vals, frames)
    density_auc = normalized_auc(density_vals, frames)

    return {
        "n_frames": len(df_trial),
        "start_frame": float(frames[0]) if len(frames) > 0 else np.nan,
        "end_frame": float(frames[-1]) if len(frames) > 0 else np.nan,
        "gcc_peak": gcc_peak_val,
        "density_peak": density_peak_val,
        "gcc_high_threshold": float(gcc_high_thr),
        "density_high_threshold": float(density_thr) if not np.isnan(density_thr) else np.nan,
        "gcc_high_duration": gcc_high_dur,
        "density_high_duration": density_high_dur,
        "gcc_auc_norm": gcc_auc,
        "density_auc_norm": density_auc,
    }


def draw_boxplot_on_ax(
    ax,
    data_dict: Dict[int, List[float]],
    ylabel: str,
    title: str,
):
    """Draw a colored boxplot with overlaid sample points."""
    keys = sorted(data_dict.keys())
    data = [[v for v in data_dict[k] if not np.isnan(v)] for k in keys]
    box_colors = GROUP_COLORS[: len(keys)]

    ax.set_facecolor("white")
    bp = ax.boxplot(
        data,
        tick_labels=[f"{k} bees" for k in keys],
        widths=0.28,
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        boxprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
        medianprops=dict(linewidth=2.0),
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(to_rgba(color, alpha=0.32))
        patch.set_edgecolor(color)
        patch.set_linewidth(2.4)

    for whisker, color in zip(bp["whiskers"], np.repeat(box_colors, 2)):
        whisker.set_color(color)

    for cap, color in zip(bp["caps"], np.repeat(box_colors, 2)):
        cap.set_color(color)

    for median, color in zip(bp["medians"], box_colors):
        median.set_color(color)

    # Overlay jittered points to show individual trials.
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, box_colors), start=1):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.045, 0.045, size=len(vals))
        x = np.full(len(vals), i, dtype=float) + jitter
        ax.scatter(
            x,
            vals,
            s=26,
            color=color,
            alpha=0.62,
            edgecolors=color,
            linewidths=0.8,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12, width=1.8, length=6, labelrotation=20)
    ax.tick_params(axis="y", labelsize=12, width=1.8, length=6)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_color("black")
    ax.grid(False)
    ax.set_title("")


def plot_boxplot(
    data_dict: Dict[int, List[float]],
    ylabel: str,
    title: str,
    save_path: str,
):
    """Create and save a boxplot figure."""
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=300)
    fig.patch.set_facecolor("white")
    draw_boxplot_on_ax(
        ax,
        data_dict=data_dict,
        ylabel=ylabel,
        title=title,
    )
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.close()


def draw_example_time_series_on_ax(ax, df_trial: pd.DataFrame, trial_title: str):
    """Draw GCC ratio and edge density on twin y-axes."""
    frames = df_trial["frame"].to_numpy()
    gcc_vals = df_trial["gcc_ratio"].to_numpy()
    density_vals = df_trial["edge_density"].to_numpy()

    ax.set_facecolor("white")

    # Left axis: GCC ratio.
    line1 = ax.plot(frames, gcc_vals, label="GCC ratio", linewidth=1.8, color=TIME_SERIES_BLUE)[0]
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("GCC ratio", fontsize=11, color=TIME_SERIES_BLUE)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(axis="x", labelsize=9, width=1.2, length=4)
    ax.tick_params(axis="y", labelsize=9, width=1.2, length=4, colors=TIME_SERIES_BLUE)
    ax.grid(False)
    ax.set_title(trial_title, fontsize=12, color=TIME_SERIES_RED)

    # Right axis: edge density.
    ax2 = ax.twinx()
    line2 = ax2.plot(frames, density_vals, label="Edge density", linewidth=1.8, color=TIME_SERIES_RED)[0]
    ax2.set_ylim(-0.05, 0.55)
    ax2.set_ylabel("Edge density", fontsize=11, color=TIME_SERIES_RED)
    ax2.tick_params(axis="y", labelsize=9, width=1.2, length=4, colors=TIME_SERIES_RED)

    for spine in ["left", "bottom", "top"]:
        ax.spines[spine].set_linewidth(1.6)
        ax.spines[spine].set_visible(True)
    ax.spines["left"].set_color(TIME_SERIES_BLUE)
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_visible(False)

    ax2.spines["right"].set_linewidth(1.6)
    ax2.spines["right"].set_color(TIME_SERIES_RED)
    ax2.spines["top"].set_linewidth(1.6)
    ax2.spines["top"].set_color("black")
    ax2.spines["top"].set_visible(True)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    ax.legend(
        [line1, line2],
        ["GCC ratio", "Edge density"],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )


def plot_example_time_series(df_trial: pd.DataFrame, trial_title: str, save_path: str):
    """Create and save one example time-series figure."""
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=300)
    fig.patch.set_facecolor("white")
    draw_example_time_series_on_ax(ax, df_trial=df_trial, trial_title=trial_title)
    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.close()


def main():
    data_root = r"./data"
    out_root = r"./network_summary_results_2"

    bee_groups = [20, 30, 40]
    epsilon = 80.0
    down_t = 1

    gcc_high_thr = 0.8
    density_high_thr_mode = "relative"
    density_relative_ratio = 0.8
    density_absolute_thr = None

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "trial_level"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "figures"), exist_ok=True)

    # Save the analysis configuration for reproducibility.
    config = {
        "data_root": data_root,
        "bee_groups": bee_groups,
        "epsilon": epsilon,
        "down_t": down_t,
        "gcc_high_thr": gcc_high_thr,
        "density_high_thr_mode": density_high_thr_mode,
        "density_relative_ratio": density_relative_ratio,
        "density_absolute_thr": density_absolute_thr,
    }
    with open(os.path.join(out_root, "analysis_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary_rows = []
    example_plot_done = set()
    figure_sources = []

    for bee_n in bee_groups:
        group_dir = os.path.join(data_root, str(bee_n))
        trial_files = sorted(glob.glob(os.path.join(group_dir, "*.npy")))

        if len(trial_files) == 0:
            print(f"[Warning] No .npy files found in {group_dir}")
            continue

        print(f"\n=== Processing {bee_n} bees: {len(trial_files)} trials ===")
        trial_out_dir = os.path.join(out_root, "trial_level", str(bee_n))
        os.makedirs(trial_out_dir, exist_ok=True)

        for trial_idx, trial_file in enumerate(trial_files, start=1):
            print(f"  -> Trial {trial_idx}: {os.path.basename(trial_file)}")

            pcs = np.load(trial_file)
            df_trial = analyze_one_trial(pcs=pcs, epsilon=epsilon, down_t=down_t)
            df_trial["bee_count"] = bee_n
            df_trial["trial_id"] = trial_idx
            df_trial["trial_file"] = os.path.basename(trial_file)

            # Save frame-level metrics for this trial.
            base_name = os.path.splitext(os.path.basename(trial_file))[0]
            df_trial.to_csv(os.path.join(trial_out_dir, f"{base_name}_timeseries.csv"), index=False)

            # Save one example time-series figure per group.
            if bee_n not in example_plot_done:
                fig_name = f"{bee_n}_bees_example_timeseries.svg"
                plot_example_time_series(
                    df_trial=df_trial,
                    trial_title=f"{bee_n} bees - example trial",
                    save_path=os.path.join(out_root, "figures", fig_name),
                )
                figure_sources.append(
                    {
                        "figure_file": fig_name,
                        "figure_type": "example_timeseries",
                        "metric": "gcc_ratio,edge_density",
                        "bee_count": bee_n,
                        "source_trial_file": os.path.basename(trial_file),
                    }
                )
                example_plot_done.add(bee_n)

            summary = extract_trial_summary(
                df_trial=df_trial,
                gcc_high_thr=gcc_high_thr,
                density_high_thr_mode=density_high_thr_mode,
                density_relative_ratio=density_relative_ratio,
                density_absolute_thr=density_absolute_thr,
                down_t=down_t,
            )
            summary["bee_count"] = bee_n
            summary["trial_id"] = trial_idx
            summary["trial_file"] = os.path.basename(trial_file)
            summary_rows.append(summary)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_root, "trial_summary_metrics.csv"), index=False)

    figure_metric_map = {
        "gcc_peak": "GCC peak",
        "density_peak": "Edge density peak",
        "gcc_high_duration": "GCC high-duration",
        "density_high_duration": "Density high-duration",
        "gcc_auc_norm": "Normalized GCC AUC",
        "density_auc_norm": "Normalized density AUC",
    }

    for metric, ylabel in figure_metric_map.items():
        # Collect one value list per bee-count group.
        group_dict = {
            bee_n: df_summary.loc[df_summary["bee_count"] == bee_n, metric].tolist() for bee_n in bee_groups
        }
        fig_name = f"{metric}_boxplot.svg"
        plot_boxplot(
            data_dict=group_dict,
            ylabel=ylabel,
            title=f"{ylabel} across different bee group sizes",
            save_path=os.path.join(out_root, "figures", fig_name),
        )
        figure_sources.append(
            {
                "figure_file": fig_name,
                "figure_type": "boxplot",
                "metric": metric,
                "bee_count": "20,30,40",
                "source_trial_file": "trial_summary_metrics.csv",
            }
        )

    # Save a figure-to-data mapping table.
    pd.DataFrame(figure_sources).to_csv(
        os.path.join(out_root, "figures", "figure_data_sources.csv"),
        index=False,
    )

    print("\n======================================")
    print("Network summary analysis finished.")
    print(f"Results saved to: {out_root}")
    print("Generated:")
    print("- trial_level/*_timeseries.csv")
    print("- trial_summary_metrics.csv")
    print("- figures/*_boxplot.svg")
    print("- figures/*_example_timeseries.svg")
    print("- figures/figure_data_sources.csv")
    print("======================================")


if __name__ == "__main__":
    main()
