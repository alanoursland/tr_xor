import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from analysis.utils import targets_to_class_labels, extract_activation_type, get_linear_layers
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from analysis.geometry import compute_angles_between, compute_norm_ratios


def plot_hyperplanes(weights, biases, x, y, title, filename=None):
    input_dim = weights.shape[-1]
    if input_dim != 2:
        # print(f"⚠️ Skipping plot: hyperplane visualization only supported for 2D inputs (got input_dim={input_dim})")
        return

    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    weights = weights.detach().cpu()  # (n_units, 2)
    biases = biases.detach().cpu()  # (n_units,)
    mean = torch.zeros(input_dim)

    plt.figure(figsize=(6, 6))

    # XOR input points
    y_labels = targets_to_class_labels(y_cpu.unsqueeze(1) if y_cpu.ndim == 1 else y_cpu)
    for xi, yi_label in zip(x_cpu, y_labels):
        marker = "o" if yi_label == 0 else "^"
        plt.scatter(xi[0], xi[1], marker=marker, s=100, color="black", edgecolors="k", linewidths=1)

    # Draw each hyperplane and normal
    for i, (W, b) in enumerate(zip(weights, biases)):
        norm = torch.norm(W)
        normal = W / norm

        distance = (W @ mean + b) / norm
        projection_on_plane = mean - distance * normal
        perp = torch.tensor([-normal[1], normal[0]])

        scale_factor = 5
        pt1 = projection_on_plane + perp * scale_factor
        pt2 = projection_on_plane - perp * scale_factor

        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color="black", linewidth=1.5, linestyle="--", label=f"Neuron {i}")

        plt.arrow(
            projection_on_plane[0].item(),
            projection_on_plane[1].item(),
            normal[0].item() * 0.5,
            normal[1].item() * 0.5,
            head_width=0.15,
            head_length=0.2,
            fc="#333333",
            ec="#333333",
            alpha=1.0,
            length_includes_head=True,
            width=0.03,
            zorder=3,
        )

    # Final plot adjustments
    plt.title(title, fontsize=16, weight="bold", pad=12)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.axis("equal")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tight_layout()

    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()


def plot_epoch_distribution(
    run_results: List[Dict[str, Any]], plot_config: Dict[str, Any], output_dir: Path, experiment_name: str
) -> None:
    """
    Plot a sorted curve of training epoch counts across all runs.

    Args:
        run_results: List of results from all runs
        plot_config: Dictionary with plot settings (from configure_analysis_from_config)
        output_dir: Path to save plots if enabled
    """
    epoch_counts = [r.get("epochs_completed", 0) for r in run_results]
    sorted_epochs = sorted(epoch_counts)

    # Plot settings
    plt.style.use(plot_config.get("style", "default"))
    dpi = plot_config.get("dpi", 300)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(sorted_epochs, marker="o", linestyle="-", linewidth=1.5)
    ax.set_title(f"Sorted Epoch Counts: {experiment_name}")
    ax.set_xlabel("Run (sorted)")
    ax.set_ylabel("Epochs Completed")
    ax.grid(True)

    plt.tight_layout()

    # Save if configured
    if plot_config.get("save_plots", False):
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"epoch_distribution.{plot_config.get('format', 'png')}"
        plt.savefig(filename, format=plot_config.get("format", "png"))
        print(f"✓ Saved epoch distribution plot to {filename}")

    # Optionally show plot if interactive
    if plot_config.get("interactive", False):
        plt.show()

    plt.close()


def plot_weight_angle_and_magnitude_vs_epochs(
    run_results: List[Dict[str, Any]], layer_name: str, output_dir: Path, experiment_name: str
):
    all_angles = []
    all_ratios = []
    all_epochs = []

    for result in run_results:
        run_dir = result["run_dir"]
        run_id = result["run_id"]
        epochs_completed = result.get("epochs_completed", None)
        if epochs_completed is None:
            continue

        try:
            w_final = torch.load(run_dir / "model.pt", map_location="cpu")[f"{layer_name}.weight"]
            w_init = torch.load(run_dir / "model_init.pt", map_location="cpu")[f"{layer_name}.weight"]
        except Exception as e:
            print(f"⚠️ Run {run_id}: Failed to load weights for layer {layer_name}: {e}")
            continue

        if w_final.ndim == 1:
            w_final = w_final.unsqueeze(0)
            w_init = w_init.unsqueeze(0)

        angles = compute_angles_between(w_init, w_final)
        norm_ratios = compute_norm_ratios(w_init, w_final)  # w_init.norm().item() / w_final.norm().item()

        all_angles.append(angles)
        all_ratios.append(norm_ratios)
        all_epochs.append(epochs_completed)

    if not all_angles:
        print(f"⚠️ No valid data for layer {layer_name}")
        return

    max_angles = [max(run_angles) for run_angles in all_angles]
    mean_ratios = [np.mean(r) for r in all_ratios]

    # === Plot 1: Angle vs Epochs ===
    fig_angle, ax_angle = plt.subplots(figsize=(6, 4), dpi=300)
    ax_angle.scatter(max_angles, all_epochs, alpha=0.8)
    ax_angle.set_title(f"Angle Between W_init and W_final\n{experiment_name} {layer_name}")
    ax_angle.set_xlabel("Angle (degrees)")
    ax_angle.set_ylabel("Epochs Completed")
    ax_angle.grid(True)
    plt.tight_layout()
    angle_path = output_dir / f"{experiment_name}_{layer_name}_angle_vs_epochs.png"
    fig_angle.savefig(angle_path)
    plt.close(fig_angle)

    # === Plot 2: Norm Ratio vs Epochs ===
    fig_ratio, ax_ratio = plt.subplots(figsize=(6, 4), dpi=300)
    ax_ratio.scatter(mean_ratios, all_epochs, alpha=0.8)
    ax_ratio.set_title(f"Norm(W_init)/Norm(W_final) vs Epochs\n{experiment_name} {layer_name}")
    ax_ratio.set_xlabel("Norm Ratio")
    ax_ratio.set_ylabel("Epochs Completed")
    ax_ratio.grid(True)
    plt.tight_layout()
    ratio_path = output_dir / f"{experiment_name}_{layer_name}_normratio_vs_epochs.png"
    fig_ratio.savefig(ratio_path)
    plt.close(fig_ratio)

    print(f"✓ Saved angle plot to:     {angle_path}")
    print(f"✓ Saved norm ratio plot to: {ratio_path}")


def plot_failure_angle_histogram(
    success_angles: List[float], failure_angles: List[float], output_path: Path, title: str
):

    plt.figure(figsize=(6, 4), dpi=300)

    if success_angles:
        plt.hist(success_angles, bins=30, alpha=0.6, label="Success (100%)")
    if failure_angles:
        plt.hist(failure_angles, bins=15, alpha=0.8, label="Failure (50%)", color="red")

    plt.axvline(90, color="black", linestyle="--", label="90° (perpendicular)")
    plt.xlabel("Initial Angle Difference to Ideal (degrees)")
    plt.ylabel("Unit Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
