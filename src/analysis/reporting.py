from typing import Dict, List, Tuple, Any
import torch
import numpy as np


def process_prototype_surface_distances(analysis_results):
    """
    Extract and process prototype surface distance test results.
    
    Returns:
        Dict[str, Dict[int, Dict[int, List[float]]]]: 
        Nested structure: layer → unit → class → list of distances
    """
    # Extract prototype surface distance test results
    distance_entries = analysis_results.get("prototype_surface", {}).get("distance_test", [])

    # Initialize nested structure: layer → unit → class → list of distances
    distance_by_layer_unit = {}

    for entry in distance_entries:
        layer_distances = entry.get("layer_distances", {})

        for layer_name, layer_data in layer_distances.items():
            unit_distances = layer_data.get("unit_distances")  # list of tensors/lists, one per unit
            labels = layer_data.get("labels")

            if unit_distances is None or labels is None:
                continue

            # Convert labels to numpy
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            elif not isinstance(labels, np.ndarray):
                labels = np.array(labels)

            for unit_idx, unit_dists in enumerate(unit_distances):
                # Convert distances to numpy
                if isinstance(unit_dists, torch.Tensor):
                    unit_dists = unit_dists.detach().cpu().numpy()
                elif not isinstance(unit_dists, np.ndarray):
                    unit_dists = np.array(unit_dists)

                for i in range(len(labels)):
                    label = labels[i]
                    dist = unit_dists[i]

                    # Handle one-hot or tensor labels
                    if isinstance(label, np.ndarray) and label.ndim == 1 and label.shape[0] > 1:
                        label = np.argmax(label)
                    elif isinstance(label, (torch.Tensor, np.generic)):
                        label = int(label)

                    if label not in (0, 1):
                        continue

                    # Initialize structure
                    if layer_name not in distance_by_layer_unit:
                        distance_by_layer_unit[layer_name] = {}
                    if unit_idx not in distance_by_layer_unit[layer_name]:
                        distance_by_layer_unit[layer_name][unit_idx] = {0: [], 1: []}

                    distance_by_layer_unit[layer_name][unit_idx][label].append(dist)

    return distance_by_layer_unit

def generate_report_header(config) -> str:
    """Generate the report header with experiment name and description."""
    name = config.execution.experiment_name
    description = config.description or "No description provided."

    report = f"# 🧪 Experiment Report: `{name}`\n\n"
    report += f"**Description**: {description}\n\n"

    return report


def generate_overview_section(config) -> str:
    """Generate the overview section with training configuration details."""
    report = "## 🎯 Overview\n\n"

    # Training Configuration
    report += f"* **Total runs**: {config.execution.num_runs}\n"
    report += f"* **Loss function**: {config.training.loss_function.__class__.__name__}\n"
    report += f"* **Optimizer**: {config.training.optimizer.__class__.__name__}\n"
    if config.training.batch_size:
        report += f"* **Batch size**: {config.training.batch_size}\n"
    report += f"* **Max epochs**: {config.training.epochs}\n"

    # Early Stopping Criteria
    if config.training.stop_training_loss_threshold is not None:
        report += f"* **Stops when loss < {config.training.stop_training_loss_threshold:.1e}**\n"

    if config.training.loss_change_threshold is not None and config.training.loss_change_patience is not None:
        report += (
            f"* **Stops if loss does not improve by ≥ {config.training.loss_change_threshold:.1e} "
            f"over {config.training.loss_change_patience} epochs**\n"
        )

    report += "\n---\n\n"
    return report


def generate_accuracy_section(distributions, total_runs) -> str:
    """Generate the classification accuracy section."""
    acc_bins = distributions.get("accuracy_distribution", {}).get("bins", {})

    report = "## 🎯 Classification Accuracy\n\n"

    for acc in sorted(acc_bins.keys(), reverse=True):
        count = acc_bins.get(acc, 0)
        if count > 0:
            report += f"* {count}/{total_runs} runs achieved {int(100*acc)}% accuracy\n"

    report += "\n---\n\n"
    return report


def generate_convergence_section(convergence_timing, config) -> str:
    """Generate the convergence timing section."""
    if not config.analysis.convergence_analysis:
        return ""

    report = "## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)\n\n"
    report += "| Percentile | Epochs |\n| ---------- | ------ |\n"

    percentiles = convergence_timing.get("percentiles", {})

    if percentiles:
        labels = ["0th", "10th", "25th", "50th", "75th", "90th", "100th"]
        for label in labels:
            value = percentiles.get(label, "N/A")
            report += f"| {label:<10} | {value}     |\n"
    else:
        report += "| N/A        | No convergence data available |\n"

    report += "\n---\n\n"
    return report


def generate_loss_distribution_section(basic_stats) -> str:
    """Generate the final loss distribution section."""
    report = "## 📉 Final Loss Distribution\n\n"

    final_losses = basic_stats.get("summary", {}).get("final_losses", {})

    if final_losses:
        # Extract additional statistics we need
        mean_loss = final_losses.get("mean", 0.0)
        min_loss = final_losses.get("min", 0.0)
        max_loss = final_losses.get("max", 0.0)

        # Calculate variance from the raw metrics if available
        raw_metrics = basic_stats.get("raw_metrics", {})
        final_loss_values = raw_metrics.get("final_losses", [])

        if final_loss_values:
            variance = np.var(final_loss_values)
        else:
            variance = 0.0

        report += f"* **Mean final loss**: {mean_loss:.2e}\n\n"
        report += f"* **Variance**: {variance:.2e}\n\n"
        report += f"* **Range**:\n\n"
        report += f"  * 0th percentile: {min_loss:.2e}\n"
        report += f"  * 100th percentile: {max_loss:.2e}\n\n"
    else:
        report += "* **No final loss data available**\n\n"

    report += "\n---\n\n"
    return report


def generate_geometry_section(distance_by_layer_unit):
    """Generate the prototype surface geometry section."""
    report = "## 📏 Prototype Surface Geometry\n\n"

    for layer_name, units in distance_by_layer_unit.items():
        report += f"### Layer: `{layer_name}`\n\n"

        for unit_idx, class_dists in units.items():
            d0 = np.array(class_dists[0])
            d1 = np.array(class_dists[1])

            if len(d0) == 0 or len(d1) == 0:
                continue  # Skip units with missing data

            d0_mean, d0_std = d0.mean(), d0.std()
            d1_mean, d1_std = d1.mean(), d1.std()
            ratio = d1_mean / (d0_mean + 1e-12)

            report += f"- **Unit {unit_idx}**\n"
            report += f"  - Mean distance to class 0: `{d0_mean:.2e} ± {d0_std:.2e}`\n"
            report += f"  - Mean distance to class 1: `{d1_mean:.5f} ± {d1_std:.2e}`\n"
            report += f"  - Separation ratio (class1/class0): `{ratio:.2f}`\n\n"

    report += "\n---\n\n"
    return report

def generate_weight_reorientation_section(weight_reorientation):
    """Generate the weight reorientation analysis section."""
    per_layer_analysis = weight_reorientation.get("per_layer_analysis", {})

    if not per_layer_analysis:
        return "No weight reorientation data available.\n\n"

    report = ""
    
    for layer_name, layer_stats in per_layer_analysis.items():
        angle_data = layer_stats.get("angle_analysis", {})
        norm_data = layer_stats.get("norm_ratio_analysis", {})

        report += f"### Layer: `{layer_name}` – Angle Between Initial and Final Weights\n\n"
        report += "| Percentile | Angle Range (°) | Mean Epochs to Convergence |\n"
        report += "| ---------- | ---------------- | -------------------------- |\n"

        if angle_data:
            for percentile_range, stats in angle_data.items():
                low, high = stats["range"]
                mean_epochs = stats["mean_epochs"]
                report += f"| {percentile_range:<10} | {low:.1f} – {high:.1f}       | {mean_epochs:.1f}                       |\n"
        else:
            report += "| N/A        | No data available | N/A                        |\n"

        report += f"\n### Layer: `{layer_name}` – Initial / Final Norm Ratio\n\n"
        report += "| Percentile | Ratio Range | Mean Epochs to Convergence |\n"
        report += "| ---------- | ------------ | -------------------------- |\n"

        if norm_data:
            for percentile_range, stats in norm_data.items():
                low, high = stats["range"]
                mean_epochs = stats["mean_epochs"]
                report += f"| {percentile_range:<10} | {low:.2f} – {high:.2f}  | {mean_epochs:.1f}                       |\n"
        else:
            report += "| N/A        | No data available | N/A                        |\n"

        report += "\n---\n\n"
    
    return report

def generate_combined_norm_ratio_section(weight_reorientation):
    """Generate the combined norm ratio analysis section."""
    report = "### ◼ Initial / Final Norm Ratio (All Layers Combined)\n\n"
    report += "| Percentile | Ratio Range | Mean Epochs to Convergence |\n"
    report += "| ---------- | ------------ | -------------------------- |\n"

    # Pull from per_layer_analysis
    per_layer_analysis = weight_reorientation.get("per_layer_analysis", {})

    # Collect all norm ratios and epochs from all layers
    all_ratios = []
    all_epochs = []

    for layer_stats in per_layer_analysis.values():
        layer_ratios = layer_stats.get("norm_ratio_analysis", {})
        for bin_data in layer_ratios.values():
            low, high = bin_data["range"]
            count = bin_data["count"]
            mean_epochs = bin_data["mean_epochs"]
            # Store per-bin representative values (flatten bins across layers)
            all_ratios.append((low, high, mean_epochs, count))
            all_epochs.extend([mean_epochs] * count)

    if all_ratios:
        # Re-bin across all ratios (flattened)
        # You could re-bucket if needed, but here's a simple way to output the flattened bins:
        for i, (low, high, mean_epochs, count) in enumerate(sorted(all_ratios, key=lambda x: x[0])):
            label = f"{i+1:>2}"
            report += f"| {label:<10} | {low:.2f} – {high:.2f}  | {mean_epochs:.1f}                       |\n"
    else:
        report += "| N/A        | No data available | N/A                        |\n"

    report += "\n---\n\n"
    return report

def generate_mirror_analysis_section(analysis_results, config) -> str:
    """Generate the mirror weight symmetry analysis section."""
    if not config.analysis.mirror_pair_detection:
        return ""

    report = "## 🔍 Mirror Weight Symmetry\n\n"

    mirror_data = analysis_results.get("prototype_surface", {}).get("mirror_test", [])
    total_runs = len(mirror_data)
    perfect_threshold = 1e-3  # similarity diff from -1.0

    mirror_sims = []
    detected_runs = 0
    perfect_mirrors = 0

    for entry in mirror_data:
        pairs = entry.get("mirror_pairs", [])
        if pairs:
            detected_runs += 1
            sim = pairs[0][2]  # cosine similarity (e.g. -0.99998)
            mirror_sims.append(sim)
            if abs(sim + 1.0) < perfect_threshold:
                perfect_mirrors += 1

    if mirror_sims:
        sims_tensor = torch.tensor(mirror_sims)
        mean_sim = sims_tensor.mean().item()
        std_sim = sims_tensor.std().item()
        mean_error = abs(mean_sim + 1.0)

        report += f"* **Mirror pairs detected**: {detected_runs} / {total_runs} runs\n"
        report += f"* **Perfect mirror symmetry** (cosine ~ -1.0): {perfect_mirrors} runs\n"
        report += f"* **Mean mirror similarity**: {mean_sim:.5f} ± {std_sim:.5f}\n"
        report += f"* **Mean mirror error (|cos + 1|)**: {mean_error:.5f}\n"
    else:
        report += "* No mirror pairs detected in any run.\n"

    report += "\n---\n\n"
    return report

def generate_analysis_report(
    analysis_results: Dict[str, Any],
    config: Any,  # Replace Any with actual ExperimentConfig type
    template: str = "comprehensive",
) -> str:
    """
    Generate a Markdown-formatted analysis report of the experiment.

    Args:
        analysis_results: Dict with experiment summary, accuracy distribution, convergence stats.
        config: The ExperimentConfig for metadata.
        template: Currently only 'comprehensive' is supported.

    Returns:
        Markdown string report.
    """
    ############################################################################################

    loss_change_threshold = 0.01  # replaced with config.training.loss_change_threshold

    # Extract top-level blocks from analysis_results
    # Path: basic_stats
    basic_stats = analysis_results.get("basic_stats", {})
    # Path: basic_stats.summary
    summary = basic_stats.get("summary", {})
    # Path: basic_stats.experiment_info
    experiment_info = basic_stats.get("experiment_info", {})
    # Path: basic_stats.distributions
    distributions = basic_stats.get("distributions", {})
    # Path: basic_stats.distributions.accuracy_distribution.bins
    acc_bins = distributions.get("accuracy_distribution", {}).get("bins", {})
    # Path: basic_stats.success_metrics
    success_metrics = basic_stats.get("success_metrics", {})
    # Path: basic_stats.summary.final_losses
    final_losses = summary.get("final_losses", {})

    # Extract prototype surface data from analysis_results
    # Path: prototype_surface
    prototype_surface = analysis_results.get("prototype_surface", {})
    # Path: prototype_surface.mirror_test
    mirror_test = prototype_surface.get("mirror_test", [])
    # Path: prototype_surface.distance_test
    distance_test = prototype_surface.get("distance_test", [])

    # Metrics extraction
    # Path: basic_stats.experiment_info.total_runs
    total_runs = experiment_info.get("total_runs", "N/A")
    # Path: basic_stats.success_metrics.perfect_accuracy_rate
    avg_acc = success_metrics.get("perfect_accuracy_rate", 0.0)
    # Path: basic_stats.distributions.accuracy_distribution.bins."1.0"
    success_runs = acc_bins.get(1.0, 0)
    # Path: basic_stats.success_metrics.convergence_rate
    conv_rate = success_metrics.get("convergence_rate", 0.0) * 100
    # Path: basic_stats.summary.final_losses.min
    best_loss = final_losses.get("min", 0.0)
    # Path: basic_stats.summary.final_losses.max
    worst_loss = final_losses.get("max", 0.0)

    # Mirror pattern check
    # Accesses 'mirror_count' within each item of prototype_surface.mirror_test[]
    mirror_detected = any(run.get("mirror_count", 0) > 0 for run in mirror_test)
    mirror_flag = "✅ Detected" if mirror_detected else "❌ None detected"

    # prototype surface interpretation flags
    proto_surface_ok = "✔️" if distance_test else "⚠️ Not available"
    geometry_ok = "✔️" if distance_test else "⚠️ Not available"
    prototype_support = "✅" if (avg_acc == 1.0 and distance_test) else "⚠️ Partial"

    ############################################################################################
    # Collect distances from hyperplanes
    ############################################################################################

    distance_by_layer_unit = process_prototype_surface_distances(analysis_results)

    ############################################################################################

    # Accuracy validation
    failed_runs = total_runs - success_runs if isinstance(total_runs, int) and isinstance(success_runs, int) else "?"
    all_success = success_runs == total_runs if isinstance(total_runs, int) and isinstance(success_runs, int) else False
    stop_threshold = getattr(config.training, "stop_training_loss_threshold", loss_change_threshold)

    # Extract convergence timing data
    convergence_timing = analysis_results.get("convergence_timing", {})
    # print(f"convergence_timing = {convergence_timing}")
    percentiles = convergence_timing.get("percentiles", {})

    # Extract weight reorientation data
    weight_reorientation = analysis_results.get("weight_reorientation", {})
    failure_angles = analysis_results.get("failure_angle_analysis", {})

    ############################################################################################
    report = ""

    name = config.execution.experiment_name
    description = config.description or experiment_info.get("description", "No description provided.")

    # Start Markdown report
    report += generate_report_header(config)
    report += generate_overview_section(config)
    report += generate_accuracy_section(distributions, total_runs)
    report += generate_convergence_section(convergence_timing, config)
    report += generate_geometry_section(distance_by_layer_unit)
    report += generate_weight_reorientation_section(weight_reorientation)
    report += generate_combined_norm_ratio_section(weight_reorientation)

    ############################################################################################

    report += generate_loss_distribution_section(basic_stats)

    ############################################################################################

    report += "## 🎯 Hyperplane Clustering\n\n"

    hyperplane_clustering = analysis_results.get("hyperplane_clustering", {})

    if not hyperplane_clustering:
        report += "* **No clustering data available**\n\n"
    else:
        for layer_name, layer_result in hyperplane_clustering.items():
            cluster_info = layer_result.get("cluster_info", {})
            n_clusters = layer_result.get("n_clusters", 0)
            noise_points = layer_result.get("noise_points", 0)

            report += f"### 🔹 Layer `{layer_name}`\n"
            report += f"* **Number of clusters discovered**: {n_clusters}\n"
            if noise_points > 0:
                report += f"* **Noise points**: {noise_points}\n"
            report += "\n"

            for cluster_name, info in cluster_info.items():
                # Extract just the numeric cluster ID
                cluster_id = cluster_name.rsplit("_", 1)[-1]
                size = info["size"]
                weight_centroid = info["weight_centroid"]
                bias_centroid = info["bias_centroid"]

                report += f"#### ◼ Cluster {cluster_id}\n"
                report += f"* **Size**: {size} runs\n"
                report += f"* **Weight centroid**: [{', '.join(f'{w:.6f}' for w in weight_centroid)}]\n"
                report += f"* **Bias centroid**: [{', '.join(f'{b:.6f}' for b in bias_centroid)}]\n"

                # Try to generate a readable hyperplane equation if 2D
                if len(weight_centroid) == 2 and len(bias_centroid) == 1:
                    w0, w1 = weight_centroid
                    b0 = bias_centroid[0]
                    report += f"* **Hyperplane equation**: {w0:.6f}x₁ + {w1:.6f}x₂ + {b0:.6f} = 0\n"

                report += "\n"

            report += "\n"

        report += "---\n\n"

    ############################################################################################

    if config.analysis.dead_data_analysis:
        report += "## 💀 Dead Data Point Analysis\n\n"

        dead_data = analysis_results["dead_data"]
        dead_counts = dead_data["dead_counts"]
        dead_class0_counts = dead_data["dead_class0_counts"]
        dead_class1_counts = dead_data["dead_class1_counts"]
        accuracies = dead_data["accuracies"]

        # Build mapping: accuracy → [dead_count]
        acc_summary = {}
        for dead, dead0, dead1, acc in zip(dead_counts, dead_class0_counts, dead_class1_counts, accuracies):
            if acc not in acc_summary:
                acc_summary[acc] = {
                    "alive": 0,
                    "dead": 0,
                    "class0_dead": 0,
                    "class1_dead": 0,
                }
            if dead == 0:
                acc_summary[acc]["alive"] += 1
            else:
                acc_summary[acc]["dead"] += 1
                if dead0 > 0:
                    acc_summary[acc]["class0_dead"] += 1
                if dead1 > 0:
                    acc_summary[acc]["class1_dead"] += 1

        for acc in sorted(acc_summary.keys(), reverse=True):
            summary = acc_summary[acc]
            acc_percent = int(acc * 100)
            if summary["alive"] > 0:
                report += f"* {summary['alive']} runs with **no dead inputs** reached {acc_percent}% accuracy\n"
            if summary["dead"] > 0:
                report += f"* {summary['dead']} runs with **dead inputs** reached {acc_percent}% accuracy\n"
                report += (
                    f"|    {summary['class0_dead']} runs with class-0 dead inputs reached {acc_percent}% accuracy\n"
                )
                report += (
                    f"|    {summary['class1_dead']} runs with class-1 dead inputs reached {acc_percent}% accuracy\n"
                )

        report += "\n---\n\n"

    ############################################################################################

    report += generate_mirror_analysis_section(analysis_results, config)

    ############################################################################################

    if config.analysis.failure_angles:
        report += "## 🧭 Geometric Analysis of Failure Modes\n\n"
        report += "We tested whether failed runs began with hyperplanes nearly perpendicular to ideal orientations.\n"
        report += "Results are shown per layer, aggregating across all units in each layer.\n\n"

        for layer_name, layer_data in failure_angles.items():
            stats_s = layer_data["summary"]["success_stats"]["angle_diff"]
            stats_f = layer_data["summary"]["failure_stats"]["angle_diff"]

            count_s = len(layer_data["success"])
            count_f = len(layer_data["failure"])

            report += f"### Layer: `{layer_name}`\n\n"
            report += (
                f"* **Success units (n={count_s})** – mean angle diff: {stats_s['mean']:.2f}° ± {stats_s['std']:.2f}°\n"
            )
            report += (
                f"* **Failure units (n={count_f})** – mean angle diff: {stats_f['mean']:.2f}° ± {stats_f['std']:.2f}°\n"
            )
            report += "* Failed units tend to cluster near 90°, consistent with the no-torque trap hypothesis.\n\n"

        report += "See `failure_angle_histogram.png` for visual confirmation.\n\n"

    ############################################################################################

    return report
