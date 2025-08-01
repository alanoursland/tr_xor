from typing import Dict, List, Tuple, Any
import torch
import numpy as np


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
    """Generate the convergence timing section broken down by accuracy tier."""
    report = "## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)\n\n"

    def format_table(title: str, timing_data: dict) -> str:
        section = f"### {title}\n\n"
        section += f"**Number of runs:** {timing_data.get('count', 0)}\n\n"
        section += "| Percentile | Epochs |\n| ---------- | ------ |\n"

        percentiles = timing_data.get("percentiles", {})
        labels = ["0th", "10th", "25th", "50th", "75th", "90th", "100th"]

        if percentiles:
            for label in labels:
                value = percentiles.get(label, "N/A")
                section += f"| {label:<10} | {value}     |\n"
        else:
            section += "| N/A        | No data available |\n"

        section += "\n"
        return section

    if convergence_timing:
        report += format_table("✅ All Runs", convergence_timing.get("all_runs", {}))
        report += format_table("🌟 Perfect Accuracy (100%)", convergence_timing.get("perfect_accuracy", {}))
        report += format_table("⚠️ Sub-Perfect Accuracy (<100%)", convergence_timing.get("subperfect_accuracy", {}))
    else:
        report += "No convergence data available.\n"

    report += "---\n\n"
    return report


def generate_hyperplane_distance_section(analysis_results):
    """
    Generate a clean, readable section for hyperplane distance clustering data.
    Updated to work with the new hook-based analysis structure.
    """
    distance_data = analysis_results.get("distance_to_hyperplanes", {})
    
    if not distance_data:
        return ""
    
    report = "## 📏 Hyperplane Distance Clusters\n\n"
    report += "Analysis based on L2 distances from actual layer inputs to hyperplanes, grouped by model predictions.\n\n"
    
    for layer_name, layer_data in distance_data.items():
        # Updated to use 'distance_analysis' instead of 'metric_space_analysis'
        distance_analysis = layer_data.get('distance_analysis', {})
        clusters = distance_analysis.get('clusters', [])
        total_hyperplanes = distance_analysis.get('total_hyperplanes', 0)
        noise_count = distance_analysis.get('noise_count', 0)
        
        # Layer header
        report += f"### Layer: `{layer_name}`\n\n"
        
        # Summary stats
        summary_parts = [f"{total_hyperplanes} hyperplanes"]
        if len(clusters) > 0:
            summary_parts.append(f"{len(clusters)} behavior patterns")
        if noise_count > 0:
            summary_parts.append(f"{noise_count} outliers")
        
        report += f"**Summary**: {', '.join(summary_parts)}\n\n"
        
        if not clusters:
            report += "No distinct behavior patterns found.\n\n"
            continue
        
        # Table format for better readability
        report += "| Pattern | Size | Class 0 Distance | Class 1 Distance | Separation | Runs |\n"
        report += "|---------|------|------------------|------------------|------------|------|\n"
        
        for i, cluster in enumerate(clusters, 1):
            cluster_id = cluster['cluster_id']
            size = cluster['size']
            centroid = cluster['centroid']
            std = cluster['std']
            hyperplanes = cluster['hyperplanes']
            
            # Format distances nicely
            mean_dist_class0, mean_dist_class1 = centroid
            std_class0, std_class1 = std
            
            dist0_str = f"{mean_dist_class0:.2f} ± {std_class0:.2f}"
            dist1_str = f"{mean_dist_class1:.2f} ± {std_class1:.2f}"
            
            # Separation interpretation
            separation_ratio = mean_dist_class1 / (mean_dist_class0 + 1e-12)
            if separation_ratio > 2.0:
                sep_str = f"{separation_ratio:.1f}× (Class 1)"
            elif separation_ratio < 0.5:
                sep_str = f"{1/separation_ratio:.1f}× (Class 0)"
            else:
                sep_str = f"{separation_ratio:.2f} (Mixed)"
            
            # All runs (no abbreviation as requested)
            run_ids = sorted(list(set(hp['run_id'] for hp in hyperplanes)))
            runs_str = ", ".join(map(str, run_ids))
            
            report += f"| {i} | {size} | {dist0_str} | {dist1_str} | {sep_str} | {runs_str} |\n"
        
        report += "\n"
    
    report += "---\n\n"
    return report


def generate_weight_reorientation_section(weight_reorientation):
    """Generate the weight reorientation analysis section."""
    per_layer_analysis = weight_reorientation.get("per_layer_analysis", {})

    if not per_layer_analysis:
        return ""

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
                report += (
                    f"| {percentile_range:<10} | {low:.2f} – {high:.2f}  | {mean_epochs:.1f}                       |\n"
                )
        else:
            report += "| N/A        | No data available | N/A                        |\n"

        report += "\n---\n\n"

    return report


def generate_combined_norm_ratio_section(weight_reorientation):
    """Generate the combined norm ratio analysis section."""

    # Extract and process norm ratio data from all layers
    per_layer_analysis = weight_reorientation.get("per_layer_analysis", {})
    if not per_layer_analysis:
        return ""

    all_ratios = []
    all_epochs = []

    for layer_stats in per_layer_analysis.values():
        layer_ratios = layer_stats.get("norm_ratio_analysis", {})
        for bin_data in layer_ratios.values():
            low = bin_data["range"][0]
            high = bin_data["range"][1]
            count = bin_data["count"]
            mean_epochs = bin_data["mean_epochs"]
            # Store per-bin representative values (flatten bins across layers)
            all_ratios.append((low, high, mean_epochs, count))
            all_epochs.extend([mean_epochs] * count)

    # Sort ratios for consistent ordering
    sorted_ratios = sorted(all_ratios, key=lambda x: x[0])

    # Format statistics into markdown report section
    report = "### ◼ Initial / Final Norm Ratio (All Layers Combined)\n\n"
    report += "| Percentile | Ratio Range | Mean Epochs to Convergence |\n"
    report += "| ---------- | ------------ | -------------------------- |\n"

    if sorted_ratios:
        for i, (low, high, mean_epochs, count) in enumerate(sorted_ratios):
            label = f"{i+1:>2}"
            report += f"| {label:<10} | {low:.2f} – {high:.2f}  | {mean_epochs:.1f}                       |\n"
    else:
        report += "| N/A        | No data available | N/A                        |\n"

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
            p25 = np.percentile(final_loss_values, 25)
            p50 = np.percentile(final_loss_values, 50)
            p75 = np.percentile(final_loss_values, 75)
        else:
            variance = 0.0
            p25 = p50 = p75 = 0.0

        report += f"* **Mean final loss**: {mean_loss:.2e}\n\n"
        report += f"* **Variance**: {variance:.2e}\n\n"
        report += f"* **Range**:\n\n"
        report += f"  * 0th percentile: {min_loss:.2e}\n"
        report += f"  * 25th percentile: {p25:.2e}\n"
        report += f"  * 50th percentile (median): {p50:.2e}\n"
        report += f"  * 75th percentile: {p75:.2e}\n"
        report += f"  * 100th percentile: {max_loss:.2e}\n\n"
    else:
        report += "* **No final loss data available**\n\n"

    report += "\n---\n\n"
    return report


def generate_hyperplane_clustering_section(analysis_results):
    """Generate the hyperplane clustering section in Markdown with improved formatting."""
    report = "## 🎯 Hyperplane Clustering\n\n"

    hyperplane_clustering = analysis_results.get("hyperplane_clustering", {})

    if not hyperplane_clustering:
        report += "* **No clustering data available**\n\n"
        return report

    for layer_name, layer_result in hyperplane_clustering.items():
        report += f"### 🔹 Layer: `{layer_name}`\n\n"

        clustering_params = layer_result.get("clustering_params", {})
        eps = clustering_params.get("eps", "N/A")
        min_samples = clustering_params.get("min_samples", "N/A")

        report += f"**DBSCAN Parameters**: eps=`{eps}`, min_samples=`{min_samples}`\n\n"

        # Process each parameter type (weight, bias, etc.)
        for param_key, param_data in layer_result.items():
            if param_key in ("layer_name", "clustering_params"):
                continue  # skip metadata

            param_label = param_data.get("param_label", param_key)
            clusters = param_data.get("param_data", [])
            n_clusters = param_data.get("n_clusters", 0)
            noise_count = param_data.get("noise_count", 0)

            report += f"#### Parameter: `{param_label}`\n\n"
            
            # Summary statistics
            report += f"* **Total clusters found**: {n_clusters}\n"
            if noise_count > 0:
                report += f"* **Noise points**: {noise_count}\n"
            
            if not clusters:
                report += "* No clusters to display\n\n"
                continue
                
            report += "\n"

            # Sort clusters by size (largest first)
            sorted_clusters = sorted(clusters, key=lambda x: x["size"], reverse=True)

            # Create a table for better readability
            report += "| Cluster | Size | Centroid | Std Dev | Runs |\n"
            report += "|---------|------|----------|---------|------|\n"

            for cluster in sorted_clusters:
                cid = cluster["cluster_label"]
                size = cluster["size"]
                centroid = cluster["centroid"]
                std = cluster["std"]
                run_ids = cluster["run_ids"]

                # Format centroid and std dev nicely
                centroid_str = "[" + ", ".join(f"{v:.3f}" for v in centroid) + "]"
                std_str = "[" + ", ".join(f"{s:.3f}" for s in std) + "]"
                
                runs_str = ", ".join(str(r) for r in set(run_ids))

                report += f"| {cid} | {size} | `{centroid_str}` | `{std_str}` | {runs_str} |\n"

            report += "\n"

        report += "---\n\n"

    return report
    """Generate the hyperplane clustering section in Markdown."""
    report = "## 🎯 Hyperplane Clustering\n\n"

    hyperplane_clustering = analysis_results.get("hyperplane_clustering", {})

    if not hyperplane_clustering:
        report += "* **No clustering data available**\n\n"
    else:
        for layer_name, layer_result in hyperplane_clustering.items():
            report += f"### 🔹 Layer `{layer_name}`\n"

            clustering_params = layer_result.get("clustering_params", {})
            eps = clustering_params.get("eps")
            min_samples = clustering_params.get("min_samples")

            report += f"* **DBSCAN eps**: `{eps}`\n"
            report += f"* **DBSCAN min_samples**: `{min_samples}`\n\n"

            for param_key, param_data in layer_result.items():
                if param_key in ("layer_name", "clustering_params"):
                    continue  # skip meta

                param_label = param_data.get("param_label")
                clusters = param_data.get("param_data", [])
                n_clusters = param_data.get("n_clusters", 0)
                noise_points = param_data.get("noise_count", 0)

                report += f"**Parameter `{param_label}`**\n\n"
                report += f"* Clusters: **{n_clusters}**\n"
                if noise_points > 0:
                    report += f"* Noise points: **{noise_points}**\n"
                report += "\n"

                for cluster in clusters:
                    cid = cluster["cluster_label"]
                    size = cluster["size"]
                    centroid = cluster["centroid"]
                    std = cluster["std"]
                    run_ids = cluster["run_ids"]

                    report += f"#### ◼ Cluster `{cid}`\n"
                    report += f"* **Size**: {size}\n"
                    report += f"* **Centroid**: [{', '.join(f'{v:.6f}' for v in centroid)}]\n"
                    report += f"* **Std Dev**: [{', '.join(f'{s:.6f}' for s in std)}]\n"
                    report += f"* **Runs**: {', '.join(str(r) for r in run_ids)}\n"
                    report += "\n"

            report += "\n"

        report += "---\n\n"

    return report

def generate_dead_data_analysis_section(analysis_results, config):
    """Generate the dead data point analysis section."""
    if not config.analysis.dead_data_analysis:
        return ""

    # Extract dead data information
    dead_data = analysis_results["dead_data"]
    dead_counts = dead_data["dead_counts"]
    dead_class0_counts = dead_data["dead_class0_counts"]
    dead_class1_counts = dead_data["dead_class1_counts"]
    accuracies = dead_data["accuracies"]

    # Calculate dead data statistics by accuracy
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

    # Format statistics into markdown report section
    report = "## 💀 Dead Data Point Analysis\n\n"

    for acc in sorted(acc_summary.keys(), reverse=True):
        summary = acc_summary[acc]
        acc_percent = int(acc * 100)
        alive_count = summary["alive"]
        dead_count = summary["dead"]
        class0_dead_count = summary["class0_dead"]
        class1_dead_count = summary["class1_dead"]

        if alive_count > 0:
            report += f"* {alive_count} runs with **no dead inputs** reached {acc_percent}% accuracy\n"

        if dead_count > 0:
            report += f"* {dead_count} runs with **dead inputs** reached {acc_percent}% accuracy\n"
            report += f"|    {class0_dead_count} runs with class-0 dead inputs reached {acc_percent}% accuracy\n"
            report += f"|    {class1_dead_count} runs with class-1 dead inputs reached {acc_percent}% accuracy\n"

    report += "\n---\n\n"
    return report


def generate_mirror_analysis_section(analysis_results, config) -> str:
    """Generate the mirror weight symmetry analysis section."""
    if not config.analysis.mirror_weight_detection:
        return ""

    # Extract and calculate mirror statistics
    mirror_data = analysis_results.get("mirror_weights", [])
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

    # Calculate summary statistics
    if mirror_sims:
        sims_tensor = torch.tensor(mirror_sims)
        mean_sim = sims_tensor.mean().item()
        std_sim = sims_tensor.std().item()
        mean_error = abs(mean_sim + 1.0)
        has_mirror_data = True
    else:
        mean_sim = 0.0
        std_sim = 0.0
        mean_error = 0.0
        has_mirror_data = False

    # Format statistics into markdown report section
    report = "## 🔍 Mirror Weight Symmetry\n\n"

    if has_mirror_data:
        report += f"* **Mirror pairs detected**: {detected_runs} / {total_runs} runs\n"
        report += f"* **Perfect mirror symmetry** (cosine ~ -1.0): {perfect_mirrors} runs\n"
        report += f"* **Mean mirror similarity**: {mean_sim:.5f} ± {std_sim:.5f}\n"
        report += f"* **Mean mirror error (|cos + 1|)**: {mean_error:.5f}\n"
    else:
        report += "* No mirror pairs detected in any run.\n"

    report += "\n---\n\n"
    return report


def generate_failure_analysis_section(analysis_results, config):
    """Generate the geometric analysis of failure modes section."""
    if not config.analysis.failure_angle_analysis:
        return ""

    failure_angles = analysis_results.get("failure_angle_analysis", {})

    report = "## 🧭 Geometric Analysis of Failure Modes\n\n"
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


    # Extract top-level blocks from analysis_results
    # Path: basic_stats
    basic_stats = analysis_results.get("basic_stats", {})
    # Path: basic_stats.experiment_info
    experiment_info = basic_stats.get("experiment_info", {})
    # Path: basic_stats.distributions
    distributions = basic_stats.get("distributions", {})

    # Metrics extraction
    # Path: basic_stats.experiment_info.total_runs
    total_runs = experiment_info.get("total_runs", "N/A")

    ############################################################################################

    # Extract convergence timing data
    convergence_timing = analysis_results.get("convergence_timing", {})
    # print(f"convergence_timing = {convergence_timing}")

    # Extract weight reorientation data
    weight_reorientation = analysis_results.get("weight_reorientation", {})

    ############################################################################################
    report = ""

    # Start Markdown report
    report += generate_report_header(config)
    report += generate_overview_section(config)
    report += generate_accuracy_section(distributions, total_runs)
    report += generate_convergence_section(convergence_timing, config)
    report += generate_hyperplane_distance_section(analysis_results)
    report += generate_weight_reorientation_section(weight_reorientation)
    report += generate_combined_norm_ratio_section(weight_reorientation)
    report += generate_loss_distribution_section(basic_stats)
    report += generate_hyperplane_clustering_section(analysis_results)
    report += generate_dead_data_analysis_section(analysis_results, config)
    report += generate_mirror_analysis_section(analysis_results, config)
    report += generate_failure_analysis_section(analysis_results, config)

    return report
