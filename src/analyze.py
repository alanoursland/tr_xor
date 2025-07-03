# analyze.py - Post-Experiment Analysis and Visualization for Prototype Surface Experiments

"""
Comprehensive analysis module for prototype surface experiments.
Provides geometric analysis, visualization, and prototype surface validation tools.
Focuses on prototype surface investigation, distance field analysis, and
comparative studies across activation functions and training runs.
"""

import json
import math
import numpy as np
import sys
import torch
import traceback

from collections import defaultdict
from configs import ExperimentConfig, get_experiment_config, list_experiments
from pathlib import Path
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Any
from analysis.loader import load_experiment_data, load_all_run_results
from analysis.utils import extract_activation_type, get_linear_layers
from analysis.stats import compute_basic_statistics, compute_summary_statistics
from analysis.visualization import (
    plot_hyperplanes,
    plot_failure_angle_histogram,
    plot_epoch_distribution,
    plot_weight_angle_and_magnitude_vs_epochs,
)
from analysis.geometry import compute_angles_between, compute_norm_ratios, analyze_hyperplane_metric_clustering
from analysis.reporting import generate_analysis_report


def analyze_dead_data(run_results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Analyze which input points are dead at initialization (no active ReLU units),
    and correlate with final accuracy.
    """

    model = config.model
    data = config.data
    x = data.x  # [num_points, input_dim]
    y = data.y.squeeze().int()  # [num_points], class labels as 0 or 1

    results = {
        "dead_counts": [],
        "dead_class0_counts": [],
        "dead_class1_counts": [],
        "accuracies": [],
    }

    for i, result in enumerate(run_results):
        run_dir = result["run_dir"]
        model_init = torch.load(run_dir / "model_init.pt", map_location="cpu")
        model.load_state_dict(model_init)
        data = config.data

        model.eval()
        with torch.no_grad():
            activations = model.forward_components(x)  # Get per-unit pre-ReLU outputs

            # Assume output is a list or tensor: [num_points, num_units]
            if isinstance(activations, tuple):
                activations = activations[0]  # if model returns (output, components)

            relu_out = torch.relu(activations)  # [num_points, num_units]
            is_dead = torch.all(relu_out == 0, dim=1)  # [num_points]

            total_dead = int(is_dead.sum().item())
            dead_class0 = int((is_dead & (y == 0)).sum().item())
            dead_class1 = int((is_dead & (y == 1)).sum().item())

            results["dead_counts"].append(total_dead)
            results["dead_class0_counts"].append(dead_class0)
            results["dead_class1_counts"].append(dead_class1)
            results["accuracies"].append(result.get("accuracy", 0.0))

    return results


def summarize_all_runs(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics across all runs.

    Args:
        all_results: List of all run results

    Returns:
        Dictionary of summary statistics
    """
    if not all_results:
        return {}

    # Extract key metrics
    final_losses = [r.get("final_loss", float("inf")) for r in all_results]
    accuracies = [r.get("accuracy", 0.0) for r in all_results]
    training_times = [r.get("training_time", 0.0) for r in all_results]

    # Convert to tensors for easy computation
    losses_tensor = torch.tensor([l for l in final_losses if l != float("inf")])
    accuracies_tensor = torch.tensor(accuracies)
    times_tensor = torch.tensor(training_times)

    summary = {
        "total_runs": len(all_results),
        "successful_runs": len(losses_tensor),
        "failed_runs": len(all_results) - len(losses_tensor),
        # Loss statistics
        "loss_stats": {
            "mean": losses_tensor.mean().item() if len(losses_tensor) > 0 else float("inf"),
            "std": losses_tensor.std().item() if len(losses_tensor) > 0 else 0.0,
            "min": losses_tensor.min().item() if len(losses_tensor) > 0 else float("inf"),
            "max": losses_tensor.max().item() if len(losses_tensor) > 0 else float("inf"),
            "median": losses_tensor.median().item() if len(losses_tensor) > 0 else float("inf"),
        },
        # Accuracy statistics
        "accuracy_stats": {
            "mean": accuracies_tensor.mean().item(),
            "std": accuracies_tensor.std().item(),
            "min": accuracies_tensor.min().item(),
            "max": accuracies_tensor.max().item(),
            "median": accuracies_tensor.median().item(),
        },
        # Training time statistics
        "time_stats": {
            "mean": times_tensor.mean().item(),
            "std": times_tensor.std().item(),
            "min": times_tensor.min().item(),
            "max": times_tensor.max().item(),
            "total": times_tensor.sum().item(),
        },
        # Success metrics
        "convergence_rate": sum(1 for r in all_results if r.get("converged", False)) / len(all_results),
        "perfect_accuracy_rate": sum(1 for r in all_results if r.get("perfect_accuracy", False)) / len(all_results),
        # Best runs
        "best_loss_run": min(all_results, key=lambda r: r.get("final_loss", float("inf")))["run_id"],
        "best_accuracy_run": max(all_results, key=lambda r: r.get("accuracy", 0.0))["run_id"],
    }

    return summary


def compute_hyperplane_metric_vector(
    weight: torch.Tensor, 
    bias: torch.Tensor, 
    x_input: torch.Tensor, 
    y_labels: torch.Tensor,
    run_id: int,
    layer_name: str,
    unit_idx: int
) -> Tuple[np.ndarray, Dict]:
    """
    Compute the 2D metric vector for a single hyperplane.
    
    Returns:
        metric_vector: [mean_dist_class0, mean_dist_class1] or None if invalid
        hyperplane_info: metadata about this hyperplane
    """
    # Skip near-zero weights
    weight_norm = torch.norm(weight)
    if weight_norm < 1e-8:
        return None, None
    
    # Convert labels to class indices
    if y_labels.ndim > 1:
        y_indices = torch.argmax(y_labels, dim=1)
    else:
        y_indices = y_labels.long()
    
    # Compute distances to hyperplane: |w^T x + b| / ||w||
    distances = torch.abs(weight @ x_input.T + bias) / weight_norm
    
    # Compute mean distances per class
    class_means = {}
    for class_idx in torch.unique(y_indices):
        mask = y_indices == class_idx
        if mask.sum() > 0:  # Ensure class has samples
            class_means[int(class_idx)] = distances[mask].mean().item()
    
    # Ensure we have both classes
    if 0 not in class_means or 1 not in class_means:
        return None, None
    
    # Create metric vector: [mean_dist_class0, mean_dist_class1]
    metric_vector = np.array([class_means[0], class_means[1]])
    
    # Metadata about this hyperplane
    hyperplane_info = {
        "run_id": run_id,
        "layer_name": layer_name,
        "unit_idx": unit_idx,
        "weight_norm": weight_norm.item(),
        "weight": weight.detach().cpu().numpy(),
        "bias": bias.item(),
        "mean_dist_class0": class_means[0],
        "mean_dist_class1": class_means[1],
        "separation_ratio": class_means[1] / (class_means[0] + 1e-12)
    }
    
    return metric_vector, hyperplane_info


def format_hyperplane_clustering_results(
    layer_name: str,
    metric_vectors: np.ndarray,
    hyperplane_info: List[Dict],
    labels: np.ndarray,
    n_clusters: int,
    noise_count: int,
    eps: float,
    min_samples: int
) -> Dict[str, Any]:
    """Format clustering results for hyperplane metric vectors."""
    
    result = {
        "layer_name": layer_name,
        "clustering_params": {"eps": eps, "min_samples": min_samples},
        "metric_space_analysis": {
            "total_hyperplanes": len(metric_vectors),
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "clusters": []
        }
    }
    
    # Process each cluster
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise
            continue
            
        mask = labels == cluster_id
        cluster_vectors = metric_vectors[mask]
        cluster_hyperplanes = [hyperplane_info[i] for i in range(len(hyperplane_info)) if mask[i]]
        
        cluster_data = {
            "cluster_id": int(cluster_id),
            "size": int(mask.sum()),
            "centroid": cluster_vectors.mean(axis=0).tolist(),
            "std": cluster_vectors.std(axis=0).tolist(),
            "metric_range": {
                "class0_dist": [cluster_vectors[:, 0].min(), cluster_vectors[:, 0].max()],
                "class1_dist": [cluster_vectors[:, 1].min(), cluster_vectors[:, 1].max()]
            },
            "hyperplanes": []
        }
        
        # Add hyperplane details
        for hp_info in cluster_hyperplanes:
            cluster_data["hyperplanes"].append({
                "run_id": hp_info["run_id"],
                "unit_idx": hp_info["unit_idx"],
                "weight_norm": hp_info["weight_norm"],
                "separation_ratio": hp_info["separation_ratio"]
            })
        
        # Sort hyperplanes by run_id for consistency
        cluster_data["hyperplanes"].sort(key=lambda x: (x["run_id"], x["unit_idx"]))
        
        result["metric_space_analysis"]["clusters"].append(cluster_data)
    
    # Sort clusters by size (largest first)
    result["metric_space_analysis"]["clusters"].sort(key=lambda x: x["size"], reverse=True)
    
    return result


def analyze_mirror_weights(run_results):
    """
    Detect mirror weight pairs (w_i ≈ -w_j) in ReLU models.
    Test if ReLU models learn mirror weight pairs (w_i ≈ -w_j).
    """
    mirror_results = []

    for result in run_results:
        if result.get("accuracy", 0) < 0.75:  # Only test successful models
            continue

        # Load model weights
        state_dict = result["model_state_dict"]
        W = state_dict["linear1.weight"]  # First layer weights

        # Normalize weights for comparison
        W_norm = W / W.norm(dim=1, keepdim=True)

        # Find mirror pairs
        mirror_pairs = []
        for i in range(W_norm.shape[0]):
            for j in range(i + 1, W_norm.shape[0]):
                # Check if w_i ≈ -w_j
                cosine_sim = (W_norm[i] @ W_norm[j]).item()
                if cosine_sim < -0.95:  # Close to -1
                    mirror_pairs.append((i, j, cosine_sim))

        mirror_results.append(
            {"run_id": result["run_id"], "mirror_pairs": mirror_pairs, "mirror_count": len(mirror_pairs)}
        )

    return mirror_results


def analyze_failure_angles(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    w_ideal_A = torch.tensor([0.54, -0.54])
    w_ideal_B = torch.tensor([-0.54, 0.54])

    layer_results = {}

    for r in run_results:
        acc = r.get("accuracy", 0.0)
        run_dir = r["run_dir"]
        init_path = run_dir / "model_init.pt"

        try:
            init_weights = torch.load(init_path, map_location="cpu")
        except Exception as e:
            print(f"⚠️ Skipping run {r['run_id']}: failed to load init weights ({e})")
            raise e

        for name, tensor in init_weights.items():
            if not name.endswith(".weight"):
                continue
            if tensor.ndim != 2 or tensor.shape[1] != 2:
                continue  # Only process 2D input layers

            layer_name = name.replace(".weight", "")

            if layer_name not in layer_results:
                layer_results[layer_name] = {"success": [], "failure": []}

            for i in range(tensor.shape[0]):
                w = tensor[i]
                angle_to_A = compute_angles_between(w.unsqueeze(0), w_ideal_A.unsqueeze(0))[0]
                angle_to_B = compute_angles_between(w.unsqueeze(0), w_ideal_B.unsqueeze(0))[0]
                angle = min(angle_to_A, angle_to_B)

                if acc >= 0.99:
                    layer_results[layer_name]["success"].append(angle)
                elif abs(acc - 0.5) < 1e-3:
                    layer_results[layer_name]["failure"].append(angle)

    # Add summary stats per layer
    for layer_name, data in layer_results.items():
        layer_results[layer_name]["summary"] = {
            "success_stats": compute_summary_statistics({"angle_diff": data["success"]}),
            "failure_stats": compute_summary_statistics({"angle_diff": data["failure"]}),
        }

    return layer_results


def analyze_accuracy_distribution(run_results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Analyze accuracy distribution patterns across training runs.

    Args:
        run_results: List of results from all training runs
        config: Experiment configuration for context

    Returns:
        Dictionary containing comprehensive accuracy analysis
    """
    if not run_results:
        return {"error": "No run results provided"}

    print(f"  Analyzing accuracy distribution for {len(run_results)} runs...")

    # Extract accuracy data
    accuracies = [r.get("accuracy", 0.0) for r in run_results]

    if not accuracies:
        return {"error": "No accuracy data found in run results"}

    # Create comprehensive accuracy analysis
    accuracy_analysis = {
        "raw_data": {"accuracies": accuracies, "num_runs": len(accuracies)},
        "summary_statistics": compute_accuracy_summary_stats(accuracies),
        "distribution_analysis": analyze_xor_accuracy_distribution(accuracies),
    }

    return accuracy_analysis


def analyze_weight_reorientation(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze weight reorientation patterns (angles and norm ratios) across all linear layers
    and training runs. Supports multiple layers per model.

    Args:
        run_results: List of results from all training runs

    Returns:
        Dictionary containing layer-wise reorientation analysis
    """
    # Aggregated data per layer
    layer_data = {}

    for result in run_results:
        run_dir = result["run_dir"]
        run_id = result["run_id"]
        epochs = result.get("epochs_completed", None)
        layer_map = result.get("model_linear_layers", {})

        if epochs is None or not layer_map:
            continue

        try:
            model_final = torch.load(run_dir / "model.pt", map_location="cpu")
            model_init = torch.load(run_dir / "model_init.pt", map_location="cpu")
        except Exception as e:
            print(f"⚠️ Run {run_id}: Failed to load weight vectors: {e}")
            raise e

        for layer_name in layer_map.keys():
            try:
                w_final = model_final[f"{layer_name}.weight"].squeeze()
                w_init = model_init[f"{layer_name}.weight"].squeeze()

                angles_per_unit = compute_angles_between(w_init, w_final)
                ratios_per_unit = compute_norm_ratios(w_init, w_final)

                # Initialize storage for layer if not already
                if layer_name not in layer_data:
                    layer_data[layer_name] = {"angles": [], "norm_ratios": [], "epochs_completed": []}

                layer_data[layer_name]["angles"].extend(angles_per_unit)
                layer_data[layer_name]["norm_ratios"].extend(ratios_per_unit)
                layer_data[layer_name]["epochs_completed"].extend([epochs] * len(angles_per_unit))

            except Exception as e:
                print(f"⚠️ Run {run_id}, layer {layer_name}: Failed to process weights: {e}")
                raise e

    # Final analysis output
    output = {"per_layer_analysis": {}, "raw_data": layer_data}

    for layer_name, data in layer_data.items():
        angle_stats = compute_percentile_bins(data["angles"], data["epochs_completed"], metric="angle")
        ratio_stats = compute_percentile_bins(data["norm_ratios"], data["epochs_completed"], metric="norm_ratio")

        output["per_layer_analysis"][layer_name] = {"angle_analysis": angle_stats, "norm_ratio_analysis": ratio_stats}

    return output


def compute_percentile_bins(values: List[float], epochs: List[int], metric: str) -> Dict[str, Any]:
    """
    Bin data by percentiles and compute mean epochs for each bin.

    Args:
        values: List of metric values (angles or norm ratios)
        epochs: List of corresponding epochs completed
        metric_name: Name of the metric for labeling

    Returns:
        Dictionary with percentile bin statistics
    """
    if not values:
        return {}

    # Convert to numpy for percentile calculations
    values_array = np.array(values)
    epochs_array = np.array(epochs)

    # Define percentile boundaries
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    boundaries = np.percentile(values_array, percentiles)

    bin_stats = {}

    # Create bins: 0-10%, 10-25%, 25-50%, 50-75%, 75-90%, 90-100%
    bin_ranges = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100)]

    for i, (low_pct, high_pct) in enumerate(bin_ranges):
        low_val = boundaries[percentiles.index(low_pct)]
        high_val = boundaries[percentiles.index(high_pct)]

        # Find values in this range
        if i == len(bin_ranges) - 1:  # Last bin includes upper boundary
            mask = (values_array >= low_val) & (values_array <= high_val)
        else:
            mask = (values_array >= low_val) & (values_array < high_val)

        if np.any(mask):
            bin_epochs = epochs_array[mask]
            mean_epochs = np.mean(bin_epochs)

            bin_stats[f"{low_pct}–{high_pct}%"] = {
                "range": (low_val, high_val),
                "mean_epochs": mean_epochs,
                "count": len(bin_epochs),
            }

    return bin_stats


def extract_layer_parameters(run_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract all layer parameters from all runs.
    Groups by layer name, preserves all param names.

    Args:
        run_results: List of results from all runs.

    Returns:
        Dict[layer_name, List[Dict[param_name -> value]]]
    """
    layer_data = defaultdict(list)

    for result in run_results:
        run_id = result.get("run_id", "?")
        try:
            state_dict = result["model_state_dict"]
            layer_params = defaultdict(dict)

            for key, tensor in state_dict.items():
                # print(f"key = {key}")
                if '.' not in key:
                    continue  # skip non-layer keys if any

                prefix, param = key.rsplit('.', 1)
                layer_params[prefix][param] = tensor.cpu().numpy()

            for layer_name, params in layer_params.items():
                # print(f"layer_name = {layer_name}, params={params}")
                params["run_id"] = run_id
                layer_data[layer_name].append(params)

        except Exception as e:
            print(f"⚠️ Run {run_id}: Failed to extract parameters: {e}")
            raise e

    return dict(layer_data)


def collate_layer_entries(layer_entries):
    """
    Robustly collate *any* param keys in layer_entries.
    Handles single param (like scale1) or multiple (like linear1).

    Returns:
        {
            'weight': np.ndarray,
            'bias': np.ndarray,
            ...
            'run_ids': list
        }
    """
    if not layer_entries:
        return {}

    # Discover param keys (ignore 'run_id')
    example_keys = list(layer_entries[0].keys())
    param_keys = [k for k in example_keys if k != "run_id"]
    base_run_ids = [entry['run_id'] for entry in layer_entries]
    # print(f"layer_entries = {layer_entries}")
    # print(f"example_keys = {param_keys}")

    result = {}
    unit_ids = []

    for key in param_keys:
        param_list = [entry[key] for entry in layer_entries]
        arr = np.stack(param_list)

        if arr.ndim == 3:
            # Linear: (runs, units, features)
            runs, units, features = arr.shape
            flat_arr = arr.reshape(-1, features)
            # Expand run_id/unit_id
            run_ids = np.repeat(base_run_ids, units)
            unit_ids.extend(list(np.tile(np.arange(units), runs)))

        elif arr.ndim == 2:
            # Scale: (runs, units)
            runs, units = arr.shape
            flat_arr = arr.reshape(-1, 1)
            run_ids = np.repeat(base_run_ids, units)
            unit_ids.extend(list(np.tile(np.arange(units), runs)))

        elif arr.ndim == 1:
            runs = arr.shape[0]
            units = 1
            flat_arr = arr.reshape(-1, 1)
            run_ids = base_run_ids  # one-to-one
            unit_ids.extend([None] * runs)  # No units to repeat

        else:
            raise ValueError(f"Unsupported param shape: {arr.shape}")

        result[key] = flat_arr

    # Run IDs — always a flat list, repeat later if needed
    result['run_ids'] = np.repeat(base_run_ids, units).tolist()
    result['unit_ids'] = unit_ids

    return result

def cluster_units(
    weights_array: np.ndarray, eps: float, min_samples: int, metric: str = "euclidean"
) -> Tuple[np.ndarray, int, int]:
    """
    Perform DBSCAN clustering on weight data.

    Args:
        weights_array: Array of weight vectors to cluster
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples per cluster

    Returns:
        Tuple of (cluster_labels, n_clusters, noise_count)
    """
    # print(f"eps = {eps}")
    # print(f"min_samples = {min_samples}")
    # print(f"metric = {metric}")
    # print(f"weights_array = {weights_array}")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(weights_array)
    labels = clustering.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = int(np.sum(labels == -1))

    return labels, n_clusters, noise_count

def format_clustering_results(
    layer_name: str,
    layer_results: Dict,
    run_ids: List,
    eps: float,
    min_samples: int,
) -> Dict[str, Any]:
    """
    Format clustering results into the expected output structure.

    Args:
        layer_name: Name of the layer
        labels: Cluster labels from DBSCAN
        weights_array: Original weight data
        biases_array: Original bias data
        run_ids: Run identifiers
        n_clusters: Number of clusters found
        noise_count: Number of noise points
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter

    Returns:
        Formatted clustering results dictionary
    """
    layer_info = {
        "layer_name": layer_name,
        "clustering_params": {"eps": eps, "min_samples": min_samples},
    }
    for param_key, param_value in layer_results.items():
        param_cluster_labels = param_value["labels"]    # cluster ids
        param_cluster_array = param_value["array"]      # cluster data
        param_n_clusters = param_value["n_clusters"] # number of clusters
        param_noise_count = param_value["noise_count"] # number of noise points

        layer_info[param_key] = {
            "param_label": param_key,
            "param_data": [],
            "n_clusters": param_n_clusters,
            "noise_count": param_noise_count
        }
        param_list = layer_info[param_key]["param_data"]

        for cluster_label in np.unique(param_cluster_labels):
            if cluster_label == -1:  # Skip noise points
                continue

            mask = param_cluster_labels == cluster_label
            cluster_array = param_cluster_array[mask]
            cluster_run_ids = [run_ids[i] for i in range(len(run_ids)) if mask[i]]

            param_list.append({
                "cluster_label": cluster_label,
                "size": int(np.sum(mask)),
                "run_ids": cluster_run_ids,
                "centroid": cluster_array.mean(axis=0).tolist(),
                "std": cluster_array.std(axis=0).tolist(),
            })

    # print(f"layer_info = {layer_info}")
    return layer_info

def analyze_hyperplane_clustering(
    run_results: List[Dict[str, Any]], config: ExperimentConfig, eps: float = 0.1, min_samples: int = 2
) -> Dict[str, Any]:
    """
    Cluster final hyperplane weights across runs, layer by layer, using DBSCAN.

    Args:
        run_results: List of results from all training runs.
        eps: DBSCAN epsilon parameter (distance threshold).
        min_samples: DBSCAN minimum samples per cluster.

    Returns:
        Dictionary mapping layer names to their clustering results.
    """
    accuracy_threshold = config.analysis.accuracy_threshold
    successful_runs = [r for r in run_results if r.get("accuracy", 0) >= accuracy_threshold]
    if not successful_runs:
        print("⚠️ No runs met the accuracy threshold. Skipping clustering.")
        return {}

    # Extract all parameter data from runs
    layer_data = extract_layer_parameters(successful_runs)

    # print(f"layer_data.keys() = {layer_data.keys()}")

    cluster_results = {}
    for layer_name, layer_entries in layer_data.items():
        try:
            # Prepare data for clustering
            collated_parameters   = collate_layer_entries(layer_entries)
            run_ids = collated_parameters  ["run_ids"]

            # print(f"collated_parameters = {collated_parameters  }")

            layer_results = {}
            for param_key, flat_array in collated_parameters.items():
                if param_key == "run_ids":
                    continue
                if param_key == "unit_ids":
                    continue

                # print(f"{layer_name} {param_key}")
                labels, n_clusters, noise_count = cluster_units(flat_array, eps, min_samples)

                # 👉 Store the raw cluster output in a clean shape:
                layer_results[param_key] = {
                    "labels": labels,
                    "array": flat_array,
                    "n_clusters": n_clusters,
                    "noise_count": noise_count
                }

            # Format results
            cluster_results[layer_name] = format_clustering_results(layer_name, layer_results, run_ids, eps, min_samples)

        except Exception as e:
            cluster_results[layer_name] = {"error": f"Failed to cluster: {str(e)}"}
            raise e

    return cluster_results


def compute_accuracy_summary_stats(accuracies: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive summary statistics for accuracy values.

    Args:
        accuracies: List of accuracy values

    Returns:
        Dictionary of summary statistics
    """
    if not accuracies:
        return {}

    acc_tensor = torch.tensor(accuracies, dtype=torch.float32)

    stats = {
        "count": len(accuracies),
        "mean": acc_tensor.mean().item(),
        "std": acc_tensor.std().item() if len(accuracies) > 1 else 0.0,
        "min": acc_tensor.min().item(),
        "max": acc_tensor.max().item(),
        "median": acc_tensor.median().item(),
        "range": acc_tensor.max().item() - acc_tensor.min().item(),
    }

    # Quartiles
    if len(accuracies) > 3:
        stats["q25"] = torch.quantile(acc_tensor, 0.25).item()
        stats["q75"] = torch.quantile(acc_tensor, 0.75).item()
        stats["iqr"] = stats["q75"] - stats["q25"]
    else:
        stats["q25"] = stats["min"]
        stats["q75"] = stats["max"]
        stats["iqr"] = stats["range"]

    # Additional statistics
    if stats["mean"] > 0:
        stats["coefficient_of_variation"] = stats["std"] / stats["mean"]
    else:
        stats["coefficient_of_variation"] = 0.0

    # Skewness approximation (Pearson's second skewness coefficient)
    if stats["std"] > 0:
        stats["skewness"] = 3 * (stats["mean"] - stats["median"]) / stats["std"]
    else:
        stats["skewness"] = 0.0

    return stats


def analyze_xor_accuracy_distribution(accuracies: List[float]) -> Dict[str, Any]:
    """
    Analyze XOR-specific accuracy distribution (discrete levels).

    Args:
        accuracies: List of accuracy values

    Returns:
        Dictionary of XOR accuracy analysis
    """
    # XOR can only achieve these accuracy levels (out of 4 total points)
    xor_levels = {
        0.0: "0/4 correct (complete failure)",
        0.25: "1/4 correct (minimal learning)",
        0.5: "2/4 correct (partial learning)",
        0.75: "3/4 correct (near success)",
        1.0: "4/4 correct (perfect solution)",
    }

    # Count occurrences at each level
    level_counts = {level: 0 for level in xor_levels.keys()}
    level_percentages = {}

    for accuracy in accuracies:
        # Round to nearest XOR level
        if accuracy <= 0.125:
            level_counts[0.0] += 1
        elif accuracy <= 0.375:
            level_counts[0.25] += 1
        elif accuracy <= 0.625:
            level_counts[0.5] += 1
        elif accuracy <= 0.875:
            level_counts[0.75] += 1
        else:
            level_counts[1.0] += 1

    # Convert to percentages
    total_runs = len(accuracies)
    for level in level_counts:
        level_percentages[level] = (level_counts[level] / total_runs) * 100

    return {
        "distribution_type": "discrete_xor",
        "level_counts": level_counts,
        "level_percentages": level_percentages,
        "level_descriptions": xor_levels,
    }


def plot_run_hyperplanes(run_results: List[Dict], config: ExperimentConfig, output_dir: Path) -> None:
    """
    Generate XOR geometric visualizations (prototype surface plots) per run.
    Loads each model from its run directory and plots learned hyperplanes.
    """

    for run_result in run_results:
        run_id = run_result.get("run_id")
        config_summary = run_result.get("config_summary")
        experiment_name = config_summary.get("experiment_name")

        # Load a fresh instance of the model from state dict
        model = config.model
        state_dict = {
            k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
            for k, v in run_result["model_state_dict"].items()
        }
        model.load_state_dict(state_dict)
        model.eval()

        # Create output directory
        plot_dir = output_dir / f"{run_id:03d}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Prepare XOR data
        x = config.data.x
        y = config.data.y

        # Plot using the styled helper
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_plot_path = plot_dir / f"hyperplanes_{name}.png"
                plot_hyperplanes(
                    module.weight,
                    module.bias,
                    x=x,
                    y=y,
                    title=f"{experiment_name} Run {run_id:03d} (Trained) \n Layer {name}",
                    filename=layer_plot_path,
                )
        init_model_path = run_result["run_dir"] / "model_init.pt"
        if init_model_path.exists():
            # Reconstruct the model and load initial weights
            initial_model = config.model
            initial_weights = torch.load(init_model_path, map_location="cpu")
            initial_model.load_state_dict(initial_weights)

            # Plot using the same helper
            for name, module in initial_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    layer_plot_path = plot_dir / f"init_hyperplanes_{name}.png"
                    plot_hyperplanes(
                        module.weight,
                        module.bias,
                        x=x,
                        y=y,
                        title=f"{experiment_name} Run {run_id:03d} (Initial) — Layer {name}",
                        filename=layer_plot_path,
                    )


def export_analysis_data(analysis_results: Dict, output_dir: Path, filename: str = "analysis_data.json") -> None:
    """
    Save the structured analysis results to a JSON file.

    Args:
        analysis_results: Dictionary containing computed analysis metrics and results.
        output_dir: Path to the directory where the file will be saved.
        filename: Name of the output JSON file.
    """
    output_path = output_dir / filename

    # Convert any non-serializable objects (e.g., torch tensors, numpy arrays)
    def safe_convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: safe_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_convert(v) for v in obj]
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_convert(analysis_results), f, indent=2, ensure_ascii=False)


def main() -> int:
    """
    Main entry point for analysis script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse command line arguments
        if len(sys.argv) != 2:
            print("Usage: python analyze.py <experiment_name>")
            print("\nAvailable experiments:")
            available_experiments = list_experiments()
            for exp in available_experiments:
                print(f"  - {exp}")
            return 1

        experiment_name = sys.argv[1]
        print(f"Analyzing experiment results: {experiment_name}")
        print("=" * 50)

        # Load the experiment configuration
        print("Loading experiment configuration...")
        try:
            config = get_experiment_config(experiment_name)
            print(f"✓ Configuration loaded successfully")
        except KeyError:
            print(f"✗ Unknown experiment: {experiment_name}")
            print("\nAvailable experiments:")
            available_experiments = list_experiments()
            for exp in available_experiments:
                print(f"  - {exp}")
            return 1
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            traceback.print_exc()
            return 1

        # Display experiment info
        print(f"\nExperiment Details:")
        print(f"  Description: {config.description}")
        print(f"  Model Type: {type(config.model).__name__}")
        print(f"  Expected Runs: {config.execution.num_runs}")

        # Get the linear layer information
        linear_layers = get_linear_layers(config.model)

        # Check if results directory exists
        results_dir = Path("results") / experiment_name
        if not results_dir.exists():
            print(f"✗ Results directory not found: {results_dir}")
            print("  Run the experiment first using: python run.py {experiment_name}")
            return 1

        print(f"✓ Results directory found: {results_dir}")

        # Load experiment data and results
        print("Loading experiment data and results...")
        experiment_data = load_experiment_data(config)
        run_results = load_all_run_results(results_dir, config)
        print(f"✓ Loaded results from {len(run_results)} runs")

        # Perform comprehensive analysis
        print("\nPerforming analysis...")
        print("-" * 30)

        analysis_results = {}

        # Basic statistics and aggregation
        if True:
            print("📊 Computing basic statistics...")
            analysis_results["basic_stats"] = compute_basic_statistics(run_results, config)
            print("  ✓ Basic statistics computed")

        # Accuracy and convergence analysis
        if True:
            print("🎯 Analyzing accuracy patterns...")
            analysis_results["accuracy"] = analyze_accuracy_distribution(run_results, config)
            print("  ✓ Accuracy analysis completed")

        # Convergence timing analysis
        if True:  # Add this to your analysis plan
            print("⏱️ Analyzing convergence timing...")
            convergence_epochs = [run_data.get("epochs_completed", None) for run_data in run_results]
            percentiles_data = {
                "0th": int(np.percentile(convergence_epochs, 0)),
                "10th": int(np.percentile(convergence_epochs, 10)),
                "25th": int(np.percentile(convergence_epochs, 25)),
                "50th": int(np.percentile(convergence_epochs, 50)),
                "75th": int(np.percentile(convergence_epochs, 75)),
                "90th": int(np.percentile(convergence_epochs, 90)),
                "100th": int(np.percentile(convergence_epochs, 100)),
            }
            analysis_results["convergence_timing"] = {
                "epochs_list": convergence_epochs,
                "percentiles": percentiles_data,
            }
            print("  ✓ Convergence timing analysis completed")

        # Weight reorientation analysis
        if config.analysis.parameter_displacement:
            print("🔄 Analyzing weight reorientation...")
            analysis_results["weight_reorientation"] = analyze_weight_reorientation(run_results)
            print("  ✓ Weight reorientation analysis completed")

        # Hyperplane clustering analysis
        if config.analysis.hyperplane_clustering:
            print("🎯 Analyzing hyperplane clustering...")
            analysis_results["hyperplane_clustering"] = analyze_hyperplane_clustering(run_results, config)
            print("  ✓ Hyperplane clustering analysis completed")

        # # Geometric analysis (hyperplanes, prototype regions)
        # if "geometric_analysis" in analysis_plan:
        #     print("📐 Performing geometric analysis...")
        #     analysis_results["geometric"] = analyze_learned_geometry(
        #         run_results, experiment_data, config, plot_config
        #     )
        #     print("  ✓ Geometric analysis completed")

        # # Weight pattern analysis
        # if config.analysis.weight_analysis:
        #     print("⚖️  Analyzing weight patterns...")
        #     analysis_results["weights"] = analyze_weight_patterns(
        #         run_results, config
        #     )
        #     print("  ✓ Weight analysis completed")
       
        # Distance to hyperplanes analysis
        if config.analysis.distance_to_hyperplanes:
            print("📏 Analyzing distance to hyperplanes...")
            analysis_results["distance_to_hyperplanes"] = analyze_hyperplane_metric_clustering(run_results, experiment_data, config)
            print("  ✓ Distance to hyperplanes analysis completed")

        # Mirror weight detection
        if config.analysis.mirror_weight_detection:
            print("🔍 Analyzing mirror weights...")
            analysis_results["mirror_weights"] = analyze_mirror_weights(run_results)
            print("  ✓ Mirror weight analysis completed")
            
        if config.analysis.failure_angle_analysis:
            print("📐 Analyzing failure angles ...")
            analysis_results["failure_angle_analysis"] = analyze_failure_angles(run_results)
            print("  ✓ Failure angle analysis completed")

            # print failure angles for each layer
            for layer_name, layer_data in analysis_results["failure_angle_analysis"].items():
                success_angles = layer_data.get("success", [])
                failure_angles = layer_data.get("failure", [])

                plot_failure_angle_histogram(
                    success_angles=success_angles,
                    failure_angles=failure_angles,
                    output_path=results_dir / "plots" / f"{experiment_name}_{layer_name}_failure_angle_histogram.png",
                    title=f"{experiment_name} – {layer_name}",
                )

        # Dead data analysis
        if config.analysis.dead_data_analysis:
            print("💀 Analyzing data data in initial model ...")
            analysis_results["dead_data"] = analyze_dead_data(run_results, config)
            print("  ✓ data data in initial model analysis completed")

        # Generate visualizations
        if config.analysis.plot_hyperplanes:
            print("📈 Plotting Hyperplanes...")
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            plot_run_hyperplanes(run_results=run_results, config=config, output_dir=plots_dir)
            print(f"  ✓ Hyperplane plots saved to {plots_dir}")

        # Generate convergence plots
        if config.analysis.plot_epoch_distribution:
            output_dir = Path("results") / config.execution.experiment_name
            plot_epoch_distribution(
                run_results,
                plot_config={
                    "save_plots": True,
                    "format": config.analysis.plot_format,
                    "dpi": config.analysis.plot_dpi,
                    "interactive": False,
                },
                output_dir=output_dir,
                experiment_name=config.execution.experiment_name,
            )

        if config.analysis.plot_parameter_displacement:
            output_dir = Path("results") / config.execution.experiment_name
            layer_names = run_results[0].get("model_linear_layers", [])
            for layer_name in layer_names:
                plot_weight_angle_and_magnitude_vs_epochs(
                    run_results=run_results,
                    layer_name=layer_name,
                    output_dir=output_dir,
                    experiment_name=config.execution.experiment_name,
                )

        # Generate comprehensive report
        print("📄 Generating analysis report...")
        report = generate_analysis_report(analysis_results, config, template="comprehensive")

        # Save report
        report_path = results_dir / f"analysis_{experiment_name}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  ✓ Report saved to {report_path}")

        # Export analysis data
        if True:
            print("💾 Exporting analysis data...")
            export_analysis_data(analysis_results, results_dir, "analysis_data.json")
            print("  ✓ Analysis data exported")

        print("-" * 30)
        print("✓ Analysis completed successfully!")

        return 0

    except KeyboardInterrupt:
        print("\n✗ Analysis interrupted by user")
        return 130

    except Exception as e:
        print(f"\n✗ Unexpected error during experiment analysis:")
        print(f"  {type(e).__name__}: {e}")

        print("\nFull traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
