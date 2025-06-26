# analyze.py - Post-Experiment Analysis and Visualization for Prototype Surface Experiments

"""
Comprehensive analysis module for prototype surface experiments.
Provides geometric analysis, visualization, and prototype surface validation tools.
Focuses on prototype surface investigation, distance field analysis, and
comparative studies across activation functions and training runs.
"""

import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import traceback

from collections import defaultdict
from configs import ExperimentConfig, ExperimentType, get_experiment_config, list_experiments
from pathlib import Path
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Any
from analysis.loader import load_experiment_data, load_all_run_results
from analysis.utils import targets_to_class_labels, extract_activation_type, get_linear_layers
from analysis.stats import compute_basic_statistics, compute_summary_statistics

def configure_analysis_from_config(config: ExperimentConfig) -> Tuple[List[str], Dict[str, Any]]:
    """
    Configure analysis pipeline based on experiment configuration.
    
    Args:
        config: Complete experiment configuration
        
    Returns:
        Tuple of (analysis_plan, plot_config) where:
        - analysis_plan: List of analysis types to perform
        - plot_config: Dictionary of visualization configuration
    """
    analysis_plan = []
    
    # Always include basic statistics
    analysis_plan.append('basic_stats')
    
    # Always analyze accuracy and convergence for training experiments
    analysis_plan.append('accuracy_analysis')
    
    analysis_plan.append('convergence_timing')
    analysis_plan.append('weight_reorientation')
    analysis_plan.append('hyperplane_clustering')
    
    # Geometric analysis
    if config.analysis.geometric_analysis:
        analysis_plan.extend([
            'hyperplane_plots',
            'prototype_regions'
        ])
        
        if config.analysis.decision_boundary_analysis:
            analysis_plan.append('decision_boundaries')
            
        if config.analysis.distance_field_analysis:
            analysis_plan.append('distance_fields')
    
    # Weight analysis
    if config.analysis.weight_analysis:
        analysis_plan.append('weight_patterns')
        
        if config.analysis.mirror_pair_detection:
            analysis_plan.append('mirror_weights')
            
        if config.analysis.symmetry_analysis:
            analysis_plan.append('weight_symmetry')
            
        if config.analysis.weight_evolution_tracking:
            analysis_plan.append('weight_evolution')
            
        if config.analysis.weight_clustering:
            analysis_plan.append('weight_clustering')
    
    # Activation analysis
    if config.analysis.activation_analysis:
        analysis_plan.extend([
            'activation_patterns',
            'activation_landscapes'
        ])
        
        if config.analysis.sparsity_analysis:
            analysis_plan.append('activation_sparsity')
            
        if config.analysis.zero_activation_regions:
            analysis_plan.append('zero_activation_analysis')
    
    # Convergence and training dynamics
    if config.analysis.convergence_analysis:
        analysis_plan.append('convergence_analysis')
        
        if config.analysis.training_dynamics:
            analysis_plan.append('training_dynamics')
            
        if config.analysis.loss_landscape_analysis:
            analysis_plan.append('loss_landscape')
    
    # Cross-run comparison (only if multiple runs)
    if config.analysis.cross_run_comparison and config.execution.num_runs > 1:
        analysis_plan.extend([
            'run_consistency',
            'solution_stability'
        ])
        
        if config.analysis.statistical_analysis:
            analysis_plan.append('statistical_analysis')
            
        if config.analysis.stability_analysis:
            analysis_plan.append('stability_metrics')
    
    # Prototype surface theory validation
    if config.analysis.prototype_surface_analysis:
        analysis_plan.append('prototype_surface')
        
        if config.analysis.separation_order_analysis:
            analysis_plan.append('separation_order')
            
        if config.analysis.minsky_papert_metrics:
            analysis_plan.append('minsky_papert_analysis')
    
    # Problem-specific analysis
    if config.data.problem_type == ExperimentType.XOR:
        analysis_plan.extend([
            'xor_specific_analysis',
            'xor_accuracy_distribution'
        ])
    elif config.data.problem_type == ExperimentType.PARITY:
        analysis_plan.append('parity_specific_analysis')
    
    # Model-specific analysis
    model_type = type(config.model).__name__
    
    if model_type == 'Model_Abs1':
        analysis_plan.extend([
            'absolute_value_theory',
            'single_unit_analysis',
            'abs_distance_validation'
        ])
    elif 'ReLU' in model_type:
        analysis_plan.extend([
            'relu_decomposition_analysis',
            'relu_mirror_validation'
        ])
    elif 'Sigmoid' in model_type:
        analysis_plan.append('sigmoid_saturation_analysis')
    
    # Activation-specific analysis based on model architecture
    activation_type = extract_activation_type(config.model)
    if activation_type == 'abs':
        analysis_plan.append('absolute_value_separation_order_validation')
    elif activation_type == 'relu':
        if config.analysis.mirror_pair_detection:
            analysis_plan.append('mirror_pair_validation')
    
    analysis_plan.append('export_data')

    # Remove duplicates while preserving order
    analysis_plan = list(dict.fromkeys(analysis_plan))
    
    # Configure visualization settings
    plot_config = {
        'save_plots': config.analysis.save_plots,
        'format': config.analysis.plot_format,
        'dpi': config.analysis.plot_dpi,
        'interactive': config.analysis.interactive_plots,
        'style': config.analysis.plot_style,
        
        # Analysis spatial configuration
        'bounds': config.analysis.analysis_bounds,
        'resolution': config.analysis.analysis_resolution,
        
        # Problem-specific visualization settings
        'problem_type': config.data.problem_type,
        'data_points': config.data.x,
        'labels': config.data.y,
        
        # Model-specific visualization settings
        'model_type': model_type,
        'activation_type': activation_type,
        'num_runs': config.execution.num_runs,
        
        # Training context for visualization
        'epochs': config.training.epochs,
        'loss_change_threshold': getattr(config.training, 'loss_change_threshold', 1e-6),
        
        # Experiment metadata
        'experiment_name': config.execution.experiment_name,
        'description': config.description
    }
    
    # Add problem-specific plot settings
    if config.data.problem_type == ExperimentType.XOR:
        plot_config.update({
            'expected_accuracy_levels': [0.0, 0.25, 0.5, 0.75, 1.0],
            'xor_corners': config.data.x,
            'decision_boundary_focus': True
        })
    
    # Add model-specific plot settings
    if model_type == 'Model_Abs1':
        plot_config.update({
            'single_unit_focus': True,
            'distance_field_emphasis': True,
            'prototype_surface_highlight': True
        })
    
    return analysis_plan, plot_config

def analyze_dead_data(run_results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Analyze which input points are dead at initialization (no active ReLU units),
    and correlate with final accuracy.
    """

    model = config.model.__class__()
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
    final_losses = [r.get('final_loss', float('inf')) for r in all_results]
    accuracies = [r.get('accuracy', 0.0) for r in all_results]
    training_times = [r.get('training_time', 0.0) for r in all_results]
    
    # Convert to tensors for easy computation
    losses_tensor = torch.tensor([l for l in final_losses if l != float('inf')])
    accuracies_tensor = torch.tensor(accuracies)
    times_tensor = torch.tensor(training_times)
    
    summary = {
        'total_runs': len(all_results),
        'successful_runs': len(losses_tensor),
        'failed_runs': len(all_results) - len(losses_tensor),
        
        # Loss statistics
        'loss_stats': {
            'mean': losses_tensor.mean().item() if len(losses_tensor) > 0 else float('inf'),
            'std': losses_tensor.std().item() if len(losses_tensor) > 0 else 0.0,
            'min': losses_tensor.min().item() if len(losses_tensor) > 0 else float('inf'),
            'max': losses_tensor.max().item() if len(losses_tensor) > 0 else float('inf'),
            'median': losses_tensor.median().item() if len(losses_tensor) > 0 else float('inf')
        },
        
        # Accuracy statistics
        'accuracy_stats': {
            'mean': accuracies_tensor.mean().item(),
            'std': accuracies_tensor.std().item(),
            'min': accuracies_tensor.min().item(),
            'max': accuracies_tensor.max().item(),
            'median': accuracies_tensor.median().item()
        },
        
        # Training time statistics
        'time_stats': {
            'mean': times_tensor.mean().item(),
            'std': times_tensor.std().item(),
            'min': times_tensor.min().item(),
            'max': times_tensor.max().item(),
            'total': times_tensor.sum().item()
        },
        
        # Success metrics
        'convergence_rate': sum(1 for r in all_results if r.get('converged', False)) / len(all_results),
        'perfect_accuracy_rate': sum(1 for r in all_results if r.get('perfect_accuracy', False)) / len(all_results),
        
        # Best runs
        'best_loss_run': min(all_results, key=lambda r: r.get('final_loss', float('inf')))['run_id'],
        'best_accuracy_run': max(all_results, key=lambda r: r.get('accuracy', 0.0))['run_id'],
    }
    
    return summary

def analyze_prototype_surface(run_results, experiment_data, config):
    # Test 1: Distance of points to hyperplanes
    distance_test = test_distance_to_hyperplanes(run_results, experiment_data, config)
    
    # Test 2: Mirror weight detection  
    mirror_test = test_mirror_weights(run_results)
    
    return {
        'distance_test': distance_test,
        'mirror_test': mirror_test
    }

def test_distance_to_hyperplanes(run_results: List[Dict[str, Any]], experiment_data: Dict[str, Any], config) -> List[Dict[str, Any]]:
    """
    Test if classification correlates with distance to learned hyperplanes
    for each linear layer in the model, using correct layer-specific input data.

    Args:
        run_results: List of run result dicts (with model_state_dict and model_linear_layers)
        experiment_data: Dict with 'x_train' and 'y_train'
        config: Experiment config (used to reconstruct model)

    Returns:
        List of per-run results, each with layer-wise distances and labels.
    """
    x_train = experiment_data['x_train']
    y_train = experiment_data['y_train']
    results = []

    for result in run_results:
        if result.get('accuracy', 0) < 0.75:
            continue

        model = config.model
        model.load_state_dict(result["model_state_dict"])
        model.eval()

        layer_inputs = {}
        current = x_train

        # Run forward pass and record inputs to each Linear layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_inputs[name] = current  # input to this linear layer
                current = module(current)
            elif isinstance(module, (torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid)):
                current = module(current)
            elif name == "scale":
                current = model.scale(current)
            # skip other modules for now

        run_result = {
            "run_id": result["run_id"],
            "layer_distances": {}
        }

        state_dict = result["model_state_dict"]

        for layer_name, x_input in layer_inputs.items():
            W = state_dict[f"{layer_name}.weight"]
            b = state_dict[f"{layer_name}.bias"]

            distances = []
            for i in range(W.shape[0]):
                w = W[i]
                bias = b[i]
                d = torch.abs(w @ x_input.T + bias) / torch.norm(w)
                distances.append(d)  # list of tensors, one per unit

            run_result["layer_distances"][layer_name] = {
                "unit_distances": distances,
                "labels": y_train
            }

        results.append(run_result)

    return results

def test_mirror_weights(run_results):
    """
    Test if ReLU models learn mirror weight pairs (w_i ≈ -w_j).
    """
    mirror_results = []
    
    for result in run_results:
        if result.get('accuracy', 0) < 0.75:  # Only test successful models
            continue
            
        # Load model weights
        state_dict = result['model_state_dict']
        W = state_dict['linear1.weight']  # First layer weights
        
        # Normalize weights for comparison
        W_norm = W / W.norm(dim=1, keepdim=True)
        
        # Find mirror pairs
        mirror_pairs = []
        for i in range(W_norm.shape[0]):
            for j in range(i+1, W_norm.shape[0]):
                # Check if w_i ≈ -w_j
                cosine_sim = (W_norm[i] @ W_norm[j]).item()
                if cosine_sim < -0.95:  # Close to -1
                    mirror_pairs.append((i, j, cosine_sim))
        
        mirror_results.append({
            'run_id': result['run_id'],
            'mirror_pairs': mirror_pairs,
            'mirror_count': len(mirror_pairs)
        })
    
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
            continue

        for name, tensor in init_weights.items():
            if not name.endswith(".weight"):
                continue
            if tensor.ndim != 2 or tensor.shape[1] != 2:
                continue  # Only process 2D input layers

            layer_name = name.replace(".weight", "")

            if layer_name not in layer_results:
                layer_results[layer_name] = {
                    "success": [],
                    "failure": []
                }

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

def plot_failure_angle_histogram(
    success_angles: List[float],
    failure_angles: List[float],
    output_path: Path,
    title: str
):

    plt.figure(figsize=(6, 4), dpi=300)

    if success_angles:
        plt.hist(success_angles, bins=30, alpha=0.6, label="Success (100%)")
    if failure_angles:
        plt.hist(failure_angles, bins=15, alpha=0.8, label="Failure (50%)", color='red')

    plt.axvline(90, color='black', linestyle='--', label='90° (perpendicular)')
    plt.xlabel("Initial Angle Difference to Ideal (degrees)")
    plt.ylabel("Unit Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

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
        return {'error': 'No run results provided'}
    
    print(f"  Analyzing accuracy distribution for {len(run_results)} runs...")
    
    # Extract accuracy data
    accuracies = [r.get('accuracy', 0.0) for r in run_results]
    
    if not accuracies:
        return {'error': 'No accuracy data found in run results'}
    
    # Create comprehensive accuracy analysis
    accuracy_analysis = {
        'raw_data': {
            'accuracies': accuracies,
            'num_runs': len(accuracies)
        },
        
        'summary_statistics': compute_accuracy_summary_stats(accuracies),
        'distribution_analysis': analyze_xor_accuracy_distribution(accuracies),
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
            continue

        for layer_name in layer_map.keys():
            try:
                w_final = model_final[f"{layer_name}.weight"].squeeze()
                w_init = model_init[f"{layer_name}.weight"].squeeze()

                angles_per_unit = compute_angles_between(w_init, w_final)
                ratios_per_unit = compute_norm_ratios(w_init, w_final)

                # Initialize storage for layer if not already
                if layer_name not in layer_data:
                    layer_data[layer_name] = {
                        "angles": [],
                        "norm_ratios": [],
                        "epochs_completed": []
                    }

                layer_data[layer_name]["angles"].extend(angles_per_unit)
                layer_data[layer_name]["norm_ratios"].extend(ratios_per_unit)
                layer_data[layer_name]["epochs_completed"].extend([epochs] * len(angles_per_unit))

            except Exception as e:
                print(f"⚠️ Run {run_id}, layer {layer_name}: Failed to process weights: {e}")
                continue

    # Final analysis output
    output = {
        "per_layer_analysis": {},
        "raw_data": layer_data
    }

    for layer_name, data in layer_data.items():
        angle_stats = compute_percentile_bins(data["angles"], data["epochs_completed"], metric="angle")
        ratio_stats = compute_percentile_bins(data["norm_ratios"], data["epochs_completed"], metric="norm_ratio")

        output["per_layer_analysis"][layer_name] = {
            "angle_analysis": angle_stats,
            "norm_ratio_analysis": ratio_stats
        }

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
    bin_ranges = [
        (0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100)
    ]
    
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
                "count": len(bin_epochs)
            }
    
    return bin_stats

def analyze_hyperplane_clustering(run_results: List[Dict[str, Any]], eps: float = 0.1, min_samples: int = 2) -> Dict[str, Any]:
    """
    Cluster final hyperplane weights across runs, layer by layer, using DBSCAN.

    Args:
        run_results: List of results from all training runs.
        eps: DBSCAN epsilon parameter (distance threshold).
        min_samples: DBSCAN minimum samples per cluster.

    Returns:
        Dictionary mapping layer names to their clustering results.
    """
    layer_data = defaultdict(list)  # layer_name → list of (weights, bias, run_id)

    for result in run_results:
        run_id = result.get("run_id", "?")
        try:
            state_dict = result["model_state_dict"]
            for key in state_dict:
                if ".weight" in key and "bias" not in key:
                    layer_name = key.replace(".weight", "")
                    w_final = state_dict[key].cpu().numpy()
                    b_final = state_dict.get(f"{layer_name}.bias", None)
                    if b_final is not None:
                        b_final = b_final.cpu().numpy()
                    else:
                        b_final = np.zeros(w_final.shape[0])
                    layer_data[layer_name].append((w_final, b_final, run_id))
        except Exception as e:
            print(f"⚠️ Run {run_id}: Failed to extract weights: {e}")
            continue

    results = {}

    for layer_name, entries in layer_data.items():
        try:
            weights, biases, run_ids = zip(*entries)
            weights_array = np.array([w.flatten() for w in weights])
            biases_array = np.array([b.flatten() for b in biases])

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(weights_array)
            labels = clustering.labels_
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            noise_count = int(np.sum(labels == -1))

            cluster_info = {}
            for label in unique_labels:
                if label == -1:
                    continue
                mask = labels == label
                cluster_weights = weights_array[mask]
                cluster_biases = biases_array[mask]
                cluster_run_ids = [run_ids[i] for i in range(len(run_ids)) if mask[i]]
                cluster_info[f"{layer_name}_cluster_{label}"] = {
                    "size": int(np.sum(mask)),
                    "run_ids": cluster_run_ids,
                    "weight_centroid": cluster_weights.mean(axis=0).tolist(),
                    "bias_centroid": cluster_biases.mean(axis=0).tolist(),
                    "weight_std": cluster_weights.std(axis=0).tolist(),
                    "bias_std": cluster_biases.std(axis=0).tolist()
                }

            results[layer_name] = {
                "clustering_params": {"eps": eps, "min_samples": min_samples},
                "n_clusters": n_clusters,
                "noise_points": noise_count,
                "cluster_info": cluster_info
            }
        except Exception as e:
            results[layer_name] = {"error": f"Failed to cluster: {str(e)}"}

    return results

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
        'count': len(accuracies),
        'mean': acc_tensor.mean().item(),
        'std': acc_tensor.std().item() if len(accuracies) > 1 else 0.0,
        'min': acc_tensor.min().item(),
        'max': acc_tensor.max().item(),
        'median': acc_tensor.median().item(),
        'range': acc_tensor.max().item() - acc_tensor.min().item()
    }
    
    # Quartiles
    if len(accuracies) > 3:
        stats['q25'] = torch.quantile(acc_tensor, 0.25).item()
        stats['q75'] = torch.quantile(acc_tensor, 0.75).item()
        stats['iqr'] = stats['q75'] - stats['q25']
    else:
        stats['q25'] = stats['min']
        stats['q75'] = stats['max']
        stats['iqr'] = stats['range']
    
    # Additional statistics
    if stats['mean'] > 0:
        stats['coefficient_of_variation'] = stats['std'] / stats['mean']
    else:
        stats['coefficient_of_variation'] = 0.0
    
    # Skewness approximation (Pearson's second skewness coefficient)
    if stats['std'] > 0:
        stats['skewness'] = 3 * (stats['mean'] - stats['median']) / stats['std']
    else:
        stats['skewness'] = 0.0
    
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
        0.0: '0/4 correct (complete failure)',
        0.25: '1/4 correct (minimal learning)',
        0.5: '2/4 correct (partial learning)',
        0.75: '3/4 correct (near success)',
        1.0: '4/4 correct (perfect solution)'
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
        'distribution_type': 'discrete_xor',
        'level_counts': level_counts,
        'level_percentages': level_percentages,
        'level_descriptions': xor_levels,        
    }

def plot_hyperplanes(weights, biases, x, y, title, filename=None):
    input_dim = weights.shape[-1]
    if input_dim != 2:
        # print(f"⚠️ Skipping plot: hyperplane visualization only supported for 2D inputs (got input_dim={input_dim})")
        return

    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    weights = weights.detach().cpu()  # (n_units, 2)
    biases = biases.detach().cpu()    # (n_units,)
    mean = torch.zeros(input_dim)

    plt.figure(figsize=(6, 6))

    # XOR input points
    y_labels = targets_to_class_labels(y_cpu.unsqueeze(1) if y_cpu.ndim == 1 else y_cpu)
    for xi, yi_label  in zip(x_cpu, y_labels):
        marker = 'o' if yi_label == 0 else '^'
        plt.scatter(xi[0], xi[1], marker=marker, s=100, color='black', edgecolors='k', linewidths=1)

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

        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                 color='black', linewidth=1.5, linestyle='--', label=f'Neuron {i}')

        plt.arrow(
            projection_on_plane[0].item(), projection_on_plane[1].item(),
            normal[0].item() * 0.5, normal[1].item() * 0.5,
            head_width=0.15, head_length=0.2,
            fc='#333333', ec='#333333', alpha=1.0,
            length_includes_head=True, width=0.03, zorder=3
        )

    # Final plot adjustments
    plt.title(title, fontsize=16, weight='bold', pad=12)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tight_layout()

    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

    plt.close()

def generate_experiment_visualizations(
    run_results: List[Dict],
    config: ExperimentConfig,
    output_dir: Path
) -> None:
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
                    filename=layer_plot_path
                )
        init_model_path = run_result['run_dir'] / "model_init.pt"
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
                        filename=layer_plot_path
                    )


def plot_epoch_distribution(run_results: List[Dict[str, Any]], plot_config: Dict[str, Any], output_dir: Path, experiment_name: str) -> None:
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

def compute_angles_between(w_init: torch.Tensor, w_final: torch.Tensor) -> List[float]:
    """
    Vectorized computation of angles (in degrees) between corresponding rows of two weight matrices.

    Args:
        w_init: Tensor of shape [n_units, n_features]
        w_final: Tensor of same shape

    Returns:
        List of angles in degrees
    """
    if w_init.dim() == 1:
        w_init = w_init.unsqueeze(0)
        w_final = w_final.unsqueeze(0)

    assert w_init.shape == w_final.shape, "Mismatched weight shapes"

    # Normalize each row (unit vector per neuron)
    w1_norm = w_init / (w_init.norm(dim=1, keepdim=True) + 1e-8)
    w2_norm = w_final / (w_final.norm(dim=1, keepdim=True) + 1e-8)

    # Compute dot products per row
    cos_theta = (w1_norm * w2_norm).sum(dim=1).clamp(-1.0, 1.0)

    # Convert to degrees
    angles = torch.acos(cos_theta) * (180.0 / math.pi)
    return angles.tolist()

def compute_norm_ratios(w_init: torch.Tensor, w_final: torch.Tensor) -> List[float]:
    """
    Vectorized computation of norm ratios (init / final) per row.

    Args:
        w_init: Tensor of shape [n_units, n_features]
        w_final: Tensor of same shape

    Returns:
        List of norm ratios per neuron
    """
    if w_init.dim() == 1:
        w_init = w_init.unsqueeze(0)
        w_final = w_final.unsqueeze(0)

    assert w_init.shape == w_final.shape, "Mismatched weight shapes"

    init_norms = w_init.norm(dim=1)
    final_norms = w_final.norm(dim=1) + 1e-8  # prevent divide-by-zero
    ratios = init_norms / final_norms

    return ratios.tolist()

def plot_weight_angle_and_magnitude_vs_epochs(run_results: List[Dict[str, Any]], layer_name: str, output_dir: Path, experiment_name: str):
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

def generate_analysis_report(
    analysis_results: Dict[str, Any],
    config: Any,  # Replace Any with actual ExperimentConfig type
    template: str = "comprehensive"
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
    if template != "comprehensive":
        raise ValueError(f"Unsupported template: {template}")

    ############################################################################################

    loss_change_threshold = 0.01 # replaced with config.training.loss_change_threshold

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

    # Now:
    # distance_by_layer_unit[layer][unit][class] = list of distances

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

    name = config.execution.experiment_name
    description = config.description or experiment_info.get("description", "No description provided.")

    # Start Markdown report
    report = f"# 🧪 Experiment Report: `{name}`\n\n"
    report += f"**Description**: {description}\n\n"

    ############################################################################################

    loss_fn = config.training.loss_function

    report += "## 🎯 Overview\n\n"

    # 🏃 Training Configuration
    report += f"* **Total runs**: {config.execution.num_runs}\n"
    report += f"* **Loss function**: {config.training.loss_function.__class__.__name__}\n"
    report += f"* **Optimizer**: {config.training.optimizer.__class__.__name__}\n"
    if config.training.batch_size:
        report += f"* **Batch size**: {config.training.batch_size}\n"
    report += f"* **Max epochs**: {config.training.epochs}\n"

    # ⏹ Early Stopping Criteria
    if config.training.stop_training_loss_threshold is not None:
        report += f"* **Stops when loss < {config.training.stop_training_loss_threshold:.1e}**\n"

    if (
        config.training.loss_change_threshold is not None
        and config.training.loss_change_patience is not None
    ):
        report += (
            f"* **Stops if loss does not improve by ≥ {config.training.loss_change_threshold:.1e} "
            f"over {config.training.loss_change_patience} epochs**\n"
        )

    report += "\n---\n\n"

    ############################################################################################

    acc_bins = distributions.get("accuracy_distribution", {}).get("bins", {})

    report += "## 🎯 Classification Accuracy\n\n"

    for acc in sorted(acc_bins.keys(), reverse=True):
        count = acc_bins.get(acc, 0)
        if count > 0:
            report += f"* {count}/{total_runs} runs achieved {int(100*acc)}% accuracy\n"


    report += "\n---\n\n"

    ############################################################################################

    if config.analysis.convergence_analysis:
        report += "## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)\n\n"
        report += "| Percentile | Epochs |\n| ---------- | ------ |\n"

        if percentiles:
            labels = ["0th", "10th", "25th", "50th", "75th", "90th", "100th"]
            for label in labels:
                value = percentiles.get(label, "N/A")
                report += f"| {label:<10} | {value}     |\n"
        else:
            report += "| N/A        | No convergence data available |\n"

        report += "\n---\n\n"

    ############################################################################################

    report += "\n## 📏 Prototype Surface Geometry\n\n"

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

    ############################################################################################

    per_layer_analysis = weight_reorientation.get("per_layer_analysis", {})

    if per_layer_analysis:
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
    else:
        report += "No weight reorientation data available.\n\n"

    ############################################################################################

    report += "### ◼ Initial / Final Norm Ratio (All Layers Combined)\n\n"
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

    ############################################################################################

    report += "## 📉 Final Loss Distribution\n\n"

    if final_losses:
        # Extract additional statistics we need
        mean_loss = final_losses.get("mean", 0.0)
        min_loss = final_losses.get("min", 0.0) 
        max_loss = final_losses.get("max", 0.0)
        
        # Calculate variance from the raw metrics if available
        raw_metrics = basic_stats['raw_metrics']
        final_loss_values = raw_metrics['final_losses']
        
        variance = np.var(final_loss_values)
        
        report += f"* **Mean final loss**: {mean_loss:.2e}\n\n"
        report += f"* **Variance**: {variance:.2e}\n\n"
        report += f"* **Range**:\n\n"
        report += f"  * 0th percentile: {min_loss:.2e}\n"
        report += f"  * 100th percentile: {max_loss:.2e}\n\n"
    else:
        report += "* **No final loss data available**\n\n"

    report += "\n---\n\n"

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
                report += f"|    {summary['class0_dead']} runs with class-0 dead inputs reached {acc_percent}% accuracy\n"
                report += f"|    {summary['class1_dead']} runs with class-1 dead inputs reached {acc_percent}% accuracy\n"
        
        report += "\n---\n\n"

    ############################################################################################

    if config.analysis.mirror_pair_detection:
        report += "## 🔍 Mirror Weight Symmetry\n\n"

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
            report += f"* **Success units (n={count_s})** – mean angle diff: {stats_s['mean']:.2f}° ± {stats_s['std']:.2f}°\n"
            report += f"* **Failure units (n={count_f})** – mean angle diff: {stats_f['mean']:.2f}° ± {stats_f['std']:.2f}°\n"
            report += "* Failed units tend to cluster near 90°, consistent with the no-torque trap hypothesis.\n\n"

        report += "See `failure_angle_histogram.png` for visual confirmation.\n\n"

    ############################################################################################

    return report

def export_analysis_data(
    analysis_results: Dict,
    output_dir: Path,
    filename: str = "analysis_data.json"
) -> None:
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
            return 1
    
        # Display experiment info
        print(f"\nExperiment Details:")
        print(f"  Description: {config.description}")
        print(f"  Model Type: {type(config.model).__name__}")
        print(f"  Problem Type: {config.data.problem_type}")
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

        # Configure analysis based on experiment config
        print("\nConfiguring analysis pipeline...")
        analysis_plan, plot_config = configure_analysis_from_config(config)
        print(f"✓ Analysis plan configured ({len(analysis_plan)} analysis types)")
        # print(analysis_plan)
        
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
        if "basic_stats" in analysis_plan:
            print("📊 Computing basic statistics...")
            analysis_results["basic_stats"] = compute_basic_statistics(run_results, config)
            print("  ✓ Basic statistics computed")

        # Accuracy and convergence analysis
        if "accuracy_analysis" in analysis_plan:
            print("🎯 Analyzing accuracy patterns...")
            analysis_results["accuracy"] = analyze_accuracy_distribution(run_results, config)
            print("  ✓ Accuracy analysis completed")
            
        # Convergence timing analysis  
        if "convergence_timing" in analysis_plan:  # Add this to your analysis plan
            print("⏱️ Analyzing convergence timing...")
            convergence_epochs = [run_data.get("epochs_completed", None) for run_data in run_results]
            percentiles_data = {
                        "0th": int(np.percentile(convergence_epochs, 0)),
                        "10th": int(np.percentile(convergence_epochs, 10)),
                        "25th": int(np.percentile(convergence_epochs, 25)),
                        "50th": int(np.percentile(convergence_epochs, 50)),
                        "75th": int(np.percentile(convergence_epochs, 75)),
                        "90th": int(np.percentile(convergence_epochs, 90)),
                        "100th": int(np.percentile(convergence_epochs, 100))
                    }
            analysis_results["convergence_timing"] = {
                "epochs_list": convergence_epochs,
                "percentiles": percentiles_data
            }
            print("  ✓ Convergence timing analysis completed")

        # Weight reorientation analysis
        if "weight_reorientation" in analysis_plan:
            print("🔄 Analyzing weight reorientation...")
            analysis_results["weight_reorientation"] = analyze_weight_reorientation(run_results)
            print("  ✓ Weight reorientation analysis completed")

        # Hyperplane clustering analysis
        if "hyperplane_clustering" in analysis_plan:
            print("🎯 Analyzing hyperplane clustering...")
            analysis_results["hyperplane_clustering"] = analyze_hyperplane_clustering(run_results)
            print("  ✓ Hyperplane clustering analysis completed")


        # # Geometric analysis (hyperplanes, prototype regions)
        # if "geometric_analysis" in analysis_plan:
        #     print("📐 Performing geometric analysis...")
        #     analysis_results["geometric"] = analyze_learned_geometry(
        #         run_results, experiment_data, config, plot_config
        #     )
        #     print("  ✓ Geometric analysis completed")

        # # Weight pattern analysis
        # if "weight_analysis" in analysis_plan:
        #     print("⚖️  Analyzing weight patterns...")
        #     analysis_results["weights"] = analyze_weight_patterns(
        #         run_results, config
        #     )
        #     print("  ✓ Weight analysis completed")

        # Prototype surface tests
        if "prototype_surface" in analysis_plan:
            print("🔬 Analyzing prototype surface ...")
            analysis_results["prototype_surface"] = analyze_prototype_surface(
                run_results, experiment_data, config
            )
            print("  ✓ Prototype surface analysis completed")

        if config.analysis.failure_angles:
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
                    title=f"{experiment_name} – {layer_name}"
                )

        # Dead data analysis
        if config.analysis.dead_data_analysis:
            print("💀 Analyzing data data in initial model ...")
            analysis_results["dead_data"] = analyze_dead_data(run_results, config)
            print("  ✓ data data in initial model analysis completed")

        # Generate visualizations
        if config.analysis.save_plots:
            print("📈 Generating visualizations...")
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            generate_experiment_visualizations(
                run_results=run_results,
                config=config,
                output_dir=plots_dir
            )
            print(f"  ✓ Visualizations plots saved to {plots_dir}")

        # Generate convergence plots
        if config.analysis.convergence_analysis:
            output_dir = Path("results") / config.execution.experiment_name
            plot_epoch_distribution(run_results, plot_config={
                'save_plots': config.analysis.save_plots,
                'format': config.analysis.plot_format,
                'dpi': config.analysis.plot_dpi,
                'interactive': config.analysis.interactive_plots,
                'style': config.analysis.plot_style
            }, 
            output_dir=output_dir,
            experiment_name=config.execution.experiment_name)

        if config.analysis.convergence_analysis:
            output_dir = Path("results") / config.execution.experiment_name
            layer_names = run_results[0].get("model_linear_layers", [])
            for layer_name in layer_names:
                plot_weight_angle_and_magnitude_vs_epochs(
                    run_results=run_results,
                    layer_name=layer_name,
                    output_dir=output_dir,
                    experiment_name=config.execution.experiment_name
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
        if "export_data" in analysis_plan:
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
