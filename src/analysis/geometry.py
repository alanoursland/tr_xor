import torch
import math
import numpy as np
import torch.nn as nn

from typing import Dict, List, Tuple, Any


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


def create_boundary_focused_grid(
    data_points: torch.Tensor, bounds: List[Tuple[float, float]], resolution: int = 50
) -> torch.Tensor:
    """
    Create sampling grid with higher density near data points for boundary analysis.

    Args:
        data_points: Original data points
        bounds: Analysis bounds for each dimension
        resolution: Base resolution for grid

    Returns:
        Grid points with higher density near data
    """
    if len(bounds) != 2:
        return None  # Only implemented for 2D

    # Create base uniform grid
    x_range = torch.linspace(bounds[0][0], bounds[0][1], resolution)
    y_range = torch.linspace(bounds[1][0], bounds[1][1], resolution)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="ij")
    base_grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Add additional points near each data point
    focused_points = []
    focus_radius = 0.3  # Radius around each data point for focused sampling
    focus_resolution = 10  # Points per dimension in focused region

    for point in data_points:
        # Create small grid around this point
        x_focus = torch.linspace(
            max(point[0] - focus_radius, bounds[0][0]), min(point[0] + focus_radius, bounds[0][1]), focus_resolution
        )
        y_focus = torch.linspace(
            max(point[1] - focus_radius, bounds[1][0]), min(point[1] + focus_radius, bounds[1][1]), focus_resolution
        )
        xx_focus, yy_focus = torch.meshgrid(x_focus, y_focus, indexing="ij")
        focus_grid = torch.stack([xx_focus.flatten(), yy_focus.flatten()], dim=1)
        focused_points.append(focus_grid)

    # Combine base grid with focused grids
    all_focused = torch.cat(focused_points, dim=0)
    combined_grid = torch.cat([base_grid, all_focused], dim=0)

    # Remove duplicates (approximately)
    unique_grid = remove_duplicate_points(combined_grid, tolerance=1e-3)

    return unique_grid


def create_high_dim_sample_grid(bounds: List[Tuple[float, float]], num_samples: int = 1000) -> torch.Tensor:
    """
    Create sample points for high-dimensional analysis.

    Args:
        bounds: Analysis bounds for each dimension
        num_samples: Number of sample points to generate

    Returns:
        Sample points for high-dimensional analysis
    """
    n_dims = len(bounds)
    samples = torch.zeros(num_samples, n_dims)

    for i, (min_val, max_val) in enumerate(bounds):
        samples[:, i] = torch.linspace(min_val, max_val, num_samples)

    # Add some random samples for better coverage
    random_samples = torch.zeros(num_samples // 2, n_dims)
    for i, (min_val, max_val) in enumerate(bounds):
        random_samples[:, i] = torch.rand(num_samples // 2) * (max_val - min_val) + min_val

    return torch.cat([samples, random_samples], dim=0)


def remove_duplicate_points(points: torch.Tensor, tolerance: float = 1e-3) -> torch.Tensor:
    """
    Remove approximately duplicate points from tensor.

    Args:
        points: Tensor of points
        tolerance: Distance tolerance for considering points duplicate

    Returns:
        Tensor with duplicate points removed
    """
    if len(points) == 0:
        return points

    unique_points = [points[0]]

    for point in points[1:]:
        # Check if this point is close to any existing unique point
        distances = torch.norm(torch.stack(unique_points) - point.unsqueeze(0), dim=1)
        if distances.min() > tolerance:
            unique_points.append(point)

    return torch.stack(unique_points)


##################################################################
# Hyperplane Distances
##################################################################

def analyze_hyperplane_distances_with_hooks(
    run_results: List[Dict[str, Any]], 
    experiment_data: Dict[str, Any], 
    config,
    eps: float = 0.1, 
    min_samples: int = 2
) -> Dict[str, Any]:
    """
    Proper hyperplane analysis using input hooks to capture actual layer inputs.
    """
    x_train = experiment_data["x_train"]
    y_train = experiment_data["y_train"]
    
    # Only analyze successful runs
    accuracy_threshold = getattr(config.analysis, 'accuracy_threshold', 1.00)
    successful_runs = [r for r in run_results if r.get("accuracy", 0) >= accuracy_threshold]
    
    if not successful_runs:
        print(f"⚠️  WARNING: No successful runs to analyze (accuracy_threshold = {accuracy_threshold}).")
        return {}
    
    # Collect distance data per layer
    layer_results = {}
    
    for run_result in successful_runs:
        # Load model
        model = config.model
        model.load_state_dict(run_result["model_state_dict"])
        model.eval()
        
        run_id = run_result["run_id"]
        
        # Hook linear layers to capture inputs
        layer_inputs = {}
        hooks = []
        
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # input is a tuple, we want the first element
                layer_inputs[layer_name] = input[0].detach().clone()
            return hook_fn
        
        # Register hooks on all linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Forward pass to capture inputs and get predictions
        with torch.no_grad():
            predictions = model(x_train)
            predicted_classes = torch.argmax(predictions, dim=1) if predictions.dim() > 1 else (predictions > 0.5).long()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Now analyze each layer with its actual inputs
        for layer_name, layer_module in model.named_modules():
            if not isinstance(layer_module, nn.Linear):
                continue
                
            if layer_name not in layer_inputs:
                print(f"Warning: No input captured for layer {layer_name}")
                continue
            
            actual_inputs = layer_inputs[layer_name]  # [batch_size, input_dim]
            
            # Initialize layer results
            if layer_name not in layer_results:
                layer_results[layer_name] = {
                    "distance_vectors": [],
                    "hyperplane_metadata": []
                }
            
            # Get layer parameters
            weight = layer_module.weight  # [n_units, input_dim]
            bias = layer_module.bias      # [n_units]
            
            # Compute distance vectors for each hyperplane
            for unit_idx in range(weight.shape[0]):
                w = weight[unit_idx]  # [input_dim]
                b = bias[unit_idx]    # scalar
                
                distance_vector, metadata = compute_hyperplane_distances_proper(
                    w, b, actual_inputs, predicted_classes, 
                    run_id, layer_name, unit_idx
                )
                
                if distance_vector is not None:
                    layer_results[layer_name]["distance_vectors"].append(distance_vector)
                    layer_results[layer_name]["hyperplane_metadata"].append(metadata)
    
    # Cluster distance vectors for each layer
    clustering_results = {}
    
    for layer_name, layer_data in layer_results.items():
        distance_vectors = np.array(layer_data["distance_vectors"])
        metadata = layer_data["hyperplane_metadata"]
        
        if len(distance_vectors) == 0:
            clustering_results[layer_name] = {"error": "No valid hyperplanes found"}
            continue
        
        # Perform DBSCAN clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(distance_vectors)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int(np.sum(labels == -1))
        
        # Format results
        clustering_results[layer_name] = format_distance_clustering_results(
            layer_name, distance_vectors, metadata, labels, 
            n_clusters, noise_count, eps, min_samples
        )

    return clustering_results


def compute_hyperplane_distances_proper(
    weight: torch.Tensor,
    bias: torch.Tensor, 
    layer_inputs: torch.Tensor,
    predicted_classes: torch.Tensor,
    run_id: int,
    layer_name: str,
    unit_idx: int
) -> Tuple[np.ndarray, Dict]:
    """
    Compute L2 distances from actual layer inputs to hyperplane, grouped by predicted class.
    
    Args:
        weight: Hyperplane weight vector [input_dim]
        bias: Hyperplane bias scalar
        layer_inputs: Actual inputs to this layer [batch_size, input_dim] 
        predicted_classes: Model predictions [batch_size]
        
    Returns:
        distance_vector: [mean_dist_class0, mean_dist_class1] or None
        metadata: Information about this hyperplane
    """
    # Skip near-zero weights
    weight_norm = torch.norm(weight)
    if weight_norm < 1e-8:
        return None, None
    
    # Compute L2 distances to hyperplane: |w^T x + b| / ||w||
    # This is the perpendicular distance from each point to the hyperplane
    distances = torch.abs(layer_inputs @ weight + bias) / weight_norm  # [batch_size]
    
    # Group by predicted class
    class_distances = {}
    for class_idx in torch.unique(predicted_classes):
        mask = predicted_classes == class_idx
        if mask.sum() > 0:
            class_distances[int(class_idx)] = distances[mask]
    
    # Ensure we have both classes
    if 0 not in class_distances or 1 not in class_distances:
        return None, None
    
    # Compute mean distances per predicted class
    mean_dist_class0 = class_distances[0].mean().item()
    mean_dist_class1 = class_distances[1].mean().item()
    
    # Create distance vector
    distance_vector = np.array([mean_dist_class0, mean_dist_class1])
    
    # Metadata
    metadata = {
        "run_id": run_id,
        "layer_name": layer_name,
        "unit_idx": unit_idx,
        "weight_norm": weight_norm.item(),
        "weight": weight.detach().cpu().numpy(),
        "bias": bias.item(),
        "mean_dist_class0": mean_dist_class0,
        "mean_dist_class1": mean_dist_class1,
        "separation_ratio": mean_dist_class1 / (mean_dist_class0 + 1e-12),
        "input_dim": layer_inputs.shape[1],
        "n_samples_class0": class_distances[0].numel(),
        "n_samples_class1": class_distances[1].numel()
    }
    
    return distance_vector, metadata


def format_distance_clustering_results(
    layer_name: str,
    distance_vectors: np.ndarray,
    metadata: List[Dict],
    labels: np.ndarray,
    n_clusters: int,
    noise_count: int,
    eps: float,
    min_samples: int
) -> Dict[str, Any]:
    """Format clustering results for distance vectors."""
    
    result = {
        "layer_name": layer_name,
        "clustering_params": {"eps": eps, "min_samples": min_samples},
        "analysis_method": "actual_layer_inputs_L2_distance",
        "distance_analysis": {
            "total_hyperplanes": len(distance_vectors),
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
        cluster_vectors = distance_vectors[mask]
        cluster_metadata = [metadata[i] for i in range(len(metadata)) if mask[i]]
        
        cluster_data = {
            "cluster_id": int(cluster_id),
            "size": int(mask.sum()),
            "centroid": cluster_vectors.mean(axis=0).tolist(),
            "std": cluster_vectors.std(axis=0).tolist(),
            "distance_range": {
                "class0_dist": [cluster_vectors[:, 0].min(), cluster_vectors[:, 0].max()],
                "class1_dist": [cluster_vectors[:, 1].min(), cluster_vectors[:, 1].max()]
            },
            "hyperplanes": []
        }
        
        # Add hyperplane details
        for hp_meta in cluster_metadata:
            cluster_data["hyperplanes"].append({
                "run_id": hp_meta["run_id"],
                "unit_idx": hp_meta["unit_idx"],
                "weight_norm": hp_meta["weight_norm"],
                "separation_ratio": hp_meta["separation_ratio"],
                "input_dim": hp_meta["input_dim"],
                "n_samples_class0": hp_meta["n_samples_class0"],
                "n_samples_class1": hp_meta["n_samples_class1"]
            })
        
        # Sort hyperplanes by run_id for consistency
        cluster_data["hyperplanes"].sort(key=lambda x: (x["run_id"], x["unit_idx"]))
        
        result["distance_analysis"]["clusters"].append(cluster_data)
    
    # Sort clusters by size (largest first)
    result["distance_analysis"]["clusters"].sort(key=lambda x: x["size"], reverse=True)
    
    return result


# Integration function to replace the old analysis
def analyze_hyperplane_metric_clustering(
    run_results: List[Dict[str, Any]], 
    experiment_data: Dict[str, Any], 
    config,
    eps: float = 0.1, 
    min_samples: int = 2
) -> Dict[str, Any]:
    """
    Wrapper function that uses the proper hook-based analysis.
    This replaces the old analyze_hyperplane_metric_clustering function.
    """
    return analyze_hyperplane_distances_with_hooks(
        run_results, experiment_data, config, eps, min_samples
    )

##################################################################
