# analyze.py - Post-Experiment Analysis and Visualization for Prototype Surface Experiments

"""
Comprehensive analysis module for prototype surface experiments.
Provides geometric analysis, visualization, and prototype surface validation tools.
Focuses on prototype surface investigation, distance field analysis, and
comparative studies across activation functions and training runs.
"""

import numpy as np
import plotly.graph_objects as go
import torch
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
from configs import ExperimentConfig, ExperimentType, get_experiment_config, list_experiments
import matplotlib.pyplot as plt
import json


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
    if config.analysis.prototype_surface_validation:
        analysis_plan.append('prototype_validation')
        
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
        'convergence_threshold': getattr(config.training, 'convergence_threshold', 1e-6),
        
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

def extract_activation_type(model: torch.nn.Module) -> str:
    """
    Extract the primary activation type from a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        String identifying the activation type ('relu', 'abs', 'sigmoid', etc.)
    """
    model_name = type(model).__name__.lower()
    
    if 'abs' in model_name:
        return 'abs'
    elif 'relu' in model_name:
        return 'relu'
    elif 'sigmoid' in model_name:
        return 'sigmoid'
    elif 'tanh' in model_name:
        return 'tanh'
    else:
        # Try to inspect the model for activation functions
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                return 'relu'
            elif isinstance(module, torch.nn.Sigmoid):
                return 'sigmoid'
            elif isinstance(module, torch.nn.Tanh):
                return 'tanh'
            # Check for custom activations
            elif hasattr(module, 'forward') and 'abs' in str(module.forward).lower():
                return 'abs'
    
    return 'unknown'

def load_experiment_data(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Load experiment data and context from configuration.
    
    Args:
        config: Complete experiment configuration
        
    Returns:
        Dictionary containing experiment data and analysis context
    """
    # Get the original training data from config
    x_train = config.data.x.clone()  # Clone to avoid modifying original
    y_train = config.data.y.clone()
    
    # Determine problem-specific analysis bounds and parameters
    if config.data.problem_type == ExperimentType.XOR:
        # XOR-specific settings
        analysis_bounds = [(-2.5, 2.5), (-2.5, 2.5)]  # Slightly larger than data bounds
        expected_accuracy_levels = [0.0, 0.25, 0.5, 0.75, 1.0]  # Only possible XOR accuracies
        problem_dimensionality = 2
        is_boolean_problem = True
        class_names = ['False', 'True']
        
        # XOR corner points for reference
        corner_points = x_train.clone()
        
    elif config.data.problem_type == ExperimentType.PARITY:
        # Parity problem settings
        n_bits = x_train.shape[1]
        analysis_bounds = [(-1.5, 1.5)] * n_bits  # Bounds for each dimension
        expected_accuracy_levels = [i / (2**n_bits) for i in range(2**n_bits + 1)]
        problem_dimensionality = n_bits
        is_boolean_problem = True
        class_names = ['Even', 'Odd']
        
        # All boolean hypercube corners
        corner_points = x_train.clone()
        
    else:
        # Generic problem settings
        # Infer bounds from data with some padding
        data_min = x_train.min(dim=0)[0]
        data_max = x_train.max(dim=0)[0]
        padding = (data_max - data_min) * 0.2  # 20% padding
        analysis_bounds = [(min_val.item() - pad.item(), max_val.item() + pad.item()) 
                          for min_val, max_val, pad in zip(data_min, data_max, padding)]
        
        # Determine expected accuracy levels based on data
        unique_labels = torch.unique(y_train)
        num_classes = len(unique_labels)
        expected_accuracy_levels = [i / len(x_train) for i in range(len(x_train) + 1)]
        
        problem_dimensionality = x_train.shape[1]
        is_boolean_problem = set(y_train.tolist()).issubset({0.0, 1.0})
        class_names = [f'Class_{int(label)}' for label in unique_labels]
        corner_points = None
    
    # Generate analysis grids for visualization
    if problem_dimensionality == 2:
        # Create dense grid for 2D visualization
        resolution = config.analysis.analysis_resolution
        x_range = torch.linspace(analysis_bounds[0][0], analysis_bounds[0][1], resolution)
        y_range = torch.linspace(analysis_bounds[1][0], analysis_bounds[1][1], resolution)
        xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
        visualization_grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Create boundary sampling grid (higher resolution near data points)
        boundary_grid = create_boundary_focused_grid(x_train, analysis_bounds, resolution=resolution//2)
        
    else:
        # For higher dimensions, create sample points for analysis
        visualization_grid = create_high_dim_sample_grid(analysis_bounds, num_samples=1000)
        boundary_grid = None
    
    # Compute data statistics
    data_statistics = compute_data_statistics(x_train, y_train)
    
    # Create analysis-specific data structures
    analysis_data = {
        # Original training data
        'x_train': x_train,
        'y_train': y_train,
        
        # Problem characterization
        'problem_type': config.data.problem_type,
        'problem_dimensionality': problem_dimensionality,
        'is_boolean_problem': is_boolean_problem,
        'num_classes': len(torch.unique(y_train)),
        'num_training_samples': len(x_train),
        
        # Analysis bounds and grids
        'analysis_bounds': analysis_bounds,
        'visualization_grid': visualization_grid,
        'boundary_grid': boundary_grid,
        
        # Expected behavior
        'expected_accuracy_levels': expected_accuracy_levels,
        'class_names': class_names,
        
        # Reference points
        'corner_points': corner_points,
        'data_centroid': x_train.mean(dim=0),
        'class_centroids': compute_class_centroids(x_train, y_train),
        
        # Statistical properties
        'data_statistics': data_statistics,
        
        # Model context
        'model_type': type(config.model).__name__,
        'activation_type': extract_activation_type(config.model),
        'model_parameters': count_model_parameters(config.model),
        
        # Training context
        'training_epochs': config.training.epochs,
        'num_runs': config.execution.num_runs,
        'convergence_threshold': getattr(config.training, 'convergence_threshold', 1e-6),
        
        # Experiment metadata
        'experiment_name': config.execution.experiment_name,
        'description': config.description,
        'config_hash': generate_config_hash(config)
    }
    
    return analysis_data

def create_boundary_focused_grid(data_points: torch.Tensor, bounds: List[Tuple[float, float]], 
                                resolution: int = 50) -> torch.Tensor:
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
    xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
    base_grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Add additional points near each data point
    focused_points = []
    focus_radius = 0.3  # Radius around each data point for focused sampling
    focus_resolution = 10  # Points per dimension in focused region
    
    for point in data_points:
        # Create small grid around this point
        x_focus = torch.linspace(
            max(point[0] - focus_radius, bounds[0][0]),
            min(point[0] + focus_radius, bounds[0][1]),
            focus_resolution
        )
        y_focus = torch.linspace(
            max(point[1] - focus_radius, bounds[1][0]),
            min(point[1] + focus_radius, bounds[1][1]),
            focus_resolution
        )
        xx_focus, yy_focus = torch.meshgrid(x_focus, y_focus, indexing='ij')
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

def compute_data_statistics(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the dataset.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary of data statistics
    """
    return {
        # Basic statistics
        'num_samples': len(x),
        'num_features': x.shape[1],
        'num_classes': len(torch.unique(y)),
        
        # Input statistics
        'input_mean': x.mean(dim=0),
        'input_std': x.std(dim=0),
        'input_min': x.min(dim=0)[0],
        'input_max': x.max(dim=0)[0],
        'input_range': x.max(dim=0)[0] - x.min(dim=0)[0],
        
        # Label statistics
        'label_distribution': torch.bincount(y.long()) / len(y),
        'is_balanced': torch.std(torch.bincount(y.long()).float()) < 0.1,
        
        # Geometric properties
        'data_diameter': compute_data_diameter(x),
        'nearest_neighbor_distances': compute_nearest_neighbor_distances(x),
        'class_separation': compute_class_separation_distance(x, y)
    }

def compute_class_centroids(x: torch.Tensor, y: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Compute centroid for each class.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary mapping class labels to centroid coordinates
    """
    centroids = {}
    unique_labels = torch.unique(y)
    
    for label in unique_labels:
        mask = (y == label)
        centroids[int(label.item())] = x[mask].mean(dim=0)
    
    return centroids

def compute_data_diameter(x: torch.Tensor) -> float:
    """
    Compute the diameter (maximum pairwise distance) of the dataset.
    
    Args:
        x: Input data tensor
        
    Returns:
        Maximum pairwise Euclidean distance
    """
    distances = torch.cdist(x, x, p=2)
    return distances.max().item()

def compute_nearest_neighbor_distances(x: torch.Tensor) -> torch.Tensor:
    """
    Compute nearest neighbor distances for each point.
    
    Args:
        x: Input data tensor
        
    Returns:
        Tensor of nearest neighbor distances
    """
    distances = torch.cdist(x, x, p=2)
    # Set diagonal to infinity to exclude self-distances
    distances.fill_diagonal_(float('inf'))
    return distances.min(dim=1)[0]

def compute_class_separation_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute minimum distance between points of different classes.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Minimum inter-class distance
    """
    min_distance = float('inf')
    unique_labels = torch.unique(y)
    
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            mask1 = (y == label1)
            mask2 = (y == label2)
            
            x1 = x[mask1]
            x2 = x[mask2]
            
            # Compute all pairwise distances between classes
            distances = torch.cdist(x1, x2, p=2)
            min_distance = min(min_distance, distances.min().item())
    
    return min_distance

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


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def generate_config_hash(config: ExperimentConfig) -> str:
    """
    Generate unique hash for configuration identification.
    
    Args:
        config: Configuration to hash
        
    Returns:
        Unique hash string
    """
    import hashlib
    import json
    
    # Create a simplified config dict for hashing (excluding objects)
    hash_dict = {
        'model_type': type(config.model).__name__,
        'epochs': config.training.epochs,
        'problem_type': config.data.problem_type.value if config.data.problem_type else None,
        'num_runs': config.execution.num_runs,
        'description': config.description
    }
    
    config_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def load_all_run_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Load results from all training runs in an experiment.
    
    Args:
        results_dir: Path to experiment results directory
        
    Returns:
        List of dictionaries containing results from each run
    """
    runs_dir = results_dir / "runs"
    
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Get all run directories (should be numbered: 000, 001, 002, etc.)
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                      key=lambda d: int(d.name))
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {runs_dir}")
    
    all_results = []
    failed_runs = []
    
    print(f"Loading results from {len(run_dirs)} runs...")
    
    for run_dir in run_dirs:
        run_id = int(run_dir.name)
        
        try:
            run_result = load_single_run_result(run_dir, run_id)
            all_results.append(run_result)
            
        except Exception as e:
            failed_runs.append((run_id, str(e)))
            print(f"⚠️  Failed to load run {run_id}: {e}")
    
    if failed_runs:
        print(f"⚠️  Failed to load {len(failed_runs)} out of {len(run_dirs)} runs")
        for run_id, error in failed_runs:
            print(f"   Run {run_id}: {error}")
    
    if not all_results:
        raise ValueError("No valid run results could be loaded")
    
    print(f"✓ Successfully loaded {len(all_results)} run results")
    
    # Sort results by run_id for consistency
    all_results.sort(key=lambda x: x['run_id'])
    
    return all_results


def load_single_run_result(run_dir: Path, run_id: int) -> Dict[str, Any]:
    """
    Load results from a single training run.
    
    Args:
        run_dir: Path to individual run directory
        run_id: Run identifier
        
    Returns:
        Dictionary containing complete run results
    """
    result = {
        'run_id': run_id,
        'run_dir': run_dir
    }
    
    # Load model state dict (required)
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model_state_dict = torch.load(model_path, map_location='cpu')
        result['model_state_dict'] = model_state_dict
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    # Load training statistics (required)
    stats_path = run_dir / "stats.pt"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    try:
        training_stats = torch.load(stats_path, map_location='cpu')
        result.update(training_stats)  # Merge stats into result
    except Exception as e:
        raise RuntimeError(f"Failed to load stats from {stats_path}: {e}")
    
    # Load config summary (optional but recommended)
    config_path = run_dir / "config_summary.pt"
    if config_path.exists():
        try:
            config_summary = torch.load(config_path, map_location='cpu')
            result['config_summary'] = config_summary
        except Exception as e:
            print(f"⚠️  Failed to load config summary for run {run_id}: {e}")
    
    # Load any checkpoints (optional)
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        result['checkpoints'] = load_run_checkpoints(checkpoints_dir)
    
    # Load additional analysis files if they exist
    additional_files = {
        'training_log.json': 'training_log',
        'hyperparameters.json': 'hyperparameters',
        'model_info.json': 'model_info'
    }
    
    for filename, key in additional_files.items():
        file_path = run_dir / filename
        if file_path.exists():
            try:
                if filename.endswith('.json'):
                    import json
                    with open(file_path, 'r') as f:
                        result[key] = json.load(f)
                elif filename.endswith('.pt'):
                    result[key] = torch.load(file_path, map_location='cpu')
            except Exception as e:
                print(f"⚠️  Failed to load {filename} for run {run_id}: {e}")
    
    # Validate required fields and compute derived metrics
    result = validate_and_enhance_run_result(result)
    
    return result


def load_run_checkpoints(checkpoints_dir: Path) -> List[Dict[str, Any]]:
    """
    Load training checkpoints from a run.
    
    Args:
        checkpoints_dir: Directory containing checkpoint files
        
    Returns:
        List of checkpoint dictionaries sorted by epoch
    """
    checkpoint_files = sorted([f for f in checkpoints_dir.iterdir() 
                              if f.is_file() and f.name.endswith('.pt')],
                             key=lambda f: extract_epoch_from_filename(f.name))
    
    checkpoints = []
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            checkpoint['checkpoint_file'] = checkpoint_file
            checkpoints.append(checkpoint)
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint {checkpoint_file}: {e}")
    
    return checkpoints


def extract_epoch_from_filename(filename: str) -> int:
    """
    Extract epoch number from checkpoint filename.
    
    Args:
        filename: Checkpoint filename (e.g., "checkpoint_epoch_100.pt")
        
    Returns:
        Epoch number
    """
    import re
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        # Try to extract any number from filename
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0


def validate_and_enhance_run_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate run result and add derived metrics.
    
    Args:
        result: Raw run result dictionary
        
    Returns:
        Enhanced run result with validation and derived metrics
    """
    # Ensure required fields exist
    required_fields = ['run_id', 'model_state_dict']
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Required field '{field}' missing from run result")
    
    # Ensure we have basic training metrics
    if 'final_loss' not in result:
        if 'loss_history' in result and result['loss_history']:
            result['final_loss'] = result['loss_history'][-1]
        else:
            result['final_loss'] = float('inf')
    
    if 'accuracy' not in result:
        result['accuracy'] = 0.0
    
    if 'training_time' not in result:
        result['training_time'] = 0.0
    
    # Add derived metrics
    result['converged'] = result['final_loss'] < 0.01  # Default convergence threshold
    result['perfect_accuracy'] = result['accuracy'] >= 0.99
    
    # Compute training efficiency metrics
    if 'loss_history' in result and result['loss_history']:
        result['training_efficiency'] = compute_training_efficiency(result['loss_history'])
    
    # Add model analysis metrics
    if 'model_state_dict' in result:
        result['model_analysis'] = analyze_model_state_dict(result['model_state_dict'])
    
    # Add timestamp if available
    if 'run_dir' in result:
        run_dir = result['run_dir']
        if run_dir.exists():
            # Use directory modification time as proxy for completion time
            result['completion_timestamp'] = run_dir.stat().st_mtime
    
    return result


def compute_training_efficiency(loss_history: List[float]) -> Dict[str, float]:
    """
    Compute training efficiency metrics from loss history.
    
    Args:
        loss_history: List of loss values over training
        
    Returns:
        Dictionary of efficiency metrics
    """
    if not loss_history or len(loss_history) < 2:
        return {'efficiency_score': 0.0, 'convergence_rate': 0.0}
    
    loss_tensor = torch.tensor(loss_history)
    
    # Compute rate of loss decrease
    loss_decrease = loss_tensor[0] - loss_tensor[-1]
    epochs = len(loss_history)
    convergence_rate = loss_decrease / epochs if epochs > 0 else 0.0
    
    # Compute efficiency score (how quickly loss decreased)
    # Find epoch where loss reached 90% of final improvement
    target_loss = loss_tensor[0] - 0.9 * loss_decrease
    convergence_epoch = epochs  # default to full training
    
    for i, loss in enumerate(loss_history):
        if loss <= target_loss:
            convergence_epoch = i + 1
            break
    
    efficiency_score = (epochs - convergence_epoch) / epochs if epochs > 0 else 0.0
    
    # Compute loss smoothness (measure of training stability)
    if len(loss_history) > 1:
        loss_diffs = torch.diff(loss_tensor)
        loss_volatility = torch.std(loss_diffs).item()
    else:
        loss_volatility = 0.0
    
    return {
        'convergence_rate': convergence_rate,
        'efficiency_score': efficiency_score,
        'loss_volatility': loss_volatility,
        'convergence_epoch': convergence_epoch,
        'total_epochs': epochs
    }


def analyze_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Analyze model state dictionary for key properties.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary of model analysis results
    """
    analysis = {
        'parameter_count': 0,
        'layer_count': 0,
        'weight_statistics': {},
        'bias_statistics': {}
    }
    
    weight_tensors = []
    bias_tensors = []
    
    for name, tensor in state_dict.items():
        analysis['parameter_count'] += tensor.numel()
        
        if 'weight' in name:
            analysis['layer_count'] += 1
            weight_tensors.append(tensor)
        elif 'bias' in name:
            bias_tensors.append(tensor)
    
    # Analyze weights
    if weight_tensors:
        all_weights = torch.cat([w.flatten() for w in weight_tensors])
        analysis['weight_statistics'] = {
            'mean': all_weights.mean().item(),
            'std': all_weights.std().item(),
            'min': all_weights.min().item(),
            'max': all_weights.max().item(),
            'norm': torch.norm(all_weights).item()
        }
    
    # Analyze biases
    if bias_tensors:
        all_biases = torch.cat([b.flatten() for b in bias_tensors])
        analysis['bias_statistics'] = {
            'mean': all_biases.mean().item(),
            'std': all_biases.std().item(),
            'min': all_biases.min().item(),
            'max': all_biases.max().item(),
            'norm': torch.norm(all_biases).item()
        }
    
    # Detect potential issues
    analysis['potential_issues'] = detect_model_issues(state_dict, analysis)
    
    return analysis


def detect_model_issues(state_dict: Dict[str, torch.Tensor], analysis: Dict[str, Any]) -> List[str]:
    """
    Detect potential issues in trained model.
    
    Args:
        state_dict: Model state dictionary
        analysis: Model analysis results
        
    Returns:
        List of detected issues
    """
    issues = []
    
    # Check for exploding gradients
    if 'weight_statistics' in analysis and analysis['weight_statistics']:
        weight_max = abs(analysis['weight_statistics']['max'])
        weight_norm = analysis['weight_statistics']['norm']
        
        if weight_max > 10.0:
            issues.append(f"Large weight values detected (max: {weight_max:.2f})")
        
        if weight_norm > 100.0:
            issues.append(f"Large weight norm detected ({weight_norm:.2f})")
    
    # Check for vanishing gradients
    if 'weight_statistics' in analysis and analysis['weight_statistics']:
        weight_std = analysis['weight_statistics']['std']
        if weight_std < 1e-6:
            issues.append(f"Very small weight variations (std: {weight_std:.2e})")
    
    # Check for dead neurons (all weights near zero)
    for name, tensor in state_dict.items():
        if 'weight' in name:
            if torch.norm(tensor) < 1e-6:
                issues.append(f"Potential dead neuron in {name}")
    
    return issues


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

def compute_basic_statistics(run_results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Compute comprehensive basic statistics across all runs.
    
    Args:
        run_results: List of results from all training runs
        config: Experiment configuration for context
        
    Returns:
        Dictionary containing basic statistics and metrics
    """
    if not run_results:
        return {'error': 'No run results provided'}
    
    print(f"  Computing statistics for {len(run_results)} runs...")
    
    # Extract key metrics from all runs
    metrics = extract_run_metrics(run_results)
    
    # Compute summary statistics
    summary_stats = compute_summary_statistics(metrics)
    
    # Compute distribution statistics
    distribution_stats = compute_distribution_statistics(metrics, config)
    
    # Compute success/failure statistics
    success_stats = compute_success_statistics(run_results, config)
    
    # Compute training dynamics statistics
    training_stats = compute_training_dynamics_statistics(run_results)
    
    # Compute model statistics
    model_stats = compute_model_statistics(run_results)
    
    # Create comprehensive statistics dictionary
    basic_stats = {
        'experiment_info': {
            'total_runs': len(run_results),
            'experiment_name': config.execution.experiment_name,
            'description': config.description,
            'model_type': type(config.model).__name__,
            'problem_type': config.data.problem_type.value if config.data.problem_type else 'unknown',
            'training_epochs': config.training.epochs
        },
        
        'summary': summary_stats,
        'distributions': distribution_stats,
        'success_metrics': success_stats,
        'training_dynamics': training_stats,
        'model_properties': model_stats,
        
        # Raw metrics for further analysis
        'raw_metrics': metrics
    }
    
    return basic_stats


def extract_run_metrics(run_results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extract numerical metrics from all run results.
    
    Args:
        run_results: List of run result dictionaries
        
    Returns:
        Dictionary mapping metric names to lists of values
    """
    metrics = {
        'final_losses': [],
        'best_losses': [],
        'accuracies': [],
        'training_times': [],
        'convergence_epochs': [],
        'parameter_counts': [],
        'weight_norms': [],
        'bias_norms': []
    }
    
    for result in run_results:
        # Loss metrics
        metrics['final_losses'].append(result.get('final_loss', float('inf')))
        metrics['best_losses'].append(result.get('best_loss', result.get('final_loss', float('inf'))))
        
        # Performance metrics
        metrics['accuracies'].append(result.get('accuracy', 0.0))
        
        # Training metrics
        metrics['training_times'].append(result.get('training_time', 0.0))
        
        # Convergence metrics
        if 'training_efficiency' in result:
            metrics['convergence_epochs'].append(
                result['training_efficiency'].get('convergence_epoch', 0)
            )
        else:
            metrics['convergence_epochs'].append(0)
        
        # Model metrics
        if 'model_analysis' in result:
            model_analysis = result['model_analysis']
            metrics['parameter_counts'].append(model_analysis.get('parameter_count', 0))
            
            if 'weight_statistics' in model_analysis:
                metrics['weight_norms'].append(
                    model_analysis['weight_statistics'].get('norm', 0.0)
                )
            else:
                metrics['weight_norms'].append(0.0)
                
            if 'bias_statistics' in model_analysis:
                metrics['bias_norms'].append(
                    model_analysis['bias_statistics'].get('norm', 0.0)
                )
            else:
                metrics['bias_norms'].append(0.0)
        else:
            metrics['parameter_counts'].append(0)
            metrics['weight_norms'].append(0.0)
            metrics['bias_norms'].append(0.0)
    
    # Remove invalid values and convert to tensors for computation
    for key, values in metrics.items():
        # Filter out inf and nan values
        valid_values = [v for v in values if not (np.isinf(v) or np.isnan(v))]
        metrics[key] = valid_values
    
    return metrics


def compute_summary_statistics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics (mean, std, min, max, etc.) for each metric.
    
    Args:
        metrics: Dictionary of metric lists
        
    Returns:
        Dictionary of summary statistics for each metric
    """
    summary = {}
    
    for metric_name, values in metrics.items():
        if not values:
            summary[metric_name] = {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
            continue
        
        values_tensor = torch.tensor(values, dtype=torch.float32)
        
        summary[metric_name] = {
            'count': len(values),
            'mean': values_tensor.mean().item(),
            'std': values_tensor.std().item() if len(values) > 1 else 0.0,
            'min': values_tensor.min().item(),
            'max': values_tensor.max().item(),
            'median': values_tensor.median().item(),
            'q25': torch.quantile(values_tensor, 0.25).item() if len(values) > 3 else values_tensor.min().item(),
            'q75': torch.quantile(values_tensor, 0.75).item() if len(values) > 3 else values_tensor.max().item()
        }
        
        # Add coefficient of variation if mean is not zero
        if summary[metric_name]['mean'] != 0:
            summary[metric_name]['cv'] = summary[metric_name]['std'] / abs(summary[metric_name]['mean'])
        else:
            summary[metric_name]['cv'] = 0.0
    
    return summary


def compute_distribution_statistics(metrics: Dict[str, List[float]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Compute distribution-specific statistics, especially for discrete outcomes.
    
    Args:
        metrics: Dictionary of metric lists
        config: Experiment configuration for context
        
    Returns:
        Dictionary of distribution statistics
    """
    distributions = {}
    
    # Accuracy distribution (especially important for XOR)
    if metrics['accuracies']:
        accuracies = metrics['accuracies']
        
        if config.data.problem_type == ExperimentType.XOR:
            # XOR has discrete accuracy levels: 0%, 25%, 50%, 75%, 100%
            acc_bins = {0.0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
            
            for acc in accuracies:
                # Round to nearest XOR accuracy level
                if acc <= 0.125:
                    acc_bins[0.0] += 1
                elif acc <= 0.375:
                    acc_bins[0.25] += 1
                elif acc <= 0.625:
                    acc_bins[0.5] += 1
                elif acc <= 0.875:
                    acc_bins[0.75] += 1
                else:
                    acc_bins[1.0] += 1
            
            distributions['accuracy_distribution'] = {
                'type': 'discrete_xor',
                'bins': acc_bins,
                'perfect_rate': acc_bins[1.0] / len(accuracies),
                'failure_rate': acc_bins[0.0] / len(accuracies),
                'partial_success_rate': (acc_bins[0.25] + acc_bins[0.5] + acc_bins[0.75]) / len(accuracies)
            }
        else:
            # Generic continuous accuracy distribution
            acc_tensor = torch.tensor(accuracies)
            distributions['accuracy_distribution'] = {
                'type': 'continuous',
                'histogram': compute_histogram(acc_tensor, bins=10),
                'perfect_rate': sum(1 for acc in accuracies if acc >= 0.99) / len(accuracies),
                'high_performance_rate': sum(1 for acc in accuracies if acc >= 0.8) / len(accuracies)
            }
    
    # Loss distribution
    if metrics['final_losses']:
        loss_tensor = torch.tensor(metrics['final_losses'])
        distributions['loss_distribution'] = {
            'histogram': compute_histogram(loss_tensor, bins=20),
            'convergence_rate': sum(1 for loss in metrics['final_losses'] if loss < 0.01) / len(metrics['final_losses']),
            'low_loss_rate': sum(1 for loss in metrics['final_losses'] if loss < 0.1) / len(metrics['final_losses'])
        }
    
    # Training time distribution
    if metrics['training_times']:
        time_tensor = torch.tensor(metrics['training_times'])
        distributions['training_time_distribution'] = {
            'histogram': compute_histogram(time_tensor, bins=15),
            'fast_runs_rate': sum(1 for t in metrics['training_times'] if t < np.percentile(metrics['training_times'], 25)) / len(metrics['training_times'])
        }
    
    return distributions


def compute_success_statistics(run_results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """
    Compute success/failure statistics with problem-specific criteria.
    
    Args:
        run_results: List of run results
        config: Experiment configuration
        
    Returns:
        Dictionary of success metrics
    """
    total_runs = len(run_results)
    
    # Basic success criteria
    converged_runs = sum(1 for r in run_results if r.get('converged', False))
    perfect_accuracy_runs = sum(1 for r in run_results if r.get('accuracy', 0) >= 0.99)
    failed_runs = sum(1 for r in run_results if r.get('final_loss', float('inf')) == float('inf'))
    
    # Problem-specific success criteria
    if config.data.problem_type == ExperimentType.XOR:
        # XOR-specific success metrics
        perfect_xor_runs = sum(1 for r in run_results if r.get('accuracy', 0) >= 1.0)
        partial_success_runs = sum(1 for r in run_results if r.get('accuracy', 0) >= 0.75)
        complete_failure_runs = sum(1 for r in run_results if r.get('accuracy', 0) <= 0.25)
        
        success_metrics = {
            'perfect_solution_rate': perfect_xor_runs / total_runs,
            'high_success_rate': partial_success_runs / total_runs,
            'complete_failure_rate': complete_failure_runs / total_runs,
            'learning_success_rate': (total_runs - complete_failure_runs) / total_runs
        }
    else:
        # Generic success metrics
        high_accuracy_runs = sum(1 for r in run_results if r.get('accuracy', 0) >= 0.8)
        low_accuracy_runs = sum(1 for r in run_results if r.get('accuracy', 0) <= 0.6)
        
        success_metrics = {
            'high_accuracy_rate': high_accuracy_runs / total_runs,
            'low_accuracy_rate': low_accuracy_runs / total_runs,
            'learning_success_rate': (total_runs - low_accuracy_runs) / total_runs
        }
    
    # Common success metrics
    success_metrics.update({
        'total_runs': total_runs,
        'convergence_rate': converged_runs / total_runs,
        'perfect_accuracy_rate': perfect_accuracy_runs / total_runs,
        'failure_rate': failed_runs / total_runs,
        'success_rate': (total_runs - failed_runs) / total_runs,
        
        # Reliability metrics
        'consistency_score': compute_consistency_score(run_results),
        'reliability_score': compute_reliability_score(run_results)
    })
    
    return success_metrics


def compute_training_dynamics_statistics(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about training dynamics across runs.
    
    Args:
        run_results: List of run results
        
    Returns:
        Dictionary of training dynamics statistics
    """
    dynamics_stats = {}
    
    # Collect efficiency metrics
    efficiency_scores = []
    convergence_rates = []
    loss_volatilities = []
    
    for result in run_results:
        if 'training_efficiency' in result:
            eff = result['training_efficiency']
            efficiency_scores.append(eff.get('efficiency_score', 0.0))
            convergence_rates.append(eff.get('convergence_rate', 0.0))
            loss_volatilities.append(eff.get('loss_volatility', 0.0))
    
    if efficiency_scores:
        dynamics_stats['efficiency'] = {
            'mean_efficiency': np.mean(efficiency_scores),
            'mean_convergence_rate': np.mean(convergence_rates),
            'mean_volatility': np.mean(loss_volatilities),
            'stable_training_rate': sum(1 for v in loss_volatilities if v < 0.1) / len(loss_volatilities)
        }
    
    # Analyze loss curves if available
    loss_histories = [r.get('loss_history', []) for r in run_results if 'loss_history' in r]
    if loss_histories:
        dynamics_stats['loss_curves'] = analyze_loss_curve_patterns(loss_histories)
    
    return dynamics_stats


def compute_model_statistics(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about learned model properties.
    
    Args:
        run_results: List of run results
        
    Returns:
        Dictionary of model statistics
    """
    model_stats = {}
    
    # Collect model analysis data
    weight_norms = []
    bias_norms = []
    parameter_counts = []
    potential_issues = []
    
    for result in run_results:
        if 'model_analysis' in result:
            analysis = result['model_analysis']
            
            parameter_counts.append(analysis.get('parameter_count', 0))
            
            if 'weight_statistics' in analysis:
                weight_norms.append(analysis['weight_statistics'].get('norm', 0.0))
            
            if 'bias_statistics' in analysis:
                bias_norms.append(analysis['bias_statistics'].get('norm', 0.0))
            
            if 'potential_issues' in analysis:
                potential_issues.extend(analysis['potential_issues'])
    
    if weight_norms:
        model_stats['weight_properties'] = {
            'mean_weight_norm': np.mean(weight_norms),
            'std_weight_norm': np.std(weight_norms),
            'weight_norm_range': (min(weight_norms), max(weight_norms))
        }
    
    if bias_norms:
        model_stats['bias_properties'] = {
            'mean_bias_norm': np.mean(bias_norms),
            'std_bias_norm': np.std(bias_norms),
            'bias_norm_range': (min(bias_norms), max(bias_norms))
        }
    
    if parameter_counts:
        model_stats['parameter_properties'] = {
            'parameter_count': parameter_counts[0] if parameter_counts else 0,  # Should be same for all
            'parameter_consistency': len(set(parameter_counts)) == 1  # Check if all models have same param count
        }
    
    # Analyze common issues
    if potential_issues:
        issue_counts = {}
        for issue in potential_issues:
            issue_type = issue.split('(')[0].strip()  # Extract issue type
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        model_stats['common_issues'] = issue_counts
        model_stats['issue_rate'] = len([r for r in run_results if r.get('model_analysis', {}).get('potential_issues', [])]) / len(run_results)
    
    return model_stats


def compute_histogram(data: torch.Tensor, bins: int = 10) -> Dict[str, Any]:
    """
    Compute histogram of data for distribution analysis.
    
    Args:
        data: Data tensor
        bins: Number of histogram bins
        
    Returns:
        Dictionary containing histogram data
    """
    if len(data) == 0:
        return {'counts': [], 'bin_edges': [], 'bin_centers': []}
    
    counts, bin_edges = torch.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'counts': counts.tolist(),
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers.tolist(),
        'normalized_counts': (counts / counts.sum()).tolist()
    }


def compute_consistency_score(run_results: List[Dict[str, Any]]) -> float:
    """
    Compute consistency score based on variance in key metrics.
    
    Args:
        run_results: List of run results
        
    Returns:
        Consistency score (higher = more consistent)
    """
    if len(run_results) <= 1:
        return 1.0
    
    accuracies = [r.get('accuracy', 0) for r in run_results]
    losses = [r.get('final_loss', float('inf')) for r in run_results if r.get('final_loss', float('inf')) != float('inf')]
    
    consistency_scores = []
    
    if accuracies:
        acc_std = np.std(accuracies)
        acc_mean = np.mean(accuracies)
        if acc_mean > 0:
            consistency_scores.append(1.0 - min(acc_std / acc_mean, 1.0))
    
    if losses:
        loss_std = np.std(losses)
        loss_mean = np.mean(losses)
        if loss_mean > 0:
            consistency_scores.append(1.0 - min(loss_std / loss_mean, 1.0))
    
    return np.mean(consistency_scores) if consistency_scores else 0.0


def compute_reliability_score(run_results: List[Dict[str, Any]]) -> float:
    """
    Compute reliability score based on success rate and consistency.
    
    Args:
        run_results: List of run results
        
    Returns:
        Reliability score (higher = more reliable)
    """
    if not run_results:
        return 0.0
    
    # Success rate component
    successful_runs = sum(1 for r in run_results if r.get('final_loss', float('inf')) != float('inf'))
    success_rate = successful_runs / len(run_results)
    
    # Consistency component
    consistency = compute_consistency_score(run_results)
    
    # Combined reliability (weighted average)
    reliability = 0.7 * success_rate + 0.3 * consistency
    
    return reliability


def analyze_loss_curve_patterns(loss_histories: List[List[float]]) -> Dict[str, Any]:
    """
    Analyze patterns in loss curves across runs.
    
    Args:
        loss_histories: List of loss curves from different runs
        
    Returns:
        Dictionary of loss curve analysis results
    """
    if not loss_histories:
        return {}
    
    # Find common patterns
    smooth_curves = 0
    oscillating_curves = 0
    plateauing_curves = 0
    early_convergence = 0
    
    for loss_curve in loss_histories:
        if len(loss_curve) < 10:
            continue
        
        loss_tensor = torch.tensor(loss_curve)
        
        # Check for smoothness (low variance in derivatives)
        if len(loss_curve) > 1:
            derivatives = torch.diff(loss_tensor)
            derivative_variance = torch.var(derivatives).item()
            
            if derivative_variance < 0.01:
                smooth_curves += 1
            elif derivative_variance > 0.1:
                oscillating_curves += 1
        
        # Check for plateauing (little change in final 25% of training)
        final_quarter = loss_curve[-len(loss_curve)//4:]
        if len(final_quarter) > 1 and (max(final_quarter) - min(final_quarter)) < 0.01:
            plateauing_curves += 1
        
        # Check for early convergence (reaches low loss in first 50% of training)
        halfway_point = len(loss_curve) // 2
        if loss_curve[halfway_point] < 0.1:
            early_convergence += 1
    
    total_curves = len(loss_histories)
    
    return {
        'total_curves_analyzed': total_curves,
        'smooth_curve_rate': smooth_curves / total_curves,
        'oscillating_curve_rate': oscillating_curves / total_curves,
        'plateauing_curve_rate': plateauing_curves / total_curves,
        'early_convergence_rate': early_convergence / total_curves
    }

def validate_prototype_theory(run_results, experiment_data, config):
    # Test 1: Distance of points to hyperplanes
    distance_test = test_distance_to_hyperplanes(run_results, experiment_data)
    
    # Test 2: Mirror weight detection  
    mirror_test = test_mirror_weights(run_results)
    
    return {
        'distance_test': distance_test,
        'mirror_test': mirror_test
    }

def test_distance_to_hyperplanes(run_results, experiment_data):
   """
   Test if classification correlates with distance to learned hyperplanes.
   """
   x_train = experiment_data['x_train']
   y_train = experiment_data['y_train']
   
   results = []
   
   for result in run_results:
       if result.get('accuracy', 0) < 0.75:  # Only test successful models
           continue
           
       # Load model weights
       state_dict = result['model_state_dict']
       W = state_dict['linear1.weight']  # Assuming first layer is 'linear1'
       b = state_dict['linear1.bias']
       
       # Compute distances from each point to each hyperplane
       distances = []
       for i in range(W.shape[0]):  # For each neuron
           w = W[i]
           bias = b[i]
           
           # Distance = |Wx + b| / ||W||
           point_distances = torch.abs(w @ x_train.T + bias) / torch.norm(w)
           distances.append(point_distances)
       
       # Find minimum distance to any hyperplane for each point
       min_distances = torch.stack(distances).min(dim=0)[0]
       
       results.append({
           'run_id': result['run_id'],
           'distances': min_distances,
           'labels': y_train
       })
   
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

def plot_single_hyperplane_model(model, x, y, title, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import torch

    model.eval()

    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()

    W = model.linear1.weight.detach()[0].cpu()  # (2,)
    b = model.linear1.bias.detach()[0].cpu()    # scalar

    mean = torch.zeros(2)

    plt.figure(figsize=(6, 6))

    # XOR input points
    for xi, yi in zip(x_cpu, y_cpu):
        marker = 'o' if yi == 0 else '^'
        plt.scatter(xi[0], xi[1], marker=marker, s=100, color='black', edgecolors='k', linewidths=1)

    # Draw the hyperplane and normal
    norm = torch.norm(W)
    normal = W / norm
    distance = (W @ mean + b) / norm
    projection_on_plane = mean - distance * normal
    perp = torch.tensor([-normal[1], normal[0]])

    scale_factor = 5
    pt1 = projection_on_plane + perp * scale_factor
    pt2 = projection_on_plane - perp * scale_factor

    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
             color='#333333', linewidth=1.5, linestyle='--')

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
        plot_path = plot_dir / "hyperplanes.png"

        # Prepare XOR data
        x = config.data.x
        y = config.data.y

        # Plot using the styled helper
        plot_single_hyperplane_model(
            model=model,
            x=x,
            y=y,
            title=f"{experiment_name} Run {run_id:03d}",
            filename=plot_path
        )

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

    # Extract top-level blocks from analysis_results
    # Path: basic_stats
    stats = analysis_results.get("basic_stats", {})
    # Path: basic_stats.summary
    summary = stats.get("summary", {})
    # Path: basic_stats.experiment_info
    experiment_info = stats.get("experiment_info", {})
    # Path: basic_stats.distributions
    distributions = stats.get("distributions", {})
    # Path: basic_stats.distributions.accuracy_distribution.bins
    acc_bins = distributions.get("accuracy_distribution", {}).get("bins", {})
    # Path: basic_stats.success_metrics
    success_metrics = stats.get("success_metrics", {})
    # Path: basic_stats.summary.final_losses
    final_losses = summary.get("final_losses", {})

    # Extract prototype surface validation from analysis_results
    # Path: prototype_validation
    prototype_validation = analysis_results.get("prototype_validation", {})
    # Path: prototype_validation.mirror_test
    mirror_test = prototype_validation.get("mirror_test", [])
    # Path: prototype_validation.distance_test
    distance_test = prototype_validation.get("distance_test", [])

    # Core experiment metadata from config and experiment_info
    name = config.execution.experiment_name
    # Path: basic_stats.experiment_info.description
    description = config.description or experiment_info.get("description", "No description provided.")

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
    # Accesses 'mirror_count' within each item of prototype_validation.mirror_test[]
    mirror_detected = any(run.get("mirror_count", 0) > 0 for run in mirror_test)
    mirror_flag = "✅ Detected" if mirror_detected else "❌ None detected"

    # prototype surface interpretation flags
    proto_surface_ok = "✔️" if distance_test else "⚠️ Not available"
    geometry_ok = "✔️" if distance_test else "⚠️ Not available"
    prototype_support = "✅" if (avg_acc == 1.0 and distance_test) else "⚠️ Partial"


    # Extract prototype surface distance test results
    distance_entries = analysis_results.get("prototype_validation", {}).get("distance_test", [])

    # Accumulate distances per class
    class0_distances = []
    class1_distances = []

    for entry in distance_entries:
        dists = entry["distances"]
        labels = entry["labels"]
        for dist, label in zip(dists, labels):
            if label == 0.0:
                class0_distances.append(dist)
            elif label == 1.0:
                class1_distances.append(dist)

    # Convert to numpy arrays for stats
    d0 = np.array(class0_distances)
    d1 = np.array(class1_distances)

    # Compute stats
    class0_mean = d0.mean()
    class0_std = d0.std()
    class1_mean = d1.mean()
    class1_std = d1.std()

    # Start Markdown report
    report = f"# 🧪 Experiment Report: `{name}`\n\n"
    report += f"**Description**: {description}\n\n"

    report += "## 🎯 Summary\n"
    report += f"- Total runs: {total_runs}\n"
    report += f"- Successful runs (100% accuracy): {success_runs}\n"
    report += f"- Average accuracy: {avg_acc:.2f}\n"
    report += f"- Convergence rate (< 0.01 loss): {conv_rate:.1f}%\n\n"

    report += "## 📊 Accuracy Distribution\n"
    report += "| Accuracy | Runs |\n|----------|------|\n"
    accuracy_labels = {
        1.0: "100%",
        0.75: "75%",
        0.5: "50%",
        0.25: "25%",
        0.0: "0%"
    }
    # Accesses keys like "1.0", "0.75" etc. within basic_stats.distributions.accuracy_distribution.bins
    for key, label in accuracy_labels.items():
        report += f"| {label} | {acc_bins.get(key, 0)} |\n"

    weight_stats = analysis_results.get("basic_stats", {}).get("summary", {}).get("weight_norms", {})
    weight_mean = weight_stats.get("mean", 0.0)
    weight_std = weight_stats.get("std", 0.0)

    report += "\n## 🔍 Convergence Statistics\n"
    report += f"- Best final loss: {best_loss:.8e}\n"
    report += f"- Worst final loss: {worst_loss:.8e}\n"

    report += "\n## 🧠 Hyperplane Analysis\n"
    report += f"- Class 0 points mean distance to hyperplane: {class0_mean:.4e} ± {class0_std:.1e}\n"
    report += f"- Class 1 points mean distance to hyperplane: {class1_mean:.5f} ± {class1_std:.1e}\n"
    report += f"- Mean ||W|| (weight norm): {weight_mean:.6f} ± {weight_std:.1e}\n"

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
        print(analysis_plan)
        
        # Load experiment data and results
        print("Loading experiment data and results...")
        experiment_data = load_experiment_data(config)
        run_results = load_all_run_results(results_dir)
        print(f"✓ Loaded results from {len(run_results)} runs")

        # Perform comprehensive analysis
        print("\nPerforming analysis...")
        print("-" * 30)
        
        analysis_results = {}

        # 1. Basic statistics and aggregation
        if "basic_stats" in analysis_plan:
            print("📊 Computing basic statistics...")
            analysis_results["basic_stats"] = compute_basic_statistics(run_results, config)
            print("  ✓ Basic statistics computed")

        # 2. Accuracy and convergence analysis
        if "accuracy_analysis" in analysis_plan:
            print("🎯 Analyzing accuracy patterns...")
            analysis_results["accuracy"] = analyze_accuracy_distribution(run_results, config)
            print("  ✓ Accuracy analysis completed")
            
        # # 3. Geometric analysis (hyperplanes, prototype regions)
        # if "geometric_analysis" in analysis_plan:
        #     print("📐 Performing geometric analysis...")
        #     analysis_results["geometric"] = analyze_learned_geometry(
        #         run_results, experiment_data, config, plot_config
        #     )
        #     print("  ✓ Geometric analysis completed")

        # # 4. Weight pattern analysis
        # if "weight_analysis" in analysis_plan:
        #     print("⚖️  Analyzing weight patterns...")
        #     analysis_results["weights"] = analyze_weight_patterns(
        #         run_results, config
        #     )
        #     print("  ✓ Weight analysis completed")

        # 5. Prototype surface validation
        if "prototype_validation" in analysis_plan:
            print("🔬 Validating prototype surface predictions...")
            analysis_results["prototype_validation"] = validate_prototype_theory(
                run_results, experiment_data, config
            )
            print("  ✓ Prototype surface validation completed")

        # 6. Generate visualizations
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

        # 7. Generate comprehensive report
        print("📄 Generating analysis report...")
        report = generate_analysis_report(analysis_results, config, template="comprehensive")
        
        # Save report
        report_path = results_dir / f"analysis_{experiment_name}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  ✓ Report saved to {report_path}")

        # 8. Export analysis data
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
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
