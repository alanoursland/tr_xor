import json
import hashlib
import torch
import re

from typing import Dict, List, Tuple, Any
from pathlib import Path
from configs import ExperimentConfig, ExperimentType
from analysis.grid import create_boundary_focused_grid, create_high_dim_sample_grid
from analysis.stats import compute_data_statistics, compute_class_centroids
from analysis.utils import extract_activation_type, count_model_parameters, get_linear_layers, analyze_model_state_dict

# load_experiment_data, load_all_run_results, load_single_run_result

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
        'loss_change_threshold': getattr(config.training, 'loss_change_threshold', 1e-6),
        
        # Experiment metadata
        'experiment_name': config.execution.experiment_name,
        'description': config.description,
        'config_hash': generate_config_hash(config)
    }
    
    return analysis_data

def load_all_run_results(results_dir: Path, config: ExperimentConfig) -> List[Dict[str, Any]]:
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
            run_result = load_single_run_result(run_dir, run_id, config)
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

def load_single_run_result(run_dir: Path, run_id: int, config: ExperimentConfig) -> Dict[str, Any]:
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
        result['model_linear_layers'] = get_linear_layers(config.model)
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
    loss_change_threshold = 0.01 # replaced with config.training.loss_change_threshold

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
    result['converged'] = result['final_loss'] < loss_change_threshold
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

def generate_config_hash(config: ExperimentConfig) -> str:
    """
    Generate unique hash for configuration identification.
    
    Args:
        config: Configuration to hash
        
    Returns:
        Unique hash string
    """
    
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

