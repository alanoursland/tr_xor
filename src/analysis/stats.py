import torch
import numpy as np

from typing import Dict, List, Tuple, Any
from analysis.utils import targets_to_class_labels
from configs import ExperimentConfig, get_experiment_config, list_experiments

def compute_data_statistics(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the dataset.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary of data statistics
    """
    # Label statistics
    y_indices = targets_to_class_labels(y)

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
        
        'label_distribution': torch.bincount(y_indices) / len(y),
        'is_balanced': torch.std(torch.bincount(y_indices).float()) < 0.1,
        
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
    y_indices = targets_to_class_labels(y)
    unique_labels = torch.unique(y_indices)

    min_distance = float('inf')

    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:
            mask1 = (y_indices == label1)
            mask2 = (y_indices == label2)

            x1 = x[mask1]
            x2 = x[mask2]

            # Skip if either class is empty
            if x1.numel() == 0 or x2.numel() == 0:
                continue

            # Ensure 2D inputs for cdist
            if x1.ndim == 1:
                x1 = x1.unsqueeze(0)
            if x2.ndim == 1:
                x2 = x2.unsqueeze(0)

            distances = torch.cdist(x1, x2, p=2)
            min_distance = min(min_distance, distances.min().item())

    return min_distance if min_distance < float('inf') else 0.0

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
            'training_epochs': config.training.epochs
        },
        
        'summary': summary_stats,
        'distributions': distribution_stats,
        'success_metrics': success_stats,
        'training_dynamics': training_stats,
        'model_properties': model_stats,
        
        # Raw metrics for further analysis - MAKE SURE THIS LINE EXISTS
        'raw_metrics': metrics
    }

    return basic_stats

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
    loss_change_threshold = 0.01 # replaced with config.training.loss_change_threshold
    
    # Accuracy distribution (especially important for XOR)
    if metrics['accuracies']:
        accuracies = metrics['accuracies']
        
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
    
    # Loss distribution
    if metrics['final_losses']:
        loss_tensor = torch.tensor(metrics['final_losses'])
        distributions['loss_distribution'] = {
            'histogram': compute_histogram(loss_tensor, bins=20),
            'convergence_rate': sum(1 for loss in metrics['final_losses'] if loss < loss_change_threshold) / len(metrics['final_losses']),
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

