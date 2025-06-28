import torch
import math

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

