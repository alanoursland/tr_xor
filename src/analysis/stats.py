import torch

from typing import Dict, List, Tuple, Any
from analysis.utils import targets_to_class_labels

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

