# data.py - Dataset Generation and Management for PSL Experiments

"""
Dataset generation and management module for Prototype Surface Learning (PSL) experiments.
Handles creation of XOR datasets, higher-dimensional parity problems, input space sampling
for visualization, and data preprocessing utilities with emphasis on geometric properties.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union, Callable
from pathlib import Path
import itertools
from enum import Enum


# ==============================================================================
# Data Types and Enums
# ==============================================================================

class NormalizationType(Enum):
    """Enumeration of supported normalization methods."""
    NONE = "none"
    STANDARDIZE = "standardize"  # Zero mean, unit variance
    MIN_MAX = "min_max"         # Scale to [0, 1]
    CENTER = "center"           # Zero mean only
    UNIT_NORM = "unit_norm"     # Unit L2 norm


class GridType(Enum):
    """Enumeration of grid sampling types for visualization."""
    UNIFORM = "uniform"         # Uniform grid spacing
    RANDOM = "random"          # Random sampling
    ADAPTIVE = "adaptive"      # Adaptive density based on data
    HALTON = "halton"          # Low-discrepancy sequence


# ==============================================================================
# XOR Data Generation
# ==============================================================================

def generate_xor_data(normalized: bool = True, center_origin: bool = True, 
                     dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate standard XOR truth table dataset.
    
    Args:
        normalized: Whether to use normalized input values (-1, 1) vs (0, 1)
        center_origin: Whether to center data at origin
        dtype: Data type for tensors
        
    Returns:
        Tuple of (inputs, labels) where inputs is (4, 2) and labels is (4,)
    """
    pass


def generate_xor_variants(scales: List[float], rotations: List[float], 
                         translations: List[Tuple[float, float]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate multiple transformed versions of XOR dataset for robustness testing.
    
    Args:
        scales: List of scaling factors to apply
        rotations: List of rotation angles in radians
        translations: List of (x, y) translation vectors
        
    Returns:
        List of (inputs, labels) tuples for each transformation
    """
    pass


def create_xor_with_noise(noise_std: float, num_samples: int = 4, 
                         base_normalized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create XOR dataset with added Gaussian noise for robustness experiments.
    
    Args:
        noise_std: Standard deviation of Gaussian noise
        num_samples: Number of samples per XOR point (if > 4, creates multiple noisy versions)
        base_normalized: Whether base XOR uses normalized coordinates
        
    Returns:
        Tuple of (noisy_inputs, labels) 
    """
    pass


def generate_xor_interpolations(num_points: int = 100, include_boundaries: bool = True) -> torch.Tensor:
    """
    Generate interpolated points between XOR corners for decision boundary analysis.
    
    Args:
        num_points: Number of interpolation points to generate
        include_boundaries: Whether to include original XOR points
        
    Returns:
        Tensor of interpolated points for analysis
    """
    pass


# ==============================================================================
# Parity Problem Generation
# ==============================================================================

def generate_parity_data(n_bits: int, signed: bool = True, 
                        dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate n-bit parity problem dataset (complete truth table).
    
    Args:
        n_bits: Number of input bits
        signed: Whether to use {-1, 1} encoding vs {0, 1}
        dtype: Data type for tensors
        
    Returns:
        Tuple of (inputs, labels) where inputs is (2^n_bits, n_bits) and labels is (2^n_bits,)
    """
    pass


def create_boolean_hypercube(n_dims: int, signed: bool = True) -> torch.Tensor:
    """
    Generate all 2^n vertices of n-dimensional boolean hypercube.
    
    Args:
        n_dims: Number of dimensions
        signed: Whether to use {-1, 1} vs {0, 1} encoding
        
    Returns:
        Tensor of shape (2^n_dims, n_dims) with all boolean combinations
    """
    pass


def sample_parity_subset(n_bits: int, num_samples: int, 
                        balanced: bool = True, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random subset of n-bit parity problem for large n.
    
    Args:
        n_bits: Number of input bits
        num_samples: Number of samples to generate
        balanced: Whether to ensure equal numbers of 0s and 1s
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_inputs, parity_labels)
    """
    pass


def generate_structured_parity_samples(n_bits: int, structure_type: str = "corners") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate structured samples from parity space (corners, edges, faces, etc.).
    
    Args:
        n_bits: Number of input bits
        structure_type: Type of structure to sample ("corners", "edges", "faces", "random")
        
    Returns:
        Tuple of (structured_inputs, parity_labels)
    """
    pass


# ==============================================================================
# Visualization Support Data
# ==============================================================================

def sample_input_space(bounds: Tuple[Tuple[float, float], ...], resolution: int = 100, 
                      grid_type: GridType = GridType.UNIFORM) -> torch.Tensor:
    """
    Generate dense sampling of input space for visualization and analysis.
    
    Args:
        bounds: Tuple of (min, max) bounds for each dimension
        resolution: Number of points per dimension (for uniform grid)
        grid_type: Type of sampling grid to use
        
    Returns:
        Tensor of sampled points with shape (num_points, num_dims)
    """
    pass


def generate_decision_boundary_samples(model: torch.nn.Module, bounds: Tuple[Tuple[float, float], ...], 
                                     num_points: int = 1000, 
                                     boundary_threshold: float = 0.1) -> torch.Tensor:
    """
    Generate samples near model decision boundaries for detailed analysis.
    
    Args:
        model: Trained model to analyze
        bounds: Input space bounds
        num_points: Number of boundary samples to generate
        boundary_threshold: Distance threshold for boundary proximity
        
    Returns:
        Tensor of points near decision boundaries
    """
    pass


def create_prototype_region_samples(model: torch.nn.Module, region_id: int, 
                                  bounds: Tuple[Tuple[float, float], ...], 
                                  density: int = 50) -> torch.Tensor:
    """
    Generate dense samples within specific prototype regions for analysis.
    
    Args:
        model: Model defining prototype regions
        region_id: ID of specific region to sample
        bounds: Input space bounds
        density: Sampling density within region
        
    Returns:
        Tensor of points within specified prototype region
    """
    pass


def sample_hyperplane_vicinity(weights: torch.Tensor, bias: float, 
                              bounds: Tuple[Tuple[float, float], ...], 
                              distance_range: Tuple[float, float] = (-0.5, 0.5),
                              num_points: int = 1000) -> torch.Tensor:
    """
    Sample points in vicinity of specified hyperplane.
    
    Args:
        weights: Hyperplane weight vector
        bias: Hyperplane bias term
        bounds: Input space bounds
        distance_range: Range of distances from hyperplane to sample
        num_points: Number of points to generate
        
    Returns:
        Tensor of points near hyperplane
    """
    pass


def create_activation_analysis_grid(bounds: Tuple[Tuple[float, float], ...], 
                                   resolution: Tuple[int, ...]) -> torch.Tensor:
    """
    Create specialized grid for activation function analysis and visualization.
    
    Args:
        bounds: Input space bounds for each dimension
        resolution: Resolution (number of points) for each dimension
        
    Returns:
        Tensor grid optimized for activation analysis
    """
    pass


# ==============================================================================
# Data Preprocessing Utilities
# ==============================================================================

def normalize_dataset(x: torch.Tensor, method: NormalizationType = NormalizationType.STANDARDIZE,
                     parameters: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Normalize dataset using specified method, returning normalization parameters.
    
    Args:
        x: Input data tensor
        method: Normalization method to use
        parameters: Pre-computed normalization parameters (for test data)
        
    Returns:
        Tuple of (normalized_data, normalization_parameters)
    """
    pass


def center_at_origin(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center data at origin by subtracting mean.
    
    Args:
        x: Input data tensor
        
    Returns:
        Tuple of (centered_data, original_mean)
    """
    pass


def apply_rotation(x: torch.Tensor, angle: float, center: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply 2D rotation to data points.
    
    Args:
        x: Input data tensor (N, 2)
        angle: Rotation angle in radians
        center: Center point for rotation (default: origin)
        
    Returns:
        Rotated data tensor
    """
    pass


def apply_scaling(x: torch.Tensor, scale_factors: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Apply scaling transformation to data.
    
    Args:
        x: Input data tensor
        scale_factors: Scalar or per-dimension scaling factors
        
    Returns:
        Scaled data tensor
    """
    pass


def apply_translation(x: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """
    Apply translation to data points.
    
    Args:
        x: Input data tensor
        translation: Translation vector
        
    Returns:
        Translated data tensor
    """
    pass


def add_gaussian_noise(x: torch.Tensor, std: float, seed: Optional[int] = None) -> torch.Tensor:
    """
    Add Gaussian noise to data for robustness experiments.
    
    Args:
        x: Input data tensor
        std: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Noisy data tensor
    """
    pass


def apply_nonlinear_transformation(x: torch.Tensor, transformation: str = "polynomial", 
                                 degree: int = 2, coefficients: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply nonlinear transformation to create more complex datasets.
    
    Args:
        x: Input data tensor
        transformation: Type of transformation ("polynomial", "sinusoidal", "radial")
        degree: Degree of transformation (for polynomial)
        coefficients: Custom transformation coefficients
        
    Returns:
        Transformed data tensor
    """
    pass


# ==============================================================================
# Batch Management
# ==============================================================================

def create_training_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, 
                           shuffle: bool = True, drop_last: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create training batches from dataset with optional shuffling.
    
    Args:
        x: Input data tensor
        y: Label tensor
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before batching
        drop_last: Whether to drop incomplete final batch
        
    Returns:
        List of (batch_x, batch_y) tuples
    """
    pass


def create_full_batch_iterator(datasets: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create iterator for small datasets that use full-batch training (like XOR).
    
    Args:
        datasets: List of (x, y) dataset tuples
        
    Returns:
        List of full datasets for batch processing
    """
    pass


def balance_classes(x: torch.Tensor, y: torch.Tensor, method: str = "oversample") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Balance class representation in dataset.
    
    Args:
        x: Input data tensor
        y: Label tensor
        method: Balancing method ("oversample", "undersample", "synthetic")
        
    Returns:
        Tuple of (balanced_x, balanced_y)
    """
    pass


def create_stratified_split(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8, 
                           seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create stratified train/validation split maintaining class proportions.
    
    Args:
        x: Input data tensor
        y: Label tensor
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (x_train, x_val, y_train, y_val)
    """
    pass


def create_cross_validation_folds(x: torch.Tensor, y: torch.Tensor, n_folds: int = 5, 
                                 shuffle: bool = True, seed: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create cross-validation folds for robust evaluation.
    
    Args:
        x: Input data tensor
        y: Label tensor
        n_folds: Number of folds to create
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
        
    Returns:
        List of (x_train, x_val, y_train, y_val) tuples for each fold
    """
    pass


# ==============================================================================
# Dataset Validation
# ==============================================================================

def verify_xor_labels(x: torch.Tensor, y: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """
    Verify that XOR dataset labels are correct according to XOR truth table.
    
    Args:
        x: Input tensor (should be XOR inputs)
        y: Label tensor (should be XOR outputs)
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if labels are correct XOR values
    """
    pass


def check_parity_consistency(x: torch.Tensor, y: torch.Tensor, n_bits: int) -> bool:
    """
    Validate that parity problem labels are consistent with input patterns.
    
    Args:
        x: Input tensor (boolean patterns)
        y: Label tensor (parity values)
        n_bits: Expected number of bits
        
    Returns:
        True if parity labels are correct
    """
    pass


def analyze_class_separability(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    Analyze linear separability and other geometric properties of dataset.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary containing separability metrics
    """
    pass


def validate_dataset_geometry(x: torch.Tensor, expected_properties: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate geometric properties of dataset (dimensionality, bounds, distribution).
    
    Args:
        x: Input data tensor
        expected_properties: Dictionary of expected geometric properties
        
    Returns:
        Dictionary of validation results
    """
    pass


def check_dataset_completeness(x: torch.Tensor, y: torch.Tensor, problem_type: str) -> Dict[str, Any]:
    """
    Check if dataset contains all expected samples for given problem type.
    
    Args:
        x: Input data tensor
        y: Label tensor
        problem_type: Type of problem ("xor", "parity", "custom")
        
    Returns:
        Dictionary with completeness analysis
    """
    pass


def analyze_data_distribution(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze statistical properties of data distribution.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary containing distribution statistics
    """
    pass


# ==============================================================================
# Geometric Analysis Support
# ==============================================================================

def compute_data_centroid(x: torch.Tensor, y: torch.Tensor, class_id: Optional[int] = None) -> torch.Tensor:
    """
    Compute centroid of data points, optionally for specific class.
    
    Args:
        x: Input data tensor
        y: Label tensor
        class_id: Specific class to compute centroid for (None for all data)
        
    Returns:
        Centroid coordinates
    """
    pass


def compute_class_separation_metrics(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    Compute various metrics quantifying separation between classes.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary of separation metrics
    """
    pass


def find_closest_points_between_classes(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Find closest points between different classes in dataset.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Tuple of (point1, point2, distance) for closest inter-class points
    """
    pass


def compute_convex_hulls_by_class(x: torch.Tensor, y: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Compute convex hull for each class in dataset.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary mapping class IDs to convex hull vertices
    """
    pass


def analyze_geometric_complexity(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    Analyze geometric complexity of classification problem.
    
    Args:
        x: Input data tensor
        y: Label tensor
        
    Returns:
        Dictionary of complexity metrics
    """
    pass


# ==============================================================================
# Custom Dataset Creation
# ==============================================================================

def create_custom_boolean_problem(truth_table: Dict[Tuple[int, ...], int], 
                                 signed: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create custom boolean classification problem from truth table.
    
    Args:
        truth_table: Dictionary mapping input tuples to output labels
        signed: Whether to use {-1, 1} vs {0, 1} encoding
        
    Returns:
        Tuple of (inputs, labels) for custom problem
    """
    pass


def generate_synthetic_linearly_separable(num_samples: int, num_features: int, 
                                         margin: float = 1.0, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic linearly separable dataset with specified margin.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of input features
        margin: Separation margin between classes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (synthetic_inputs, binary_labels)
    """
    pass


def generate_synthetic_nonlinearly_separable(num_samples: int, num_features: int, 
                                           complexity: float = 1.0, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic dataset requiring nonlinear separation.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of input features
        complexity: Complexity of nonlinear boundary
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (synthetic_inputs, binary_labels)
    """
    pass


def load_external_dataset(filepath: Path, format: str = "auto") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load dataset from external file (CSV, NPY, etc.).
    
    Args:
        filepath: Path to dataset file
        format: File format ("csv", "npy", "json", "auto")
        
    Returns:
        Tuple of (loaded_inputs, loaded_labels)
    """
    pass


def save_dataset(x: torch.Tensor, y: torch.Tensor, filepath: Path, 
                format: str = "npy", metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save dataset to file with optional metadata.
    
    Args:
        x: Input data tensor
        y: Label tensor
        filepath: Output file path
        format: Save format ("npy", "csv", "json")
        metadata: Optional metadata to save with dataset
    """
    pass