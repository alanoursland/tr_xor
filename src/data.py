# data.py - Dataset Generation and Management for Prototype Surface Experiments

"""
Dataset generation and management module for prototype surface experiments.
Handles creation of XOR datasets, higher-dimensional parity problems, input space sampling
for visualization, and data preprocessing utilities with emphasis on geometric properties.
"""

import torch

# ==============================================================================
# Simple Dataset Creation
# ==============================================================================

def xor_data_centered() -> torch.Tensor:
    """
    Generate centered XOR input data.
    
    Returns:
        Tensor of shape (4, 2) with XOR input points: [-1,-1], [1,-1], [-1,1], [1,1]
    """
    return torch.tensor([
        [-1.0, -1.0],
        [ 1.0, -1.0], 
        [-1.0,  1.0],
        [ 1.0,  1.0]
    ], dtype=torch.float32)


def xor_labels_T1() -> torch.Tensor:
    """
    Generate XOR labels where True=1.
    
    Returns:
        Tensor of shape (4,) with XOR labels: [0, 1, 1, 0]
    """
    return torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)

def xor_labels_one_hot() -> torch.Tensor:
    """
    One-hot float targets for XOR with two output units (0=False, 1=True).
    Suitable for use with BCEWithLogitsLoss.

    Returns:
        Tensor of shape (4, 2), where each row is:
            [1.0, 0.0] for class 0 (XOR=False),
            [0.0, 1.0] for class 1 (XOR=True)
    """
    return torch.tensor([
        [1.0, 0.0],  # XOR(0, 0)
        [0.0, 1.0],  # XOR(0, 1)
        [0.0, 1.0],  # XOR(1, 0)
        [1.0, 0.0],  # XOR(1, 1)
    ], dtype=torch.float32)
