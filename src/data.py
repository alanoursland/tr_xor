# data.py - Dataset Generation and Management for PSL Experiments

"""
Dataset generation and management module for Prototype Surface Learning (PSL) experiments.
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
