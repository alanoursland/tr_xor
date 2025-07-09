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
    return torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)


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
    return torch.tensor(
        [
            [1.0, 0.0],  # XOR(0, 0)
            [0.0, 1.0],  # XOR(0, 1)
            [0.0, 1.0],  # XOR(1, 0)
            [1.0, 0.0],  # XOR(1, 1)
        ],
        dtype=torch.float32,
    )

def accuracy_binary_threshold(output: torch.Tensor, target: torch.Tensor) -> float:
    # Squeeze output in case shape is (N, 1)
    output = output.squeeze()

    # Apply threshold at 0.5
    preds = (output >= 0.5).float()

    # Compute accuracy
    return (preds == target).float().mean().item()


def accuracy_one_hot(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.argmax(output, dim=1)
    true = torch.argmax(target, dim=1)
    return (preds == true).float().mean().item()
