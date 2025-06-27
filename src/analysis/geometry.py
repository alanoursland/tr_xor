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

