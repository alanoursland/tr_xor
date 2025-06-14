# models.py - Neural Network Definitions for Prototype Surface Experiments

"""
Neural network model architectures for prototype surface experiments.
Provides configurable multi-layer perceptrons with specialized focus on geometric
interpretability and prototype surface theory validation. Includes custom activation functions
and analysis methods for prototype surface investigation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod


# ==============================================================================
# Custom Models
# ==============================================================================


class Model_Abs1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 1)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.abs(x)
        return x.squeeze()
    
    def init_normal(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.linear1.bias)
        return self
    
    def init_kaiming(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        return self
    
    def init_xavier(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        return self

    def init_tiny(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.linear1.bias)
        return self

    def init_large(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=4.0)
        nn.init.zeros_(self.linear1.bias)
        return self
    
class Model_ReLU1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = x.sum(dim=1, keepdim=True)
        return x.squeeze()
    
    def forward_components(self, x):
        """
        Returns pre-activation outputs of each linear unit before ReLU is applied.
        """
        return self.linear1(x)

    def init_normal(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.linear1.bias)
        return self
    
    def reinit_dead_data(self, init_fn, data, max_attempts=100, min_threshold=0):
        """
        Reinitialize the model until no data points are dead (produce all negative pre-activations).
        
        Args:
            init_fn: Function to call for reinitialization (e.g., self.init_normal)
            data: Input data tensor to check for dead data
            max_attempts: Maximum number of reinit attempts before giving up
        
        Returns:
            bool: True if successful (no dead data), False if max_attempts exceeded
        """
        for attempt in range(max_attempts):
            # Apply the initialization function
            init_fn()
            
            # Get pre-activation values for all data points
            with torch.no_grad():
                pre_activations = self.forward_components(data)  # Shape: [n_data, n_nodes]
            
            # Check if any data point produces negative values across ALL nodes
            # A data point is "dead" if all its pre-activations are negative
            all_negative_per_datapoint = (pre_activations <= min_threshold).all(dim=1)  # Shape: [n_data]
            
            # If no data points are dead, we're done
            if not all_negative_per_datapoint.any():
                print(f"{attempt} reinitializations")
                return self
            
        
        # Failed to find initialization without dead data
        raise RuntimeError(f"reinit_dead_data failed after {max_attempts} attempts")

    def init_bounded_hypersphere(self, weight_init_fn, radius, data_mean=None):
        """
        Bounded Hypersphere Initialization: Initialize weights using weight_init_fn,
        then set biases so hyperplanes are tangent to a hypersphere of given radius.
        
        Args:
            weight_init_fn: Function to initialize weights (e.g., lambda: nn.init.kaiming_normal_(self.linear1.weight))
            radius: Radius of the enclosing hypersphere
            data_mean: Center of the hypersphere (defaults to zero vector)
        
        Returns:
            self: Returns the model instance for method chaining
        """
        # Initialize weights using the provided function
        weight_init_fn()
        
        # Set default data mean to zero vector if not provided
        if data_mean is None:
            data_mean = torch.zeros(self.linear1.in_features)
        
        # Set biases for hypersphere tangency
        with torch.no_grad():
            for i in range(self.linear1.out_features):
                # print(f"Node {i}")
                w = self.linear1.weight[i]
                # print(f"  w={w}")
                w_norm = torch.norm(w)
                # print(f"  w_norm={w_norm}")
                hypersphere_point = data_mean - w * radius / w_norm
                # print(f"  hypersphere_point={hypersphere_point}")
                # b = ||W|| * radius for tangent hyperplane pointing inward
                self.linear1.bias[i] = -torch.dot(w, hypersphere_point)
                # print(f"  bias={self.linear1.bias[i]}")
        
        return self

class ParametricAbs(nn.Module):
    """
    Parametric absolute value: f(x) = α * |x + β| + γ

    Learnable parameters allow the network to adapt the prototype surface location
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, gamma: float = 0.0, learnable: bool = False):
        super(ParametricAbs, self).__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))
            self.register_buffer("gamma", torch.tensor(gamma))
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parametric absolute value activation."""
        pass

    def extra_repr(self) -> str:
        """String representation for debugging."""
        pass

