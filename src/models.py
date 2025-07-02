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
# Custom Activations and Layers
# ==============================================================================

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


class Abs(nn.Module):
    """
    A neural network module that applies the absolute value function element-wise.

    This class wraps the torch.abs function in an nn.Module, allowing it to be
    seamlessly integrated into a model's architecture (e.g., within an
    nn.Sequential container) and be discoverable by hooks.

    Shape:
        - Input: (N, *) where * means any number of additional dimensions.
        - Output: (N, *), with the same shape as the input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the absolute value function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A new tensor with the element-wise absolute value of the input.
        """
        return torch.abs(x)


class Sum(nn.Module):
    """
    A neural network module that sums a tensor along a specified dimension.

    This class wraps the torch.sum function in an nn.Module, allowing it to be
    integrated into a model's architecture and be discoverable by hooks.

    Args:
        dim (int): The dimension along which to sum.
        keepdim (bool): Whether the output tensor has `dim` retained or not.
                        Defaults to False.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the sum operation to the input tensor.
        """
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)


class StaticScale(nn.Module):
    def __init__(self, features, device=None):
        """
        A non-learnable scaling layer that multiplies each input feature by a fixed scale.

        Args:
            features (int): Number of input features (dimensions).
            device (torch.device or str, optional): Device to place the scale buffer on.
        """
        super().__init__()
        weight = torch.ones(features, dtype=torch.float32, device=device)
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x * self.weight


class Confidence(nn.Module):
    """
    A custom PyTorch layer that scales the input by a single learnable parameter.

    This layer multiplies the input tensor by a scalar 'confidence' parameter,
    which is learned during the training process.

    Args:
        initial_value (float, optional): The initial value for the confidence
                                         parameter. Defaults to 1.0.

    Shape:
        - Input: (N, *) where * means any number of additional dimensions.
        - Output: (N, *), same shape as the input.
    """

    def __init__(self, initial_value: float = 1.0):
        super(Confidence, self).__init__()
        # Initialize the single learnable parameter.
        # We wrap it in nn.Parameter to ensure it is registered as a model parameter
        # and will be updated during the training process (e.g., by an optimizer).
        self.confidence = nn.Parameter(torch.tensor(initial_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor scaled by the confidence parameter.
        """
        return x * torch.square(self.confidence)


class MagnitudeEater(nn.Module):
    def __init__(self, max_points: int):
        super().__init__()
        self.max_points = max_points
        # Store the running average and point count directly
        self.register_buffer('running_avg', torch.tensor(1.0))
        self.register_buffer('point_count', torch.tensor(0.0))

    def forward(self, x):
        if not self.training:
            return x

        # Amplify the input using the stored running average
        # print(running_avg)
        # setting x = x*running_avg is too much regularization
        # let running_avg = gain+1
        # and the equation above be x = x*(1+gain)
        # x = x + x*gain
        # now we can weaken the regularization by adding a factor
        # x = x + k*x*gain
        gain = self.running_avg.detach() - 1
        amplified_x = x + (0.1 * x * gain)

        # Update the sliding window average using the stable math
        with torch.no_grad():
            # Get stats for the new batch
            N = x.shape[0]
            # print(f"N = {N}")
            A_N = torch.mean(torch.norm(x, p=2, dim=1)) # New average
            # print(f"A_N = {A_N}")
            
            # Current state
            M = self.point_count
            # print(f"M = {M}")
            A_M = self.running_avg
            # print(f"A_M = {A_M}")
            Max = self.max_points
            # print(f"Max = {Max}")

            # Number of points to discard
            d = F.relu((M + N) - Max)
            # print(f"d = {d}")

            # New total count in the window
            M_final = (M - d) + N
            # print(f"M_final = {M_final}")

            # Calculate the final average using the derived formula
            # Avoid division by zero if the new count is zero
            if M_final > 0:
                numerator = (A_M * (M - d)) + (A_N * N)
                A_final = numerator / M_final
            else: # Should not happen in practice if N > 0
                A_final = torch.tensor(1.0) 
            # print(f"M_final = {A_final}")

            # Update state for the next iteration
            self.running_avg = A_final
            self.point_count = M_final
            
        return amplified_x

    def init(self):
        self.running_avg.zero_()
        self.point_count.zero_()

# ==============================================================================
# Custom Models
# ==============================================================================


class Model_Abs1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 1)
        self.activation = Abs()
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x.squeeze()

    def init_normal(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.linear1.bias)
        return self

    def init_kaiming(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
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
        self.activation = nn.ReLU()
        self.sum_layer = Sum(dim=1, keepdim=True)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.sum_layer(x)
        return x.squeeze()

    def forward_components(self, x):
        """
        Returns pre-activation outputs of each linear unit before ReLU is applied.
        """
        return self.linear1(x)

    @torch.no_grad()
    def init_normal(self):
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.linear1.bias)
        return self

    @torch.no_grad()
    def init_mirror(self):
        """
        Initializes the layer's weights and biases in mirror pairs.
        The first half of the neurons are initialized from a normal distribution,
        and the second half are initialized as the negation of the first half.
        """
        midpoint = self.linear1.weight.shape[0] // 2
        self.linear1.weight[midpoint:] = -self.linear1.weight[:midpoint].clone()
        self.linear1.bias[midpoint:] = -self.linear1.bias[:midpoint].clone()

        return self

    @torch.no_grad()
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

    @torch.no_grad()
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


class Model_Xor2(nn.Module):
    def __init__(self, middle, activation):
        super().__init__()
        self.linear1 = nn.Linear(2, middle)
        self.activation = activation
        self.scale1 = StaticScale(middle)
        self.linear2 = nn.Linear(middle, 2)
        self.scale2 = StaticScale(2)
        self.init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.scale1(x)
        x = self.linear2(x)
        x = self.scale2(x)
        return x

    def init(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

        nn.init.ones_(self.scale1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        nn.init.ones_(self.scale2.weight)

        return self

    @torch.no_grad()
    def decompose_weight(self, linear: nn.Linear, scale: StaticScale):
        W = linear.weight
        b = linear.bias

        row_norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W /= row_norms
        b /= row_norms.squeeze()

        scale.weight *= row_norms.squeeze()


    @torch.no_grad()
    def absorb_scale(self, scale: StaticScale, linear: nn.Linear):
        W = linear.weight
        W *= scale.weight.unsqueeze(0)
        scale.weight.fill_(1.0)
        
    @torch.no_grad()
    def normalization_propagation(self):
        """Apply manual weight normalization and propagate scaling factors."""
        self.decompose_weight(self.linear1, self.scale1)
        self.absorb_scale(self.scale1, self.linear2)
        self.decompose_weight(self.linear2, self.scale2)
        print("Normalization Propagation done.")

class Model_Xor2_Confidence(nn.Module):
    def __init__(self, middle, activation):
        super().__init__()
        self.linear1 = nn.Linear(2, middle)
        self.activation = activation
        self.scale1 = StaticScale(middle)
        self.linear2 = nn.Linear(middle, 2)
        self.scale2 = StaticScale(middle)
        self.confidence = Confidence()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.scale1(x)
        x = self.linear2(x)
        x = self.scale2(x)
        x = self.confidence(x)
        return x

    def init(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

        nn.init.ones_(self.scale1.weight)

        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        nn.init.ones_(self.scale2.weight)

        nn.init.ones_(self.confidence.confidence)

        return self

class Model_Xor2_Eater(nn.Module):
    def __init__(self, middle, activation, max_points):
        super().__init__()
        self.linear1 = nn.Linear(2, middle)
        self.activation = activation
        self.eater1 = MagnitudeEater(max_points)
        self.linear2 = nn.Linear(middle, 2)
        self.eater2 = MagnitudeEater(max_points)
        self.init()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.eater1(x)
        x = self.linear2(x)
        x = self.eater2(x)
        return x

    def init(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)

        self.eater1.init()
        
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
        self.eater2.init()


        return self
