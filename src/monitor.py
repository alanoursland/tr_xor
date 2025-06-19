# monitor.py - Prototype Surface Health Monitor for ReLU Networks

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MonitorMetrics:
    """Single timestep metrics from the monitor."""
    step: int
    dead_data_fraction: float  # Fraction of examples with all ReLUs off
    torque_ratio: float        # Mean ratio of tangential to total gradient
    bias_grad_ratio: float     # Mean |∇b| / |b| ratio (bias movement)
    bias_drift: float          # Mean step-to-step change in bias vectors (||b_t - b_{t-1}||)
    dead_neurons: List[int] = field(default_factory=list)  # Indices of dead neurons
    
    @property
    def has_dead_data_issue(self) -> bool:
        return self.dead_data_fraction > 0.0
    
    @property 
    def has_torque_issue(self) -> bool:
        return self.torque_ratio < 0.01

    @property
    def has_bias_issue(self) -> bool:
        return self.bias_grad_ratio < 0.01

    @property
    def has_bias_freeze_issue(self) -> bool:
        return self.bias_drift < 1e-4

class PrototypeSurfaceMonitor:
    """
    Monitor for detecting dead-data collapse and no-torque traps in ReLU networks.
    
    Usage:
        monitor = PrototypeSurfaceMonitor(model)
        
        # In training loop:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Check health before optimizer step
        metrics = monitor.check()
        if metrics.has_dead_data_issue:
            print(f"Warning: {metrics.dead_data_fraction:.1%} dead data")
            
        optimizer.step()
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize monitor for a model.
        
        Args:
            model: PyTorch model to monitor (must have ReLU-like activations)
        """
        self.model = model
        self.history: List[MonitorMetrics] = []
        self.step_count = 0
        self._prev_bias = {n: p.detach().clone()
                   for n, p in model.named_parameters() if 'bias' in n}

        # Find and register hooks on ReLU layers
        self._register_hooks()

    def to(self, device: torch.device) -> None:
        """
        Move internal tensors to the specified device (e.g., CUDA or CPU).
        """
        self.device = device
        for name in self._prev_bias:
            self._prev_bias[name] = self._prev_bias[name].to(device)

    def check(self) -> MonitorMetrics:
        """
        Compute current health metrics. Call after backward() but before optimizer.step().
        
        Returns:
            MonitorMetrics for current step
        """
        # Compute metrics from cached activations and gradients
        metrics = self._compute_metrics()
        self.history.append(metrics)
        self.step_count += 1
        return metrics
    
    def get_history(self) -> List[MonitorMetrics]:
        """Get full history of metrics."""
        return self.history
    
    def clear_history(self) -> None:
        """Clear stored metrics history."""
        self.history = []
        self.step_count = 0
    
    def summary(self) -> Dict[str, float]:
        """
        Get summary statistics over recent history.
        
        Returns:
            Dict with keys like 'mean_dead_fraction', 'max_dead_fraction', etc.
        """
        if not self.history:
            return {}
            
        recent = self.history[-100:]  # Last 100 steps
        
        return {
            'mean_dead_fraction': sum(m.dead_data_fraction for m in recent) / len(recent),
            'max_dead_fraction': max(m.dead_data_fraction for m in recent),
            'mean_torque_ratio': sum(m.torque_ratio for m in recent) / len(recent),
            'min_torque_ratio': min(m.torque_ratio for m in recent),
            'steps_with_issues': sum(1 for m in recent if m.has_dead_data_issue or m.has_torque_issue),
            'mean_bias_grad_ratio': sum(m.bias_grad_ratio for m in recent) / len(recent),
            'min_bias_grad_ratio': min(m.bias_grad_ratio for m in recent),
            'mean_bias_drift': sum(m.bias_drift for m in recent) / len(recent),
            'min_bias_drift': min(m.bias_drift for m in recent),
        }
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on ReLU layers."""
        self.relu_layers = []
        self.activations = {}
        self.gradients = {}
        
        # Find all ReLU-like layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU)):
                self.relu_layers.append((name, module))
                
                # Register forward hook to capture pre-activation values
                def make_forward_hook(layer_name):
                    def hook(module, input, output):
                        # Store pre-activation values (input to ReLU)
                        self.activations[layer_name] = input[0].detach()
                    return hook
                
                module.register_forward_hook(make_forward_hook(name))
                
        def make_grad_hook(layer_name):
            def hook(grad):
                self.gradients[layer_name] = grad.detach()
            return hook

        # Also need to capture gradients w.r.t. weights
        # Find Linear/Conv layers that feed into ReLUs
        # register_hook on .weight only works after .backward() has been called. 
        # Ensure it’s not missed by optimizers that clear gradients immediately.
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.weight.requires_grad:
                    module.weight.register_hook(make_grad_hook(name + '_weight'))
                if module.bias is not None and module.bias.requires_grad:
                    module.bias.register_hook(make_grad_hook(name + '_bias'))

    def _compute_metrics(self) -> MonitorMetrics:
        """Compute metrics from current activations and gradients."""
        
        # 1. Dead-data detection
        dead_data_fraction = compute_dead_data_fraction(self.activations)
        
        # 2. No-torque detection
        weight_torque, bias_ratio = compute_mean_torque_ratio(self.gradients, self.model)
        
        # 3. Dead neurons (expensive, check less frequently)
        dead_neurons = []
        if self.step_count % 100 == 0:
            dead_neurons = find_dead_neurons(self.activations)
        
        # 4. Bias drift
        bias_drifts = []
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                drift = (param - self._prev_bias[name]).norm()
                bias_drifts.append(drift)
                self._prev_bias[name] = param.detach().clone()

        avg_bias_drift = torch.stack(bias_drifts).mean().item() if bias_drifts else 0.0

        return MonitorMetrics(
                step=self.step_count,
                dead_data_fraction=dead_data_fraction,
                torque_ratio=weight_torque,
                bias_grad_ratio=bias_ratio,
                dead_neurons=dead_neurons,
                bias_drift=avg_bias_drift
)

# Utility functions
def compute_dead_data_fraction(activations: Dict[str, torch.Tensor]) -> float:
    """
    Compute fraction of samples with no active ReLUs across any layer.
    
    Args:
        activations: Dict mapping layer names to pre-activation tensors
        
    Returns:
        Fraction of "dead" samples (0.0 to 1.0)
    """
    if not activations:
        return 0.0
        
    dead_samples = None
    
    for acts in activations.values():
        # Flatten spatial dimensions: [batch, channels, ...] -> [batch, features]
        batch_size = acts.shape[0]
        flat_acts = acts.view(batch_size, -1)
        
        # Check which samples have NO active neurons in this layer
        layer_dead = (flat_acts <= 0).all(dim=1)  # [batch]
        
        # Accumulate: sample is dead if dead in ANY layer
        if dead_samples is None:
            dead_samples = layer_dead
        else:
            dead_samples = dead_samples | layer_dead
    
    return dead_samples.float().mean().item()


def compute_mean_torque_ratio(gradients: Dict[str, torch.Tensor],
                              model: nn.Module) -> Tuple[float, float]:
    """
    Returns:
        (avg_weight_torque, avg_bias_grad_ratio)
    """
    weight_ratios = []
    bias_ratios   = []

    # ---------- weights ----------
    for pname, grad in gradients.items():
        if 'weight' not in pname or grad is None:
            continue
        weight = dict(model.named_parameters()).get(pname.replace('_weight', '.weight'))
        if weight is None:
            continue
        ratios = compute_layer_torque_ratios(weight, grad)
        if ratios.numel() > 0:
            weight_ratios.append(ratios)

    avg_weight_ratio = (
        torch.cat(weight_ratios).mean().item()
        if weight_ratios else 1.0
    )

    # ---------- biases ----------
    for pname, grad in gradients.items():
        if 'bias' not in pname or grad is None: 
            continue
        bias = dict(model.named_parameters()).get(pname.replace('_bias', '.bias'))
        if bias is None: 
            continue
        bias_ratios.append(grad.norm() / (bias.norm() + 1e-8))

    avg_bias_ratio = (
        torch.tensor(bias_ratios).mean().item()
        if bias_ratios else 1.0
    )

    return avg_weight_ratio, avg_bias_ratio

def compute_layer_torque_ratios(weight: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Compute torque ratios for all neurons in a single layer.
    
    Args:
        weight: Weight tensor [out_features, in_features, ...]
        grad: Gradient tensor (same shape as weight)
        
    Returns:
        Torque ratios for each output neuron
    """
    # Reshape to [num_outputs, -1]
    W = weight.view(weight.shape[0], -1)
    G = grad.view(grad.shape[0], -1)
    
    # Compute norms
    W_norms = W.norm(dim=1, keepdim=True)  # [num_outputs, 1]
    G_norms = G.norm(dim=1)  # [num_outputs]
    
    # Skip near-zero weights/gradients
    valid_mask = (W_norms.squeeze() > 1e-8) & (G_norms > 1e-8)
    if not valid_mask.any():
        return torch.tensor([])
    
    # Normalized weight directions
    W_hat = W / (W_norms + 1e-8)  # [num_outputs, features]
    
    # Radial gradient component: (G · W_hat) * W_hat
    radial_magnitude = (G * W_hat).sum(dim=1, keepdim=True)  # [num_outputs, 1]
    G_radial = radial_magnitude * W_hat  # [num_outputs, features]
    
    # Tangential component and ratio
    G_tangential = G - G_radial
    torque_ratios = G_tangential.norm(dim=1) / (G_norms + 1e-8)
    
    return torque_ratios[valid_mask]


# This is very expensive.
def find_dead_neurons(activations: Dict[str, torch.Tensor]) -> List[str]:
    """
    Find neurons that never activate across the batch.
    
    Args:
        activations: Dict mapping layer names to pre-activation tensors
        
    Returns:
        List of dead neuron identifiers (e.g., "layer1_42")
    """
    dead_neurons = []
    
    for layer_name, acts in activations.items():
        # Flatten spatial dimensions
        batch_size = acts.shape[0]
        flat_acts = acts.view(batch_size, -1)
        
        # Find neurons that are never positive
        never_active = (flat_acts <= 0).all(dim=0)
        dead_indices = never_active.nonzero(as_tuple=True)[0].tolist()
        
        # Create identifiers
        dead_neurons.extend(f"{layer_name}_{i}" for i in dead_indices)
    
    return dead_neurons