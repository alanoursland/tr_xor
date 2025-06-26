import torch
import torch.nn as nn

from typing import Dict, List, Tuple, Any

def get_linear_layers(model: nn.Module) -> Dict[str, nn.Linear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

def extract_activation_type(model: torch.nn.Module) -> str:
    """
    Extract the primary activation type from a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        String identifying the activation type ('relu', 'abs', 'sigmoid', etc.)
    """
    model_name = type(model).__name__.lower()
    
    if 'abs' in model_name:
        return 'abs'
    elif 'relu' in model_name:
        return 'relu'
    elif 'sigmoid' in model_name:
        return 'sigmoid'
    elif 'tanh' in model_name:
        return 'tanh'
    else:
        # Try to inspect the model for activation functions
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                return 'relu'
            elif isinstance(module, torch.nn.Sigmoid):
                return 'sigmoid'
            elif isinstance(module, torch.nn.Tanh):
                return 'tanh'
            # Check for custom activations
            elif hasattr(module, 'forward') and 'abs' in str(module.forward).lower():
                return 'abs'
    
    return 'unknown'

def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def analyze_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Analyze model state dictionary for key properties.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary of model analysis results
    """
    analysis = {
        'parameter_count': 0,
        'layer_count': 0,
        'weight_statistics': {},
        'bias_statistics': {}
    }
    
    weight_tensors = []
    bias_tensors = []
    
    for name, tensor in state_dict.items():
        analysis['parameter_count'] += tensor.numel()
        
        if 'weight' in name:
            analysis['layer_count'] += 1
            weight_tensors.append(tensor)
        elif 'bias' in name:
            bias_tensors.append(tensor)
    
    # Analyze weights
    if weight_tensors:
        all_weights = torch.cat([w.flatten() for w in weight_tensors])
        analysis['weight_statistics'] = {
            'mean': all_weights.mean().item(),
            'std': all_weights.std().item(),
            'min': all_weights.min().item(),
            'max': all_weights.max().item(),
            'norm': torch.norm(all_weights).item()
        }
    
    # Analyze biases
    if bias_tensors:
        all_biases = torch.cat([b.flatten() for b in bias_tensors])
        analysis['bias_statistics'] = {
            'mean': all_biases.mean().item(),
            'std': all_biases.std(unbiased=False).item(),
            'min': all_biases.min().item(),
            'max': all_biases.max().item(),
            'norm': torch.norm(all_biases).item()
        }
    
    # Detect potential issues
    analysis['potential_issues'] = detect_model_issues(state_dict, analysis)
    
    return analysis

def detect_model_issues(state_dict: Dict[str, torch.Tensor], analysis: Dict[str, Any]) -> List[str]:
    """
    Detect potential issues in trained model.
    
    Args:
        state_dict: Model state dictionary
        analysis: Model analysis results
        
    Returns:
        List of detected issues
    """
    issues = []
    
    # Check for exploding gradients
    if 'weight_statistics' in analysis and analysis['weight_statistics']:
        weight_max = abs(analysis['weight_statistics']['max'])
        weight_norm = analysis['weight_statistics']['norm']
        
        if weight_max > 10.0:
            issues.append(f"Large weight values detected (max: {weight_max:.2f})")
        
        if weight_norm > 100.0:
            issues.append(f"Large weight norm detected ({weight_norm:.2f})")
    
    # Check for vanishing gradients
    if 'weight_statistics' in analysis and analysis['weight_statistics']:
        weight_std = analysis['weight_statistics']['std']
        if weight_std < 1e-6:
            issues.append(f"Very small weight variations (std: {weight_std:.2e})")
    
    # Check for dead neurons (all weights near zero)
    for name, tensor in state_dict.items():
        if 'weight' in name:
            if torch.norm(tensor) < 1e-6:
                issues.append(f"Potential dead neuron in {name}")
    
    return issues

def targets_to_class_labels(y):
    if y.ndim == 2 and y.shape[1] > 1:
        # Convert one-hot classification to labels
        y_indices = torch.argmax(y, dim=1)
    else:
        # Convert binary thresholding to labels
        y_indices = y.long()
    return y_indices

