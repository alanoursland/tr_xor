# monitor.py - Prototype Surface Health Monitor for ReLU Networks

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple, Callable
from dataclasses import dataclass, field
from models import Abs, Sum

class SharedHookManager:
    """
    Manages the registration of hooks and storage of captured data for a model.
    This object is intended to be shared by one or more monitors to prevent
    redundant hook registration.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.forward_outputs: Dict[str, torch.Tensor] = {}

        self.model_output: Optional[torch.Tensor] = None
        self.output_gradient: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        # --- Standard hooks for Linear and ReLU layers ---
        for name, module in self.model.named_modules():
            # Note: A more robust implementation might check for any activation
            # function, not just ReLU. For now, this is fine.
            if isinstance(module, (nn.ReLU, Abs, Sum)):
                module.register_forward_hook(self._make_forward_hook(name))
            elif isinstance(module, nn.Linear):
                module.weight.register_hook(self._make_grad_hook(name + "_weight"))
                if module.bias is not None:
                    module.bias.register_hook(self._make_grad_hook(name + "_bias"))

        # --- Automatic hook for the final model output ---
        # Heuristic: The last module in the sequence is the one producing the output.
        last_module = list(self.model.modules())[-1]
        last_module.register_forward_hook(self._make_output_capture_hook())

    def _make_forward_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            # Use a consistent naming scheme
            self.activations[name + "_in"] = input[0].detach().clone()
            self.forward_outputs[name + "_out"] = output.detach().clone()
        return hook

    def _make_grad_hook(self, name: str) -> Callable:
        def hook(grad):
            self.gradients[name] = grad.detach().clone()
        return hook
        
    def _make_output_capture_hook(self) -> Callable:
        """
        Creates a forward hook that captures the model's final output tensor
        and attaches a gradient hook to it.
        """
        def hook(module, input, output):
            # 1. Capture the prediction tensor (the output of the last layer)
            self.model_output = output.detach().clone()
            
            # 2. Attach the gradient hook to this specific output tensor
            def grad_hook(grad):
                self.output_gradient = grad.detach().clone()
            # Ensure the output requires a gradient before attaching the hook
            if output.requires_grad:
                output.register_hook(grad_hook)
        return hook

    def to(self, device: torch.device) -> None:
        """Moves all cached tensors to the specified device."""
        def move_dict(d: Dict[str, torch.Tensor]):
            for k in d:
                if d[k] is not None:
                    d[k] = d[k].to(device)
        
        move_dict(self.activations)
        move_dict(self.forward_outputs)
        move_dict(self.gradients)
        
        # Handle the tensors directly, checking if they exist first.
        if self.model_output is not None:
            self.model_output = self.model_output.to(device)
        if self.output_gradient is not None:
            self.output_gradient = self.output_gradient.to(device)
    
    def clear(self):
        """Clears all stored data for the next training step."""
        self.activations.clear()
        self.forward_outputs.clear()
        self.gradients.clear()

        # --- CORRECTED ---
        # To "clear" a tensor, set its reference to None.
        self.model_output = None
        self.output_gradient = None

class BaseMonitor:
    """
    BaseMonitor provides shared infrastructure for monitoring neural network
    training dynamics. It supports:

    - Automatic registration of forward and gradient hooks for ReLU and Linear layers
    - Storage of per-layer activations and gradients
    - Easy transfer of cached tensors between devices

    Derived monitors can extend this class to implement specific training diagnostics
    or interventions such as dead neuron detection or sample-level gradient tracking.

    Subclasses should override:
        • check(x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) → bool
        • fix(x: Tensor, y: Tensor) → None
    """
    def __init__(
        self,
        hook_manager: SharedHookManager,
        dataset_size: int,
        classifier_threshold: float = 0.5,
    ):  
        self.manager = hook_manager
        self.step_count = 0
        self.dataset_size = dataset_size
        self.classifier_threshold = classifier_threshold

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        """Provides direct access to the shared activations dictionary."""
        return self.manager.activations

    @property
    def forward_outputs(self) -> Dict[str, torch.Tensor]:
        """Provides direct access to the shared forward_outputs dictionary."""
        return self.manager.forward_outputs

    @property
    def gradients(self) -> Dict[str, torch.Tensor]:
        """Provides direct access to the shared gradients dictionary."""
        return self.manager.gradients

    @property
    def model(self) -> nn.Module:
        """Provides direct access to the model via the shared manager."""
        return self.manager.model

    def to(self, device: torch.device) -> None:
        self.manager.to(device)

    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        """
        Subclasses should override this method to implement custom monitoring logic.
        Returns True if training should continue normally, or False to trigger intervention.
        """
        raise NotImplementedError("Subclasses must implement the check() method.")

    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        """
        Subclasses should override this method to implement corrective intervention
        when a monitored condition is violated.
        """
        raise NotImplementedError("Subclasses must implement the fix() method.")

class DeadSampleMonitor(BaseMonitor):
    """
    Detects samples that are both misclassified and receive zero gradient 
    flow through all hidden neurons (i.e., "dead-and-wrong") for multiple epochs.

    Flags these early-training failures and allows targeted correction.

    Inherits:
        - Hook and activation plumbing from BaseMonitor
        - check() and fix() methods should be implemented for active behavior
    """
    def __init__(
        self,
        hook_manager: SharedHookManager,
        dataset_size: int,
        patience: int = 3,
        classifier_threshold: float = 0.5,
    ):
        super().__init__(
            dataset_size=dataset_size,
            classifier_threshold=classifier_threshold,
            hook_manager = hook_manager
        )
        self.pre_activation_key = f"activation_in"
        self.patience = patience
        self.dead_counter = torch.zeros(dataset_size, dtype=torch.int32)

    # ---------------------------------------------------------------
    #  ---- EARLY-FAILURE HEURISTIC IMPLEMENTATION ------------------
    # ---------------------------------------------------------------
    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        """
        Checks for samples that are both misclassified and have zero gradient flow.
        
        This method identifies "dead-and-wrong" samples by:
        1.  Calculating a gradient-flow score for each sample. This score is zero
            if all pre-ReLU activations for a sample are negative, indicating no
            gradient will pass through the layer.
        2.  Identifying which samples are misclassified based on the model's output.
        3.  Tracking a "dead streak" counter for each sample in the dataset.
        4.  Flagging samples whose streak of being "dead-and-wrong" exceeds a
            predefined `patience` threshold.

        Args:
            x: The data points for the current batch.
            y: The ground truth labels for the current batch.
            batch_idx: The global indices of the samples in the current batch.

        Returns:
            bool: True if no samples are flagged, False if an intervention is needed.
        """
        # --- 1. Retrieve captured tensors from the hook manager ---
        # (B, H) - Pre-activation values for the first hidden layer
        pre_activations = self.activations["activation1_in"]
        # (B,)   - Final model predictions for the batch
        predictions = self.manager.model_output.squeeze(1)

        # --- 2. Identify misclassified samples ---
        # Convert predictions to binary classes (0 or 1)
        predicted_classes = (predictions > self.classifier_threshold).int()
        is_misclassified = predicted_classes.ne(y.int())

        # --- 3. Calculate the gradient-flow score for each sample ---
        # This score estimates how much gradient can flow back through the ReLU neurons.
        # Note: This manually infers the gradient for Mean Squared Error loss.
        # dL/dŷ = 2 * (ŷ - y) / B, where B is batch size.
        manual_mse_gradient = 2.0 / predictions.size(0) * (predictions - y)
        
        # ReLU passes gradient only if its input (pre_activation) is > 0.
        active_neurons_mask = (pre_activations > 0).float()
        
        # The score is the sum of absolute gradients flowing through active neurons.
        # A score of 0 means no gradient can flow back for that sample.
        gradient_flow_score = active_neurons_mask.mul(manual_mse_gradient.abs().unsqueeze(1)).sum(dim=1)

        # --- 4. Identify samples that are both "dead" and "misclassified" ---
        has_zero_gradient_flow = (gradient_flow_score == 0)
        is_dead_and_misclassified = has_zero_gradient_flow & is_misclassified

        # --- 5. Update the streak counter for the entire dataset ---
        # Move necessary tensors to the CPU to update the counter
        batch_idx_cpu = torch.as_tensor(batch_idx, dtype=torch.long)
        dead_and_wrong_cpu = is_dead_and_misclassified.cpu()
        
        # For samples in the current batch:
        # - If dead-and-wrong, increment their counter.
        # - Otherwise, reset their counter to zero.
        current_counts = self.dead_counter[batch_idx_cpu]
        updated_counts = torch.where(dead_and_wrong_cpu, current_counts + 1, 0)
        self.dead_counter[batch_idx_cpu] = updated_counts

        # --- 6. Check if any sample's streak has exceeded patience ---
        flagged_samples_in_batch = torch.nonzero(updated_counts > self.patience).flatten()
        
        self.step_count += 1

        if len(flagged_samples_in_batch) > 0:
            # Retrieve the global indices of the flagged samples for logging
            flagged_global_idx = batch_idx_cpu[flagged_samples_in_batch].tolist()
            # print(f"⚠️ Early-failure detector: dead samples {flagged_global_idx} "
            #       f" (streak > {self.patience} epochs)")
            return False  # Signal that an intervention is required

        return True # All checks passed

    @torch.no_grad()
    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]):
        """
        Identifies samples that have been flagged for intervention and applies a
        corrective nudge.

        This method:
        1.  Uses the `dead_counter` to find which samples in the current batch have
            exceeded the `patience` threshold.
        2.  For each flagged sample, it calls a specialized fix function (hardcoded
            for now).
        3.  Resets the `dead_counter` for the corrected samples to 0.
        """
        # --- 1. Identify which samples in THIS BATCH need fixing ---
        batch_idx_cpu = torch.as_tensor(batch_idx, dtype=torch.long)
        
        # Get the streak count for each sample in the current batch
        batch_dead_counts = self.dead_counter[batch_idx_cpu]
        
        # Find the *local* indices within the batch of samples that need fixing
        local_idx_to_fix = torch.nonzero(batch_dead_counts > self.patience).flatten()

        if len(local_idx_to_fix) == 0:
            return  # No samples in this batch require fixing

        # --- 2. Prepare for the fix ---
        flagged_global_idx = batch_idx_cpu[local_idx_to_fix].tolist()
        print(f"⚠️ Correcting persistently dead samples (global indices: {flagged_global_idx})")

        model = self.model
        W = model.linear1.weight
        b = model.linear1.bias

        # --- 3. Apply the chosen fix to each flagged sample ---
        for i in local_idx_to_fix:
            # Get the specific input tensor for the one sample we are fixing
            sample_input = x[i]
            
            # --- Call the hardcoded, specialized fix method ---
            self._fix_weight_nudge(sample_input, W, b)
            # To use a different fix, you would change the line above, e.g.:
            # self._fix_noise_injection(sample_input, W, b)

        # --- 4. Reset the counter for the samples that were just fixed ---
        self.dead_counter[flagged_global_idx] = 0
        print(f"   ➤ Reset patience counter for samples {flagged_global_idx}.")

    # 92% get 100% accuracy
    @torch.no_grad()
    def _fix_weight_nudge(self, xi: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        """Applies a minimal-norm weight nudge to a single dead sample."""
        zi = xi @ W.detach().T + b.detach() # Pre-activations for this one sample
        
        # Find the neuron closest to activating for this sample
        norms = W.detach().norm(dim=1) + 1e-12
        distances = zi / norms
        closest_neuron_idx = torch.argmin(distances.abs()).item()

        # Calculate the minimal update to activate that neuron
        z_closest = zi[closest_neuron_idx].item()
        norm_x_sq = xi @ xi + 1e-12
        eps = 1e-4  # Target activation value

        alpha = max(0.0, eps - z_closest) / norm_x_sq
        delta_w = alpha * xi
        
        # Apply the fix
        W[closest_neuron_idx] += delta_w

    # 100% get 100% accuracy
    @torch.no_grad()
    def _fix_noise_injection(self, xi: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        """Injects scaled random noise to nudge a single dead sample."""
        zi = xi @ W.detach().T + b.detach()
        norms = W.detach().norm(dim=1) + 1e-12
        distances = zi / norms
        closest_neuron_idx = torch.argmin(distances.abs()).item()
        
        z_closest = zi[closest_neuron_idx].item()
        norm_x = xi.norm().item() + 1e-12
        eps = 1e-4

        projected_deficit = eps - z_closest
        noise_magnitude = projected_deficit / norm_x
        
        noise_direction = torch.randn_like(xi)
        noise_direction = noise_direction / (noise_direction.norm() + 1e-12)
        delta_w = noise_magnitude * noise_direction

        W[closest_neuron_idx] += delta_w

    # 80% get 100% accuracy
    @torch.no_grad()
    def _fix_bias_nudge(self, xi: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        """Applies a bias nudge to a single dead sample."""
        zi = xi @ W.detach().T + b.detach()
        norms = W.detach().norm(dim=1) + 1e-12
        distances = zi / norms
        closest_neuron_idx = torch.argmin(distances.abs()).item()

        # Nudge the bias just enough to make the pre-activation positive
        delta_b = -zi[closest_neuron_idx].item() + 1e-4
        b[closest_neuron_idx] += delta_b

    # 98% get 100% accuracy
    @torch.no_grad()
    def _fix_weight_bias_nudge(self, xi: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        """Applies a joint weight and bias nudge to a single dead sample."""
        zi = xi @ W.detach().T + b.detach()
        norms = W.detach().norm(dim=1) + 1e-12
        distances = zi / norms
        closest_neuron_idx = torch.argmin(distances.abs()).item()

        # Augmented input: [x; 1] for joint update
        x_aug = torch.cat([xi, torch.ones(1, device=xi.device)])
        norm_x_aug_sq = x_aug @ x_aug + 1e-12

        z_closest = zi[closest_neuron_idx].item()
        eps = 1e-4
        delta_needed = max(0.0, eps - z_closest)

        # Calculate the minimal joint update vector
        delta_theta = (delta_needed / norm_x_aug_sq) * x_aug
        delta_w = delta_theta[:-1]
        delta_b = delta_theta[-1].item()

        # Apply the fix
        W[closest_neuron_idx] += delta_w
        b[closest_neuron_idx] += delta_b

class BoundsMonitor(BaseMonitor):
    def __init__(
        self,
        hook_manager: SharedHookManager,
        dataset_size: int,
        radius: float,
        origin: torch.Tensor = None,
    ):
        super().__init__(hook_manager, dataset_size)
        self.radius = radius
        self.origin = origin if origin is not None else torch.zeros(hook_manager.model.linear1.in_features)

    @torch.no_grad()
    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        W = self.model.linear1.weight.detach()  # (H, D)
        b = self.model.linear1.bias.detach()    # (H,)
        origin = self.origin.to(W.device)       # (D,)

        num = W @ origin + b                    # (H,)
        denom = W.norm(dim=1) + 1e-12           # (H,)
        dists = num.abs() / denom               # (H,)

        mask = dists > self.radius
        self.flagged_neurons = torch.nonzero(mask, as_tuple=False).flatten()

        # for j in self.flagged_neurons.tolist():
        #     print(f"⚠️  Neuron {j} outside radius: distance = {dists[j]:.4f} > {self.radius:.4f}")

        return len(self.flagged_neurons) == 0

    @torch.no_grad()
    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        # print(f"⚠️    Zeroing node biases due to bounds violation...")
        for j in self.flagged_neurons.tolist():
            if self.model.linear1.bias is not None:
                self.model.linear1.bias[j].zero_()

class LoggingMonitor(BaseMonitor):
    """
    LoggingMonitor extends BaseMonitor with additional diagnostic state
    or logic as needed (currently replicates DeadSampleMonitor’s init
    structure but may be specialized for logging purposes).
    """
    def __init__(
        self,
        hook_manager: SharedHookManager,
        dataset_size: int,
        patience: int = 3,
        classifier_threshold: float = 0.5,
    ):
        super().__init__(
            hook_manager,
            dataset_size=dataset_size,
            classifier_threshold=classifier_threshold, 
        )
        self.patience = patience
        self.dead_counter = torch.zeros(dataset_size, dtype=torch.int32)

    # ---------------------------------------------------------------
    #  --------- pretty printing as before --------------------------
    # ---------------------------------------------------------------
    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]):
        print(f"\n=== Step {self.step_count} Monitor Report ===")

        if self.activations:
            print("\n[Activations]")
            for name, act in self.activations.items():
                print(f"→ {name}: {act.shape}")
                print(act)

        if self.forward_outputs:
            print("\n[Forward Outputs (post-ReLU)]")
            for name, out in self.forward_outputs.items():
                print(f"→ {name}: {out.shape}")
                print(out)

        if self.gradients:
            print("\n[Gradients]")
            for name, grad in self.gradients.items():
                print(f"→ {name}: {grad.shape}")
                print(grad)

        print("\n[Weights and Biases]")
        for name, param in self.model.named_parameters():
            flat = param.detach().view(-1).tolist()
            compact = ", ".join(f"{v:+.4f}" for v in flat)
            print(f"→ {name}: [{compact}]")

        if 'linear1_weight' in self.gradients:
            weight_grad = self.gradients['linear1_weight']
            bias_grad   = self.gradients.get('linear1_bias',
                                             torch.zeros_like(weight_grad[:, 0]))
            print("\n[Per-Neuron Error Status]")
            for i in range(weight_grad.size(0)):
                w_norm = weight_grad[i].norm().item()
                b_val  = bias_grad[i].item()
                print(f"Neuron {i}: ‖∇w‖ = {w_norm:.6f}, ∇b = {b_val:.6f}")

        # print("====================================\n")
        self.step_count += 1

# In monitor.py

class CompositeMonitor(BaseMonitor):
    def __init__(self, monitors: Sequence[BaseMonitor]):
        if not monitors:
            raise ValueError("CompositeMonitor requires at least one monitor.")

        # ---- Intelligent Initialization ----
        # 1. We'll get the model and shared manager from the first monitor in the list.
        #    (We assume all monitors in the list share the same model and manager).
        manager = monitors[0].manager
        dataset_size = monitors[0].dataset_size # Also needed for super()

        # 2. Now, properly initialize itself as a BaseMonitor.
        super().__init__(
            hook_manager=manager,
            dataset_size=dataset_size,
        )

        # 3. Store the list of monitors it will delegate to.
        self.monitors = monitors

    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        # Delegate the 'check' call to each child monitor.
        results = [m.check(x, y, batch_idx) for m in self.monitors]
        return all(results)

    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        # Delegate the 'fix' call to each child monitor.
        for m in self.monitors:
            m.fix(x, y, batch_idx)

    def to(self, device: torch.device):
        # The parent 'to' method already calls manager.to(device),
        # so we don't need to do anything extra here. The shared state
        # is handled correctly.
        super().to(device)