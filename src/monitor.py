# monitor.py - Prototype Surface Health Monitor for ReLU Networks

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple, Callable, Any
from dataclasses import dataclass, field
from models import Abs, Sum
import copy
import json
from pathlib import Path
from configs import ExperimentConfig


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
        hook_manager: Optional[SharedHookManager],
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
        if self.manager:
            return self.manager.activations
        return {}

    @property
    def forward_outputs(self) -> Dict[str, torch.Tensor]:
        """Provides direct access to the shared forward_outputs dictionary."""
        if self.manager:
            return self.manager.forward_outputs
        return {}

    @property
    def gradients(self) -> Dict[str, torch.Tensor]:
        """Provides direct access to the shared gradients dictionary."""
        if self.manager:
            return self.manager.gradients
        return {}

    @property
    def model(self) -> nn.Module:
        """Provides direct access to the model via the shared manager."""
        return self.manager.model

    def to(self, device: torch.device) -> None:
        if self.manager:
            self.manager.to(device)

    # === TRAINING LIFECYCLE HOOKS ===

    def start_run(self, run_id: int) -> None:
        pass

    def end_run(self, run_id: int, final_stats: Dict[str, Any]) -> None:
        pass

    def start_epoch(self, epoch: int) -> None:
        pass

    def end_epoch(self, epoch: int, epoch_loss: float) -> None:
        pass

    # === OPTIMIZATION STEP HOOKS ===

    def before_forward(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        pass

    def after_forward(
        self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int], outputs: torch.Tensor, loss: torch.Tensor
    ) -> None:
        pass

    def before_backward(self, loss: torch.Tensor) -> None:
        pass

    def after_backward(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        pass

    # === EXISTING CORE METHODS ===
    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        """
        Main monitoring check - called after backward pass.

        This is the primary monitoring method where most detection logic should go.
        Called after gradients are computed but before parameter updates.

        Args:
            x: Input batch
            y: Target batch
            batch_idx: Indices of samples in this batch

        Returns:
            True if training should continue normally, False if intervention needed
        """
        return True

    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        """
        Apply corrective intervention when check() returns False.

        Args:
            x: Input batch
            y: Target batch
            batch_idx: Indices of samples in this batch
        """
        pass

    def after_optimizer_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        """
        Called after the optimizer step (parameters have been updated).

        Args:
            x: Input batch
            y: Target batch
            batch_idx: Indices of samples in this batch
        """
        pass

    # === EARLY STOPPING HOOKS ===

    def should_stop_early(
        self, epoch: int, current_loss: float, best_loss: float, epochs_without_improvement: int
    ) -> bool:
        """
        Allow monitor to request early stopping based on custom criteria.

        Args:
            epoch: Current epoch number
            current_loss: Loss for current epoch
            best_loss: Best loss seen so far
            epochs_without_improvement: Number of epochs without improvement

        Returns:
            True if training should stop early, False otherwise
        """
        return False

    def on_early_stop(self, reason: str, epoch: int) -> None:
        """
        Called when early stopping is triggered (by any mechanism).

        Args:
            reason: Reason for early stopping (e.g., "loss_threshold", "convergence", "monitor_request")
            epoch: Epoch at which stopping occurred
        """
        pass


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
            dataset_size=dataset_size, classifier_threshold=classifier_threshold, hook_manager=hook_manager
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
        pre_activations = self.activations["activation_in"]
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
        has_zero_gradient_flow = gradient_flow_score == 0
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

        return True  # All checks passed

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
        print(f"👻 {self.step_count} Correcting persistently dead samples (global indices: {flagged_global_idx})")

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

    # 92% get 100% accuracy
    @torch.no_grad()
    def _fix_weight_nudge(self, xi: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        """Applies a minimal-norm weight nudge to a single dead sample."""
        zi = xi @ W.detach().T + b.detach()  # Pre-activations for this one sample

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
    """
    Monitors if the decision boundaries of neurons have drifted too far from a
    central point in the input space.

    This monitor calculates the orthogonal distance from a specified `origin` point
    to the hyperplane defined by each neuron in the first linear layer. If this
    distance exceeds a given `radius`, the monitor flags the neuron for
    intervention.

    This can be a form of regularization, preventing neurons from becoming
    unresponsive to data near the origin.
    """

    def __init__(
        self,
        hook_manager: SharedHookManager,
        dataset_size: int,
        radius: float,
        patience: int = 3,
        origin: torch.Tensor = None,
    ):
        """
        Initializes the BoundsMonitor.

        Args:
            hook_manager: The shared manager for hooks and captured data.
            dataset_size: The total number of samples in the dataset.
            radius: The maximum allowed distance from the origin to a neuron's
                    decision boundary.
            patience: The number of consecutive steps a neuron must be out of
                      bounds before it is flagged for intervention.
            origin: The reference point in the input space. If None, it defaults
                    to a zero vector with the same dimension as the model's input.
        """
        super().__init__(hook_manager, dataset_size)
        self.radius = radius
        self.patience = patience

        # Determine the number of neurons to monitor.
        num_neurons = self.model.linear1.out_features

        # Initialize a counter to track consecutive violations for each neuron.
        self.violation_counter = torch.zeros(num_neurons, dtype=torch.int32)

        # If no origin is provided, create a default zero vector.
        if origin is None:
            input_dimensionality = self.model.linear1.in_features
            self.origin = torch.zeros(input_dimensionality)
        else:
            self.origin = origin

        # A small constant to prevent division by zero.
        self.EPSILON = 1e-12

    @torch.no_grad()
    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        """
        Checks if any neuron's decision boundary has been outside the allowed
        radius for a duration longer than `patience`.

        The distance is calculated as: distance = |w·x_0 + b| / ||w||.

        Returns:
            True if all neurons are within bounds or patience has not been
            exceeded, False otherwise.
        """
        weights = self.model.linear1.weight  # Shape: (num_neurons, in_features)
        biases = self.model.linear1.bias  # Shape: (num_neurons,)
        origin = self.origin.to(weights.device)
        self.violation_counter = self.violation_counter.to(weights.device)

        # --- 1. Calculate Distances (same as before) ---
        hyperplane_eval_at_origin = weights @ origin + biases
        numerator = hyperplane_eval_at_origin.abs()
        weight_norms = weights.norm(dim=1) + self.EPSILON
        distances = numerator / weight_norms

        # --- 2. Update Violation Counters ---
        is_outside_radius = distances > self.radius

        # If a neuron is out of bounds, increment its counter.
        # Otherwise, reset its counter to zero.
        current_counts = self.violation_counter
        updated_counts = torch.where(is_outside_radius, current_counts + 1, 0)
        self.violation_counter = updated_counts

        # --- 3. Flag Neurons Exceeding Patience ---
        # A neuron is flagged only if its violation streak is greater than patience.
        needs_fixing = updated_counts > self.patience
        self.flagged_neurons = torch.nonzero(needs_fixing).flatten()

        # For debugging:
        # if self.flagged_neurons.numel() > 0:
        #      print(f"📏 Neuron(s) {self.flagged_neurons.tolist()} have been out of bounds "
        #            f"for {self.patience} steps.")

        self.step_count += 1
        return len(self.flagged_neurons) == 0

    @torch.no_grad()
    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        """
        Applies a corrective fix by resetting the bias of any flagged neuron
        and also resetting its violation counter.
        """
        if not self.flagged_neurons.numel():
            return  # No neurons to fix

        flagged_indices = self.flagged_neurons.tolist()

        print(
            f"🌐 {self.step_count} Correcting {len(flagged_indices)} neuron(s) "
            f"{flagged_indices} by resetting bias due to persistent bounds violation..."
        )

        for neuron_idx in flagged_indices:
            # Apply the fix by zeroing the bias.
            if self.model.linear1.bias is not None:
                self.model.linear1.bias[neuron_idx].zero_()

            # Reset the violation counter for the fixed neuron.
            self.violation_counter[neuron_idx] = 0


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

        if "linear1_weight" in self.gradients:
            weight_grad = self.gradients["linear1_weight"]
            bias_grad = self.gradients.get("linear1_bias", torch.zeros_like(weight_grad[:, 0]))
            print("\n[Per-Neuron Error Status]")
            for i in range(weight_grad.size(0)):
                w_norm = weight_grad[i].norm().item()
                b_val = bias_grad[i].item()
                print(f"Neuron {i}: ‖∇w‖ = {w_norm:.6f}, ∇b = {b_val:.6f}")

        # print("====================================\n")
        self.step_count += 1


class AnnealingMonitor(BaseMonitor):
    """
    Monitors the training process and applies Error-Driven Annealing.

    This monitor calculates the entropy of the per-example error distribution.
    If the entropy is low (indicating an unhealthy, "spiky" error distribution),
    it flags the state as uncertain. The `fix` method then injects scaled,
    random noise into the model parameters to "kick" the optimizer out of a
    potential local minimum.
    """

    def __init__(
        self,
        hook_manager: SharedHookManager,
        dataset_size: int,
        loss_fn: nn.Module,
        base_noise_level: float = 0.001,
        annealing_threshold: float = 0.1,
    ):
        """
        Initializes the AnnealingMonitor.

        Args:
            hook_manager: The shared manager for hooks and captured data.
            dataset_size: The total number of samples in the dataset.
            loss_fn: The loss function used to calculate per-example error.
            base_noise_level: A scaling factor for the injected noise.
            annealing_threshold: The uncertainty value above which noise is injected.
        """
        super().__init__(hook_manager, dataset_size)
        self.loss_fn = loss_fn
        self.base_noise_level = base_noise_level
        self.annealing_threshold = annealing_threshold

        # This will store the temperature calculated in `check` for use in `fix`
        self.error_magnitude: float = 0.0
        self.temperature: float = 0.0

        # Create a deep copy of the loss function to avoid side effects
        self._internal_loss_fn = copy.deepcopy(loss_fn)
        # Set its reduction to 'none' to get per-example losses
        self._internal_loss_fn.reduction = "none"

    def _calculate_uncertainty(self, per_example_loss: torch.Tensor) -> float:
        """Calculates the normalized uncertainty from per-example losses."""
        losses = per_example_loss.detach()
        if torch.sum(losses) <= 1e-9:
            return 0.0  # No error, no uncertainty

        batch_size = len(losses)
        if batch_size <= 1:
            return 0.0  # Entropy is not meaningful for a single item

        # Normalize losses into a probability distribution
        p = losses / torch.sum(losses)

        # Calculate actual entropy, adding epsilon for numerical stability
        h_actual = -torch.sum(p * torch.log2(p + 1e-9))

        # Calculate max entropy for a uniform distribution
        h_max = torch.log2(torch.tensor(batch_size, dtype=torch.float))

        # Return normalized uncertainty (0 to 1)
        uncertainty = (h_max - h_actual) / h_max
        return uncertainty.item()

    @torch.no_grad()
    def check(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        """
        Calculates the error entropy uncertainty. Returns False if it exceeds the
        threshold, signaling that a `fix` is required.
        """
        # The hook manager has already captured the model's output from the forward pass
        predictions = self.manager.model_output
        # print(f"y = {y}")
        # print(f"predictions = {predictions}")

        # Calculate per-example loss to get the error distribution
        per_example_loss = self._internal_loss_fn(predictions, y.unsqueeze(1))
        # print(f"per_example_loss = {per_example_loss}")

        # Calculate the "temperature" and store it for the `fix` method
        self.error_magnitude = torch.linalg.norm(per_example_loss, ord=2)
        self.temperature = self._calculate_uncertainty(per_example_loss)

        self.step_count += 1

        # If temperature is above threshold, return False to trigger an intervention
        return self.temperature <= self.annealing_threshold

    @torch.no_grad()
    def fix(self, x: torch.Tensor, y: torch.Tensor, batch_idx: Sequence[int]) -> None:
        """
        Injects scaled multiplicative random noise into all model parameters based on the
        temperature calculated during the `check` phase.
        """
        if self.temperature <= self.annealing_threshold:
            return  # Nothing to fix

        # msg = f"🔥 {self.step_count} Annealing temp {self.temperature:.4f} mag {self.error_magnitude:.4f}"
        # print(msg)

        for param in self.model.parameters():
            # 1. Generate standard Gaussian noise (mean=0, var=1)
            noise = torch.randn_like(param)

            # 2. Scale the noise by the base level and current temperature
            scaled_noise = noise * self.base_noise_level * self.error_magnitude * self.temperature * self.temperature

            # 3. Apply the additive noise to the parameter in-place
            param.add_(scaled_noise)


class CompositeMonitor(BaseMonitor):
    def __init__(self, monitors: Sequence[BaseMonitor]):
        if not monitors:
            raise ValueError("CompositeMonitor requires at least one monitor.")

        # ---- Intelligent Initialization ----
        # 1. We'll get the model and shared manager from the first monitor in the list.
        #    (We assume all monitors in the list share the same model and manager).
        manager = monitors[0].manager
        dataset_size = monitors[0].dataset_size  # Also needed for super()

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


class ParameterTraceMonitor(BaseMonitor):
    """
    Monitor that captures model parameters at each epoch and saves detailed
    training traces to JSON files.

    This monitor creates a comprehensive record of parameter evolution during
    training, useful for analyzing convergence dynamics, parameter trajectories,
    and training stability.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dataset_size: int,
        trace_subdir: str = "parameter_traces",
        save_frequency: int = 1,  # Save every N epochs
    ):
        """
        Initialize the parameter trace monitor.

        Args:
            hook_manager: Shared hook manager for model access
            dataset_size: Size of the training dataset
            output_dir: Base output directory for experiments
            experiment_name: Name of the current experiment
            trace_subdir: Subdirectory name for trace files
            save_frequency: Save parameters every N epochs (1 = every epoch)
        """
        super().__init__(hook_manager=None, dataset_size=dataset_size)

        self.config = config
        self.trace_subdir = trace_subdir
        self.save_frequency = save_frequency
        self.current_run_id = -1

        # Training data storage
        self.trace_data = {"metadata": {}, "epochs": []}

        # Track when we've captured initial state
        self.initial_state_captured = False

    def start_run(self, run_id: int) -> None:
        """Initialize trace data for a new run and capture initial model state."""
        super().start_run(run_id)
        self.current_run_id = run_id

        print("start_run")

        experiment_name = self.config.execution.experiment_name
        output_dir = self.config.execution.output_dir

        # Reset trace data for new run
        self.trace_data = {
            "metadata": {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "model_architecture": self.config.model.__class__.__name__,
                "parameter_count": sum(p.numel() for p in self.config.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.config.model.parameters() if p.requires_grad),
            },
            "epochs": [],
        }

        self.initial_state_captured = False

        # Create output directory if it doesn't exist
        self.trace_dir = Path(output_dir) / experiment_name / self.trace_subdir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        # Capture and save initial model state
        self._capture_initial_state()

    def _capture_initial_state(self) -> None:
        """Capture the initial state of the model at the start of training."""
        
        # Extract all parameters using the same logic as _capture_epoch_data
        initial_parameters = {}
        for name, param in self.config.model.named_parameters():
            # Convert to CPU and detach to avoid GPU memory issues
            param_cpu = param.detach().cpu()

            # Convert to nested lists for JSON serialization
            if param_cpu.dim() == 0:  # Scalar parameter
                initial_parameters[name] = param_cpu.item()
            elif param_cpu.dim() == 1:  # 1D parameter (bias)
                initial_parameters[name] = param_cpu.tolist()
            elif param_cpu.dim() == 2:  # 2D parameter (weight matrix)
                initial_parameters[name] = param_cpu.tolist()
            else:  # Higher dimensional parameters
                initial_parameters[name] = param_cpu.tolist()
        
        # Add initial state to trace data (JSON serializable)
        self.trace_data["initial_state"] = {
            "epoch": -1,  # Use -1 to indicate this is pre-training
            "parameters": initial_parameters
        }
        
        # Update metadata
        self.trace_data["metadata"]["initial_state_captured"] = True
        
        print(f"Initial state captured with {len(initial_parameters)} parameter groups")

    def end_epoch(self, epoch: int, epoch_loss: float) -> None:
        """Capture parameters at the end of each epoch."""
        # Save parameters at specified frequency
        if epoch % self.save_frequency == 0:
            self._capture_epoch_data(epoch, epoch_loss)

    def end_run(self, run_id: int, final_stats: Dict[str, Any]) -> None:
        """Finalize and save the trace data for this run."""
        # Update metadata with final statistics
        self.trace_data["metadata"].update(
            {
                "total_epochs": len(self.trace_data["epochs"]),
                "final_accuracy": final_stats.get("accuracy", 0.0),
                "final_loss": final_stats.get("final_loss", float("inf")),
                "best_loss": final_stats.get("best_loss", float("inf")),
                "training_time": final_stats.get("training_time", 0.0),
                "epochs_completed": final_stats.get("epochs_completed", 0),
            }
        )

        # Save to file
        self._save_trace_file(run_id)

    def _capture_epoch_data(self, epoch: int, loss: float) -> None:
        """Capture current model state."""
        # Extract all parameters
        parameters = {}
        for name, param in self.config.model.named_parameters():
            # Convert to CPU and detach to avoid GPU memory issues
            param_cpu = param.detach().cpu()

            # Convert to nested lists for JSON serialization
            if param_cpu.dim() == 0:  # Scalar parameter
                parameters[name] = param_cpu.item()
            elif param_cpu.dim() == 1:  # 1D parameter (bias)
                parameters[name] = param_cpu.tolist()
            elif param_cpu.dim() == 2:  # 2D parameter (weight matrix)
                parameters[name] = param_cpu.tolist()
            else:  # Higher dimensional parameters
                parameters[name] = param_cpu.tolist()

        # Create epoch record
        epoch_data = {"epoch": epoch, "loss": float(loss), "parameters": parameters}

        self.trace_data["epochs"].append(epoch_data)

    def _save_trace_file(self, run_id: int) -> None:
        """Save trace data to JSON file."""
        filename = f"run_{run_id:03d}_trace.json"
        filepath = self.trace_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(self.trace_data, f, indent=2)

            print(f"📊 Saved parameter trace: {filepath}")

        except Exception as e:
            print(f"❌ Failed to save parameter trace {filepath}: {e}")

    @staticmethod
    def load_trace_file(filepath: str) -> Dict[str, Any]:
        """Load a parameter trace file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_experiment_traces(experiment_dir: str, trace_subdir: str = "parameter_traces") -> List[Dict[str, Any]]:
        """Load all trace files from an experiment directory."""
        trace_dir = Path(experiment_dir) / trace_subdir
        
        if not trace_dir.exists():
            return []
        
        traces = []
        for trace_file in sorted(trace_dir.glob("run_*_trace.json")):
            try:
                traces.append(ParameterTraceMonitor.load_trace_file(trace_file))
            except Exception as e:
                print(f"Warning: Failed to load {trace_file}: {e}")
        
        return traces

    @staticmethod
    def extract_parameter_trajectory(trace_data: Dict[str, Any], param_name: str) -> List[Any]:
        """Extract the trajectory of a specific parameter across epochs."""
        trajectory = []
        for epoch_data in trace_data["epochs"]:
            if param_name in epoch_data["parameters"]:
                trajectory.append(epoch_data["parameters"][param_name])
        return trajectory

    @staticmethod
    def extract_loss_trajectory(trace_data: Dict[str, Any]) -> List[float]:
        """Extract the loss trajectory across epochs."""
        return [epoch_data["loss"] for epoch_data in trace_data["epochs"]]