# monitor.py - Prototype Surface Health Monitor for ReLU Networks

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple, Callable
from dataclasses import dataclass, field

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
        • check(targets: Tensor, batch_idx: Sequence[int]) → bool
        • fix(x: Tensor, y: Tensor) → None
    """
    def __init__(
        self,
        model: nn.Module,
        dataset_size: int,
        classifier_threshold: float = 0.5,
    ):
        self.model = model
        self.step_count = 0
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.forward_outputs: Dict[str, torch.Tensor] = {}

        self.dataset_size = dataset_size
        self.classifier_threshold = classifier_threshold

        self._register_hooks()

    def to(self, device: torch.device) -> None:
        def move_dict(d: Dict[str, torch.Tensor]):
            for k in d:
                d[k] = d[k].to(device)
        move_dict(self.activations)
        move_dict(self.forward_outputs)
        move_dict(self.gradients)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(self._make_forward_hook(name + "_relu_out"))
            elif isinstance(module, nn.Linear):
                module.weight.register_hook(self._make_grad_hook(name + "_weight"))
                if module.bias is not None:
                    module.bias.register_hook(self._make_grad_hook(name + "_bias"))

    def _make_forward_hook(self, name):
        def hook(module, input, output):
            self.activations[name + "_in"]  = input[0].detach().clone()
            self.forward_outputs[name + "_out"] = output.detach().clone()
        return hook

    def _make_grad_hook(self, name):
        def hook(grad):
            self.gradients[name] = grad.detach().clone()
        return hook

    def check(self, targets: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        """
        Subclasses should override this method to implement custom monitoring logic.
        Returns True if training should continue normally, or False to trigger intervention.
        """
        raise NotImplementedError("Subclasses must implement the check() method.")

    def fix(self, x: torch.Tensor, y: torch.Tensor) -> None:
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
        model: nn.Module,
        dataset_size: int,
        patience: int = 3,
        classifier_threshold: float = 0.5,
    ):
        super().__init__(
            model=model,
            dataset_size=dataset_size,
            classifier_threshold=classifier_threshold,
        )
        self.patience = patience
        self.dead_counter = torch.zeros(dataset_size, dtype=torch.int32)

    # ---------------------------------------------------------------
    #  ---- EARLY-FAILURE HEURISTIC IMPLEMENTATION ------------------
    # ---------------------------------------------------------------
    @torch.no_grad()
    def update_failure_detector(
        self,
        z: torch.Tensor,              # (B, H) pre-ReLU activations
        dL_dy: torch.Tensor,          # (B,)   per-sample dL/dŷ
        preds: torch.Tensor,          # (B,)   raw model outputs
        targets: torch.Tensor,        # (B,)   true labels   (0/1)
        batch_idx: Sequence[int],     # global sample indices
    ):
        """
        Call once per mini-batch *before* backward().
        Works for binary or scalar regression outputs.
        """
        result = True
        # gradient-flow score S_k  (ReLU passes grad only if z>0)
        S = (z > 0).float().mul_(dL_dy.abs().unsqueeze(1)).sum(dim=1)  # (B,)

        wrong = (preds > self.classifier_threshold).int().ne_(targets.int())
        dead_and_wrong = (S == 0) & wrong

        idx = torch.as_tensor(batch_idx, device=S.device)
        cpu_idx = idx.cpu()
        cpu_dw  = dead_and_wrong.cpu()

        # update streak counter
        self.dead_counter[cpu_idx] = torch.where(
            cpu_dw.bool(),
            self.dead_counter[cpu_idx] + 1,
            torch.zeros_like(self.dead_counter[cpu_idx]),
        )

        # flag any sample whose streak >= patience
        flagged = torch.nonzero(self.dead_counter[cpu_idx] > self.patience).flatten()
        if len(flagged):
            result = False
            ids = cpu_idx[flagged].tolist()
            # print(f"⚠️\tEarly-failure detector: dead samples {ids} "
            #       f"(>{self.patience} epochs) "
            #     #   f"{self.dead_counter[cpu_idx]} "
            #       )
        return result

    # ---------------------------------------------------------------
    #  --------- pretty printing as before --------------------------
    # ---------------------------------------------------------------
    def check(self, targets: torch.Tensor, batch_idx: Sequence[int]):
        z = self.activations["relu1_relu_out_in"]      # (B, H)
        a = self.forward_outputs["relu1_relu_out_out"] # (B, H)
        preds = a.sum(dim=1)                           # (B,)

        # Infer dL/dŷ for MSE manually
        dL_dy = 2.0 / preds.size(0) * (preds - targets)

        result = self.update_failure_detector(z, dL_dy, preds, targets, batch_idx)

        self.step_count += 1
        return result

    @torch.no_grad()
    def fix(self, x: torch.Tensor, y: torch.Tensor):
        self.fix_weight_nudge(x, y)

    # 92% get 100% accuracy
    @torch.no_grad()
    def fix_noise_injection(self, x: torch.Tensor, y: torch.Tensor):
        """
        For each dead-and-wrong sample, inject scaled random noise into
        the closest neuron's weight vector to help activate the sample.
        """
        model = self.model
        W = model.linear1.weight          # (H, D)
        b = model.linear1.bias            # (H,)
        norms = W.detach().norm(dim=1)    # (H,)

        z = x @ W.detach().T + b.detach() # (B, H)
        preds = self.forward_outputs["relu1_relu_out_out"].sum(dim=1)
        wrong = (preds > self.classifier_threshold).int() != y.int()

        dead_mask = (z <= 0).all(dim=1) & wrong
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()

        if len(dead_indices) == 0:
            print("✔️  No dead-and-wrong samples to fix (noise injection).")
            return

        for i in dead_indices:
            xi = x[i]                     # (D,)
            zi = z[i]                     # (H,)
            di = zi / norms               # signed scaled distances
            closest = torch.argmin(di.abs()).item()

            zj = zi[closest].item()
            norm_x = xi.norm().item() + 1e-12  # avoid div/0
            eps = 1e-4

            projected_deficit = eps - zj
            noise_magnitude = projected_deficit / norm_x

            noise_direction = torch.randn_like(xi)
            noise_direction = noise_direction / (noise_direction.norm() + 1e-12)
            delta_w = noise_magnitude * noise_direction

            W[closest] += delta_w

            print(f"🧪 Dead sample {i.item()} (label = {y[i].item()})")
            # print(f"    Input x = {xi.tolist()}")
            # print(f"    Pre-activations z = {zi.tolist()}")
            # print(f"    ➤ Closest neuron index: {closest} (|distance| = {di[closest].abs().item():.4f})")
            print(f"      Injected noise: ∥Δw∥ = {delta_w.norm().item():.5f}")
            # print(f"      Δw = {delta_w.tolist()}")

    # 100% get 100% accuracy
    @torch.no_grad()
    def fix_weight_nudge(self, x: torch.Tensor, y: torch.Tensor):
        """
        For each flagged dead-and-wrong sample, compute and print:
        • distance to each neuron's ReLU hyperplane (z = wᵀx + b)
        • index of the closest neuron
        • minimal-norm update to weights to activate the dead point
        """
        model = self.model
        W = model.linear1.weight          # (H, D)
        b = model.linear1.bias            # (H,)
        norms = W.detach().norm(dim=1)    # (H,)

        z = x @ W.detach().T + b.detach() # (B, H)
        preds = self.forward_outputs["relu1_relu_out_out"].sum(dim=1)
        wrong = (preds > self.classifier_threshold).int() != y.int()

        dead_mask = (z <= 0).all(dim=1) & wrong
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()

        if len(dead_indices) == 0:
            return
        print(f"⚠️    Correcting dead samples {[i.item() for i in dead_indices]}")
        for i in dead_indices:
            xi = x[i]                     # (D,)
            zi = z[i]                     # (H,)
            di = zi / norms               # signed scaled distances
            closest = torch.argmin(di.abs()).item()

            zj = zi[closest].item()
            norm_x2 = xi @ xi + 1e-12     # avoid divide-by-zero
            eps = 1e-4                    # small positive activation threshold

            # Calculate the minimal weight update vector
            alpha = max(0.0, eps - zj) / norm_x2
            delta_w = alpha * xi         # shape (D,)
            W[closest] += delta_w

            # print(f"🧠  Dead sample {i.item()} (label = {y[i].item()})")
            # print(f"    Input x = {xi.tolist()}")
            # print(f"    Pre-activations z = {zi.tolist()}")
            # print(f"    Distances to ReLU hyperplanes = {di.tolist()}")
            # print(f"    ➤ Closest neuron index: {closest} (|distance| = {di[closest].abs().item():.4f})")
            # print(f"      Projection patch: nudging weight[{closest}] by {delta_w.tolist()}")

    # 80% get 100% accuracy
    @torch.no_grad()
    def fix_bias_nudge(self, x: torch.Tensor, y: torch.Tensor):
        W = self.model.linear1.weight.detach()   # (H, D)
        b = self.model.linear1.bias.detach()     # (H,)
        norms = W.norm(dim=1)                    # (H,)

        z = x @ W.T + b                          # (B, H)
        preds = self.forward_outputs["relu1_relu_out_out"].sum(dim=1)  # (B,)
        wrong = (preds > self.classifier_threshold).int() != y.int()

        dead_mask = (z <= 0).all(dim=1) & wrong  # (B,)
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()

        if len(dead_indices) == 0:
            print("✔️  No dead-and-wrong samples to fix.")
            return

        for i in dead_indices:
            xi = x[i]
            zi = z[i]               # (H,)
            di = zi / norms         # signed scaled distances
            closest = torch.argmin(di.abs())
            delta = -z[i, closest].item() + 1e-4   # just above zero
            self.model.linear1.bias[closest] += delta

            print(f"  Dead sample {i.item()} (label = {y[i].item()})")
            print(f"    Input x = {xi.tolist()}")
            print(f"    Pre-activations z = {zi.tolist()}")
            print(f"    Distances to ReLU hyperplanes = {di.tolist()}")
            print(f"    Closest neuron index: {closest.item()} (|distance| = {di[closest].abs().item():.4f})")
            print(f"      Nudging bias[{closest}] by {delta:.5f} to activate sample.")

    # 98% get 100% accuracy
    @torch.no_grad()
    def fix_weight_bias_nudge(self, x: torch.Tensor, y: torch.Tensor):
        """
        For each dead-and-wrong sample, compute the minimal-norm joint update
        to both weight and bias to activate the sample (z = wᵀx + b > 0).
        This is a constrained least-norm projection in augmented space.
        """
        model = self.model
        W = model.linear1.weight           # (H, D)
        b = model.linear1.bias             # (H,)
        norms = W.detach().norm(dim=1)     # (H,)

        z = x @ W.detach().T + b.detach()  # (B, H)
        preds = self.forward_outputs["relu1_relu_out_out"].sum(dim=1)
        wrong = (preds > self.classifier_threshold).int() != y.int()

        dead_mask = (z <= 0).all(dim=1) & wrong
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()

        if len(dead_indices) == 0:
            # print("✔️  No dead-and-wrong samples to fix (weight+bias).")
            return

        print(f"⚠️    Dead samples detected. Correcting...  ")
        for i in dead_indices:
            xi = x[i]                     # (D,)
            zi = z[i]                     # (H,)
            di = zi / norms               # scaled distances
            closest = torch.argmin(di.abs()).item()

            # Augmented input: [x; 1] for bias
            x_aug = torch.cat([xi, torch.ones(1, device=xi.device)])  # (D+1,)
            norm_x_aug_sq = x_aug @ x_aug + 1e-12

            zj = zi[closest].item()
            eps = 1e-4
            delta_needed = max(0.0, eps - zj)

            delta_theta = (delta_needed / norm_x_aug_sq) * x_aug      # (D+1,)
            delta_w = delta_theta[:-1]
            delta_b = delta_theta[-1].item()

            W[closest] += delta_w
            b[closest] += delta_b

            # print(f"⚙️  Dead sample {i.item()} (label = {y[i].item()})")
            # print(f"    Input x = {xi.tolist()}")
            # print(f"    Pre-activations z = {zi.tolist()}")
            # print(f"    Distances to ReLU hyperplanes = {di.tolist()}")
            # print(f"    ➤ Closest neuron index: {closest} (|distance| = {di[closest].abs().item():.4f})")
            # print(f"      Joint min-norm patch:")
            # print(f"        Δw = {delta_w.tolist()}")
            # print(f"        Δb = {delta_b:.5f}")

class BoundsMonitor(BaseMonitor):
    def __init__(
        self,
        model: nn.Module,
        dataset_size: int,
        radius: float,
        origin: torch.Tensor = None,
    ):
        super().__init__(model, dataset_size)
        self.radius = radius
        self.origin = origin if origin is not None else torch.zeros(model.linear1.in_features)

    @torch.no_grad()
    def check(self, targets: torch.Tensor, batch_idx: Sequence[int]) -> bool:
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
    def fix(self, x: torch.Tensor, y: torch.Tensor) -> None:
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
        model: nn.Module,
        dataset_size: int,
        patience: int = 3,
        classifier_threshold: float = 0.5,
    ):
        super().__init__(
            model=model,
            dataset_size=dataset_size,
            classifier_threshold=classifier_threshold,
        )
        self.patience = patience
        self.dead_counter = torch.zeros(dataset_size, dtype=torch.int32)

    # ---------------------------------------------------------------
    #  --------- pretty printing as before --------------------------
    # ---------------------------------------------------------------
    def check(self, targets: torch.Tensor, batch_idx: Sequence[int]):
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

    # ---------------------------------------------------------------
    #  helper for toy XOR case (unchanged) --------------------------
    # ---------------------------------------------------------------
    def compute_per_example_gradients(self, inputs, targets, criterion):
        grads = []
        print("\n[Per-Example Gradients and Losses]")
        for i in range(inputs.size(0)):
            self.model.zero_grad()
            out   = self.model(inputs[i].unsqueeze(0))
            loss  = criterion(out, targets[i].unsqueeze(0))
            print(f"Data point {i}: loss = {loss.item():.6f}")
            loss.backward(retain_graph=True)
            w = self.model.linear1.weight.grad.detach().clone()
            b = self.model.linear1.bias.grad.detach().clone()
            grads.append((w, b))
        return grads

    def clear(self):
        self.activations.clear()
        self.forward_outputs.clear()
        self.gradients.clear()

class CompositeMonitor(BaseMonitor):
    def __init__(self, monitors: Sequence[BaseMonitor]):
        self.monitors = monitors
        self.model = monitors[0].model
        self.step_count = 0

    def check(self, targets: torch.Tensor, batch_idx: Sequence[int]) -> bool:
        results = [m.check(targets, batch_idx) for m in self.monitors]
        return all(results)

    def fix(self, x: torch.Tensor, y: torch.Tensor):
        for m in self.monitors:
            m.fix(x, y)

    def to(self, device: torch.device):
        for m in self.monitors:
            m.to(device)
