Here is the draft of `abs2_overview.md` written in standard prose:

---

# Experiment Report: Single Absolute Value Unit with BCE Loss (abs2)

## Experiment Overview

This experiment evaluates whether a minimal neural network consisting of a single linear neuron with an absolute value activation can solve the XOR classification task using a two-output Binary Cross-Entropy (BCE) loss. It is the first experiment in the `abs2` series and builds directly on prior work from `abs1`, extending the prototype surface framework to a classification setting that uses logit outputs and a binary classification objective.

The central question is whether the network can still align a learned prototype surface with the XOR structure and successfully classify based on deviations from this surface, even when optimized under BCE rather than Mean Squared Error (MSE). The experiment also highlights how the network utilizes this single deviation signal to generate class-specific predictions across multiple random initializations.

## Model Architecture

The architecture is deliberately minimal and consists of the following layers:

* A linear layer that maps 2D inputs to a single scalar projection.
* An absolute value activation, which interprets this projection as an unsigned distance from a learned surface.
* A static, non-learnable scaling layer to preserve geometric interpretability by preventing internal weight magnitudes from drifting arbitrarily.
* A second linear layer that maps the scaled distance signal to two output logits—one for each class (XOR = False, XOR = True).

This setup enables the model to act as a prototype surface evaluator: the absolute value unit measures deviation from a central surface, while the final linear layer interprets that distance in terms of class affinity.

## Loss Function and Targets

The model is trained using `nn.BCEWithLogitsLoss`, which expects raw logits and internally applies the sigmoid function. The XOR labels are provided as one-hot float vectors:

* \[1.0, 0.0] for XOR = False
* \[0.0, 1.0] for XOR = True

This configuration allows each output neuron to independently evaluate whether the input belongs to its corresponding class. The use of two outputs emphasizes that class decisions are made by comparing affinities, not by passing a single score through a threshold.

## Data Configuration

The XOR dataset is centered for geometric symmetry. Inputs are:

* (-1, -1)
* (-1, 1)
* (1, -1)
* (1, 1)

with corresponding one-hot targets based on the XOR truth table. This centering ensures that learned surfaces can be interpreted with respect to the origin, making visualization and geometric analysis more consistent across experiments.

## Training Configuration

* Loss function: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
* Optimizer: Adam (learning rate = 0.01, betas = (0.9, 0.99))
* Maximum epochs: 400
* Early stopping: triggered if loss falls below 1e-7 or stops improving by more than 1e-24 over 10 epochs
* Batch size: full (4 data points)
* Runs: 50, each with different random initialization
* Initialization: Kaiming Normal for the first linear layer; Xavier Normal for the second; static scale set to 1.0

This configuration allows reliable and interpretable comparisons across training runs while focusing on the convergence behavior of a single-unit architecture.

## Experimental Focus

The experiment is designed to test three main hypotheses:

1. That a single absolute value unit can solve XOR classification under a 2-output BCE loss, despite the minimal model capacity.
2. That the model converges to a stable geometric solution where the learned surface intersects the XOR=False class, and deviations from this surface distinguish class membership.
3. That downstream layers can reliably transform a scalar geometric distance into meaningful class logits without requiring additional hidden units or nonlinearities.

The results of this experiment will clarify how well the prototype surface learning framework generalizes from regression-based training to logit-based classification, and whether a single-unit deviation signal is sufficient to drive high-confidence, surface-based decision making.
