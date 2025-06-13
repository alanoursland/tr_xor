# **Discussion: Two ReLU Units Summed for XOR Classification**

## 1. Overview

The `relu1` experiment investigates a minimal ReLU-based model designed as a functional analog of the absolute value model (`abs1`). It consists of two opposing linear units, each passed through a ReLU activation, and summed. Although mathematically capable of implementing an absolute value—via the identity `abs(x) = relu(x) + relu(-x)`—the `relu1` model did **not** reliably learn to classify XOR.

Across 50 runs, **only 29 reached 100% accuracy**, while the remaining 21 converged to suboptimal solutions. These failures were not due to training instability or divergence, but rather **stable convergence to incorrect geometries**. This discrepancy, when compared to the consistent success of `abs1`, highlights an important insight: **functional equivalence does not imply optimization equivalence**.

## 2. Interpreting the Failures

### 2.1. Initial Observations

The failed runs all achieved a low final loss, yet did not correctly classify all XOR inputs. This suggests that the model converged to **local minima** that satisfied the loss function numerically but not semantically. The outputs for class-1 points hovered near—but not above—the 0.5 threshold, making them misclassified despite the overall error being small.

This behavior was not seen in the absolute value model, which consistently aligned its prototype surface with the class-0 inputs and achieved high-contrast outputs for the `True` class. In contrast, the `relu1` model often failed to build that symmetry, resulting in **weaker geometric separation** and **flatter decision boundaries**.

### 2.2. Hypothesis: Dead Data Points at Initialization

This theory is supported by our analysis: across 50 runs, **no models with non-dead inputs failed to reach 100% accuracy**, while **all 21 failed runs had at least one dead input** at initialization.

In particular, class-1 dead points were strongly correlated with failure. No model with a dead class-1 input achieved 100% accuracy. In contrast, dead class-0 inputs were often—but not always—tolerated.

These findings support the theory that **dead inputs starve the model of gradient signal**, preventing the prototype surface from aligning with important decision boundaries. Presense of dead nodes do not guarantee model failure, but all of the models without dead nodes successfully trained to 100%

### 2.3 Mirror Symmetry: Emergent in All Successful Runs

We also analyzed whether the two ReLU units formed mirror-symmetric pairs—i.e., weights pointing in opposite directions with equal magnitude, and biases of equal magnitude but opposite sign. This structure mimics the hardcoded symmetry of the `abs1` model.

Our results show that:

* **All 29 runs that reached 100% accuracy exhibited at least one mirror-symmetric pair**.
* Of those, **15 showed near-perfect symmetry**, with cosine similarity between mirrored weights close to –1.0.
* Among failed runs (75% accuracy), mirror symmetry was often absent or imperfect.

This suggests that **emergent mirror symmetry is a necessary condition for successfully solving XOR in this architecture**, even though the model is not explicitly constrained to discover it. The absence of symmetry appears to trap the model in suboptimal configurations, reinforcing the idea that symmetry provides both geometric clarity and optimization guidance.

## 3. Implications for Prototype Surface Learning

These results reinforce a key tenet of Prototype Surface Learning (PSL): that learning is a geometric process driven by **gradient flow from class-aligned surfaces**. In `abs1`, that flow is always present. In `relu1`, it is **fragile and conditional**, dependent on whether the initialization allows class-0 points to influence the gradient.

More broadly, this highlights a **critical role for inductive bias** in neural architectures. Although `relu1` can represent the same function as `abs1`, it lacks the structural guarantee that the absolute value provides. This makes the optimization landscape far more treacherous, even for a trivial dataset like XOR.

## 4. Directions for Further Investigation

This experiment opens several lines for deeper study:

* **Initialization diagnostics**: Track which points are "dead" at initialization and correlate with training failure.
* **Gradient flow analysis**: Measure per-point gradient magnitude to confirm whether class-0 points go silent in failed runs.
* **Structural regularization**: Explore architectural tweaks (e.g. bias initialization, asymmetric weights) that encourage early symmetry and prevent dead regions.
* **Activation comparisons**: Evaluate whether other symmetric or piecewise-linear activations show similar failure dynamics.
* **Initialization diagnostics**: We implemented a dead input detector that confirmed a strong correlation between dead points and training failure.
* **Mirror symmetry tracking**: We measured emergent symmetry between ReLU units and found it to be common but inconsistent.

While `relu1` is theoretically sufficient to solve XOR, its failure to do so reliably underscores the value of geometric interpretability—and of architectures that **make desirable gradients inevitable**, not optional.
