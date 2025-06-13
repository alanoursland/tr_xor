# Experiment Report: Symmetric ReLU Pair for XOR Classification

## Experiment Overview

This experiment investigates how a minimal ReLU-based network—composed of two opposing linear units passed through ReLU and summed—learns to solve the XOR problem. This model is designed as a structural analog of the single-unit absolute value network, based on the identity:

$$
|x| = \text{ReLU}(x) + \text{ReLU}(-x)
$$

The purpose is not to test whether the model can reach 100% accuracy (it can), but to analyze **how** it learns when the underlying symmetry of the absolute value is not baked in, but must emerge through optimization. We focus on convergence dynamics, symmetry formation, and geometric structure in weight space.

## Model Architecture

### Network Structure

The network is composed of two opposing linear units, each feeding into a ReLU activation. Their outputs are summed directly to produce a scalar output:

```
Input (2D) → Linear Layer (2 units) → ReLU → Sum → Output
```

There is no hidden layer or learnable output weights; the final layer simply sums the activations with fixed weights `[1, 1]`.

### Mathematical Description

Given input $x = [x_1, x_2]$, the model computes:

$$
y = \text{ReLU}(w_1 \cdot x + b_1) + \text{ReLU}(w_2 \cdot x + b_2)
$$

This is functionally equivalent to the absolute value transformation if:

* $w_2 = -w_1$
* $b_2 = -b_1$

But the model does not enforce this symmetry — it must be learned through gradient descent.

## Theoretical Framing

In the context of prototype surface learning, this model tests whether **a symmetric prototype surface** can be constructed from **asymmetric components**. Each ReLU unit defines an activation region corresponding to a half-space. The sum of two such ReLU fields can mimic a symmetric surface, but only if the weights and biases of the two units are mirror images.

This experiment studies whether such symmetry emerges reliably from standard training, and how its formation compares to the explicit structure imposed by the absolute value.

---

## Dead Data Point Hypothesis

During early analysis of models that failed to achieve 100% accuracy, we identified a recurring structural pattern: some input points produced **zero activation** across all ReLU units at initialization. We refer to these as **dead data points**.

### Definition

An input is considered *dead* if, at model initialization, it lies outside the activation region of **all** ReLU units — i.e., it falls entirely into the negative half-space for each linear unit. Formally, an input \$x\_i\$ is dead if:

$$
\text{ReLU}(w_1 \cdot x_i + b_1) = \text{ReLU}(w_2 \cdot x_i + b_2) = 0
$$

This condition means that the input generates **no signal**, contributes **no loss gradient**, and is effectively **invisible to training** until it activates. If the model fails to adjust its weights to activate the point, it may never learn from it.

### Motivation

We introduced this metric after observing that several failed training runs shared similar decision boundaries, and many of them had inputs that were dead from the start. We hypothesize that **dead data points at initialization are predictive of learning failure**, particularly when they correspond to critical class-1 examples (which must be pushed off-surface to solve XOR).

This analysis does not claim dead inputs are always harmful—but when they occur early and persist, they may severely restrict the model's capacity to reshape its decision surface.

Further experiments (e.g., reinitializing models until no data is dead) are planned to evaluate this hypothesis more directly.

---

## Training Configuration

* **Loss Function**: Mean Squared Error (MSE)
* **Convergence Threshold**: Training stops if loss falls below **1e-7**
* **Stagnation Criterion**: Training also stops if loss does not improve by at least **1e-24** over **10 consecutive epochs**
* **Optimizer**: Adam (learning rate 0.01, β₁ = 0.9, β₂ = 0.99)
* **Batch Size**: Full batch (4 XOR examples)
* **Epochs**: Maximum of 200
* **Runs**: 50 per initialization strategy

---

## Initialization Strategies

We initialize the model using a normal distribution $\mathcal{N}(0, 0.5^2)$.

All biases are initialized to zero.

## Data Configuration

The network is trained on the centered XOR dataset:

* **Inputs**: $[(-1, -1), (1, -1), (-1, 1), (1, 1)]$
* **Labels**: $[0, 1, 1, 0]$

Centering ensures that any emergent symmetry in the solution can be interpreted cleanly, and matches the conditions used in the `abs1` experiment.

## Geometry and Learning Metrics

To study symmetry and surface formation, we compute the following:

### 1. **Convergence Epochs**

* **Definition**: Number of epochs to reach loss < 1e-7.
* **Purpose**: Indicates training stability and sensitivity to initialization.
* **Interpretation**: Faster convergence may correlate with symmetry-aligned initial weights.

### 2. **Component Symmetry Metrics**

We measure whether the two ReLU units form a symmetric pair:

* **Weight Angle Deviation**: Angle between $w_1$ and $-w_2$
* **Bias Symmetry**: Difference $b_1 + b_2$

These capture how closely the model approximates the structure of an absolute value.

### 3. **Signed and Absolute Input Distances**

Each input $x_i$ is projected onto each weight:

$$
d_1(x_i) = \frac{w_1 \cdot x_i + b_1}{\|w_1\|}, \quad d_2(x_i) = \frac{w_2 \cdot x_i + b_2}{\|w_2\|}
$$

* **Purpose**: Shows how the two half-spaces contribute to classification.
* **Interpretation**: Helps identify whether the model constructs symmetric or asymmetric regions of influence.

### 4. **Weight Angle and Norm Evolution**

* **Angle**: Between initial and final weight vectors for each unit
* **Norm Ratio**: $\|w^{(0)}\| / \|w^{(T)}\|$

These show whether training proceeds by rotation, scaling, or both, and whether the two units evolve similarly.

### 5. **Classification Accuracy**

As in the `abs1` experiment, this is tracked for completeness. All models are expected to eventually reach 100% accuracy, but our focus is on **how** they get there.

## Experimental Focus

This experiment explores:

1. Whether the model consistently learns the symmetric structure of the absolute value.
2. How symmetry failure correlates with convergence speed and geometry.
3. How training behavior differs from `abs1`, despite functional equivalence being possible.
4. What initialization scales are most conducive to emergent symmetry.

## Comments

* This model reveals how **implicit structure must be discovered**, not imposed.
* Divergence in symmetry is treated as meaningful geometry, not failure.
* Convergence success does not imply correct surface interpretation — only structural symmetry does.
