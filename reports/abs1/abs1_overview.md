# Experiment Report: Single Absolute Value Unit for XOR Classification

## Experiment Overview

This experiment investigates how a minimal neural network—a single linear neuron followed by an absolute value activation—learns to solve the XOR problem. The study is motivated by a geometric interpretation of learning in neural networks: that classification emerges from proximity to learned surfaces, not from discrete thresholding. The goal is not to demonstrate feasibility (all models achieve 100% accuracy), but to characterize **how** the model learns—specifically, how the decision boundary evolves to **intersect class prototypes**, and how it moves, scales, and aligns with the input data across different initialization strategies.


## Model Architecture

### Network Structure

The architecture is intentionally minimal:

```
Input (2D) → Linear Layer (1 unit) → Absolute Value → Output
```

There are no hidden layers, gates, or composite nonlinearities—just one linear unit computing a scalar projection, followed by an absolute value transformation.

### Mathematical Description

Given input $x = [x_1, x_2]$, the model computes:

$$
y = |w_1 x_1 + w_2 x_2 + b|
$$

where:

* $w_1, w_2$ are learnable weights,
* $b$ is a learnable bias,
* $|\cdot|$ is the absolute value activation.

This output defines a scalar field over the input space. The value is zero on the learned hyperplane $Wx + b = 0$, and increases with signed distance from this surface. This hyperplane is interpreted as the model's prototype surface for class 0.

## Theoretical Framing

Our prototype surface theory proposes that neural networks learn structured, class-characterizing surfaces rather than just boundaries. The absolute value activation makes this interpretation geometrically transparent:

* The hyperplane $Wx + b = 0$ defines a **prototype surface** for class 0.
* Inputs on this surface yield zero activation.
* Class 1 points lie at a **measurable distance** from this surface.
* The model classifies by encoding relative proximity: class 0 is "on" the surface, class 1 is “off” the surface.

This experiment treats the network as a geometric object, asking how the shape, scale, and orientation of this surface evolve during learning.

## Training Configuration

* **Loss Function**: Mean Squared Error (MSE)
* **Convergence Threshold**: Training stops if loss falls below **1e-7**
* **Optimizer**: Adam (learning rate 0.01, β₁ = 0.9, β₂ = 0.99)
* **Batch Size**: Full batch (4 XOR examples)
* **Epochs**: Maximum of 200
* **Runs**: 50 per initialization strategy

## Initialization Strategies

We test five initialization regimes that differ only in the standard deviation of the initial weights:

1. **Tiny**: $\mathcal{N}(0, 0.1^2)$
2. **Normal**: $\mathcal{N}(0, 0.5^2)$
3. **Xavier**: Xavier/Glorot normal
4. **Kaiming**: He normal
5. **Large**: $\mathcal{N}(0, 4.0^2)$

In all cases, the bias is initialized to zero. These settings explore how different starting geometries (direction and scale) influence learning dynamics, not accuracy.

## Data Configuration

The dataset is the centered version of the XOR problem:

* **Inputs**: $[(-1, -1), (1, -1), (-1, 1), (1, 1)]$
* **Labels**: $[0, 1, 1, 0]$

Centering ensures symmetric geometry around the origin. With bias initialized to zero, the model’s initial prototype surface passes through the center of the input space.

## Geometry and Learning Metrics

We compute the following metrics across all runs to characterize how the model learns. 

### 1. **Convergence Epochs**

* **Definition**: The number of training steps taken to reach loss < 1e-7.
* **Purpose**: Measures how quickly a given initialization leads to successful training.
* **Analysis**: Used as a distribution across runs to compare learning stability and efficiency.

### 2. **Signed and Absolute Distance to Hyperplane**

For each input point $x_i$, we compute:

$$
d(x_i) = \frac{w_1 x_{i1} + w_2 x_{i2} + b}{\|w\|}, \quad\quad |d(x_i)|
$$

* **Purpose**: Reflects how class-0 inputs align with the learned surface and how far class-1 inputs lie from it.
* **Interpretation**: Encodes the **geometric separation** the model learns.

### 3. **Weight Angle (Initial → Final)**

* **Definition**: Angle in degrees between the initial weight vector $w^{(0)}$ and the final weight vector $w^{(T)}$.
* **Purpose**: Indicates how directionally “committed” learning is — whether convergence happens via rotation, scaling, or both.

### 4. **Weight Norm Ratio**

* **Definition**: $\|w^{(0)}\| / \|w^{(T)}\|$
* **Purpose**: Describes how much the learning trajectory involves **rescaling** versus reorientation.
* **Interpretation**: Large changes in norm may indicate slow convergence or overcorrection.

### 5. **Classification Accuracy (for completeness)**

* **Definition**: Outputs are binarized with a threshold at 0.5. Accuracy is the percentage of matching labels across the 4 examples.
* **Note**: All trained models reach **100% accuracy**. This metric is retained only to define that baseline.

## Experimental Focus

This experiment does not aim to evaluate correctness (all models converge). Instead, it investigates:

1. How convergence speed varies by initialization.
2. How surface orientation and scale evolve.
3. How closely the learned hyperplane aligns with class-0 inputs.
4. What trajectories successful runs take through weight space.

## Comments

* We treat divergence in epoch counts as geometry, not error.
* We treat the model as a **surface learner**, not a classifier.
* Variations in performance are not errors but **signals of different geometries** emerging from different initial conditions.

