# Experiment Report: Symmetric ReLU Pair for XOR Classification

## Experiment Overview

This experiment investigates how a minimal ReLU-based network—composed of two opposing linear units passed through ReLU and summed—learns to solve the XOR problem. This model is designed as a structural analog of the single-unit absolute value network (`abs1`), based on the identity:

$$|x| = \text{ReLU}(x) + \text{ReLU}(-x)$$

Preliminary runs revealed that, unlike the robust `abs1` model, this architecture is surprisingly fragile and frequently fails to converge to a correct solution under a standard initialization. This report documents a series of experiments designed to diagnose and correct these failures. We began by testing the **Dead Data Point Hypothesis**, which posited that failures were caused by inputs having no initial gradient signal. After this was largely confirmed, we identified and tested a secondary **Margin Hypothesis**, which posited that failures also occurred when hyperplanes initialized too close to data points.

This multi-stage investigation analyzes how different initialization heuristics affect convergence, reliability, and the emergence of the geometric structures required to solve the problem.

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

During early analysis of models that failed to achieve 100% accuracy, we identified a recurring structural pattern: some input points produced **zero activation** across all ReLU units in a layer at initialization. We refer to these as **dead data points**.

### Definition

An input is considered *dead* if, at model initialization, it lies outside the activation region of **all** ReLU units in a layer — i.e., it falls entirely into the negative half-space for each linear unit. Formally, an input $x_i$ is dead if:

$$
\text{ReLU}(w_1 \cdot x_i + b_1) = \text{ReLU}(w_2 \cdot x_i + b_2) = 0
$$

This condition means that the input generates **no signal**, contributes **no loss gradient**, and is effectively **invisible to training** until it activates. If the model fails to adjust its weights to activate the point, it may never learn from it.

### Motivation

We introduced this metric after observing that several failed training runs shared similar decision boundaries, and many of them had inputs that were dead from the start. We hypothesize that **dead data points at initialization are predictive of learning failure**.

---

### A Secondary Failure Mode: The Margin Hypothesis

After correcting for dead data points, a small number of training failures still occurred. Analysis of these cases suggested a new hypothesis: failures can also happen if a neuron's hyperplane initializes **too close to a data point**.

If the initial margin is too small, a slight weight update during early optimization can push the hyperplane across a nearby data point. This can "deactivate" the point's influence on that neuron by moving it into the ReLU's zero-gradient region, effectively halting learning for that point-neuron pair and potentially trapping the model in a suboptimal state. This "Margin Hypothesis" suggests that a "safe" initialization requires not only that all points are "live," but also that they have sufficient distance from the initial decision boundaries.

---

## Training Configuration

* **Loss Function**: Mean Squared Error (MSE)
* **Convergence Threshold**: Training stops if loss falls below **1e-7**
* **Stagnation Criterion**: Training also stops if loss does not improve by at least **1e-24** over **10 consecutive epochs**
* **Optimizer**: Adam (learning rate 0.01, β₁ = 0.9, β₂ = 0.99)
* **Batch Size**: Full batch (4 XOR examples)
* **Epochs**: Maximum of 800.
* **Runs**: 50 to 500 per initialization strategy.

---

## Initialization Strategies

We test a sequence of initialization strategies to diagnose and resolve the model's training failures. In all cases, the base weight initialization uses a normal distribution $\mathcal{N}(0, 0.5^2)$ and biases are initialized to zero.

1.  **Standard Init (`relu1_normal`)**: The baseline condition. The model uses weights as-is after sampling, with no checks. This condition revealed the model's fragility.

2.  **Re-init on Dead (`relu1_reinit`)**: The first intervention. The model is re-initialized until no data points are "dead," ensuring all inputs have an initial gradient signal from at least one neuron.

3.  **Re-init with Margin (`relu1_reinit_margin`)**: The final set of interventions. In addition to the "live data" check, this condition also re-initializes if any hyperplane is within a specified margin $\epsilon$ of any data point. We tested this condition with progressively larger margins of **$\epsilon=0.1$**, **$\epsilon=0.2$**, and **$\epsilon=0.3$**.

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

$$d_1(x_i) = \frac{w_1 \cdot x_i + b_1}{\|w_1\|}, \quad d_2(x_i) = \frac{w_2 \cdot x_i + b_2}{\|w_2\|}$$

* **Purpose**: Shows how the two half-spaces contribute to classification.
* **Interpretation**: Helps identify whether the model constructs symmetric or asymmetric regions of influence.

### 4. **Weight Angle and Norm Evolution**

* **Angle**: Between initial and final weight vectors for each unit
* **Norm Ratio**: $\|w^{(0)}\| / \|w^{(T)}\|$

These show whether training proceeds by rotation, scaling, or both, and whether the two units evolve similarly.

### 5. **Classification Accuracy**

As in the `abs1` experiment, this is tracked for completeness, but our focus is on **how** the models get there.

## Experimental Focus

This experiment follows an iterative, hypothesis-driven process. The primary goals evolved to include:

1.  Diagnosing the cause of the high failure rate in the standard ReLU model.
2.  Testing the **Dead Data Point Hypothesis** by comparing the `Standard Init` and `Re-init on Dead` conditions.
3.  Testing the **Margin Hypothesis** by applying progressively larger initialization margins.
4.  Identifying a set of initialization heuristics sufficient to achieve a 100% success rate on this problem.
5.  Analyzing how these interventions affect the emergence of symmetric geometric structures in the final learned model.

## Comments

* This model reveals how **implicit structure must be discovered**, not imposed.
* Divergence in symmetry is treated as meaningful geometry, not failure.
* Convergence success does not imply correct surface interpretation — only structural symmetry does.