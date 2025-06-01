# Experiment Report: Single Absolute Value Unit for XOR Classification

## Experiment Overview

This experiment investigates whether a minimal neural network—a single linear neuron followed by an absolute value activation—can learn to solve the XOR problem. The study is motivated by a geometric interpretation of how neural networks learn class-defining structures through prototype surfaces. Specifically, the goal is to demonstrate that even a model as simple as one node in one layer can learn a solution to XOR, not through linear separation, but by learning a geometric surface that characterizes one class and distinguishes it from the other based on distance.

## Model Architecture

### Network Structure

We test a minimal model architecture consists of a single linear transformation followed by an absolute value activation:

```
Input (2D) → Linear Layer (1 unit) → Absolute Value → Output
```

There are no hidden layers, no gating mechanisms, and no additional nonlinearities—just a single neuron acting directly on the input space.

### Mathematical Description

Given input $x = [x_1, x_2]$, the model computes:

$$
y = |w_1 x_1 + w_2 x_2 + b|
$$

where:

* $w_1, w_2$ are learnable weights,
* $b$ is a learnable bias,
* $|\cdot|$ is the absolute value activation.

The output $y$ represents a scalar field over the input space, increasing with distance from a learned hyperplane.

## Theoretical Foundation

This experiment is grounded in prototype surface theory, which reinterprets the role of neurons in terms of geometry. Rather than acting as simple threshold gates, neurons are hyperplanes that intersect feature prototypes. The linear component $Wx + b$ defines a hyperplane in input space. The absolute value activation then transforms this into a symmetric distance function: zero on the surface, positive elsewhere.

Under this interpretation:

* The neuron **represents a prototype surface** for one class—specifically, the set of inputs that yield zero output.
* Points **on the surface** are treated as exemplars of one class (e.g., class 0).
* Points **far from the surface** produce higher output, distinguishing the opposing class (e.g., class 1).
* Classification emerges naturally by comparing the distance from this learned surface.

For the XOR problem, the ideal solution aligns the prototype surface such that it passes through the class-0 inputs, while the class-1 inputs lie equidistant on either side. The absolute value activation captures this symmetric relationship directly.

## Training Configuration

* **Optimizer**: Adam (learning rate 0.01, β₁ = 0.9, β₂ = 0.99)
* **Loss Function**: Mean Squared Error (MSE)
* **Epochs**: 200 per run
* **Batch Size**: Full batch (4 XOR examples)
* **Runs**: 500 per initialization strategy

## Initialization Strategies

To explore the robustness of this minimal model, the experiment tests five initialization methods:

1. **Tiny**: Weights from $\mathcal{N}(0, 0.1^2)$
2. **Normal**: Weights from $\mathcal{N}(0, 0.5^2)$
3. **Xavier**: Xavier/Glorot normal initialization
4. **Kaiming**: He normal initialization
5. **Large**: Weights from $\mathcal{N}(0, 4.0^2)$

In all cases, the bias $b$ is initialized to zero.

These variations help assess the consistency and sensitivity of learning. They are not necessary to validate the core hypothesis, but they inform how optimization dynamics interact with geometry in this simple setting.

## Data Configuration

The dataset uses a centered form of the XOR problem:

* **Inputs**: $[(-1, -1), (1, -1), (-1, 1), (1, 1)]$
* **Labels**: $[0, 1, 1, 0]$

Centering ensures that the data is symmetric around the origin. With the bias initialized to zero, the initial prototype surface passes through the mean of the data. This follows standard practice in neural network training where centering inputs or activations is known to improve optimization dynamics. For background on this practice, see LeCun et al. (2012) or Ioffe & Szegedy (2015) on normalization in deep networks.

## Comments

1. MSE is an uncommon choice for classification problems. We are trying to keep this model simple by using a single output that returns either 0 or 1. XOR is simple enough that this works. It may not be possible in more complex datasets.

2. While we use MSE for error, for classification we use nearest neighbor to the target values 0 and 1.

3. With four data points, we are looking for perfect learning on the training set. We are not studying the generalization problem in this experiment. We are studing the geometric representation of solutions.

## Experimental Objectives

1. **Demonstrate Feasibility**
   Show that a single neuron with absolute value activation can learn to solve the XOR classification task without hidden layers.

2. **Understand Geometric Feature Encoding**
   Analyze how the neuron represents a class-specific surface in input space, and how this surface aligns with labeled examples.

3. **Support Prototype Surface Learning Theory**
   Empirically examine whether successful training leads to a configuration where class-0 inputs lie on or near the learned surface and class-1 inputs lie at a distance.

4. **Explore Optimization Robustness**
   Investigate how different initialization strategies affect convergence, and whether the model can consistently learn the correct surface geometry.

5. **Identify Future Questions via Failures**
   Although results are not discussed here, one aim of this work is to lay the groundwork for understanding why this simple model sometimes fails to learn, despite having sufficient capacity. These failure cases will be examined in the companion results report.

Great—here’s an **Evaluation Metrics** section that fits seamlessly with the revised overview and highlights the aspects most relevant to your theory-driven goals:

## Evaluation Metrics

To evaluate whether the model successfully learns the XOR task through the lens of prototype surface learning, the following metrics are collected across runs:

### 1. **Final Loss (Error)**

* **Metric**: Mean Squared Error (MSE) over the 4 XOR examples.
* **Purpose**: Measures how closely the model’s outputs match target values; used to assess convergence.

### 2. **Classification Accuracy**

* **Metric**: Percentage of correctly classified examples, using a threshold of 0.5 to binarize outputs. It is effectively a nearest neighbor classification to the target data.
* **Purpose**: Provides a discrete measure of success; 100% accuracy corresponds to perfect XOR classification.

### 3. **Accuracy Distribution**

* **Metric**: Distribution of final accuracy across all runs (expected values: 0%, 25%, 50%, 75%, 100%).
* **Purpose**: Characterizes variability in convergence outcomes; helps identify partial or inconsistent learning patterns.

### 4. **Distance to Hyperplane**

* **Metric**: For each input point $x_i$, compute the signed distance:

  $$
  d(x_i) = \frac{w_1 x_{i1} + w_2 x_{i2} + b}{\|w\|}
  $$

  and the absolute distance:

  $$
  |d(x_i)| = \frac{|w_1 x_{i1} + w_2 x_{i2} + b|}{\|w\|}
  $$
* **Purpose**: Validates prototype surface theory predictions. Class-0 points should lie near the surface (small distance), while class-1 points should lie far (large distance). These distances directly reflect the learned geometry.

### 5. **Hyperplane Normal Length**

* **Metric**: Euclidean norm of the learned weight vector:

  $$
  \|w\| = \sqrt{w_1^2 + w_2^2}
  $$
* **Purpose**: In the model $y = |Wx + b|$, the output is a **scaled distance** from the input point $x$ to the learned prototype surface $Wx + b = 0$. The true Euclidean distance is:

  $$
  \frac{|Wx + b|}{\|w\|}
  $$

  Therefore, $\|w\|$ serves as the scaling factor applied to the actual distance. Tracking $\|w\|$ lets us interpret model outputs geometrically and assess how the learned surface expresses distance. It also helps distinguish whether high output values result from increased separation or merely larger scaling.

