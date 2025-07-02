# On the Magnitude Eater: A Feedback-Based Regularizer

## Abstract

Standard deep neural networks are often invariant to the scaling of their weights, a property that complicates the optimization landscape and hinders model analysis. We introduce and analyze a novel, training-time-only regularizer, termed the **Magnitude Eater**. Unlike methods that directly re-parameterize weights or normalize activations, the Magnitude Eater implements a feedback control mechanism. It scales its input by a running average of historical input magnitudes, which in turn scales the backpropagated gradient. We formally analyze the training dynamics under this mechanism and show that the only stable equilibrium for the system occurs when the preceding layer's output vectors have an expected L2 norm of one. This induces self-regularization, improves model identifiability, and adds no computational overhead at inference time.

-----

## 1\. Introduction and Motivation

A well-known property of many deep neural networks is the ambiguity of scale in their internal representations. The function computed by a network can remain identical under transformations that re-scale the weights of adjacent layers. This ambiguity introduces a redundant degree of freedom in the optimization process, complicating the loss landscape with plateaus and non-identifiable solutions. This makes the comparison of learned parameters across different models or training runs fundamentally ill-posed.

Furthermore, the lack of control over the magnitude of layer outputs (activations) can lead to training instability, contributing to the problems of vanishing and exploding gradients. While techniques like Batch Normalization address this by standardizing layer inputs, they do so by introducing learned affine transformation parameters and a dependency on batch statistics.

We propose a different approach: a regularizer that creates an incentive for the network to **learn to self-regulate** the magnitude of its own representations.

-----

## 2\. The Magnitude Eater Layer

### 2.1. Formal Definition

The Magnitude Eater is a stateful layer active only during training. It maintains two state variables, or buffers:

  * `A_t`: The running average of L2 norms of input vectors at step `t`.
  * `M_t`: The number of data points represented in the average `A_t`.

The layer is defined by a single hyperparameter, `Max`, representing the maximum size of the data window for the running average.

### 2.2. Mechanism of Action

Let `x_t` be the input to the Magnitude Eater from a preceding layer at training step `t`. The forward pass is defined as:

$$y_t = x_t \cdot A_{t-1}$$

where `y_t` is the output passed to the subsequent layer, and `A_{t-1}` is the running average from the previous step.

After the forward pass, the layer's state is updated. Let `N_t` be the number of samples in the batch `x_t`, and let `A_{x_t}` be the mean L2 norm of the vectors in that batch. The number of old data points to discard, `d_t`, is calculated to maintain the window size:

$$d_t = \text{ReLU}((M_{t-1} + N_t) - \text{Max})$$

The running average `A_t` and count `M_t` are then updated using a numerically stable, memory-efficient sliding window approximation:

$$A_t = \frac{(A_{t-1} \cdot (M_{t-1} - d_t)) + (A_{x_t} \cdot N_t)}{(M_{t-1} - d_t) + N_t}$$
$$M_t = (M_{t-1} - d_t) + N_t$$

At inference time (`model.eval()`), the layer becomes an identity function: `y_t = x_t`.

-----

## 3\. Analysis of Training Dynamics

We now prove that the Magnitude Eater creates a strong incentive for the optimizer to find solutions where the expected norm of the layer's input, `E[||x_t||]`, converges to 1.

### 3.1. Gradient Propagation Analysis

Let `L` be the total loss function of the network, which is a function of the Magnitude Eater's output, `y_t`. Let `w` be the weights of the layer preceding the Magnitude Eater, such that `x_t = f(w, z_{t-1})`, where `z_{t-1}` is the input to that layer. The learning signal that updates `w` is proportional to the gradient of the loss with respect to `x_t`.

Using the chain rule, this gradient is:

$$\frac{\partial L}{\partial x_t} = \frac{\partial L}{\partial y_t} \cdot \frac{\partial y_t}{\partial x_t}$$

From the forward pass definition, `y_t = x_t \cdot A_{t-1}`. During the forward and backward pass for a given step `t`, `A_{t-1}` is a pre-computed scalar constant. Therefore, its derivative is straightforward:

$$\frac{\partial y_t}{\partial x_t} = A_{t-1}$$

Substituting this back, we arrive at the core of the dynamic:

$$\frac{\partial L}{\partial x_t} = \frac{\partial L}{\partial y_t} \cdot A_{t-1}$$

### 3.2. Interpretation and Proof of Behavior

This result shows that the gradient signal (`∂L/∂y_t`) flowing from the subsequent parts of the network is **scaled by the historical average magnitude `A_{t-1}`** before it updates the weights of the preceding layer.

We analyze the behavior in three regimes:

  * **Case 1: Systemic Over-estimation (`A_{t-1} > 1`)**. This occurs when the preceding layer consistently produces vectors with a norm greater than 1. In this state, the gradient flowing back to that layer is **amplified**. A larger gradient leads to a more aggressive weight update. This punishes the layer for producing large-magnitude outputs, creating a strong pressure for the optimizer to reduce the norms of the weights `w` to decrease `||x_t||`.

  * **Case 2: Systemic Under-estimation (`A_{t-1} < 1`)**. This occurs when `||x_t||` is consistently less than 1. Here, the gradient is **dampened**. A diminished learning signal is suboptimal for convergence. The optimizer is thus implicitly incentivized to increase the norms of `w` to produce a larger `||x_t||`, thereby restoring a stronger gradient signal.

  * **Case 3: Equilibrium (`A_{t-1} \approx 1`)**. This is the stable fixed point of the system. When the historical average magnitude is approximately 1, the gradient is passed through almost unchanged (`∂L/∂x_t ≈ ∂L/∂y_t`). The Magnitude Eater becomes transparent to the optimization process. At this point, the distorting influence of the scaling factor is neutralized, and the optimizer can focus efficiently on learning the optimal *orientation* of the output vectors `x_t`.

The optimizer's objective is to minimize `L`. The analysis shows that this optimization is most stable and efficient when the gradient scaling factor `A_{t-1}` is 1. Therefore, the optimizer is guided by this implicit meta-objective to seek solutions where `E[||x_t||] \to 1`.

-----

## 4\. Discussion

The Magnitude Eater presents a unique form of regularization that differs fundamentally from existing methods. Unlike **Batch Normalization**, it does not standardize inputs based on batch statistics and introduces no new learnable parameters. Unlike **Weight Normalization**, it does not re-parameterize the weight vectors themselves. Instead, it creates a feedback control system that acts on the gradient dynamics directly, guiding the optimizer toward solutions that lie on a manifold of normalized representations.

The primary limitation is the introduction of a new hyperparameter, `Max`, which controls the memory of the system. However, this parameter is arguably more intuitive than the momentum term in an exponential moving average. Future work should include empirical studies to validate its performance against other regularizers on benchmark datasets and explore its impact on the geometry of the loss landscape.

## 5\. Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MagnitudeEater(nn.Module):
    def __init__(self, max_points: int):
        super().__init__()
        self.max_points = max_points
        # Store the running average and point count as persistent buffers
        self.register_buffer('running_avg', torch.tensor(1.0))
        self.register_buffer('point_count', torch.tensor(0.0))

    def forward(self, x):
        if not self.training:
            return x

        # Amplify the input using the stored running average
        amplified_x = x * self.running_avg.detach()

        # Update the sliding window average outside of the gradient graph
        with torch.no_grad():
            N = x.shape[0]
            A_N = torch.mean(torch.norm(x, p=2, dim=1))
            M = self.point_count
            A_M = self.running_avg
            
            d = F.relu((M + N) - self.max_points)
            M_final = (M - d) + N

            if M_final > 0:
                numerator = (A_M * (M - d)) + (A_N * N)
                A_final = numerator / M_final
            else:
                A_final = torch.tensor(1.0)

            self.running_avg = A_final
            self.point_count = M_final
            
        return amplified_x
