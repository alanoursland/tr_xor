# Manual Weight Normalization with Static Scaling Layers

## Abstract

Standard parameterizations of deep neural networks contain implicit, free-floating multiplicative factors within their weight matrices. While a network's functional output is invariant to the redistribution of these factors, their presence can introduce poorly understood complexities into the optimization process. This document explores a method for re-parameterizing a network to make these multiplicative dimensions explicit and controllable. We will introduce a "calculus" of equivalence-preserving transformations that allow for the systematic isolation of weight vector directions from their magnitudes. By pushing all per-output scaling factors towards the output of the network and consolidating them, we can achieve a canonical representation. This form is immediately useful for model interpretation and analysis, and may also offer benefits for regularization and training stability when integrated into the training loop.


## Historical Context and Motivation

The challenge of training deep neural networks effectively has led to a variety of normalization techniques.

#### Predecessors: L2 Regularization
For decades, the standard approach to controlling model complexity was **L2 Regularization** (weight decay). This method adds a penalty to the loss function proportional to the squared magnitude of the weights (`||W||²`), encouraging the optimizer to find solutions with smaller weights. However, this approach conflates the learning of a weight's magnitude and its direction.

#### The Rise of Activation Normalization: Batch Normalization
In 2015, Sergey Ioffe and Christian Szegedy introduced **Batch Normalization** (`BatchNorm`), which normalizes the *activations* (the outputs of a layer) across a mini-batch. This dramatically stabilized training by reducing "internal covariate shift," allowing for much deeper networks. However, `BatchNorm`'s dependency on the mini-batch, its differing behavior between training and inference, and its computational overhead motivated the search for alternatives.

#### An Alternative: Weight Normalization
In 2016, Tim Salimans and Diederik P. Kingma proposed **Weight Normalization**, re-parameterizing the weight matrix `W` into a scalar magnitude `g` and a directional vector `v`. The key insight was that decoupling the optimization of the weight vector's length from its direction could accelerate and stabilize training without the drawbacks of `BatchNorm`. The methods described herein build directly upon this principle of decoupling.

## A Calculus for Network Reparameterization

A neural network can be viewed not as a fixed architecture, but as a mathematical expression that can be refactored without changing its input-output function. The following transformations form a basic calculus that allows us to manipulate this expression, with the goal of isolating and propagating multiplicative factors.

### I. Weight-Norm Decomposition

Any linear node can be decomposed into a directional component and a scalar magnitude. This is the foundational operation of our calculus.

* **Transformation:** `linear(W) ==> unit_row_linear(V) -> scale_vector(g)`

* **Derivation:**     "A linear transformation is given by `y = Wx + b`. We can re-parameterize `W` based on the L2 norm of its **row vectors**. Let `w_i` be the i-th row of `W`. We define a scaling **vector** `g` where each element is the norm of the corresponding row:
    $$g_i = ||w_i||_2$$
    The new weight matrix `V` is composed of unit-norm row vectors `v_i = w_i / ||w_i||`. By definition, each row of `V` has a unit L2 norm. The transformation is thus:
    $$y = g \odot (Vx) + b$$
    where `odot` denotes element-wise (Hadamard) product. This decouples the magnitude of each output feature from its directional computation."


### II. ReLU Commutation

A vector multiplication can be commuted with a ReLU activation, enabling the propagation of magnitudes past nonlinearities.

* **Transformation:** `scale_vector(g) -> ReLU ==> ReLU -> scale_vector(g)`

* **Derivation:**         "This transformation relies on the property that ReLU is a positive homogeneous function, which holds for element-wise multiplication. For any input vector `z` and any vector `g` with non-negative elements, the following holds:
        $$\text{ReLU}(g \odot z) = g \odot \text{ReLU}(z)$$
Since the vector `g` in our calculus originates from row norms, its elements are always non-negative, and this commutative property always holds."

### III. Scale Absorption and Propagation

A key insight from this calculus is that the propagation of magnitudes is a strictly **sequential** process that must proceed from the input towards the output. The directional component of a layer (`V`) and the directional component of an incoming scale vector (`u_in`) interfere with each other, creating a new magnitude that cannot be known ahead of time. This transformation describes how to calculate that new, combined magnitude.

* **Transformation:** `scale_vector(g_in) -> linear(W) ==> unit_row_linear(V_out) -> scale_vector(g_out)`

* **Derivation and Intuition:**
    Our goal is to understand the result of the sequence `scale_vector(g_in) -> linear(W)`. Let's follow the insightful approach of decomposing each part into its magnitude and directional components.

    1.  **Decomposition:** The incoming scaling vector `g_in` can be split into its scalar magnitude `s_in = ||g_in||_2` and its unit-norm direction `u_in = g_in / s_in`. Similarly, each row `w_i` of the weight matrix `W` can be split into its magnitude `g_{w,i} = ||w_i||_2` and its direction `v_i = w_i / g_{w,i}`.

    2.  **The Core Interaction:** The operation `W \cdot (g_{in} \odot z)` is equivalent to a new linear layer `W_{temp}` acting on `z`, where each row of the new layer is the element-wise (Hadamard) product of the original row and the incoming scale vector:
        $$\text{row}_i(W_{temp}) = \text{row}_i(W) \odot g_{in}$$
        Using our decomposed parts, this becomes:
        $$\text{row}_i(W_{temp}) = (g_{w,i} \cdot v_i) \odot (s_{in} \cdot u_{in}) = (g_{w,i} \cdot s_{in}) \cdot (v_i \odot u_{in})$$

    3.  **Analysis:** This elegant result reveals two distinct parts:
        * An **easy part:** A simple scalar product of the original magnitudes, `g_{w,i} \cdot s_{in}`.
        * A **hard part:** The Hadamard product of the two *unit-norm* directional vectors, `v_i \odot u_{in}`.

    The magnitude of this final vector is what defines our new outgoing scale, `g_out`. The magnitude of a product is the product of the magnitudes, so:
    $$g_{out, i} = ||\text{row}_i(W_{temp})||_2 = (g_{w,i} \cdot s_{in}) \cdot ||v_i \odot u_{in}||_2$$
    This formula proves that the final magnitude depends critically on the term `||v_i \odot u_{in}||_2`—the result of the interaction between the two directional vectors.

* **Geometric Interpretation: The Hadamard Norm and Directional Interference**

    One might hope that the Hadamard product of two unit vectors (`v_i`, `u_in`) would also be a unit vector, but this is not the case. The term `||v_i \odot u_{in}||_2` is a measure of **directional interference** or "component-wise alignment".

    Imagine two unit vectors as representing the direction of two flashlights, each with a total power of 1.
    * If both vectors are identical (e.g., `v_i = u_in = [1, 0, 0]`), their energy is perfectly concentrated in the same component. The Hadamard product is `[1, 0, 0]`, and its norm is 1. There is no interference.
    * If the vectors are orthogonal (e.g., `v_i = [1, 0]` and `u_in = [0, 1]`), their energy is in completely different components. The Hadamard product is `[0, 0]`, and its norm is 0. The interference is perfectly destructive.
    * For any other angle between them, the norm will be between 0 and 1. For `v_i=[1,0]` and `u_in=[\sqrt{0.5}, \sqrt{0.5}]`, the norm of their Hadamard product is `\sqrt{0.5}`.

    This "loss of magnitude" due to directional interference is why `||v_i \odot u_{in}||_2 \le 1`. This factor must be explicitly calculated for each row.

* **Consolidation:**
    The final resulting `g_out` and `V_out` are therefore:
    1.  **The new scale vector `g_out`**, which is the vector of row norms of `W_{temp}`:
        $$g_{out, i} = || \text{row}_i(W_{temp}) ||_2 = \sqrt{\sum_j (W_{ij} \cdot g_{in, j})^2}$$
    2.  **The new unit-norm matrix `V_out`**, which is `W_{temp}` normalized by `g_out`:
        $$\text{row}_i(V_{out}) = \frac{\text{row}_i(W_{temp})}{g_{out, i}}$$

    The necessity of this explicit calculation at each layer is what makes the overall consolidation a fundamentally sequential process.

## Propagating and Consolidating Magnitudes

Using this calculus, we can systematically transform a standard network into a canonical form where all directional components are separated from a single, consolidated output scaling vector. Consider a simple `linear - ReLU - linear` network:

1.  **Initial State:** `linear(W1) -> ReLU -> linear(W2)`

2.  **Normalize First Layer:** We apply **Transformation I** to the first layer.
    `[unit_row_linear(V1) -> scale_vector(g1)] -> ReLU -> linear(W2)`
    where `g1` is the vector of row norms of `W1`.

3.  **Hop the ReLU:** The `scale_vector(g1)` is adjacent to the ReLU. Using **Transformation II**, we commute them.
    `unit_row_linear(V1) -> ReLU -> [scale_vector(g1) -> linear(W2)]`

4.  **Propagate the Scale:** The sequence in brackets is an instance of **Transformation III**. We absorb the vector `g1` into `W2` and re-normalize. This creates a new final weight matrix `V2` and a new final scaling vector `g2`.
    `unit_row_linear(V1) -> ReLU -> [unit_row_linear(V2) -> scale_vector(g2)]`

The final, functionally identical network is `unit_row_linear(V1) -> ReLU -> unit_row_linear(V2) -> scale_vector(g2)`. The directions of the original weight matrices are now captured in the unit-norm layers `V1` and `V2`, and their combined magnitudes have been consolidated into a single **scaling vector** `g2` at the very end of the network.

### A Framework for Interpretation and Analysis

The result of this consolidation process is a final scaling vector, `g_final`, applied at the end of the network. This vector is rich with information. For more detailed analysis, we can decompose it further into a global magnitude and a relative directional component.
$$g_{final} = g_{scalar} \odot v_{vector}$$
where:
* **$g_{scalar} = ||g_{final}||_2$** is a single scalar representing the network's true **global gain**. This is the ultimate measure of the function's total amplification and is an ideal, simple target for regularization.
* **$v_{vector} = g_{final} / ||g_{final}||_2$** is a unit-norm vector representing the **relative gain direction**. It reveals which output channels the network has learned to amplify more than others, independent of the overall gain. For a classifier, this can be interpreted as a learned prior on the confidence of certain classes.

This hierarchical view allows us to separate the analysis of 'how much' the network amplifies overall (`g_scalar`) from 'where' it directs that amplification (`v_vector`).

### Implications, Applications, and Conclusion

The existence of this sequence of equivalence transformations reveals a fundamental property of many neural networks: they are invariant to re-parameterizations that redistribute vector magnitudes across layers. The standard parameterization contains "free dimensions" corresponding to the norms of its weight matrices. How these free dimensions affect training dynamics and model analysis is a complex and important area of research.

By performing this manual normalization, we transform the network into a canonical representation where these dimensions are no longer free-floating. This process has two primary applications:

**1. As a Tool for Static Analysis (Interpretation):** As discussed, this is an immediately applicable method for understanding a pre-trained model. It provides a way to calculate the true magnitude of a network's function, compare models fairly, and diagnose the role of individual layers, independent of the specific parameterization.

**2. As a Framework for Regularization and Optimization:** The canonical form suggests novel training paradigms.
   * **Regularization:** The final scaling vector, $g_{final}$, can be explicitly constrained or penalized as a highly interpretable form of regularization.
   * **Optimization:** Forcing all linear layers to operate on a unit-norm manifold may simplify the optimization problem, allowing the optimizer to focus solely on finding the correct weight directions while the magnitude is handled separately. This could involve training the directional matrices $V_i$ alongside a single, learnable $g_{final}$ parameter.

In conclusion, viewing a neural network as a malleable mathematical expression allows us to use a simple calculus to isolate its core directional components from its vector magnitudes. This provides a powerful framework for analysis and reveals intrinsic properties of the network parameter space that have profound implications for interpretation, optimization, and generalization.

## Application to Dynamic Training and Regularization

While the calculus is powerful as a tool for post-hoc static analysis, its principles can be integrated directly into the training loop to create a novel and more direct form of regularization. Standard methods like L2 regularization are often indirect, penalizing a mixture of magnitude and direction. By periodically converting the network to its canonical form during training, we can apply targeted penalties to the disentangled components of the model.

We term this approach **Periodic Canonical Regularization (PCR)**. This method addresses the inherent scale ambiguity introduced by loss functions like Binary Cross-Entropy, providing a stabilizing force that is more direct and interpretable than traditional weight decay.

### The Periodic Canonical Regularization (PCR) Algorithm

The PCR algorithm punctuates a standard training process with a canonical re-parameterization and regularization step, executed every `C` training cycles.

The process for a single PCR step is as follows:

1.  **Train (`C` cycles):** The network is trained for `C` steps using a standard optimizer (e.g., Adam) on the conventional weight parameters `W_i`.

2.  **Pause & Canonicalize:** After the `C`-th step, training is paused. The full calculus is applied to the current network state to compute the canonical components: the set of unit-norm directional matrices `V_i` and the final consolidated scaling vector `g_final`.

3.  **Apply Principled Regularization:** Regularization is applied directly to the desired canonical component. The most effective target is the network's global gain, `g_scalar = ||g_final||_2`. A penalty can be applied in two ways:
    * **Soft Penalty:** A term such as `λ * (g_scalar)^2` is added to the loss function.
    * **Hard Constraint:** The global gain is explicitly capped at a maximum value, `g_scalar = min(g_scalar, max_gain)`.

4.  **Reconstitute Network & Optimizer State:** The network's standard weight parameters `W_i` are rebuilt from the (now potentially regularized) canonical components. Critically, the optimizer's internal state is also transformed to remain consistent (see below).

5.  **Resume Training:** Standard training is resumed for another `C` cycles.

### Preserving Optimizer State During Re-parameterization

A significant challenge of PCR is the "Optimizer State Disruption." When `W` is updated to a new `W_new` during the canonicalization step, the optimizer's internal moments (`m` and `v` in Adam) become stale. Simply resetting them would discard valuable learning history.

The proposed solution is to **transform the optimizer's state in lockstep with the weight transformation.** This can be achieved with a principled heuristic based on how rescaling affects gradients. When a weight vector (e.g., a row `w`) is scaled by a factor `α` to become `w_new = α * w`, the moments should be updated as follows:

* **First Moment (Momentum):** The momentum vector `m` is scaled linearly.
    $$m_{new} = \alpha \cdot m_{old}$$
* **Second Moment (Variance):** The variance vector `v`, which tracks squared gradients, is scaled quadratically.
    $$v_{new} = \alpha^2 \cdot v_{old}$$

In the context of our calculus, each transformation (e.g., normalizing a row `w_i` by its norm `||w_i||_2`) is a rescaling operation where `α_i = 1 / ||w_i||_2`. By applying these corresponding updates to the `m` and `v` tensors for each row, we can preserve the optimizer's learned momentum and adaptive learning rates across the re-parameterization step.

This stateful transformation makes the PCR algorithm stable and efficient, turning it from a theoretical concept into a practically viable training strategy. It represents a promising area for future research, exploring the trade-offs between computational cost (the frequency `C`) and the benefits of this highly direct and interpretable form of model regularization.

### Practical Considerations and Future Directions

The calculus presented here forms a foundational framework. Extending it to arbitrary modern architectures requires addressing several practical considerations:

* **Bias Terms:** The propagation of scale vectors also affects bias terms ($g \odot (Vx+b) = (g \odot Vx) + g \odot b$). A complete implementation must explicitly track the scaling and accumulation of biases through the network.
* **Skip Connections:** Architectures like ResNets introduce additive junctions ($x + F(x)$), which prevent the simple factoring-out of a scaling vector. Applying this method to such networks may require analyzing branches separately or developing new transformations for handling additive operations.

These challenges represent promising avenues for future work in extending this powerful analytical framework.