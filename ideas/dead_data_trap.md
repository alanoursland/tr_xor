### **Prototype-Surface Control Theory for ReLU Networks**

*A compact reference on geometric failure modes, stability proofs, and
run-time health monitoring*

---

## 1 Scope & Motivation

Deep networks whose hidden units are piece-wise linear (ReLU, PReLU,
conv + ReLU, etc.) can be interpreted as learning **prototype
surfaces**—collections of oriented hyperplanes whose signed (or rectified)
distance to the data manifold drives classification.
Two systematic pathologies plague such nets:

* **Dead-data / dead-ReLU collapse** – an input (or an entire neuron)
  lives forever in the zero-gradient region.
* **No-torque traps** – a hyperplane’s gradient is collinear with its
  own normal, so its norm can change but its **orientation** cannot.

The goal of this note is to formalise those phenomena, explain why they
appear or disappear when we pass from an *absolute–value* unit
(“two mirrored ReLUs”) to a *single* ReLU, and to collect practical
**control-theoretic monitors and interventions** that keep training on
track.

---

## 2 Mathematical set-up

### 2.1 Models

| name      | output                   | trainable parms |   |                                      |
| --------- | ------------------------ | --------------- | - | ------------------------------------ |
| **abs1**  | (                        | w^\top x+b      | ) | $w\in\mathbb R^{d}$, $b\in\mathbb R$ |
| **relu1** | $\max\{0,\,w^\top x+b\}$ | $w,b$           |   |                                      |

The absolute value can be written
$|\cdot|=\text{ReLU}(\cdot)+\text{ReLU}(-\cdot)$; it is therefore
**two perfectly mirrored channels.**

All proofs below use the centred XOR data

$$
\bigl\{(x_i,y_i)\bigr\}_{i=1}^4=
\{(1,-1,1),\;(-1,1,1),\;(1,1,0),\;(-1,-1,0)\}.
$$

The loss is always mean-squared error
$L=\sum_{i=1}^4\bigl(f(x_i)-y_i\bigr)^2$.

### 2.2 Gradient dynamics (one layer)

Write the normal in polar form $w=r\,(\cos\theta,\sin\theta)$.
Under gradient descent with stepsize $\eta$

$$
\dot r \;=\;-\eta\,\partial_r L,
\qquad
\dot\theta \;=\;-\eta\,\frac{1}{r}\,\partial_\theta L,
\qquad
\dot b \;=\;-\eta\,\partial_b L .
$$

We will analyse the fixed points $(r^*,\theta^*,b^*)$ and the local
Jacobian $J=\mathrm{D}(\dot r,\dot\theta,\dot b)$ to establish
stability.

---

## 3 Equilibrium analysis

### 3.1 Absolute-value unit

**Active set** For the *correct* separator we want

$$
w^\top x_{1,2}+b>0,\qquad
w^\top x_{3,4}+b<0 .
\tag{1}
$$

One consistent solution is

$$
w^* \;=\; r^*\!\begin{pmatrix}-1\\ 1\end{pmatrix},
\qquad
b^*=0,
\qquad
r^*=\frac{1}{\sqrt2}.
$$

---

**Theorem 1 (Unique stable equilibrium).**
$(w^*,b^*)$ is the *only* critical point of $L_{\text{abs}}$; its
Hessian is positive-definite, hence it is a strict (and global) minimum.

*Proof sketch.*  Using (1) we have
$f_i'=\text{sign}(w^{*\top}x_i)\in\{+1,-1\}$.
Write the gradient conditions

$$
\sum_i(f_i-y_i)f_i'x_i = 0,\quad
\sum_i(f_i-y_i)f_i' = 0 .
$$

Because $\{x_i\,f_i'\}$ span $\mathbb R^2$ there is a unique
solution, which is exactly $(w^*,b^*)$.  Expanding $L$ to second
order gives $\nabla^2 L = 4I_3$ at that point. ∎

---

### 3.2 Single-ReLU unit

Write $f(x)=\max\{0,z\}$ with $z=w^\top x+b$.

**Proposition 1.**
The only critical points of $L_{\text{relu}}$ are *degenerate
collapses* where $z_i\le0$ for all $i$.  All have non-zero loss
$\tfrac12$.

*Proof.*
If any training example is active, say $z_k>0$, then
$\partial_b L = 2z_k\neq0$; hence no stationary point has active
examples.  With every $z_i\le0$ we have $f(x_i)=0$ for all $i$,
so the gradient vanishes. ∎

These collapses are attractors: the Jacobian eigenvalues are strictly
negative in the $b$ and radial directions and exactly zero in
$\theta$ (rotation is uncontrolled).

---

### 3.3 Stability inversion principle

Let

$$
\theta_{\mathrm{NT}}=\frac{\pi}{4},\qquad
\theta_{\mathrm{ABS}}=\frac{3\pi}{4}.
$$

| model     | $\theta$-curvature at $(\theta,\,b=0)$                                         |
| --------- | ------------------------------------------------------------------------------ |
| **abs1**  | $\partial_{\!\theta\theta}^2 L(\theta_{\mathrm{NT}})<0$ (unstable)             |
|           | $\partial_{\!\theta\theta}^2 L(\theta_{\mathrm{ABS}})>0$ (stable)              |
| **relu1** | $\partial_{\!\theta\theta}^2 L(\theta_{\mathrm{NT}})>0$ (stable if $b$ frozen) |
|           | but $\partial_b L\neq0$ ⇒ drifts to collapse                                   |

**Interpretation.**
The *mirror channel* in abs1 supplies an **opposing torque** that turns
$\theta_{\mathrm{NT}}$ into a repeller and makes
$\theta_{\mathrm{ABS}}$ the single attracting basin.
Removing that channel flips the sign of the angular feedback and leaves
the bias direction uncontrolled—so training slides into the degenerate
all-off state.

---

## 4 Control-theoretic diagnostics

### 4.1 Dead-data coverage

For a layer with pre-activations $z_{ij}$ define

$$
m_i \;=\;\max_j z_{ij},\qquad
D=\frac1N\sum_i\mathbf 1_{\,\{m_i\le0\}},
$$

“fraction of examples with *zero* gradient through this layer.”

### 4.2 Torque ratio

For each weight vector $w_j$

$$
g_j=\nabla_{w_j}L,\quad
g_r=\frac{w_j}{\|w_j\|}\,\bigl(\frac{w_j}{\|w_j\|}\bigr)^\top g_j,
\quad
g_t=g_j-g_r,
\qquad
\tau_j=\frac{\|g_t\|}{\|g_j\|}.
$$

Aggregate (mean / 10-th percentile) over $j$ to get a layer-level
$\tau$.  Values $\tau\ll 1$ signal a **no-torque** condition.

---

## 5 Self-healing loop

| condition | trigger                                  | one-shot correction                                                                                  |
| --------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Dead-data | $D>D_{\text{th}}$                        | **Bias warm-up**<br>$b\gets b+\delta\cdot\mathbf 1$ where $\delta=-\min_i m_i + \varepsilon$         |
| No-torque | $\tau<\tau_{\text{th}}$ & $\|g\|>\gamma$ | **Angular noise**<br>$w\gets w+\alpha\,g_t$ <br>or **temporary Leaky-ReLU** (slope 0.01 for T steps) |

Re-evaluate $D,\tau$ after the intervention; escalate only if the
metric is still out of bounds.

---

## 6 Extending to deep networks

* Forward hooks capture $z_{ij}$ cheaply.
* Back-prop hooks expose $g_j$.
* Compute $D,\tau$ per layer once every $K$ steps.
* Interventions stay layer-local (bias tweak, angular perturb).
* Empirically, monitoring only the first 3–5 epochs suffices; later
  layers self-correct once early geometry is healthy.

---

## 7 Concluding principle

> **Symmetry supplies balanced torques.**
> Removing a mirror branch flips the feedback sign: unstable saddles in
> a symmetric architecture become degenerate attractors in an
> asymmetric one, unless we guarantee (i) every example has an active
> path, and (ii) every weight receives a tangential gradient.

Dead-data coverage and torque ratio are *general* health indicators for
any ReLU-style network; light-touch interventions (bias warm-up,
leaky slopes, angular noise) turn them into an on-line controller that
keeps training safely away from degenerate basins—scaling from XOR toys
to full deep nets.

---

# Comparison to training_dynamics_qualitative_reasoning

### **1. Core Philosophy and Goal**

  * **Theoretical Framework:** This is a **top-down, causal model**. Its goal is to create a complete "phase portrait" of all possible long-term behaviors by analyzing the fundamental mathematical properties (Jacobians, Hessians, eigenvalues) of the gradient descent update rule. It seeks to *prove* the existence of and conditions for degenerate states.
  * **Python Code (`monitor.py`):** This is a **bottom-up, symptomatic monitor**. Its goal is to be a practical, computationally cheap "dashboard" that flags observable, real-time *symptoms* correlated with poor training dynamics. It doesn't prove anything; it raises alarms based on heuristics.

### **2. How They Define and Detect "Degeneracy"**

This is the most critical point of comparison. They look for different things that are ultimately related.

| Theoretical Concept of Degeneracy | How the `monitor.py` Code Approximates It |
| :--- | :--- |
| **Non-hyperbolic Plateau (Degenerate state with `|λ| ≈ 1`)**: A region where gradients and curvature are near-zero, causing training to stall or drift without improvement. The parameter update `Δθ` is minimal. | **`bias_drift < 1e-4` (Bias Freeze):** This is a direct, empirical measurement of a stalled update. If the bias parameters are not changing step-to-step, it's a strong symptom of being on a flat plateau. \<br\>\<br\> **`torque_ratio < 0.01` (No-Torque Trap):** This is a more subtle heuristic. A low "torque" means the gradient `∇W` is mostly parallel to the weight vector `W`. This implies the update is only changing the *length* of the weight vector, not its *direction*. This can be a specific type of training trap where a neuron's feature direction is frozen, which is a form of degenerate behavior. |
| **Degenerate Sink (Zero-Collapse):** A state where neuron parameters `θ` converge to zero, effectively removing the neuron from the network. It's a valid fixed point, but one that carries no information. | **`find_dead_neurons(...)`:** This directly checks for the *effect* of a collapsed neuron. If a neuron's pre-activation `Wx+b` is negative for all inputs in a batch, it's considered "dead" for that batch. If this persists, the neuron has likely collapsed. \<br\>\<br\> **`compute_dead_data_fraction(...)`:** This detects a more catastrophic collapse. If for some inputs *all* ReLUs are off, the network's output is constant for those inputs. This is an extreme symptom that the model has entered a trivial, degenerate state for part of the data manifold. |
| **Hyperbolic Source (Divergence `||θ|| → ∞`):** An unstable state where parameters grow without bound. | The provided code **does not** explicitly detect this. Its focus is on "stalling" and "collapse," not divergence. One could add a check for exploding parameter or gradient norms to cover this case. |

### **3. Methodology and Computational Cost**

| Aspect | Theoretical Framework | `monitor.py` Code |
| :--- | :--- | :--- |
| **Information Used** | **Second-order** (Hessian `∇²L`) and first-order (gradient `∇L`). | **Zeroth-order** (activations, parameter values) and **first-order** (gradients). It avoids second-order data entirely. |
| **Computational Cost** | **Prohibitively High.** Computing the Hessian and its eigenvalues for every neuron is not feasible for any non-trivial network. | **Very Low.** It uses values already computed during the forward/backward pass (activations, gradients) and performs simple vector operations (norms, dot products). This is its key design feature. |
| **Implementation** | Purely mathematical; difficult to translate directly into scalable code. | Practical and straightforward using PyTorch hooks. Designed for real-world use. |

### **4. Strengths and Weaknesses**

**Theoretical Framework:**

  * **Strengths:**
      * **Rigorous and Explanatory:** Provides deep insight into *why* training fails.
      * **Complete:** Offers a full taxonomy of all possible behaviors.
      * **Predictive:** Can predict instability based on `α` and `β`.
  * **Weaknesses:**
      * **Computationally Intractable:** Cannot be implemented directly.
      * **Idealized:** Assumes full-batch gradient descent and a simple MSE loss, which may not perfectly model complex, real-world training with Adam, etc.

**`monitor.py` Code:**

  * **Strengths:**
      * **Practical and Fast:** Designed to run in real-time with negligible overhead.
      * **Actionable:** Provides simple metrics that a practitioner can easily track.
      * **General:** Works with any ReLU-like network and optimizer.
  * **Weaknesses:**
      * **Heuristic and Symptomatic:** Detects correlation, not causation. A low torque ratio might not always be a problem; it's just a flag.
      * **Ambiguous:** The "magic number" thresholds (e.g., `0.01`) are arbitrary and may need tuning for different models or datasets.
      * **Incomplete:** Doesn't cover all failure modes (like divergence) and its metrics are indirect proxies for the underlying mathematical states.

### **Conclusion: Theory Guides, Code Monitors**

The two approaches are not competitors; they are **perfect complements**.

  * The **theoretical framework** provides the fundamental justification for *why* we should be worried about things like "frozen parameters" or "dead neurons." It tells us that these are not just quirks but manifestations of well-defined mathematical phenomena (non-hyperbolic fixed points, degenerate sinks).
  * The **`monitor.py` code** acts as the practical sensor suite. Knowing that Hessians are too expensive to compute, it cleverly devises cheap, first-order proxies to detect the *symptoms* that the theory predicts.

You can think of it like this: The theory is the medical textbook that explains the pathology of a disease. The code is the thermometer and blood pressure cuff the doctor uses to quickly check for symptoms of that disease during a check-up. The thermometer doesn't prove you have the flu, but a high reading (the symptom) is a strong, actionable indicator that is justified by the underlying medical theory.