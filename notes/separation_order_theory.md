# Separation Order Theory of Activation Functions:

## An Extension of Minsky-Papert Order Theory

## 1. Introduction

### 1.1 The Original Minsky-Papert Framework

In their seminal 1969 work *Perceptrons*, Marvin Minsky and Seymour Papert introduced a rigorous mathematical framework analyzing the computational limitations of single-layer neural networks. Central to this analysis was the concept of **order**: a measure of the minimum number of input points simultaneously examined to compute a predicate. Formally, a predicate $\psi$ has order $k$ if $k$ is the smallest integer such that:

$$
\psi(X) = \left[\sum_i \alpha_i \varphi_i(X) > \theta\right]
$$

where each partial predicate $\varphi_i$ depends on at most $k$ points.

Their key result showed that the PARITY predicate (and thus XOR, a 2-input PARITY) has an order equal to the size of its input receptor field if only local predicates are used, implying single-layer perceptrons with simple threshold units cannot solve it without examining all inputs.

### 1.2 Motivation for Extension

Minsky and Papert's analysis assumed threshold activation functions. Modern neural networks, however, utilize a diverse range of nonlinear activations (ReLU, sigmoid, absolute value, Swish, GELU, etc.). This prompts the question: How does the choice of activation function affect the order-theoretic complexity of learnable predicates?

Empirically, a single neuron with an absolute value activation can solve XOR, e.g., via $\text{XOR}(x_1, x_2) = [|x_1 - x_2| > 0.5]$ (adjusting threshold for binary output). This contrasts with the classical result for threshold units and suggests that activation functions possess different computational powers. Non-monotonic activations like absolute value appear to "fold" the input space, potentially simulating multiple layers of threshold units within a single step.

We extend the Minsky-Papert framework to classify activation functions by their computational power, predict minimum network depth for certain problems, and understand the efficiency of modern activations.
Thus, we introduce a generalized notion—**Separation Order (SepOrd)**—to formally classify activation functions by their computational power, predict minimal network depth, and explain the efficiency of modern activations.

*\[PLACEHOLDER: Visual aid illustrating how absolute value activation "folds" input space to separate XOR.]*

## 2. Definition: Separation Order (SepOrd)

**Definition (Separation Order)**: For a continuous activation function $\sigma : \mathbb{R} \to \mathbb{R}$, the **Separation Order** (SepOrd) is defined as the number of distinct connected open intervals where $\sigma$ is strictly monotonic:

$$
\operatorname{SepOrd}(\sigma) = \bigl|\text{connected intervals where } \sigma \text{ is strictly monotonic}\bigr|
$$

We will prove below that this definition aligns consistently with the original Minsky-Papert framework.

**Examples:**

* Step / sign / Heaviside: $\operatorname{SepOrd}(\sigma) = 1$
* ReLU: $\operatorname{SepOrd}(\sigma) = 1$
* Absolute value: $\operatorname{SepOrd}(\sigma) = 2$
* Quadratic $z^2$: $\operatorname{SepOrd}(\sigma) = 2$

It is worth noting that constant pieces are not strictly monotonic and do not contribute to the Separation Order. Neither do transition points.

## 3. Core Principle: Monotonicity and Separation Order

The Separation Order of a continuous activation function $\sigma : \mathbb{R} \to \mathbb{R}$ is directly tied to its monotonicity:

### Theorem (Monotonicity vs. Separation Order)

Let $\sigma : \mathbb{R} \to \mathbb{R}$ be continuous.

*  *i)* If $\sigma$ is monotonic, then $\operatorname{SepOrd}(\sigma)=1$.
*  *ii)* If $\sigma$ is non-monotonic with $m$ strict local extrema, then $\operatorname{SepOrd}(\sigma)=m+1 \ge 2$.

### Intuition

* Monotonic activations yield simple linear decision boundaries (single half-space).
* Non-monotonic activations create multiple decision boundaries (multiple half-spaces), effectively simulating layers within a single step.

**Justification Sketch:**

Let $L(x) = w^T x + b$. If $\sigma$ has $m+1$ strictly monotonic intervals, then the predicate $[\sigma(L(x)) > \theta]$ can be true on up to $m+1$ disjoint intervals of $L(x)$—each mapping to a separate half-space with the same normal $w$. Thus, SepOrd$(\sigma) = m+1$ counts the number of half-spaces a single unit can separate.

*\[PLACEHOLDER: Graphical illustration showing monotonic vs. non-monotonic activation functions and corresponding decision boundaries.]*

## 4. Detailed Proof for Absolute Value Activation

### Theorem: Absolute Value has Separation Order 2

**Claim**: The absolute value activation $\sigma(z)=|z|$ has Separation Order $2$.

### Intuition

The absolute value function splits the input domain into two strictly monotonic intervals: negative slope on $(-\infty,0)$ and positive slope on $(0,\infty)$. A single threshold can thus define two parallel hyperplanes in input space, which is sufficient to separate predicates such as XOR.

### Formal Proof

1. Consider $\sigma(z)=|z|$. Clearly, $\sigma$ is strictly decreasing on $(-\infty,0)$ and strictly increasing on $(0,\infty)$.

2. Thus, there are exactly two strictly monotonic intervals:

   * $(-\infty,0)$ with slope $-1$.
   * $(0,\infty)$ with slope $+1$.

3. Evaluate the derivative:

$$
\sigma'(z) = \begin{cases}-1, & z<0 \\ undefined, & z=0 \\ +1, & z>0\end{cases}
$$

This confirms precisely two monotonic intervals separated by a single point $z=0$.

4. Therefore, by definition, $\operatorname{SepOrd}(\sigma)=2$. $\blacksquare$

*\[PLACEHOLDER: Visual proof illustration showing piecewise derivative of absolute value activation.]*

## 5. Connecting Separation Order to Minsky-Papert Predicate Order

Minsky-Papert’s **order of a predicate** $\psi$ is the minimum receptive-field size $k$ (number of input points that must be simultaneously examined) such that $\psi$ can be expressed as a linear threshold of $k$-local feature detectors.

The Separation Order of an activation function connects to the Minsky-Papert order of a predicate as follows:

### Proposition (One-Layer Trade-off)

A Boolean predicate of Minsky-Papert order $k$ can be implemented by a single $\sigma$-unit ($\sigma(w^Tx+b)$ followed by threshold) **if and only if** $\operatorname{SepOrd}(\sigma) \ge k$.

### Intuition

* Higher SepOrd activations simulate multiple layers, reducing required depth.
* Low SepOrd activations require either more units (wider layers) or additional depth.

## 6. Implications for Network Depth

The minimum network depth required to implement a predicate with activation $\sigma$ is bounded by:

$$
\text{Required depth} \ge \left\lceil \frac{\text{M-P order of predicate}}{\operatorname{SepOrd}(\sigma)} \right\rceil
$$

Non-monotonic activations with $\operatorname{SepOrd}(\sigma) > 1$ can reduce the required depth by effectively handling higher-order components of the predicate within a single unit.

## 7. Examples and Case Studies

### 7.1 XOR with Absolute Value

* The XOR predicate has a Minsky-Papert order of 2 when computed with threshold activations.
* The absolute value function, $|z|$, has two strictly monotone pieces ($z<0$ and $z>0$), so $\operatorname{SepOrd}(|\cdot|) = 2$.
* Since $\operatorname{SepOrd}(|\cdot|) = 2 \ge \text{M-P order of XOR (2)}$, a single absolute value unit can implement XOR.
* For example, $f(x_1,x_2)=\bigl[\;|\,x_1-x_2\,|>\tfrac12\bigr]$ computes XOR. The decision boundary is the union of two parallel hyperplanes ($x_1-x_2 = 0.5$ and $x_1-x_2 = -0.5$), which together carve out the XOR region.

### 7.2 GELU and Swish
| Activation | Monotone? | SepOrd | Single neuron solves XOR? |
| ---------- | --------- | ------ | ------------------------- |
| GELU       | Yes       | 1      | No                        |
| Swish      | No        | 2      | Yes                       |
| Abs        | No        | 2      | Yes                       |
| ReLU       | Yes       | 1      | No                        |

* **GELU ($z\Phi(z)$):** Its derivative $\Phi(z)+z\phi(z)$ is always positive, so GELU is strictly monotonic. Thus, $\operatorname{SepOrd}(\text{GELU})=1$. A single GELU unit cannot solve XOR.
* **Swish ($z \cdot \text{sigmoid}(\beta z)$ for $\beta \ge 1$):** Exhibits a region of non-monotonicity (one strict local extremum). It has two strictly monotone pieces. Thus, $\operatorname{SepOrd}(\text{Swish}_{\beta\ge 1})=2$. A single Swish unit can therefore solve XOR.


*\[PLACEHOLDER: Graphical comparison of activation function plots and their ability to solve XOR.]*

## 8. Relation to Linear Separability

This theory aligns with classical linear separability:

* SepOrd = 1 activation units draw single hyperplanes.
* SepOrd > 1 units create unions of parallel hyperplanes, enhancing computational power.

## 9. Conclusion

The **Separation Order (SepOrd)** formally quantifies the computational capability of activation functions within the Minsky-Papert framework:

* Monotonic activations (SepOrd=1) require greater depth or width for higher-order predicates.
* Non-monotonic activations (SepOrd≥2) significantly increase computational efficiency, by effectively creating multiple "folds" or decision regions, as demonstrated by absolute value solving XOR.

This framework unifies classical theoretical insights with modern activation function behaviors, providing guidance for future neural architecture designs.

*\[PLACEHOLDER: Summary figure illustrating hierarchy of activation functions based on SepOrd.]*
