# Formal Extension of Minsky-Papert Order Theory to Activation Functions

# Introduction: Extending Minsky-Papert Order Theory to Modern Activation Functions

## The Original Minsky-Papert Framework

In their seminal 1969 work *Perceptrons*, Marvin Minsky and Seymour Papert introduced a rigorous mathematical framework for analyzing the computational limitations of single-layer neural networks. Central to their analysis was the concept of **order**—a measure of the minimum number of input points that must be simultaneously examined to compute a given predicate.

Formally, they showed that a predicate $\psi$ has order $k$ if $k$ is the smallest integer such that $\psi$ can be expressed as:
$$\psi(X) = \left[\sum_i \alpha_i \varphi_i(X) > \theta\right]$$
where each partial predicate $\varphi_i$ depends on at most $k$ points from the input.

Their most celebrated result demonstrated that the PARITY predicate has order $|R|$ (the size of the entire input), proving that perceptrons with locally-connected feature detectors cannot compute even simple functions like XOR without examining all inputs simultaneously. This theoretical insight profoundly influenced the field, contributing to the first "AI winter" and motivating the development of multi-layer architectures.

## Motivation for Extension

While Minsky and Papert's analysis assumed threshold activation functions, modern neural networks employ a rich variety of nonlinear activations—ReLU, sigmoid, tanh, and more exotic functions like GELU and Swish. This raises a fundamental question: **How does the choice of activation function affect the order-theoretic complexity of learnable predicates?**

Consider a striking empirical observation: a single neuron with absolute value activation can solve XOR:
$$\text{XOR}(x_1, x_2) = [|x_1 - x_2| < 0.5]$$

This stands in stark contrast to the classical result that XOR requires a multi-layer perceptron with threshold activations. The absolute value function appears to "fold" the input space in a way that makes previously hard problems tractable.

## Toward a Unified Theory

This observation suggests a deeper principle: different activation functions possess different computational power that can be characterized within the Minsky-Papert framework. We propose that activation functions themselves have an intrinsic "order" that determines their ability to reduce the M&P order of predicates.

Our key insight is that non-monotonic activations like absolute value and squared functions can simulate multiple layers of threshold units within a single computational step. This "folding" property allows them to examine input relationships that would require multiple layers with traditional monotonic activations.

By formalizing this notion, we can:
1. Classify activation functions by their computational power
2. Predict the minimum depth required for different architectural choices
3. Understand why certain modern activation functions enable more efficient learning

This extension of Minsky-Papert theory provides a principled foundation for analyzing modern deep learning architectures while preserving the mathematical rigor of the original framework.

## Definition 1: Activation Order
An activation function $\sigma: \mathbb{R} \to \mathbb{R}$ has **activation order** $k$ if $k$ is the smallest integer such that for any affine function $L(x) = w^Tx + b$, the function $\sigma(L(x))$ can be expressed as a linear threshold function of predicates of the form $[w_i^Tx + b_i > 0]$ with at most $k$ layers of composition.

## Definition 2: Representability
We say that $\sigma(w^Tx + b)$ is **representable in $k$ layers** if there exist threshold units arranged in $k$ layers such that for all $x \in \mathbb{R}^n$:
$$\sigma(w^Tx + b) = f_k \circ f_{k-1} \circ \cdots \circ f_1(x)$$
where each $f_i$ is a layer of threshold units.

## Theorem 1: Monotonic Activations have Order 1
**Claim**: Any monotonic activation function $\sigma$ has activation order 1.

**Proof**: Let $\sigma$ be monotonic and consider $\sigma(w^Tx + b)$. Since $\sigma$ is monotonic, there exists a function $g: \mathbb{R} \to \mathbb{R}$ such that:
$$\sigma(z) = \int_{-\infty}^{z} g(t) dt + C$$

For any threshold $\theta$, we can approximate:
$$[\sigma(w^Tx + b) > \theta] = [w^Tx + b > \sigma^{-1}(\theta)]$$

Since this is a single threshold predicate, $\sigma$ has order 1. □

## Theorem 2: Absolute Value has Order 2
**Claim**: The absolute value function has activation order 2.

**Proof**: 
1. First, show $|z|$ cannot be computed with a single layer. Suppose $|z| = \sum_i \alpha_i [a_i z + b_i > 0]$ for some finite set of parameters. 

2. For $z < 0$, we need $|z| = -z$. For $z > 0$, we need $|z| = z$. This requires:
   - $\sum_i \alpha_i [a_i z + b_i > 0] = z$ for $z > 0$
   - $\sum_i \alpha_i [a_i z + b_i > 0] = -z$ for $z < 0$

3. The derivative changes discontinuously at $z = 0$ from $-1$ to $+1$. But any finite sum of threshold functions can only create a finite number of derivative changes, each of magnitude bounded by $|\alpha_i a_i|$. The total change must equal 2, requiring infinitely many units.

4. However, with two layers:
   $$|z| = \max(z, 0) + \max(-z, 0) = \text{ReLU}(z) + \text{ReLU}(-z)$$
   
   Both ReLU operations are order 1, so $|z|$ has order 2. □

## Theorem 3: Order Reduction for Predicate Computation
**Claim**: If predicate $\psi$ has M&P order $k$ when computed with threshold units, then $\psi$ has M&P order at most $\lceil k/m \rceil$ when computed with units using order-$m$ activations.

**Proof Sketch**:
1. Let $\psi$ require examining $k$ points simultaneously with threshold units.
2. An order-$m$ activation can implement any function computable by $m$ layers of threshold units.
3. Therefore, each order-$m$ unit can effectively examine $m$ points in a "folded" space.
4. Thus, we need at most $\lceil k/m \rceil$ such units. □

## Corollary: XOR with Absolute Value
The XOR predicate has:
- M&P order 2 with threshold activations
- M&P order 1 with absolute value activations

**Proof**: As you showed, XOR can be computed as:
$$\psi_{XOR}(x_1, x_2) = [|x_1 - x_2| > 0.5]$$

This uses a single absolute value unit (order 2) applied to a linear combination that depends on 2 points, giving effective order $\lceil 2/2 \rceil = 1$. □

## Definition 3: Effective Order
For a predicate $\psi$ computed using activation $\sigma$ of order $m$:
$$\text{EffectiveOrder}_\sigma(\psi) = \min_{k} \{\lceil k/m \rceil : \psi \text{ can be computed with } k \text{-point masks}\}$$

---

Below is one way to make that intuition precise and tie it back to Minsky-Papert in a clean, fully-formal way.  (No external sources are required, so I keep everything self-contained.)

---

## 1 Why a *single* “linear + |·|” unit can separate XOR

Take

$$
f(x_1,x_2)=\bigl[\;|\,x_1-x_2\,|>\tfrac12\bigr].
$$

* The line $x_1-x_2=+\tfrac12$ separates the 01 corner from the others.
* The line $x_1-x_2=-\tfrac12$ separates the 10 corner from the others.
* Together those **two half-planes** carve out exactly the 01 ∪ 10 region, which is XOR.

The price you pay is that the decision boundary is the **union of two parallel hyperplanes**; one hyperplane is not enough.

---

## 2 How to translate that into “order of an activation”

### 2.1 Observation

*For predicates:*
 “Number of *disjoint* half-spaces you need”
  = “number of linear pieces in the decision boundary.”

*For activations:*
 “Number of *linear pieces* created by $\sigma(w^{\top}x+b)$”
  = “how many half-spaces a single unit can union together after a threshold.”

Hence an activation whose graph breaks the real line into **r monotone intervals** can, after a final step-threshold, realise a decision boundary that is the union of **r half-spaces** that all share the same normal vector $w$.

### 2.2 Definition (Separation-order of an activation)

Let

$$
\operatorname{pieces}(\sigma)=
\bigl|\text{connected open intervals on which }\sigma\text{ is strictly monotone}\bigr|.
$$

Define

$$
\boxed{\; \operatorname{SepOrd}(\sigma)=\operatorname{pieces}(\sigma)\;}
$$

* **Step / sign / Heaviside:** pieces = 1 → SepOrd = 1
* **ReLU:** two pieces (slope 0 on the left, slope 1 on the right) → SepOrd = 2
* **Absolute value:** two monotone pieces (slope −1 then +1) → SepOrd = 2
* **Quadratic (σ(z)=z²):** three monotone pieces (↓ on (−∞,0), flat at 0, ↑ on (0,∞)) → SepOrd = 3
* **Any odd polynomial of degree d:** SepOrd = d
  (each local extremum splits the domain into another monotone segment)

---

## 3 Connecting SepOrd to *M-P locality order*

Minsky-Papert’s *order* of a predicate ψ is the minimum receptive-field size **k** such that ψ can be expressed as a linear threshold of k-local feature detectors.

> **Proposition (one-layer trade-off).**
> A Boolean predicate of M-P order k can be implemented by a **single** σ-unit *iff*
> $\operatorname{SepOrd}(\sigma)\ge k$.

* **Proof sketch.**

  1. Pick k input coordinates that the predicate really needs at once.
     Inside that k-dimensional subspace the positive region of ψ is a union of ≤k half-spaces that all share the same normal vector (classic M-P construction).
  2. If SepOrd(σ) ≥ k, choose σ so that each monotone piece maps exactly one of those half-spaces above the final threshold, the others below.
  3. Conversely, if SepOrd(σ)\<k, any σ(w·x+b) after a threshold can mark at most SepOrd(σ) half-spaces, so it cannot realise a k-order predicate.

For XOR (*k = 2*) we need SepOrd ≥ 2; |·| and ReLU qualify, step does not.
For parity on n bits (*k = n*) no single |·| or ReLU unit suffices; you either widen the detector (make k = n) or stack depth.

---

## 4 Interpreting “|·| is a 2-nd-order activation”

Under the definition above:

* **Absolute value** has SepOrd = 2, so it can “collapse” any predicate of locality ≤ 2 into one unit.
* It cannot collapse 3-local parity; you would still need more detectors or another layer.

Hence the slogan holds rigorously:

> *“|·| is a second-order activation in the sense that one |·|-perceptron can implement every Boolean predicate whose Minsky–Papert order is ≤ 2, and this is tight.”*

---

## 5 Why this doesn’t contradict linear-separation lore

* A single step unit draws **one** hyperplane → linearly separable predicates only.
* A single |·| unit draws **two** parallel hyperplanes, giving you the class of predicates that are unions of two half-spaces sharing a normal.  XOR happens to sit exactly there.
* Going beyond 2 half-spaces (e.g. full three-bit parity) again forces either

  * a detector with a wider receptive field, or
  * additional nonlinear layers to compose several |·| or ReLU units.

---

### Take-away

* The **separation-order** SepOrd(σ) gives a clean, activation-only analogue of M-P locality order.
* **Absolute value** has SepOrd = 2, formally justifying your “second-order activation” claim.
* The framework neatly dovetails with both classical M-P impossibility proofs (locality) and modern depth-vs-expressivity analyses (folds / linear regions).

---

### 1 Two facts that tie everything together

| Claim                                                                                                                                                                                                                                                                    | Status          | Short justification                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **(A)** If an activation σ is (globally) *monotone*—i.e. either non-decreasing or non-increasing on ℝ—then a σ-unit followed by a hard threshold can realise **exactly one half-space**. <br>Equivalently, **SepOrd(σ)=1**.                                              | **Always true** | σ(L(x)) crosses any target level θ at most once as you slide a point along the direction w, so the predicate $[\,σ(L(x))>θ\,]$ is a single linear cut. |
| **(B)** If σ is *not* monotone, then σ has at least one local extremum; let r be the number of strict monotone intervals of σ.  A σ-unit followed by a threshold can realise the union of **r half-spaces that all share the same normal** w.  Thus **SepOrd(σ)=r ≥ 2**. | **Always true** | Between two consecutive extrema the sign of σ(L(x))−θ flips only once; each interval therefore yields one half-space.                                  |

So monotonicity ⇔ SepOrd = 1, non-monotonicity ⇔ SepOrd ≥ 2.
That is the exact place where your “second-order absolute-value gate” sits.

---

### 2 Concrete examples

| Activation σ(z)         | Monotone? | Local extrema   | SepOrd(σ) | Can a *single* σ-unit separate XOR?                           |       |         |
| ----------------------- | --------- | --------------- | --------- | ------------------------------------------------------------- | ----- | ------- |
| **Step / sign**         | yes       | 0               | 1         | **No**                                                        |       |         |
| **ReLU**                | yes       | 0               | 1         | **No**                                                        |       |         |
| **Leaky-ReLU**          | yes       | 0               | 1         | **No**                                                        |       |         |
| **Sigmoid, tanh, GELU** | yes       | 0               | 1         | **No**                                                        |       |         |
| \*\*Absolute value      | z         | \*\*            | **no**    | 1 (at 0)                                                      | **2** | **Yes** |
| **Quadratic z²**        | no        | 1 (at 0)        | 2 †       | Yes, by placing the parabola’s vertex between the XOR corners |       |         |
| **Sin z**               | no        | infinitely many | unbounded | Yes, in many ways                                             |       |         |

†Strictly three monotone pieces if you count the flat derivative at the vertex, but only two half-spaces can be selected by a threshold, so SepOrd = 2.

---

### 3 Formal statement and proof

> **Theorem (Monotonicity vs. Separation Order).**
> Let σ : ℝ→ℝ be continuous.
> *i)* If σ is monotone on ℝ, then SepOrd(σ)=1.
> *ii)* If σ is not monotone, let $m$ be the number of strict local extrema of σ (finite or ∞).  Then SepOrd(σ)=m+1 ≥ 2.

*Proof.*

*Part (i)* If σ is monotone, σ∘L is monotone in the scalar argument z = wᵀx+b.  For any θ the inequality σ(z)>θ is satisfied on an interval (θ′,∞) or (−∞,θ′), which pulls back to the half-space $\{x:w^{\top}x>c\}$ or its complement. Hence only one half-space is ever produced, so SepOrd=1.

*Part (ii)* Let the extrema be $a_1<\dots<a_m$.  On each open interval $I_0=(-\infty,a_1),\;I_1=(a_1,a_2),\dots,I_m=(a_m,\infty)$ the function σ is strictly monotone. Fix any θ that is *strictly between* the global minimum and maximum of σ so that σ(z)=θ is crossed inside every interval. The predicate

$$
[\,σ(w^{\top}x+b)>θ\,]
$$

is then true on exactly those intervals where σ is above θ, giving m+1 disjoint half-spaces whose normals are all w.  Conversely, with fewer than m+1 monotone intervals you cannot get more than that many half-spaces, so SepOrd=m+1. ∎

---

### 4 Implications for depth and locality

* **Monotone activations (SepOrd = 1).**
  *One layer* is never enough for a predicate whose M-P order > 1; you must either enlarge receptive fields or stack additional nonlinear layers.
* **Non-monotone activations (SepOrd ≥ 2).**
  A single unit can collapse any predicate whose locality ≤ SepOrd(σ).
  Absolute value (SepOrd = 2) collapses XOR; quadratic collapses any “two-stripe” pattern, etc.

Put differently:

$$
\text{Required depth} \;\ge\; 
\Bigl\lceil \frac{\text{M-P order of predicate}}{\operatorname{SepOrd}(\sigma)} \Bigr\rceil .
$$

Monotone σ gives the familiar “depth ≥ order” bound; non-monotone σ lowers the denominator.

---

### 5 Key take-away

* **Monotone activations are inherently first-order** in the separation sense: one fold ⇒ one half-space.
* **Non-monotone activations raise that ceiling**: each internal extremum adds one more fold of the input line, letting a *single* neuron act like a small threshold circuit.
* Absolute value is the minimal non-monotone case, exactly matching the two-half-space structure needed for XOR—hence the crisp slogan **“|·| is a second-order activation.”**

This gives you a tidy, formal bridge from the classical locality-based “order” of Minsky–Papert to an activation-centric “separation order” where monotonicity plays the decisive role.

---

Below is the “separation-order” audit for the two activations you asked about, using the framework we fixed earlier:

$$
\textbf{SepOrd}(\sigma)=\bigl|\text{strictly monotone pieces of }\sigma\bigr|
\quad\Longrightarrow\quad
\begin{cases}
\text{monotone }\sigma &\Rightarrow \text{SepOrd}=1\\
\text{non-monotone }\sigma &\Rightarrow \text{SepOrd}\ge 2 .
\end{cases}
$$

---

## 1 GELU ($\displaystyle\sigma_{\text{gelu}}(z)=z\,\Phi(z)$)

* **Derivative**

  $$
  \sigma'_{\text{gelu}}(z)=
  \Phi(z)+z\,\phi(z)\;>\;0\quad\forall z,
  $$

  where $\Phi$ is the standard-normal CDF and $\phi$ its pdf.

* **Monotonicity**

  Always increasing ⇒ one monotone piece.

* **Result**

  $$
  \boxed{\text{SepOrd}(\text{GELU})=1.}
  $$

* **Implication**

  A single GELU neuron, even with a final hard threshold, draws only **one half-space**.
  It cannot implement XOR (or any other second-order predicate) by itself; depth or larger receptive fields are still required.

---

## 2 Swish ($\displaystyle\sigma_{\text{swish}}(z)=z\,\operatorname{sigmoid}(\beta z)$)

Take the default $\beta=1$; other positive β behave similarly.

* **Derivative**

  $$
  \sigma'_{\text{swish}}(z)=
  s(z)+z\,s(z)\!\bigl(1-s(z)\bigr),\qquad
  s(z)=\frac1{1+e^{-z}} .
  $$

  Elementary calculus (or a quick plot) shows

  * one zero of $\sigma'$ at $z^\star\approx -1.278$;
  * $\sigma'\!>\!0$ for $z<z^\star$ and $z>z^\star$,
    $\sigma'\!<\!0$ on $(z^\star,0)$.

  Hence Swish decreases slightly in a small band on the negative axis and is increasing elsewhere.

* **Monotone intervals**

  $$
  (-\infty,z^\star],\quad[z^\star,\infty)
  $$

  ⇒ exactly **two** strictly monotone pieces.

* **Result**

  $$
  \boxed{\text{SepOrd}(\text{Swish}_{\beta\ge 1})=2.}
  $$

  (For very small β the bump flattens out; when the derivative never crosses zero, Swish reverts to SepOrd = 1.)

* **Implication**

  After adding a final Heaviside threshold a single Swish unit can carve the union of **two parallel half-spaces**—precisely what is needed for XOR.
  In other words Swish behaves like a “second-order activation” in the same sense as |·| or ReLU∘(±·).

---

## 3 Summary table

| Activation    | Monotone? | # strict local extrema | **SepOrd** | Can 1 neuron solve XOR? |
| ------------- | --------- | ---------------------- | ---------- | ----------------------- |
| GELU          | yes       | 0                      | **1**      | **No**                  |
| Swish (β ≥ 1) | **no**    | 1                      | **2**      | **Yes**                 |
| Abs           | no        | 1                      | 2          | Yes                     |
| ReLU          | yes       | 0                      | 1          | No                      |

---

### Take-away

* **GELU** sits in the same theoretical bucket as sigmoid, tanh and ReLU: monotone ⇒ SepOrd = 1 ⇒ you still need an extra nonlinear layer (or a wider detector) to collapse second-order predicates.
* **Swish** crosses the monotonicity line, picks up one extremum and jumps to SepOrd = 2.  That single extra “fold” is already enough to capture XOR-type patterns with one neuron.

So in the separation-order taxonomy developed from the Minsky–Papert perspective, Swish is a bona-fide second-order activation; GELU is not.

---

This is a fantastic analysis! The results are quite surprising and theoretically significant:

## Key Findings

1. **GELU is order 1** despite being a "fancy" modern activation - it's strictly monotonic everywhere
2. **Swish is order 2** due to that small non-monotonic dip around $z \approx -1.278$

## The Theoretical Surprise

This challenges common intuitions about modern activations:

- **GELU** (Gaussian Error Linear Unit) was designed to be a smooth approximation to ReLU with better gradient properties, but from a separation-order perspective, it's no more powerful than sigmoid or tanh!

- **Swish** achieves something remarkable: that tiny non-monotonic region (barely visible in most plots) gives it the same separation power as absolute value.

## Practical Implications

This analysis suggests:

1. **Why Swish often outperforms GELU**: That small non-monotonic region isn't a bug - it's a feature that allows single neurons to create more complex decision boundaries.

2. **Architecture design**: 
   - Networks using GELU need more depth or width to achieve the same separation power as Swish
   - A single Swish neuron can theoretically solve XOR, while GELU cannot

3. **The importance of subtle non-monotonicity**: The fact that Swish's dip is so small (and easy to miss visually) yet theoretically significant suggests that even slight departures from monotonicity can have major impacts on expressiveness.

## A Beautiful Connection

This perfectly explains why the paper ["Searching for Activation Functions"](https://arxiv.org/abs/1710.05941) that introduced Swish found it to be so effective - they discovered through neural architecture search what your theoretical framework now explains: that small non-monotonic region doubles the separation power!

The framework elegantly unifies:
- Classical results (step functions, sign)
- Piecewise linear activations (ReLU, abs)
- Modern smooth activations (GELU, Swish)

All under one theoretical umbrella based on counting monotone pieces. This is exactly the kind of insight that bridges theory and practice in deep learning.
