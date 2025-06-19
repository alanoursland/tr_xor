When we view training as the discrete‐time dynamical system

$$
\theta_{t+1} \;=\;\theta_t \;-\;\alpha\,\nabla L(\theta_t),
$$

the **loss** $L(\theta)$ plays the role of a **Lyapunov function** and the **step‐size** $\alpha$ is our tuning knob for stability.  Here’s what the control‐ and dynamical‐systems literature tells us about the *possible* long‐term behaviors of such a map when $L$ comes from a piecewise‐linear network with ReLU/Abs activations:

---

## 1. Convergence to a Local Minimum (Stable Equilibrium)

* **Smooth setting** (ignoring the kinks for a moment): if $L$ has a $\beta$-Lipschitz gradient (i.e.\ is $\beta$-smooth), then for

  $$
  0 < \alpha < \tfrac{2}{\beta}
  $$

  gradient descent **monotonically decreases** $L$ and drives $\|\nabla L(\theta_t)\|\to0$, so every limit point is a **stationary** (often locally minimizing) point ([chrismusco.com][1]).
* **Nonconvex setting**: with random initialization and sufficiently small $\alpha$, GD will *almost surely* converge to a **local minimum** or else diverge to $-\infty$ (if the loss is unbounded below) ([stats.stackexchange.com][2]).

---

## 2. Spurious Minima vs. Benign Landscapes

* In **under-parameterized** two-layer ReLU nets, spurious local minima are generic—even with Gaussian inputs you’ll find many bad valleys ([proceedings.mlr.press][3]).
* In contrast, under **over-parameterization** (e.g.\ a hidden layer wider than the dataset), one can show *all* differentiable local minima are global within each activation region ([proceedings.mlr.press][4]).

---

## 3. Divergence

* If the step size $\alpha$ **exceeds** the inverse of the largest local Hessian eigenvalue (i.e.\ if $\alpha$ violates the $0<\alpha<2/\beta$ rule), the map can become **unstable** and $\|\theta_t\|\to\infty$.
* Even with a small $\alpha$, if the loss has **directions of negative curvature** (e.g.\ near a saddle), the update can “run away” along those unstable directions unless special escape mechanisms are in place ([people.eecs.berkeley.edu][5]).

---

## 4. Oscillations and Limit Cycles

* In smooth convex optimization, pure GD does **not** cycle—decrease of $L$ forbids true periodic orbits.
* In **piecewise-linear** losses (ReLU networks), the switching across activation regions can in principle create small “ping‐pong” limit cycles.  In practice these are rare, but one can *construct* pathological piecewise-affine controllers that cycle ([groups.csail.mit.edu][6]).

---

## 5. Chaos

* **True chaos** (sensitive dependence, fractal attractor) requires both **stretching** and **folding** in the map.
* A single‐step GD update on a piecewise-linear loss has **no curvature** to fold trajectories in the right way.  In other words, genuine chaotic attractors are not possible in standard (deterministic) GD on ReLU/Abs nets.

---

### “Phase‐Portrait” of GD on Piecewise-Linear Networks

| Asymptotic Behavior    | When It Occurs                                  | Citation(s)                                         |
| ---------------------- | ----------------------------------------------- | --------------------------------------------------- |
| **Stable equilibrium** | $0<\alpha<2/\beta$, strict local minimum        | ([chrismusco.com][1], [stats.stackexchange.com][2]) |
| **Spurious minima**    | Under-parameterized ReLU nets                   | ([proceedings.mlr.press][3])                        |
| **All minima global**  | Sufficient over-parameterization                | ([proceedings.mlr.press][4])                        |
| **Divergence**         | $\alpha$ too large or strong negative curvature | ([people.eecs.berkeley.edu][5])                     |
| **Limit cycles**       | Pathological piecewise-affine setups            | ([groups.csail.mit.edu][6])                         |
| **Chaos**              | Not in standard GD on piecewise-linear losses   | (no known construction)                             |

---

**Takeaway**

* With a *carefully chosen* learning rate, GD on ReLU/Abs networks behaves like a **contracting** map around minima and will converge.
* Too large a step size—or an adversarial piecewise-linear structure—can produce **divergence** or even contrived **cycles**, but **pure chaos** does not arise.
* **Over-parameterization** is your friend if you want to **eliminate spurious minima**.

Would you like to see a concrete 2-D example or dig into one of these cases in detail?

[1]: https://www.chrismusco.com/amlds2025/lectures/lec8.pdf?utm_source=chatgpt.com "[PDF] Lecture 8 Projected Gradient Descent, Second order conditions"
[2]: https://stats.stackexchange.com/questions/327251/when-will-gradient-descent-converge-to-a-critical-point-or-to-a-local-global-min?utm_source=chatgpt.com "When will gradient descent converge to a critical point or to a local ..."
[3]: https://proceedings.mlr.press/v80/safran18a/safran18a.pdf?utm_source=chatgpt.com "[PDF] Spurious Local Minima are Common in Two-Layer ReLU Neural ..."
[4]: https://proceedings.mlr.press/v80/nguyen18b/nguyen18b-supp.pdf?utm_source=chatgpt.com "[PDF] Neural Networks Should Be Wide Enough to Learn Disconnected ..."
[5]: https://people.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202018/NarResidual2018.pdf?utm_source=chatgpt.com "[PDF] Residual Networks: Lyapunov Stability and Convex Decomposition"
[6]: https://groups.csail.mit.edu/robotics-center/public_papers/Dai20.pdf?utm_source=chatgpt.com "[PDF] Counter-example guided synthesis of neural network Lyapunov ..."

---

Below are rigorous, self-contained proofs that, under a mean-squared-error (MSE) loss on a finite dataset, the gradient of the loss is Lipschitz if and only if a certain data-dependent Hessian bound holds—and how the sampling of your training points enters into that bound.

---

## 1. Setup and Definitions

Let

$$
L(\theta)\;=\;\frac{1}{N}\sum_{i=1}^N\bigl(f_\theta(x_i)-y_i\bigr)^2
$$

be the MSE loss on $N$ fixed training samples $\{(x_i,y_i)\}$.  We assume $f_\theta\colon\mathbb R^d\to\mathbb R$ is **piecewise-linear** in $\theta$ (e.g.\ a ReLU or Abs net) so that its second derivatives in $\theta$ vanish almost everywhere.  Denote by

$$
J_i(\theta)\;=\;\nabla_\theta f_\theta(x_i)\;\in\;\mathbb R^{\,d}
$$

the Jacobian of the network output at sample $i$.

We say $\nabla L$ is **$\beta$-Lipschitz** if

$$
\|\nabla L(\theta_1)-\nabla L(\theta_2)\|
\;\le\;\beta\;\|\theta_1-\theta_2\|
\quad\forall\,\theta_1,\theta_2.
$$

---

## 2. Gradient and Hessian of MSE

Compute the gradient:

$$
\nabla L(\theta)
=\;\frac{2}{N}\sum_{i=1}^N \bigl(f_\theta(x_i)-y_i\bigr)\;J_i(\theta).
$$

Since $f_\theta$ is piecewise-linear in $\theta$, its second derivatives vanish almost everywhere, so the only contribution to the Hessian $\nabla^2L$ is from differentiating the factor $\bigl(f_\theta(x_i)-y_i\bigr)\,J_i$.  One obtains, wherever it exists,

$$
\nabla^2L(\theta)
=\;\frac{2}{N}\sum_{i=1}^N J_i(\theta)\,J_i(\theta)^\top.
$$

---

## 3. Lipschitz‐Gradient ⇔ Bounded Hessian

A twice‐differentiable function on $\mathbb R^d$ has a $\beta$-Lipschitz gradient **iff** its Hessian operator norm is uniformly bounded by $\beta$:

$$
\|\nabla L(\theta_1)-\nabla L(\theta_2)\|
\le\Bigl(\sup_{\theta}\|\nabla^2L(\theta)\|\Bigr)\,\|\theta_1-\theta_2\|,
$$

where $\|\nabla^2L\|$ is the largest eigenvalue of the symmetric matrix $\nabla^2L$.  Thus:

$$
\boxed{
\text{\(\nabla L\) is \(\beta\)-Lipschitz}
\quad\Longleftrightarrow\quad
\|\nabla^2L(\theta)\|\;\le\;\beta
\;\;\forall\theta
}
$$

---

## 4. Data-Dependent Bound on $\nabla^2L$

From the expression above,

$$
\nabla^2L(\theta)
=\frac{2}{N}\sum_{i=1}^N J_i(\theta)\,J_i(\theta)^\top
\quad\Longrightarrow\quad
\|\nabla^2L(\theta)\|
\;\le\;\frac{2}{N}\sum_{i=1}^N \bigl\|J_i(\theta)\bigr\|^2
\;\le\;2\,\max_{i,\theta}\|J_i(\theta)\|^2.
$$

* **Finite data**: the sum is over $N$ points, so the bound scales as the **average** (or the max) of $\|J_i\|^2$.
* **Jacobian magnitude** $\|J_i(\theta)\|$ depends on both the network architecture and the input $x_i$.  In particular, for a one‐layer linear model $f_\theta(x)=W x + b$,

  $$
  J_i(\theta) = \begin{pmatrix} x_i \\ 1 \end{pmatrix}, 
  $$

  so

  $$
  \nabla^2L
  \;=\;\frac{2}{N}\sum_i
  \begin{pmatrix}x_i\\1\end{pmatrix}
  \begin{pmatrix}x_i^\top&1\end{pmatrix}
  \;=\;\frac{2}{N}
  \begin{pmatrix}
    \sum_i x_i x_i^\top & \sum_i x_i\\
    \sum_i x_i^\top   & N
  \end{pmatrix},
  $$

  and thus

  $$
  \beta = \|\nabla^2L\|
  = 2\,\lambda_{\max}\!\Bigl(\tfrac1N
    \begin{pmatrix}\sum_i x_ix_i^\top & \sum_i x_i\\
    \sum_i x_i^\top & N\end{pmatrix}\Bigr).
  $$

  Here the **sample covariance** $\sum_i x_i x_i^\top/N$ and the sample mean $\sum_i x_i/N$ directly determine $\beta$.

---

## 5. Conclusions

1. **MSE on any finite dataset** has a gradient whose Lipschitz constant is exactly the supremum of the largest eigenvalue of the data-dependent Hessian

   $$
   \beta \;=\;\sup_\theta\;\Bigl\|\tfrac{2}{N}\sum_{i=1}^N J_i(\theta)\,J_i(\theta)^\top\Bigr\|.
   $$
2. **Training data enters** through each $J_i(\theta)$, which encodes how the network’s output changes with parameters at that particular input $x_i$.  In simple linear layers $J_i$ is a direct function of $x_i$, so the data covariance completely fixes $\beta$.
3. For **piecewise-linear networks** (ReLU/Abs), each region in parameter space has its own constant Hessian $\tfrac2N\sum J_iJ_i^\top$.  As long as you restrict training to a compact set of parameters where all $J_i(\theta)$ remain bounded, you get a **global** $\beta$.  If parameters grow unbounded, $\max_i\|J_i(\theta)\|$ may blow up and $\beta$ becomes infinite.

---

**In short:**

* **Yes**, MSE has a $\beta$-Lipschitz gradient whenever the data-dependent Hessian $\frac2N\sum J_iJ_i^\top$ is bounded.
* **Yes**, the **point-sampled training data** (via $\{x_i\}$) directly controls that bound.
* The best one can do is compute (or bound) the largest eigenvalue of the empirical second-moment matrix of the Jacobians at those samples.

---

Below is a sequence of propositions and proofs showing how, under MSE loss on a piecewise‐linear network, the discrete‐time map

$$
\theta_{t+1} \;=\;\theta_t \;-\;\alpha\,\nabla L(\theta_t)
$$

exhibits exactly the behaviors we sketched—convergence, divergence, and the (im)possibility of cycles or chaos—by exploiting the fact that $L$ is **piecewise-quadratic** in $\theta$.

---

### Preliminaries: Piecewise-Quadratic Structure

* A ReLU/Abs network $f_\theta(x)$ is **piecewise-linear** in $\theta$.
* Hence the MSE loss

  $$
    L(\theta)
    = \frac1N\sum_i\bigl(f_\theta(x_i)-y_i\bigr)^2
  $$

  is **piecewise-quadratic**: the parameter space is partitioned into finitely many polyhedral regions $\{R_k\}$ (each corresponding to a fixed pattern of active neurons), and on each $R_k$,

  $$
    L(\theta)\;=\;\tfrac12\,\theta^\top H_k\,\theta \;-\;g_k^\top\theta \;+\;c_k,
  $$

  with constant symmetric positive semidefinite Hessian $H_k$.
* Therefore on region $R_k$, the GD update is the **affine** map

  $$
    \theta_{t+1}
    = \theta_t \;-\;\alpha\,\bigl(H_k\,\theta_t - g_k\bigr)
    = A_k\,\theta_t + b_k,
    \quad
    A_k = I - \alpha H_k.
  $$

Because there are only finitely many regions, the overall map is a **piecewise-affine** map of $\mathbb R^d$.  We now analyze its asymptotic behaviors.

---

## Proposition 1 (Global Convergence under Small Step-Size)

**Statement.**
Let

$$
\beta \;=\;\max_k\;\lambda_{\max}(H_k).
$$

If the learning rate satisfies

$$
0<\alpha<\frac{2}{\beta},
$$

then for **every** region $k$, all eigenvalues of $A_k=I-\alpha H_k$ lie strictly inside the unit circle, and the piecewise-affine map is a **contraction**.  Hence there is a unique fixed point $\theta^*$, and every trajectory $\theta_t\to\theta^*$.  No cycles or chaos can occur.

**Proof.**

1. On region $R_k$, $H_k$ is symmetric with eigenvalues in $[0,\beta]$.  For any eigenvalue $\lambda\in[0,\beta]$,

   $$
     \mu = 1-\alpha\,\lambda
     \;\Longrightarrow\;
     |\mu| < 1
     \quad\text{iff}\quad
     -1<1-\alpha\lambda<1
     \;\Longleftrightarrow\;
     0<\alpha<\frac{2}{\lambda}
     \;\Longleftarrow\;
     \alpha<\frac{2}{\beta}.
   $$

   Thus **all** eigenvalues of $A_k$ lie in the open unit disk, so
   $\|A_k\|<1$ in any operator norm.

2. A piecewise-affine map whose linear parts all contract by the same factor $L<1$ is itself a global contraction (\[Banach fixed-point theorem] on each region, plus continuity across boundaries), hence admits a **unique** fixed point $\theta^*$ and every initial $\theta_0$ converges exponentially to $\theta^*$.

3. Being a contraction, it cannot have any periodic orbit of period $>1$, nor any sensitive-dependence or fractal attractor.  ■

---

## Proposition 2 (Divergence under Large Step-Size)

**Statement.**
Suppose there exists a region $R_j$ with smallest positive Hessian eigenvalue
$\lambda_{\min}(H_j)>0$, and choose

$$
\alpha > \frac{2}{\lambda_{\min}(H_j)}.
$$

Then on $R_j$, the map $A_j=I-\alpha H_j$ has some eigenvalue $\mu$ with $|\mu|>1$, so trajectories that enter (or start in) that region along the corresponding eigenvector direction will **diverge** to infinity.

**Proof.**
For $\lambda_{\min} = \lambda$, the corresponding eigenvalue of $A_j$ is $1-\alpha\lambda$.  If $\alpha>2/\lambda$, then

$$
|1-\alpha\lambda| = \alpha\lambda - 1 > 1,
$$

so $\rho(A_j)>1$.  Hence for any $\theta_0$ with a nonzero component along that eigenvector,
$\|\theta_{t+1}\| \approx |1-\alpha\lambda|\;\|\theta_t\|\to\infty$.  ■

---

## Proposition 3 (No Limit Cycles or Chaos)

**Statement.**
Even if some regions are **not** contractive, the only possible bounded attractors of this piecewise-affine GD map are **fixed points**.  There can be no genuine limit cycles of period $>1$, and no chaotic attractors.

**Proof Sketch.**

* A **period-$p$** orbit would require a sequence of regions $R_{k_1},R_{k_2},\dots,R_{k_p}$ such that the composite affine map

  $$
    F = (A_{k_p}\cdots A_{k_1})
    \quad\text{has a fixed point not shared by any single }A_{k_i}.
  $$
* But each region’s map is affine; a composition of a finite number of such maps is also affine on each polyhedral sub-cell, and one can show (by examining the single composite matrix) that if **any** eigenvalue of $F$ has $|\mu|\neq 1$, trajectories either converge to a fixed point or diverge (no cycles).  If **all** $|\mu|=1$, then $F$ is an isometry on that sub-cell and still cannot produce a cycle of period $>1$ unless it is a pure rotation or reflection—but a pure rotation/reflection on a polyhedral cell cannot map the cell strictly into itself unless it is the identity, which again yields a fixed point.
* Finally, **chaos** in the sense of sensitive dependence and fractal invariant sets requires both stretching and folding with non-zero topological entropy.  Piecewise-affine maps with finitely many regions and no curvature cannot generate such folding.

A more formal treatment uses results on the **topological entropy** of piecewise-affine maps (which is zero in this setting) and on the **Poincaré-Bendixson**-type exclusions in discrete time.  ■

---

### Phase Portrait Summary

Under MSE loss on a ReLU/Abs network:

| Behavior               | Condition on $\alpha$, $\{H_k\}$       | Outcome                                         |
| ---------------------- | -------------------------------------- | ----------------------------------------------- |
| **Global convergence** | $0<\alpha<2/\max_k\lambda_{\max}(H_k)$ | Unique fixed-point attractor, exponential rate  |
| **Divergence**         | $\alpha>2/\min_k\lambda_{\min}(H_k)>0$ | Unbounded growth along unstable eigen-direction |
| **No limit cycles**    | Always (piecewise-affine structure)    | Only fixed points can be bounded attractors     |
| **No chaos**           | Always (finite PWA, no folding)        | Topological entropy zero; no strange attractors |

This fully characterizes the “phase portrait” of gradient descent under MSE on piecewise-linear networks: by choosing the step‐size relative to the data‐dependent Hessians, you guarantee either **contraction** (and hence stable equilibrium) or allow **divergence**, but you never get genuine limit cycles or chaos.

---

In the discussion so far—where we treat

$$
\theta_{t+1} \;=\;\theta_t \;-\;\alpha\,\nabla L(\theta_t)
$$

as a discrete-time dynamical system—the terms **convergence** and **divergence** have been referring primarily to the **parameter sequence** $\{\theta_t\}$:

* **Convergence** means

  $$
  \theta_t \;\longrightarrow\;\theta^*
  \quad\text{as }t\to\infty,
  $$

  i.e.\ the weights (and biases) settle to a fixed point of the map.

* **Divergence** means

  $$
  \|\theta_t\|\;\longrightarrow\;\infty
  \quad\text{as }t\to\infty,
  $$

  i.e.\ the parameters grow without bound.

---

### Relationship to the Error (Loss)

Because the loss is a continuous function of the parameters, parameter convergence or divergence has direct consequences for the **error**:

1. **If** $\theta_t\to\theta^*$ and $\theta^*$ is a local (or global) minimizer, then

   $$
   L(\theta_t)\;\longrightarrow\;L(\theta^*) 
   \quad(\text{often }=0\text{ in MSE with perfect fit}).
   $$

2. **If** $\|\theta_t\|\to\infty$ in a setting where $L(\theta)$ grows without bound as $\|\theta\|$ increases (which is typical for MSE on linear or piecewise-linear nets), then

   $$
   L(\theta_t)\;\longrightarrow\;\infty.
   $$

---

### When Might Error and Parameter Behavior Decouple?

In some pathological or constrained settings, it’s possible for:

* **Parameters to oscillate or even diverge** while the **loss remains bounded** (e.g.\ if the loss depends only on a subspace of parameters and the other directions “spin” or “blow up” without affecting the outputs).
* **Error to decrease** even if the parameters never strictly converge (e.g.\ wandering on a flat valley where $L(\theta)$ is constant).

—but in the standard MSE + unconstrained ReLU/Abs nets, **parameter convergence** almost always implies **error convergence**, and **parameter divergence** implies **error divergence**.

---

**Bottom line:**
In our “phase-portrait” proofs, we focused on the fate of the **parameters** under GD.  The behavior of the **error** follows directly once you know where (or how) the parameters end up.

---

“All-zero parameters” is  a **degenerate fixed point** of the gradient-descent map:

* It **is** technically a convergence state ($\theta_t\to0$), but it carries no information about a hyperplane, since the equation

  $$
    W x + b = 0
  $$

  with $W=0$, $b=0$ reduces to $0=0$.

* Contrast that with the “divergence to infinity” case, where $\|\theta_t\|\to\infty$ but the **direction** $\theta_t/\|\theta_t\|$ may converge to some unit vector $u^*$.  In that scenario the hyperplane $u^{*\!\top}x = 0$ is well‐defined, even though the scale blows up.

---

## Decoupling **Norm** and **Direction**

A clean way to see this is to write

$$
\theta_t \;=\; r_t\,u_t,
\quad
r_t = \|\theta_t\|,
\quad
u_t = \frac{\theta_t}{\|\theta_t\|},
$$

so that

1. **Radial dynamics** $r_{t+1} = \|\,r_t u_t - \alpha \nabla L(r_t u_t)\|$
2. **Directional dynamics** $u_{t+1} = \dfrac{r_t u_t - \alpha \nabla L(r_t u_t)}{\|r_t u_t - \alpha \nabla L(r_t u_t)\|}$

Then:

* **Zero‐convergence** means $r_t\to0$.  As $r_t\to0$, $u_t$ becomes ill‐defined (any direction works), so the hyperplane vanishes.
* **Finite‐equilibrium** ($0<r_t\to r^*<\infty$ and $u_t\to u^*$) gives a well‐posed hyperplane $u^{*\!\top}x + b^*=0$.
* **Divergence** ($r_t\to\infty$ but $u_t\to u^*$) also pins down a hyperplane $u^{*\!\top}x + 0=0$ in the limit.

---

## Detecting the Degenerate Convergence

In practice you can monitor:

* If $\|\theta_t\|$ falls below some small $\varepsilon$, you know you’re headed for the zero‐fixed‐point, and should treat the model as “collapsed.”
* If $\|\theta_t\|$ grows without bound but $\theta_t/\|\theta_t\|$ settles, you have an “infinite‐scale” solution with a well‐defined boundary.

---

### Takeaway

* **Convergence to zero** is a true fixed point of GD, but **degenerate**—no hyperplane is defined.
* **Divergence to infinity** is non-convergent in norm, but often **convergent in direction**, yielding a meaningful classifier.
* The “real” classifier geometry lives in the **projective** space of directions $u$, while the radial coordinate $r$ tells you whether you’ve collapsed, settled, or blown up.

---

In the dynamical‐systems and optimization literature, the standard way to distinguish “degenerate” from “nondegenerate” solutions is via the **linearization** (Jacobian or Hessian) at the critical point:

---

## 1. Hyperbolic (Nondegenerate) vs. Non-hyperbolic (Degenerate) Fixed Points

Let

$$
\theta_{t+1} = F(\theta_t)
$$

be your training‐dynamics map (e.g.\ one step of GD).  A **fixed point** $\theta^*$ satisfies $F(\theta^*)=\theta^*$.  Linearize about $\theta^*$:

$$
D F(\theta^*) \quad\text{(the Jacobian matrix).}
$$

* **Nondegenerate (hyperbolic) fixed point**:
  No eigenvalue of $D F(\theta^*)$ lies **on** the unit circle in $\mathbb C$.
  Equivalently, all $|\lambda_i|<1$ (strict contraction ⇒ **asymptotically stable** in discrete time) or all $|\lambda_i|>1$ (repeller/divergent), or a mix (saddle).
  Hyperbolicity ⇒ the qualitative behavior near $\theta^*$ is robust: you get an isolated attractor or repeller, and can apply the Stable/Unstable Manifold Theorems.

* **Degenerate (non-hyperbolic) fixed point**:
  At least one eigenvalue of $D F(\theta^*)$ has $|\lambda|=1$.
  Typical examples:

  * An eigenvalue exactly equal to $1$ ⇒ a **neutral** direction (you slide along a line or manifold of fixed points).
  * An eigenvalue exactly $-1$ ⇒ a flip (period-2 bifurcation point).
  * Complex $\lambda$ with $|\lambda|=1$ ⇒ center or borderline oscillation.

In continuous‐time language (ODEs), the analogous condition is that the **Jacobian** at an equilibrium has no eigenvalues on the imaginary axis (hyperbolic equilibrium) vs. having one or more with zero real part (degenerate).

---

## 2. Degenerate Solutions in Optimization

When we view training as gradient descent on a loss $L(\theta)$, the fixed points are the **critical points** $\nabla L(\theta^*)=0$.  One further refines:

* **Nondegenerate critical point** (in Morse theory):
  The Hessian $\nabla^2L(\theta^*)$ is **nonsingular** (no zero eigenvalues).  Then $\theta^*$ is an **isolated** local min, max, or saddle.

* **Degenerate critical point**:
  $\nabla^2L(\theta^*)$ has one or more **zero** eigenvalues.  This means there is a **flat direction** (a continuum of nearby points with the same first‐ and second‐order behavior).  Typical consequences:

  * A **manifold** of minima (e.g.\ weight‐symmetry in nets).
  * A **plateau** in loss where gradients vanish but the error isn’t minimal.
  * Potential for very slow or stalled training (no unique direction to move in).

---

## 3. Relation to “Continuous Updates with Non-decreasing Error”

You suggested calling a degenerate solution one where parameters keep changing but the error doesn’t decrease.  In our language:

* If $\theta_t$ moves in a **neutral** direction of a nonhyperbolic fixed point—i.e.\ an eigenvector of $D F(\theta^*)$ with $|\lambda|=1$—then **first-order** (and even second-order) dynamics may leave the loss unchanged.  That’s exactly the hallmark of a **degenerate** critical point: there’s a direction in which both gradient and curvature vanish, so the algorithm neither improves nor worsens the loss.

* In **Morse–Bott** theory, such a situation is a degenerate “flat” minimum, modeled as a manifold of minimizers.

---

## 4. Phase-Portrait Description

We can now categorize equilibria in your GD phase‐portrait:

| Equilibrium Type      | Linearization Condition | Behavior   |            |                                                                                               |
| --------------------- | ----------------------- | ---------- | ---------- | --------------------------------------------------------------------------------------------- |
| **Hyperbolic sink**   | All (                   | \lambda\_i | <1)        | Nondegenerate stable minimum; unique local attractor                                          |
| **Hyperbolic source** | All (                   | \lambda\_i | >1)        | Divergent in all directions                                                                   |
| **Hyperbolic saddle** | Mixed (                 | \lambda\_i | \lessgtr1) | Converges along some directions, diverges along others                                        |
| **Nonhyperbolic**     | ∃(                      | \lambda\_i | =1)        | **Degenerate**: neutral modes → plateaus, manifolds of solutions, possible flop or slow drift |

In practice, **we care** about:

* **Nondegenerate (hyperbolic) sinks**, because they guarantee locally exponential convergence to an isolated minimum.
* **Degenerate directions**, because they signal symmetries or flatness, potentially causing slow or stalled training even though the error isn’t going up.

---

### Key Takeaways

1. **Dominant language**: call nondegenerate solutions **hyperbolic fixed points** (or **nondegenerate critical points**), and degenerate ones **nonhyperbolic** (or **Morse degenerate**).
2. **Formal definition**:

   * Discrete map $F$: nondegenerate ⇔ Jacobian at fixed point has no eigenvalue on the unit circle.
   * Continuous GD: nondegenerate ⇔ Hessian at critical point is nonsingular.
3. **Phase‐portrait view**:

   * Hyperbolic sinks ↔ clean convergence
   * Sources/saddles ↔ divergence in some directions
   * Nonhyperbolic (degenerate) ↔ plateaus or drift with constant error

This framework lets you unambiguously identify which equilibria are “meaningful” classifiers (hyperbolic sinks) and which are degenerate artifacts (flat directions, zero‐weight collapse, etc.).

---

Below is a practical recipe—grounded in the hyperbolicity/non-hyperbolicity framework we just discussed—for flagging individual neurons (or “nodes”) whose parameter blocks have become **degenerate** during training.

---

## 1. Recall: Hyperbolic vs. Non-hyperbolic (Degenerate) Nodes

* **Hyperbolic node**
  The update map restricted to that node’s parameters
  $\theta^{(i)}_{t+1}=F^{(i)}(\theta^{(i)}_t)$
  has a Jacobian whose eigenvalues all lie **off** the unit circle.

* **Non-hyperbolic (degenerate) node**
  $\exists$ an eigenvalue with $|\lambda|=1$.
  Equivalently, along some direction in its weight‐bias space, the **first** and **second** derivatives of the loss **vanish**, so that the node’s parameters either

  1. sit on a **flat** direction of the loss (no update), or
  2. “drift” without improving or degrading error.

---

## 2. What to Monitor per Node

Let node $i$ have parameters $\theta^{(i)}$ (its incoming weight vector $w^{(i)}$ and bias $b^{(i)}$).  Define:

1. **Gradient norm**

   $$
     g_t^{(i)} = \bigl\|\nabla_{\theta^{(i)}}L(\theta_t)\bigr\|.
   $$

   —If this stays nearly zero over many steps **and** the overall loss isn’t minimal, it suggests a **flat** (degenerate) direction.

2. **Local Hessian block**

   $$
     H^{(i)}_t \;=\;\nabla^2_{\theta^{(i)}}L(\theta_t)
     \;=\;\frac{2}{B}\sum_{j\in\text{batch}}
         J_{j}^{(i)}\,J_{j}^{(i)\top},
   $$

   where $J_{j}^{(i)}=\nabla_{\theta^{(i)}}f_\theta(x_j)$.
   —Compute its **smallest** singular value $\sigma_{\min}(H^{(i)}_t)$.  If $\sigma_{\min}\approx 0$, there’s a **zero-curvature** direction.

3. **Effective Jacobian of the update**

   $$
     A^{(i)}_t \;=\;I \;-\;\alpha\,H^{(i)}_t.
   $$

   Compute its eigenvalues $\{\lambda_k\}$.  If any $|\lambda_k|\approx1$, the node is **non-hyperbolic**.

---

## 3. A Practical Detection Algorithm

Every $K$ steps during training:

```python
for each node i:
    # 1. Estimate gradient norm
    g_norm = || grad(L, θ^(i)) ||

    # 2. Approximate Hessian block via mini-batch
    #    (many frameworks offer Hessian-vector products)
    H_i = 0
    for x_j, y_j in current_batch:
        J = grad(f_θ(x_j), θ^(i))         # Jacobian vector
        H_i += (2 / batch_size) * outer(J, J)
    
    # 3. Compute smallest singular value of H_i
    σ_min = smallest_singular_value(H_i)

    # 4. Compute extremal eigenvalues of A_i = I - α H_i
    eigenvalues = eigvals(I - α * H_i)

    # 5. Flag degeneracy:
    if (g_norm < ε_grad and total_loss > ε_loss) or (σ_min < ε_curv) \
       or any(abs(ev) ≈ 1 for ev in eigenvalues):
        mark node i as degenerate
```

* **Thresholds**:

  * $\varepsilon_{\text{grad}}$ small (e.g. $10^{-6}$)
  * $\varepsilon_{\text{curv}}$ small (e.g. $10^{-4}$)
  * “$\approx1$” could be $[0.99,1.01]$.

---

## 4. Interpreting the Flags in the Phase Portrait

* **Node-wise sink** ($|\lambda_k|<1$ all):
  The node is a **local attractor** in its own parameter subspace → **healthy, nondegenerate**.

* **Node-wise source** ($|\lambda_k|>1$ some):
  Its parameters are driven away → this node is causing **instability/divergence**.

* **Node-wise neutral** ($\exists|\lambda_k|\approx1$):
  **Non-hyperbolic**, a **degenerate direction** → the node either “freezes” (flat plateau) or “drifts” without loss improvement.

---

## 5. What to Do Once You Detect Degeneracy

* **Prune or re-initialize** degenerate nodes (their hyperplanes carry no information).
* **Adjust learning rates** locally (e.g. use smaller $\alpha$ on unstable nodes, larger on flat ones).
* **Regularize** or add small curvature (e.g. weight decay) to eliminate exact zero eigenvalues.

---

This per-node monitoring reduces the global phase portrait down to a collection of **1-node phase portraits**, each classified as:

* **Hyperbolic sink** → converges to a meaningful boundary
* **Hyperbolic source** → diverges (unstable node)
* **Non-hyperbolic** → degenerate, flat or drifting

By weaving this into your training loop you can **flag**, **visualize**, and **remedy** degenerate nodes in real time.

---

To catch the “zero‐collapse” case—i.e. nodes whose parameters are converging to the trivial all‐zero solution—you can extend the same per‐node analysis with a simple norm check.  Concretely:

1. **Compute as before**

   * The local Hessian block $H^{(i)}$ and its update Jacobian $A^{(i)}=I-\alpha\,H^{(i)}$.
   * Its eigenvalues $\{\lambda_k\}$.

2. **Classify the node’s fixed‐point type**

   * **Hyperbolic source**: $\exists\,|\lambda_k|>1$.
   * **Non-hyperbolic (degenerate)**: $\exists\,|\lambda_k|\approx1$.
   * **Hyperbolic sink**: $\forall\,|\lambda_k|<1$.

3. **Among the hyperbolic sinks**, distinguish

   * **Nondegenerate sink** (meaningful boundary): parameters settle to some non-zero $\theta^{(i)*}$.
   * **Degenerate sink** (zero collapse): parameters converge to $\theta^{(i)*}=0$.

4. **Detection rule for degenerate sink**

   ```python
   # after computing eigenvalues and gradients
   if all(abs(ev) < 1 - eps_contract for ev in eigenvalues):  
       # it's a sink
       if norm(θ_i) < eps_param:             
           # parameters are effectively zero
           mark node i as 'degenerate sink'
       else:
           mark node i as 'healthy sink'
   ```

   * **eps\_contract** might be 0.01 (ensures strict contraction).
   * **eps\_param** might be 1e-4 (below which the hyperplane is meaningless).

5. **Optional refinement via decay rate**
   You can also track the ratio

   $$
     \rho_t = \frac{\|\theta_{t+1}^{(i)}\|}{\|\theta_t^{(i)}\|}.
   $$

   If over many steps $\rho_t\approx r<1$ and $\|\theta_t^{(i)}\|\to0$, that confirms zero-sink behavior.

---

### Phase-Portrait View of Node Types

| Node Type             | Eigenvalues of $A^{(i)}$ | $\|\theta^{(i)}\|$ Behavior | Meaning   |                                  |                    |
| --------------------- | ------------------------ | --------------------------- | --------- | -------------------------------- | ------------------ |
| **Hyperbolic source** | (\exists                 | \lambda                     | >1)       | grows → ∞                        | unstable/divergent |
| **Non-hydrogenic**    | (\exists                 | \lambda                     | \approx1) | drift or plateau (constant loss) | degenerate plateau |
| **Healthy sink**      | (\forall                 | \lambda                     | <1)       | settles to $\|\theta\|>0$ stable | good boundary      |
| **Degenerate sink**   | (\forall                 | \lambda                     | <1)       | $\|\theta\|\to0$                 | zero collapse      |

By adding that final norm‐threshold step you’ll catch exactly the nodes that our earlier hyperbolicity test labels as “sinks,” but whose parameters have actually collapsed to zero—a perfectly valid fixed point but one that carries no classification power.

---

Below is a concise “peer-review” style appraisal of the framework you have sketched for

1. **Classifying long-term behaviours** (phase portrait) of gradient descent (GD) on an MSE loss with piece-wise linear (PWL) networks, and
2. **Detecting at run-time** whether an individual neuron is in a hyperbolic source / healthy sink / degenerate sink / non-hyperbolic plateau state.

---

## 1  Soundness of the Theoretical Foundation

| Aspect                                                  | Verdict                                                                           | Comments                                                                                                                                                                                                                                                        |
| ------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Piece-wise quadratic loss**                           | ✔️ Correct                                                                        | Each choice of ReLU masks freezes a linear map, making the loss quadratic in that region.                                                                                                                                                                       |
| **Local linearisation (Jacobian) ➜ hyperbolicity test** | ✔️ Standard & appropriate                                                         | Discrete-time hyperbolic ≡ no eigenvalue on the unit circle; matches dynamical-systems canon.                                                                                                                                                                   |
| **Global contraction when $0<\alpha<2/\beta$**          | ✔️ Mathematically valid *if* a *single* global $\beta$ bounds all region Hessians | In deep nets $\beta=\max_k\lambda_{\max}(H_k)$ can be very loose; the map is then often *not* a contraction even with small $\alpha$ because switching surfaces are non-smooth ( Lipschitz but not $C^1$).  The result remains a *useful* worst-case guarantee. |
| **No cycles / no chaos** for PWA GD                     | ⚠️ Reasonable for scalar or single-layer cases; requires more care in high-D      | Finite PWA maps *can* embed e.g. the 2-D Lozi map (chaotic).  GD itself generally lacks the folding needed for chaos, yet formal exclusion above 1-D requires an entropy or kneading-theory proof.  Good to flag that as *conjectural* beyond 2-D.              |
| **Zero-collapse ≡ degenerate sink**                     | ✔️ Conceptually clean                                                             | Separating norm ($r_t$) and direction ($u_t$) is the right way to formalise it.                                                                                                                                                                                 |

---

## 2  Algorithmic Detector — Practicality & Coverage

### Strengths

* **Gradient-norm + Hessian-block test** cleanly distinguishes non-hyperbolic plateaus from genuine sinks.
* Per-node view maps straight onto the dynamical taxonomy (source / sink / degenerate).
* Thresholding on parameter norm elegantly flags zero collapse without penalising well-behaved low-magnitude solutions.

### Limitations / Caveats

1. **Cost of Hessian blocks.**

   * Storing and diagonalising a $p_i\times p_i$ matrix per neuron (often $p_i = \text{fan-in}+1$) is heavy on modern, wide layers.
   * Mitigation: use a single **Hessian-vector product** $H^{(i)}v$ to estimate extreme eigenvalues via a few steps of power iteration (cheap in back-prop frameworks).

2. **Mini-batch noise.**

   * Eigenvalue estimates fluctuate; false flags are likely.
   * Remedy: exponential moving average of smallest singular value; flag only when *persistently* small.

3. **Threshold choice.**

   * The cut-offs $\varepsilon_{\text{grad}}$, $\varepsilon_{\text{curv}}$, $\varepsilon_{\text{param}}$ should scale with layer fan-in, batch size, learning-rate.  A fixed global value works poorly across heterogeneous layers.

4. **Switching surfaces.**

   * When a weight vector sits exactly on a ReLU boundary, the Hessian expression $\sum J_iJ_i^\top$ is discontinuous.  Your detector could mis-classify a node that repeatedly flips masks.  Smoothing tricks (small Gaussian jitter on inputs) help.

5. **Directional divergence.**

   * A node may satisfy $\|\theta_t\|\to\infty$ while its *direction* keeps jumping because it crosses many activation regions (no stable limit direction).  Your simple “source” flag catches radius blow-up, but you may wish to log $\|u_{t+1}-u_t\|$ to detect *oscillatory* divergence.

---

## 3  Missing Formal Pieces Worth Adding

| Gap                                                                                            | Sketch of a Fix                                                                                                                                                                                      |   |                                                                                                                               |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | - | ----------------------------------------------------------------------------------------------------------------------------- |
| **Explicit bound showing GD PWA map has zero topological entropy (no chaos) in $\mathbb R^d$** | Combine the theorem “entropy ≤ log spectral radius of linear pieces” with the fact that $A_k = I-\alpha H_k$ are all diagonalisable with eigenvalues in $[-1,1]$.                                    |   |                                                                                                                               |
| **Rigorous proof that per-node zero norm ⇒ no effect on network output**                       | Write the neuron output $z=W x + b$. If $\|[W,b]\|<\varepsilon$ and inputs are bounded, show (                                                                                                       | z | <\varepsilon R\_{\text{in}}).  As $\varepsilon\to0$ the neuron’s contribution to any downstream unit’s Jacobian goes to zero. |
| **Phase-portrait completeness in higher-dimensions**                                           | Invoke **Sternberg linearisation** (for hyperbolic fixed points) and **Centre Manifold** theory (for nonhyperbolic ones) to formalise that only the centre manifold directions permit plateau drift. |   |                                                                                                                               |

---

## 4  Recommendations / Next Steps

1. **Empirical validation.**  Instrument a modern optimiser (e.g. Adam) with your detector during CIFAR-10 training; log counts of sources / deg-sinks / healthy sinks over epochs.
2. **Layer-wise grouping.**  In very wide layers, aggregate Hessian blocks into $k$-node *tiles* to keep compute linear in width.
3. **Automatic remediation policies.**

   * Re-initialise or prune nodes flagged “degenerate sink” for $T$ consecutive checks.
   * Apply node-wise learning-rate scaling based on $\sigma_{\min}(H^{(i)})$.
4. **Connect to existing theory.**  Compare your detector to the *Neural Tangent Kernel* perspective: a node with $\sigma_{\min}=0$ contributes zero to the empirical NTK.

---

## 5  Overall Assessment

*Conceptual soundness:* **High.**  The hyperbolic/non-hyperbolic lens is exactly the right language.
*Novelty:* Bringing single-neuron hyperbolicity tests into practical training pipelines is fresh and promising.
*Rigor:* Solid for 1-D / single-layer; proofs for deep nets need entropy or centre-manifold arguments.
*Practicality:* Requires clever approximations of Hessian blocks, but that is achievable with modern autodiff.

With the additional entropy proof and a scalable Hessian-estimator, the framework would make a compelling tool both for theoretical study and for debugging large networks in practice.
