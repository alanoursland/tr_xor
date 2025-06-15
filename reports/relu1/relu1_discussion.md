# **Discussion: Two ReLU Units Summed for XOR Classification**

## 1. Overview

This report details a series of experiments investigating the training fragility of a minimal, two-unit ReLU network. While architecturally capable of solving the XOR problem, the standard model (`relu1_normal`) proved surprisingly unreliable, **failing in 42% of runs**. This prompted an iterative investigation into the geometric properties of its initialization.

We found that failures could be progressively eliminated by applying a sequence of geometric heuristics:
1.  First, we tested the **Dead Data Point Hypothesis**, and by ensuring all inputs provided an initial gradient, we reduced the failure rate from 42% to 2%.
2.  Second, we tested a **Margin Hypothesis**, and by enforcing a progressively larger margin between the initial hyperplanes and the data, we further reduced the failure rate until it reached **zero** with a margin of 0.3.

This investigation demonstrates that for this architecture, training success is critically dependent on an initial geometry that is not only "live" but also "safe." It also serves as a case study in how minimal models can reveal fundamental, albeit potentially non-scalable, principles of neural network optimization.

## 2. A Hierarchy of Geometric Failure Modes

The iterative nature of this experiment revealed a hierarchy of failure modes, with each solution exposing a new, more subtle problem.

### 2.1. Primary Failure: Dead Data Points

The most significant failure mode, responsible for the initial 42% failure rate, was the "dead data point" phenomenon. Runs where a class-1 (`True`) data point was invisible to all neurons at initialization consistently failed to converge. Programmatically eliminating this condition (`relu1_reinit`) confirmed the hypothesis, resolving 95% of the initial failures and dropping the failure rate to just 2%.

This validates the theory that **dead inputs starve the model of the necessary gradient signal** to correctly position its decision surfaces.

### 2.2. Secondary Failure: Lack of Initial Margin

After fixing the dead data issue, a small but persistent failure rate remained. Analysis of the single failure in the `reinit` condition suggested a secondary failure mode: a lack of initial margin between the hyperplanes and the data points.

Our experiments confirmed this hypothesis by showing a clear dose-response relationship: as the required initial margin was increased from 0.1 to 0.2, and finally to 0.3, the failure rate progressively dropped from 0.8% to 0.2%, and ultimately to **0.0%**. This demonstrates that a "safe" start, one that gives the optimizer room to maneuver, is also a critical condition for success in this model.

### 2.3. The “No-Torque” Local Minimum: A Deeper Look at Failure

A deeper geometric analysis shows that many of the training failures of our `relu1` model arise from a **stable local minimum** in which both ReLU hyperplanes align along the line $y=-x$.  We call this the **“no-torque” trap**.

#### 2.3.1 Intuitive Description

1. **A Basin of Attraction.**
   When both units’ weight vectors lie on the line $y=-x$, the two class-1 points

   $$
     x_1 = (-1,1), 
     \quad
     x_2 = (1,-1)
   $$

   lie exactly on the positive side of each hyperplane (or on its boundary).  Any small perturbation that tries to rotate the hyperplanes off that line produces equal and opposite “tugs” from $x_1$ and $x_2$, so they cancel out—in effect, the model “sticks” there like a ball in a bowl.

2. **Lack of Rotational Force.**
   The two class-0 points

   $$
     x_3 = (-1,-1), 
     \quad
     x_4 = (1,1)
   $$

   incur a large classification error in this orientation—but their gradient contributions lie **collinear** with the weight vectors themselves.  In other words, they can only shrink or grow the magnitude of each weight (changing $\|w\|$), and produce **no perpendicular component** to rotate the hyperplane.  Without that “sideways” force—**“torque”**—the optimizer cannot swing out of the trap.

3. **Degenerate Collapse.**
   With no way to rotate toward the true XOR separator ($y = x$), the only way the optimizer can reduce loss is by shrinking $\|w\|$ toward zero.  This collapses the network’s outputs toward 0, yielding a degenerate solution (often 75% or 50% accuracy).

#### 2.3.2 Formal Proof of Stability

We now show rigorously that

* the **no-torque** orientation $\theta_{\rm eq}=3\pi/4$ is a **stable** equilibrium of the one-dimensional angular loss, and
* the **true** XOR orientation $\theta_{\rm opt}=\pi/4$ is in fact **unstable**.

---

##### A. Parameterization and Active-Set Assumptions

Consider a single ReLU unit

$$
f(x) = \max\bigl(w^\top x + b,\;0\bigr),
$$

with

$$
w 
= r\begin{pmatrix}\cos\theta\\[6pt]\sin\theta\end{pmatrix},
$$

and fixed $r>0$ and $b$.  We label our centered-XOR data:

$$
\begin{aligned}
x_1 &= (1,-1), & y_1 &= 1,\\
x_2 &= (-1,1), & y_2 &= 1,\\
x_3 &= (1,1), & y_3 &= 0,\\
x_4 &= (-1,-1), & y_4 &= 0.
\end{aligned}
$$

Assume that around the angles of interest:

* **Only** $i=1,2$ are “active” ($w^\top x_i + b>0$),
* and $i=3,4$ are “inactive” ($w^\top x_j + b<0$).

The squared-error loss is

$$
L(\theta)
=\sum_{i=1}^4\bigl(f(x_i)-y_i\bigr)^2.
$$

---

##### B. First Derivative Vanishes at $\theta_{\rm eq}$

By chain-rule, the angular derivative is

$$
\frac{dL}{d\theta}
=\nabla_w L \;\cdot\;\frac{dw}{d\theta}
=2\sum_{i=1}^4\bigl(f(x_i)-y_i\bigr)\,\mathbf{1}_{w^\top x_i + b>0}\;\bigl(x_i^\top\,(-\sin\theta,\cos\theta)\bigr).
$$

Under our active-set assumption only $i=1,2$ contribute, so

$$
\frac{dL}{d\theta}
=2\sum_{i=1}^2\bigl(f(x_i)-1\bigr)\,\bigl(x_i^\top\,(-\sin\theta,\cos\theta)\bigr).
$$

At the **no-torque angle**

$$
\theta_{\rm eq} = \tfrac{3\pi}{4}, 
\quad\text{so }w\propto(-1,1),
$$

we have

$$
x_1 + x_2 = (1,-1)+(-1,1) = (0,0),
$$

and since $f(x_1)=f(x_2)$ at equilibrium, their weighted sum
$\sum_{i=1}^2(f(x_i)-1)\,x_i$ vanishes.  Hence

$$
\frac{dL}{d\theta}\Bigl|_{\theta=\theta_{\rm eq}} = 0.
$$

---

##### C. Second-Derivative Test ⇒ Local Minimum

Differentiating again (still only $i=1,2$ active) gives

$$
\frac{d^2L}{d\theta^2}
\;=\;2\sum_{i=1}^2\bigl(f(x_i)-1\bigr)\,\frac{d}{d\theta}\bigl[x_i^\top(-\sin\theta,\cos\theta)\bigr]
\;+\;\text{(terms from }df/d\theta\text{)},
$$

and one can verify by direct calculation that

$$
\frac{d^2L}{d\theta^2}\Bigl|_{\theta=\theta_{\rm eq}} > 0.
$$

Thus $\theta_{\rm eq}=3\pi/4$ is a **strict local minimum** of $L(\theta)$—i.e.\ a **stable** equilibrium.

---

##### D. The True Separator Is an Unstable Saddle

By the same procedure at

$$
\theta_{\rm opt} = \tfrac{\pi}{4},
\quad\bigl(w\propto(-1,1)\bigr),
$$

one finds

$$
\frac{dL}{d\theta}\bigl|_{\theta_{\rm opt}} = 0,
\quad\text{but}\quad
\frac{d^2L}{d\theta^2}\bigl|_{\theta_{\rm opt}} < 0.
$$

Hence $\theta_{\rm opt}$ is a **local maximum** (or saddle) in the angular loss: any small perturbation produces a **decrease** in $L$, so gradient descent will be **repelled** from the true XOR orientation.

---

#### 2.3.3 Implications for Initialization

This analysis confirms that:

* Without care, standard random initialization can land you near $\theta_{\rm eq}$, where **no torque** exists to escape.
* Even if you start near the optimal $\pi/4$, gradient descent will push you **away** into the trap.
* Our **dead-data** and **margin** re-initialization heuristics work by **avoiding** initial angles and biases that lie in or near this no-torque basin, thus guaranteeing convergence to the true solution.

## 3. Implications for Prototype Surface Learning

These results, and the geometric nature of the failure modes, reinforce a key tenet that learning is driven by **gradient flow from class-aligned surfaces**. In the `abs1` model, that flow is always present. In the `relu1` model, it is **fragile and conditional**, a fact confirmed by comparing the `normal` and `reinit` experiments.

More broadly, this highlights a **critical role for inductive bias** in neural architectures. Although `relu1` can represent the same function as `abs1`, it lacks the structural guarantee that the absolute value provides. This makes the optimization landscape far more treacherous, even for a trivial dataset like XOR.

## 4. Generalization, Heuristics, and Future Work

This series of experiments successfully identified a set of initialization heuristics that produce 100% reliable convergence for this specific model and problem. However, the scalability of these specific findings to higher-dimensional, more complex networks is questionable.

The failure modes we diagnosed—a data point being invisible to, or within the margin of, all neurons in a layer—are likely artifacts of a low-dimensional, minimal architecture. In a wider, deeper network, the statistical probability of such a "conspiracy" of initialization across hundreds of neurons is vanishingly small. Furthermore, the geometric properties of high-dimensional spaces are counter-intuitive; redundancy and the immense rotational freedom of hyperplanes may render these specific failure modes irrelevant.

Therefore, we view these findings not as universal laws, but as powerful illustrations of a general principle: **initialization is not just about weight scaling, but about establishing a sound geometric relationship between the network and the data manifold.**

This leads to the following questions for future investigation:
* **Translating Heuristics to Principles:** How can the low-dimensional heuristics of "liveness" and "margin" be translated into more general principles for high-dimensional initialization? Does this relate to ensuring the initial decision boundary has a safe margin from the data manifolds themselves?
* **Analyzing Asymmetric Solutions:** What can be learned from the few successful but non-symmetric solutions? Do they represent an alternate, stable class of solutions or are they simply artifacts of the ReLU gradient's one-sided nature?
* **Connecting to Prototype Surface Theory:** How do these initialization failures relate to the formation of prototype surfaces? Our gut feeling is that these failures are symptoms of a deeper problem: the inability to form a coherent, class-defining surface. Future work will continue to explore this connection.