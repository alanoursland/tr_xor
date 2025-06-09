### 🌀 Angular Momentum Updates for Neural Networks

**Angular momentum updates** are a conceptual extension of standard gradient-based optimization that treat neural network weight vectors not just as points in Euclidean space, but as **directional entities**. Rather than focusing solely on minimizing loss via direct gradient descent, angular momentum methods explicitly track and update the **direction of weight vectors**, separating **angular motion** (changes in direction) from **radial scaling** (changes in magnitude).

---

#### 🧠 Key Idea

Standard optimizers like SGD or Adam adjust weights in full vector space, which couples both direction and magnitude. Angular momentum updates decouple these components:

* **Angular update**: rotate the weight vector toward the descent direction, tracking “velocity” over the unit hypersphere (e.g., via projected gradients or exponential moving averages on the sphere).
* **Radial update**: adjust the norm (scaling) of the weight vector separately, optionally holding it fixed to isolate directional learning.

---

#### 🧭 Motivation

Many learning tasks—especially those involving geometric interpretability (e.g., prototype surfaces, decision boundaries)—rely more on the **orientation** of weights than their scale. In such contexts:

* Large angular corrections correlate with longer training times.
* Conventional optimizers may waste steps adjusting norm when only a rotation is needed.
* Angular momentum preserves directional inertia, encouraging smoother, faster alignment.

---

#### 🧪 Use Cases

* Interpretable models where decision boundaries are defined by vector orientation.
* Geometrically motivated tasks (e.g., prototype learning, contrastive learning).
* Low-dimensional setups where weight rotation dominates learning dynamics (e.g., XOR).

---

Angular momentum updates offer a promising path toward **geometry-aware learning**, particularly in models where **directional alignment** is more important than scale.
