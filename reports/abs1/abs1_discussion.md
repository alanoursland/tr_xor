# **Discussion: Single Absolute Value Unit for XOR Classification**

## 1. **Overview**

Briefly restate the nature of the demonstration:

* A single linear unit with absolute value activation was trained to solve XOR.
* Results are reported across multiple initialization strategies.
* This discussion summarizes consistent observations and raises open questions for further study.

---

## 2. **Observed Geometry of Successful Models**

### 2.1. One Class Lies on the Learned Surface

* In all successful runs, the learned hyperplane intersects the two inputs labeled `False` in the XOR dataset.
* The two `True` points lie symmetrically on either side of this hyperplane.
* The solution is geometrically stable across runs—up to sign flips (negation of the weight vector), which produce the same surface.

### 2.2. Output Reflects Scaled Distance

* The model computes $y = |Wx + b|$, producing a signed, scaled distance from the learned surface.
* In successful cases, the norm $\|W\|$ consistently converges to approximately $1/\sqrt{2}$.
* This matches the L2 distance of the `True` inputs from the decision boundary: $\sqrt{2}$.
* The model thus appears to learn a scale such that $|Wx + b| \approx \text{distance}$, where:

  $$
  y = |\lambda^{-1/2}(v \cdot x + b)|,\quad v = \frac{W}{\|W\|}
  $$

  This can be rewritten as a Mahalanobis-type expression:

  $$
  y = |\lambda^{-1/2}(v - \mu)|,\quad \mu \in \{x : Wx + b = 0\}
  $$

  where $\mu$ is any point on the learned hyperplane.

---

## 3. **Interpretation Ambiguity in Single-Unit Outputs**

* Although the unit outputs high values for `True` inputs and low values for `False` inputs, it learns a surface that intersects the `False` points.
* This raises an observation: under the traditional interpretation of neurons as detectors of high-activation "features," the unit appears to “recognize” the `True` class—yet it learns to align with the `False` class geometrically.
* This duality—surface alignment with one class, activation peak for the other—highlights representational ambiguity in single-node models.
* In this architecture, the output itself does not explicitly represent class membership. It represents a magnitude related to surface distance.

---

## 4. **Failure Behavior**

* Runs that did not achieve perfect accuracy often settled on 50% accuracy.
* These failures may reflect early saturation, optimization issues, or failure to align the surface with the correct class points.
* Without dynamic convergence tracking, it is unclear whether these are true local minima or merely under-trained.
* The current experiment design used fixed-epoch training. Future experiments will train to convergence and record dynamic stopping conditions.

---

## 5. **General Properties of the Model**

### 5.1. Capacity to Intersect Any Two Points

* In 2D, a single affine hyperplane can intersect any pair of points in the dataset.
* The solution observed here always intersects the two `False` points.
* Whether this is the only geometrically stable XOR-aligned configuration—or simply the easiest to find—remains an open question.

---

## 6. **Open Questions**

* Why does the learned surface consistently intersect the `False` inputs?
* Are there other valid surfaces that the model could converge to but doesn't?
* What structure governs the failure cases—are they true local minima, poor convergence paths, or sensitivity to initialization?
* How do starting weights relate to success or failure?
* Can we identify attractor basins in parameter space?
* What can we learn by clustering successful and failed final states?

---

Let me know when you're ready to start drafting this into full prose. Or if you'd like a diagrammatic summary of the geometry and the class alignment behavior.
