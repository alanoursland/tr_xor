# **Discussion: Single Absolute Value Unit for XOR Classification**

## 1. **Overview**

This experiment demonstrated that a minimal neural network—consisting of a single linear neuron followed by an absolute value activation—can successfully learn to classify the XOR dataset. Across multiple weight initialization strategies, the model frequently converged to a solution that achieves perfect training accuracy. These outcomes were not only quantitatively robust but also geometrically consistent: the learned surface aligned with the two class-0 inputs, while class-1 inputs were positioned symmetrically on either side.

The results support the central hypothesis of this work: that a single-node model can represent a class-defining prototype surface and distinguish categories by measuring deviation from it. More broadly, the experiment serves as a concrete test case for the Prototype Surface Learning (PSL) theory, which interprets neural network activations as geometric distance functions rather than feature intensities.

This discussion highlights key geometric patterns observed in successful solutions, notes ambiguous aspects of interpretation in single-unit networks, outlines early observations about failure modes, and raises theoretical and empirical questions for future investigation. The findings are not definitive; rather, they are the first step in grounding a geometric theory of learning in empirical behavior—even in the simplest nonlinear setting.

---

## 2. **Observed Geometry of Successful Models**

### 2.1. One Class Lies on the Learned Surface

Across all successful training runs, the learned surface—defined by the zero set of the pre-activation \$Wx + b\$—passed directly through the two inputs labeled `False` in the XOR dataset: \$(-1, -1)\$ and \$(1, 1)\$. These points consistently fell on or near the surface, yielding near-zero output after the absolute value activation. The two `True` inputs, in contrast, were positioned symmetrically on either side of this hyperplane.

This alignment was stable across runs and initialization methods, modulo a sign flip in the weight vector \$W\$, which does not affect the model’s output due to the symmetry of the absolute value function. These sign-inverted solutions represent the same surface geometry and produce identical outputs. The consistency of this geometric configuration supports the PSL framework's prediction: classification emerges through the placement of a prototype surface that includes one class and measures the deviation of the other.

Yes—you did give a clear explanation earlier, and it hasn’t been explicitly written out yet in any section. Here's what you said, paraphrased:

> The surface intersects the `False` inputs because those are the points with label 0, and the model is trained to output 0 for those inputs. Since \$y = |Wx + b|\$, the only way to output 0 is for the linear transformation \$Wx + b\$ to equal 0—i.e., the input lies **on the surface**. So minimizing loss naturally pulls the surface toward the 0-labeled class.

This outcome is not incidental. Because the target value for the `False` class is 0, and the model's output is \$y = |Wx + b|\$, the only way to exactly match the target is to place the input **on the surface**—i.e., to satisfy \$Wx + b = 0\$. The model thus learns to anchor the surface on the 0-labeled inputs as a direct consequence of minimizing squared error. In this architecture, class 0 is not just geometrically central; it is **functionally preferred** by the optimization objective.

### 2.2. Output Reflects Scaled Distance

The model’s output, \$y = |Wx + b|\$, reflects a signed, scaled distance from the learned surface. Empirically, the weight norm \$|W|\$ in successful runs converged to approximately \$1/\sqrt{2}\$, implying that the surface was scaled such that the class-1 inputs—positioned at a Euclidean distance of \$\sqrt{2}\$ from the surface—produced outputs near 1. This suggests the model not only learned a surface aligned with class-0 but also scaled its output to match a meaningful distance measure.

Formally, the output behavior is consistent with the structure of a 1D Mahalanobis distance function:

$$
y = \left|\lambda^{-1/2}(v^\top x + b)\right|,\quad \text{with } v = \frac{W}{\|W\|}
$$

where \$\lambda\$ is a scalar variance term and \$v\$ is the unit vector normal to the surface. This expression corresponds exactly to a Mahalanobis distance computation along a single principal component, as described in Oursland (2024). Here, the surface passes through some \$\mu\$ such that \$Wx + b = 0\$, and the network’s output becomes:

$$
y = \left|\lambda^{-1/2}(v^\top (x - \mu))\right|
$$

Thus, while the model was not explicitly trained to perform a Mahalanobis computation, it effectively learned a solution that mirrors it—centered on the class-0 inputs and measuring scaled distance to them. This supports the theoretical claim that absolute-value activations can induce surface-based, statistically grounded representations, even in minimal settings.

---

## 3. **Interpretation Ambiguity in Single-Unit Outputs**

The observed behavior of the single-node model exposes a subtle but important ambiguity in how neural network outputs are interpreted—particularly in minimal architectures. Although the network reliably outputs higher values for `True`-labeled XOR inputs and near-zero values for `False` inputs, it does so by learning a surface that **intersects** the `False` class. From a traditional standpoint—where higher activation is often taken to indicate stronger feature presence—this could suggest the unit is recognizing the `True` class. However, under the prototype surface learning (PSL) perspective, the opposite is true: the model is **anchored to the `False` class**, and its output grows with **distance** from the prototype surface that represents it.

This tension between activation magnitude and geometric alignment reveals a representational duality. In this setup, the neuron is not a gate detecting presence or absence of a class-specific feature—it is a **distance function**, anchored to one class and used to measure how far inputs deviate from it. The low-activation (near-zero) values signal membership in the prototype class (`False`), while high-activation values correspond to inputs outside that region (`True`).

This duality is further complicated by the binary nature of the task. Because only one surface is learned, and the model must distinguish two classes, the distinction between recognition and rejection becomes entangled. The same output serves both as a distance measure and a basis for binary decision-making. In multiclass settings, where multiple surfaces could each represent their own prototype region, this ambiguity would likely diminish. But in the single-unit XOR case, it highlights how minimal models can conflate **geometric structure** with **semantic interpretation**, underscoring the need for care when assigning meaning to scalar outputs in prototype-based architectures.

---

## 4. **General Properties of the Model**

### 4.1. Capacity to Intersect Any Two Points

In two-dimensional space, any pair of distinct points defines a unique line. A single linear neuron, therefore, has sufficient capacity to place its decision surface through any two of the four XOR inputs. The model architecture is not constrained to intersect only the `False` points; in principle, it could align its surface with any class pairing.

Empirically, however, the solution consistently aligns with the `False`-labeled inputs, positioning the learned surface to pass through those two examples while leaving the `True` inputs on opposite sides. This raises an important question: is this configuration uniquely stable under gradient descent, or is it simply the most accessible minimum given the loss landscape, initialization symmetry, or label encoding?

One plausible explanation is that the model treats the `False` class—mapped to an output of zero—as the prototype set. Since the loss function penalizes deviations from target values, driving the output toward zero for `False` inputs naturally pulls the learned surface toward those points. This creates a structural bias toward anchoring the surface on the zero-valued class. Whether alternative alignments (e.g., passing through one `True` and one `False` point) are even locally stable, or merely suboptimal and rarely discovered, remains an open question for further geometric and optimization-based analysis.

### 4.2. Analytical Optimality of the Learned Surface

While the network has the capacity to align its surface with any two points in the XOR dataset, the consistent convergence to a surface intersecting the `False` inputs is not merely an empirical artifact—it is analytically optimal under the model and loss function.

We derived the total squared loss symbolically for the model $y = |w_1 x_1 + w_2 x_2 + b|$ on the centered XOR dataset. The result is:

$$
L(w_1, w_2, b) = 4b^2 + 4w_1^2 + 4w_2^2 - 2|b - w_1 + w_2| - 2|b + w_1 - w_2| + 2
$$

This expression arises from summing squared errors over all four data points, with target outputs of 0 for the `False` class and 1 for the `True` class.

The only way for the model to output exactly zero is for the linear term $Wx + b$ to equal zero—i.e., for the input to lie **on the surface** defined by the affine equation. Therefore, the optimization objective naturally favors placing this surface through the class-0 inputs. Since those two points lie on a shared diagonal, and the class-1 inputs lie symmetrically across from it, this configuration also minimizes error for the `True` class by placing them at equal and maximal distance from the surface.

Thus, the surface intersecting the `False` inputs is not just a convenient or likely solution—it is the **globally optimal configuration** for this architecture and task. This result reinforces the geometric consistency observed across successful runs and provides a formal basis for the behavior predicted by Prototype Surface Learning theory.

---

## 5. **Open Questions**

* What structure governs the failure cases—are they true local minima, poor convergence paths, or sensitivity to initialization?
* How do starting weights relate to success or failure?
* Can we identify attractor basins in parameter space?
* What can we learn by clustering successful and failed final states?

