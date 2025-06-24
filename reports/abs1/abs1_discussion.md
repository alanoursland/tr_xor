# **Discussion: Single Absolute Value Unit for XOR Classification**

## 1. Overview

This experiment demonstrated that a minimal neural network—a single linear neuron with an absolute value activation—consistently learns to classify the XOR dataset. All runs converged to a geometrically optimal and stable solution. This outcome provides strong support for the central hypothesis of this work: that even a single-node model can learn a class-defining prototype surface and distinguish categories by measuring deviation from it. These runs show a single |·| neuron consistently learns one prototype surface anchored on the False points and uses distance from that surface to classify.

The results support the central hypothesis of this work: that a single-node model can represent a class-defining prototype surface and distinguish categories by measuring deviation from it. More broadly, the experiment serves as a concrete test case for the Prototype Surface Learning (PSL) theory, which interprets neural network activations as geometric distance functions rather than magnitude indicating feature confidence.

This discussion highlights key geometric patterns observed in successful solutions, notes ambiguous aspects of interpretation in single-unit networks, outlines early observations about convergence rates, and raises theoretical and empirical questions for future investigation. The findings are not definitive; rather, they are the first step in grounding a geometric theory of learning in empirical behavior—even in the simplest nonlinear setting.

## 2. The Universal Geometry of the Learned Solution

Across all 250 runs, every model converged to one of two specific, symmetric, and optimal geometric configurations. This consistency provides a stable foundation for analyzing the varied paths taken to reach this final state.

### 2.1. The Learned Surface Intersects the `False` Class

The learned surface, defined by the hyperplane $Wx + b = 0$, consistently passed directly through the two inputs labeled `False` (class 0). This is not an incidental outcome. Because the target value for the `False` class is 0 and the model's output is $y = |Wx + b|$, the only way for the model to perfectly match the target and minimize the loss is to learn parameters that ensure $Wx + b = 0$ for those inputs. The optimization objective itself functionally compels the model to anchor its prototype surface on the 0-labeled class.


### 2.2. Output Reflects Scaled Distance to the Surface

The final geometry for all runs was not only correctly oriented but also precisely scaled. The mean absolute distance of `True` (class 1) inputs to the learned surface was consistently \~1.41421, placing the two `True` points at exactly √2 units away from the learned prototype surface (mean 1.41421 ± 1e-7). This implies that the norm of the final weight vector, $\|W\|$, was also optimized to a specific value—approximately $1/\sqrt{2}$—so that the model’s output magnitude for `True` inputs approaches 1. Thus, the network does not simply orient a surface: it learns both its optimal *placement* and *scaling*.

The affine form of the surface equation:

$$
y = Wx + b
$$

can be decomposed geometrically as a projection operator:

$$
y = \frac{v \cdot x}{\sqrt{\lambda}} + b
$$

where $v$ is a unit vector defining the direction of projection, and $\lambda$ is a positive scaling factor (analogous to an eigenvalue). Choosing any point $\mu$ **on the learned decision boundary**—that is, any $\mu$ satisfying $W\mu + b = 0$—we can rewrite:

$$
b = -\frac{v \cdot \mu}{\sqrt{\lambda}}
\quad \Rightarrow \quad
y = \frac{v \cdot (x - \mu)}{\sqrt{\lambda}}
$$

This formulation shows that the model computes a signed, scaled projection of the input relative to a surface centered at $\mu$, with orientation $v$, and scaling $1/\sqrt{\lambda}$. The choice of $\mu$ introduces translation invariance along the surface: any point on the decision boundary defines the same geometric decomposition.

This interpretation reinforces the prototype surface hypothesis. The model is not merely learning a separating hyperplane; it is constructing a **directional deviation measure** from a class-defining region. The neuron behaves not as a binary detector, but as a calibrated distance function from the prototype surface—a surface that, by training objective, must intersect the class-0 examples and symmetrically distance itself from class-1 inputs.

## 3. Learning Dynamics and Convergence Speed

To understand why different initializations take markedly different amounts of time to reach the same optimal solution, we explored factors affecting convergence. The data strongly suggests that the required rescaling of the weights is a dominant factor, while the re-orientation of the surface is secondary.

A plausible explanation is that when the initial weight norm is far from the optimal norm, the optimizer spends the majority of its epochs correcting this scale discrepancy. For instance, the `Large` initialization produced the slowest convergence by a wide margin (median 644 epochs) and required the most extreme weight shrinking, with an initial-to-final norm ratio ranging from 1.85 to 17.79. Conversely, the fastest initializations, `Tiny` and `Normal` (medians of 147 and 138 epochs, respectively), started with smaller weights that needed to be scaled up. This suggests that growing weights toward an optimum is a more efficient optimization path.

Even when minimal re-orientation was needed, performance was still dictated by scale. In the `Large` initialization runs, trials requiring the smallest angle changes were still extremely slow, averaging 628.8 epochs to converge. This reinforces the conclusion that correcting the initial weight scale, rather than its orientation, was the primary bottleneck in the training process.

The initial orientation and magnitudes both affect convergence speed and need to be studied jointly. We can't draw any strong conclusions from the data. Although, we suspect that the L2-distance between the intial and final states may correlate more strongly with the training epochs. This aspect of our experiment invites more investigation.

## 4. Interpretation and Representational Duality

The consistency of the final learned geometry, arrived at from widely different starting points and along varied time courses, strengthens a key theoretical takeaway. The model's single neuron does not act as a simple feature detector. Instead, it learns to be a **distance function**, anchored to the `False` class prototype. The near-zero activations for the `False` class signal membership ("on the surface"), while high-activation values for the `True` class signal deviation or distance from that prototype.

This duality—where low activation signals class membership—is a core tenet of the PSL framework. The significant effort the model expends correcting its internal scale, with some runs taking over 1,500 epochs, to perfectly realize this specific geometric relationship underscores its centrality to the learning objective. Future work might investigate if these scaling dynamics hold in higher-dimensional problems and how they interact with more complex architectures and optimizers.

## 5. **Interpretation Ambiguity in Single-Unit Outputs**

The observed behavior of the single-node model exposes a subtle but important ambiguity in how neural network outputs are interpreted—particularly in minimal architectures. Although the network reliably outputs higher values for `True`-labeled XOR inputs and near-zero values for `False` inputs, it does so by learning a surface that **intersects** the `False` class. From a traditional standpoint—where higher activation is often taken to indicate stronger feature presence—this could suggest the unit is recognizing the `True` class. However, under the prototype surface learning (PSL) perspective, the opposite is true: the model is **anchored to the `False` class**, and its output grows with **distance** from the prototype surface that represents it.

This tension between activation magnitude and geometric alignment reveals a representational duality. In this setup, the neuron is not a gate detecting presence or absence of a class-specific feature—it is a **distance function**, anchored to one class and used to measure how far inputs deviate from it. The low-activation (near-zero) values signal membership in the prototype class (`False`), while high-activation values correspond to inputs outside that region (`True`).

This duality is further complicated by the binary nature of the task. Because only one surface is learned, and the model must distinguish two classes, the distinction between recognition and rejection becomes entangled. The same output serves both as a distance measure and a basis for binary decision-making. In multiclass settings, where multiple surfaces could each represent their own prototype region, this ambiguity would likely diminish. But in the single-unit XOR case, it highlights how minimal models can conflate **geometric structure** with **semantic interpretation**, underscoring the need for care when assigning meaning to scalar outputs in prototype-based architectures.

## 6. **General Properties of the Model**

### 6.1. Capacity to Intersect Any Two Points

In two-dimensional space, any pair of distinct points defines a unique line. A single linear neuron, therefore, has sufficient capacity to place its decision surface through any two of the four XOR inputs. The model architecture is not constrained to intersect only the `False` points; in principle, it could align its surface with any class pairing.

Empirically, however, the solution consistently aligns with the `False`-labeled inputs, positioning the learned surface to pass through those two examples while leaving the `True` inputs on opposite sides. This raises an important question: is this configuration uniquely stable under gradient descent, or is it simply the most accessible minimum given the loss landscape, initialization symmetry, or label encoding?

One plausible explanation is that the model treats the `False` class—mapped to an output of zero—as the prototype set. Since the loss function penalizes deviations from target values, driving the output toward zero for `False` inputs naturally pulls the learned surface toward those points. This creates a structural bias toward anchoring the surface on the zero-valued class. Whether alternative alignments (e.g., passing through one `True` and one `False` point) are even locally stable, or merely suboptimal and rarely discovered, remains an open question for further geometric and optimization-based analysis.

### 6.2. Analytical Optimality of the Learned Surface

While the network has the capacity to align its surface with any two points in the XOR dataset, the consistent convergence to a surface intersecting the `False` inputs is not merely an empirical artifact—it is analytically optimal under the model and loss function.

We derived the total squared loss symbolically for the model $y = |w_1 x_1 + w_2 x_2 + b|$ on the centered XOR dataset. The result is:

$$
L(w_1, w_2, b) = 4b^2 + 4w_1^2 + 4w_2^2 - 2|b - w_1 + w_2| - 2|b + w_1 - w_2| + 2
$$

This expression arises from summing squared errors over all four data points, with target outputs of 0 for the `False` class and 1 for the `True` class.

The only way for the model to output exactly zero is for the linear term $Wx + b$ to equal zero—i.e., for the input to lie **on the surface** defined by the affine equation. Therefore, the optimization objective naturally favors placing this surface through the class-0 inputs. Since those two points lie on a shared diagonal, and the class-1 inputs lie symmetrically across from it, this configuration also minimizes error for the `True` class by placing them at equal and maximal distance from the surface.

Thus, the surface intersecting the `False` inputs is not just a convenient or likely solution—it is the **globally optimal configuration** for this architecture and task. This result reinforces the geometric consistency observed across successful runs and provides a formal basis for the behavior predicted by Prototype Surface Learning theory.



