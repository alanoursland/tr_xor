# **Prototype Surface Learning Theory**

## **1. Conceptual Context**

In conventional interpretations of neural networks, individual neurons are often viewed as feature detectors, with larger activation values indicating stronger membership in or presence of a learned feature. Our theory reframes how neurons detect features: rather than higher activations indicating stronger feature presence, each neuron evaluates proximity to a learned prototype surface, with feature membership determined by deviation from this hyperplane. The hyperplane decision boundary is defined by intersection of a set of data prototypes. The structure of the neuron’s computation — particularly the linear transformation followed by a nonlinear activation — naturally supports this interpretation through geometric properties. Importantly, this theoretical framework applies to backpropagation training and standard feedforward neural network architectures that utilize linear transformations followed by ReLU activations, requiring no modifications to existing network designs or training procedures.

---

## **2. Mathematical Foundation**

### **2.1 Signed Distance from Linear Transformation**

At the core of this theory is the observation that the pre-activation output of a neuron, computed as:

$$
z = Wx + b
$$

can be interpreted as a **signed, scaled distance** from the input vector $x$ to a hyperplane in the input space defined by the weight vector $W$ and bias $b$. The hyperplane equation $Wx + b = 0$ partitions the space into two half-spaces:

* The region where $Wx + b \leq 0$
* The region where $Wx + b > 0$

The sign of $z$ indicates on which side of the hyperplane the input lies, and its magnitude (when scaled by $\|W\|_2$) gives the distance to the surface.

### **2.2 Absolute Distance and Geometric Interpretation**

The absolute geometric distance from an input point $x$ to the hyperplane is given by:

$$
d(x, \mathcal{H}) = \frac{|Wx + b|}{\|W\|_2}
$$

This formulation captures proximity to the learned prototype surface without regard for direction. It reveals that a linear layer alone encodes rich geometric information about input location relative to learned surfaces.

### **2.3 Decomposing Absolute Value Using ReLU**

An important identity relating absolute value to ReLU is:

$$
|z| = \operatorname{ReLU}(z) + \operatorname{ReLU}(-z)
$$

This decomposition shows that absolute distance can be recovered by summing two one-sided activations. Each ReLU term computes distance from only one side of the surface.

### **2.4 ReLU as One-Sided Prototype Surface Recognition**

The ReLU activation function:

$$
\operatorname{ReLU}(z) = \max(0, z)
$$

outputs positive values only when the input lies in the half-space where $Wx + b > 0$. For inputs in the region $Wx + b \leq 0$, the output is zero.

Thus, ReLU can be viewed as a one-sided distance evaluator: it activates only when the input lies **outside** a defined region, and it remains zero when the input lies **within** it. The region where ReLU outputs zero — i.e., the negative half-space — is interpreted here as the **prototype region**: a learned set of feature patterns to which the neuron is “tuned.”

### **2.5 Absolute-Value as Two-Sided Prototype Locus**

While ReLU marks the entire negative half-space $\{x : Wx + b \le 0\}$ as the prototype region (via zero activation), the absolute-value activation $g(z) = |z|$ reinterprets prototype membership in a more sharply localized way. Instead of extending a region, it selects a **prototype locus**—a *measure-zero* set precisely on the affine hyperplane where the pre-activation vanishes.

Let $f(x) = |Wx + b|$. Then $f(x) = 0$ if and only if $x \in H = \{x : Wx + b = 0\}$. In contrast to ReLU, there is no extension to one side: the activation is strictly positive off the hyperplane, and symmetric in both directions. The zero set is no longer a half-space but the **surface itself**.

We can interpret this as a shift in the prototype encoding principle. With absolute value, a unit declares a prototype not by inclusion, but by **coincidence**: the prototype is the exact affine slice where the activation vanishes. Furthermore, the **output of the unit now encodes distance** from this prototype locus. In fact, since

$$
|Wx + b| = \text{ReLU}(Wx + b) + \text{ReLU}(-Wx - b),
$$

the absolute-value unit can be understood as simultaneously measuring distance on both sides of the surface, using two opposing ReLU-like detectors.

This behavior makes absolute-value networks function less like inclusion-based classifiers and more like **surface detectors**: each hidden neuron describes an affine prototype, and outputs the deviation from it. The network as a whole then aggregates these deviations to guide classification. As we will see later in the XOR case (Section §4.1), this property allows a single $|\cdot|$ unit to carve out linearly inseparable regions through symmetric combination—something ReLU requires multiple units to achieve.

In short, absolute-value activations preserve the prototype surface idea but collapse the prototype region to a locus. They transition the model’s geometry from region-growing to **distance field encoding**, setting up a different but still compatible basis for prototype surface learning.

---

## **3. Prototype Recognition Mechanism**

### **3.1 Linear Output as Prototype Set Membership**

Before applying any activation function, the output of a neuron is given by the affine transformation:

$$
y = Wx + b
$$

This equation defines an **N-dimensional hyperplane** in input space. The set of points that satisfy the equation $Wx + b = 0$ forms this hyperplane, which can be interpreted as the **boundary of a prototype region**. It divides the input space into two half-spaces:

* Points where $Wx + b = 0$: lie **exactly on the prototype surface**
* Points where $Wx + b < 0$: lie on the **prototype-recognized side**
* Points where $Wx + b > 0$: lie on the **non-prototype side**

Importantly, this surface can be constructed to intersect multiple distinct prototype points. In $\mathbb{R}^N$, a single hyperplane can intersect up to $N$ **affinely independent** points. (If we're counting *distinct* prototypes in general position that the surface passes through, it's $N$; if we exclude the bias term and consider only directions, it may be $N-1$.)

Thus, a neuron can encode **membership in a prototype set** by shaping a hyperplane to intersect multiple examples and generalize to all points lying within the same geometric relation — specifically, those that are **collinear** or linearly dependent with the intersected points. These collinear extensions represent the **implicit generalization** power of the linear layer: it treats an entire subspace of similar inputs as sharing a prototype identity.

### **3.2 ReLU Expands the Recognized Set**

When a ReLU activation is applied to the output:

$$
\text{ReLU}(Wx + b)
$$

the neuron becomes sensitive only to inputs on one side of the hyperplane — specifically, the region where $Wx + b > 0$, producing a positive value. However, the more interesting behavior is what happens when the output is **zero**:

* For all inputs where $Wx + b \leq 0$, the neuron outputs zero.
* This includes not only the points exactly on the surface $Wx + b = 0$, but also the **entire negative half-space**.

This zero-output region defines a **superset** of the original prototype set. It includes:

* The intersected prototype points on the hyperplane
* All collinear points aligned with those prototypes
* All additional points lying **in the negative half-space**, which are geometrically similar in the sense of lying on the “prototype side” of the learned boundary

ReLU therefore acts as a **set extender**: it generalizes the prototype membership from the exact surface (defined by the linear equation) to **an entire half-space**. From a functional perspective, the neuron treats all these points — infinitely many of them — as **equivalent** by assigning them the same output (zero).

This reinterpretation turns a conventional assumption on its head: **zero activation is not a sign of inactivity, but a sign of inclusion**. It indicates that the input lies within a recognized region — defined not as a point but as a **spatially extended prototype set**. In this view, a zero activation can be interpreted as an 'acceptance' of the input into the prototype set, while a positive activation signals 'rejection' or measures the input's deviation from this recognized region.

---

### **3.3 Absolute-Value Encodes Unsigned Distance**

Whereas ReLU extends a single prototype surface into an entire half-space of inclusion, the absolute-value activation performs no such expansion. Instead, it treats the prototype as the **zero set** $\{x : Wx + b = 0\}$, and outputs a value proportional to the **distance** from that set. The geometry is not one of region growth, but of distance field encoding.

Given a neuron with parameters $W \in \mathbb{R}^{1 \times d}$, $b \in \mathbb{R}$, the pre-activation value $z = Wx + b$ represents a signed measure of displacement from the surface $H = \{x : Wx + b = 0\}$. Applying the absolute-value activation yields

$$
|z| = |Wx + b|,
$$

which corresponds to the **unsigned (algebraic) distance** from $x$ to the surface $H$, up to a normalizing factor.

More precisely, the perpendicular Euclidean distance from a point $x$ to the surface $H$ is:

$$
\text{dist}(x, H) = \frac{|Wx + b|}{\|W\|}.
$$

Thus, a neuron with $g(z) = |z|$ implicitly defines a prototype surface and computes the (scaled) distance from it.

This shift has two implications:

1. **Distance over inclusion.** Unlike ReLU, which gives a binary membership signal (zero if in the prototype region, positive otherwise), the absolute-value unit furnishes a continuous score—useful not for gating, but for evaluating deviation. Classification becomes a matter of **comparing distances** to multiple prototype surfaces, possibly using downstream learned weights to shape their relative influence.

2. **Symmetry of response.** Absolute-value units respond identically to displacements on either side of the prototype surface. This enables downstream layers to form symmetric or anti-symmetric combinations, particularly important in problems like XOR where parity matters.

When networks use absolute-value activations, hidden units act more like **distance sensors** than region classifiers. This aligns closely with traditional metric learning, but from a surface-based, rather than centroid-based, perspective.

From a prototype surface learning standpoint, the use of $|\cdot|$ enforces a strict localization: **the prototype is the surface itself**, and all outputs describe how far the input has drifted from it. This behavior complements the ReLU-based regime, offering a dual view—**proximity measured from both sides, rather than inclusion from one.**

---

## **4. Network-Level Behavior and Feature Surfaces**

Neural networks operate by composing many such prototype surface evaluators across layers. Each layer takes the output of previous layers — transformed distance measures — and combines them through further linear and nonlinear operations. The result is a hierarchical construction of complex, nonlinear surfaces in feature space.

* **Early layers** recognize basic prototype regions — broad geometric patterns in the input
* **Deeper layers** integrate and refine these into more selective, abstract prototype sets
* The final layers classify inputs based on **composite proximity** to these learned prototype surfaces

Rather than making decisions by crossing fixed decision boundaries, the network evaluates the **degree of deviation from multiple prototype sets**, allowing rich and flexible classification based on geometric structure in the data. It's noteworthy that this understanding of network behavior as composing prototype surface evaluators applies directly to common existing architectures employing linear layers and ReLU activations, without necessitating changes to their design or training methods.
