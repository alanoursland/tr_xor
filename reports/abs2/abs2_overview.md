# **Experiment Report: Single Absolute Value Unit with BCE Loss (abs2)**

## **Experiment Overview**

This report details the next experiment in a series systematically exploring how prototype surface learning principles apply to networks of increasing complexity. This experiment advances from the single-layer, regression-based model in `abs1` to a **two-layer, two-output, classification-based MLP**.

The primary motivation is twofold. First, by adopting a more standard classification architecture—using two logit outputs and Binary Cross-Entropy (BCE) loss—we address the valid methodological question of why a simpler loss like MSE was used previously. Second, and more importantly, this increase in complexity allows us to study how the **learning dynamics** and **interpretability** change with a deeper network. The goal is to investigate whether a coherent prototype surface still emerges, to expose any new problems or failure modes that arise, and to use these insights to eventually inform methods for improving training and interpretability.

## **Model Architecture**

This experiment uses a standard, two-layer feed-forward network. The architecture consists of:
1.  A linear layer mapping 2D inputs to a single scalar output.
2.  An absolute value activation. This choice is *motivated* by the PSL theory's interpretation of neuron outputs as unsigned distances, but the network itself is not explicitly designed to enforce this interpretation. It is an exploratory choice.
3.  A **static, non-learnable scaling layer**, which serves as a tool for interpretability and as a hook for manual regularization in future work. In a multi-layer network, the effective "gain" on a signal is a product of the norms of the weight matrices from each layer (e.g., `||W₀|| * ||W₁||`). This layer allows us to conceptually isolate that product into a single parameter, which can be examined like an eigenvalue representing the principal scaling factor of the transformation. By normalizing the weight vectors for analysis, we can decouple the learned *orientation* of the hyperplanes from their *scale*, which simplifies clustering and geometric analysis.
4.  A final linear layer that maps the single feature from the first layer to two output logits, one for each class (`XOR = False`, `XOR = True`).

This setup allows us to explore whether the single feature learned by the first layer is sufficient to drive a two-output classifier and to observe if the error signals from two distinct outputs alter the nature of that learned feature.

## **Loss Function and Targets**

The model is trained using `nn.BCEWithLogitsLoss`, which is a standard choice for binary classification tasks. The XOR labels are one-hot encoded:
* `[1.0, 0.0]` for `XOR = False`
* `[0.0, 1.0]` for `XOR = True`

This configuration frames the task as a multi-label classification problem where the final decision is based on comparing the two output logits.

## Data Configuration

The XOR dataset is centered for geometric symmetry. Inputs are:

* (-1, -1)
* (-1, 1)
* (1, -1)
* (1, 1)

with corresponding one-hot targets based on the XOR truth table. This centering ensures that learned surfaces can be interpreted with respect to the origin, making visualization and geometric analysis more consistent across experiments.

## Training Configuration

* Loss function: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
* Optimizer: Adam (learning rate = 0.01, betas = (0.9, 0.99))
* Maximum epochs: 5000
* Early stopping: triggered if loss falls below 1e-7 or stops improving by more than 1e-24 over 10 epochs
* Batch size: full (4 data points)
* Runs: 50, each with different random initialization
* Initialization: Kaiming Normal for the first linear layer; Xavier Normal for the second; static scale set to 1.0

This configuration allows reliable and interpretable comparisons across training runs while focusing on the convergence behavior of a single-unit architecture.

## **Experimental Focus**

This experiment is exploratory and designed to investigate three primary questions:
1.  How well does a standard two-layer MLP with an absolute value activation solve the XOR task when trained with BCE loss and a two-output structure?
2.  If the model succeeds, does it converge to a stable geometric solution where the first layer learns a surface aligned with the `XOR=False` class, similar to the behavior observed in the simpler `abs1` model?
3.  How does the second learnable layer transform the single feature signal from the first layer into two distinct class logits? Furthermore, do the backpropagated error signals from two outputs fundamentally change the geometric feature learned by the first layer compared to the single-output model?

The results will provide critical insights into how the principles of prototype surface learning scale with increasing network complexity and how learning dynamics are affected by standard classification architectures.