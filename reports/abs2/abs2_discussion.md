# **Discussion: Single Absolute Value Unit with BCE Loss (abs2) [DRAFT]**

## 1\. Overview

This experiment successfully demonstrated that a two-layer MLP, using a single absolute value unit as its primary feature extractor, can solve the XOR task within a standard classification framework. All 50 runs achieved 100% classification accuracy.

The key finding from this initial analysis is that moving to a standard classification framework (BCE loss with two logit outputs) fundamentally alters the learning dynamics and the final learned geometry compared to the simpler regression-based `abs1` model. While functionally successful, the model converges much more slowly to its low-loss target and discovers a more varied set of geometric solutions, prompting the need for more nuanced analysis.

## 2\. A Tale of Two Convergences: Accuracy vs. Loss

The most striking dynamic observed in this experiment is the vast difference between the time to achieve functional correctness and the time to fully minimize the loss function.

Informal observation showed that the model learns to classify all four XOR points correctly—achieving 100% accuracy—very rapidly, typically within 50 to 200 epochs. In stark contrast, reaching the stringent loss threshold of `< 1.0e-7` required over 3,000 epochs on average.

This suggests a two-phase learning process:

1.  **Rapid Geometric Learning:** The network first learns the correct general geometry required to separate the classes. The single feature learned by the absolute value unit becomes effective for classification very early in training.
2.  **Slow Confidence Refinement:** The long tail of the optimization process is spent slowly adjusting the weights of both layers to fine-tune the output logits. The BCE loss pushes the model not just to be correct, but to be *confident* in its correctness (i.e., producing logits that result in probabilities very close to 0.0 and 1.0). This final, high-confidence state is much harder and slower to achieve than the initial geometric separation.

## 3\. The Geometry of the Learned Solution: A New Duality

The most significant finding of this experiment is the fundamental change in the geometry of the learned prototype surface. In the `abs1` model, the MSE loss function, with its target of 0 for the `False` class, effectively forced the learned hyperplane to intersect the `False` class points. This experiment reveals that the BCE loss framework provides no such constraint.

As a result, the model is free to discover one of at least two distinct, equally valid geometric solutions:

  * **Mode 1: `False` Class as Prototype:** In some runs, the learned hyperplane is anchored on the `XOR=False` class points (circles), mirroring the solution from `abs1`.

  * **Mode 2: `True` Class as Prototype:** In other runs, the hyperplane is anchored on the `XOR=True` class points (triangles).

This discovery is critical. It demonstrates that the specific placement of the prototype surface is highly dependent on the loss function. The BCE objective only requires that the final logits are correct; it does not care which class is used as the geometric anchor. The optimizer is free to converge to whichever mode provides an effective gradient path. This finding complicates the interpretation of the learned surface but also deepens our understanding of how geometric representations are shaped by training objectives.

## 4\. Analysis of Learning Dynamics and Model Parameters

The new complexities of this architecture are also reflected in the learning dynamics and final weight distributions.

### The Irrelevance of Initial Geometry

Unlike in `abs1`, where the initial scale of the weights was a strong predictor of convergence time, the learning dynamics in this experiment show no clear correlation between the initial weight geometry (angle or norm ratio) and the time required to reach the final low-loss state. The slow convergence appears to be an intrinsic property of the optimization landscape for this architecture, not a consequence of a poor starting position.

### Interpreting the "Messy" Clusters

The preliminary clustering analysis, which found 11 clusters for the `linear1` weights, is almost certainly an artifact of unconstrained weight magnitudes. It is hypothesized that different runs converge to solutions with similar hyperplane *orientations* but different weight vector *lengths*. These differences in scale confound the clustering algorithm. It is expected that normalizing the weight vectors before clustering would reveal the four expected underlying solution orientations (two diagonal lines, each with two possible perpendicular weight vectors).

### The Consistent Logic of the Final Layer

Despite the variety of solutions found in the first layer, the second linear layer (`linear2`) learns a simple and consistent logic. It reliably functions as a linear classifier operating on the scalar distance feature from the first layer. It consistently learns one of two symmetric weight configurations (approximating `[+C, -C]` or `[-C, +C]`) to correctly map the distance feature to the two opposing class logits.

## 5\. Implications and Future Work

This experiment successfully demonstrates that the core principles of our prototype surface theory can generalize from a simple regression model to a more standard, multi-layer classification architecture. However, this generalization is not seamless and introduces significant new complexities. The choice of loss function and the addition of a second learnable layer have a profound impact on the learned geometry, the stability of the solution, and the optimization dynamics.

These findings open up several avenues for future investigation:

  * **Refining Analysis:** A more sophisticated analysis is required to properly characterize the multi-modal geometric solutions. This includes re-clustering based on normalized weights and classifying runs based on which class serves as the prototype anchor.
  * **Controlling the Geometry:** Can we develop regularization techniques or architectural constraints to guide the network towards a single, more stable geometric solution? The static scaling layer included in this architecture serves as a potential hook for such future work.
  * **Scaling Complexity:** How do these dynamics evolve in even deeper or wider networks? Understanding how these simple geometric principles scale is critical to applying them to more complex, real-world problems.

  