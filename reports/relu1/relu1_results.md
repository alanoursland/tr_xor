# **Results: Two ReLU Units Summed for XOR Classification**

## 1. Overview

This document summarizes empirical results from the `relu1_normal` experiment, which investigates a minimal ReLU-based architecture applied to the centered XOR problem. The model consists of two linear units followed by ReLU activations whose outputs are summed to form the prediction. Training was conducted using a fixed initialization scheme over 50 independent runs.

Unlike the `abs1` model—which converged successfully in every case—this experiment **failed to reach 100% accuracy in 21 out of 50 runs**, despite using the same data and optimizer. This result demonstrates that the ReLU decomposition of the absolute value is not guaranteed to preserve the same optimization behavior, even under ideal conditions.

In addition to overall accuracy and convergence metrics, we also examined **initial dead activations** and **emergent mirror symmetry** between units. These offered new insights into the conditions that lead to convergence or failure.

## 2. Classification Accuracy

| Accuracy (%) | Run Count |
| ------------ | --------- |
| 100%         | 29        |
| 75%          | 21        |

No runs resulted in partial or zero accuracy beyond these two discrete outcomes, suggesting a binary success/failure dynamic in the learned solutions.

## 3. Final Loss Distribution

The models that failed to achieve perfect accuracy also retained noticeably higher loss values at termination. Mean final loss across all runs was **1.05e-01**, with a maximum loss of **2.52e-01**.

## 4. Convergence Timing (Epochs to Loss < 1e-7)

Although many runs did converge to the configured loss threshold, the overall convergence distribution was slower than in the absolute value model:

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 33     |
| 50th       | 123    |
| 100th      | 275    |

Convergence time alone was not a reliable predictor of final classification success.

## 5. Prototype Geometry and Hyperplane Clustering

The final learned hyperplanes exhibited broader variability and weaker class-0 alignment compared to `abs1`. Cluster analysis revealed five distinct solution modes, with many noisy outliers—further suggesting that training often failed to converge on a coherent decision surface.

## 6. Dead Data Point Analysis

We introduced a metric to identify **dead data points**—inputs that produce zero activation across all ReLU units at initialization. These points have no gradient influence early in training and may reduce the model’s ability to learn appropriate decision boundaries.

Findings:

* 13 runs with **no dead inputs** reached 100% accuracy  
* 16 runs with **dead inputs** reached 100% accuracy  
  * 16 of these had class-0 dead inputs  
  * 0 of these had class-1 dead inputs  

* 21 runs with **dead inputs** reached 75% accuracy  
  * 13 of these had class-0 dead inputs  
  * 20 of these had class-1 dead inputs  

Notably, **no model with dead inputs from class 1 reached 100% accuracy**, suggesting these inputs are crucial for correct classification. This supports the hypothesis that **initial representational coverage** of the dataset strongly influences final outcomes.

## 7. Mirror Symmetry Formation

The model was architecturally inspired by the identity $|x| = \text{ReLU}(x) + \text{ReLU}(-x)$, but no symmetry was enforced. We measured how often and how precisely the two learned components became mirror reflections of one another.

Findings:

* **Mirror pairs detected**: 29 / 50 runs  
* **Perfect mirror symmetry** (cosine similarity ≈ –1.0): 15 runs  
* **Mean mirror similarity**: –0.99335 ± 0.00970  
* **Mean mirror error** ($|\cos + 1|$): 0.00665

Symmetry was common but not guaranteed, and perfect alignment was only achieved in about half the symmetric cases. This reflects the **difficulty of learning symmetry from scratch** via gradient descent.
