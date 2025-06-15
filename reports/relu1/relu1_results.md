# **Results: Two ReLU Units Summed for XOR Classification**

## 1. Overview

This document summarizes the empirical results from two related experiments (`relu1_normal` and `relu1_reinit`) investigating a minimal ReLU-based architecture for the XOR problem. The key difference between the experiments is the initialization strategy. The `relu1_normal` condition uses a standard initialization, which frequently results in "dead data points" that provide no initial gradient. The `relu1_reinit` condition programmatically re-initializes the model until no data points are dead.

The results show a dramatic improvement in the `reinit` condition, with the failure rate dropping from **42% to 2%**. This confirms that dead data points are the primary, but not sole, cause of training failure. Analysis of the single remaining failure case in the `reinit` condition points to a more subtle geometric failure mode. While most successful runs learned the required mirror-symmetric structure, the final hyperplanes were often slightly asymmetric, suggesting limitations imposed by the one-sided ReLU gradient.
## 2. Classification Accuracy

The re-initialization strategy to eliminate dead data points was highly effective, resolving the vast majority of training failures.

| Initialization Strategy | 100% Accuracy (Runs) | 75% Accuracy (Runs) | Failure Rate |
| :--- | :--- | :--- | :--- |
| **Standard Init (`relu1_normal`)** | 29 / 50 | 21 / 50 | 42% |
| **Re-init on Dead (`relu1_reinit`)** | 49 / 50 | 1 / 50 | 2% |

## 3. Dead Data Point Analysis

This metric was central to testing the hypothesis that a lack of initial gradient flow causes training failure.

### Standard Init (`relu1_normal`)

Under standard initialization, dead data points were common and strongly correlated with failure.

* 13 runs with **no dead inputs** reached 100% accuracy.
* 16 runs with **dead inputs** reached 100% accuracy.
    * All 16 of these had only class-0 inputs dead.
* 21 runs with **dead inputs** reached 75% accuracy.
    * 20 of these had class-1 inputs dead.

Notably, **no model with dead inputs from the `True` class (class 1) reached 100% accuracy**.

### Re-init on Dead (`relu1_reinit`)

This condition was designed to eliminate dead data points from the start.

* By design, 0 runs had dead inputs at initialization.
* Despite this, **1 run with no dead inputs still failed** to reach 100% accuracy, converging to 75% instead.

## 4. Mirror Symmetry Formation

Emergent mirror symmetry was a strong indicator of success across both conditions.

| Initialization Strategy | Mirror Pairs Detected |
| :--- | :--- |
| **Standard Init (`relu1_normal`)** | **29 / 50 runs** (perfectly matching the 29 successful runs) |
| **Re-init on Dead (`relu1_reinit`)** | **47 / 50 runs** |

In the `reinit` condition, the three runs that did not form a mirror pair corresponded to the single failed run and two successful runs whose angles were too different to consider mirror pairs, although the structure of those solutions still matched the mirrored solutions. 

## 5. Single Failure Case Analysis (`relu1_reinit`)

The single run in the `reinit` condition that failed provides insight into a more subtle failure mode. Visual analysis of this run's initial and final states reveals a specific geometric pattern.

* **Initial State**: At initialization, the two ReLU hyperplanes were positioned close to each other and nearly perpendicular to the ideal separating boundaries. As shown in the initial hyperplane plot (`reinit_failure_plots/hyperplanes_initial.png`), one hyperplane started very close to the class-1 data points.

* **Final State**: During training, one hyperplane appears to have quickly adjusted to zero out both class-0 data points (`(-1, -1)` and `(1, 1)`). In doing so, both class-1 points (`(-1, 1)` and `(1, -1)`) were pushed into the negative (deactivated) region of both ReLU units, halting their gradient flow and preventing the model from correcting their classification. The final hyperplane plot (`reinit_failure_plots/hyperplanes_final.png`) illustrates this outcome.

This suggests a new hypothesis for this failure: if an initial hyperplane is too close to a data point, it can be pushed across that point during early optimization, effectively deactivating it before the model has a chance to learn a globally optimal solution. A potential mitigation could be a "margin-based" re-initialization, which would ensure all data points are a minimum distance from the initial hyperplanes.

## 6. Final Loss & Convergence (Standard Init)

*The following metrics refer to the `relu1_normal` experiment runs.*

The models that failed to achieve perfect accuracy also retained noticeably higher loss values at termination. Mean final loss across all runs was **1.05e-01**, with a maximum loss of **2.52e-01**. The overall convergence distribution was slower than in the `abs1` model.

| Percentile | Epochs |
| :--- | :--- |
| 0th | 33 |
| 50th | 123 |
| 100th | 275 |

## 7. Prototype Geometry and Hyperplane Clustering (Standard Init)

*The following metrics refer to the `relu1_normal` experiment runs.*

The final learned hyperplanes exhibited broader variability and weaker class-0 alignment compared to `abs1`. Cluster analysis revealed five distinct solution modes, with many noisy outliers—further suggesting that training often failed to converge on a coherent decision surface.