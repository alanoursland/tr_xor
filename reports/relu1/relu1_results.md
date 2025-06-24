# **Results: Two ReLU Units Summed for XOR Classification**

## 1. Overview of Experimental Sequence & Accuracy

This document summarizes empirical results from a sequence of experiments on the `relu1` architecture. Each experiment used a progressively more constrained initialization strategy to test hypotheses about training failures. The following table shows the final classification accuracy for each condition. The margin thresholds (0.1, 0.2, 0.3) were chosen arbitrarily to observe the trend.

### Classification Accuracy Across All Conditions

| Initialization Strategy | Total Runs | Success Rate | Failure Rate |
| :--- | :--- | :--- | :--- |
| **1. Standard Init (`relu1_normal`)** | 50 | 58.0% (29 runs) | 42.0% (21 runs) |
| **2. Re-init on Dead (`relu1_reinit`)** | 50 | 98.0% (49 runs) | 2.0% (1 run) |
| **3. Re-init + 0.1 Margin** | 500 | 99.2% (496 runs) | 0.8% (4 runs) |
| **4. Re-init + 0.2 Margin** | 500 | 99.8% (499 runs) | 0.2% (1 run) |
| **5. Re-init + 0.3 Margin** | 500 | **100.0%** (500 runs) | **0.0%** (0 runs) |
| **6. In-Training Monitoring (`relu1_monitor`)**| 1000 | **100.0%** (1000 runs) | **0.0%** (0 runs) |

## 2. Results of Margin-Based Initialization

Following the analysis of the `Re-init on Dead` condition, a series of experiments were run to test the Margin Hypothesis by enforcing a progressively larger initialization margin ($\epsilon$).

* **With $\epsilon = 0.1$**: The failure rate was **0.8%** (4 failures in 500 runs).
* **With $\epsilon = 0.2$**: The failure rate was further reduced to **0.2%** (1 failure in 500 runs).
* **With $\epsilon = 0.3$**: All failure modes were suppressed, resulting in a **100% success rate** over 500 runs.

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

* **Final State**: During training, one hyperplane appears to have quickly adjusted to zero out both class-0 data points (`(-1, -1)` and `(1, 1)`). In doing so, both class-1 points (`(-1, 1)` and `(-1, 1)`) were pushed into the negative (deactivated) region of both ReLU units, halting their gradient flow and preventing the model from correcting their classification. The final hyperplane plot (`reinit_failure_plots/hyperplanes_final.png`) illustrates this outcome.

This suggests a new hypothesis for this failure: if an initial hyperplane is too close to a data point, it can be pushed across that point during early optimization, effectively deactivating it before the model has a chance to learn a globally optimal solution. A potential mitigation could be a "margin-based" re-initialization, which would ensure all data points are a minimum distance from the initial hyperplanes.

## 6. Results of In-Training Monitoring and Correction (`relu1_monitor`)

This experiment used the `Standard Init` condition but enabled the real-time health monitor to correct for emergent failures during training.

### 6.1. Primary Outcome: Classification Accuracy
The active monitoring and correction strategy was highly effective, resulting in a **100% success rate** over 1000 runs.

### 6.2. Diagnostic Finding: Overcoming Initialization Failures
The monitor's success is directly attributable to its ability to correct for the failure modes identified in the baseline experiment.

* The monitoring system successfully rescued runs that began with known failure conditions. **751 runs that started with 'dead' data points were corrected by the monitor to achieve 100% accuracy.**
* Crucially, this includes **487 runs where a class-1 (True) input was dead at initialization**—a condition that was previously predictive of certain failure in the unmonitored baseline experiment.

## 7. Final Loss & Convergence (Standard Init)

*The following metrics refer to the `relu1_normal` experiment runs.*

The models that failed to achieve perfect accuracy also retained noticeably higher loss values at termination. Mean final loss across all runs was **1.05e-01**, with a maximum loss of **2.52e-01**. The overall convergence distribution was slower than in the `abs1` model.

| Percentile | Epochs |
| :--- | :--- |
| 0th | 33 |
| 50th | 123 |
| 100th | 275 |

## 8. Prototype Geometry and Hyperplane Clustering (Standard Init)

*The following metrics refer to the `relu1_normal` experiment runs.*

The final learned hyperplanes exhibited broader variability and weaker class-0 alignment compared to `abs1`. Cluster analysis revealed five distinct solution modes, with many noisy outliers—further suggesting that training often failed to converge on a coherent decision surface.