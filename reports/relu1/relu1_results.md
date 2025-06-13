# **Results: Two ReLU Units Summed for XOR Classification**

## 1. Overview

This document summarizes empirical results from the `relu1_normal` experiment, which investigates a minimal ReLU-based architecture applied to the centered XOR problem. The model consists of two linear units followed by ReLU activations whose outputs are summed to form the prediction. Training was conducted using a fixed initialization scheme over 50 independent runs.

Unlike the `abs1` model—which converged successfully in every case—this experiment **failed to reach 100% accuracy in 21 out of 50 runs**, despite using the same data and optimizer. This result demonstrates that the ReLU decomposition of the absolute value is not guaranteed to preserve the same optimization behavior, even under ideal conditions.

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

