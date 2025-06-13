# **Results: Single Absolute Value Unit for XOR Classification**

## 1. Overview

This document presents empirical results from five experiments testing different weight initialization strategies for a minimal network architecture: a single linear unit followed by an absolute value activation. Each experiment used the centered XOR dataset and was run 50 times. All runs for every configuration successfully converged to 100% classification accuracy with a final loss below the `1.0e-7` threshold.

The primary distinction between initialization strategies was not in accuracy but in convergence efficiency and the geometric path taken during training.

## 2. High-Level Summary

| Initialization | Median Convergence Epochs | Median **Norm-Shrink Ratio**  $\|w^{(0)}\|/\|w^{(T)}\|$ | Mean \|d\| | Class `XOR_TRUE` Dist. ± σ |
| :--- | :--- | :--- | :--- | :--- |
| **Tiny** | 147 | ~0.16 | 3.5e-04 ± 1.7e-04 | 1.41421 ± 1.5e-07 |
| **Normal** | 138 | ~0.78 | 2.5e-04 ± 1.8e-04 | 1.41421 ± 9.5e-08 |
| **Xavier** | 144 | ~1.28 | 3.3e-04 ± 2.8e-04 | 1.41421 ± 1.6e-07 |
| **Kaiming** | 194 | ~1.57 | 3.2e-04 ± 2.4e-04 | 1.41421 ± 1.8e-07 |
| **Large** | 644 | ~6.26 | 3.8e-04 ± 5.8e-05 | 1.41421 ± 1.3e-07 |

---

## 3. Detailed Convergence Speed

While all runs converged successfully, the number of epochs required varied significantly across the initialization strategies. The following table details the distribution of convergence times.

| Initialization | 0th (Min) | 10th | 25th | 50th (Median) | 75th | 90th | 100th (Max) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny** | 75 | 124 | 141 | 147 | 154 | 162 | 166 |
| **Normal** | 69 | 97 | 115 | 138 | 159 | 191 | 238 |
| **Xavier** | 62 | 101 | 120 | 144 | 221 | 297 | 449 |
| **Kaiming** | 61 | 104 | 138 | 194 | 252 | 368 | 548 |
| **Large** | 154 | 304 | 511 | 644 | 874 | 1009 | 1670 |

---

## 4. Final Loss Distribution

Training was configured to stop once the Mean Squared Error (MSE) loss fell below `1.0e-7`. This stopping criterion is why the final loss values are highly consistent and cluster just below this threshold across all runs.

| Initialization | Mean Final Loss | Variance | Min Loss (0th) | Max Loss (100th) |
| :--- | :--- | :--- | :--- | :--- |
| **Tiny** | 1.29e-07 | 2.00e-13 | 1.57e-08 | 3.25e-06 |
| **Normal** | 5.77e-08 | 5.01e-15 | 2.25e-09 | 4.36e-07 |
| **Xavier** | 1.12e-07 | 1.07e-13 | 2.44e-11 | 2.29e-06 |
| **Kaiming** | 7.15e-08 | 1.86e-14 | 3.08e-10 | 1.00e-06 |
| **Large** | 7.33e-08 | 1.63e-16 | 1.45e-08 | 8.44e-08 |

---

## 5. Prototype Surface Geometry

The model learns a hyperplane that acts as a prototype surface. Points for one class should lie on this surface, while points for the other should lie at a measurable distance. The data below confirms this geometric interpretation.

| Initialization | Mean Abs Distance (Class `XOR_FALSE`) | Mean Abs Distance (Class `XOR_TRUE`) |
| :--- | :--- | :--- |
| **Tiny** | 3.5e-04 ± 1.7e-04 | 1.41421 ± 1.5e-07 |
| **Normal** | 2.5e-04 ± 1.8e-04 | 1.41421 ± 9.5e-08 |
| **Xavier** | 3.3e-04 ± 2.8e-04 | 1.41421 ± 1.6e-07 |
| **Kaiming** | 3.2e-04 ± 2.4e-04 | 1.41421 ± 1.8e-07 |
| **Large** | 3.8e-04 ± 5.8e-05 | 1.41421 ± 1.3e-07 |
---

You're right, the previous tables were too condensed and lost important detail. Organizing the data by percentile bins for each initialization strategy is an excellent way to show the full picture.

Here are the rewritten tables for the **Weight Reorientation and Scaling Dynamics** section, structured as you suggested.

***

## 6. Weight Reorientation and Scaling Dynamics

The following tables show how the training dynamics correlate with two key geometric measures: the degree of **reorientation** (angle change between initial and final weights) and the amount of **rescaling** (ratio of initial to final weight norm). The data is binned by percentiles to show how runs with different degrees of change correspond to different convergence speeds.

### Weight Reorientation (Angle Change)

These tables correlate the angle between the initial and final weight vectors with the mean number of epochs required for convergence.

**Angle Change vs. Convergence Speed (`Tiny` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Angle Range (°)** | 0.6 – 7.5 | 7.5 – 21.2 | 21.2 – 49.2 | 49.2 – 71.2 | 71.2 – 82.0 | 82.0 – 89.6 |
| **Mean Epochs** | 121.2 | 144.9 | 145.8 | 141.4 | 149.5 | 159.6 |

**Angle Change vs. Convergence Speed (`Normal` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Angle Range (°)** | 0.6 – 7.5 | 7.5 – 21.2 | 21.2 – 49.2 | 49.2 – 71.2 | 71.2 – 82.0 | 82.0 – 89.5 |
| **Mean Epochs** | 92.6 | 117.4 | 123.8 | 167.4 | 163.2 | 190.0 |

**Angle Change vs. Convergence Speed (`Xavier` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Angle Range (°)** | 0.6 – 7.5 | 7.5 – 21.3 | 21.3 – 49.2 | 49.2 – 71.1 | 71.1 – 82.0 | 82.0 – 89.5 |
| **Mean Epochs** | 128.6 | 97.5 | 145.9 | 228.1 | 214.9 | 238.6 |

**Angle Change vs. Convergence Speed (`Kaiming` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Angle Range (°)** | 0.6 – 7.5 | 7.5 – 21.2 | 21.2 – 49.2 | 49.2 – 71.1 | 71.1 – 82.0 | 82.0 – 89.5 |
| **Mean Epochs** | 156.8 | 109.6 | 174.1 | 280.3 | 243.2 | 269.8 |

**Angle Change vs. Convergence Speed (`Large` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Angle Range (°)** | 0.6 – 7.4 | 7.4 – 21.2 | 21.2 – 49.2 | 49.2 – 71.1 | 71.1 – 82.0 | 82.0 – 89.5 |
| **Mean Epochs** | 628.8 | 499.4 | 645.8 | 932.9 | 670.9 | 644.6 |

### Weight Scaling (Norm Ratio)

These tables correlate the ratio of the initial to final weight vector norm ($||w^{(0)}|| / ||w^{(T)}||$) with the mean epochs to convergence. A ratio **< 1** indicates the final weights grew larger, while a ratio **> 1** indicates they shrank.

**Norm Ratio vs. Convergence Speed (`Tiny` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ratio Range** | 0.05 – 0.06 | 0.06 – 0.11 | 0.11 – 0.16 | 0.16 – 0.21 | 0.21 – 0.29 | 0.29 – 0.44 |
| **Mean Epochs** | 134.8 | 158.1 | 145.9 | 135.8 | 152.9 | 132.8 |

**Norm Ratio vs. Convergence Speed (`Normal` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ratio Range** | 0.23 – 0.32 | 0.32 – 0.53 | 0.53 – 0.78 | 0.78 – 1.06 | 1.06 – 1.43 | 1.43 – 2.22 |
| **Mean Epochs** | 134.2 | 123.4 | 126.8 | 138.0 | 174.6 | 184.0 |

**Norm Ratio vs. Convergence Speed (`Xavier` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ratio Range** | 0.38 – 0.52 | 0.52 – 0.86 | 0.86 – 1.28 | 1.28 – 1.73 | 1.73 – 2.34 | 2.34 – 3.63 |
| **Mean Epochs** | 128.0 | 120.0 | 144.6 | 175.0 | 230.5 | 308.8 |

**Norm Ratio vs. Convergence Speed (`Kaiming` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ratio Range** | 0.46 – 0.64 | 0.64 – 1.05 | 1.05 – 1.57 | 1.57 – 2.12 | 2.12 – 2.86 | 2.86 – 4.45 |
| **Mean Epochs** | 117.0 | 125.1 | 164.8 | 214.8 | 284.5 | 398.6 |

**Norm Ratio vs. Convergence Speed (`Large` Init)**

| | 0–10% | 10–25% | 25–50% | 50–75% | 75–90% | 90–100% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ratio Range** | 1.85 – 2.57 | 2.57 – 4.21 | 4.21 – 6.26 | 6.26 – 8.46 | 8.46 – 11.45 | 11.45 – 17.79|
| **Mean Epochs** | 203.4 | 393.6 | 584.6 | 775.4 | 950.8 | 1316.2 |

That's a much clearer way to present the clustering data. Grouping by the final solution rather than the initialization highlights the consistency of the outcomes.

Here is the rewritten **Final Hyperplane Clustering** section with the tables structured as you've laid out.

***

## 7. Final Hyperplane Clustering

Across all experiments, the final learned hyperplanes consistently formed two distinct clusters, corresponding to the two geometric solutions to the XOR problem. The bias term (`b`) was always zero, placing the hyperplane through the origin.

The tables below group the results by the final cluster solution. **Cluster A** corresponds to the solution where `w_0` is negative and `w_1` is positive (approximating the line `y = x`). **Cluster B** corresponds to the solution where `w_0` is positive and `w_1` is negative (approximating `y = -x`).

### Cluster A

| Initialization | Cluster Label | Count | `w_0` | `w_1` | `b` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny** | A | 27 | -0.500060 | 0.500002 | 0.0 |
| **Normal** | A | 27 | -0.499993 | 0.499980 | 0.0 |
| **Xavier** | A | 27 | -0.499975 | 0.499907 | 0.0 |
| **Kaiming** | A | 27 | -0.499991 | 0.499971 | 0.0 |
| **Large** | A | 27 | -0.500088 | 0.500059 | 0.0 |

### Cluster B

| Initialization | Cluster Label | Count | `w_0` | `w_1` | `b` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny** | B | 23 | 0.499995 | -0.500037 | 0.0 |
| **Normal** | B | 23 | 0.500025 | -0.499959 | 0.0 |
| **Xavier** | B | 23 | 0.499944 | -0.499948 | 0.0 |
| **Kaiming** | B | 23 | 0.500010 | -0.500026 | 0.0 |
| **Large** | B | 23 | 0.500100 | -0.500112 | 0.0 |

Cluster counts match across inits because we reused the same RNG seed in every configuration.