# **Results: Single Absolute Value Unit for XOR Classification**

## 1. Overview

This document presents empirical results from five experiments testing different weight initialization strategies for a minimal network architecture: a single linear unit followed by an absolute value activation. Each experiment used the centered XOR dataset and was run 50 times per configuration.

---

## 2. Summary of Results

| Initialization | 100% Accuracy Rate | Avg Accuracy | Convergence Rate (<0.01 Loss) | Mean ‖W‖ ± σ     | Class 0 Dist. ± σ | Class 1 Dist. ± σ |
| -------------- | ------------------ | ------------ | ----------------------------- | ---------------- | ----------------- | ----------------- |
| Tiny           | 100%               | 1.00         | 100%                          | 0.7071 ± 2.2e-05 | 4.0e-05 ± 2.8e-05 | 1.4142 ± 6.5e-08  |
| Normal         | 100%               | 1.00         | 100%                          | 0.7070 ± 1.5e-03 | 8.8e-04 ± 2.7e-03 | 1.4142 ± 1.1e-05  |
| Xavier         | 100%               | 1.00         | 96%                           | 0.7137 ± 5.6e-02 | 2.4e-02 ± 6.9e-02 | 1.4123 ± 9.3e-03  |
| Kaiming        | 98%                | 0.98         | 86%                           | 0.7284 ± 1.2e-01 | 4.6e-02 ± 1.0e-01 | 1.4097 ± 1.2e-02  |
| Large          | 20%                | 0.20         | 14%                           | 2.8043 ± 2.1e+00 | 7.8e-02 ± 8.3e-02 | 1.4096 ± 6.7e-03  |

---

## 3. Accuracy Distribution

### 3.1. High-Performing Initializations

* **Tiny, Normal, Xavier**: All 50 runs achieved 100% accuracy.

### 3.2. Moderate Performance

* **Kaiming**: 49 runs reached 100% accuracy; 1 run ended at 50%.

### 3.3. Low Performance

* **Large**: Only 10 runs achieved 100% accuracy. The majority (39) plateaued at 50%, and one run scored 0%.

---

## 4. Convergence Behavior

### 4.1. Final Loss Statistics

* **Best loss values** ranged from $10^{-12}$ to $10^{-10}$ in successful runs.
* **Worst loss values** ranged from $8.2 \times 10^{-1}$ (kaiming) to $9.97 \times 10^1$ (large), indicating significant divergence in failed cases.

### 4.2. Convergence Rates

* **100% convergence** for tiny and normal.
* Xavier achieved 96%, kaiming 86%, and large only 14%.

---

## 5. Hyperplane Geometry

### 5.1. Distance to Surface

For successful runs:

* **Class-0 points** had low mean distances to the learned hyperplane, indicating near-surface alignment.
* **Class-1 points** had mean distances ≈ √2, consistent with geometric expectations from the centered XOR configuration.

### 5.2. Norm of Weight Vector

* Models converging to correct geometry consistently learned a weight norm near $\|W\| = 0.707$, implying the surface was scaled to produce output values of 0 and √2 before activation.

---

## 6. Observational Notes

* **Accuracy**: Models that successfully learned XOR consistently achieved 100% accuracy using only a single absolute value unit.
* **Loss**: Converged runs reached extremely low final loss values (often < $10^{-10}$), indicating near-perfect output matching.
* **Distance to Surface**: Class-0 points consistently lie near the learned surface, while class-1 points lie approximately √2 units away, as expected from the geometric formulation.
* **Weight Norm**: Successful runs yield weight norms close to 0.707, consistent with scaling the output of off surface classes to one.
