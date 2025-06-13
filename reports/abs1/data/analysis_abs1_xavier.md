# 🧪 Experiment Report: `abs1_xavier`

**Description**: Centered XOR with single absolute value unit xavier init.

## 🎯 Overview

* **Total runs**: 50
* **Training stops when loss < 1.0e-07**
* ✅ All runs achieved 100% classification accuracy

---

## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 62     |
| 10th       | 101     |
| 25th       | 120     |
| 50th       | 144     |
| 75th       | 221     |
| 90th       | 297     |
| 100th      | 449     |

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 3.3e-04 ± 2.8e-04

* **Mean absolute distance of class 1 points to surface**: 1.41421 ± 1.6e-07


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.6 – 7.5     | 128.6                       |
| 10–25%     | 7.5 – 21.3     | 97.5                       |
| 25–50%     | 21.3 – 49.2     | 145.9                       |
| 50–75%     | 49.2 – 71.1     | 228.1                       |
| 75–90%     | 71.1 – 82.0     | 214.9                       |
| 90–100%    | 82.0 – 89.5     | 238.6                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.38 – 0.52 | 128.0                       |
| 10–25%     | 0.52 – 0.86 | 120.0                       |
| 25–50%     | 0.86 – 1.28 | 144.6                       |
| 50–75%     | 1.28 – 1.73 | 175.0                       |
| 75–90%     | 1.73 – 2.34 | 230.5                       |
| 90–100%    | 2.34 – 3.63 | 308.8                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 1.12e-07

* **Variance**: 1.07e-13

* **Range**:

  * 0th percentile: 2.44e-11
  * 100th percentile: 2.29e-06


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 2

### ◼ Cluster 0

* **Size**: 27 runs
* **Weight centroid**: [-0.499975, 0.499907]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: -0.499975x₁ + 0.499907x₂ + 0.000000 = 0

### ◼ Cluster 1

* **Size**: 23 runs
* **Weight centroid**: [0.499944, -0.499948]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: 0.499944x₁ + -0.499948x₂ + 0.000000 = 0

---

