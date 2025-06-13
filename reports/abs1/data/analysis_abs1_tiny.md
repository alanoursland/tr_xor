# 🧪 Experiment Report: `abs1_tiny`

**Description**: Centered XOR with single absolute value unit and tiny normal init.

## 🎯 Overview

* **Total runs**: 50
* **Training stops when loss < 1.0e-07**
* ✅ All runs achieved 100% classification accuracy

---

## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 75     |
| 10th       | 124     |
| 25th       | 141     |
| 50th       | 147     |
| 75th       | 154     |
| 90th       | 162     |
| 100th      | 166     |

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 3.5e-04 ± 1.7e-04

* **Mean absolute distance of class 1 points to surface**: 1.41421 ± 1.5e-07


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.6 – 7.5     | 121.2                       |
| 10–25%     | 7.5 – 21.2     | 144.9                       |
| 25–50%     | 21.2 – 49.2     | 145.8                       |
| 50–75%     | 49.2 – 71.2     | 141.4                       |
| 75–90%     | 71.2 – 82.0     | 149.5                       |
| 90–100%    | 82.0 – 89.6     | 159.6                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.05 – 0.06 | 134.8                       |
| 10–25%     | 0.06 – 0.11 | 158.1                       |
| 25–50%     | 0.11 – 0.16 | 145.9                       |
| 50–75%     | 0.16 – 0.21 | 135.8                       |
| 75–90%     | 0.21 – 0.29 | 152.9                       |
| 90–100%    | 0.29 – 0.44 | 132.8                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 1.29e-07

* **Variance**: 2.00e-13

* **Range**:

  * 0th percentile: 1.57e-08
  * 100th percentile: 3.25e-06


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 2

### ◼ Cluster 0

* **Size**: 27 runs
* **Weight centroid**: [-0.500060, 0.500002]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: -0.500060x₁ + 0.500002x₂ + 0.000000 = 0

### ◼ Cluster 1

* **Size**: 23 runs
* **Weight centroid**: [0.499995, -0.500037]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: 0.499995x₁ + -0.500037x₂ + 0.000000 = 0

---

