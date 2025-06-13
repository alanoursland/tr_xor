# 🧪 Experiment Report: `abs1_normal`

**Description**: Centered XOR with single absolute value unit and normal init.

## 🎯 Overview

* **Total runs**: 50
* **Training stops when loss < 1.0e-07**
* ✅ All runs achieved 100% classification accuracy

---

## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 69     |
| 10th       | 97     |
| 25th       | 115     |
| 50th       | 138     |
| 75th       | 159     |
| 90th       | 191     |
| 100th      | 238     |

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 2.5e-04 ± 1.8e-04

* **Mean absolute distance of class 1 points to surface**: 1.41421 ± 9.5e-08


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.6 – 7.5     | 92.6                       |
| 10–25%     | 7.5 – 21.2     | 117.4                       |
| 25–50%     | 21.2 – 49.2     | 123.8                       |
| 50–75%     | 49.2 – 71.2     | 167.4                       |
| 75–90%     | 71.2 – 82.0     | 163.2                       |
| 90–100%    | 82.0 – 89.5     | 190.0                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.23 – 0.32 | 134.2                       |
| 10–25%     | 0.32 – 0.53 | 123.4                       |
| 25–50%     | 0.53 – 0.78 | 126.8                       |
| 50–75%     | 0.78 – 1.06 | 138.0                       |
| 75–90%     | 1.06 – 1.43 | 174.6                       |
| 90–100%    | 1.43 – 2.22 | 184.0                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 5.77e-08

* **Variance**: 5.01e-15

* **Range**:

  * 0th percentile: 2.25e-09
  * 100th percentile: 4.36e-07


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 2

### ◼ Cluster 0

* **Size**: 27 runs
* **Weight centroid**: [-0.499993, 0.499980]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: -0.499993x₁ + 0.499980x₂ + 0.000000 = 0

### ◼ Cluster 1

* **Size**: 23 runs
* **Weight centroid**: [0.500025, -0.499959]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: 0.500025x₁ + -0.499959x₂ + 0.000000 = 0

---

