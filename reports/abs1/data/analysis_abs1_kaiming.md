# 🧪 Experiment Report: `abs1_kaiming`

**Description**: Centered XOR with single absolute value unit and kaiming init.

## 🎯 Overview

* **Total runs**: 50
* **Training stops when loss < 1.0e-07**
* ✅ All runs achieved 100% classification accuracy

---

## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 61     |
| 10th       | 104     |
| 25th       | 138     |
| 50th       | 194     |
| 75th       | 252     |
| 90th       | 368     |
| 100th      | 548     |

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 3.2e-04 ± 2.4e-04

* **Mean absolute distance of class 1 points to surface**: 1.41421 ± 1.8e-07


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.6 – 7.5     | 156.8                       |
| 10–25%     | 7.5 – 21.2     | 109.6                       |
| 25–50%     | 21.2 – 49.2     | 174.1                       |
| 50–75%     | 49.2 – 71.1     | 280.3                       |
| 75–90%     | 71.1 – 82.0     | 243.2                       |
| 90–100%    | 82.0 – 89.5     | 269.8                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.46 – 0.64 | 117.0                       |
| 10–25%     | 0.64 – 1.05 | 125.1                       |
| 25–50%     | 1.05 – 1.57 | 164.8                       |
| 50–75%     | 1.57 – 2.12 | 214.8                       |
| 75–90%     | 2.12 – 2.86 | 284.5                       |
| 90–100%    | 2.86 – 4.45 | 398.6                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 7.15e-08

* **Variance**: 1.86e-14

* **Range**:

  * 0th percentile: 3.08e-10
  * 100th percentile: 1.00e-06


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 2

### ◼ Cluster 0

* **Size**: 27 runs
* **Weight centroid**: [-0.499991, 0.499971]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: -0.499991x₁ + 0.499971x₂ + 0.000000 = 0

### ◼ Cluster 1

* **Size**: 23 runs
* **Weight centroid**: [0.500010, -0.500026]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: 0.500010x₁ + -0.500026x₂ + 0.000000 = 0

---

