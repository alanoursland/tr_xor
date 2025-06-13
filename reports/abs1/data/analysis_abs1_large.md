# 🧪 Experiment Report: `abs1_large`

**Description**: Centered XOR with single absolute value unit and large normal init.

## 🎯 Overview

* **Total runs**: 50
* **Training stops when loss < 1.0e-07**
* ✅ All runs achieved 100% classification accuracy

---

## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 154     |
| 10th       | 304     |
| 25th       | 511     |
| 50th       | 644     |
| 75th       | 874     |
| 90th       | 1009     |
| 100th      | 1670     |

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 3.8e-04 ± 5.8e-05

* **Mean absolute distance of class 1 points to surface**: 1.41421 ± 1.3e-07


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.6 – 7.4     | 628.8                       |
| 10–25%     | 7.4 – 21.2     | 499.4                       |
| 25–50%     | 21.2 – 49.2     | 645.8                       |
| 50–75%     | 49.2 – 71.1     | 932.9                       |
| 75–90%     | 71.1 – 82.0     | 670.9                       |
| 90–100%    | 82.0 – 89.5     | 644.6                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 1.85 – 2.57 | 203.4                       |
| 10–25%     | 2.57 – 4.21 | 393.6                       |
| 25–50%     | 4.21 – 6.26 | 584.6                       |
| 50–75%     | 6.26 – 8.46 | 775.4                       |
| 75–90%     | 8.46 – 11.45 | 950.8                       |
| 90–100%    | 11.45 – 17.79 | 1316.2                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 7.33e-08

* **Variance**: 1.63e-16

* **Range**:

  * 0th percentile: 1.45e-08
  * 100th percentile: 8.44e-08


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 2

### ◼ Cluster 0

* **Size**: 27 runs
* **Weight centroid**: [-0.500088, 0.500059]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: -0.500088x₁ + 0.500059x₂ + 0.000000 = 0

### ◼ Cluster 1

* **Size**: 23 runs
* **Weight centroid**: [0.500100, -0.500112]
* **Bias centroid**: 0.000000
* **Hyperplane equation**: 0.500100x₁ + -0.500112x₂ + 0.000000 = 0

---

