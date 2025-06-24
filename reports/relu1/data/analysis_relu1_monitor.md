# 🧪 Experiment Report: `relu1_monitor`

**Description**: Centered XOR with two nodes, ReLU, sum, normal init, and early-failure degeneracy detection.

## 🎯 Overview

* **Total runs**: 1000
* **Loss function**: MSELoss
* **Optimizer**: Adam
* **Max epochs**: 2000
* **Stops when loss < 1.0e-07**

---

## 🎯 Classification Accuracy

* 1000/1000 runs achieved 100% accuracy

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 2.4e-02 ± 5.6e-02

* **Mean absolute distance of class 1 points to surface**: 1.30958 ± 1.2e-01


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.0 – 9.7     | 157.7                       |
| 10–25%     | 9.7 – 23.6     | 150.6                       |
| 25–50%     | 23.6 – 49.5     | 162.9                       |
| 50–75%     | 49.5 – 85.7     | 204.3                       |
| 75–90%     | 85.7 – 120.8     | 204.3                       |
| 90–100%    | 120.8 – 179.0     | 238.1                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.01 – 0.33 | 232.7                       |
| 10–25%     | 0.33 – 0.54 | 205.6                       |
| 25–50%     | 0.54 – 0.79 | 175.0                       |
| 50–75%     | 0.79 – 1.02 | 164.9                       |
| 75–90%     | 1.02 – 1.20 | 176.6                       |
| 90–100%    | 1.20 – 2.05 | 190.1                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 6.32e-08

* **Variance**: 1.27e-14

* **Range**:

  * 0th percentile: 2.11e-11
  * 100th percentile: 2.17e-06


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 4
* **Noise points**: 21

### ◼ Cluster 0

* **Size**: 470 runs
* **Weight centroid**: [-0.534258, 0.547827]
* **Bias centroid**: -0.082128
* **Hyperplane equation**: -0.534258x₁ + 0.547827x₂ + -0.082128 = 0

### ◼ Cluster 1

* **Size**: 500 runs
* **Weight centroid**: [0.539231, -0.541184]
* **Bias centroid**: -0.080457
* **Hyperplane equation**: 0.539231x₁ + -0.541184x₂ + -0.080457 = 0

### ◼ Cluster 2

* **Size**: 6 runs
* **Weight centroid**: [-0.957234, 0.501064]
* **Bias centroid**: -0.458368
* **Hyperplane equation**: -0.957234x₁ + 0.501064x₂ + -0.458368 = 0

### ◼ Cluster 3

* **Size**: 3 runs
* **Weight centroid**: [0.689979, -0.501416]
* **Bias centroid**: -0.191650
* **Hyperplane equation**: 0.689979x₁ + -0.501416x₂ + -0.191650 = 0

---

## 💀 Dead Data Point Analysis

* 249 runs with **no dead inputs** reached 100% accuracy
* 751 runs with **dead inputs** reached 100% accuracy
|    512 runs with class-0 dead inputs reached 100% accuracy
|    487 runs with class-1 dead inputs reached 100% accuracy

---

## 🔍 Mirror Weight Symmetry

* **Mirror pairs detected**: 983 / 1000 runs
* **Perfect mirror symmetry** (cosine ~ -1.0): 475 runs
* **Mean mirror similarity**: -0.99406 ± 0.00954
* **Mean mirror error (|cos + 1|)**: 0.00594

---

