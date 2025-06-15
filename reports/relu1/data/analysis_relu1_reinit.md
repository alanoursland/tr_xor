# 🧪 Experiment Report: `relu1_reinit`

**Description**: Centered XOR with two nodes, ReLU, sum, and normal init. If dead data is detected, model is reinitialized.

## 🎯 Overview

* **Total runs**: 50
* **Loss function**: MSELoss
* **Optimizer**: Adam
* **Max epochs**: 800
* **Stops when loss < 1.0e-07**
* **Stops if loss does not improve by ≥ 1.0e-24 over 10 epochs**

---

## 🎯 Classification Accuracy

* 49/50 runs achieved 100% accuracy
* 1/50 runs achieved 75% accuracy

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 2.2e-02 ± 4.4e-02

* **Mean absolute distance of class 1 points to surface**: 1.30798 ± 1.3e-01


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.3 – 10.5     | 95.0                       |
| 10–25%     | 10.5 – 17.0     | 111.3                       |
| 25–50%     | 17.0 – 36.1     | 116.9                       |
| 50–75%     | 36.1 – 59.5     | 128.9                       |
| 75–90%     | 59.5 – 78.6     | 163.3                       |
| 90–100%    | 78.6 – 88.3     | 169.9                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.01 – 0.25 | 131.5                       |
| 10–25%     | 0.25 – 0.48 | 125.3                       |
| 25–50%     | 0.48 – 0.75 | 128.4                       |
| 50–75%     | 0.75 – 1.00 | 115.5                       |
| 75–90%     | 1.00 – 1.17 | 136.1                       |
| 90–100%    | 1.17 – 2.91 | 158.0                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 5.01e-03

* **Variance**: 1.22e-03

* **Range**:

  * 0th percentile: 1.35e-09
  * 100th percentile: 2.50e-01


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 3
* **Noise points**: 15

### ◼ Cluster 0

* **Size**: 13 runs
* **Weight centroid**: [0.512249, -0.510244]
* **Bias centroid**: -0.023228
* **Hyperplane equation**: 0.512249x₁ + -0.510244x₂ + -0.023228 = 0

### ◼ Cluster 1

* **Size**: 20 runs
* **Weight centroid**: [-0.514092, 0.517274]
* **Bias centroid**: -0.030854
* **Hyperplane equation**: -0.514092x₁ + 0.517274x₂ + -0.030854 = 0

### ◼ Cluster 2

* **Size**: 2 runs
* **Weight centroid**: [0.520314, -0.760138]
* **Bias centroid**: -0.280705
* **Hyperplane equation**: 0.520314x₁ + -0.760138x₂ + -0.280705 = 0

---

## 💀 Dead Data Point Analysis

* 49 runs with **no dead inputs** reached 100% accuracy
* 1 runs with **no dead inputs** reached 75% accuracy

---

## 🔍 Mirror Weight Symmetry

* **Mirror pairs detected**: 47 / 50 runs
* **Perfect mirror symmetry** (cosine ~ -1.0): 25 runs
* **Mean mirror similarity**: -0.99432 ± 0.00947
* **Mean mirror error (|cos + 1|)**: 0.00568

---

