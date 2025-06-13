# 🧪 Experiment Report: `relu1_normal`

**Description**: Centered XOR with two nodes, ReLU, sum, and kaiming init.

## 🎯 Overview

* **Total runs**: 50
* **Loss function**: MSELoss
* **Optimizer**: Adam
* **Max epochs**: 800
* **Stops when loss < 1.0e-07**
* **Stops if loss does not improve by ≥ 1.0e-24 over 10 epochs**

---

## 🎯 Classification Accuracy

* 29/50 runs achieved 100% accuracy
* 21/50 runs achieved 75% accuracy

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 6.9e-02 ± 1.6e-01

* **Mean absolute distance of class 1 points to surface**: 1.27205 ± 3.1e-01


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.3 – 8.9     | 80.3                       |
| 10–25%     | 8.9 – 22.5     | 109.2                       |
| 25–50%     | 22.5 – 38.6     | 127.9                       |
| 50–75%     | 38.6 – 61.8     | 131.7                       |
| 75–90%     | 61.8 – 72.9     | 154.8                       |
| 90–100%    | 72.9 – 171.8     | 141.7                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.01 – 0.40 | 118.6                       |
| 10–25%     | 0.40 – 0.61 | 134.2                       |
| 25–50%     | 0.61 – 0.88 | 129.7                       |
| 50–75%     | 0.88 – 1.22 | 136.9                       |
| 75–90%     | 1.22 – 1.91 | 116.5                       |
| 90–100%    | 1.91 – 5.36 | 105.9                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 1.05e-01

* **Variance**: 1.52e-02

* **Range**:

  * 0th percentile: 1.67e-09
  * 100th percentile: 2.52e-01


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 5
* **Noise points**: 26

### ◼ Cluster 0

* **Size**: 2 runs
* **Weight centroid**: [-0.515136, 0.707638]
* **Bias centroid**: -0.222633
* **Hyperplane equation**: -0.515136x₁ + 0.707638x₂ + -0.222633 = 0

### ◼ Cluster 1

* **Size**: 9 runs
* **Weight centroid**: [-0.524666, 0.510748]
* **Bias centroid**: -0.035200
* **Hyperplane equation**: -0.524666x₁ + 0.510748x₂ + -0.035200 = 0

### ◼ Cluster 2

* **Size**: 3 runs
* **Weight centroid**: [-0.510803, 0.506450]
* **Bias centroid**: -0.017289
* **Hyperplane equation**: -0.510803x₁ + 0.506450x₂ + -0.017289 = 0

### ◼ Cluster 3

* **Size**: 8 runs
* **Weight centroid**: [0.522852, -0.514912]
* **Bias centroid**: -0.037724
* **Hyperplane equation**: 0.522852x₁ + -0.514912x₂ + -0.037724 = 0

### ◼ Cluster 4

* **Size**: 2 runs
* **Weight centroid**: [0.688900, -0.515857]
* **Bias centroid**: -0.204663
* **Hyperplane equation**: 0.688900x₁ + -0.515857x₂ + -0.204663 = 0

---

## 💀 Dead Data Point Analysis

* 13 runs with **no dead inputs** reached 100% accuracy
* 16 runs with **dead inputs** reached 100% accuracy
|    16 runs with class-0 dead inputs reached 100% accuracy
|    0 runs with class-1 dead inputs reached 100% accuracy
* 21 runs with **dead inputs** reached 75% accuracy
|    13 runs with class-0 dead inputs reached 75% accuracy
|    20 runs with class-1 dead inputs reached 75% accuracy

---

## 🔍 Mirror Weight Symmetry

* **Mirror pairs detected**: 29 / 50 runs
* **Perfect mirror symmetry** (cosine ~ -1.0): 15 runs
* **Mean mirror similarity**: -0.99335 ± 0.00970
* **Mean mirror error (|cos + 1|)**: 0.00665

---

