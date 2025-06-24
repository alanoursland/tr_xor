# 🧪 Experiment Report: `relu1_mirror`

**Description**: Centered XOR with two nodes, ReLU, sum, and mirrored normal init.

## 🎯 Overview

* **Total runs**: 1000
* **Loss function**: MSELoss
* **Optimizer**: Adam
* **Max epochs**: 800
* **Stops when loss < 1.0e-07**
* **Stops if loss does not improve by ≥ 1.0e-24 over 10 epochs**

---

## 🎯 Classification Accuracy

* 984/1000 runs achieved 100% accuracy
* 16/1000 runs achieved 50% accuracy

---


## 🧠 Prototype Surface Geometry

* **Mean absolute distance of class 0 points to surface**: 2.1e-02 ± 4.2e-02

* **Mean absolute distance of class 1 points to surface**: 1.30711 ± 1.3e-01


---

## 🔁 Weight Reorientation

### ◼ Angle Between Initial and Final Weights (Degrees)

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | --------------- | -------------------------- |
| 0–10%      | 0.0 – 7.6     | 48.9                       |
| 10–25%     | 7.6 – 19.0     | 53.0                       |
| 25–50%     | 19.0 – 40.7     | 72.6                       |
| 50–75%     | 40.7 – 63.3     | 126.2                       |
| 75–90%     | 63.3 – 77.5     | 163.8                       |
| 90–100%    | 77.5 – 89.9     | 171.8                       |

---

### ◼ Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ----------- | -------------------------- |
| 0–10%      | 0.02 – 0.34 | 102.0                       |
| 10–25%     | 0.34 – 0.56 | 96.9                       |
| 25–50%     | 0.56 – 0.81 | 100.9                       |
| 50–75%     | 0.81 – 1.04 | 96.1                       |
| 75–90%     | 1.04 – 1.24 | 118.0                       |
| 90–100%    | 1.24 – 3.19 | 126.3                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 8.09e-03

* **Variance**: 3.93e-03

* **Range**:

  * 0th percentile: 6.39e-14
  * 100th percentile: 5.00e-01


---

## 🎯 Hyperplane Clustering

* **Number of clusters discovered**: 6
* **Noise points**: 6

### ◼ Cluster 0

* **Size**: 491 runs
* **Weight centroid**: [0.540062, -0.544747]
* **Bias centroid**: -0.085945
* **Hyperplane equation**: 0.540062x₁ + -0.544747x₂ + -0.085945 = 0

### ◼ Cluster 1

* **Size**: 491 runs
* **Weight centroid**: [-0.539468, 0.546498]
* **Bias centroid**: -0.086905
* **Hyperplane equation**: -0.539468x₁ + 0.546498x₂ + -0.086905 = 0

### ◼ Cluster 2

* **Size**: 6 runs
* **Weight centroid**: [-0.198882, -0.202459]
* **Bias centroid**: -0.444444
* **Hyperplane equation**: -0.198882x₁ + -0.202459x₂ + -0.444444 = 0

### ◼ Cluster 3

* **Size**: 2 runs
* **Weight centroid**: [0.180944, 0.147659]
* **Bias centroid**: -0.381845
* **Hyperplane equation**: 0.180944x₁ + 0.147659x₂ + -0.381845 = 0

### ◼ Cluster 4

* **Size**: 2 runs
* **Weight centroid**: [0.190107, 0.230935]
* **Bias centroid**: -0.457744
* **Hyperplane equation**: 0.190107x₁ + 0.230935x₂ + -0.457744 = 0

### ◼ Cluster 5

* **Size**: 2 runs
* **Weight centroid**: [-0.332330, -0.341883]
* **Bias centroid**: -0.688481
* **Hyperplane equation**: -0.332330x₁ + -0.341883x₂ + -0.688481 = 0

---

## 💀 Dead Data Point Analysis

* 984 runs with **no dead inputs** reached 100% accuracy
* 16 runs with **no dead inputs** reached 50% accuracy

---

## 🔍 Mirror Weight Symmetry

* **Mirror pairs detected**: 984 / 984 runs
* **Perfect mirror symmetry** (cosine ~ -1.0): 984 runs
* **Mean mirror similarity**: -1.00000 ± 0.00000
* **Mean mirror error (|cos + 1|)**: 0.00000

---

## 🧭 Geometric Analysis of Failure Modes

We tested whether failed runs began with initial hyperplanes nearly perpendicular to the ideal.

* **Success runs (n=984)** – mean angle diff: 45.15° ± 25.63°
* **Failure runs (n=16)** – mean angle diff: 87.64° ± 1.66°
* Failed runs are tightly clustered near 90°, consistent with the no-torque trap hypothesis.

See `failure_angle_histogram.png` for visual confirmation.

