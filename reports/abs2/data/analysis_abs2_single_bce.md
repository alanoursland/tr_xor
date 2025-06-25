# 🧪 Experiment Report: `abs2_single_bce`

**Description**: Centered XOR with 2-output BCE loss using a single Abs unit.

## 🎯 Overview

* **Total runs**: 50
* **Loss function**: BCEWithLogitsLoss
* **Optimizer**: Adam
* **Max epochs**: 5000
* **Stops when loss < 1.0e-07**
* **Stops if loss does not improve by ≥ 1.0e-24 over 10 epochs**

---

## 🎯 Classification Accuracy

* 50/50 runs achieved 100% accuracy

---

## ⏱️ Convergence Timing (Epochs to MSE < 1e-7)

| Percentile | Epochs |
| ---------- | ------ |
| 0th        | 2934     |
| 10th       | 3016     |
| 25th       | 3082     |
| 50th       | 3131     |
| 75th       | 3208     |
| 90th       | 3242     |
| 100th      | 3332     |

---


## 📏 Prototype Surface Geometry

### Layer: `linear1`

- **Unit 0**
  - Mean distance to class 0: `5.38e-01 ± 6.86e-01`
  - Mean distance to class 1: `0.87689 ± 6.86e-01`
  - Separation ratio (class1/class0): `1.63`

### Layer: `linear2`

- **Unit 0**
  - Mean distance to class 0: `3.27e+00 ± 2.02e+00`
  - Mean distance to class 1: `4.02360 ± 2.25e+00`
  - Separation ratio (class1/class0): `1.23`

- **Unit 1**
  - Mean distance to class 0: `3.37e+00 ± 1.90e+00`
  - Mean distance to class 1: `3.96957 ± 2.39e+00`
  - Separation ratio (class1/class0): `1.18`


---

### Layer: `linear1` – Angle Between Initial and Final Weights

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | ---------------- | -------------------------- |
| 0–10%      | 0.1 – 7.0       | 3220.2                       |
| 10–25%     | 7.0 – 11.3       | 3161.6                       |
| 25–50%     | 11.3 – 22.3       | 3127.8                       |
| 50–75%     | 22.3 – 44.0       | 3111.5                       |
| 75–90%     | 44.0 – 62.2       | 3130.1                       |
| 90–100%    | 62.2 – 89.0       | 3109.0                       |

### Layer: `linear1` – Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ------------ | -------------------------- |
| 0–10%      | 0.01 – 0.12  | 3029.2                       |
| 10–25%     | 0.12 – 0.17  | 3019.9                       |
| 25–50%     | 0.17 – 0.36  | 3145.2                       |
| 50–75%     | 0.36 – 0.50  | 3179.9                       |
| 75–90%     | 0.50 – 0.55  | 3210.4                       |
| 90–100%    | 0.55 – 0.66  | 3192.4                       |

---

### Layer: `linear2` – Angle Between Initial and Final Weights

| Percentile | Angle Range (°) | Mean Epochs to Convergence |
| ---------- | ---------------- | -------------------------- |
| 0–10%      | 1.7 – 16.6       | 3168.2                       |
| 10–25%     | 16.6 – 26.0       | 3170.0                       |
| 25–50%     | 26.0 – 48.0       | 3079.2                       |
| 50–75%     | 48.0 – 86.5       | 3103.4                       |
| 75–90%     | 86.5 – 123.1       | 3161.2                       |
| 90–100%    | 123.1 – 179.6       | 3234.0                       |

### Layer: `linear2` – Initial / Final Norm Ratio

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ------------ | -------------------------- |
| 0–10%      | 0.02 – 0.04  | 3082.4                       |
| 10–25%     | 0.04 – 0.06  | 3062.8                       |
| 25–50%     | 0.06 – 0.09  | 3117.4                       |
| 50–75%     | 0.09 – 0.11  | 3171.2                       |
| 75–90%     | 0.11 – 0.14  | 3196.8                       |
| 90–100%    | 0.14 – 0.24  | 3180.2                       |

---

### ◼ Initial / Final Norm Ratio (All Layers Combined)

| Percentile | Ratio Range | Mean Epochs to Convergence |
| ---------- | ------------ | -------------------------- |
|  1         | 0.01 – 0.12  | 3029.2                       |
|  2         | 0.02 – 0.04  | 3082.4                       |
|  3         | 0.04 – 0.06  | 3062.8                       |
|  4         | 0.06 – 0.09  | 3117.4                       |
|  5         | 0.09 – 0.11  | 3171.2                       |
|  6         | 0.11 – 0.14  | 3196.8                       |
|  7         | 0.12 – 0.17  | 3019.9                       |
|  8         | 0.14 – 0.24  | 3180.2                       |
|  9         | 0.17 – 0.36  | 3145.2                       |
| 10         | 0.36 – 0.50  | 3179.9                       |
| 11         | 0.50 – 0.55  | 3210.4                       |
| 12         | 0.55 – 0.66  | 3192.4                       |

---

## 📉 Final Loss Distribution

* **Mean final loss**: 9.70e-08

* **Variance**: 1.24e-15

* **Range**:

  * 0th percentile: 3.32e-08
  * 100th percentile: 3.19e-07


---

## 🎯 Hyperplane Clustering

### 🔹 Layer `linear1`
* **Number of clusters discovered**: 11
* **Noise points**: 8

#### ◼ Cluster 0
* **Size**: 6 runs
* **Weight centroid**: [2.300308, 2.300167]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: 2.300308x₁ + 2.300167x₂ + 0.000000 = 0

#### ◼ Cluster 1
* **Size**: 3 runs
* **Weight centroid**: [2.680283, -2.680066]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: 2.680283x₁ + -2.680066x₂ + 0.000000 = 0

#### ◼ Cluster 2
* **Size**: 2 runs
* **Weight centroid**: [-2.671327, -2.670847]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: -2.671327x₁ + -2.670847x₂ + 0.000000 = 0

#### ◼ Cluster 3
* **Size**: 2 runs
* **Weight centroid**: [2.378139, -2.378346]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: 2.378139x₁ + -2.378346x₂ + 0.000000 = 0

#### ◼ Cluster 4
* **Size**: 2 runs
* **Weight centroid**: [-2.846759, -2.846374]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: -2.846759x₁ + -2.846374x₂ + 0.000000 = 0

#### ◼ Cluster 5
* **Size**: 13 runs
* **Weight centroid**: [-2.535009, 2.534941]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: -2.535009x₁ + 2.534941x₂ + 0.000000 = 0

#### ◼ Cluster 6
* **Size**: 2 runs
* **Weight centroid**: [-2.827012, 2.826115]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: -2.827012x₁ + 2.826115x₂ + 0.000000 = 0

#### ◼ Cluster 7
* **Size**: 3 runs
* **Weight centroid**: [2.524856, 2.525915]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: 2.524856x₁ + 2.525915x₂ + 0.000000 = 0

#### ◼ Cluster 8
* **Size**: 3 runs
* **Weight centroid**: [2.901796, -2.901515]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: 2.901796x₁ + -2.901515x₂ + 0.000000 = 0

#### ◼ Cluster 9
* **Size**: 4 runs
* **Weight centroid**: [2.136560, -2.135422]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: 2.136560x₁ + -2.135422x₂ + 0.000000 = 0

#### ◼ Cluster 10
* **Size**: 2 runs
* **Weight centroid**: [-2.984362, -2.984096]
* **Bias centroid**: [0.000000]
* **Hyperplane equation**: -2.984362x₁ + -2.984096x₂ + 0.000000 = 0


### 🔹 Layer `scale`
* **Number of clusters discovered**: 1

#### ◼ Cluster 0
* **Size**: 50 runs
* **Weight centroid**: [1.000000]
* **Bias centroid**: [0.000000]


### 🔹 Layer `linear2`
* **Number of clusters discovered**: 11
* **Noise points**: 21

#### ◼ Cluster 0
* **Size**: 4 runs
* **Weight centroid**: [6.968760, -7.481071]
* **Bias centroid**: [-14.831604, 14.900865]

#### ◼ Cluster 1
* **Size**: 2 runs
* **Weight centroid**: [-6.993798, 6.582438]
* **Bias centroid**: [14.979403, -15.017406]

#### ◼ Cluster 2
* **Size**: 2 runs
* **Weight centroid**: [5.486222, -5.987028]
* **Bias centroid**: [-14.566658, 15.374281]

#### ◼ Cluster 3
* **Size**: 2 runs
* **Weight centroid**: [-6.919537, 6.401582]
* **Bias centroid**: [14.984613, -14.619282]

#### ◼ Cluster 4
* **Size**: 3 runs
* **Weight centroid**: [-5.874110, 5.560886]
* **Bias centroid**: [14.891899, -15.191213]

#### ◼ Cluster 5
* **Size**: 3 runs
* **Weight centroid**: [6.175931, -6.580673]
* **Bias centroid**: [-15.185235, 14.901633]

#### ◼ Cluster 6
* **Size**: 2 runs
* **Weight centroid**: [6.630861, -7.079667]
* **Bias centroid**: [-14.885347, 15.021505]

#### ◼ Cluster 7
* **Size**: 4 runs
* **Weight centroid**: [-6.936885, 6.256909]
* **Bias centroid**: [15.232782, -14.918238]

#### ◼ Cluster 8
* **Size**: 3 runs
* **Weight centroid**: [-6.453022, 6.166180]
* **Bias centroid**: [14.895932, -15.819618]

#### ◼ Cluster 9
* **Size**: 2 runs
* **Weight centroid**: [-8.242989, 7.608178]
* **Bias centroid**: [14.899640, -14.879039]

#### ◼ Cluster 10
* **Size**: 2 runs
* **Weight centroid**: [-6.598021, 6.151974]
* **Bias centroid**: [14.897385, -15.035958]


---

